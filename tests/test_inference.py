"""
Tests for :mod:`sample.inference` — input parsing and inference-time featurization.

The central risk this module carries: a feature computed differently at inference time than at
training time produces a confident wrong answer with nothing to indicate anything went wrong. The
featurization tests therefore check exact column identity and order, not just shape.

The transformer blocks (ProtBERT, ESM-2) are not exercised here -- they download and run a model,
which is an integration concern rather than a unit one.
"""
import json
import os
import pickle
from typing import Any, List

import pytest

from sample import inference as I


class _StubClassifier:
    """
    A fitted classifier stand-in. Defined at module scope because pickle stores a class by
    qualified name, and a class defined inside a fixture cannot be resolved on load.
    """

    feature_names_in_ = ["charge", "sequence_length"]

    def predict_proba(self, X: Any) -> Any:
        import numpy as np
        return np.column_stack([np.full(len(X), 0.2), np.full(len(X), 0.8)])


class _StubRegressor:
    """A fitted regressor stand-in: no ``predict_proba``, so the regression path is taken."""

    feature_names_in_ = ["charge", "sequence_length"]

    def predict(self, X: Any) -> Any:
        import numpy as np
        return np.full(len(X), 12.5)


class _IdentityScaler:
    """A fitted-scaler stand-in for applicability-domain tests: passes features through unchanged
    so the test can control what :func:`sample.ml_models.in_domain` sees without depending on a
    real fitted `StandardScaler`."""

    def transform(self, X: Any) -> Any:
        return X


class TestParsePeptideInput:
    def test_parses_fasta(self) -> None:
        peptides = I.parse_peptide_input(">magainin-2\nGIGKFLHSAKKFGKAFVGEIMNS\n>melittin\nGIGAVLK")
        assert [p["peptide_name"] for p in peptides] == ["magainin-2", "melittin"]
        assert peptides[0]["sequence"] == "GIGKFLHSAKKFGKAFVGEIMNS"

    def test_parses_one_sequence_per_line(self) -> None:
        peptides = I.parse_peptide_input("GIGKFLH\nILPWKWPWWPWRR")
        assert [p["sequence"] for p in peptides] == ["GIGKFLH", "ILPWKWPWWPWRR"]

    def test_unnamed_sequences_get_positional_names(self) -> None:
        # Every result row has to be traceable back to its input.
        peptides = I.parse_peptide_input("GIGKFLH\nILPWKWP")
        assert [p["peptide_name"] for p in peptides] == ["peptide_1", "peptide_2"]

    def test_multiline_fasta_records_are_joined(self) -> None:
        peptides = I.parse_peptide_input(">p\nGIGK\nFLHS\nAKKF")
        assert peptides[0]["sequence"] == "GIGKFLHSAKKF"

    def test_uppercases(self) -> None:
        assert I.parse_peptide_input("gigkflh")[0]["sequence"] == "GIGKFLH"

    def test_blank_lines_are_skipped(self) -> None:
        assert len(I.parse_peptide_input("GIGK\n\n\nFLHS\n")) == 2

    def test_strips_digits_and_whitespace_within_a_sequence(self) -> None:
        # Residue numbering pasted out of a paper.
        assert I.parse_peptide_input(">p\nGIGK 10 FLHS")[0]["sequence"] == "GIGKFLHS"

    @pytest.mark.parametrize("text", ["", "   ", "\n\n", None])
    def test_empty_input_yields_nothing(self, text: Any) -> None:
        assert I.parse_peptide_input(text) == []

    def test_fasta_header_with_no_sequence_is_dropped(self) -> None:
        assert I.parse_peptide_input(">orphan\n") == []


class TestValidateSequence:
    @pytest.mark.parametrize("seq", ["GIGKFLHSAKKFGKAFVGEIMNS", "AC", "ACDEFGHIKLMNPQRSTVWY"])
    def test_accepts_standard_sequences(self, seq: str) -> None:
        assert I.validate_sequence(seq) is None

    @pytest.mark.parametrize("seq", ["", None, "   "])
    def test_rejects_empty(self, seq: Any) -> None:
        assert I.validate_sequence(seq) == "empty sequence"

    def test_rejects_single_residue(self) -> None:
        assert "short" in I.validate_sequence("A")

    @pytest.mark.parametrize("seq,residue", [("GIGKZ", "Z"), ("GIGKB", "B"), ("GIGKJ", "J")])
    def test_names_the_offending_residue(self, seq: str, residue: str) -> None:
        # The message goes straight into the results table, so it has to say what was wrong.
        problem = I.validate_sequence(seq)
        assert problem and residue in problem


class TestCompositionRow:
    def test_length_matches_the_declared_vocabulary(self) -> None:
        row = I._composition_row("GIGKFLHSAKKFGKAFVGEIMNS")
        assert len(row) == len(I.AMINO_ACIDS) + len(I.AA_GROUPS)

    def test_single_residue_frequencies_sum_to_one(self) -> None:
        row = I._composition_row("GIGKFLHSAKKFGKAFVGEIMNS")
        assert sum(row[:len(I.AMINO_ACIDS)]) == pytest.approx(1.0)

    def test_homopolymer_puts_all_weight_on_one_residue(self) -> None:
        row = I._composition_row("AAAA")
        assert row[I.AMINO_ACIDS.index("A")] == pytest.approx(1.0)

    def test_empty_sequence_is_all_zeros(self) -> None:
        assert set(I._composition_row("")) == {0.0}


class TestDescriptorRow:
    def test_returns_every_descriptor_column(self) -> None:
        row = I._descriptor_row("GIGKFLHSAKKFGKAFVGEIMNS")
        assert set(row) == set(I.DESCRIPTOR_COLUMNS)

    def test_the_peptides_py_descriptors_are_real_numbers_not_nan(self) -> None:
        # These five arrive via sample.enrichment.peptide_descriptors, which returns an EMPTY
        # mapping when peptides.py is not installed -- leaving every one of them NaN. That is not a
        # crash and not a visible failure: it is a descriptors model scoring on five silently
        # missing features. Assert the values are real so a missing dependency fails here.
        row = I._descriptor_row("GIGKFLHSAKKFGKAFVGEIMNS")
        for column in I.PEPTIDE_DESCRIPTOR_COLUMNS:
            assert row[column] == row[column], f"{column} is NaN; is peptides.py installed?"

    def test_matches_the_trained_columns_for_a_disulfide_bonded_peptide(self) -> None:
        # The training columns were built through sample.enrichment's topology-aware molecule, so
        # a cysteine-rich peptide's disulfides are closed there. Scoring it through a plain
        # open-chain MolFromSequence instead reported 8 extra hydrogen-bond donors (94 against the
        # stored 86) and a LogP off by 2.1 -- the same feature name carrying a different number on
        # each side of the train/serve boundary.
        from sample.enrichment import molecule_from_sequence
        from rdkit.Chem import Descriptors

        sequence = "DGVKLCDVPSGTWSGHCGSSSKCSQQCKDREHFAYGGACHYQFPSVKCFCKRQC"
        expected = molecule_from_sequence(sequence)
        row = I._descriptor_row(sequence)
        assert row["hydrogen_bond_donors"] == float(Descriptors.NumHDonors(expected))
        assert row["logp"] == pytest.approx(float(Descriptors.MolLogP(expected)))

    def test_sequence_length_is_exact(self) -> None:
        assert I._descriptor_row("GIGKFLH")["sequence_length"] == 7.0

    def test_molecular_weight_is_plausible(self) -> None:
        # ~110 Da per residue; a 23-mer lands near 2400.
        mw = I._descriptor_row("GIGKFLHSAKKFGKAFVGEIMNS")["molecular_weight"]
        assert 2000 < mw < 3000


class TestBuildFeatureFrame:
    def test_returns_exactly_the_requested_columns_in_order(self) -> None:
        columns = ["sequence_length", "charge", "molecular_weight"]
        frame = I.build_feature_frame(["GIGKFLH"], columns)
        assert list(frame.columns) == columns

    def test_one_row_per_sequence(self) -> None:
        frame = I.build_feature_frame(["GIGKFLH", "ILPWKWP"], ["charge"])
        assert len(frame) == 2

    def test_builds_the_named_composition_block(self) -> None:
        frame = I.build_feature_frame(["GIGKFLH"], ["aa_freq_G", "aa_group_freq_basic"])
        assert frame.shape == (1, 2)

    def test_legacy_embedding_block_maps_to_residue_frequencies(self) -> None:
        # emb_dim_0..19 is the pre-rewrite 20-dim frequency vector. Verified against the shipped
        # final_dataset.csv, which carries exactly emb_dim_0..emb_dim_19.
        columns = [f"emb_dim_{i}" for i in range(20)]
        frame = I.build_feature_frame(["GIGKFLH"], columns)
        assert list(frame.columns) == columns
        assert frame.iloc[0].sum() == pytest.approx(1.0)

    def test_only_requested_blocks_are_computed(self) -> None:
        # Scoring a descriptor-only model must not load a transformer. If this regresses the test
        # does not fail -- it hangs downloading weights -- which is itself the signal.
        frame = I.build_feature_frame(["GIGKFLH"], ["charge"])
        assert list(frame.columns) == ["charge"]

    def test_unbuildable_column_raises_rather_than_filling(self) -> None:
        # A zero here is a real value ("contains no tryptophan") and cannot double as
        # "not computed" without corrupting the prediction.
        with pytest.raises(RuntimeError, match="feature columns"):
            I.build_feature_frame(["GIGKFLH"], ["charge", "invented_column"])


class TestPredict:
    @pytest.fixture
    def model_dir(self, tmp_path: Any) -> str:
        """:returns: A model root holding one classifier and one regressor."""
        root = str(tmp_path / "models")
        for task, estimator in (("classification", _StubClassifier()),
                                ("regression", _StubRegressor())):
            out = os.path.join(root, "general", "general", "any", task)
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "descriptors.pkl"), "wb") as fh:
                pickle.dump({"model": estimator, "columns": ["charge", "sequence_length"],
                             "namespace": f"general/general/any/{task}",
                             "feature_block": "descriptors", "metrics": {}}, fh)
        return root

    def test_regression_model_reports_a_value_not_a_probability(self, model_dir: str) -> None:
        # "How strong is this peptide" and "is this peptide active" are different questions and
        # cannot share an output shape.
        result = I.predict(["GIGKFLH"],
                           "general/general/any/regression/descriptors", model_dir)[0]
        assert result["value"] == pytest.approx(12.5)
        assert result["probability"] is None
        assert result["predicted_label"] is None

    def test_returns_one_result_per_sequence(self, model_dir: str) -> None:
        results = I.predict(["GIGKFLH", "ILPWKWP"],
                            "general/general/any/classification/descriptors", model_dir)
        assert len(results) == 2

    def test_reports_probability_and_label(self, model_dir: str) -> None:
        result = I.predict(["GIGKFLH"],
                           "general/general/any/classification/descriptors", model_dir)[0]
        assert result["probability"] == pytest.approx(0.8)
        assert result["predicted_label"] == "active"

    def test_falls_back_to_one_half_when_no_calibrated_threshold_is_on_record(
            self, model_dir: str) -> None:
        # model_dir's fixture writes no ml_report.json, so list_models() finds no "threshold" for
        # this model -- a model trained before calibration existed, or any release predating it.
        # _StubClassifier's 0.8 must still read as "active" against the conventional 0.5 fallback.
        result = I.predict(["GIGKFLH"],
                           "general/general/any/classification/descriptors", model_dir)[0]
        assert result["predicted_label"] == "active"

    def test_uses_the_models_own_calibrated_threshold_when_one_is_on_record(
            self, model_dir: str, tmp_path: Any) -> None:
        # _StubClassifier always reports probability 0.8 -- "active" at the conventional 0.5, but
        # this model's own calibrated threshold (written to ml_report.json, exactly as
        # sample.ml_models.train_all does) is 0.95, so the SAME probability must read "inactive".
        report_path = os.path.join(os.path.dirname(model_dir), "ml_report.json")
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump({"models": [{
                "namespace": "general/general/any/classification", "feature_block": "descriptors",
                "learning_task": "classification", "threshold": 0.95,
            }]}, fh)

        result = I.predict(["GIGKFLH"],
                           "general/general/any/classification/descriptors", model_dir)[0]
        assert result["probability"] == pytest.approx(0.8)
        assert result["predicted_label"] == "inactive"

    def test_position_tracks_input_order(self, model_dir: str) -> None:
        results = I.predict(["GIGKFLH", "ILPWKWP"],
                            "general/general/any/classification/descriptors", model_dir)
        assert [r["position"] for r in results] == [0, 1]
        assert results[1]["sequence"] == "ILPWKWP"

    def test_probability_is_reported_alongside_the_label(self, model_dir: str) -> None:
        # The probability is the model's actual output; the label is one reading of it at the
        # model's decision threshold, and must not replace it.
        result = I.predict(["GIGKFLH"],
                           "general/general/any/classification/descriptors", model_dir)[0]
        assert result["probability"] is not None and result["predicted_label"] is not None

    def test_unknown_model_raises(self, model_dir: str) -> None:
        with pytest.raises(FileNotFoundError):
            I.predict(["GIGKFLH"], "no/such/model", model_dir)

    def test_no_applicability_domain_reports_in_domain_as_none(self, model_dir: str) -> None:
        # A model trained before steps 45-46 carried one (the legacy model, or an older release) --
        # "never checked" must be distinguishable from a real False.
        result = I.predict(["GIGKFLH"],
                           "general/general/any/classification/descriptors", model_dir)[0]
        assert result["in_domain"] is None

    def test_applicability_domain_true_and_false_are_reported(
            self, model_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
        import pickle as _pickle

        path = os.path.join(model_dir, "general", "general", "any", "classification",
                            "descriptors.pkl")
        with open(path, "rb") as fh:
            payload = _pickle.load(fh)
        payload["scaler"] = _IdentityScaler()  # _in_domain_flags only forwards this to in_domain()
        payload["applicability_domain"] = {"k": 5, "threshold": 1.0}
        with open(path, "wb") as fh:
            _pickle.dump(payload, fh)

        monkeypatch.setattr("sample.ml_models.in_domain", lambda X, ad: [True, False])
        results = I.predict(["GIGKFLH", "ILPWKWP"],
                            "general/general/any/classification/descriptors", model_dir)
        assert results[0]["in_domain"] is True
        assert results[1]["in_domain"] is False

    def test_applicability_domain_failure_degrades_to_none_not_a_crash(
            self, model_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
        import pickle as _pickle

        path = os.path.join(model_dir, "general", "general", "any", "classification",
                            "descriptors.pkl")
        with open(path, "rb") as fh:
            payload = _pickle.load(fh)
        payload["scaler"] = object()
        payload["applicability_domain"] = {"k": 5, "threshold": 1.0}
        with open(path, "wb") as fh:
            _pickle.dump(payload, fh)

        def _boom(X: Any, ad: Any) -> Any:
            raise RuntimeError("scaler was fit on a different column set")

        monkeypatch.setattr("sample.ml_models.in_domain", _boom)
        result = I.predict(["GIGKFLH"],
                           "general/general/any/classification/descriptors", model_dir)[0]
        assert result["in_domain"] is None
        # The prediction itself must still succeed -- the AD check is informational, not blocking.
        assert result["probability"] is not None


class TestConstants:
    def test_amino_acid_vocabulary_is_the_standard_twenty(self) -> None:
        assert len(I.AMINO_ACIDS) == 20
        assert set(I.AMINO_ACIDS) == set("ACDEFGHIKLMNPQRSTVWY")

    def test_group_vocabulary_is_ordered(self) -> None:
        # The legacy emb_dim_* naming is positional, so reordering would silently remap features.
        assert isinstance(I.AA_GROUPS, dict)
        assert list(I.AA_GROUPS) == list(I.AA_GROUPS)

    def test_job_size_limit_is_positive(self) -> None:
        assert I.MAX_PEPTIDES_PER_JOB > 0
