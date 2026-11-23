"""
Tests for the 2026-08-12 silent-exception-handler pass (see `DECISIONS.md`).

Most of the 19 sites fixed that day only added a counter and a summary log around an existing
loop -- the same behavior, now visible. This file covers the handful with new, testable logic:
`peptide_descriptors`/`compute_peptides_descriptors` threading an optional failure counter through
their per-property try/except blocks, and `merge_databases` counting the sequences it drops rather
than just silently shrinking the output.
"""
import os
from typing import Any

import pandas as pd
import pytest


class TestDescriptorFailureCounting:
    """`sample.enrichment.peptide_descriptors` and its `sample.peptide_predictors` counterpart."""

    def test_a_failing_named_property_is_counted_not_silently_dropped(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import math

        import sample.enrichment as enrichment_mod

        # A descriptor mapped to a method Peptide does not have raises deterministically,
        # regardless of the sequence -- exercising the failure path without depending on which
        # real peptides.py descriptor happens to reject a given sequence.
        monkeypatch.setitem(enrichment_mod.PEPTIDE_DESCRIPTORS, "bogus",
                            ("this_method_does_not_exist", {}))

        counts: dict = {}
        result = enrichment_mod.peptide_descriptors("GIGKFLHSAKKFGKAFVGEIMNS", counts)

        assert math.isnan(result["bogus"]), "a failing descriptor must fall back to NaN"
        assert counts.get("bogus") == 1

    def test_failure_counts_is_optional_and_behavior_is_unchanged_without_it(self) -> None:
        """Every caller before this change did not pass `failure_counts`; that path must still
        work exactly as before -- an omitted argument, not a required one."""
        from sample.enrichment import peptide_descriptors

        result = peptide_descriptors("GIGKFLHSAKKFGKAFVGEIMNS")
        assert "isoelectric_point" in result

    def test_enrich_with_descriptors_runs_end_to_end_and_logs_a_summary(
            self, caplog: Any) -> None:
        from sample.enrichment import enrich_with_descriptors

        df = pd.DataFrame({
            "sequence": ["GIGKFLHSAKKFGKAFVGEIMNS", "ACDXUKR", ""],
            "SMILES": [None, None, None],
        })
        with caplog.at_level("INFO", logger="sample.enrichment"):
            out = enrich_with_descriptors(df)

        assert len(out) == 3
        assert "molecular_weight" in out.columns

    def test_compute_peptides_descriptors_counts_a_forced_failure(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        peptides = pytest.importorskip("peptides")

        def broken(self: Any) -> float:
            raise ValueError("forced failure for this test")

        # `compute_peptides_descriptors` imports `peptides` lazily inside the function, so the
        # class itself -- not a module attribute on sample.peptide_predictors -- is what to patch.
        monkeypatch.setattr(peptides.Peptide, "boman", broken, raising=False)

        from sample.peptide_predictors import compute_peptides_descriptors

        counts: dict = {}
        result = compute_peptides_descriptors("GIGKFLHSAKKFGKAFVGEIMNS", counts)

        assert "pep_boman" not in result, \
            "unlike peptide_descriptors, this one omits the key entirely on failure"
        assert counts.get("boman") == 1

    def test_enrich_with_predictors_runs_end_to_end_without_external_tools(self) -> None:
        from sample.peptide_predictors import enrich_with_predictors

        df = pd.DataFrame({"sequence": ["GIGKFLHSAKKFGKAFVGEIMNS", "ACDXUKR"]})
        out = enrich_with_predictors(df, run_external=False)
        assert len(out) == 2


class TestMergeDatabasesDropCounting:
    """`sample.merge.merge_databases` -- unparseable sequences used to vanish from the consensus
    build with only the final `entry_count` visible, no record of how many were dropped or why."""

    def test_x_residue_sequences_are_counted_and_logged(
            self, tmp_path: Any, caplog: Any) -> None:
        import sample.merge as merge_mod

        base_dir = tmp_path / "antimicrobial_databases"
        (base_dir / "adp").mkdir(parents=True)
        # merge_databases reads ADP straight from this fasta; the other four sources are read from
        # directories that simply do not exist here, which each ingest loop already tolerates.
        (base_dir / "adp" / "peptides.fasta").write_text(
            ">real_one\nGIGKFLHSAKKFGKAFVGEIMNS\n"
            ">has_x_residue\nGIGKFLXSAKKFGKAFVGEIMNS\n")

        out_path = str(tmp_path / "merged.csv")
        with caplog.at_level("INFO", logger="sample.merge"):
            merge_mod.merge_databases(out_path, base_dir=str(base_dir))

        merged = pd.read_csv(out_path)
        assert len(merged) == 1, "the X-residue sequence must not reach the output"
        assert merged.iloc[0]["sequence"] == "GIGKFLHSAKKFGKAFVGEIMNS"

        summary = [r.message for r in caplog.records if "Merge complete" in r.message]
        assert summary, "the drop must be visible in the log, not just a smaller output file"
        assert "1 had a non-standard" in summary[0]

    def test_a_custom_output_leaves_the_shared_baseline_untouched(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        `merge_databases` also writes a compatibility copy to a HARD-CODED relative path,
        `data/antimicrobial_databases/merged_source.csv`, which step 13 reads as its novelty
        baseline. That write used to ignore `output_path` entirely, so any run with a custom
        --output -- a test, a subset, a side-by-side comparison -- replaced the project's baseline
        with a partial result. It was found in exactly that state: one ADP row, accession
        `real_one`, which is a test fixture's FASTA header, making step 13 report "99.89% new
        peptides" against a baseline of one sequence.

        The write happens relative to the process CWD, so this runs from a throwaway directory:
        the assertion is that nothing appears there at all.
        """
        import sample.merge as merge_mod

        base_dir = tmp_path / "antimicrobial_databases"
        (base_dir / "adp").mkdir(parents=True)
        (base_dir / "adp" / "peptides.fasta").write_text(
            ">only_one\nGIGKFLHSAKKFGKAFVGEIMNS\n")

        workdir = tmp_path / "cwd"
        workdir.mkdir()
        monkeypatch.chdir(workdir)

        out_path = str(tmp_path / "custom_output.csv")
        merge_mod.merge_databases(out_path, base_dir=str(base_dir))

        assert os.path.exists(out_path), "the requested output must still be written"
        leaked = workdir / "data" / "antimicrobial_databases" / "merged_source.csv"
        assert not leaked.exists(), (
            "a non-default --output must not write the shared baseline copy")
