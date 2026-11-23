"""
Tests for :mod:`sample.ml_models` — the model namespace, scoring and the quality gate.

The scoring helpers are tested against cases whose correct answer is known independently (a
perfect ranker scores 1.0, a coin flip scores 0.5), so a metric wired to the wrong argument or
inverted shows up as a wrong number rather than merely a different one.

Most training runs are not exercised here: they fit hundreds of trees over thousands of rows. The
one exception is :class:`TestTrainAllEndToEnd`, a small (160-row) real run of the whole
:func:`~sample.ml_models.train_all` pipeline -- everything else in this file tests one function in
isolation, and nothing here catches a real integration break between them (a changed key name, a
column that stops lining up) the way one full run does.
"""
import json
import os
import pickle
from typing import Any, Dict, List

# `sample` must be imported before numpy/pandas: sample/__init__.py preloads sqlite3 to pin down
# which libstdc++ the process resolves against before a heavier native library grabs an
# incompatible one first -- see that module's docstring. Importing pandas/numpy here first defeats
# the guard when this file is run in isolation (`pytest tests/test_ml_models.py`).
from sample import ml_models as M

import numpy as np
import pandas as pd
import pytest


class TestSlug:
    @pytest.mark.parametrize("raw,expected", [
        ("Escherichia coli", "escherichia_coli"),
        ("MIC", "mic"),
        ("HeLa/S3", "hela_s3"),
        ("  spaced  ", "spaced"),
    ])
    def test_produces_filesystem_safe_lowercase(self, raw: str, expected: str) -> None:
        assert M._slug(raw) == expected

    @pytest.mark.parametrize("raw", ["", "   ", "///"])
    def test_degenerate_input_gets_a_placeholder(self, raw: str) -> None:
        # An empty path segment would collapse the namespace directory structure.
        assert M._slug(raw) == "unspecified"


class TestTaskSpecNamespace:
    def test_general_task_namespace(self) -> None:
        spec = M.TaskSpec("general", "general", "any", "", "classification", pd.Series())
        assert spec.namespace == "general/general/any/classification"

    def test_target_appears_only_when_scoped(self) -> None:
        scoped = M.TaskSpec("general", "organism_specific", "any", "Escherichia coli",
                            "classification", pd.Series())
        assert scoped.namespace == \
            "general/organism_specific/escherichia_coli/any/classification"

    def test_name_is_the_namespace_flattened(self) -> None:
        spec = M.TaskSpec("antibacterial", "general", "MIC", "", "classification", pd.Series())
        assert spec.name == spec.namespace.replace("/", "__")
        assert "/" not in spec.name

    def test_namespace_encodes_every_dimension(self) -> None:
        # The path IS the model's identity: what kind of peptide, tested against what, measured
        # how, predicted as what.
        spec = M.TaskSpec("anticancer", "cell_type_specific", "IC50", "HeLa",
                          "regression", pd.Series())
        assert spec.namespace == "anticancer/cell_type_specific/hela/ic50/regression"


class TestLearningTaskFor:
    @pytest.mark.parametrize("assay", ["MIC", "IC50", "EC50", "MBC", "HC50"])
    def test_potency_assays_are_regression(self, assay: str) -> None:
        # A published potency is a continuous value; predicting it is regression.
        assert M.learning_task_for(assay) == "regression"

    @pytest.mark.parametrize("assay", ["Hemolysis", "Bactericidal", "not-an-assay", ""])
    def test_everything_else_is_classification(self, assay: str) -> None:
        assert M.learning_task_for(assay) == "classification"

    def test_whitespace_is_tolerated(self) -> None:
        assert M.learning_task_for("  MIC  ") == "regression"


class TestBuildTasksCellCategories:
    def _dataset(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT", "DDDD"],
            "is_decoy": [0, 0, 0, 0, 1],
        })

    @staticmethod
    def _potent(sequences: List[str], assay: str = "IC50", um: float = 1.0) -> pd.DataFrame:
        # All below ACTIVITY_THRESHOLD_UM (5 uM default), so every row counts as active.
        return pd.DataFrame({"sequence": sequences, "activity_type": assay,
                             "value_lower": um, "activity_unit": "uM"})

    def test_pools_cell_lines_by_cellosaurus_category(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # Three different exact cell lines, all breast cancer -- individually far below
        # MIN_POSITIVES_PER_TASK*2, but the pooled cancer_type_specific task should see all three.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 2)
        monkeypatch.setattr(
            "sample.cell_lines.build_cell_line_categories",
            lambda names, **k: {n: ("breast_cancer", True) for n in names})

        activities = self._potent(["AAAA", "CCCC", "GGGG"])
        activities["target_cell_line"] = ["MCF-7", "MDA-MB-231", "SK-BR-3"]
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        tasks = M.build_tasks(self._dataset(), str(activities_csv))
        cancer_tasks = [t for t in tasks if t.scope == "cancer_type_specific"]
        assert len(cancer_tasks) == 1
        assert cancer_tasks[0].target == "breast_cancer"
        assert cancer_tasks[0].assay_type == "IC50"
        assert int((cancer_tasks[0].y == 1).sum()) == 3

    def test_below_threshold_category_is_not_a_task(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 60)
        monkeypatch.setattr(
            "sample.cell_lines.build_cell_line_categories",
            lambda names, **k: {n: ("breast_cancer", True) for n in names})

        activities = self._potent(["AAAA"])
        activities["target_cell_line"] = ["MCF-7"]
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        tasks = M.build_tasks(self._dataset(), str(activities_csv))
        assert not [t for t in tasks if t.scope == "cancer_type_specific"]

    def test_measured_but_not_potent_enough_is_not_a_positive(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # A real IC50 was measured, but it sits well above ACTIVITY_THRESHOLD_UM -- "was tested"
        # is not "was active", and this must not count as a positive on either axis.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 1)
        monkeypatch.setattr(
            "sample.cell_lines.build_cell_line_categories",
            lambda names, **k: {n: ("breast_cancer", True) for n in names})

        activities = self._potent(["AAAA"], um=500.0)
        activities["target_cell_line"] = ["MCF-7"]
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        tasks = M.build_tasks(self._dataset(), str(activities_csv))
        assert not [t for t in tasks if t.scope in ("cancer_type_specific", "cell_type_specific")]

    def test_unclassified_cell_lines_do_not_create_a_cancer_type_task(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 1)
        monkeypatch.setattr("sample.cell_lines.build_cell_line_categories", lambda names, **k: {})

        activities = self._potent(["AAAA"])
        activities["target_cell_line"] = ["SomeObscureLine"]
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        tasks = M.build_tasks(self._dataset(), str(activities_csv))
        assert not [t for t in tasks if t.scope == "cancer_type_specific"]
        # The exact-cell-line axis still sees it (at its own, higher threshold) -- being
        # unclassified by Cellosaurus only removes it from the pooled category, not entirely.
        assert "cell_lines" in M._label_lookup(str(activities_csv))

    def test_homo_sapiens_anticancer_rows_become_a_cancer_task_not_an_organism_one(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # DBAASP/DRAMP record some anticancer IC50s with target_organism="Homo sapiens" and no
        # cell line at all -- real signal, but "organism_specific/Homo sapiens" would misread as
        # this peptide's activity against humans (toxicity), not what it actually is (cancer
        # potency with unspecified cell line).
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 1)
        activities = self._potent(["AAAA", "CCCC", "GGGG", "TTTT"])
        activities["target_group"] = "anticancer"
        activities["target_organism"] = "Homo sapiens"
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        tasks = M.build_tasks(self._dataset(), str(activities_csv))
        assert not [t for t in tasks if t.scope == "organism_specific"]
        cancer_tasks = [t for t in tasks if t.scope == "cancer_type_specific"
                       and t.task == "classification"]
        assert len(cancer_tasks) == 1
        assert cancer_tasks[0].target == "cancer_unspecified_cell_line"
        assert cancer_tasks[0].assay_type == "IC50"
        assert int((cancer_tasks[0].y == 1).sum()) == 4

    def test_homo_sapiens_is_excluded_from_organisms_only_never_from_other_groups_wrongly_labeled_cancer(
            self, tmp_path: Any) -> None:
        # A non-anticancer "Homo sapiens" row (e.g. a general LD50 toxicity assay, target_group
        # "other") is real host-toxicity data, not cancer potency -- it must not be mislabeled
        # into the cancer_unspecified_cell_line bucket just because the organism string matches.
        activities = pd.DataFrame({
            "sequence": ["AAAA"],
            "target_group": "other",
            "target_organism": "Homo sapiens",
        })
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        labels = M._label_lookup(str(activities_csv))
        organisms = {v for values in labels["organisms"].values() for v in values}
        assert "Homo sapiens" not in organisms
        cell_cats = {v for values in labels["cell_categories"].values() for v in values}
        assert "cancer_unspecified_cell_line" not in cell_cats


class TestMedianBasedClassificationThreshold:
    """Classification activity is decided on the MEDIAN of a peptide's repeat measurements for the
    same (target, assay) pair, not on whichever single measurement happens to be most potent."""

    def test_median_of_repeat_measurements_decides_activity_not_any_single_reading(
            self, tmp_path: Any) -> None:
        # AAAA: MIC vs E. coli reported as 2, 8, 20 uM -- median 8 uM, above the 5 uM threshold,
        # so INACTIVE, even though one individual reading (2 uM) would have crossed it alone.
        # BBBB: 1, 2, 9 uM -- median 2 uM, ACTIVE.
        activities = pd.DataFrame({
            "sequence": ["AAAA", "AAAA", "AAAA", "BBBB", "BBBB", "BBBB"],
            "activity_type": "MIC",
            "value_lower": [2.0, 8.0, 20.0, 1.0, 2.0, 9.0],
            "activity_unit": "uM",
            "target_organism": "Escherichia coli",
        })
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        labels = M._label_lookup(str(activities_csv))
        pair = ("Escherichia coli", "MIC")
        assert pair not in labels["active_organism_assay"].get("AAAA", set())
        assert pair in labels["active_organism_assay"].get("BBBB", set())
        # The regression side keeps every individual measurement, unaffected by the
        # classification decision -- median-aggregation for regression happens later, in
        # build_tasks, on this same full list.
        assert sorted(labels["potency_organism_assay"]["AAAA"][pair]) == sorted(
            [-np.log10(v * 1e-6) for v in (2.0, 8.0, 20.0)])

    def test_median_also_applies_through_the_cell_line_category_pathway(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # Regression test for a real ordering bug: active_cell_category_assay is built by
        # relabeling active_cell_line_assay through the Cellosaurus category map, so it must see
        # the MEDIAN-decided cell-line activity, not run before that decision exists. A cell line
        # with the same straddling measurements as AAAA above (median 8 uM, inactive) must stay
        # out of the pooled cancer-type task, not be pooled in on the strength of one reading.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 1)
        monkeypatch.setattr(
            "sample.cell_lines.build_cell_line_categories",
            lambda names, **k: {n: ("breast_cancer", True) for n in names})

        activities = pd.DataFrame({
            "sequence": ["AAAA", "AAAA", "AAAA"],
            "activity_type": "IC50",
            "value_lower": [2.0, 8.0, 20.0],
            "activity_unit": "uM",
            "target_cell_line": "MCF-7",
        })
        activities_csv = tmp_path / "activities.csv"
        activities.to_csv(activities_csv, index=False)

        labels = M._label_lookup(str(activities_csv))
        pair = ("MCF-7", "IC50")
        assert pair not in labels["active_cell_line_assay"].get("AAAA", set())
        assert ("breast_cancer", "IC50") not in labels["active_cell_category_assay"].get(
            "AAAA", set())


class TestBuildTasksTargetScoping:
    def _dataset(self, n_real: int = 40) -> pd.DataFrame:
        seqs = [f"{'A' * (i % 5 + 6)}{'C' * (i % 3)}" for i in range(n_real)]
        rows = [{"sequence": s, "is_decoy": 0} for s in seqs]
        rows += [{"sequence": "D" * (i % 5 + 6), "is_decoy": 1} for i in range(n_real)]
        return pd.DataFrame(rows)

    def _activities(self, df: pd.DataFrame, assay: str, um: float = 1.0, **extra: Any) -> pd.DataFrame:
        real = df[df["is_decoy"] == 0]["sequence"]
        row = {"sequence": real, "activity_type": assay, "value_lower": um,
              "activity_unit": "uM", "target_group": "antibacterial"}
        row.update(extra)
        return pd.DataFrame(row)

    def test_every_task_is_classification_or_regression_never_anything_else(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        activities_csv = tmp_path / "activities.csv"
        self._activities(df, "MIC", target_organism="Escherichia coli").to_csv(
            activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv))
        assert tasks  # sanity: the run actually produced something
        assert all(t.task in ("classification", "regression") for t in tasks)

    def test_organism_specific_mic_gets_both_a_classification_and_a_regression_task(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # Varied potencies (not every peptide at the exact same value) so the regression target
        # carries real variance, matching how it will actually look on real data.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        real = df[df["is_decoy"] == 0]["sequence"]
        rows = pd.DataFrame({
            "sequence": real, "activity_type": "MIC",
            "value_lower": [1.0 + (i % 10) for i in range(len(real))],
            "activity_unit": "uM", "target_group": "antibacterial",
            "target_organism": "Escherichia coli",
        })
        activities_csv = tmp_path / "activities.csv"
        rows.to_csv(activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv), min_target_examples=5)
        mic_tasks = {t.task: t for t in tasks
                    if t.scope == "organism_specific" and t.assay_type == "MIC"}
        assert set(mic_tasks) == {"classification", "regression"}
        # Classification: 1.0-4.0 uM values (i % 10 in 0..3) are below the 5 uM bar, 5.0-10.0
        # are not -- only the first group is positive.
        assert int((mic_tasks["classification"].y == 1).sum()) < len(real)
        # Regression: every real peptide with a measurement gets a target, active or not.
        assert int(mic_tasks["regression"].y.notna().sum()) == len(real)

    def test_regression_target_is_potency_not_raw_value(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        # A single fixed value (5 uM), so every peptide's regression target should be exactly
        # -log10(5e-6), not the raw 5.0 or a decoy-shaped 0/1 label.
        self._activities(df, "MIC", um=5.0, target_organism="Escherichia coli").to_csv(
            tmp_path / "activities.csv", index=False)

        reg = next(t for t in M.build_tasks(df, str(tmp_path / "activities.csv"),
                                            min_target_examples=5)
                  if t.scope == "organism_specific" and t.task == "regression")
        import math
        expected = -math.log10(5.0e-6)
        assert reg.y.dropna().to_numpy() == pytest.approx(expected)
        # Decoys have no measured potency -- NaN, not a fabricated value.
        assert reg.y[df["is_decoy"] == 1].isna().all()

    def test_assay_with_no_target_at_all_gets_no_pooled_task(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # MIC/IC50/... with neither a target organism nor a target cell line is exactly the
        # "generic per-metric" model this axis must not build -- an assay this common only ever
        # earns a task when it is scoped to something. Applies to regression too: there is no
        # "regress on MIC against no one in particular" task either.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        activities_csv = tmp_path / "activities.csv"
        self._activities(df, "MIC").to_csv(activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv))
        assert not [t for t in tasks if t.assay_type.lower() == "mic"]

    def test_organism_specific_tasks_are_split_one_assay_at_a_time(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # A peptide active by MIC against E. coli and a peptide active by MBC against E. coli must
        # not be pooled into one "active against E. coli, any assay" task.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        mic = self._activities(df, "MIC", target_organism="Escherichia coli")
        mbc = self._activities(df, "MBC", target_organism="Escherichia coli")
        activities_csv = tmp_path / "activities.csv"
        pd.concat([mic, mbc]).to_csv(activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv), min_target_examples=5)
        organism_tasks = [t for t in tasks
                          if t.scope == "organism_specific" and t.task == "classification"]
        namespaces = {t.namespace for t in organism_tasks}
        assert any(ns.endswith("/mic/classification") for ns in namespaces)
        assert any(ns.endswith("/mbc/classification") for ns in namespaces)
        # Each is its own task, not one task carrying both.
        assert len(organism_tasks) == 2

    def test_hemolysis_gets_its_own_peptide_type_not_a_generic_toxicity_bucket(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        activities_csv = tmp_path / "activities.csv"
        self._activities(df, "Hemolysis", target_group="other").to_csv(
            activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv))
        hemo = [t for t in tasks if t.assay_type == "Hemolysis" and t.task == "classification"]
        assert len(hemo) == 1
        assert hemo[0].peptide_type == "hemolytic"
        assert hemo[0].scope == "general"
        assert hemo[0].target == ""

    def test_systemic_toxicity_assay_gets_a_no_target_toxicity_task(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # LD50 against "Homo sapiens" (no cell line) is systemic toxicity, not a per-organism
        # potency question -- it belongs on the no-target toxicity axis.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        activities_csv = tmp_path / "activities.csv"
        self._activities(df, "LD50", target_group="other",
                         target_organism="Homo sapiens").to_csv(activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv))
        toxicity = [t for t in tasks if t.peptide_type == "toxicity" and t.task == "classification"]
        assert len(toxicity) == 1
        assert toxicity[0].assay_type == "LD50"
        assert toxicity[0].target == ""
        assert toxicity[0].scope == "general"

    def test_ld50_against_a_real_organism_is_organism_specific_not_toxicity(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # The same assay (LD50), scored against an actual pathogen rather than the human host, is
        # a target-specific lethality question and belongs with the other organism-specific tasks.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        activities_csv = tmp_path / "activities.csv"
        self._activities(df, "LD50", target_group="antiparasitic",
                         target_organism="Trypanosoma cruzi").to_csv(activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv), min_target_examples=5)
        assert not [t for t in tasks if t.peptide_type == "toxicity"]
        organism_tasks = [t for t in tasks
                          if t.scope == "organism_specific" and t.task == "classification"]
        assert len(organism_tasks) == 1
        assert organism_tasks[0].target == "Trypanosoma cruzi"
        assert organism_tasks[0].assay_type == "LD50"

    def test_strain_qualified_organism_names_pool_into_one_species_level_task(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # "Escherichia coli", "...W" and "...C" are the same organism_specific question --
        # fragmenting them into three separate tasks would silently drop real signal that
        # individually never clears the per-target threshold.
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        variants = pd.concat([
            self._activities(df.iloc[:14], "MIC", target_organism="Escherichia coli"),
            self._activities(df.iloc[14:27], "MIC", target_organism="Escherichia coli W"),
            self._activities(df.iloc[27:40], "MIC", target_organism="Escherichia coli C"),
        ])
        activities_csv = tmp_path / "activities.csv"
        variants.to_csv(activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv), min_target_examples=5)
        ecoli = [t for t in tasks if t.scope == "organism_specific" and t.assay_type == "MIC"
                and t.task == "classification"]
        assert len(ecoli) == 1
        assert ecoli[0].target == "Escherichia coli"
        assert int((ecoli[0].y == 1).sum()) == 40

    def test_measured_above_threshold_is_not_a_classification_positive_but_still_regresses(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 5)
        df = self._dataset()
        activities_csv = tmp_path / "activities.csv"
        # Well above ACTIVITY_THRESHOLD_UM (5 uM) -- measured, but not active by this criterion.
        self._activities(df, "MIC", um=500.0, target_organism="Escherichia coli").to_csv(
            activities_csv, index=False)

        tasks = M.build_tasks(df, str(activities_csv), min_target_examples=5)
        organism_tasks = {t.task: t for t in tasks if t.scope == "organism_specific"}
        # No classification positive -- 500 uM never clears the activity bar.
        assert "classification" not in organism_tasks
        # But the measurement is still real potency data, worth regressing on regardless of
        # whether it counts as "active" by the classification bar.
        assert "regression" in organism_tasks
        assert int(organism_tasks["regression"].y.notna().sum()) == 40


class TestRegressionHelpers:
    def test_make_model_regression_has_predict_but_not_predict_proba(self) -> None:
        model = M._make_model(is_regression=True)
        assert hasattr(model, "predict")
        assert not hasattr(model, "predict_proba")

    def test_make_model_classification_has_predict_proba(self) -> None:
        assert hasattr(M._make_model(is_regression=False), "predict_proba")

    def test_primary_metric(self) -> None:
        assert M._primary_metric(is_regression=True) == "r2"
        assert M._primary_metric(is_regression=False) == "roc_auc"

    def test_predict_scores_returns_values_for_a_regressor(self) -> None:
        import numpy as np

        class _Reg:
            def predict(self, X: Any) -> Any:
                return np.array([1.5, 2.5])
        out = M._predict_scores(_Reg(), np.zeros((2, 3)), is_regression=True)
        assert list(out) == [1.5, 2.5]

    def test_bootstrap_samples_regression_reports_regression_metrics(self) -> None:
        import numpy as np
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 50)
        y_pred = y_true + rng.normal(0, 0.1, 50)  # good predictions
        samples = M._bootstrap_samples(y_true, y_pred, n_resamples=40, is_regression=True)
        assert "r2" in samples and "pearson_r" in samples and "mae" in samples
        assert "roc_auc" not in samples

    def test_bootstrap_samples_regression_skips_zero_variance_resamples(self) -> None:
        import numpy as np
        # All-identical y_true: every resample has zero variance, R^2 undefined -- must skip, not
        # crash or emit inf.
        y_true = np.full(20, 3.0)
        samples = M._bootstrap_samples(y_true, np.arange(20.0), n_resamples=20, is_regression=True)
        assert samples == {}


class TestRegressionQualityGate:
    def _record(self, **overrides: Any) -> Dict[str, Any]:
        record = {
            "learning_task": "regression",
            "holdout": {"r2": 0.5},
            "holdout_ci95": {"r2": [0.3, 0.7]},
            "y_randomization": {"mean_r2": 0.0},
            "null_comparison": {"r2": {"primary_test": "kruskal", "primary_pvalue": 0.001}},
        }
        record.update(overrides)
        return record

    def test_a_good_regression_model_passes(self) -> None:
        passed, reasons = M.assess_quality(self._record())
        assert passed and reasons == []

    def test_low_holdout_r2_fails(self) -> None:
        passed, reasons = M.assess_quality(self._record(holdout={"r2": 0.05}))
        assert not passed and any("R2" in r for r in reasons)

    def test_y_randomized_r2_above_zero_fails(self) -> None:
        passed, reasons = M.assess_quality(self._record(y_randomization={"mean_r2": 0.4}))
        assert not passed and any("leakage" in r for r in reasons)

    def test_regression_record_is_not_judged_by_the_classification_gate(self) -> None:
        # A regression record has no roc_auc; assess_quality must route it to the R^2 gate, not
        # reject it for a missing ROC-AUC it was never going to have.
        passed, reasons = M.assess_quality(self._record())
        assert passed
        assert not any("ROC-AUC" in r for r in reasons)


class TestFeatureBlocks:
    """
    ESM-2 is deliberately excluded here -- this project's own comparison found it ahead on
    ROC-AUC/F1/MCC but consistently behind on recall against both the fixed-hyperparameter
    XGBoost and a full ESM-2 fine-tune, on the two tasks with a real external-tool comparison.
    These tests confirm descriptors/composition/dpc/ctd are the only blocks ever returned, and
    that ESM-2 (and anything else) is correctly ignored rather than merged in.
    """

    def test_identifies_the_descriptors_block(self) -> None:
        frame = pd.DataFrame({"molecular_weight": [1.0], "charge": [2.0]})
        assert M.feature_blocks(frame) == {"descriptors": ["molecular_weight", "charge"]}

    def test_descriptors_block_is_named_explicitly_not_globbed(self) -> None:
        # The `pep_*`/`stab_*` columns are a decoy-leakage trap (100% populated for real
        # peptides, 0% for decoys -- see feature_blocks' own docstring); they must never appear
        # in the descriptors block, no matter how close their names look to the real ones.
        frame = pd.DataFrame({
            "molecular_weight": [1.0], "pep_molecular_weight": [1.0], "pep_charge": [2.0],
            "stab_isoelectric_point": [7.0],
        })
        assert M.feature_blocks(frame) == {"descriptors": ["molecular_weight"]}

    def test_identifies_the_composition_block(self) -> None:
        frame = pd.DataFrame({"aa_freq_A": [0.1], "aa_group_freq_basic": [0.2]})
        assert M.feature_blocks(frame) == {"composition": ["aa_freq_A", "aa_group_freq_basic"]}

    def test_identifies_the_dpc_and_ctd_blocks(self) -> None:
        frame = pd.DataFrame({"dpc_AA": [0.1], "dpc_AC": [0.2],
                              "ctd_comp_charge_g1": [0.3], "ctd_tran_charge_12": [0.4]})
        assert M.feature_blocks(frame) == {
            "dpc": ["dpc_AA", "dpc_AC"],
            "ctd": ["ctd_comp_charge_g1", "ctd_tran_charge_12"]}

    def test_identifies_every_block_together(self) -> None:
        frame = pd.DataFrame({"molecular_weight": [1.0], "aa_freq_A": [0.1],
                              "dpc_AA": [0.2], "ctd_comp_charge_g1": [0.3]})
        assert M.feature_blocks(frame) == {
            "descriptors": ["molecular_weight"], "composition": ["aa_freq_A"],
            "dpc": ["dpc_AA"], "ctd": ["ctd_comp_charge_g1"]}

    def test_esm2_and_other_columns_are_ignored(self) -> None:
        frame = pd.DataFrame({
            "molecular_weight": [1.0], "aa_freq_A": [0.1], "protbert_dim_0": [0.5],
            "esm2_dim_0": [0.3],
        })
        assert M.feature_blocks(frame) == {
            "descriptors": ["molecular_weight"], "composition": ["aa_freq_A"]}

    def test_no_recognized_columns_yields_no_blocks(self) -> None:
        assert M.feature_blocks(pd.DataFrame({"sequence": ["GIGK"]})) == {}
        assert M.feature_blocks(pd.DataFrame({"esm2_dim_0": [0.3]})) == {}


class TestAddDerivedFeatureColumns:
    def test_adds_dpc_and_ctd_columns_matching_the_row_builders(self) -> None:
        from sample.inference import CTD_COLUMNS, DPC_COLUMNS, _ctd_row, _dpc_row

        df = pd.DataFrame({"sequence": ["KWKLFKKIGAVLKVL", "GIGKFLHSAKKFGKA"]})
        out = M.add_derived_feature_columns(df)
        assert "dpc_AA" in out.columns and "ctd_comp_charge_g1" in out.columns
        for i, seq in enumerate(df["sequence"]):
            expected_dpc = _dpc_row(seq)
            expected_ctd = _ctd_row(seq)
            got_dpc = out.loc[i, list(DPC_COLUMNS)].to_numpy(dtype=float)
            got_ctd = out.loc[i, list(CTD_COLUMNS)].to_numpy(dtype=float)
            assert np.allclose(got_dpc, expected_dpc, atol=1e-5)
            assert np.allclose(got_ctd, expected_ctd, atol=1e-5)

    def test_is_idempotent_when_columns_already_present(self) -> None:
        df = pd.DataFrame({"sequence": ["KWKLFKKIGAVLKVL"]})
        once = M.add_derived_feature_columns(df)
        twice = M.add_derived_feature_columns(once)
        assert twice is once  # returned unchanged, not recomputed


class TestClassificationScores:
    def test_perfect_ranking_scores_one(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9])
        scores = M._scores(y_true, y_prob)
        assert scores["roc_auc"] == pytest.approx(1.0)
        assert scores["accuracy"] == pytest.approx(1.0)

    def test_inverted_ranking_scores_zero_auc(self) -> None:
        # Catches a metric computed with the arguments the wrong way round.
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        assert M._scores(y_true, y_prob)["roc_auc"] == pytest.approx(0.0)

    def test_uninformative_scores_are_chance(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        assert M._scores(y_true, y_prob)["roc_auc"] == pytest.approx(0.5)

    def test_reports_the_requested_metric_set(self) -> None:
        scores = M._scores(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        assert {"accuracy", "balanced_accuracy", "recall", "precision", "f1",
                "roc_auc"} <= set(scores)

    def test_all_metrics_are_finite_floats(self) -> None:
        scores = M._scores(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        for name, value in scores.items():
            assert isinstance(value, float), name
            assert np.isfinite(value), name

    def test_single_class_does_not_raise(self) -> None:
        # A fold can land on one class; that must degrade, not abort the run.
        scores = M._scores(np.array([1, 1, 1]), np.array([0.6, 0.7, 0.8]))
        assert isinstance(scores, dict)

    def test_a_non_default_threshold_changes_the_call(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.4, 0.6, 0.7])
        at_half = M._scores(y_true, y_prob, threshold=0.5)
        at_low = M._scores(y_true, y_prob, threshold=0.35)
        # Lowering the cutoff can only call MORE rows positive, never fewer.
        assert at_low["recall"] >= at_half["recall"]


class TestBestF1Threshold:
    def test_matches_a_brute_force_grid_search(self) -> None:
        from sklearn.metrics import f1_score

        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, 200)
        y_prob = np.clip(y_true * 0.6 + rng.normal(0, 0.25, 200) + 0.2, 0, 1)

        threshold = M._best_f1_threshold(y_true, y_prob)
        chosen_f1 = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        best_grid_f1 = max(
            f1_score(y_true, (y_prob >= thr).astype(int), zero_division=0)
            for thr in np.arange(0.01, 1.0, 0.01))
        assert chosen_f1 >= best_grid_f1 - 1e-9

    def test_single_class_falls_back_to_one_half(self) -> None:
        assert M._best_f1_threshold(np.array([1, 1, 1]), np.array([0.2, 0.6, 0.9])) == 0.5

    def test_positive_scores_clustered_low_pull_the_threshold_down(self) -> None:
        # The exact shape scale_pos_weight leaves behind: the training loss is corrected for
        # imbalance, but the fitted decision boundary is not, so positives can still cluster under
        # 0.5. The calibrated cutoff should follow the data, not stay pinned at 0.5.
        y_true = np.array([0] * 80 + [1] * 20)
        y_prob = np.concatenate([np.full(80, 0.1), np.full(20, 0.3)])
        assert M._best_f1_threshold(y_true, y_prob) < 0.5


class TestOofCalibrateThreshold:
    """
    _oof_calibrate_threshold is fit_and_validate's shared OOF-pooling core -- both the fixed-
    hyperparameter path's real per-fold refits and a cheap-refit-capable architecture's frozen
    refits call through this one function, so a bug here would affect both calibration paths.
    """

    def _signal_data(self, n: int = 200, seed: int = 0) -> Any:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        X = np.column_stack([y * 3 + rng.normal(0, 1, n),
                             rng.normal(0, 1, n)]).astype(np.float32)
        groups = np.arange(n)
        return X, y, groups

    def test_below_two_folds_returns_one_half_without_fitting_anything(self) -> None:
        calls: List[Any] = []
        factory = lambda **kw: calls.append(1) or M._make_model(**kw)  # noqa: E731
        X, y, groups = self._signal_data()
        assert M._oof_calibrate_threshold(X, y, groups, factory, {}, 1) == 0.5
        assert not calls

    def test_pools_out_of_fold_predictions_across_every_fold(self) -> None:
        X, y, groups = self._signal_data()
        threshold = M._oof_calibrate_threshold(X, y, groups, M._make_model, {}, 5)
        # Real, separable signal -- the pooled OOF threshold should land somewhere sane, not at
        # a degenerate extreme (0.0/1.0) or the untouched default.
        assert 0.0 < threshold < 1.0

    def test_a_custom_factory_is_used_for_every_fold_not_make_model(self) -> None:
        calls: List[Any] = []

        def counting_factory(is_regression: bool = False, **kw: Any) -> Any:
            calls.append(1)
            return M._make_model(is_regression=is_regression, **kw)

        X, y, groups = self._signal_data()
        M._oof_calibrate_threshold(X, y, groups, counting_factory, {}, 5)
        assert len(calls) == 5  # one fit per fold, all through the given factory

    def test_too_few_groups_degrades_to_one_half_not_a_crash(self) -> None:
        X = np.zeros((3, 2), dtype=np.float32)
        y = np.array([0, 1, 1])
        groups = np.array([0, 1, 2])
        assert M._oof_calibrate_threshold(X, y, groups, M._make_model, {}, 5) == 0.5


class TestRegressionScores:
    def test_perfect_prediction(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        scores = M._regression_scores(y, y.copy())
        assert scores["r2"] == pytest.approx(1.0)
        assert scores["rmse"] == pytest.approx(0.0)
        assert scores["mae"] == pytest.approx(0.0)

    def test_correlations_are_reported(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        scores = M._regression_scores(y_true, y_pred)
        assert scores["pearson_r"] > 0.95
        assert scores["spearman_r"] > 0.95

    def test_reports_the_requested_metric_set(self) -> None:
        scores = M._regression_scores(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert {"r2", "pearson_r", "spearman_r", "mae", "rmse"} <= set(scores)

    def test_worse_prediction_scores_lower(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        good = M._regression_scores(y_true, np.array([1.1, 2.1, 3.1, 4.1]))
        bad = M._regression_scores(y_true, np.array([4.0, 1.0, 3.0, 2.0]))
        assert good["r2"] > bad["r2"]
        assert good["rmse"] < bad["rmse"]


class TestQualityGate:
    def _record(self, **overrides: Any) -> Dict[str, Any]:
        """
        :param overrides: Fields to replace on the passing baseline.
        :returns: A model record that clears the default gate unless overridden.
        """
        record = {
            "holdout": {"roc_auc": 0.85},
            "holdout_ci95": {"roc_auc": [0.72, 0.90]},
            "y_randomization": {"mean_roc_auc": 0.50},
            "n_samples": 500,
        }
        record.update(overrides)
        return record

    def test_default_thresholds_are_ordered_sensibly(self) -> None:
        gate = M.DEFAULT_QUALITY_GATE
        assert gate.min_holdout_auc > gate.max_y_random_auc
        assert gate.min_ci_lower < gate.min_holdout_auc

    def test_a_good_model_passes(self) -> None:
        passed, reasons = M.assess_quality(self._record())
        assert passed and reasons == []

    def test_low_holdout_auc_fails(self) -> None:
        passed, reasons = M.assess_quality(self._record(holdout={"roc_auc": 0.60}))
        assert not passed and reasons

    def test_y_randomization_above_chance_fails(self) -> None:
        # A model that scores well on SHUFFLED labels is reading structure, not biology --
        # duplicated sequences across folds, or leakage. This is the most important gate.
        passed, reasons = M.assess_quality(
            self._record(y_randomization={"mean_roc_auc": 0.80}))
        assert not passed and reasons

    def test_wide_confidence_interval_fails(self) -> None:
        passed, reasons = M.assess_quality(self._record(holdout_ci95={"roc_auc": [0.40, 0.99]}))
        assert not passed and reasons

    def test_failure_reasons_are_reported(self) -> None:
        # The reasons go into the report so a rejected model can be understood, not just counted.
        _, reasons = M.assess_quality(self._record(holdout={"roc_auc": 0.10}))
        assert reasons and all(isinstance(r, str) and r.strip() for r in reasons)

    def test_a_custom_gate_is_honoured(self) -> None:
        strict = M.QualityGate(min_holdout_auc=0.99,
                               min_ci_lower=0.99, max_y_random_auc=0.01)
        assert M.assess_quality(self._record(), strict)[0] is False


class TestConstants:
    def test_regression_assays_are_potency_metrics(self) -> None:
        assert {"MIC", "IC50", "EC50"} <= set(M.REGRESSION_ASSAYS)

    def test_minimum_positives_is_a_meaningful_floor(self) -> None:
        # A per-target model trained on a handful of peptides reports a number driven by fold
        # assignment rather than by biology.
        assert M.MIN_POSITIVES_PER_TASK >= 10


class TestNormalityTest:
    def test_gaussian_sample_is_not_rejected(self) -> None:
        values = np.random.default_rng(1).normal(0, 1, 200)
        result = M._normality_test(values)
        assert result["is_normal"] is True
        assert result["pvalue"] > 0.05

    def test_clearly_bimodal_sample_is_rejected(self) -> None:
        # Two tight clumps far apart -- about as far from a bell curve as a sample gets.
        values = np.concatenate([np.zeros(100), np.full(100, 100.0)])
        result = M._normality_test(values)
        assert result["is_normal"] is False
        assert result["pvalue"] < 0.05

    def test_too_few_values_reports_not_normal_without_raising(self) -> None:
        # Shapiro-Wilk is undefined below n=3; this must degrade, not crash a whole training run.
        result = M._normality_test(np.array([0.1, 0.2]))
        assert result["is_normal"] is False
        assert np.isnan(result["pvalue"])

    def test_constant_array_reports_not_normal_without_raising(self) -> None:
        # Zero variance: Shapiro-Wilk divides by the sample standard deviation.
        result = M._normality_test(np.array([0.5, 0.5, 0.5, 0.5]))
        assert result["is_normal"] is False
        assert np.isnan(result["pvalue"])

    def test_non_finite_values_are_dropped_before_testing(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 1.5, 2.5, 1.8, np.inf])
        result = M._normality_test(values)
        assert result["n"] == 5


class TestClusterGroups:
    def test_real_peptides_share_a_group_when_clustered_decoys_are_singletons(
            self, monkeypatch: Any) -> None:
        df = pd.DataFrame({
            "sequence": ["AAAAAAAAAA", "AAAAAAAAAA", "CCCCCCCCCC", "DDDDDDDDDD", "EEEEEEEEEE"],
            "is_decoy": [0, 0, 0, 1, 1],
        })

        # The clustering backend is exercised for real elsewhere (sample.families' own tests);
        # what this test checks is _cluster_groups' bookkeeping around it -- broadcasting a cluster
        # id to every row sharing a sequence, and keeping decoys out of the real-peptide space.
        def fake_cluster(sequences: Dict[str, str], identity: float = 0.9,
                         **kwargs: Any) -> Dict[str, int]:
            return {key: (1 if seq.startswith("A") else 2) for key, seq in sequences.items()}

        monkeypatch.setattr("sample.families.cdhit_available", lambda: False)
        monkeypatch.setattr("sample.families.cluster", fake_cluster)
        groups = M._cluster_groups(df, identity=0.9)

        assert groups[0] == groups[1]  # two identical real peptides share a cluster
        assert groups[2] != groups[0]  # a dissimilar real peptide does not
        assert groups[3] != groups[4]  # decoys never share a group with each other
        assert groups[3] not in {groups[0], groups[2]}  # or with any real cluster
        assert len(groups) == 5

    def test_single_real_peptide_does_not_call_the_backend(self, monkeypatch: Any) -> None:
        # Either backend needs at least two sequences to compare; one real peptide has nothing to
        # cluster against, so neither must be called on it at all.
        called: List[Any] = []
        monkeypatch.setattr("sample.families.cluster", lambda *a, **k: called.append(1) or {})
        monkeypatch.setattr("sample.families.cluster_cdhit", lambda *a, **k: called.append(1) or {})
        df = pd.DataFrame({"sequence": ["AAAA"], "is_decoy": [0]})
        groups = M._cluster_groups(df)
        assert not called
        assert len(groups) == 1

    def test_prefers_cdhit_over_pepmatch_when_it_is_installed(self, monkeypatch: Any) -> None:
        # PepMatch caps its mismatch budget at 6, so it cannot deliver GROUPING_IDENTITY for
        # anything but short peptides -- cd-hit must win the routing wherever it exists, or the
        # split is quietly clustered at ~88% identity while reporting 40%.
        used: List[str] = []
        monkeypatch.setattr("sample.families.cdhit_available", lambda: True)
        monkeypatch.setattr("sample.families.cluster_cdhit",
                            lambda seqs, **k: used.append("cdhit") or {s: 1 for s in seqs})
        monkeypatch.setattr("sample.families.cluster",
                            lambda seqs, **k: used.append("pepmatch") or {s: 1 for s in seqs})
        df = pd.DataFrame({"sequence": ["AAAA", "CCCC"], "is_decoy": [0, 0]})
        M._cluster_groups(df)
        assert used == ["cdhit"]

    @pytest.mark.parametrize("has_cdhit,backend", [(True, "cluster_cdhit"), (False, "cluster")])
    def test_work_dir_is_forwarded_to_whichever_backend_runs(
            self, monkeypatch: Any, has_cdhit: bool, backend: str) -> None:
        # This is the isolation hook a test harness (or a caller wanting steps 45-46's scratch files
        # kept apart from step 35's) needs -- verify it reaches the backend that actually runs
        # rather than being silently dropped on one of the two paths.
        seen: Dict[str, Any] = {}

        def fake(sequences: Dict[str, str], work_dir: Any = None, **kwargs: Any) -> Dict[str, int]:
            seen["work_dir"] = work_dir
            return {key: 1 for key in sequences}

        monkeypatch.setattr("sample.families.cdhit_available", lambda: has_cdhit)
        monkeypatch.setattr(f"sample.families.{backend}", fake)
        df = pd.DataFrame({"sequence": ["AAAA", "CCCC"], "is_decoy": [0, 0]})
        M._cluster_groups(df, work_dir="/tmp/isolated-scratch")
        assert seen["work_dir"] == "/tmp/isolated-scratch"


class TestFamilyGroups:
    """
    _family_groups CAN union two independent signals -- a curated family assignment and a fresh
    identity-clustering pass -- via union-find, but does not by default; see
    TestFamilyGroupsUnionIsOffByDefault for why. Tests here that exercise the union machinery pass
    ``union_families=True`` explicitly rather than relying on the default, so flipping the default
    again cannot silently turn them into tests of something else.

    Every test fixes the clustering signal to a known partition (monkeypatched, so it is not
    exercising PepMatch/cd-hit for real -- that is covered by TestClusterGroups and
    test_families_synergy.py's own TestClusterCdhit) and checks what the grouping does with it.
    """

    def _df(self, entry_ids: List[str], sequences: List[str],
           is_decoy: List[int]) -> pd.DataFrame:
        return pd.DataFrame({"entry_id": entry_ids, "sequence": sequences, "is_decoy": is_decoy})

    def _fix_clusters(self, monkeypatch: Any, cluster_by_seq: Dict[str, int]) -> None:
        # Bypasses the real backend entirely: _cluster_groups routes to cd-hit whenever it is
        # installed (true on this machine), so cdhit_available() is forced off here to exercise
        # the PepMatch path being patched below -- _cluster_groups itself (unpatched, still doing
        # its own decoy/real bookkeeping) reports the fixed partition back to _family_groups.
        monkeypatch.setattr("sample.families.cdhit_available", lambda: False)
        monkeypatch.setattr(
            "sample.families.cluster",
            lambda sequences, **k: {key: cluster_by_seq[seq] for key, seq in sequences.items()})

    def test_two_peptides_sharing_only_a_curated_family_are_unioned(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # Below any plausible identity threshold to each other (fixed to distinct clusters), but
        # both anchored to the same Pfam family -- the curated signal alone must still merge them.
        self._fix_clusters(monkeypatch, {"AAAA": 1, "CCCC": 2})
        df = self._df(["P1", "P2"], ["AAAA", "CCCC"], [0, 0])
        families_json = tmp_path / "families.json"
        families_json.write_text(json.dumps({
            "P1": {"family_id": "PF00001", "source": "pfam"},
            "P2": {"family_id": "PF00001", "source": "pfam"},
        }))
        groups = M._family_groups(df, families_json=str(families_json), union_families=True)
        assert groups[0] == groups[1]

    def test_two_peptides_sharing_only_a_cluster_are_unioned(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # Neither has a curated family at all, but the fresh clustering pass puts them together --
        # the identity signal alone must still merge them, exactly the case an earlier version of
        # this function (curated family with a fallback ONLY for uncovered rows) already handled.
        self._fix_clusters(monkeypatch, {"AAAA": 1, "AAAAAAAA": 1})
        df = self._df(["P1", "P2"], ["AAAA", "AAAAAAAA"], [0, 0])
        groups = M._family_groups(df, families_json=str(tmp_path / "absent.json"))
        assert groups[0] == groups[1]

    def test_a_family_and_cluster_bridge_transitively_merges_a_third_peptide(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # P1/P2 share a curated family; P2/P3 share a cluster; P1 and P3 share neither signal
        # directly. The point of union-find over both signals together is that P1 and P3 still
        # land in the same group, through P2 as the bridge -- this is exactly the gap the earlier
        # "prefer one, fall back for the rest" arrangement could not close.
        self._fix_clusters(monkeypatch, {"AAAA": 1, "CCCC": 2, "GGGG": 2})
        df = self._df(["P1", "P2", "P3"], ["AAAA", "CCCC", "GGGG"], [0, 0, 0])
        families_json = tmp_path / "families.json"
        families_json.write_text(json.dumps({
            "P1": {"family_id": "PF00001", "source": "pfam"},
            "P2": {"family_id": "PF00001", "source": "pfam"},
        }))
        groups = M._family_groups(df, families_json=str(families_json), union_families=True)
        assert groups[0] == groups[1] == groups[2]

    def test_peptides_with_no_shared_signal_stay_in_different_groups(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        self._fix_clusters(monkeypatch, {"AAAA": 1, "CCCC": 2})
        df = self._df(["P1", "P2"], ["AAAA", "CCCC"], [0, 0])
        groups = M._family_groups(df, families_json=str(tmp_path / "absent.json"))
        assert groups[0] != groups[1]

    def test_decoys_are_always_singletons_regardless_of_family_data(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        self._fix_clusters(monkeypatch, {"AAAA": 1, "DDDD": 100, "EEEE": 101})
        df = self._df(["P1", "D1", "D2"], ["AAAA", "DDDD", "EEEE"], [0, 1, 1])
        groups = M._family_groups(df, families_json=str(tmp_path / "absent.json"))
        assert groups[1] != groups[2]
        assert groups[1] != groups[0] and groups[2] != groups[0]

    def test_group_ids_are_stable_length_and_cover_every_row(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        self._fix_clusters(monkeypatch, {"AAAA": 1, "CCCC": 2, "DDDD": 3})
        df = self._df(["P1", "P2", "D1"], ["AAAA", "CCCC", "DDDD"], [0, 0, 1])
        groups = M._family_groups(df, families_json=str(tmp_path / "absent.json"))
        assert len(groups) == 3
        assert groups.dtype == np.int64


class TestFamilyGroupsUnionIsOffByDefault:
    """
    Unioning curated families with identity clusters is single-linkage transitive closure, and on
    the real dataset it collapsed 81 % of the antibacterial positives (11,589 of 14,283) into ONE
    unsplittable group of pure positives. StratifiedGroupKFold cannot divide a group, so that blob
    went whole to the training side and produced a holdout of 4.4 % prevalence against training's
    22.0 % -- every reported holdout metric was then measured on an unrepresentative remnant, and
    the model was never held out against its own dominant positive family.

    These tests pin the default OFF and pin the mechanism that caused it, so neither can come back
    without a failing test. They are deliberately about the chaining behaviour rather than the
    exact production numbers, which depend on data this suite does not carry.
    """

    def _df(self, entry_ids: List[str], sequences: List[str],
            is_decoy: List[int]) -> pd.DataFrame:
        return pd.DataFrame({"entry_id": entry_ids, "sequence": sequences, "is_decoy": is_decoy})

    def _fix_clusters(self, monkeypatch: Any, cluster_by_seq: Dict[str, int]) -> None:
        monkeypatch.setattr("sample.families.cdhit_available", lambda: False)
        monkeypatch.setattr(
            "sample.families.cluster",
            lambda sequences, **k: {key: cluster_by_seq[seq] for key, seq in sequences.items()})

    def test_the_default_is_off(self) -> None:
        # The constant IS the fix. If this flips, the collapse returns.
        assert M.UNION_FAMILIES_IN_GROUPING is False

    def test_by_default_a_shared_family_does_not_merge_distinct_clusters(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # Same setup as TestFamilyGroups' first test, but WITHOUT union_families=True: the curated
        # family must no longer pull two separate identity clusters together.
        self._fix_clusters(monkeypatch, {"AAAA": 1, "CCCC": 2})
        df = self._df(["P1", "P2"], ["AAAA", "CCCC"], [0, 0])
        families_json = tmp_path / "families.json"
        families_json.write_text(json.dumps({
            "P1": {"family_id": "PF00001", "source": "pfam"},
            "P2": {"family_id": "PF00001", "source": "pfam"},
        }))
        groups = M._family_groups(df, families_json=str(families_json))
        assert groups[0] != groups[1]

    def test_by_default_families_do_not_chain_separate_clusters_together(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # The collapse mechanism in miniature: four peptides in three distinct identity clusters,
        # chained into one component by two overlapping families (P1-P2 via F1, P2-P3 via F2,
        # P3-P4 via F3). Under the union every one of them lands in a single group; by default
        # they must stay as the three clusters the identity signal actually found.
        self._fix_clusters(monkeypatch, {"AAAA": 1, "CCCC": 1, "GGGG": 2, "TTTT": 3})
        df = self._df(["P1", "P2", "P3", "P4"], ["AAAA", "CCCC", "GGGG", "TTTT"], [0, 0, 0, 0])
        families_json = tmp_path / "families.json"
        families_json.write_text(json.dumps({
            "P2": {"family_id": "PF00002", "source": "pfam"},
            "P3": {"family_id": "PF00002", "source": "pfam"},
            "P4": {"family_id": "PF00003", "source": "pfam"},
        }))

        unioned = M._family_groups(df, families_json=str(families_json), union_families=True)
        assert len(set(unioned.tolist())) == 2, "the union should chain P2/P3/P4 into one group"

        default = M._family_groups(df, families_json=str(families_json))
        assert len(set(default.tolist())) == 3, "clustering alone must keep three groups"
        assert default[0] == default[1]   # same cluster, still grouped -- leakage guard intact
        assert default[2] != default[3]   # different clusters, no family bridge

    def test_the_leakage_guard_still_holds_by_default(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # The property the whole function exists for: anything the identity pass calls a cluster
        # stays in one group, so no near-identical peptide can straddle the split. Turning the
        # union off must not weaken this.
        self._fix_clusters(monkeypatch, {"AAAA": 1, "AAAAAAAA": 1, "CCCC": 2})
        df = self._df(["P1", "P2", "P3"], ["AAAA", "AAAAAAAA", "CCCC"], [0, 0, 0])
        groups = M._family_groups(df, families_json=str(tmp_path / "absent.json"))
        assert groups[0] == groups[1]
        assert groups[2] != groups[0]

    def test_families_json_is_not_even_read_by_default(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        # A corrupt family file must not be able to break grouping that no longer consults it --
        # this is what makes "off" mean off, rather than "read, then mostly ignored".
        self._fix_clusters(monkeypatch, {"AAAA": 1, "CCCC": 2})
        df = self._df(["P1", "P2"], ["AAAA", "CCCC"], [0, 0])
        corrupt = tmp_path / "families.json"
        corrupt.write_text("{not valid json at all")
        groups = M._family_groups(df, families_json=str(corrupt))
        assert len(groups) == 2


class TestFitAndValidateNoCrossValidation:
    """
    Hyperparameter selection (comparing several candidates by cross-validated score) was removed
    long before this: the quality gate already catches a badly wrong default, which was the only
    thing comparing candidates bought over just fitting one. :func:`fit_and_validate` fits ONE
    model on a single grouped train/holdout split -- no selection-oriented CV, nested or
    otherwise, and never a different model chosen by a cross-validated score.

    Cross-validation itself is back (:func:`M._cross_validate`, reported under the
    ``"cross_validation"`` key), but as a fold-to-fold-spread DIAGNOSTIC alongside the single
    holdout number, not a selection mechanism -- these tests still hold: no stale CV-as-selection
    key exists, and one model, not several candidates, is ever fit.
    """

    def _signal_data(self, n: int = 120, seed: int = 0) -> Any:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        X = np.column_stack([y * 3 + rng.normal(0, 1, n),
                             rng.normal(0, 1, n)]).astype(np.float32)
        groups = np.arange(n)  # one group per row -- enough groups for a grouped split
        entry_ids = np.array([f"P{i}" for i in range(n)])
        return X, y, groups, entry_ids

    def test_fits_once_with_fixed_hyperparameters_no_cv_fields(self) -> None:
        X, y, groups, entry_ids = self._signal_data()
        record = M.fit_and_validate(X, y, groups, entry_ids)
        assert hasattr(record["_model"], "predict_proba")
        for stale_key in ("cv_mean", "cv_std", "nested_cv", "selected_hyperparameters"):
            assert stale_key not in record
        assert "holdout" in record and "holdout_ci95" in record

    def test_holdout_is_roughly_the_configured_split_size(self) -> None:
        X, y, groups, entry_ids = self._signal_data(n=200)
        record = M.fit_and_validate(X, y, groups, entry_ids)
        n_test = len(record["_test_entry_ids"])
        n_train = len(record["_train_entry_ids"])
        assert n_train + n_test == 200
        # ~1/N_SPLITS held out (StratifiedGroupKFold's first fold), not an exact fraction since
        # groups cannot always be split perfectly evenly.
        assert 0.1 < n_test / 200 < 0.35

    def test_too_few_groups_still_fits_via_the_ungrouped_fallback(self) -> None:
        X = np.random.default_rng(0).normal(0, 1, size=(8, 2)).astype(np.float32)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        groups = np.zeros(8, dtype=np.int64)  # a single group cannot be split into 5 folds
        entry_ids = np.array([f"P{i}" for i in range(8)])
        record = M.fit_and_validate(X, y, groups, entry_ids)
        assert hasattr(record["_model"], "predict_proba")
        assert len(record["_train_entry_ids"]) + len(record["_test_entry_ids"]) == 8

    def test_classification_record_carries_a_calibrated_threshold(self) -> None:
        X, y, groups, entry_ids = self._signal_data()
        record = M.fit_and_validate(X, y, groups, entry_ids)
        assert record["threshold"] is not None
        assert 0.0 <= record["threshold"] <= 1.0

    def test_calibrate_threshold_false_pins_the_conventional_half(self) -> None:
        # sample.synergy's FICI classifier opts out: its probability is deliberately constructed
        # so 0.5 lands exactly on the domain's own synergy cutoff, and it is served at a hardcoded
        # 0.5 independent of anything this function returns.
        X, y, groups, entry_ids = self._signal_data()
        record = M.fit_and_validate(X, y, groups, entry_ids, calibrate_threshold=False)
        assert record["threshold"] == pytest.approx(0.5)

    def test_regression_record_has_no_threshold(self) -> None:
        rng = np.random.default_rng(0)
        n = 120
        y = rng.normal(0, 1, n).astype(np.float32)
        X = np.column_stack([y * 2 + rng.normal(0, 0.5, n),
                             rng.normal(0, 1, n)]).astype(np.float32)
        groups = np.arange(n)
        entry_ids = np.array([f"P{i}" for i in range(n)])
        record = M.fit_and_validate(X, y, groups, entry_ids, is_regression=True)
        assert record["threshold"] is None

    def test_holdout_metrics_are_scored_at_the_calibrated_threshold(
            self, monkeypatch: Any) -> None:
        # If fit_and_validate silently kept scoring the holdout at a hardcoded 0.5 instead of the
        # threshold it just calibrated, this would catch it: force calibration to a threshold far
        # from 0.5 and confirm the reported holdout metrics actually reflect that cutoff, not 0.5.
        X, y, groups, entry_ids = self._signal_data()
        monkeypatch.setattr(M, "_best_f1_threshold", lambda y_true, y_prob: 0.9)
        record = M.fit_and_validate(X, y, groups, entry_ids)
        assert record["threshold"] == pytest.approx(0.9)

        model = record["_model"]
        test_ids = set(record["_test_entry_ids"])
        mask = np.isin(entry_ids, list(test_ids))
        preds = model.predict_proba(X[mask])[:, 1]
        expected_recall = float(((preds >= 0.9).astype(int) & (y[mask] == 1)).sum()
                                / max(int((y[mask] == 1).sum()), 1))
        assert record["holdout"]["recall"] == pytest.approx(expected_recall)

    def test_threshold_calibration_uses_only_the_training_rows(
            self, monkeypatch: Any) -> None:
        # Out-of-fold calibration pools predictions across every fold, so _best_f1_threshold is
        # called ONCE at the end on (close to) the full ~240-row train portion, not a single
        # fold's ~48-row calibration slice -- if the code mistakenly calibrated against the
        # ~60-row TEST portion instead, this would land near 60, not 240: far enough apart that a
        # leak cannot hide here.
        X, y, groups, entry_ids = self._signal_data(n=300)
        captured: Dict[str, int] = {}
        real_best_f1 = M._best_f1_threshold

        def spy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
            captured["n_calib"] = len(y_true)
            return real_best_f1(y_true, y_prob)

        monkeypatch.setattr(M, "_best_f1_threshold", spy)
        record = M.fit_and_validate(X, y, groups, entry_ids)
        n_train = len(record["_train_entry_ids"])
        n_test = len(record["_test_entry_ids"])
        assert n_train > 3 * n_test  # sanity: the outer ~80/20 split behaved as expected

        assert captured["n_calib"] == pytest.approx(n_train, rel=0.15)
        assert captured["n_calib"] > n_test

    def test_oof_calibration_lands_close_to_the_true_optimum(self) -> None:
        # Regression test for a real bug: calibrating from a SINGLE train/calibration split (the
        # first version of this) was found, on a real trained model, to land far from the true
        # F1-optimal threshold -- noise from using only ~20% of the training portion for the
        # estimate, different by a wide margin depending on which slice was drawn. Verifies the
        # out-of-fold replacement lands close to the actual best threshold, measured independently
        # on a much larger population drawn from the same generative process (never seen by
        # fit_and_validate itself).
        rng = np.random.default_rng(1)
        n = 2000

        def _draw(n_rows: int) -> Any:
            y = rng.integers(0, 2, n_rows)
            # Positive-class scores centered BELOW 0.5 with real overlap -- the exact shape
            # scale_pos_weight leaves behind (see _best_f1_threshold), where the true optimal
            # cutoff is not simply 0.5 or obvious by inspection.
            X = np.column_stack([y * 0.4 + rng.normal(0, 0.25, n_rows),
                                 rng.normal(0, 1, n_rows)]).astype(np.float32)
            return X, y

        X, y = _draw(n)
        groups = np.arange(n)
        entry_ids = np.array([f"P{i}" for i in range(n)])
        record = M.fit_and_validate(X, y, groups, entry_ids)

        X_check, y_check = _draw(20_000)
        probs_check = record["_model"].predict_proba(X_check)[:, 1]
        from sklearn.metrics import f1_score
        best_true_thr, best_true_f1 = 0.5, -1.0
        for thr in np.arange(0.05, 0.96, 0.01):
            f1 = f1_score(y_check, (probs_check >= thr).astype(int), zero_division=0)
            if f1 > best_true_f1:
                best_true_f1, best_true_thr = f1, thr

        assert abs(record["threshold"] - best_true_thr) < 0.15


class _FakeExpensiveEstimator:
    """
    A minimal stand-in for FlamlXGBoostEstimator that exposes the SAME cheap-refit contract
    (``y_randomization_factory()``) without needing flaml installed -- lets the tests below verify
    fit_and_validate's cheap-refit detection and reordering logic in the `sample` environment,
    where flaml is not present.
    """

    def __init__(self, is_regression: bool = False,
                 scale_pos_weight: Any = None, **_ignored: Any) -> None:
        self.is_regression = is_regression
        self.scale_pos_weight = scale_pos_weight
        self._model: Any = None

    def fit(self, X: Any, y: Any) -> "_FakeExpensiveEstimator":
        self._model = M._make_model(is_regression=self.is_regression,
                                    scale_pos_weight=self.scale_pos_weight)
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: Any) -> Any:
        return self._model.predict_proba(X)

    def predict(self, X: Any) -> Any:
        return self._model.predict(X)

    def y_randomization_factory(self) -> Any:
        def factory(is_regression: bool = False, **_ignored: Any) -> Any:
            return M._make_model(is_regression=is_regression)
        return factory


class TestFitAndValidateCheapRefitDetection:
    """
    fit_and_validate must engage the cheap-refit calibration/CV/y-randomization path for a factory
    that is the class itself or a `functools.partial` around it, and must NOT engage it for an
    opaque callable (a lambda/closure) it cannot inspect -- see `_oof_calibrate_threshold`'s own
    callers for why this distinction matters (silently falling back to the slower path is safe;
    silently assuming a hook exists when it does not would crash).
    """

    def _signal_data(self, n: int = 150, seed: int = 0) -> Any:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        X = np.column_stack([y * 3 + rng.normal(0, 1, n),
                             rng.normal(0, 1, n)]).astype(np.float32)
        groups = np.arange(n)
        entry_ids = np.array([f"P{i}" for i in range(n)])
        return X, y, groups, entry_ids

    def _count_fits(self, monkeypatch: Any) -> Dict[str, int]:
        calls = {"count": 0}
        real_fit = _FakeExpensiveEstimator.fit

        def counting_fit(self: Any, X: Any, y: Any) -> Any:
            calls["count"] += 1
            return real_fit(self, X, y)

        monkeypatch.setattr(_FakeExpensiveEstimator, "fit", counting_fit)
        return calls

    def test_the_class_itself_engages_the_cheap_refit_path(self, monkeypatch: Any) -> None:
        calls = self._count_fits(monkeypatch)
        X, y, groups, entry_ids = self._signal_data()
        M.fit_and_validate(X, y, groups, entry_ids, model_factory=_FakeExpensiveEstimator,
                           n_calibration_folds=1, n_y_randomizations=2, n_cv_folds=3)
        # One real fit for the model itself; calibration/CV/y-randomization all reuse it via the
        # cheap y_randomization_factory() hook rather than calling _FakeExpensiveEstimator.fit()
        # again.
        assert calls["count"] == 1

    def test_a_functools_partial_around_the_class_also_engages_it(self, monkeypatch: Any) -> None:
        import functools

        calls = self._count_fits(monkeypatch)
        X, y, groups, entry_ids = self._signal_data()
        factory = functools.partial(_FakeExpensiveEstimator)
        M.fit_and_validate(X, y, groups, entry_ids, model_factory=factory,
                           n_calibration_folds=1, n_y_randomizations=2, n_cv_folds=3)
        assert calls["count"] == 1

    def test_an_opaque_lambda_falls_back_to_the_slow_path(self, monkeypatch: Any) -> None:
        calls = self._count_fits(monkeypatch)
        X, y, groups, entry_ids = self._signal_data()
        M.fit_and_validate(
            X, y, groups, entry_ids,
            model_factory=lambda **kw: _FakeExpensiveEstimator(**kw),
            n_calibration_folds=1, n_y_randomizations=1, n_cv_folds=1)
        # The old, slower shape: one real fit for the single-split calibration model, a SEPARATE
        # real fit for the final model -- fit_and_validate cannot see past the lambda to find
        # y_randomization_factory, so it cannot know reusing either fit's configuration for the
        # other purpose is safe.
        assert calls["count"] == 2

    def test_cheap_path_calibration_folds_are_raised_to_n_splits(self, monkeypatch: Any) -> None:
        # A caller passing n_calibration_folds=1 (FLAML's own historical "too expensive to refit"
        # value) must not get single-split-noisy calibration on the cheap-refit path any more --
        # there is no remaining reason to accept that noise once refits are cheap.
        oof_calls: Dict[str, int] = {}
        real_oof = M._oof_calibrate_threshold

        def spy(X_train: Any, y_train: Any, groups_train: Any, factory: Any,
               model_params: Any, n_folds: int) -> float:
            oof_calls["n_folds"] = n_folds
            return real_oof(X_train, y_train, groups_train, factory, model_params, n_folds)

        monkeypatch.setattr(M, "_oof_calibrate_threshold", spy)
        X, y, groups, entry_ids = self._signal_data()
        M.fit_and_validate(X, y, groups, entry_ids, model_factory=_FakeExpensiveEstimator,
                           n_calibration_folds=1, n_y_randomizations=1, n_cv_folds=1)
        assert oof_calls["n_folds"] >= M.N_SPLITS


class TestApplicabilityDomain:
    def test_in_domain_points_pass_and_a_far_outlier_fails(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.normal(0, 1, size=(200, 3))
        ad = M._applicability_domain(X_train, k=5)
        assert ad["index"] is not None
        assert ad["threshold"] > 0

        near_query = X_train[:5] + rng.normal(0, 0.01, size=(5, 3))
        far_query = np.full((3, 3), 1000.0)

        assert M.in_domain(near_query, ad).all()
        assert not M.in_domain(far_query, ad).any()

    def test_too_few_training_points_reports_no_domain(self) -> None:
        # k=5 neighbours needs at least 6 training points; one point has none.
        ad = M._applicability_domain(np.zeros((1, 3)), k=5)
        assert ad["index"] is None

    def test_in_domain_with_no_index_fails_visibly_rather_than_skipping(self) -> None:
        result = M.in_domain(np.zeros((4, 2)), {"index": None})
        assert not result.any()
        assert len(result) == 4


class TestYRandomization:
    def test_shuffled_labels_collapse_toward_chance(self) -> None:
        rng = np.random.default_rng(0)
        n = 150
        y = rng.integers(0, 2, n)
        X = np.column_stack([y * 4 + rng.normal(0, 1, n),
                             rng.normal(0, 1, n)]).astype(np.float32)
        X_train, y_train = X[:100], y[:100]
        X_test, y_test = X[100:], y[100:]

        result = M._y_randomization(X_train, y_train, X_test, y_test, n_repeats=3)
        # Near chance, not the real ~0.9+ AUC this same data supports with real labels.
        assert 0.25 <= result["mean_roc_auc"] <= 0.75
        assert len(result["per_repeat"]) == 3

    def test_a_single_class_training_set_returns_nan_not_a_crash(self) -> None:
        # Every permutation of an all-one-class label is itself all-one-class -- degenerate, must
        # not raise mid-way through a training run.
        X_train = np.zeros((3, 2), dtype=np.float32)
        y_train = np.array([1, 1, 1])
        X_test = np.zeros((2, 2), dtype=np.float32)
        y_test = np.array([0, 1])
        result = M._y_randomization(X_train, y_train, X_test, y_test, n_repeats=2)
        assert np.isnan(result["mean_roc_auc"])
        assert result["per_repeat"] == []


class TestCompareToNull:
    def test_separated_distributions_are_significant(self) -> None:
        rng = np.random.default_rng(0)
        bootstrap_samples = {"roc_auc": list(rng.normal(0.9, 0.02, 300))}
        null_per_repeat = [{"roc_auc": v} for v in rng.normal(0.5, 0.02, 5)]

        out = M._compare_to_null(bootstrap_samples, null_per_repeat)
        cmp = out["roc_auc"]
        assert cmp["real_mean"] > cmp["null_mean"]
        assert cmp["primary_pvalue"] < 0.05
        assert cmp["primary_test"] in ("t_test", "kruskal")
        assert "real_normality" in cmp and "null_normality" in cmp
        assert "t_pvalue" in cmp and "kruskal_pvalue" in cmp  # both kept, even though unused

    def test_no_null_repeats_returns_empty(self) -> None:
        assert M._compare_to_null({"roc_auc": [0.9, 0.8]}, []) == {}

    def test_non_normal_real_distribution_selects_kruskal_as_primary(self) -> None:
        rng = np.random.default_rng(2)
        # Two tight clumps -- Shapiro-Wilk reliably rejects normality on this shape.
        bootstrap_samples = {"roc_auc": list(np.concatenate([np.full(50, 0.6), np.full(50, 0.95)]))}
        null_per_repeat = [{"roc_auc": v} for v in rng.normal(0.5, 0.02, 5)]

        out = M._compare_to_null(bootstrap_samples, null_per_repeat)
        assert out["roc_auc"]["real_normality"]["is_normal"] is False
        assert out["roc_auc"]["primary_test"] == "kruskal"
        assert out["roc_auc"]["primary_pvalue"] == pytest.approx(out["roc_auc"]["kruskal_pvalue"])


class TestQualityGateNullSignificance:
    def _record(self, **overrides: Any) -> Dict[str, Any]:
        record = {
            "holdout": {"roc_auc": 0.85},
            "holdout_ci95": {"roc_auc": [0.72, 0.90]},
            "y_randomization": {"mean_roc_auc": 0.50},
            "n_samples": 500,
        }
        record.update(overrides)
        return record

    def test_significant_null_comparison_passes(self) -> None:
        record = self._record(null_comparison={
            "roc_auc": {"primary_test": "t_test", "primary_pvalue": 0.001}})
        passed, reasons = M.assess_quality(record)
        assert passed and reasons == []

    def test_non_significant_null_comparison_fails(self) -> None:
        record = self._record(null_comparison={
            "roc_auc": {"primary_test": "kruskal", "primary_pvalue": 0.5}})
        passed, reasons = M.assess_quality(record)
        assert not passed
        assert any("not significant" in r for r in reasons)

    def test_missing_null_comparison_is_not_itself_a_failure(self) -> None:
        # A record from before this field existed (or a test fixture that omits it) must be judged
        # on the other criteria alone, not rejected for a key that simply is not there.
        passed, reasons = M.assess_quality(self._record())
        assert passed and reasons == []


class TestTrainAllEndToEnd:
    """
    One small, real run of the whole pipeline -- not mocked. :func:`~sample.ml_models._cluster_groups`
    goes through the real clustering backend (cd-hit where installed, PepMatch otherwise), and
    everything downstream depends on the groups it returns; a test that mocked clustering away
    could not catch a real integration break between the two modules. Kept to 160 rows so it still
    runs in a few seconds, and points
    ``cluster_work_dir`` at a temp directory so it cannot overwrite step 35's real scratch files in
    ``data/families/`` (which share filenames like ``_cluster_target.fasta`` by design -- see
    :func:`~sample.ml_models._cluster_groups`'s ``work_dir`` docstring), and ``families_json`` at a
    path that does not exist so grouping falls back to plain near-identity clustering rather than
    reading the real release's curated Pfam-family assignment for these synthetic entry ids.
    """

    def _dataset(self, tmp_path: Any, n: int = 160, seed: int = 0) -> Any:
        rng = np.random.default_rng(seed)
        # Real peptides are drawn from a cationic, hydrophobic alphabet -- the composition that
        # actually characterises an antimicrobial peptide -- and decoys from a uniform one. The
        # signal therefore lives in the SEQUENCE, which puts it within reach of every block
        # feature_blocks offers rather than only the one column a planted scalar reaches: net
        # charge for "descriptors", residue pairs for "dpc", the charge/hydrophobicity class
        # profile for "ctd". A block failing the gate on this fixture then means the pipeline
        # broke, not that the fixture gave that block nothing to find.
        amp_alphabet = list("KKRRLLWWGGAAIFV")
        uniform_alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        rows = []
        activity_rows = []
        for i in range(n):
            is_decoy = i % 2
            length = int(rng.integers(8, 25))
            alphabet = uniform_alphabet if is_decoy else amp_alphabet
            sequence = "".join(rng.choice(alphabet, size=length))
            # Derived from the sequence, not drawn independently of it: a `charge` column that
            # contradicted its own sequence would let the descriptors block pass on signal the
            # sequence-derived blocks cannot see, which is precisely the disagreement this test
            # should fail on.
            charge = float(sum(sequence.count(r) for r in "KR")
                          - sum(sequence.count(r) for r in "DE"))
            row = {"entry_id": f"P{i:05d}", "sequence": sequence, "is_decoy": is_decoy,
                  "molecular_weight": float(length) * 110.0, "charge": charge}
            rows.append(row)
            # There is no more general (task-free) real-vs-decoy model to fall back on -- every
            # task build_tasks() builds needs real activity data, so every real peptide needs a
            # row here too, not just a target_group column on the dataset that build_tasks never
            # reads.
            if not is_decoy:
                activity_rows.append({"sequence": sequence, "target_group": "antibacterial"})
        dataset_csv = tmp_path / "dataset.csv"
        pd.DataFrame(rows).to_csv(dataset_csv, index=False)
        activities_csv = tmp_path / "activities.csv"
        pd.DataFrame(activity_rows).to_csv(activities_csv, index=False)
        return str(dataset_csv), str(activities_csv)

    def test_runs_end_to_end_and_publishes_a_dataset_file_per_kept_model(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 20)
        dataset_csv, activities_csv = self._dataset(tmp_path)
        model_dir = str(tmp_path / "models")
        report_path = str(tmp_path / "report.json")
        cluster_work_dir = str(tmp_path / "cluster_scratch")
        # A path that does not exist: _family_groups degrades cleanly to plain near-identity
        # clustering, so this synthetic run never picks up the real release's curated families.
        families_json = str(tmp_path / "no_families_here.json")

        report = M.train_all(dataset_csv, model_dir, report_path,
                             activities_csv=activities_csv, tasks_limit=1,
                             cluster_work_dir=cluster_work_dir, families_json=families_json)

        assert report["n_models_kept"] >= 1
        record = report["models"][0]
        assert os.path.exists(record["dataset_path"])
        # Confirms the isolation actually took effect, not just that the argument was accepted.
        assert os.path.exists(cluster_work_dir)

        with open(record["dataset_path"], encoding="utf-8") as fh:
            dataset_record = json.load(fh)

        train_ids = set(dataset_record["train_entry_ids"])
        test_ids = set(dataset_record["test_entry_ids"])
        assert train_ids.isdisjoint(test_ids)
        assert len(train_ids) == dataset_record["n_train"]
        assert len(test_ids) == dataset_record["n_test"]
        assert dataset_record["n_train"] + dataset_record["n_test"] == dataset_record["n_samples"]
        assert dataset_record["quality_passed"] is True
        assert "roc_auc" in dataset_record["null_comparison"]
        assert dataset_record["applicability_domain"]["k"] > 0

        # The holdout as FASTA, beside the dataset.json -- what benchmark.holdout_fasta's fast path
        # (and sample.benchmark itself, indirectly) reads instead of recomputing it.
        assert os.path.exists(record["holdout_fasta_path"])
        with open(record["holdout_fasta_path"], encoding="utf-8") as fh:
            fasta_text = fh.read()
        fasta_ids = {line[1:].strip() for line in fasta_text.splitlines() if line.startswith(">")}
        assert fasta_ids == test_ids
        assert fasta_text.count(">") == dataset_record["n_test"]

        with open(record["model_path"], "rb") as fh:
            payload = pickle.load(fh)
        assert set(payload) >= {"model", "columns", "scaler", "applicability_domain"}
        assert hasattr(payload["model"], "predict_proba")


class TestTrainTaskWithASequenceInputModelFactory:
    """
    A real, small end-to-end run of train_task() with `columns == ["sequence"]` and
    `sample.esm_finetune.ESM2FineTuneEstimator` as the `model_factory` -- not reusing
    TestTrainAllEndToEnd's fixture, whose signal lives in synthetic `esm2_dim_*` floats over
    literally-random sequences: a sequence-input model computes its own embedding from the raw
    string and would see pure noise from that fixture. Needs sequences that are actually
    composition-separable instead.

    Skipped under the default `sample` test environment: see tests/test_esm_finetune.py's module
    docstring for why (a stray torchvision incompatible with this project's torch pin breaks
    transformers' EsmModel import there) -- run under sample-ml to exercise this for real.
    """

    @pytest.fixture(autouse=True)
    def _require_esm_model(self) -> None:
        # RuntimeError, not just ImportError -- see tests/test_esm_finetune.py's module docstring
        # (a stray torchvision crashes transformers' EsmModel import chain under `sample`, but not
        # under `sample-ml`, which has no torchvision at all). pytest.importorskip only catches
        # ImportError, so this needs its own try/except rather than that helper.
        try:
            import transformers.models.esm.modeling_esm  # noqa: F401
        except (ImportError, RuntimeError):
            pytest.skip("EsmModel is not importable in this environment -- run under sample-ml.")

    def _dataset(self, n: int = 40, seed: int = 0) -> Any:
        rng = np.random.default_rng(seed)
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        rows = []
        for i in range(n):
            is_positive = i % 2 == 1
            residues = list(rng.choice(alphabet, size=12))
            if is_positive:
                residues[0] = residues[1] = "K"  # composition-separable signal
            rows.append({"entry_id": f"P{i:05d}", "sequence": "".join(residues),
                        "target_group": "antibacterial" if is_positive else None})
        return pd.DataFrame(rows)

    def test_trains_a_sequence_input_model_through_the_real_pipeline(
            self, monkeypatch: Any) -> None:
        import sample.esm_finetune as EF
        monkeypatch.setattr(EF, "MAX_EPOCHS", 2)
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 15)

        df = self._dataset()
        y = pd.Series((df["target_group"] == "antibacterial").astype(int), index=df.index)
        groups = np.arange(len(df))  # singleton groups -- no real clustering needed for this test

        record = M.train_task(
            df, y, ["sequence"], "antibacterial_test", "esm2_finetune", groups,
            learning_task="classification", model_factory=EF.ESM2FineTuneEstimator,
            n_calibration_folds=1, n_y_randomizations=1)

        assert record is not None
        assert record["threshold"] is not None
        assert 0.0 <= record["threshold"] <= 1.0
        assert record["_scaler"] is None
        assert record["_applicability_domain"]["index"] is None
        assert hasattr(record["_model"], "predict_proba")
        assert "roc_auc" in record["holdout"]



class TestMeasuredInactiveNegatives:
    """
    Real peptides measured against a task and found NOT potent belong in that task's negative
    class (:data:`sample.ml_models.USE_MEASURED_INACTIVE_NEGATIVES`).

    With decoys as the only negatives, a classification task's real question is "real peptide or
    generated string?", and artifacts of the decoy generator are available as shortcuts. These
    negatives are real sequences with real numbers that missed the bar, so no generator artifact
    separates them from the positives.
    """

    def _dataset(self) -> pd.DataFrame:
        return pd.DataFrame({
            "sequence": ["AAAA", "CCCC", "GGGG", "TTTT", "DDDD"],
            "is_decoy": [0, 0, 0, 0, 1],
        })

    @staticmethod
    def _activities(values: List[float]) -> pd.DataFrame:
        # AAAA/CCCC/GGGG all tested against one organism by MIC; TTTT never tested at all.
        return pd.DataFrame({
            "sequence": ["AAAA", "CCCC", "GGGG"],
            "activity_type": "MIC",
            "value_lower": values,
            "activity_unit": "uM",
            "target_organism": "Escherichia coli",
            "target_group": "antibacterial",
        })

    def _write(self, tmp_path: Any, frame: pd.DataFrame) -> str:
        path = tmp_path / "activities.csv"
        frame.to_csv(path, index=False)
        return str(path)

    def _organism_task(self, tmp_path: Any, values: List[float]) -> Any:
        csv = self._write(tmp_path, self._activities(values))
        tasks = M.build_tasks(self._dataset(), csv, min_task_examples=2, min_target_examples=2)
        return next(t for t in tasks
                    if t.scope == "organism_specific" and t.task == "classification")

    def test_measured_but_not_potent_peptide_becomes_a_negative(self, tmp_path: Any) -> None:
        # GGGG's MIC is 500 uM, two orders above ACTIVITY_THRESHOLD_UM -- tested, and not active.
        spec = self._organism_task(tmp_path, [1.0, 1.0, 500.0])
        assert spec.y.tolist()[:3] == [1.0, 1.0, 0.0]
        assert spec.y.iloc[4] == 0.0          # the decoy is still a negative
        assert pd.isna(spec.y.iloc[3])        # never tested -> not in either class
        assert spec.n_measured_inactive == 1

    def test_turning_the_flag_off_restores_the_decoy_only_negative_class(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(M, "USE_MEASURED_INACTIVE_NEGATIVES", False)
        spec = self._organism_task(tmp_path, [1.0, 1.0, 500.0])
        assert pd.isna(spec.y.iloc[2])        # measured-inactive drops back out of the task
        assert spec.y.iloc[4] == 0.0          # decoys unaffected
        assert spec.n_measured_inactive == 0

    def test_a_measured_inactive_peptide_overrides_class_level_presence(
            self, tmp_path: Any) -> None:
        # The activity-class axis is keyed on "was tested in this class", so GGGG was previously
        # an antibacterial POSITIVE purely for having been tested. Its measurement says otherwise.
        csv = self._write(tmp_path, self._activities([1.0, 1.0, 500.0]))
        tasks = M.build_tasks(self._dataset(), csv, min_task_examples=2, min_target_examples=2)
        spec = next(t for t in tasks if t.peptide_type == "antibacterial")
        assert spec.y.iloc[2] == 0.0
        assert spec.y.tolist()[:2] == [1.0, 1.0]

    def test_a_qualitative_annotation_with_no_number_is_never_a_negative(
            self, tmp_path: Any) -> None:
        # No convertible value means no evidence of inactivity -- it is a reported active whose
        # potency was not captured. Calling it inactive would invent a measurement.
        frame = self._activities([1.0, 1.0, 1.0])
        frame.loc[2, "value_lower"] = None
        frame.loc[2, "activity_unit"] = None
        csv = self._write(tmp_path, frame)
        tasks = M.build_tasks(self._dataset(), csv, min_task_examples=2, min_target_examples=2)
        spec = next(t for t in tasks if t.peptide_type == "antibacterial")
        assert spec.y.iloc[2] == 1.0
        assert spec.n_measured_inactive == 0


class TestPotencyBands:
    """
    Potency is three bands, not two: high (<= ACTIVITY_THRESHOLD_UM), low (between the bars) and
    inactive (> INACTIVE_THRESHOLD_UM).

    The low band is the point of these tests. It used to be swept into the negative class by the
    rule "has a number and never cleared 5 uM", which across this corpus meant 67,158 of the
    93,124 pairs called negative -- 72% -- were peptides with real, moderate activity being
    asserted as inactive. A peptide with an MIC of 20 uM is not a high-potency lead and is also
    not an inactive; it belongs to neither class, and these tests pin that down.
    """

    @staticmethod
    def _activities(um: float, assay: str = "MIC", seq: str = "AAAA") -> pd.DataFrame:
        return pd.DataFrame({"sequence": [seq], "activity_type": [assay],
                             "value_lower": [um], "activity_unit": ["uM"],
                             "target_organism": ["Escherichia coli"]})

    def _labels(self, tmp_path: Any, um: float) -> dict:
        csv = tmp_path / "activities.csv"
        self._activities(um).to_csv(csv, index=False)
        return M._label_lookup(str(csv))

    @pytest.mark.parametrize("um", [0.1, 1.0, 5.0])
    def test_high_band_is_active_and_not_inactive(self, tmp_path: Any, um: float) -> None:
        labels = self._labels(tmp_path, um)
        assert labels["active_organism_assay"].get("AAAA")
        assert not labels["inactive_organism_assay"].get("AAAA")

    @pytest.mark.parametrize("um", [5.1, 20.0, 64.0, 100.0])
    def test_low_band_is_neither_active_nor_inactive(self, tmp_path: Any, um: float) -> None:
        """The decisive case: measurable, moderately active, and a member of no class."""
        labels = self._labels(tmp_path, um)
        assert not labels["active_organism_assay"].get("AAAA")
        assert not labels["inactive_organism_assay"].get("AAAA")
        # It is still a recorded measurement -- it just does not become a label.
        assert labels["potency_organism_assay"].get("AAAA")

    @pytest.mark.parametrize("um", [100.1, 250.0, 2000.0])
    def test_inactive_band_is_inactive_and_not_active(self, tmp_path: Any, um: float) -> None:
        labels = self._labels(tmp_path, um)
        assert not labels["active_organism_assay"].get("AAAA")
        assert labels["inactive_organism_assay"].get("AAAA")

    def test_the_two_bars_are_the_documented_values(self) -> None:
        assert M.ACTIVITY_THRESHOLD_UM == 5.0
        assert M.INACTIVE_THRESHOLD_UM == 100.0

    def test_the_bands_never_overlap(self, tmp_path: Any) -> None:
        for um in (0.5, 5.0, 5.01, 50.0, 100.0, 100.01, 500.0):
            labels = self._labels(tmp_path, um)
            active = bool(labels["active_organism_assay"].get("AAAA"))
            inactive = bool(labels["inactive_organism_assay"].get("AAAA"))
            assert not (active and inactive), f"{um} uM landed in both bands"

    def test_a_low_band_peptide_is_not_a_negative_in_a_built_task(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        """End to end: the low band must not reach the label vector as a 0."""
        monkeypatch.setattr(M, "MIN_POSITIVES_PER_TASK", 1)
        rows = pd.concat([
            self._activities(1.0, seq="AAAA"),     # high -> positive
            self._activities(20.0, seq="CCCC"),    # low  -> excluded
            self._activities(500.0, seq="GGGG"),   # inactive -> negative
        ], ignore_index=True)
        csv = tmp_path / "activities.csv"
        rows.to_csv(csv, index=False)
        dataset = pd.DataFrame({"sequence": ["AAAA", "CCCC", "GGGG"], "is_decoy": [0, 0, 0]})

        tasks = [t for t in M.build_tasks(dataset, str(csv), min_task_examples=1,
                                          min_target_examples=1)
                 if t.task == "classification"]
        assert tasks, "expected at least one classification task"
        for task in tasks:
            labelled = task.y.dropna()
            assert labelled.get(0) == 1.0, "the 1 uM peptide should be a positive"
            assert 1 not in labelled.index, "the 20 uM peptide must carry no label at all"
