"""
Tests for :mod:`sample.benchmark` -- comparing a sAMPle model to an external reference tool.

The properties that matter: both tools are scored with the *same* metric code on the *same*
holdout, a metric that is undefined on a single-class holdout is reported as ``None`` rather than a
misleading zero, and an unavailable reference tool produces *no* comparison rather than a set of
fabricated negative predictions that would flatter or damage either side.
"""
from typing import Any, List

import pytest

from sample import benchmark


class _FakePredictor:
    """Stand-in for an ExternalPredictor: fixed availability, one canned row per sequence.

    reference_scores calls `predict` in chunks of REFERENCE_CHUNK_SIZE (10 by default -- see its
    docstring), so every sequence in a test with fewer than that many sequences arrives in a
    single call; `rows` must supply exactly one entry per test sequence, in that same order."""

    def __init__(self, name: str, rows: List[dict], available: bool = True) -> None:
        self.name = name
        self._rows = rows
        self._available = available

    def available(self) -> bool:
        return self._available

    def predict(self, sequences: Any) -> List[dict]:
        return list(self._rows[:len(sequences)])


class _FakeEstimator:
    """Stand-in for a trained model: predict_proba/predict read straight off a dataframe's "score"
    column, so a test can assert exactly what each row should produce without a real XGBoost fit."""

    def predict_proba(self, X: Any) -> Any:
        import numpy as np

        p = np.asarray(X["score"], dtype=float)
        return np.column_stack([1 - p, p])

    def predict(self, X: Any) -> Any:
        import numpy as np

        return np.asarray(X["score"], dtype=float)


class TestReferenceToolFor:
    def test_matches_a_hemolytic_model(self) -> None:
        tool = benchmark.reference_tool_for(
            "hemolytic/general/any/classification/descriptors")
        assert tool is not None and tool.predictor == "hemopi2"

    def test_returns_none_when_no_tool_covers_the_task(self) -> None:
        assert benchmark.reference_tool_for(
            "antibacterial/general/mic/regression/descriptors") is None

    def test_regression_namespace_never_matches_even_with_the_key_present(self) -> None:
        # antibacterial DOES have a registered tool now, but only for classification --
        # binary_metrics has nothing to do with a continuous MIC/IC50 prediction.
        assert benchmark.reference_tool_for(
            "antibacterial/general/mic/regression/descriptors") is None

    def test_matches_macrel_for_antibacterial_classification(self) -> None:
        tool = benchmark.reference_tool_for(
            "antibacterial/general/any/classification/descriptors")
        assert tool is not None and tool.predictor == "macrel"

    def test_anticancer_has_no_registered_tool(self) -> None:
        # Benchmarking is scoped to antibacterial peptide prediction tools only, by instruction --
        # antibacterial is the one case study while the FLAML-searched training path is proved
        # out. See REFERENCE_TOOLS' own comment for how to bring AntiCP 2.0 back into scope.
        assert benchmark.reference_tool_for(
            "anticancer/general/any/classification/descriptors") is None
        assert benchmark.reference_tools_for(
            "anticancer/general/any/classification/descriptors") == []

    def test_antibacterial_classification_has_macrel_amplify_ampidentifier_and_ampeppy(self) -> None:
        tools = benchmark.reference_tools_for(
            "antibacterial/general/any/classification/descriptors")
        predictors = {t.predictor for t in tools}
        assert predictors == {"macrel", "amplify", "ampidentifier", "ampeppy"}
        amplify = next(t for t in tools if t.predictor == "amplify")
        assert amplify.best_effort is True
        ampidentifier = next(t for t in tools if t.predictor == "ampidentifier")
        assert ampidentifier.best_effort is True
        ampeppy = next(t for t in tools if t.predictor == "ampeppy")
        assert ampeppy.best_effort is True


class TestBinaryMetrics:
    def test_perfect_separation_scores_one(self) -> None:
        m = benchmark.binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["n"] == 4 and m["n_positives"] == 2

    def test_single_class_holdout_has_undefined_auc(self) -> None:
        # ROC-AUC is undefined with one class present; it must be None, not silently 0.0 or 0.5.
        m = benchmark.binary_metrics([1, 1, 1], [0.2, 0.7, 0.9])
        assert m["roc_auc"] is None
        assert m["average_precision"] is None
        assert m["mcc"] == 0.0

    def test_threshold_is_reported_and_applied(self) -> None:
        m = benchmark.binary_metrics([0, 1], [0.4, 0.6], threshold=0.5)
        assert m["threshold"] == 0.5
        assert m["accuracy"] == pytest.approx(1.0)


class TestReferenceScores:
    def test_reads_scores_from_the_predictor(self, monkeypatch: Any) -> None:
        fake = _FakePredictor("hemopi2", [
            {"hemopi2_score": 0.9, "hemopi2_is_hemolytic": 1.0},
            {"hemopi2_score": 0.1, "hemopi2_is_hemolytic": 0.0},
        ])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        tool = benchmark.reference_tool_for("hemolytic/general/any/classification/descriptors")
        result = benchmark.reference_scores(tool, ["AAAA", "CCCC"])
        assert result["available"] is True
        assert result["scores"] == [0.9, 0.1]
        assert result["labels"] == [1.0, 0.0]

    def test_unavailable_tool_yields_no_scores(self, monkeypatch: Any) -> None:
        fake = _FakePredictor("hemopi2", [], available=False)
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        tool = benchmark.reference_tool_for("hemolytic/general/any/classification/descriptors")
        result = benchmark.reference_scores(tool, ["AAAA"])
        assert result["available"] is False
        assert result["scores"] == []

    def test_missing_predictor_is_unavailable(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [])
        tool = benchmark.reference_tool_for("hemolytic/general/any/classification/descriptors")
        assert benchmark.reference_scores(tool, ["AAAA"])["available"] is False

    def test_label_only_tool_falls_back_to_its_call(self, monkeypatch: Any) -> None:
        # A tool that reports a 0/1 call but no continuous score is still comparable: the label
        # becomes the score so ROC-AUC degrades gracefully rather than being all-NaN.
        fake = _FakePredictor("hemopi2", [
            {"hemopi2_score": None, "hemopi2_is_hemolytic": 1.0},
            {"hemopi2_score": None, "hemopi2_is_hemolytic": 0.0},
        ])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        tool = benchmark.reference_tool_for("hemolytic/general/any/classification/descriptors")
        result = benchmark.reference_scores(tool, ["AAAA", "CCCC"])
        assert result["scores"] == [1.0, 0.0]


class TestCompare:
    def test_reports_both_sides_and_the_delta(self) -> None:
        y = [0, 0, 1, 1]
        model = [0.1, 0.2, 0.8, 0.9]      # perfect
        reference = [0.4, 0.6, 0.4, 0.6]  # coin-flip-ish
        result = benchmark.compare(y, model, reference, reference_available=True)
        assert result["model"]["roc_auc"] == pytest.approx(1.0)
        assert result["reference"] is not None
        # The model leads on ROC-AUC, so the delta is positive.
        assert result["delta"]["roc_auc"] > 0

    def test_unavailable_reference_gives_model_only(self) -> None:
        result = benchmark.compare([0, 1], [0.2, 0.8], None, reference_available=False)
        assert result["reference"] is None
        assert result["delta"] is None
        assert result["model"]["n"] == 2

    def test_model_threshold_scores_only_the_model_side(self) -> None:
        # The model's own score sits below the conventional 0.5 but above a lower calibrated
        # threshold; the reference score sits above 0.5. Passing model_threshold must move only
        # the model's own call, leaving the reference read at the plain `threshold`.
        y = [0, 1]
        model = [0.1, 0.3]  # the positive's score sits below 0.5, above a 0.2 calibrated cutoff
        reference = [0.4, 0.6]
        result = benchmark.compare(y, model, reference, reference_available=True,
                                   threshold=0.5, model_threshold=0.2)
        assert result["model"]["threshold"] == pytest.approx(0.2)
        assert result["model"]["recall"] == pytest.approx(1.0)  # 0.3 >= 0.2 now counts as positive
        assert result["reference"]["threshold"] == pytest.approx(0.5)

    def test_omitted_model_threshold_falls_back_to_the_shared_threshold(self) -> None:
        result = benchmark.compare([0, 1], [0.2, 0.8], [0.3, 0.7], reference_available=True,
                                   threshold=0.6)
        assert result["model"]["threshold"] == pytest.approx(0.6)
        assert result["reference"]["threshold"] == pytest.approx(0.6)


class TestBenchmarkModel:
    def test_full_report_over_a_shared_holdout(self, monkeypatch: Any) -> None:
        sequences = ["AAAA", "CCCC", "DDDD", "EEEE"]
        y_true = [0, 0, 1, 1]

        # sAMPle model: perfect probabilities.
        monkeypatch.setattr(
            "sample.inference.predict",
            lambda seqs, model_id, model_dir=None: [
                {"probability": p, "predicted_label": int(p >= 0.5)}
                for p in (0.1, 0.2, 0.8, 0.9)])
        # Reference: weaker.
        fake = _FakePredictor("hemopi2", [
            {"hemopi2_score": s} for s in (0.4, 0.6, 0.5, 0.55)])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        # hemolytic also has a second, best_effort tool (ToxinPred2) since it is looked up via
        # _benchmark_only_predictors rather than EXTERNAL_PREDICTORS, mocking that alone does not
        # stop it from resolving to the REAL, installed ToxinPred2Predictor. Stood in here as
        # unavailable so this stays hermetic -- this test is about the HemoPI2 comparison only.
        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (_FakePredictor("toxinpred2", [], available=False),))

        report = benchmark.benchmark_model(
            "hemolytic/general/any/classification/descriptors", sequences, y_true)
        assert report["model_id"].startswith("hemolytic/")
        assert report["reference_tool"] == "HemoPI 2.0"
        assert report["reference_available"] is True
        assert report["model"]["roc_auc"] == pytest.approx(1.0)
        assert report["delta"]["roc_auc"] >= 0

    def test_model_scores_parameter_skips_sample_inference_predict(
            self, monkeypatch: Any) -> None:
        # benchmark_holdout passes pre-computed scores (from cached feature columns, see
        # TestScoreModelFromDataset) specifically to avoid this call -- a fresh ESM-2 forward pass
        # per benchmark run. When model_scores is given, predict must never run at all.
        def _must_not_be_called(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("benchmark_model must not call sample.inference.predict "
                                 "when model_scores is already provided")

        monkeypatch.setattr("sample.inference.predict", _must_not_be_called)
        fake = _FakePredictor("hemopi2", [{"hemopi2_score": s} for s in (0.4, 0.6, 0.5, 0.55)])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        # See test_full_report_over_a_shared_holdout's comment: hemolytic's second, best_effort
        # tool (ToxinPred2) is resolved via _benchmark_only_predictors, not EXTERNAL_PREDICTORS.
        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (_FakePredictor("toxinpred2", [], available=False),))

        report = benchmark.benchmark_model(
            "hemolytic/general/any/classification/descriptors",
            ["AAAA", "CCCC", "DDDD", "EEEE"], [0, 0, 1, 1],
            model_scores=[0.1, 0.2, 0.8, 0.9])

        assert report["model"]["roc_auc"] == pytest.approx(1.0)

    def test_raises_without_a_reference_tool(self) -> None:
        with pytest.raises(ValueError):
            benchmark.benchmark_model(
                "antibacterial/general/mic/regression/descriptors", ["AAAA"], [1])

    def test_scores_the_model_side_at_its_own_stored_calibrated_threshold(
            self, monkeypatch: Any) -> None:
        # The model's own probabilities sit below 0.5 for both true positives -- at the
        # conventional 0.5 cutoff, recall would be 0. Its stored dataset record carries a
        # calibrated threshold of 0.15, low enough to catch them; the reference tool has no such
        # record and must stay read at the plain 0.5.
        sequences = ["AAAA", "CCCC", "DDDD", "EEEE"]
        y_true = [0, 0, 1, 1]
        monkeypatch.setattr(
            "sample.inference.predict",
            lambda seqs, model_id, model_dir=None: [
                {"probability": p, "predicted_label": int(p >= 0.5)}
                for p in (0.05, 0.1, 0.2, 0.3)])
        fake = _FakePredictor("hemopi2", [{"hemopi2_score": s} for s in (0.4, 0.6, 0.5, 0.55)])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (_FakePredictor("toxinpred2", [], available=False),))
        monkeypatch.setattr(
            "sample.model_registry.get_model_dataset",
            lambda model_id, model_dir=None: {"threshold": 0.15})

        report = benchmark.benchmark_model(
            "hemolytic/general/any/classification/descriptors", sequences, y_true)

        assert report["model_threshold"] == pytest.approx(0.15)
        assert report["reference_threshold"] == pytest.approx(benchmark.DEFAULT_THRESHOLD)
        assert report["model"]["recall"] == pytest.approx(1.0)

    def test_reports_a_comparison_per_matching_tool_and_vs_chance(self, monkeypatch: Any) -> None:
        # antibacterial has four registered tools (Macrel, best-effort AMPlify, best-effort
        # AMPidentifier, best-effort amPEPpy); the report should carry all four, plus a comparison
        # against the model's own stored y-randomization null.
        sequences = ["AAAA", "CCCC", "DDDD", "EEEE"]
        y_true = [0, 0, 1, 1]

        monkeypatch.setattr(
            "sample.inference.predict",
            lambda seqs, model_id, model_dir=None: [
                {"probability": p, "predicted_label": int(p >= 0.5)}
                for p in (0.1, 0.2, 0.8, 0.9)])
        macrel = _FakePredictor("macrel", [
            {"macrel_amp_probability": s} for s in (0.3, 0.4, 0.6, 0.7)])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [macrel])
        amplify = _FakePredictor("amplify", [
            {"amplify_score": s} for s in (0.2, 0.3, 0.7, 0.8)])
        ampidentifier = _FakePredictor("ampidentifier", [
            {"ampidentifier_score": s} for s in (0.25, 0.35, 0.75, 0.85)])
        ampeppy = _FakePredictor("ampeppy", [
            {"ampeppy_amp_probability": s} for s in (0.15, 0.45, 0.65, 0.95)])
        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (amplify, ampidentifier, ampeppy))
        monkeypatch.setattr(
            "sample.model_registry.get_model_dataset",
            lambda model_id, model_dir=None: {
                "holdout": {"roc_auc": 0.91},
                "y_randomization": {"mean_roc_auc": 0.52},
                "null_comparison": {"roc_auc": {"primary_pvalue": 0.01, "primary_test": "kruskal"}},
            })

        report = benchmark.benchmark_model(
            "antibacterial/general/any/classification/descriptors", sequences, y_true)

        tool_names = {c["reference_tool"] for c in report["comparisons"]}
        assert tool_names == {"Macrel", "AMPlify", "AMPidentifier", "amPEPpy"}
        assert report["model_vs_chance"]["holdout_roc_auc"] == 0.91
        assert report["model_vs_chance"]["y_randomization_mean_roc_auc"] == 0.52
        # Backward-compatible top-level keys mirror the FIRST matching tool (Macrel, registered
        # before AMPlify, AMPidentifier and amPEPpy in REFERENCE_TOOLS).
        assert report["reference_tool"] == "Macrel"
        assert "model_ci95" in report and "roc_auc" in report["model_ci95"]


class TestReferenceScoresChunking:
    """reference_scores calls the predictor in chunks of REFERENCE_CHUNK_SIZE, for every tool --
    neither a single whole-list batch call nor one sequence per call. A chunk that comes back with
    nothing usable is retried one sequence at a time, so a bad sequence costs at most one wasted
    chunk call plus REFERENCE_CHUNK_SIZE single-sequence calls, not the whole comparison."""

    class _FailsOnOneSequenceWithinAChunk:
        """Fails (returns nothing usable) for the WHOLE call whenever it includes the one poison
        sequence -- exactly AMPlify's real, observed failure mode: a batch containing an invalid
        sequence returns nothing usable for the whole call, not just that one entry. Succeeds
        normally otherwise, whatever the call size."""

        name = "amplify"

        def available(self) -> bool:
            return True

        def predict(self, sequences: Any) -> List[dict]:
            if "BADSEQ" in sequences:
                return [{} for _ in sequences]
            return [{"amplify_score": 0.7, "amplify_is_amp": 1.0} for _ in sequences]

    def test_a_bad_sequence_only_costs_its_own_chunk_not_the_whole_comparison(
            self, monkeypatch: Any) -> None:
        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (self._FailsOnOneSequenceWithinAChunk(),))
        tool = benchmark.ReferenceTool(
            key="antibacterial", predictor="amplify", score_key="amplify_score",
            label_key="amplify_is_amp", display_name="AMPlify", best_effort=True)

        result = benchmark.reference_scores(tool, ["AAAA", "BADSEQ", "CCCC"])

        assert result["available"] is True
        assert result["scores"][0] == pytest.approx(0.7)
        assert result["scores"][2] == pytest.approx(0.7)
        import math
        assert math.isnan(result["scores"][1])

    def test_sequences_are_grouped_into_chunks_not_one_call_per_sequence_or_one_whole_batch(
            self, monkeypatch: Any) -> None:
        calls: List[List[str]] = []

        class _RecordingPredictor:
            name = "macrel"

            def available(self) -> bool:
                return True

            def predict(self, sequences: Any) -> List[dict]:
                calls.append(list(sequences))
                return [{"macrel_amp_probability": 0.5} for _ in sequences]

        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [_RecordingPredictor()])
        tool = benchmark.ReferenceTool(
            key="antibacterial", predictor="macrel", score_key="macrel_amp_probability",
            label_key=None, display_name="Macrel", best_effort=False)
        monkeypatch.setattr(benchmark, "REFERENCE_CHUNK_SIZE", 10)

        sequences = [f"SEQ{i}" for i in range(12)]
        result = benchmark.reference_scores(tool, sequences)

        assert calls == [sequences[:10], sequences[10:]]
        assert result["scores"] == [0.5] * 12

    def test_whole_chunk_failure_retries_one_sequence_at_a_time_for_just_that_chunk(
            self, monkeypatch: Any) -> None:
        calls: List[List[str]] = []

        class _CrashesOnChunkWithBadSeq:
            name = "amplify"

            def available(self) -> bool:
                return True

            def predict(self, sequences: Any) -> List[dict]:
                calls.append(list(sequences))
                if "BADSEQ" in sequences:
                    return [{} for _ in sequences]
                return [{"amplify_score": 0.9, "amplify_is_amp": 1.0} for _ in sequences]

        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (_CrashesOnChunkWithBadSeq(),))
        tool = benchmark.ReferenceTool(
            key="antibacterial", predictor="amplify", score_key="amplify_score",
            label_key="amplify_is_amp", display_name="AMPlify", best_effort=True)
        monkeypatch.setattr(benchmark, "REFERENCE_CHUNK_SIZE", 3)

        result = benchmark.reference_scores(tool, ["AAAA", "BADSEQ", "CCCC"])

        # First call: the whole 3-sequence chunk, which fails because of BADSEQ. Then exactly 3
        # single-sequence retry calls for that same chunk -- never a call mixing sizes in between.
        assert calls[0] == ["AAAA", "BADSEQ", "CCCC"]
        assert calls[1:] == [["AAAA"], ["BADSEQ"], ["CCCC"]]
        assert result["scores"][0] == pytest.approx(0.9)
        assert result["scores"][2] == pytest.approx(0.9)
        import math
        assert math.isnan(result["scores"][1])


class TestSampleEntryIds:
    """_sample_entry_ids caps a model's holdout to a practical size for benchmarking, the same
    way every time for the same model -- benchmark_holdout and holdout_fasta both call it and
    must agree on which peptides "the benchmark" means without one asking the other."""

    def test_returns_everything_unchanged_when_already_at_or_under_the_cap(self) -> None:
        ids = [f"S{i}" for i in range(50)]
        assert benchmark._sample_entry_ids(ids, size=50) == ids
        assert benchmark._sample_entry_ids(ids, size=1000) == ids

    def test_caps_at_the_requested_size(self) -> None:
        ids = [f"S{i}" for i in range(5000)]
        sample = benchmark._sample_entry_ids(ids, size=1000)
        assert len(sample) == 1000
        # Every sampled id genuinely came from the input, no duplicates.
        assert len(set(sample)) == 1000
        assert set(sample) <= set(ids)

    def test_same_seed_gives_the_same_sample_every_time(self) -> None:
        ids = [f"S{i}" for i in range(5000)]
        first = benchmark._sample_entry_ids(ids, size=100, seed=7)
        second = benchmark._sample_entry_ids(ids, size=100, seed=7)
        assert first == second

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        ids = [f"S{i}" for i in range(5000)]
        a = benchmark._sample_entry_ids(ids, size=100, seed=1)
        b = benchmark._sample_entry_ids(ids, size=100, seed=2)
        assert a != b


class TestHoldoutFasta:
    """holdout_fasta exports exactly the peptides benchmark_holdout scores a model and its
    reference tool against, so a tool with no ExternalPredictor wrapper can still be compared on
    the identical holdout by hand."""

    @staticmethod
    def _write_dataset_csv(tmp_path: Any) -> str:
        path = tmp_path / "final_dataset.csv"
        path.write_text(
            "entry_id,sequence,other_column\n"
            "SAMPLE01,AAAA,x\n"
            "SAMPLE02,CCCC,y\n"
            "SAMPLE03,DDDD,z\n"
            "SAMPLE04,EEEE,w\n")
        return str(path)

    def test_exports_only_the_saved_holdout_sequences_in_order(
            self, monkeypatch: Any, tmp_path: Any) -> None:
        dataset_csv = self._write_dataset_csv(tmp_path)
        monkeypatch.setattr(
            "sample.model_registry.get_model_dataset",
            lambda model_id, model_dir=None: {"test_entry_ids": ["SAMPLE04", "SAMPLE02"]})

        fasta = benchmark.holdout_fasta("hemolytic/general/any/classification/descriptors",
                                        dataset_csv=dataset_csv)

        assert fasta == ">SAMPLE04\nEEEE\n>SAMPLE02\nCCCC\n"

    def test_raises_without_a_stored_dataset(self, tmp_path: Any, monkeypatch: Any) -> None:
        dataset_csv = self._write_dataset_csv(tmp_path)
        monkeypatch.setattr("sample.model_registry.get_model_dataset",
                            lambda model_id, model_dir=None: None)
        with pytest.raises(ValueError):
            benchmark.holdout_fasta("hemolytic/general/any/classification/descriptors",
                                    dataset_csv=dataset_csv)

    def test_a_holdout_id_missing_from_the_dataset_csv_is_skipped_not_fatal(
            self, monkeypatch: Any, tmp_path: Any) -> None:
        dataset_csv = self._write_dataset_csv(tmp_path)
        monkeypatch.setattr(
            "sample.model_registry.get_model_dataset",
            lambda model_id, model_dir=None: {"test_entry_ids": ["SAMPLE01", "GONE"]})

        fasta = benchmark.holdout_fasta("hemolytic/general/any/classification/descriptors",
                                        dataset_csv=dataset_csv)

        assert fasta == ">SAMPLE01\nAAAA\n"

    def test_does_not_depend_on_build_tasks_matching_the_models_namespace(
            self, monkeypatch: Any, tmp_path: Any) -> None:
        # The regression this guards against: a model trained under an older build_tasks scoping
        # rule has an on-disk namespace that a later build_tasks() no longer reproduces (see
        # _holdout_eval_set, which _would_ raise here). A pure sequence export has no reason to
        # depend on that at all -- it needs test_entry_ids, nothing else.
        dataset_csv = self._write_dataset_csv(tmp_path)
        monkeypatch.setattr(
            "sample.model_registry.get_model_dataset",
            lambda model_id, model_dir=None: {"test_entry_ids": ["SAMPLE01"]})

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("holdout_fasta must not call build_tasks")

        monkeypatch.setattr("sample.ml_models.build_tasks", _boom)

        fasta = benchmark.holdout_fasta("some/namespace/that/no/longer/matches/anything",
                                        dataset_csv=dataset_csv)

        assert fasta == ">SAMPLE01\nAAAA\n"

    def test_a_holdout_larger_than_the_benchmark_sample_size_is_capped_to_the_same_ids_every_time(
            self, monkeypatch: Any, tmp_path: Any) -> None:
        # holdout_fasta must sample exactly the way benchmark_holdout does (same helper, same
        # seed) -- otherwise a peptide handed to some other tool by hand would not be one of the
        # peptides the benchmark itself actually used.
        path = tmp_path / "final_dataset.csv"
        n = 50
        lines = ["entry_id,sequence"] + [f"S{i},{'A' * (8 + i % 5)}" for i in range(n)]
        path.write_text("\n".join(lines) + "\n")
        all_ids = [f"S{i}" for i in range(n)]
        monkeypatch.setattr("sample.model_registry.get_model_dataset",
                            lambda model_id, model_dir=None: {"test_entry_ids": all_ids})
        monkeypatch.setattr(benchmark, "BENCHMARK_SAMPLE_SIZE", 10)

        fasta = benchmark.holdout_fasta("hemolytic/general/any/classification/descriptors",
                                        dataset_csv=str(path))
        ids_in_fasta = [line[1:] for line in fasta.splitlines() if line.startswith(">")]

        assert len(ids_in_fasta) == 10
        assert ids_in_fasta == benchmark._sample_entry_ids(all_ids, size=10)


class TestScoreModelFromDataset:
    """_score_model_from_dataset scores a model straight off dataset_csv's own feature columns,
    the ones the enrichment step already computed -- instead of sample.inference.predict, which
    would re-derive them (an ESM-2 forward pass, for a model using that block) from the raw
    sequence on every call. See benchmark_holdout, which uses this to avoid re-embedding a
    model's own already-embedded holdout on every benchmark run."""

    def test_scores_a_classifier_in_the_requested_entry_id_order(self, monkeypatch: Any) -> None:
        import pandas as pd

        df = pd.DataFrame({"entry_id": ["S1", "S2", "S3"],
                           "sequence": ["AAAA", "CCCC", "DDDD"],
                           "score": [0.1, 0.9, 0.5]})
        monkeypatch.setattr(
            "sample.model_registry.load_model",
            lambda model_id, model_dir=None: (
                _FakeEstimator(), ["score"], {"learning_task": "classification"}))

        # Requested out of the dataframe's own row order, and skipping S2 -- confirms this looks
        # each id up rather than assuming positional alignment with df.
        scores = benchmark._score_model_from_dataset(df, ["S3", "S1"], "some/model/id")

        assert scores == pytest.approx([0.5, 0.1])

    def test_scores_a_regressor_via_predict_not_predict_proba(self, monkeypatch: Any) -> None:
        import pandas as pd

        df = pd.DataFrame({"entry_id": ["S1"], "sequence": ["AAAA"], "score": [3.2]})
        monkeypatch.setattr(
            "sample.model_registry.load_model",
            lambda model_id, model_dir=None: (
                _FakeEstimator(), ["score"], {"learning_task": "regression"}))

        scores = benchmark._score_model_from_dataset(df, ["S1"], "some/model/id")

        assert scores == pytest.approx([3.2])

    def test_raises_when_a_feature_column_is_missing_from_the_dataset(
            self, monkeypatch: Any) -> None:
        import pandas as pd

        df = pd.DataFrame({"entry_id": ["S1"], "sequence": ["AAAA"]})
        monkeypatch.setattr(
            "sample.model_registry.load_model",
            lambda model_id, model_dir=None: (
                _FakeEstimator(), ["esm2_dim_0"], {"learning_task": "classification"}))

        with pytest.raises(ValueError):
            benchmark._score_model_from_dataset(df, ["S1"], "some/model/id")


class TestHoldoutEvalSetDerivesDpcCtdColumns:
    """
    Regression test for a real production incident: dataset_csv never stores dpc_*/ctd_*
    (add_derived_feature_columns computes them from `sequence` on demand rather than bloating the
    774MB release file with 547 extra columns) -- so a FLAML-trained model, which needs them,
    always failed _score_model_from_dataset's column check and fell back to
    sample.inference.predict, re-deriving every feature for every sampled peptide one at a time.
    Measured directly against a real benchmark run: ~20 minutes for a 1,000-peptide sample, against
    ~20 seconds to derive the columns once for the whole dataset. _holdout_eval_set must do that
    derivation itself so the fast, cached-feature path in benchmark_holdout actually has something
    to read.
    """

    def test_the_returned_dataframe_carries_dpc_and_ctd_columns(
            self, tmp_path: Any, monkeypatch: Any) -> None:
        import pandas as pd

        dataset_csv = tmp_path / "final_dataset.csv"
        pd.DataFrame([
            {"entry_id": "P1", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "is_decoy": 0},
            {"entry_id": "P2", "sequence": "ILPWKWPWWPWRR", "is_decoy": 0},
            {"entry_id": "D1", "sequence": "MFGLGGAAPQISSQQKLQAAEA", "is_decoy": 1},
        ]).to_csv(dataset_csv, index=False)
        activities_csv = tmp_path / "activities.csv"
        pd.DataFrame([
            {"sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "target_group": "antibacterial"},
            {"sequence": "ILPWKWPWWPWRR", "target_group": "antibacterial"},
        ]).to_csv(activities_csv, index=False)

        monkeypatch.setattr(
            "sample.model_registry.get_model_dataset",
            lambda model_id, model_dir=None: {"test_entry_ids": ["P1", "P2"]})
        # build_tasks' own MIN_POSITIVES_PER_TASK default (60) is far above this fixture's 2 real
        # peptides -- lowered so antibacterial/general/any/classification actually resolves,
        # exactly as tests/test_ml_models.py's own end-to-end fixture does.
        monkeypatch.setattr("sample.ml_models.MIN_POSITIVES_PER_TASK", 2)

        sequences, y_true, entry_ids, df = benchmark._holdout_eval_set(
            "antibacterial/general/any/classification/flaml", str(dataset_csv),
            str(activities_csv))

        assert "dpc_AA" in df.columns
        assert "ctd_comp_charge_g1" in df.columns
        # And the derivation must be COMPLETE, not merely present as an all-zero placeholder --
        # a real dipeptide count for a peptide that actually contains that pair.
        assert df.loc[df["entry_id"] == "P1", "dpc_GI"].iloc[0] > 0


class TestBenchmarkHoldoutPrefersCachedFeatures:
    """benchmark_holdout scores the sAMPle model from dataset_csv's cached columns rather than
    sample.inference.predict, and only falls back to predict() when that raises (a dataset built
    before this model's feature block existed)."""

    def test_scores_from_cached_columns_and_never_calls_predict(self, monkeypatch: Any) -> None:
        def _must_not_be_called(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("benchmark_holdout must not call sample.inference.predict when "
                                 "cached-feature scoring succeeds")

        monkeypatch.setattr("sample.inference.predict", _must_not_be_called)
        monkeypatch.setattr(
            benchmark, "_holdout_eval_set",
            lambda model_id, dataset_csv, activities_csv, model_dir=None: (
                ["AAAA", "CCCC"], [0, 1], ["S1", "S2"], "fake_df"))
        monkeypatch.setattr(
            benchmark, "_score_model_from_dataset",
            lambda df, entry_ids, model_id, model_dir=None: [0.1, 0.9])
        fake = _FakePredictor("hemopi2", [{"hemopi2_score": s} for s in (0.2, 0.8)])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        # hemolytic's second, best_effort tool (ToxinPred2) resolves via _benchmark_only_predictors
        # rather than EXTERNAL_PREDICTORS, so mocking that alone leaves it pointed at the REAL,
        # installed ToxinPred2Predictor -- stood in here as unavailable to keep this hermetic.
        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (_FakePredictor("toxinpred2", [], available=False),))

        report = benchmark.benchmark_holdout("hemolytic/general/any/classification/descriptors")

        assert report["model"]["roc_auc"] == pytest.approx(1.0)
        assert report["n_holdout"] == 2

    def test_falls_back_to_predict_when_cached_scoring_raises(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            benchmark, "_holdout_eval_set",
            lambda model_id, dataset_csv, activities_csv, model_dir=None: (
                ["AAAA", "CCCC"], [0, 1], ["S1", "S2"], "fake_df"))

        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise ValueError("missing feature column")

        monkeypatch.setattr(benchmark, "_score_model_from_dataset", _boom)
        monkeypatch.setattr(
            "sample.inference.predict",
            lambda seqs, model_id, model_dir=None: [
                {"probability": p, "predicted_label": int(p >= 0.5)} for p in (0.1, 0.9)])
        fake = _FakePredictor("hemopi2", [{"hemopi2_score": s} for s in (0.2, 0.8)])
        monkeypatch.setattr("sample.peptide_predictors.EXTERNAL_PREDICTORS", [fake])
        monkeypatch.setattr(benchmark, "_benchmark_only_predictors",
                            lambda: (_FakePredictor("toxinpred2", [], available=False),))

        report = benchmark.benchmark_holdout("hemolytic/general/any/classification/descriptors")

        assert report["model"]["roc_auc"] == pytest.approx(1.0)
