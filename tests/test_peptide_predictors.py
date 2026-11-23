"""
Tests for the external activity predictor adapters (`sample.peptide_predictors`).

`MacrelPredictor` had never been tested and turned out to have two real bugs, found while wiring
it into the annotator this session: it did not pass `--keep-negatives`, so macrel silently dropped
every peptide it did not call an AMP from the output rather than writing a negative row, and its
output-file filter (`.tsv`/`.tsv.gz`) never matched macrel's real filename
(`<tag>.prediction.gz`) -- so even a positive call was never actually parsed. `available()`
reported the tool as usable the whole time; every real prediction silently came back empty. These
tests are hermetic (mock `subprocess.run` to write output in macrel's real, verified shape) so they
run in the `sample` test environment, which does not have macrel installed -- what they check is
this project's own parsing logic, not macrel's.
"""
import gzip
import os
from typing import Any, Dict, List, Optional

import pytest

#: macrel v1.6.1's real output, byte for byte as produced by a live run with --keep-negatives:
#: tab-separated, a leading "#" comment line, and `is_AMP` present as its own column only because
#: negatives are being kept at all.
_MACREL_REAL_OUTPUT = (
    "# Prediction from macrel v1.6.1\n"
    "Access\tSequence\tAMP_family\tis_AMP\tAMP_probability\tHemolytic\tHemolytic_probability\n"
    "s0\tGIGKFLHSAKKFGKAFVGEIMNS\tCLP\tTrue\t0.950\tHemo\t0.911\n"
    "s1\tDDDDEEEEDDDDEEEEDDDDEEEEDDDD\tALP\tFalse\t0.455\tNonHemo\t0.069\n"
)


def _fake_macrel_run(cmd: List[str], **kwargs: Any) -> "FakeCompletedProcess":
    """Stands in for `subprocess.run(["macrel", "peptides", ...])`: writes the real output shape
    into the `--output` directory the real call would have been given, at the real filename."""
    out_dir = cmd[cmd.index("--output") + 1]
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, "macrel.out.prediction.gz"), "wt") as handle:
        handle.write(_MACREL_REAL_OUTPUT)

    class FakeCompletedProcess:
        returncode = 0

    return FakeCompletedProcess()


class TestMacrelPredictor:
    def test_parses_real_output_shape_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp.subprocess, "run", _fake_macrel_run)

        predictor = pp.MacrelPredictor()
        predictor._command = ["macrel"]  # bypass available(); this test is about parsing, not PATH
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])

        assert results[0]["macrel_is_amp"] == 1.0
        assert results[0]["macrel_amp_probability"] == pytest.approx(0.950)
        assert results[0]["macrel_hemolytic_probability"] == pytest.approx(0.911)
        assert results[1]["macrel_is_amp"] == 0.0, \
            "a peptide macrel calls negative must still produce a result, not be silently dropped"
        assert results[1]["macrel_amp_probability"] == pytest.approx(0.455)

    def test_invocation_includes_keep_negatives(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression test for the bug itself: without `--keep-negatives`, macrel omits every
        peptide it does not call an AMP from the output file entirely."""
        import sample.peptide_predictors as pp

        captured_cmd: List[List[str]] = []

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            captured_cmd.append(cmd)
            return _fake_macrel_run(cmd, **kwargs)

        monkeypatch.setattr(pp.subprocess, "run", capturing_run)

        predictor = pp.MacrelPredictor()
        predictor._command = ["macrel"]
        predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        assert captured_cmd, "subprocess.run must have been called"
        assert "--keep-negatives" in captured_cmd[0]

    def test_invocation_uses_the_resolved_command_not_a_bare_name(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression test: `_run` hardcoded a bare `"macrel"` instead of `self._command`, so it
        depended on macrel being on PATH regardless of what `available()`'s per-environment probe
        resolved -- defeating the whole reason each predictor gets its own isolated environment.
        Confirmed as a real bug on a host where the resolved absolute path exists but the bare name
        does not. `_command` here is deliberately NOT `["macrel"]`, so a hardcoded bare name would
        make this test's `subprocess.run` calls diverge from what got captured."""
        import sample.peptide_predictors as pp

        captured_cmd: List[List[str]] = []

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            captured_cmd.append(cmd)
            return _fake_macrel_run(cmd, **kwargs)

        monkeypatch.setattr(pp.subprocess, "run", capturing_run)

        predictor = pp.MacrelPredictor()
        predictor._command = ["/opt/envs/sample-macrel/bin/macrel"]
        predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        assert captured_cmd, "subprocess.run must have been called"
        assert captured_cmd[0][0] == "/opt/envs/sample-macrel/bin/macrel"

    def test_unavailable_returns_empty_without_calling_subprocess(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when unavailable")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.MacrelPredictor()
        predictor._command = None
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_output_filter_matches_the_real_filename(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`<tag>.prediction.gz` -- not `.tsv.gz` -- is what macrel actually writes; a decoy file
        with an unrelated name must not be mistaken for it."""
        import sample.peptide_predictors as pp

        def run_with_decoy_file(cmd: List[str], **kwargs: Any) -> Any:
            out_dir = cmd[cmd.index("--output") + 1]
            os.makedirs(out_dir, exist_ok=True)
            # An unrelated .tsv.gz file that must be ignored, alongside the real output.
            with gzip.open(os.path.join(out_dir, "unrelated.tsv.gz"), "wt") as handle:
                handle.write("not\tmacrel\toutput\n")
            with gzip.open(os.path.join(out_dir, "macrel.out.prediction.gz"), "wt") as handle:
                handle.write(_MACREL_REAL_OUTPUT)

            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", run_with_decoy_file)

        predictor = pp.MacrelPredictor()
        predictor._command = ["macrel"]
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        assert results[0].get("macrel_is_amp") == 1.0

    def test_subprocess_failure_returns_empty_results_not_an_exception(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("macrel crashed")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)

        predictor = pp.MacrelPredictor()
        predictor._command = ["macrel"]
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        assert results == [{}]

    def test_unavailable_when_not_on_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The same graceful-degradation path every adapter relies on: a tool missing from both
        the isolated predictor environment and `PATH` must report unavailable and contribute
        nothing, not raise."""
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp, "_predictor_command", lambda executable, env=None: None)

        predictor = pp.MacrelPredictor()
        assert predictor.available() is False
        assert predictor.predict(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_each_tool_is_probed_in_its_own_environment(self,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
        """Per-tool isolation: each adapter must look for its executable in ITS environment, not one
        shared predictors env -- otherwise the dedicated env files (sample-macrel, sample-anticp2)
        are never consulted and the pins they exist to hold are moot.

        AMPlify is checked separately, by instantiating it directly rather than through
        `EXTERNAL_PREDICTORS`: it was pulled from that list (2026-08-26, see its own comment there)
        after crashing on 7-12%+ of batches in every real production run, but the class itself is
        kept as a working reference, and its per-environment isolation is still worth verifying."""
        import sample.peptide_predictors as pp

        probed: dict = {}

        def _record(executable: str, env=None):
            probed[executable] = env
            return None

        monkeypatch.setattr(pp, "_predictor_command", _record)
        for predictor in pp.EXTERNAL_PREDICTORS:
            predictor._available = None  # clear the cached probe
            predictor.available()

        assert probed["macrel"] == "sample-macrel"
        assert probed["anticp2"] == "sample-anticp2"
        assert probed["algpred2"] == "sample-algpred2"
        # HemoPI2 stays in the shared predictors environment (its default).
        assert probed["hemopi2_classification"] == pp.PREDICTOR_ENV

        probed.clear()
        amplify = pp.AMPlifyPredictor()
        amplify.available()
        assert probed["AMPlify"] == "sample-amplify"

    def test_available_in_its_own_real_environment(self) -> None:
        """Not mocked: `sample-macrel` (built via `make env-macrel`, per
        `environments/sample-macrel.yml`'s documented reason for its own environment -- macrel's
        scikit-learn/numpy pin conflicts with hemopi2/toxinpred3's) is a real, built environment on
        this host, and macrel genuinely runs in it. This is the real, current state, not a mock --
        confirms the adapter finds the tool where the environment file says it lives."""
        import sample.peptide_predictors as pp

        assert pp.MacrelPredictor().available() is True


_ANTICP2_REAL_OUTPUT = (
    "# Sequence_ID, Sequence, Prediction, Score\n"
    ">s0,GIGKFLHSAKKFGKAFVGEIMNS,AntiCP,1.000000\n"
    ">s1,DDDDEEEEDDDDEEEEDDDDEEEEDDDD,Non-AntiCP,0.120000\n")


class TestAntiCP2Predictor:
    def test_parses_real_output_shape_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            out_csv = cmd[cmd.index("-o") + 1]
            with open(out_csv, "w", encoding="utf-8") as fh:
                fh.write(_ANTICP2_REAL_OUTPUT)

            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)

        predictor = pp.AntiCP2Predictor()
        predictor._command = ["anticp2"]
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])

        assert results[0]["anticp_is_anticancer"] == 1.0
        assert results[0]["anticp_score"] == pytest.approx(1.0)
        assert results[1]["anticp_is_anticancer"] == 0.0, \
            "a peptide called Non-AntiCP must still produce a result, not be silently dropped"
        assert results[1]["anticp_score"] == pytest.approx(0.12)

    def test_comment_header_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            out_csv = cmd[cmd.index("-o") + 1]
            with open(out_csv, "w", encoding="utf-8") as fh:
                fh.write(_ANTICP2_REAL_OUTPUT)
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.AntiCP2Predictor()
        predictor._command = ["anticp2"]
        # No exception from trying to parse the "# Sequence_ID, ..." line as a result row.
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])
        assert len(results) == 2

    def test_subprocess_failure_returns_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("anticp2 crashed")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        predictor = pp.AntiCP2Predictor()
        predictor._command = ["anticp2"]
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_unavailable_returns_empty_without_calling_subprocess(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when unavailable")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.AntiCP2Predictor()
        predictor._command = None
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]


_ALGPRED2_REAL_OUTPUT = (
    "ID,Sequence,ML_Score,Prediction\n"
    "s0,GIGKFLHSAKKFGKAFVGEIMNS,0.434,Allergen\n"
    "s1,DDDDEEEEDDDDEEEEDDDDEEEEDDDD,0.120,Non-Allergen\n")


class TestAlgPred2Predictor:
    def test_parses_real_output_shape_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            out_csv = cmd[cmd.index("-o") + 1]
            with open(out_csv, "w", encoding="utf-8") as fh:
                fh.write(_ALGPRED2_REAL_OUTPUT)

            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)

        predictor = pp.AlgPred2Predictor()
        predictor._command = ["algpred2"]
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])

        assert results[0]["algpred2_is_allergen"] == 1.0
        assert results[0]["algpred2_score"] == pytest.approx(0.434)
        assert results[1]["algpred2_is_allergen"] == 0.0, \
            "a peptide called Non-Allergen must still produce a result, not be silently dropped"
        assert results[1]["algpred2_score"] == pytest.approx(0.12)

    def test_display_all_flag_is_passed_so_non_allergens_are_not_filtered(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """algpred2's own `-d` flag defaults to showing allergen calls only; `_run` must pass
        `-d 2` so a genuinely non-allergenic peptide still gets a row in the output, not silence."""
        import sample.peptide_predictors as pp

        captured: dict = {}

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            captured["cmd"] = cmd
            out_csv = cmd[cmd.index("-o") + 1]
            with open(out_csv, "w", encoding="utf-8") as fh:
                fh.write(_ALGPRED2_REAL_OUTPUT)
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.AlgPred2Predictor()
        predictor._command = ["algpred2"]
        predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])
        assert "-d" in captured["cmd"] and captured["cmd"][captured["cmd"].index("-d") + 1] == "2"

    def test_subprocess_failure_returns_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("algpred2 crashed")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        predictor = pp.AlgPred2Predictor()
        predictor._command = ["algpred2"]
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_unavailable_returns_empty_without_calling_subprocess(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when unavailable")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.AlgPred2Predictor()
        predictor._command = None
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_available_in_its_own_real_environment(self) -> None:
        """Not mocked: `sample-algpred2` is a real, built environment on this host (see
        `environments/sample-algpred2.yml` for the scikit-learn==1.2.2 + numpy<2 pin its pickle
        needs), and algpred2 genuinely runs in it."""
        import sample.peptide_predictors as pp

        assert pp.AlgPred2Predictor().available() is True


_AMPLIFY_REAL_OUTPUT = (
    "Sequence_ID\tSequence\tLength\tCharge\tProbability_score\tAMPlify_log_scaled_score\tPrediction\n"
    "s0\tGIGKFLHSAKKFGKAFVGEIMNS\t23\t3\t0.99999297\t51.5284\tAMP\n"
    "s1\tDDDDEEEEDDDDEEEEDDDDEEEEDDDD\t28\t-8\t0.01\t2.1\tNon-AMP\n")


class TestAMPlifyPredictor:
    def test_locates_python_and_script_next_to_the_environment(self, tmp_path: Any) -> None:
        """`_command` names the `AMPlify` wrapper (from the base class's probing); `_run` must
        derive the environment's own python and AMPlify.py from its directory, not use the wrapper
        directly -- the wrapper's own shebang resolves through the CALLER's PATH and silently picks
        up the wrong interpreter (confirmed on a real host: ModuleNotFoundError: No module named
        'keras' when invoked outside its own environment)."""
        import sample.peptide_predictors as pp

        env_root = tmp_path / "sample-amplify"
        (env_root / "bin").mkdir(parents=True)
        (env_root / "bin" / "python").write_text("#!/bin/sh\n")
        script_dir = env_root / "share" / "amplify" / "src"
        script_dir.mkdir(parents=True)
        (script_dir / "AMPlify.py").write_text("# stub\n")

        predictor = pp.AMPlifyPredictor()
        predictor._command = [str(env_root / "bin" / "AMPlify")]

        captured: List[List[str]] = []

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            captured.append(cmd)
            out_dir = cmd[cmd.index("-od") + 1]
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "out.tsv"), "w", encoding="utf-8") as fh:
                fh.write(_AMPLIFY_REAL_OUTPUT)
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        import sample.peptide_predictors as pp_mod
        original_run = pp_mod.subprocess.run
        pp_mod.subprocess.run = capturing_run
        try:
            results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])
        finally:
            pp_mod.subprocess.run = original_run

        assert captured, "subprocess.run must have been called"
        assert captured[0][0] == str(env_root / "bin" / "python")
        assert captured[0][1] == str(script_dir / "AMPlify.py")
        assert results[0]["amplify_is_amp"] == 1.0
        assert results[0]["amplify_score"] == pytest.approx(0.99999297)
        assert results[1]["amplify_is_amp"] == 0.0
        assert results[1]["amplify_score"] == pytest.approx(0.01)

    def test_missing_script_returns_empty_without_calling_subprocess(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when the script cannot be located")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.AMPlifyPredictor()
        predictor._command = [str(tmp_path / "nonexistent" / "bin" / "AMPlify")]
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_subprocess_failure_returns_empty_results(self, tmp_path: Any) -> None:
        import sample.peptide_predictors as pp

        env_root = tmp_path / "sample-amplify"
        (env_root / "bin").mkdir(parents=True)
        (env_root / "bin" / "python").write_text("#!/bin/sh\n")
        script_dir = env_root / "share" / "amplify" / "src"
        script_dir.mkdir(parents=True)
        (script_dir / "AMPlify.py").write_text("# stub\n")

        predictor = pp.AMPlifyPredictor()
        predictor._command = [str(env_root / "bin" / "AMPlify")]

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("amplify crashed")

        import sample.peptide_predictors as pp_mod
        original_run = pp_mod.subprocess.run
        pp_mod.subprocess.run = failing_run
        try:
            assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]
        finally:
            pp_mod.subprocess.run = original_run

    def test_a_calledprocesserror_surfaces_its_real_stderr(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: a real production run (2027-8-24) logged nothing but `amplify
        failed: Command '[...]' returned non-zero exit status 1.` on ~1000 peptides -- the exact
        same bug already found and fixed in HemoPI2Predictor, present here too because each
        predictor's `except Exception as e: logger.warning(f"... failed: {e}")` had its own copy
        of the same mistake. Both now go through the shared `_log_subprocess_failure`."""
        import subprocess

        import sample.peptide_predictors as pp

        env_root = tmp_path / "sample-amplify"
        (env_root / "bin").mkdir(parents=True)
        (env_root / "bin" / "python").write_text("#!/bin/sh\n")
        script_dir = env_root / "share" / "amplify" / "src"
        script_dir.mkdir(parents=True)
        (script_dir / "AMPlify.py").write_text("# stub\n")

        predictor = pp.AMPlifyPredictor()
        predictor._command = [str(env_root / "bin" / "AMPlify")]

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, output=b"",
                stderr=b"tensorflow.python.framework.errors_impl.ResourceExhaustedError\n")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        messages: List[str] = []
        monkeypatch.setattr(pp.logger, "warning", lambda msg: messages.append(msg))

        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]
        assert messages
        assert "ResourceExhaustedError" in messages[0], \
            "the captured stderr must reach the log, not just the generic exit-status message"

    def test_a_sequence_amplify_would_reject_never_reaches_the_subprocess(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: AMPlify's own script drops a sequence outside its accepted footprint (here,
        one residue -- below its length-2 minimum) from its internal valid-sequence list, and
        CRASHES with an opaque tensor-shape error when that leaves the list empty, rather than
        reporting nothing scored. Confirmed against a real 12,712-peptide benchmark holdout: this
        specific case cost a full subprocess spin-up (TensorFlow/Keras import, five model loads)
        every time, for a call that could never have produced a score. It must never reach
        subprocess.run at all -- reference_scores calls this with exactly one sequence per call,
        so a single rejected sequence is the entire batch, not one bad entry among many good ones."""
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not spin up the subprocess for a sequence AMPlify's own "
                                 "script would reject anyway")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.AMPlifyPredictor()
        predictor._command = [str(tmp_path / "sample-amplify" / "bin" / "AMPlify")]

        assert predictor._run(["A"]) == [{}]

    def test_an_invalid_sequence_in_a_mixed_batch_is_excluded_but_does_not_block_the_rest(
            self, tmp_path: Any) -> None:
        import sample.peptide_predictors as pp

        env_root = tmp_path / "sample-amplify"
        (env_root / "bin").mkdir(parents=True)
        (env_root / "bin" / "python").write_text("#!/bin/sh\n")
        script_dir = env_root / "share" / "amplify" / "src"
        script_dir.mkdir(parents=True)
        (script_dir / "AMPlify.py").write_text("# stub\n")

        predictor = pp.AMPlifyPredictor()
        predictor._command = [str(env_root / "bin" / "AMPlify")]

        written_fasta: List[str] = []

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            fasta_path = cmd[cmd.index("-s") + 1]
            with open(fasta_path, encoding="utf-8") as fh:
                written_fasta.append(fh.read())
            out_dir = cmd[cmd.index("-od") + 1]
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "out.tsv"), "w", encoding="utf-8") as fh:
                fh.write("Sequence_ID\tSequence\tLength\tCharge\tProbability_score\t"
                        "AMPlify_log_scaled_score\tPrediction\n"
                        "s0\tGIGKFLHSAKKFGKAFVGEIMNS\t23\t3\t0.9\t10\tAMP\n")
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        import sample.peptide_predictors as pp_mod
        original_run = pp_mod.subprocess.run
        pp_mod.subprocess.run = capturing_run
        try:
            # "X" (a non-standard residue, e.g. from a shuffled decoy) is invalid; the other two
            # are fine and must still reach the FASTA and get scored.
            results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "X", "DDDDEEEEDDDDEEEEDDDD"])
        finally:
            pp_mod.subprocess.run = original_run

        assert written_fasta
        assert ">s1" not in written_fasta[0]
        assert ">s0" in written_fasta[0] and ">s2" in written_fasta[0]
        assert results[0]["amplify_is_amp"] == 1.0
        assert results[1] == {}


#: AMPidentifier 2.0.3's real output, byte for byte as produced by a live run of its voting-ensemble
#: default (`ampidentifier -i ... -o ...`, no `-m` flag): `predictions_voting.csv`, comma-separated,
#: with the probability already rounded to 4 places and the binary call in its own `prediction`
#: column (0/1) alongside a human-readable `label` ("AMP"/"non-AMP").
_AMPIDENTIFIER_REAL_OUTPUT = (
    "ID,sequence,probability_AMP,prediction,label\n"
    "s0,GIGKFLHSAKKFGKAFVGEIMNS,0.9959,1,AMP\n"
    "s1,DDDDEEEEDDDDEEEEDDDDEEEEDDDD,0.819,1,AMP\n")


def _fake_ampidentifier_run(cmd: List[str], **kwargs: Any) -> "FakeCompletedProcess":
    """Stands in for `subprocess.run([<ampidentifier>, "-i", ..., "-o", ...])`: writes the real
    output shape into the `-o` directory the real call would have been given, at the real,
    voting-ensemble-specific filename (`predictions_voting.csv` -- AMPidentifierPredictor never
    passes `-m`, so "voting", the tool's own default, is always the model in play)."""
    out_dir = cmd[cmd.index("-o") + 1]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions_voting.csv"), "w", encoding="utf-8") as fh:
        fh.write(_AMPIDENTIFIER_REAL_OUTPUT)

    class FakeCompletedProcess:
        returncode = 0

    return FakeCompletedProcess()


class TestAMPidentifierPredictor:
    """AMPidentifier is benchmark-only (`sample.benchmark._benchmark_only_predictors`), not part of
    the live `EXTERNAL_PREDICTORS` -- it exists purely as a fourth antibacterial comparator
    alongside Macrel and AMPlify, never as part of the live annotator's design. Its console-script
    (`self._command`, unlike AMPlify's derived python+script indirection) carries an ABSOLUTE
    shebang baked in at pip-install time, so it is invoked directly, the same way MacrelPredictor
    and AntiCP2Predictor invoke theirs."""

    def test_happy_path_parses_real_output_into_scores_and_calls(self, tmp_path: Any) -> None:
        import sample.peptide_predictors as pp

        predictor = pp.AMPidentifierPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampidentifier")]

        original_run = pp.subprocess.run
        pp.subprocess.run = _fake_ampidentifier_run
        try:
            results = predictor._run(
                ["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])
        finally:
            pp.subprocess.run = original_run

        assert results[0]["ampidentifier_score"] == pytest.approx(0.9959)
        assert results[0]["ampidentifier_is_amp"] == 1.0
        assert results[1]["ampidentifier_score"] == pytest.approx(0.819)
        assert results[1]["ampidentifier_is_amp"] == 1.0

    def test_passes_the_resolved_command_and_no_model_flag(self, tmp_path: Any) -> None:
        """`_run` must invoke `self._command` (the environment-resolved absolute path from
        `available()`'s probing) directly -- never a bare "ampidentifier" that would resolve
        through the caller's own PATH instead of the tool's isolated environment -- and must never
        pass `-m`/`--model`, since "voting" (the tool's own default and recommended mode) is what
        `predictions_voting.csv`'s fixed filename in `_run` assumes."""
        import sample.peptide_predictors as pp

        predictor = pp.AMPidentifierPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampidentifier")]

        captured: List[List[str]] = []

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            captured.append(cmd)
            return _fake_ampidentifier_run(cmd, **kwargs)

        original_run = pp.subprocess.run
        pp.subprocess.run = capturing_run
        try:
            predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])
        finally:
            pp.subprocess.run = original_run

        assert captured
        assert captured[0][0] == str(tmp_path / "bin" / "ampidentifier")
        assert "-m" not in captured[0] and "--model" not in captured[0]

    def test_missing_environment_returns_empty_results_without_calling_subprocess(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tool not being installed (`available()` false, `_command` never resolved) must
        return empty dicts for every sequence and never crash -- and never even attempt the
        subprocess call, exactly as `ExternalPredictor.predict` already guarantees for every
        adapter."""
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when the tool is not installed")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.AMPidentifierPredictor()
        monkeypatch.setattr(predictor, "available", lambda: False)

        assert predictor.predict(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDD"]) == [{}, {}]

    def test_subprocess_failure_returns_empty_results(self, tmp_path: Any) -> None:
        import sample.peptide_predictors as pp

        predictor = pp.AMPidentifierPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampidentifier")]

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("ampidentifier crashed")

        original_run = pp.subprocess.run
        pp.subprocess.run = failing_run
        try:
            assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]
        finally:
            pp.subprocess.run = original_run

    def test_a_calledprocesserror_surfaces_its_real_stderr(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same regression class as AMPlify's and hemopi2's own version of this bug: a bare
        `except Exception as e: logger.warning(f"... failed: {e}")` loses the real captured
        stderr and logs only the generic exit-status message. Must go through
        `_log_subprocess_failure`, which surfaces it."""
        import subprocess

        import sample.peptide_predictors as pp

        predictor = pp.AMPidentifierPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampidentifier")]

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, output=b"",
                stderr=b"ModuleNotFoundError: No module named '_loss'\n")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        messages: List[str] = []
        monkeypatch.setattr(pp.logger, "warning", lambda msg: messages.append(msg))

        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]
        assert messages
        assert "_loss" in messages[0], \
            "the captured stderr must reach the log, not just the generic exit-status message"

    def test_missing_output_file_returns_empty_results(self, tmp_path: Any) -> None:
        """A chunk where every sequence gets filtered out by AMPidentifier's own footprint (see
        TestAMPidentifierInputGuard) reaches subprocess.run with an effectively-trivial FASTA and
        exits 0 without ever writing `predictions_voting.csv` -- must not raise trying to open a
        file that was never produced."""
        import sample.peptide_predictors as pp

        predictor = pp.AMPidentifierPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampidentifier")]

        def no_output_run(cmd: List[str], **kwargs: Any) -> Any:
            out_dir = cmd[cmd.index("-o") + 1]
            os.makedirs(out_dir, exist_ok=True)

            class FakeCompletedProcess:
                returncode = 0

            return FakeCompletedProcess()

        original_run = pp.subprocess.run
        pp.subprocess.run = no_output_run
        try:
            assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]
        finally:
            pp.subprocess.run = original_run

    def test_a_sequence_ampidentifier_would_reject_never_reaches_the_subprocess(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: AMPidentifier's own feature extractor (modlamp's GlobalDescriptor) does a
        bare dict lookup per residue with no non-standard-residue handling, so ONE sequence with a
        residue outside the 20 standard amino acids (X/B/Z/*, ...) raises KeyError and kills
        feature extraction -- and therefore every prediction -- for the WHOLE submitted batch, not
        just that sequence. Confirmed against a real run: a single 'X'-suffixed sequence alone in a
        batch crashed with exit code 1 and no output file at all. Must never reach subprocess.run
        when it is the only sequence in the call."""
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError(
                "must not spin up the subprocess for a sequence AMPidentifier's own feature "
                "extractor would reject anyway")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.AMPidentifierPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampidentifier")]

        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNSX"]) == [{}]

    def test_an_invalid_sequence_in_a_mixed_batch_is_excluded_but_does_not_block_the_rest(
            self, tmp_path: Any) -> None:
        import sample.peptide_predictors as pp

        predictor = pp.AMPidentifierPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampidentifier")]

        written_fasta: List[str] = []

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            fasta_path = cmd[cmd.index("-i") + 1]
            with open(fasta_path, encoding="utf-8") as fh:
                written_fasta.append(fh.read())
            return _fake_ampidentifier_run(cmd, **kwargs)

        original_run = pp.subprocess.run
        pp.subprocess.run = capturing_run
        try:
            # "X" (a non-standard residue, e.g. from a shuffled decoy) is invalid; the other two
            # are fine and must still reach the FASTA and get scored (their real, mocked output
            # rows are keyed s0/s1 to match those two positions -- s2, the invalid one, is skipped
            # entirely so the position count in the mocked CSV still lines up).
            results = predictor._run(
                ["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD", "X"])
        finally:
            pp.subprocess.run = original_run

        assert written_fasta
        assert ">s2" not in written_fasta[0]
        assert ">s0" in written_fasta[0] and ">s1" in written_fasta[0]
        assert results[0]["ampidentifier_is_amp"] == 1.0
        assert results[1]["ampidentifier_is_amp"] == 1.0
        assert results[2] == {}


class TestAMPidentifierInputGuard:
    """Regression coverage for `_is_ampidentifier_valid` in isolation, mirroring
    `TestMacrelInputGuard`/`TestHemoPI2InputGuard` below for their own tools' crashing inputs."""

    def test_only_the_20_standard_residues_are_accepted(self) -> None:
        import sample.peptide_predictors as pp

        assert pp._is_ampidentifier_valid("GIGKFLHSAKKFGKAFVGEIMNS") is True
        assert pp._is_ampidentifier_valid("DDDDEEEEDDDDEEEEDDDDEEEEDDDD") is True
        # No length floor -- unlike AMPlify's length-2 minimum, a single residue is fine.
        assert pp._is_ampidentifier_valid("A") is True

        # These crash modlamp's GlobalDescriptor.calculate_MW with a bare KeyError.
        assert pp._is_ampidentifier_valid("X") is False
        assert pp._is_ampidentifier_valid("GIGKFLHSAKKFGKAFVGEIMNSX") is False
        assert pp._is_ampidentifier_valid("B") is False
        assert pp._is_ampidentifier_valid("") is False


_TOXINPRED2_REAL_OUTPUT = (
    "ID,Sequence,ML_Score,Prediction\n"
    "s0,GIGKFLHSAKKFGKAFVGEIMNS,0.474,Non-Toxin\n"
    "s1,DDDDEEEEDDDDEEEEDDDDEEEEDDDD,0.729,Toxin\n")


class TestToxinPred2Predictor:
    """ToxinPred2 is benchmark-only (`sample.benchmark._benchmark_only_predictors`), not part of
    the live `EXTERNAL_PREDICTORS` -- it predicts general toxicity, not hemolysis specifically, and
    is wired in only as an adjacent comparator for the "hemolytic" benchmark. Two real bugs were
    found against the actual installed tool (2026-09-04), both covered below: its own hybrid model
    (`-m 2`) crashes with a `NameError` on the common case of zero BLAST hits across a whole batch
    (worked around by never invoking it -- `_run` always uses `-m 1`), and its default AAC model
    crashes on a batch of exactly one sequence regardless of content, because `np.loadtxt` collapses
    a one-row feature file to 1-D and onnxruntime then rejects the rank (worked around by padding a
    lone sequence with a throwaway second one before ever calling the subprocess)."""

    def test_parses_real_output_shape_correctly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            out_csv = cmd[cmd.index("-o") + 1]
            with open(out_csv, "w", encoding="utf-8") as fh:
                fh.write(_TOXINPRED2_REAL_OUTPUT)

            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)

        predictor = pp.ToxinPred2Predictor()
        predictor._command = ["toxinpred2"]
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])

        assert results[0]["toxinpred2_is_toxic"] == 0.0
        assert results[0]["toxinpred2_score"] == pytest.approx(0.474)
        assert results[1]["toxinpred2_is_toxic"] == 1.0, \
            "a peptide called Toxin must still produce a result, not be silently dropped"
        assert results[1]["toxinpred2_score"] == pytest.approx(0.729)

    def test_model_and_display_flags_are_passed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`-m 1` (AAC-based RF, not the hybrid `-m 2` -- see class docstring for the confirmed
        `NameError` crash in the hybrid's BLAST-processing path) and `-d 2` (report every
        sequence -- `-d 1`, the tool's own default, silently drops every Non-Toxin call from the
        output, the same trap anticp2/algpred2 need `-d 2` for) must both reach the subprocess."""
        import sample.peptide_predictors as pp

        captured: dict = {}

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            captured["cmd"] = cmd
            out_csv = cmd[cmd.index("-o") + 1]
            with open(out_csv, "w", encoding="utf-8") as fh:
                fh.write(_TOXINPRED2_REAL_OUTPUT)
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.ToxinPred2Predictor()
        predictor._command = ["toxinpred2"]
        predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])
        cmd = captured["cmd"]
        assert cmd[cmd.index("-m") + 1] == "1"
        assert cmd[cmd.index("-d") + 1] == "2"

    def test_a_lone_sequence_is_padded_to_avoid_the_rank_collapse_crash(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: confirmed against the real installed tool that a batch of exactly one
        sequence crashes with `onnxruntime...InvalidArgument: Invalid rank for input: ... Got: 1
        Expected: 2`, regardless of the sequence's own content -- `np.loadtxt` on a one-row CSV
        collapses to 1-D. `_run` must submit a second, throwaway sequence whenever it would
        otherwise send only one, and that padding row must never leak into the returned result."""
        import sample.peptide_predictors as pp

        written_fasta: List[str] = []

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            with open(cmd[cmd.index("-i") + 1], encoding="utf-8") as fh:
                written_fasta.append(fh.read())
            out_csv = cmd[cmd.index("-o") + 1]
            with open(out_csv, "w", encoding="utf-8") as fh:
                fh.write("ID,Sequence,ML_Score,Prediction\n"
                        "s0,GIGKFLHSAKKFGKAFVGEIMNS,0.474,Non-Toxin\n"
                        "__toxinpred2_pad__,AAAA,0.1,Non-Toxin\n")
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.ToxinPred2Predictor()
        predictor._command = ["toxinpred2"]
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        assert written_fasta and written_fasta[0].count(">") == 2, \
            "a lone sequence must be padded with a second, throwaway one before ever reaching " \
            "the subprocess"
        assert len(results) == 1
        assert results[0]["toxinpred2_score"] == pytest.approx(0.474)

    def test_subprocess_failure_returns_empty_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("toxinpred2 crashed")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        predictor = pp.ToxinPred2Predictor()
        predictor._command = ["toxinpred2"]
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"]) == \
            [{}, {}]

    def test_a_calledprocesserror_surfaces_its_real_stderr(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import sample.peptide_predictors as pp

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, output=b"", stderr=b"NameError: name 'seqid' is not defined\n")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        messages: List[str] = []
        monkeypatch.setattr(pp.logger, "warning", lambda msg: messages.append(msg))
        predictor = pp.ToxinPred2Predictor()
        predictor._command = ["toxinpred2"]

        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"]) == \
            [{}, {}]
        assert messages
        assert "seqid" in messages[0], \
            "the captured stderr must reach the log, not just the generic exit-status message"

    def test_unavailable_returns_empty_without_calling_subprocess(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when unavailable")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.ToxinPred2Predictor()
        predictor._command = None
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_available_in_its_own_real_environment(self) -> None:
        """Not mocked: `sample-toxinpred2` is a real, built environment on this host (see
        `environments/sample-toxinpred2.yml`), and toxinpred2 genuinely runs in it."""
        import sample.peptide_predictors as pp

        assert pp.ToxinPred2Predictor().available() is True


#: amPEPpy's real output shape, byte for byte as produced by a live run of `ampep predict -m
#: pretrained_models/amPEP.model -i ... -o ...`: tab-separated, `probability_nonAMP` /
#: `probability_AMP` / `predicted` / `seq_id` columns in that fixed order, `predicted` a
#: human-readable "AMP"/"nonAMP" string rather than a 0/1 code.
_AMPEPPY_REAL_OUTPUT = (
    "probability_nonAMP\tprobability_AMP\tpredicted\tseq_id\n"
    "0.040625\t0.959375\tAMP\ts0\n"
    "0.3734375\t0.6265625\tAMP\ts1\n")


def _fake_ampeppy_run(cmd: List[str], **kwargs: Any) -> "FakeCompletedProcess":
    """Stands in for `subprocess.run([<ampep>, "predict", "-m", ..., "-i", ..., "-o", ...])`:
    writes the real output shape to the `-o` FILE (not a directory, unlike AMPidentifier/macrel)
    the real call would have been given."""
    out_tsv = cmd[cmd.index("-o") + 1]
    with open(out_tsv, "w", encoding="utf-8") as fh:
        fh.write(_AMPEPPY_REAL_OUTPUT)

    class FakeCompletedProcess:
        returncode = 0

    return FakeCompletedProcess()


class TestAmPEPpyPredictor:
    """amPEPpy is benchmark-only (`sample.benchmark._benchmark_only_predictors`), not part of the
    live `EXTERNAL_PREDICTORS` -- a fifth antibacterial comparator alongside Macrel, AMPlify and
    AMPidentifier, never part of the live annotator's design. Its console-script (`self._command`)
    carries an ABSOLUTE shebang baked in at pip-install time (confirmed by reading the installed
    script directly on a real host), so it is invoked directly, the same way
    AMPidentifierPredictor's is -- no AMPlify-style derived python+script indirection needed.

    Verified directly against the real installed tool and its bundled pretrained model
    (2026-09-04): magainin-2 (`GIGKFLHSAKKFGKAFVGEIMNS`) scores `probability_AMP=0.959375`, called
    `AMP` -- the exact numbers `_AMPEPPY_REAL_OUTPUT` above reproduces. Unlike
    macrel/AMPlify/AMPidentifier, amPEPpy never crashed on any degenerate input tried directly
    against its real CLI (a single residue, a non-standard residue, a trailing "*", even a fully
    empty sequence) -- so, unlike those tools, there is no `_is_ampeppy_valid`-style input guard to
    test here; empty sequences are still excluded in `_run`, just because scoring "nothing" is not
    a prediction worth keeping.
    """

    @staticmethod
    def _model_path(tmp_path: Any) -> Any:
        """A stand-in for the pretrained model file `_run` merely checks the existence of --
        parsing its content is scikit-learn's job, not this project's, so a stub is enough."""
        model = tmp_path / "amPEP.model"
        model.write_bytes(b"stub")
        return model

    def test_happy_path_parses_real_output_into_scores_and_calls(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp, "AMPEPPY_MODEL_PATH", str(self._model_path(tmp_path)))
        monkeypatch.setattr(pp.subprocess, "run", _fake_ampeppy_run)

        predictor = pp.AmPEPpyPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampep")]
        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])

        assert results[0]["ampeppy_amp_probability"] == pytest.approx(0.959375)
        assert results[0]["ampeppy_is_amp"] == 1.0
        assert results[1]["ampeppy_amp_probability"] == pytest.approx(0.6265625)
        # amPEPpy's real, verified behavior on this negative-control sequence is to call it AMP
        # too -- the wrapper must report the tool's actual call, not a "corrected" one.
        assert results[1]["ampeppy_is_amp"] == 1.0

    def test_passes_predict_subcommand_model_path_and_cwd(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """`_run` must invoke `ampep predict -m <AMPEPPY_MODEL_PATH> -i ... -o ...` through
        `self._command` (the environment-resolved absolute path from `available()`'s probing,
        never a bare "ampep" that would resolve through the caller's own PATH), and must run with
        an explicit `cwd`: amPEPpy's own feature-scoring step unconditionally writes a
        `<today>_scored_features.csv` side-effect file into the current working directory
        (confirmed against a real run, undocumented in its README, and unrelated to `-o`) -- without
        `cwd` pinned to a throwaway temp directory that file lands wherever this process happens to
        be running from."""
        import sample.peptide_predictors as pp

        model_path = str(self._model_path(tmp_path))
        monkeypatch.setattr(pp, "AMPEPPY_MODEL_PATH", model_path)

        captured: Dict[str, Any] = {}

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _fake_ampeppy_run(cmd, **kwargs)

        monkeypatch.setattr(pp.subprocess, "run", capturing_run)
        predictor = pp.AmPEPpyPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampep")]
        predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        cmd = captured["cmd"]
        assert cmd[0] == str(tmp_path / "bin" / "ampep")
        assert cmd[1] == "predict"
        assert cmd[cmd.index("-m") + 1] == model_path
        assert captured["kwargs"].get("cwd"), \
            "must run with an explicit cwd to contain amPEPpy's own scored_features.csv side effect"

    def test_missing_environment_returns_empty_results_without_calling_subprocess(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The tool not being installed (`available()` false, `_command` never resolved) must
        return empty dicts for every sequence and never crash -- and never even attempt the
        subprocess call, exactly as `ExternalPredictor.predict` already guarantees for every
        adapter."""
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when the tool is not installed")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        predictor = pp.AmPEPpyPredictor()
        monkeypatch.setattr(predictor, "available", lambda: False)

        assert predictor.predict(["GIGKFLHSAKKFGKAFVGEIMNS", "DDDD"]) == [{}, {}]

    def test_missing_pretrained_model_returns_empty_results_without_calling_subprocess(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """A `sample-ampeppy` conda environment can exist (so `available()` is True and
        `self._command` resolves) without `make env-ampeppy` having also cloned amPEPpy's repo for
        its pretrained model -- `_run` must check for that file itself and skip cleanly rather than
        handing `ampep predict` a `-m` path that does not exist."""
        import sample.peptide_predictors as pp

        def boom(*a: Any, **k: Any) -> Any:
            raise AssertionError("must not be called when the pretrained model is missing")

        monkeypatch.setattr(pp.subprocess, "run", boom)
        monkeypatch.setattr(pp, "AMPEPPY_MODEL_PATH", str(tmp_path / "nonexistent" / "amPEP.model"))
        predictor = pp.AmPEPpyPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampep")]

        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_subprocess_failure_returns_empty_results(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp, "AMPEPPY_MODEL_PATH", str(self._model_path(tmp_path)))

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("ampeppy crashed")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        predictor = pp.AmPEPpyPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampep")]
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_a_calledprocesserror_surfaces_its_real_stderr(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same regression class as every other adapter's own version of this bug: a bare
        `except Exception as e: logger.warning(f"... failed: {e}")` loses the real captured stderr
        and logs only the generic exit-status message. Must go through `_log_subprocess_failure`,
        which surfaces it."""
        import subprocess

        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp, "AMPEPPY_MODEL_PATH", str(self._model_path(tmp_path)))

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd, output=b"",
                stderr=b"sklearn.exceptions.InconsistentVersionWarning: boom\n")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)
        messages: List[str] = []
        monkeypatch.setattr(pp.logger, "warning", lambda msg: messages.append(msg))
        predictor = pp.AmPEPpyPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampep")]

        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]
        assert messages
        assert "InconsistentVersionWarning" in messages[0], \
            "the captured stderr must reach the log, not just the generic exit-status message"

    def test_missing_output_file_returns_empty_results(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp, "AMPEPPY_MODEL_PATH", str(self._model_path(tmp_path)))

        def no_output_run(cmd: List[str], **kwargs: Any) -> Any:
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", no_output_run)
        predictor = pp.AmPEPpyPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampep")]
        assert predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"]) == [{}]

    def test_an_empty_sequence_in_a_mixed_batch_is_excluded_but_does_not_block_the_rest(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp, "AMPEPPY_MODEL_PATH", str(self._model_path(tmp_path)))

        written_fasta: List[str] = []

        def capturing_run(cmd: List[str], **kwargs: Any) -> Any:
            with open(cmd[cmd.index("-i") + 1], encoding="utf-8") as fh:
                written_fasta.append(fh.read())
            out_tsv = cmd[cmd.index("-o") + 1]
            with open(out_tsv, "w", encoding="utf-8") as fh:
                fh.write("probability_nonAMP\tprobability_AMP\tpredicted\tseq_id\n"
                         "0.040625\t0.959375\tAMP\ts0\n"
                         "0.3734375\t0.6265625\tAMP\ts2\n")
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", capturing_run)
        predictor = pp.AmPEPpyPredictor()
        predictor._command = [str(tmp_path / "bin" / "ampep")]

        # The empty string at position 1 must never reach the FASTA, but must not block the real
        # sequences on either side of it from being submitted and scored.
        results = predictor._run(
            ["GIGKFLHSAKKFGKAFVGEIMNS", "", "DDDDEEEEDDDDEEEEDDDDEEEEDDDD"])

        assert written_fasta
        assert ">s1" not in written_fasta[0]
        assert ">s0" in written_fasta[0] and ">s2" in written_fasta[0]
        assert results[0]["ampeppy_is_amp"] == 1.0
        assert results[1] == {}
        assert results[2]["ampeppy_is_amp"] == 1.0

    def test_available_in_its_own_real_environment(self) -> None:
        """Not mocked: `sample-ampeppy` is a real, built environment on this host (see
        `environments/sample-ampeppy.yml`), and `ampep` genuinely runs in it."""
        import sample.peptide_predictors as pp

        assert pp.AmPEPpyPredictor().available() is True


class TestMacrelInputGuard:
    """Regression: one junk row cost an entire 23,368-peptide batch.

    macrel's own `normalize_seq` strips a leading "M" and then reads `seq[-1]`, so a sequence that
    is exactly "M" raises IndexError and the whole run exits non-zero. The real consensus contained
    one (SAMPLE0000015473), and step 22 lost every macrel prediction because of it.
    """

    def test_only_the_genuinely_crashing_inputs_are_excluded(self) -> None:
        import sample.peptide_predictors as pp

        # These crash macrel outright.
        assert pp._macrel_safe("M") is False
        assert pp._macrel_safe("") is False
        assert pp._macrel_safe("   ") is False
        # These do NOT -- macrel strips at most ONE leading M, and removes a trailing "*" only
        # after the index that crashes. Excluding them would silently lose real predictions,
        # which is the failure this guard exists to prevent.
        assert pp._macrel_safe("MM") is True
        assert pp._macrel_safe("*") is True
        assert pp._macrel_safe("K") is True
        assert pp._macrel_safe("MGIGKFLHSAKKF") is True

    def test_a_crashing_sequence_is_kept_out_of_the_fasta(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        written = {}

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            fasta = cmd[cmd.index("--fasta") + 1]
            with open(fasta, encoding="utf-8") as handle:
                written["fasta"] = handle.read()

            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.MacrelPredictor()
        predictor._command = ["macrel"]
        predictor._run(["GIGKFLHSAKKF", "M", "KWKLFKKIEK"])

        # The batch still goes out; only the crashing record is absent.
        assert "GIGKFLHSAKKF" in written["fasta"]
        assert "KWKLFKKIEK" in written["fasta"]
        assert "\nM\n" not in written["fasta"]

    def test_the_batch_still_returns_a_slot_per_input_sequence(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Filtering the FASTA must not shift results onto the wrong peptides: the caller indexes
        # by position, so every input still needs its own slot.
        import sample.peptide_predictors as pp

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.MacrelPredictor()
        predictor._command = ["macrel"]
        results = predictor._run(["GIGKFLHSAKKF", "M", "KWKLFKKIEK"])
        assert len(results) == 3


class TestHemoPI2InputGuard:
    """Regression: hemopi2's own CTC (Conjoint Triad) feature step needs at least one 3-mer to
    count, checks for that itself (`Max_MM == 0` in `hemopi2/Model/composition_calculate.py`), and
    on a too-short sequence prints an error and returns WITHOUT writing its output file -- which
    then crashes an unrelated concatenation step immediately after it (an uncaught `pd.read_csv`
    under `os.system`, so that failure is swallowed too), surfacing many steps later as a
    `FileNotFoundError` on a completely different file. Confirmed on a real corpus entry ('EL',
    row 2078) in a live production run, 2026-08-26."""

    def test_only_sequences_too_short_for_a_single_triad_are_excluded(self) -> None:
        import sample.peptide_predictors as pp

        # These crash hemopi2's CTC step outright: fewer than 3 residues, so zero possible triads.
        assert pp._hemopi2_safe("") is False
        assert pp._hemopi2_safe("E") is False
        assert pp._hemopi2_safe("EL") is False
        # Exactly 3 residues is the real boundary -- one triad, CTC succeeds.
        assert pp._hemopi2_safe("ELK") is True
        assert pp._hemopi2_safe("GIGKFLHSAKKF") is True

    def test_a_too_short_sequence_is_kept_out_of_the_fasta(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        written = {}

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            with open(os.path.join(kwargs["cwd"], "in.faa"), encoding="utf-8") as handle:
                written["fasta"] = handle.read()

            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.HemoPI2Predictor()
        predictor._command = ["hemopi2_classification"]
        predictor._run(["GIGKFLHSAKKF", "EL", "KWKLFKKIEK"])

        # The batch still goes out; only the crashing record is absent.
        assert "GIGKFLHSAKKF" in written["fasta"]
        assert "KWKLFKKIEK" in written["fasta"]
        assert "\nEL\n" not in written["fasta"]

    def test_the_batch_still_returns_a_slot_per_input_sequence(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            class FakeCompletedProcess:
                returncode = 0
            return FakeCompletedProcess()

        monkeypatch.setattr(pp.subprocess, "run", fake_run)
        predictor = pp.HemoPI2Predictor()
        predictor._command = ["hemopi2_classification"]
        results = predictor._run(["GIGKFLHSAKKF", "EL", "KWKLFKKIEK"])
        assert len(results) == 3


class TestPredictorBatching:
    """Regression: the whole corpus went to each tool in ONE subprocess call, so AMPlify hit its
    3600s timeout on a 23,368-peptide run and every AMPlify prediction for the corpus was lost."""

    def test_a_large_corpus_is_split_into_bounded_calls(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        calls: List[int] = []

        class CountingPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="counter", executable="counter",
                                 description="test", columns=["counter_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                calls.append(len(sequences))
                return [{"counter_score": 1.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [CountingPredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 100)

        df = pd.DataFrame({"sequence": ["GIGKFLHSAKKF"] * 250})
        out = pp.enrich_with_predictors(df)

        assert calls == [100, 100, 50], "the corpus must be split into bounded calls"
        assert out["counter_score"].notna().sum() == 250, "every peptide still gets its result"

    def test_one_failing_batch_is_recovered_by_the_per_peptide_fallback(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A whole-chunk failure that was really a transient hiccup (not a specific bad peptide)
        is now fully recovered: the fallback retries every sequence in that chunk individually,
        and each one succeeds on its own call. This is the payoff of the fallback over the old
        "batch fails, whole batch is lost" behavior."""
        import pandas as pd

        import sample.peptide_predictors as pp

        class FlakyPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="flaky", executable="flaky",
                                 description="test", columns=["flaky_score"])
                self.seen = 0

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                self.seen += 1
                if self.seen == 2:            # only the second CALL fails, a one-off hiccup
                    return [{} for _ in sequences]
                return [{"flaky_score": 1.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [FlakyPredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 100)

        df = pd.DataFrame({"sequence": ["GIGKFLHSAKKF"] * 250})
        out = pp.enrich_with_predictors(df)

        assert out["flaky_score"].notna().sum() == 250, \
            "the fallback's per-peptide retries must recover the whole failed chunk"

    def test_a_genuinely_bad_peptide_in_a_batch_is_isolated_by_the_fallback(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The real-world case this exists for: one poisoned sequence kills the whole batch call
        it is in, but only that one sequence should end up with no prediction -- not the other 99
        peptides that happened to share its batch."""
        import pandas as pd

        import sample.peptide_predictors as pp

        class PoisonedBatchPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="poison", executable="poison",
                                 description="test", columns=["poison_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                if "WWWWWWWWWW" in sequences:
                    return [{} for _ in sequences]  # the whole call dies, batched or not
                return [{"poison_score": 1.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [PoisonedBatchPredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 100)

        seqs = ["GIGKFLHSAKKF"] * 99 + ["WWWWWWWWWW"]
        out = pp.enrich_with_predictors(pd.DataFrame({"sequence": seqs}))

        assert out["poison_score"].notna().sum() == 99, \
            "only the one poisoned peptide should end up without a result"
        assert out.loc[out["sequence"] == "WWWWWWWWWW", "poison_score"].isna().all()

    def test_results_stay_aligned_with_their_input_rows(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Batching must not shift a prediction onto the wrong peptide.
        import pandas as pd

        import sample.peptide_predictors as pp

        class IndexingPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="idx", executable="idx",
                                 description="test", columns=["idx_len"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                return [{"idx_len": float(len(s))} for s in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [IndexingPredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 10)

        seqs = ["A" * (i % 7 + 3) for i in range(55)]
        out = pp.enrich_with_predictors(pd.DataFrame({"sequence": seqs}))
        assert out["idx_len"].tolist() == [float(len(s)) for s in seqs]

    def test_default_batch_size_is_100_with_per_peptide_fallback(self) -> None:
        """The default was briefly 1 (one call per peptide), after hemopi2 failed on every batch
        of a 53,614-peptide run at size 2000 with no informative error. That traded away almost
        all throughput: most of a tool's per-call time is fixed model-load overhead, and paying it
        53,614 times instead of amortising it over batches made AMPlify alone run at ~31s/peptide
        (real run, 2027-8-21/24) -- weeks for the corpus. The fallback in
        `_predict_chunk_with_fallback` gets the isolation back without the cost: batches of 100,
        retried one-at-a-time ONLY when the whole batch fails."""
        import sample.peptide_predictors as pp

        assert pp.PREDICTOR_BATCH_SIZE == 100

    def test_batch_size_one_isolates_and_names_the_failing_peptide(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        class OneBadPeptidePredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="picky", executable="picky",
                                 description="test", columns=["picky_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                assert len(sequences) == 1, "batch size 1 must mean one peptide per call"
                if sequences[0] == "WWWWWWWWWW":
                    return [{}]
                return [{"picky_score": 1.0}]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [OneBadPeptidePredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 1)

        messages: List[str] = []
        monkeypatch.setattr(pp.logger, "warning", lambda msg: messages.append(msg))

        seqs = ["GIGKFLHSAKKF", "WWWWWWWWWW", "DDDDEEEEDDDD"]
        out = pp.enrich_with_predictors(pd.DataFrame({"sequence": seqs}))

        assert out["picky_score"].notna().sum() == 2, \
            "only the one poisoned peptide should be missing a result"
        assert out.loc[out["sequence"] == "WWWWWWWWWW", "picky_score"].isna().all()
        assert any("WWWWWWWWWW" in m for m in messages), \
            "the failing peptide's own sequence must appear in the log, not just a count"


class TestHemoPI2ErrorReporting:
    """Regression: a full production run failed on every hemopi2 batch for hours with nothing but
    `hemopi2 failed: Command '[...]' returned non-zero exit status 1.` in the log -- `str(e)` on a
    `subprocess.CalledProcessError` never includes the captured stderr/stdout even though
    `capture_output=True` collected it, so the real reason was sitting in `e.stderr` the whole
    time and simply never got logged. This tests that the actual failure text now surfaces."""

    def test_stderr_is_surfaced_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import subprocess

        import sample.peptide_predictors as pp

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(
                returncode=1, cmd=cmd,
                output=b"", stderr=b"Traceback: ValueError: bad feature matrix shape\n")

        monkeypatch.setattr(pp.subprocess, "run", failing_run)

        predictor = pp.HemoPI2Predictor()
        predictor._command = ["hemopi2_classification"]

        messages: List[str] = []
        monkeypatch.setattr(pp.logger, "warning", lambda msg: messages.append(msg))

        results = predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        assert results == [{}], "a failed call must still return an empty result, not raise"
        assert messages, "the failure must be logged"
        assert "bad feature matrix shape" in messages[0], \
            "the captured stderr must reach the log, not just the generic exit-status message"

    def test_negative_returncode_is_flagged_as_a_likely_kill(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A negative returncode means the process died to a signal -- SIGKILL/-9 most often
        means the OOM killer -- which is exactly the "silent failure, no traceback" shape seen in
        the field. Worth calling out explicitly rather than just printing "-9"."""
        import subprocess

        import sample.peptide_predictors as pp

        def killed_run(cmd: List[str], **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(returncode=-9, cmd=cmd, output=b"", stderr=b"")

        monkeypatch.setattr(pp.subprocess, "run", killed_run)

        predictor = pp.HemoPI2Predictor()
        predictor._command = ["hemopi2_classification"]

        messages: List[str] = []
        monkeypatch.setattr(pp.logger, "warning", lambda msg: messages.append(msg))

        predictor._run(["GIGKFLHSAKKFGKAFVGEIMNS"])

        assert messages
        assert "killed by signal 9" in messages[0]
        assert "OOM" in messages[0]


class TestPredictorWorkers:
    """SAMPLE_PREDICTOR_WORKERS/PREDICTOR_WORKERS: concurrent batches, opt-in (default 1 = the
    exact original sequential behavior). Real motivation: each batch is an independent subprocess
    call, in principle trivially parallelizable across idle cores. But naive concurrency alone is
    NOT safe here -- a real test on a 24-core host (2026-08-25) found that two concurrent AMPlify
    (TensorFlow) subprocess calls pushed load average past 50: each call's own math library claims
    every visible core with no awareness that siblings are about to do the same. That is what
    `_capped_thread_env` exists to prevent (see its own docstring for the measured numbers)."""

    def test_default_is_sequential(self) -> None:
        import sample.peptide_predictors as pp

        assert pp.PREDICTOR_WORKERS == 1

    def test_results_stay_correctly_aligned_when_batches_finish_out_of_order(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole risk of concurrency: a later chunk's subprocess call finishing before an
        earlier one's must not scramble which prediction lands on which row."""
        import time
        import pandas as pd

        import sample.peptide_predictors as pp

        class SlowFirstBatchPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="slow_first", executable="slow_first", description="test",
                                 columns=["slow_first_len"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                if sequences and sequences[0].startswith("A"):
                    time.sleep(0.2)  # the first chunk (all "A..." seqs) finishes LAST
                return [{"slow_first_len": float(len(s))} for s in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [SlowFirstBatchPredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 10)
        monkeypatch.setattr(pp, "PREDICTOR_WORKERS", 4)

        seqs = [f"A{'A' * (i % 5)}" for i in range(10)] + [f"C{'C' * (i % 5)}" for i in range(30)]
        out = pp.enrich_with_predictors(pd.DataFrame({"sequence": seqs}))

        assert out["slow_first_len"].tolist() == [float(len(s)) for s in seqs]

    def test_concurrent_run_still_recovers_all_peptides_after_a_failure(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        class PoisonedBatchPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="poison_conc", executable="poison_conc", description="test",
                                 columns=["poison_conc_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                if "WWWWWWWWWW" in sequences:
                    return [{} for _ in sequences]
                return [{"poison_conc_score": 1.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [PoisonedBatchPredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 10)
        monkeypatch.setattr(pp, "PREDICTOR_WORKERS", 4)

        seqs = ["GIGKFLHSAKKF"] * 35 + ["WWWWWWWWWW"] + ["GIGKFLHSAKKF"] * 4
        out = pp.enrich_with_predictors(pd.DataFrame({"sequence": seqs}))

        assert out["poison_conc_score"].notna().sum() == 39
        assert out.loc[out["sequence"] == "WWWWWWWWWW", "poison_conc_score"].isna().all()


class TestCappedThreadEnv:
    """Regression for the real oversubscription incident: without this, N concurrent predictor
    subprocesses each try to claim every CPU core, since none of them know N-1 siblings are about
    to do the same thing."""

    def test_is_a_no_op_at_worker_count_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setenv("OMP_NUM_THREADS", "16")
        with pp._capped_thread_env(1):
            assert os.environ["OMP_NUM_THREADS"] == "16"  # untouched

    def test_caps_thread_env_vars_proportionally_to_worker_count(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp.os, "cpu_count", lambda: 24)
        with pp._capped_thread_env(4):
            assert os.environ["OMP_NUM_THREADS"] == "6"
            assert os.environ["OPENBLAS_NUM_THREADS"] == "6"
            assert os.environ["TF_NUM_INTRAOP_THREADS"] == "6"

    def test_never_caps_below_one_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setattr(pp.os, "cpu_count", lambda: 4)
        with pp._capped_thread_env(24):  # more workers than cores
            assert os.environ["OMP_NUM_THREADS"] == "1"

    def test_restores_previous_values_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.setenv("OMP_NUM_THREADS", "16")
        with pp._capped_thread_env(4):
            assert os.environ["OMP_NUM_THREADS"] != "16"
        assert os.environ["OMP_NUM_THREADS"] == "16"

    def test_removes_the_var_again_if_it_was_never_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.peptide_predictors as pp

        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        with pp._capped_thread_env(4):
            assert "OMP_NUM_THREADS" in os.environ
        assert "OMP_NUM_THREADS" not in os.environ

    def test_a_real_concurrent_dispatch_caps_the_env_for_the_whole_run(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The actual integration point: `_run_predictor`'s concurrent path must be wrapped in
        `_capped_thread_env`, not just have the helper exist unused."""
        import pandas as pd

        import sample.peptide_predictors as pp

        seen_during_call: List[Optional[str]] = []

        class ObservingPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="observer", executable="observer", description="test",
                                 columns=["observer_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                seen_during_call.append(os.environ.get("OMP_NUM_THREADS"))
                return [{"observer_score": 1.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [ObservingPredictor()])
        monkeypatch.setattr(pp, "PREDICTOR_BATCH_SIZE", 5)
        monkeypatch.setattr(pp, "PREDICTOR_WORKERS", 3)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

        pp.enrich_with_predictors(pd.DataFrame({"sequence": ["GIGKFLHSAKKF"] * 15}))

        assert seen_during_call, "the predictor must actually have been called"
        assert all(v is not None for v in seen_during_call), \
            "every concurrent call must see a capped OMP_NUM_THREADS, not the unset default"
        assert "OMP_NUM_THREADS" not in os.environ, "must not leak the cap past the run"


class TestEnrichOnePredictor:
    """Regression: AMPlify failed on ~7% of a 53,614-peptide corpus midway through a run that also
    computes descriptors and four other predictors, and there was no way to redo just AMPlify's
    gap without redoing everything -- 30+ hours of already-completed work. `enrich_one_predictor`
    is the fix: one tool, over an already-enriched table, touching only its own columns."""

    def test_unknown_predictor_name_raises(self) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        df = pd.DataFrame({"sequence": ["GIGKFLHSAKKF"]})
        with pytest.raises(ValueError, match="Unknown predictor"):
            pp.enrich_one_predictor(df, "not_a_real_predictor")

    def test_unavailable_predictor_leaves_columns_untouched_but_present(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        class UnavailablePredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="ghost", executable="ghost",
                                 description="test", columns=["ghost_score"])

            def available(self) -> bool:
                return False

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [UnavailablePredictor()])
        df = pd.DataFrame({"sequence": ["GIGKFLHSAKKF"]})

        out = pp.enrich_one_predictor(df, "ghost")

        assert "ghost_score" in out.columns
        assert out["ghost_score"].isna().all()

    def test_other_columns_are_carried_through_completely_unchanged(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole point: descriptors and every OTHER predictor's results must survive a
        single-predictor re-run untouched, not just the sequence column."""
        import pandas as pd

        import sample.peptide_predictors as pp

        class OkPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="ok", executable="ok", description="test",
                                 columns=["ok_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                return [{"ok_score": 1.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [OkPredictor()])
        df = pd.DataFrame({
            "sequence": ["GIGKFLHSAKKF", "DDDDEEEEDDDD"],
            "pep_mw": [1234.5, 987.6],                       # a descriptor column
            "amplify_score": [0.9, 0.1],                     # another predictor's results
        })

        out = pp.enrich_one_predictor(df, "ok")

        assert out["pep_mw"].tolist() == [1234.5, 987.6]
        assert out["amplify_score"].tolist() == [0.9, 0.1]
        assert out["ok_score"].tolist() == [1.0, 1.0]

    def test_only_missing_recomputes_just_the_gap_by_default(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        calls: List[List[str]] = []

        class GapFillPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="fill", executable="fill", description="test",
                                 columns=["fill_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                calls.append(list(sequences))
                return [{"fill_score": 9.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [GapFillPredictor()])
        df = pd.DataFrame({
            "sequence": ["GIGKFLHSAKKF", "DDDDEEEEDDDD", "AAAAAAAAAAAA"],
            "fill_score": [0.5, float("nan"), 0.7],  # only the middle row is missing a result
        })

        out = pp.enrich_one_predictor(df, "fill")

        assert calls == [["DDDDEEEEDDDD"]], "only the missing row should have been sent to the tool"
        assert out["fill_score"].tolist() == [0.5, 9.0, 0.7], \
            "existing values for already-scored rows must be preserved exactly"

    def test_all_forces_a_full_recompute_even_where_a_value_exists(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        class OverwritePredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="overwrite", executable="overwrite", description="test",
                                 columns=["overwrite_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                return [{"overwrite_score": 42.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [OverwritePredictor()])
        df = pd.DataFrame({"sequence": ["GIGKFLHSAKKF"], "overwrite_score": [0.1]})

        out = pp.enrich_one_predictor(df, "overwrite", only_missing=False)

        assert out["overwrite_score"].tolist() == [42.0]

    def test_no_missing_rows_is_a_clean_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import pandas as pd

        import sample.peptide_predictors as pp

        calls: List[Any] = []

        class ShouldNotRunPredictor(pp.ExternalPredictor):
            def __init__(self) -> None:
                super().__init__(name="unused", executable="unused", description="test",
                                 columns=["unused_score"])

            def available(self) -> bool:
                return True

            def _run(self, sequences: Any) -> List[Dict[str, float]]:
                calls.append(sequences)
                return [{"unused_score": 1.0} for _ in sequences]

        monkeypatch.setattr(pp, "EXTERNAL_PREDICTORS", [ShouldNotRunPredictor()])
        df = pd.DataFrame({"sequence": ["GIGKFLHSAKKF"], "unused_score": [0.5]})

        out = pp.enrich_one_predictor(df, "unused")

        assert not calls, "nothing was missing, so the tool must not be invoked at all"
        assert out["unused_score"].tolist() == [0.5]
