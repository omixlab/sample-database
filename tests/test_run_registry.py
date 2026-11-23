"""
Resume semantics and the per-run progress table (`sample.run_registry`).

The behaviour under test is what an operator relies on after a 52-step run dies at step 30: the
next `make` must continue at 30, must not redo 1-29, and must be able to say which run that was.
Each test here is one of the ways that can go wrong.
"""
import os
import subprocess
import sys
import time

import pytest

from sample import run_registry as R


@pytest.fixture()
def db(tmp_path):
    """A registry of its own per test; the module's default path is never touched."""
    return str(tmp_path / "runs.sqlite")


class TestRunIdentity:
    """A run has one identifier, and the stats are keyed on it."""

    def test_a_run_id_is_unique(self):
        assert len({R.new_run_id() for _ in range(50)}) == 50, (
            "the random suffix is what separates two runs started in the same second")

    def test_a_run_id_sorts_chronologically(self, monkeypatch):
        """
        The id carries a UTC timestamp prefix so `pipeline-runs` and an `ls`-style listing are in
        run order without parsing anything.

        The clock is faked rather than slept through: the prefix has one-second resolution, so two
        ids made in the same second differ only by their random suffix and their relative order is
        arbitrary. An earlier version of this test slept 10ms and passed by luck.
        """
        ids = []
        for stamp in ("2026-10-21 10:00:00", "2026-10-21 10:00:01", "2026-10-22 09:00:00"):
            fake = time.strptime(stamp, "%Y-%m-%d %H:%M:%S")
            monkeypatch.setattr(R.time, "gmtime", lambda *_a, _f=fake: _f)
            ids.append(R.new_run_id())
        assert ids == sorted(ids), f"ids must sort in creation order, got {ids}"
        assert [i.split("-")[0] for i in ids] == ["20261021T100000Z", "20261021T100001Z",
                                                  "20261022T090000Z"]

    def test_the_first_resolve_creates_a_run_and_the_next_adopts_it(self, db):
        first, adopted_first = R.resolve_run(db_path=db)
        second, adopted_second = R.resolve_run(db_path=db)
        assert adopted_first is False
        assert adopted_second is True
        assert first == second, (
            "run_step calls resolve once per step; all 52 must land under one run id")

    def test_last_run_identifies_the_run_without_being_told_which(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        assert R.last_run(db_path=db)["run_id"] == run_id

    def test_a_run_records_who_and_where_it_ran(self, db):
        run_id, _ = R.resolve_run(db_path=db, notes="nightly")
        row = R.last_run(db_path=db)
        assert row["host"] and row["status"] == R.STATUS_RUNNING
        assert row["notes"] == "nightly"
        assert row["run_id"] == run_id


class TestResume:
    """What gets skipped, and what does not."""

    def test_a_completed_step_is_skipped(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 7, "Merge", db_path=db)
        R.step_finished(run_id, 7, R.STATUS_SUCCESS, exit_code=0, db_path=db)
        assert R.should_skip(run_id, 7, db_path=db) is True

    def test_a_step_that_never_ran_is_not_skipped(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        assert R.should_skip(run_id, 7, db_path=db) is False

    @pytest.mark.parametrize("status", [R.STATUS_FAILURE, R.STATUS_INTERRUPTED,
                                        R.STATUS_SKIPPED, R.STATUS_RUNNING])
    def test_only_success_earns_a_skip(self, db, status):
        """
        A FAILURE or INTERRUPTED step has no trustworthy artifact and a SKIPPED one never ran, so
        treating any of them as done would silently build the rest of the pipeline on a gap.
        """
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 9, "Refined extraction", db_path=db)
        R.step_finished(run_id, 9, status, db_path=db)
        assert R.should_skip(run_id, 9, db_path=db) is False

    def test_from_scratch_skips_nothing(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 7, "Merge", db_path=db)
        R.step_finished(run_id, 7, R.STATUS_SUCCESS, db_path=db)
        assert R.should_skip(run_id, 7, from_scratch=True, db_path=db) is False

    def test_from_scratch_starts_a_new_run_and_closes_the_old_one(self, db):
        old, _ = R.resolve_run(db_path=db)
        R.step_started(old, 1, "Download", db_path=db)
        R.step_finished(old, 1, R.STATUS_SUCCESS, db_path=db)
        new, adopted = R.resolve_run(from_scratch=True, db_path=db)
        assert adopted is False and new != old
        assert R.should_skip(new, 1, db_path=db) is False, "a fresh run has completed nothing"
        rows = {r["run_id"]: r["status"] for r in R.run_rows(db_path=db)}
        assert rows[old] == R.STATUS_INTERRUPTED, "the abandoned run must not stay RUNNING"

    def test_a_concurrent_step_is_not_reported_as_interrupted(self, db):
        """
        A long step and a quick manual one alongside it is ordinary operation.

        Observed for real: `make step_11_polish_extraction_records` run by hand while step 9 was
        genuinely mid-flight -- 1h44m elapsed, five worker subprocesses alive -- recorded step 9
        as INTERRUPTED and bumped `resumed_count`, purely because resolve closed every RUNNING
        row it found. Only the row for the step about to run may be closed.
        """
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 9, "Refined extraction", db_path=db)      # still running
        again, adopted = R.resolve_run(db_path=db, step=11)              # another terminal
        assert again == run_id and adopted is True
        statuses = {r["step"]: r["status"] for r in R.step_rows(run_id, db_path=db)}
        assert statuses[9] == R.STATUS_RUNNING, "a live step must not be closed by another step"
        assert R.last_run(db_path=db)["resumed_count"] == 0

    def test_without_a_step_nothing_is_closed(self, db):
        """
        `pipeline-finish` resolves without naming a step; it must not reinterpret running work.
        """
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 9, "Refined extraction", db_path=db)
        R.resolve_run(db_path=db)
        assert R.step_rows(run_id, db_path=db)[0]["status"] == R.STATUS_RUNNING

    def test_a_step_killed_mid_flight_is_marked_interrupted_and_rerun(self, db):
        """
        The Ctrl-C case. The step row is left RUNNING with no end; the next invocation has to
        notice, record it as INTERRUPTED for the audit trail, and re-run it.
        """
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 1, "Download", db_path=db)
        R.step_finished(run_id, 1, R.STATUS_SUCCESS, db_path=db)
        R.step_started(run_id, 2, "Scopus", db_path=db)          # killed here

        again, adopted = R.resolve_run(db_path=db, step=2)
        assert adopted is True and again == run_id
        statuses = {r["step"]: r["status"] for r in R.step_rows(run_id, db_path=db)}
        assert statuses[2] == R.STATUS_INTERRUPTED
        assert R.should_skip(run_id, 1, db_path=db) is True, "step 1 finished; do not redo it"
        assert R.should_skip(run_id, 2, db_path=db) is False, "step 2 died; re-run it"

    def test_resolve_does_not_count_the_steps_of_one_make_as_resumes(self, db):
        """
        resolve runs once per step, so it must distinguish "second step of this make" from
        "second invocation". Only the latter is a resume.
        """
        run_id, _ = R.resolve_run(db_path=db)
        for step in (1, 2, 3):
            R.resolve_run(db_path=db)
            R.step_started(run_id, step, f"Step {step}", db_path=db)
            R.step_finished(run_id, step, R.STATUS_SUCCESS, db_path=db)
        assert R.last_run(db_path=db)["resumed_count"] == 0

    def test_a_real_resume_is_counted(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 1, "Download", db_path=db)
        R.step_finished(run_id, 1, R.STATUS_SUCCESS, db_path=db)
        R.step_started(run_id, 2, "Scopus", db_path=db)      # interrupted
        R.resolve_run(db_path=db, step=2)                    # next invocation
        assert R.last_run(db_path=db)["resumed_count"] == 1

    def test_a_resume_during_the_very_first_step_is_counted(self, db):
        """
        The case the original condition got wrong. `make step_09` alone, killed during step 9:
        nothing has reached a terminal status, so requiring a closed row recorded this genuine
        resume as `resumed_count = 0`. Seen live on run 20261021T162333Z-14f7f9.
        """
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 9, "Refined extraction", db_path=db)   # killed here
        again, adopted = R.resolve_run(db_path=db, step=9)
        assert (again, adopted) == (run_id, True)
        assert R.last_run(db_path=db)["resumed_count"] == 1
        assert R.step_rows(run_id, db_path=db)[0]["status"] == R.STATUS_INTERRUPTED

    def test_each_further_invocation_increments_the_count(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        for _ in range(3):
            R.step_started(run_id, 9, "Refined extraction", db_path=db)
            R.resolve_run(db_path=db, step=9)
        assert R.last_run(db_path=db)["resumed_count"] == 3
        assert R.step_rows(run_id, db_path=db)[0]["attempts"] == 3


class TestProgressStats:
    """The table itself."""

    def test_a_step_records_its_duration_exit_code_and_attempts(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 5, "Databases", command="sample-cli databases", db_path=db)
        R.step_finished(run_id, 5, R.STATUS_FAILURE, exit_code=2, db_path=db)
        R.step_started(run_id, 5, "Databases", db_path=db)
        R.step_finished(run_id, 5, R.STATUS_SUCCESS, exit_code=0, db_path=db)

        row = R.step_rows(run_id, db_path=db)[0]
        assert row["status"] == R.STATUS_SUCCESS
        assert row["exit_code"] == 0
        assert row["attempts"] == 2, "a retried step is one row with a count, not two rows"
        assert row["duration_seconds"] is not None and row["duration_seconds"] >= 0

    def test_item_counts_are_stored_when_a_command_reports_them(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 9, "Refined extraction", db_path=db)
        R.step_finished(run_id, 9, R.STATUS_SUCCESS, items_total=22050, items_done=11698,
                        items_skipped=9545, items_failed=3, db_path=db)
        row = R.step_rows(run_id, db_path=db)[0]
        assert (row["items_total"], row["items_done"]) == (22050, 11698)
        assert (row["items_skipped"], row["items_failed"]) == (9545, 3)

    def test_a_directory_artifact_is_measured_by_entry_count(self, db, tmp_path):
        out = tmp_path / "pubmed"
        out.mkdir()
        for i in range(3):
            (out / f"{i}.json").write_text("{}")
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 9, "Extract", artifact=str(out), db_path=db)
        R.step_finished(run_id, 9, R.STATUS_SUCCESS, db_path=db)
        row = R.step_rows(run_id, db_path=db)[0]
        assert row["artifact_present"] == 1 and row["artifact_count"] == 3

    def test_a_missing_artifact_is_recorded_as_absent(self, db, tmp_path):
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 9, "Extract", artifact=str(tmp_path / "nope"), db_path=db)
        R.step_finished(run_id, 9, R.STATUS_FAILURE, db_path=db)
        row = R.step_rows(run_id, db_path=db)[0]
        assert row["artifact_present"] == 0 and row["artifact_count"] == 0

    def test_completed_steps_lists_exactly_what_resume_will_skip(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        for step, status in ((1, R.STATUS_SUCCESS), (2, R.STATUS_FAILURE),
                             (3, R.STATUS_SUCCESS), (4, R.STATUS_SKIPPED)):
            R.step_started(run_id, step, f"Step {step}", db_path=db)
            R.step_finished(run_id, step, status, db_path=db)
        assert R.completed_steps(run_id, db_path=db) == {1, 3}

    def test_progress_renders_as_a_table(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        R.step_started(run_id, 1, "Download PubMed", db_path=db)
        R.step_finished(run_id, 1, R.STATUS_SUCCESS, db_path=db)
        text = R.format_progress(run_id, db_path=db)
        assert run_id in text and "Download PubMed" in text and R.STATUS_SUCCESS in text
        assert "1 of 1 recorded step(s) succeeded" in text

    def test_stats_from_two_runs_do_not_mix(self, db):
        first, _ = R.resolve_run(db_path=db)
        R.step_started(first, 1, "Download", db_path=db)
        R.step_finished(first, 1, R.STATUS_SUCCESS, db_path=db)
        second, _ = R.resolve_run(from_scratch=True, db_path=db)
        R.step_started(second, 2, "Scopus", db_path=db)
        R.step_finished(second, 2, R.STATUS_SUCCESS, db_path=db)
        assert [r["step"] for r in R.step_rows(first, db_path=db)] == [1]
        assert [r["step"] for r in R.step_rows(second, db_path=db)] == [2]


class TestFinishRun:
    def test_a_finished_run_is_not_adopted_by_the_next_invocation(self, db):
        run_id, _ = R.resolve_run(db_path=db)
        R.finish_run(run_id, db_path=db)
        nxt, adopted = R.resolve_run(db_path=db)
        assert adopted is False and nxt != run_id, (
            "`make all` closes the run on success; the next make is a new run, not a resume")


class TestTheCommandLineTheMakefileUses:
    """
    run_step shells out to this module between every step, so the CLI contract is load-bearing:
    `resolve` must print a bare id with no trailing newline (a recipe captures it in a shell
    variable), and `should-skip` must signal through the exit status.
    """

    @staticmethod
    def _run(db, *args):
        return subprocess.run([sys.executable, "-m", "sample.run_registry", "--db", db, *args],
                              capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def test_resolve_prints_a_bare_run_id(self, db):
        out = self._run(db, "resolve", "--from-scratch", "0")
        assert out.returncode == 0
        assert out.stdout == out.stdout.strip() and out.stdout, "no newline: $(...) keeps it clean"

    def test_should_skip_signals_through_the_exit_status(self, db):
        run_id = self._run(db, "resolve").stdout
        assert self._run(db, "should-skip", "--run-id", run_id, "--step", "1").returncode == 1
        self._run(db, "begin-step", "--run-id", run_id, "--step", "1", "--name", "x")
        self._run(db, "end-step", "--run-id", run_id, "--step", "1", "--status", "SUCCESS")
        assert self._run(db, "should-skip", "--run-id", run_id, "--step", "1").returncode == 0

    def test_show_falls_back_to_the_last_run(self, db):
        run_id = self._run(db, "resolve").stdout
        self._run(db, "begin-step", "--run-id", run_id, "--step", "3", "--name", "Papers")
        self._run(db, "end-step", "--run-id", run_id, "--step", "3", "--status", "SUCCESS")
        out = self._run(db, "show")
        assert run_id in out.stdout and "Papers" in out.stdout

    def test_show_emits_json_on_request(self, db):
        import json
        run_id = self._run(db, "resolve").stdout
        self._run(db, "begin-step", "--run-id", run_id, "--step", "1", "--name", "x")
        self._run(db, "end-step", "--run-id", run_id, "--step", "1", "--status", "SUCCESS")
        payload = json.loads(self._run(db, "show", "--json").stdout)
        assert payload["run_id"] == run_id and payload["steps"][0]["step"] == 1

    def test_the_registry_runs_without_a_conda_environment(self, db):
        """
        run_step calls this three times per step through the bare system python3. An import of
        pandas, or of anything else from site-packages, would force `conda run` per call and cost
        a minute of startup across 52 steps -- including on `make help`.
        """
        out = self._run(db, "list")
        assert out.returncode == 0, out.stderr
        source = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "sample", "run_registry.py")).read()
        for forbidden in ("import pandas", "import numpy", "from sample.", "import requests"):
            assert forbidden not in source, f"{forbidden} would drag in a conda environment"
