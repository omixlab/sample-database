"""Tests for :mod:`sample.prediction_store` — the background prediction job store."""
import os
import sqlite3
from typing import Any, Dict, List

import pytest

from sample import prediction_store as PS


@pytest.fixture
def db(tmp_path: Any) -> str:
    """
    :param tmp_path: pytest temporary directory.
    :returns: Path to an initialized, empty prediction database.
    """
    path = str(tmp_path / "predictions.db")
    PS.init_db(path)
    return path


@pytest.fixture
def rows() -> List[Dict[str, Any]]:
    """:returns: Two result rows, one scored and one failed."""
    return [
        {"position": 0, "peptide_name": "magainin-2", "sequence": "GIGKFL",
         "probability": 0.87, "predicted_label": "active", "value": None, "error": None},
        {"position": 1, "peptide_name": "bad", "sequence": "ZZZ",
         "probability": None, "predicted_label": None, "value": None,
         "error": "non-standard residue(s): Z"},
    ]


class TestDefaultDbPath:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAMPLE_PREDICTIONS_DB", "/custom/predictions.db")
        assert PS.default_db_path() == "/custom/predictions.db"

    def test_falls_back_to_data_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SAMPLE_PREDICTIONS_DB", raising=False)
        assert PS.default_db_path().endswith(os.path.join("data", "predictions.db"))

    def test_is_not_the_release_database(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The whole point of this module: user jobs must not share a file with the release
        # artifact that step 44 rebuilds and step 52 ships.
        monkeypatch.delenv("SAMPLE_PREDICTIONS_DB", raising=False)
        assert not PS.default_db_path().endswith("sample.db")


class TestInitDb:
    def test_creates_both_tables(self, db: str) -> None:
        with sqlite3.connect(db) as conn:
            names = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"jobs", "predictions"} <= names

    def test_is_idempotent(self, db: str) -> None:
        assert PS.init_db(db) == PS.init_db(db)

    def test_creates_parent_directory(self, tmp_path: Any) -> None:
        nested = str(tmp_path / "a" / "b" / "predictions.db")
        PS.init_db(nested)
        assert os.path.exists(nested)

    def test_enables_wal(self, db: str) -> None:
        # The worker writes while the webserver reads, from separate processes.
        with sqlite3.connect(db) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"

    def test_migrates_a_predictions_table_from_before_in_domain_existed(self, tmp_path: Any) -> None:
        path = str(tmp_path / "old_predictions.db")
        with sqlite3.connect(path) as conn:
            conn.execute(
                "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, "
                "status TEXT NOT NULL, n_peptides INTEGER NOT NULL DEFAULT 0, "
                "n_completed INTEGER NOT NULL DEFAULT 0, submitted_at REAL NOT NULL, "
                "started_at REAL, finished_at REAL, error TEXT, celery_task_id TEXT, meta TEXT)")
            conn.execute(
                "CREATE TABLE predictions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "job_id TEXT NOT NULL, position INTEGER NOT NULL, peptide_name TEXT, "
                "sequence TEXT NOT NULL, probability REAL, predicted_label TEXT, value REAL, "
                "error TEXT)")
            conn.execute(
                "INSERT INTO jobs (job_id, model_id, status, submitted_at) "
                "VALUES ('old-job', 'm/x', 'SUCCESS', 0)")
            conn.execute(
                "INSERT INTO predictions (job_id, position, sequence) VALUES ('old-job', 0, 'AAAA')")

        PS.init_db(path)  # the migration under test

        with sqlite3.connect(path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        assert "in_domain" in columns
        # The pre-existing row survives the migration, with the new column reading as unset.
        preserved = PS.get_predictions("old-job", db_path=path)
        assert len(preserved) == 1
        assert preserved[0]["sequence"] == "AAAA"
        assert preserved[0]["in_domain"] is None


class TestJobLifecycle:
    def test_created_job_starts_pending(self, db: str) -> None:
        PS.create_job("j1", "m/x", 3, db_path=db)
        job = PS.get_job("j1", db_path=db)
        assert job is not None
        assert job["status"] == PS.STATUS_PENDING
        assert job["n_peptides"] == 3
        assert job["n_completed"] == 0

    def test_unknown_job_is_none(self, db: str) -> None:
        assert PS.get_job("nope", db_path=db) is None

    def test_meta_round_trips_as_dict(self, db: str) -> None:
        PS.create_job("j1", "m/x", 1, meta={"k": "v"}, db_path=db)
        assert PS.get_job("j1", db_path=db)["meta"] == {"k": "v"}

    def test_missing_meta_decodes_to_empty_dict(self, db: str) -> None:
        PS.create_job("j1", "m/x", 1, db_path=db)
        assert PS.get_job("j1", db_path=db)["meta"] == {}

    @pytest.mark.parametrize("bad", ["{not json", None])
    def test_corrupt_meta_degrades_to_empty_dict(self, db: str, bad: object) -> None:
        # A job row is still useful when only its annotation is damaged; failing to read the
        # status because the meta column is malformed would lose the result entirely.
        PS.create_job("j1", "m/x", 1, db_path=db)
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE jobs SET meta = ? WHERE job_id = 'j1'", (bad,))
        assert PS.get_job("j1", db_path=db)["meta"] == {}

    def test_attach_task_id(self, db: str) -> None:
        PS.create_job("j1", "m/x", 1, db_path=db)
        PS.attach_task_id("j1", "celery-123", db_path=db)
        assert PS.get_job("j1", db_path=db)["celery_task_id"] == "celery-123"

    def test_mark_running_sets_started_at(self, db: str) -> None:
        PS.create_job("j1", "m/x", 1, db_path=db)
        PS.mark_running("j1", db_path=db)
        job = PS.get_job("j1", db_path=db)
        assert job["status"] == PS.STATUS_RUNNING
        assert job["started_at"] is not None

    def test_mark_finished_success(self, db: str) -> None:
        PS.create_job("j1", "m/x", 1, db_path=db)
        PS.mark_finished("j1", PS.STATUS_SUCCESS, db_path=db)
        job = PS.get_job("j1", db_path=db)
        assert job["status"] == PS.STATUS_SUCCESS
        assert job["finished_at"] is not None
        assert job["error"] is None

    def test_mark_finished_failure_records_the_message(self, db: str) -> None:
        PS.create_job("j1", "m/x", 1, db_path=db)
        PS.mark_finished("j1", PS.STATUS_FAILURE, error="boom", db_path=db)
        assert PS.get_job("j1", db_path=db)["error"] == "boom"

    def test_duplicate_job_id_rejected(self, db: str) -> None:
        PS.create_job("j1", "m/x", 1, db_path=db)
        with pytest.raises(sqlite3.IntegrityError):
            PS.create_job("j1", "m/y", 1, db_path=db)

    def test_terminal_statuses_are_exactly_success_and_failure(self) -> None:
        assert PS.TERMINAL_STATUSES == frozenset({PS.STATUS_SUCCESS, PS.STATUS_FAILURE})
        assert PS.STATUS_PENDING not in PS.TERMINAL_STATUSES
        assert PS.STATUS_RUNNING not in PS.TERMINAL_STATUSES


class TestSavePredictions:
    def test_writes_rows_and_updates_count(self, db: str, rows: List[Dict[str, Any]]) -> None:
        PS.create_job("j1", "m/x", 2, db_path=db)
        assert PS.save_predictions("j1", rows, db_path=db) == 2
        assert PS.get_job("j1", db_path=db)["n_completed"] == 2

    def test_empty_rows_write_nothing(self, db: str) -> None:
        PS.create_job("j1", "m/x", 0, db_path=db)
        assert PS.save_predictions("j1", [], db_path=db) == 0
        assert PS.get_job("j1", db_path=db)["n_completed"] == 0

    def test_results_come_back_in_submission_order(self, db: str) -> None:
        PS.create_job("j1", "m/x", 3, db_path=db)
        PS.save_predictions("j1", [{"position": 2, "sequence": "C"},
                                   {"position": 0, "sequence": "A"},
                                   {"position": 1, "sequence": "B"}], db_path=db)
        got = [r["sequence"] for r in PS.get_predictions("j1", db_path=db)]
        assert got == ["A", "B", "C"]

    def test_position_defaults_to_enumeration(self, db: str) -> None:
        PS.create_job("j1", "m/x", 2, db_path=db)
        PS.save_predictions("j1", [{"sequence": "A"}, {"sequence": "B"}], db_path=db)
        assert [r["position"] for r in PS.get_predictions("j1", db_path=db)] == [0, 1]

    def test_row_level_error_is_preserved(self, db: str, rows: List[Dict[str, Any]]) -> None:
        # A bad sequence must fail its own row, not the job.
        PS.create_job("j1", "m/x", 2, db_path=db)
        PS.save_predictions("j1", rows, db_path=db)
        got = PS.get_predictions("j1", db_path=db)
        assert got[0]["error"] is None and got[0]["probability"] == pytest.approx(0.87)
        assert "Z" in got[1]["error"]

    def test_accumulates_across_calls(self, db: str) -> None:
        PS.create_job("j1", "m/x", 2, db_path=db)
        PS.save_predictions("j1", [{"position": 0, "sequence": "A"}], db_path=db)
        PS.save_predictions("j1", [{"position": 1, "sequence": "B"}], db_path=db)
        assert PS.get_job("j1", db_path=db)["n_completed"] == 2

    def test_predictions_for_unknown_job_is_empty(self, db: str) -> None:
        assert PS.get_predictions("nope", db_path=db) == []

    def test_in_domain_round_trips_as_a_real_bool(self, db: str) -> None:
        PS.create_job("j1", "m/x", 2, db_path=db)
        PS.save_predictions("j1", [{"position": 0, "sequence": "A", "in_domain": True},
                                   {"position": 1, "sequence": "B", "in_domain": False}],
                            db_path=db)
        got = PS.get_predictions("j1", db_path=db)
        assert got[0]["in_domain"] is True
        assert got[1]["in_domain"] is False

    def test_missing_in_domain_round_trips_as_none(self, db: str) -> None:
        # A row from a model with no applicability domain (a legacy model, or an older release)
        # must read back as "never checked", not as a stored False.
        PS.create_job("j1", "m/x", 1, db_path=db)
        PS.save_predictions("j1", [{"position": 0, "sequence": "A"}], db_path=db)
        assert PS.get_predictions("j1", db_path=db)[0]["in_domain"] is None

    def test_foreign_key_blocks_orphan_rows(self, db: str) -> None:
        with pytest.raises(sqlite3.IntegrityError):
            PS.save_predictions("no-such-job", [{"position": 0, "sequence": "A"}], db_path=db)


class TestRecentJobs:
    def test_empty_store(self, db: str) -> None:
        assert PS.recent_jobs(db_path=db) == []

    def test_newest_first(self, db: str) -> None:
        for i in range(3):
            PS.create_job(f"j{i}", "m/x", 1, db_path=db)
        assert [j["job_id"] for j in PS.recent_jobs(db_path=db)][0] == "j2"

    def test_respects_limit(self, db: str) -> None:
        for i in range(5):
            PS.create_job(f"j{i}", "m/x", 1, db_path=db)
        assert len(PS.recent_jobs(limit=2, db_path=db)) == 2
