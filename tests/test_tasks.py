"""
Tests for :mod:`sample.tasks` — the Celery prediction task.

The task is invoked directly rather than through a broker. What matters here is its contract with
the store: a job must always reach a terminal state and record why, because the browser polls the
database and a job that dies without writing its own failure is indistinguishable from one still
running.
"""
from typing import Any, Dict, List
from unittest import mock

import pytest

from sample import prediction_store as PS
from sample import tasks as T


@pytest.fixture
def db(tmp_path: Any) -> str:
    """:returns: Path to an initialized prediction database."""
    path = str(tmp_path / "predictions.db")
    PS.init_db(path)
    return path


@pytest.fixture
def peptides() -> List[Dict[str, str]]:
    """:returns: Two valid peptides and one with a non-standard residue."""
    return [
        {"peptide_name": "magainin-2", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS"},
        {"peptide_name": "indolicidin", "sequence": "ILPWKWPWWPWRR"},
        {"peptide_name": "broken", "sequence": "GIGKZZ"},
    ]


def _fake_predict(sequences: Any, model_id: str, *args: Any, **kwargs: Any) -> List[Dict]:
    """Stands in for :func:`sample.inference.predict`, which would load a real model."""
    return [{"position": i, "sequence": s, "probability": 0.75,
             "predicted_label": "active", "value": None}
            for i, s in enumerate(sequences)]


class TestConfiguration:
    def test_broker_and_backend_are_distinct_databases(self) -> None:
        # Task state and the queue share a Redis server but not a logical database.
        assert T.BROKER_URL != T.RESULT_BACKEND

    def test_queue_name_matches_what_the_worker_consumes(self) -> None:
        assert T.QUEUE_NAME == "sample-predictions"
        assert T.celery_app.conf.task_default_queue == T.QUEUE_NAME

    def test_results_expire_because_sqlite_is_the_record_of_truth(self) -> None:
        assert T.celery_app.conf.result_expires

    def test_prefetch_is_one(self) -> None:
        # These tasks are memory-bound and may hold a loaded transformer; prefetching would
        # serialize a queue behind one process while others idle.
        assert T.celery_app.conf.worker_prefetch_multiplier == 1

    def test_time_limits_are_ordered(self) -> None:
        assert T.celery_app.conf.task_soft_time_limit < T.celery_app.conf.task_time_limit


class TestBrokerAvailable:
    def test_true_when_the_connection_succeeds(self,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(T.celery_app, "connection", lambda: mock.MagicMock())
        assert T.broker_available() is True

    def test_false_when_the_connection_raises(self,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        # The submit route depends on this to refuse cleanly rather than hand out a job id for a
        # job that will never run.
        def _boom() -> None:
            raise OSError("connection refused")
        monkeypatch.setattr(T.celery_app, "connection", _boom)
        assert T.broker_available() is False

    def test_true_in_eager_mode_without_a_broker(self,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(T.celery_app.conf, "task_always_eager", True)
        assert T.broker_available() is True


class TestPredictPeptides:
    def test_successful_run_reaches_success(self, db: str,
                                            peptides: List[Dict[str, str]]) -> None:
        PS.create_job("j1", "m/x", len(peptides), db_path=db)
        with mock.patch("sample.inference.predict", side_effect=_fake_predict):
            T.predict_peptides.apply(args=("j1", "m/x", peptides, db)).get()
        assert PS.get_job("j1", db_path=db)["status"] == PS.STATUS_SUCCESS

    def test_invalid_sequence_fails_its_row_not_the_job(
            self, db: str, peptides: List[Dict[str, str]]) -> None:
        # One typo among fifty sequences must not discard the other forty-nine.
        PS.create_job("j1", "m/x", len(peptides), db_path=db)
        with mock.patch("sample.inference.predict", side_effect=_fake_predict):
            T.predict_peptides.apply(args=("j1", "m/x", peptides, db)).get()

        job = PS.get_job("j1", db_path=db)
        rows = PS.get_predictions("j1", db_path=db)
        assert job["status"] == PS.STATUS_SUCCESS
        assert len(rows) == 3
        scored = [r for r in rows if r["error"] is None]
        failed = [r for r in rows if r["error"]]
        assert len(scored) == 2 and len(failed) == 1
        assert "Z" in failed[0]["error"]

    def test_rows_are_ordered_by_submission_position(
            self, db: str, peptides: List[Dict[str, str]]) -> None:
        PS.create_job("j1", "m/x", len(peptides), db_path=db)
        with mock.patch("sample.inference.predict", side_effect=_fake_predict):
            T.predict_peptides.apply(args=("j1", "m/x", peptides, db)).get()
        rows = PS.get_predictions("j1", db_path=db)
        assert [r["position"] for r in rows] == [0, 1, 2]
        assert rows[2]["peptide_name"] == "broken"

    def test_peptide_names_are_preserved(self, db: str,
                                         peptides: List[Dict[str, str]]) -> None:
        PS.create_job("j1", "m/x", len(peptides), db_path=db)
        with mock.patch("sample.inference.predict", side_effect=_fake_predict):
            T.predict_peptides.apply(args=("j1", "m/x", peptides, db)).get()
        names = [r["peptide_name"] for r in PS.get_predictions("j1", db_path=db)]
        assert names == ["magainin-2", "indolicidin", "broken"]

    def test_model_failure_is_recorded_not_raised(self, db: str,
                                                  peptides: List[Dict[str, str]]) -> None:
        # A job that dies without writing its own failure looks exactly like one still running.
        PS.create_job("j1", "m/x", len(peptides), db_path=db)
        with mock.patch("sample.inference.predict",
                        side_effect=RuntimeError("model exploded")):
            result = T.predict_peptides.apply(args=("j1", "m/x", peptides, db)).get()

        job = PS.get_job("j1", db_path=db)
        assert job["status"] == PS.STATUS_FAILURE
        assert "model exploded" in job["error"]
        assert "error" in result

    def test_all_invalid_still_terminates_successfully(self, db: str) -> None:
        bad = [{"peptide_name": "a", "sequence": "ZZZ"}]
        PS.create_job("j1", "m/x", 1, db_path=db)
        T.predict_peptides.apply(args=("j1", "m/x", bad, db)).get()

        assert PS.get_job("j1", db_path=db)["status"] == PS.STATUS_SUCCESS
        assert PS.get_predictions("j1", db_path=db)[0]["error"]

    def test_empty_peptide_list_terminates(self, db: str) -> None:
        PS.create_job("j1", "m/x", 0, db_path=db)
        T.predict_peptides.apply(args=("j1", "m/x", [], db)).get()
        assert PS.get_job("j1", db_path=db)["status"] == PS.STATUS_SUCCESS

    def test_summary_does_not_carry_the_results(self, db: str,
                                                peptides: List[Dict[str, str]]) -> None:
        # The return value goes to Redis and expires; nothing may depend on it for results.
        PS.create_job("j1", "m/x", len(peptides), db_path=db)
        with mock.patch("sample.inference.predict", side_effect=_fake_predict):
            result = T.predict_peptides.apply(args=("j1", "m/x", peptides, db)).get()
        assert set(result) == {"job_id", "n_rows", "seconds"}
