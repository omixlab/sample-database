"""
Tests for :mod:`sample.acquisition` -- the acquisition driver's one piece of real logic: diffing
the pipeline's running consensus CSV against the live release database, then handing exactly the
new rows to :func:`sample.incremental.update_families`.
"""
import sqlite3
from typing import Any

import pandas as pd
import pytest

from sample import acquisition


def _make_db(path: str, entry_ids: Any) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE peptides (entry_id TEXT PRIMARY KEY, sequence TEXT)")
    conn.executemany("INSERT INTO peptides (entry_id, sequence) VALUES (?, ?)",
                     [(eid, "SEQ") for eid in entry_ids])
    conn.commit()
    conn.close()


class TestFindNewPeptides:
    def test_no_consensus_file_is_empty_not_an_error(self, tmp_path: Any) -> None:
        assert acquisition.find_new_peptides(str(tmp_path / "missing.csv")) == {}

    def test_rows_absent_from_the_database_are_new(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "sample.db")
        _make_db(db_path, ["SAMPLE0000000001"])

        consensus = tmp_path / "consensus.csv"
        pd.DataFrame([
            {"entry_id": "SAMPLE0000000001", "sequence": "KLLKKLLKKLLKKLLKKL"},  # already known
            {"entry_id": "SAMPLE0000000002", "sequence": "wywywywywywywywywy"},  # new, lowercase
        ]).to_csv(consensus, index=False)

        new = acquisition.find_new_peptides(str(consensus), db_path=db_path)
        assert new == {"SAMPLE0000000002": "WYWYWYWYWYWYWYWYWY"}

    def test_rows_missing_id_or_sequence_are_skipped(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "sample.db")
        _make_db(db_path, [])

        consensus = tmp_path / "consensus.csv"
        pd.DataFrame([
            {"entry_id": "", "sequence": "KLLKKLLKKLLKKLLKKL"},
            {"entry_id": "SAMPLE0000000003", "sequence": ""},
        ]).to_csv(consensus, index=False)

        assert acquisition.find_new_peptides(str(consensus), db_path=db_path) == {}

    def test_missing_columns_is_empty_not_an_error(self, tmp_path: Any) -> None:
        db_path = str(tmp_path / "sample.db")
        _make_db(db_path, [])
        consensus = tmp_path / "consensus.csv"
        pd.DataFrame([{"foo": "bar"}]).to_csv(consensus, index=False)
        assert acquisition.find_new_peptides(str(consensus), db_path=db_path) == {}


class TestRunIncrementalUpdate:
    def test_nothing_new_reports_zero_without_touching_families(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = str(tmp_path / "sample.db")
        _make_db(db_path, [])
        monkeypatch.setattr(acquisition, "find_new_peptides", lambda *a, **k: {})

        summary = acquisition.run_incremental_update(db_path=db_path)
        assert summary == {"new_peptides": 0}

    def test_new_peptides_are_handed_to_update_families(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        db_path = str(tmp_path / "sample.db")
        _make_db(db_path, [])
        monkeypatch.setattr(acquisition, "find_new_peptides",
                            lambda *a, **k: {"NEW1": "WYWYWYWYWYWYWYWYWY"})

        captured = {}

        def fake_update_families(new_sequences: Any, **kwargs: Any) -> Any:
            captured["new_sequences"] = new_sequences
            captured["kwargs"] = kwargs
            return {"assigned": 1, "inherited": 0, "new_clusters": 1, "skipped_existing": 0,
                    "applied": kwargs.get("apply", True)}

        import sample.incremental as incremental_module
        monkeypatch.setattr(incremental_module, "update_families", fake_update_families)

        summary = acquisition.run_incremental_update(db_path=db_path, dry_run=True)
        assert captured["new_sequences"] == {"NEW1": "WYWYWYWYWYWYWYWYWY"}
        assert captured["kwargs"]["apply"] is False
        assert summary["new_peptides"] == 1
        assert summary["new_clusters"] == 1
