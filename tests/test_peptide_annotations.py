"""
Tests for :mod:`sample.peptide_annotations` -- the pre-computed, comprehensive per-peptide
annotation (predictors, Pfam/family, near-identity hits, HMM hits), with circular peptides handled
explicitly.

Near-identity search and the HMM scan are mocked throughout (they shell out / hit the database),
keeping these tests hermetic and fast; the properties under test are the assembly and storage logic.
"""
from typing import Any, Dict, List

import pytest

from sample import peptide_annotations as pa


class TestPredictorValues:
    def test_keeps_only_present_numeric_fields_with_labels(self) -> None:
        row = {"macrel_amp_probability": 0.87, "hemopi2_score": None, "unrelated_column": "x"}
        out = pa._predictor_values(row)
        assert out == {"macrel_amp_probability": {"label": "macrel AMP probability", "value": 0.87}}

    def test_drops_nan(self) -> None:
        row = {"toxinpred3_score": float("nan")}
        assert pa._predictor_values(row) == {}

    def test_empty_row(self) -> None:
        assert pa._predictor_values({}) == {}


class TestAnnotateOne:
    @pytest.fixture(autouse=True)
    def _mock_hits(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(pa, "_near_identity_hits",
                            lambda seq, self_id, limit=10: [{"entry_id": "OTHER1", "identity": 95.0,
                                                              "mismatches": 1}])
        monkeypatch.setattr(pa, "_hmm_hits",
                            lambda seq: [{"family_id": "SAMPLE_FAM_000001", "evalue": 1e-10}])

    def test_linear_peptide_has_no_circular_caveat(self) -> None:
        ann = pa.annotate_one("E1", "GIGKFL", is_circular=False,
                              predictor_row={"macrel_amp_probability": 0.9},
                              family={"family_id": "SAMPLE_FAM_000001", "source": "cluster"},
                              pfam_domains=["PF00711"])
        assert ann["is_circular"] is False
        assert "circular_caveat" not in ann
        assert ann["predictors"]["macrel_amp_probability"]["value"] == 0.9
        assert ann["pfam_domains"] == ["PF00711"]
        assert ann["family"]["family_id"] == "SAMPLE_FAM_000001"
        assert ann["near_identity_hits"][0]["entry_id"] == "OTHER1"
        assert ann["hmm_hits"][0]["family_id"] == "SAMPLE_FAM_000001"

    def test_circular_peptide_carries_the_caveat(self) -> None:
        ann = pa.annotate_one("E2", "GIGKFL", is_circular=True)
        assert ann["is_circular"] is True
        assert ann["circular_caveat"] == pa.CIRCULAR_CAVEAT
        # The sequence-based results are still computed (not skipped) -- just flagged.
        assert ann["near_identity_hits"]
        assert ann["hmm_hits"]

    def test_defaults_are_safe_when_nothing_supplied(self) -> None:
        ann = pa.annotate_one("E3", "GIGKFL", is_circular=False)
        assert ann["predictors"] == {}
        assert ann["pfam_domains"] == []
        assert ann["family"] is None


class TestStore:
    def test_save_and_get_roundtrip(self, tmp_path: Any) -> None:
        db = str(tmp_path / "ann.sqlite")
        pa.init_db(db)
        annotation = {"entry_id": "E1", "is_circular": False, "predictors": {}}
        pa.save("E1", annotation, db_path=db)
        assert pa.get("E1", db_path=db) == annotation

    def test_get_missing_entry_is_none(self, tmp_path: Any) -> None:
        db = str(tmp_path / "ann.sqlite")
        pa.init_db(db)
        assert pa.get("NOPE", db_path=db) is None

    def test_get_before_any_db_exists_is_none_not_an_error(self, tmp_path: Any) -> None:
        assert pa.get("E1", db_path=str(tmp_path / "absent.sqlite")) is None

    def test_save_upserts(self, tmp_path: Any) -> None:
        db = str(tmp_path / "ann.sqlite")
        pa.init_db(db)
        pa.save("E1", {"entry_id": "E1", "is_circular": False, "predictors": {"a": 1}}, db_path=db)
        pa.save("E1", {"entry_id": "E1", "is_circular": False, "predictors": {"a": 2}}, db_path=db)
        assert pa.get("E1", db_path=db)["predictors"] == {"a": 2}

    def test_processed_ids(self, tmp_path: Any) -> None:
        db = str(tmp_path / "ann.sqlite")
        pa.init_db(db)
        pa.save("E1", {"entry_id": "E1", "is_circular": False}, db_path=db)
        pa.save("E2", {"entry_id": "E2", "is_circular": True}, db_path=db)
        assert pa.processed_ids(db) == {"E1", "E2"}

    def test_processed_ids_empty_before_any_db(self, tmp_path: Any) -> None:
        assert pa.processed_ids(str(tmp_path / "absent.sqlite")) == set()
