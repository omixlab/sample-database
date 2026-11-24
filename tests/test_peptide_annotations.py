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


class TestDomainList:
    """
    The Pfam-accession field as the release database actually stores it.

    ``peptide_families.all_domains`` is a comma-separated STRING, and :func:`build_all` passes it
    straight through. The original code did ``list(pfam_domains or [])``, which explodes a string
    into single characters; the peptide page renders one badge per element, so the single real
    domain ``PF12121`` displayed as seven badges reading P, F, 1, 2, 1, 2, 1. Every pre-existing
    test passed an already-split list, so the suite was green while the only path that runs in
    production was wrong. These pin the string form.
    """

    def test_single_accession_string_is_not_split_into_characters(self) -> None:
        assert pa._domain_list("PF12121") == ["PF12121"]

    def test_comma_separated_string_splits_on_the_comma(self) -> None:
        assert pa._domain_list("PF16211,PF00125,PF00808") == ["PF16211", "PF00125", "PF00808"]

    def test_surrounding_whitespace_is_stripped(self) -> None:
        assert pa._domain_list("PF01097, PF00304") == ["PF01097", "PF00304"]

    def test_empty_and_none_are_empty_lists(self) -> None:
        assert pa._domain_list("") == []
        assert pa._domain_list(None) == []
        assert pa._domain_list([]) == []

    def test_trailing_comma_does_not_produce_a_blank_accession(self) -> None:
        assert pa._domain_list("PF00304,") == ["PF00304"]

    def test_already_split_list_passes_through(self) -> None:
        assert pa._domain_list(["PF00711", "PF00304"]) == ["PF00711", "PF00304"]

    def test_annotate_one_accepts_the_string_the_database_supplies(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(pa, "_near_identity_hits", lambda seq, self_id, limit=10: [])
        monkeypatch.setattr(pa, "_hmm_hits", lambda seq: [])
        ann = pa.annotate_one("E1", "GIGKFL", is_circular=False,
                              pfam_domains="PF16211,PF00125")
        assert ann["pfam_domains"] == ["PF16211", "PF00125"]


class TestPredictorsFromCSV:
    """
    Reusing already-computed predictor values instead of re-running the tools.

    The predictors are pure functions of the sequence, so a value is reusable for any record
    carrying the same sequence -- and the sequence-identity guard is the only thing standing
    between that and silently attaching another release's numbers to a peptide. These tests are
    weighted towards the guard refusing, and towards a missing/unusable source failing loudly
    rather than looking like "every predictor declined".
    """

    PRED = "macrel_amp_probability"

    def _csv(self, tmp_path: Any, rows: str, header: str = None) -> str:
        path = tmp_path / "release.csv"
        head = header or f"entry_id,sequence,{self.PRED}\n"
        path.write_text(head + rows, encoding="utf-8")
        return str(path)

    def test_matching_sequence_supplies_the_value(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,GIGKFL,0.87\n")
        out = pa._predictors_from_csv(csv, {"E1": "GIGKFL"})
        assert out == {"E1": {self.PRED: 0.87}}

    def test_sequence_mismatch_drops_the_row_rather_than_mislabelling_it(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,TOTALLYDIFFERENT,0.87\n")
        out = pa._predictors_from_csv(csv, {"E1": "GIGKFL"})
        assert out == {}, "a row whose sequence disagrees with the database must not be used"

    def test_match_ignores_case_and_surrounding_whitespace(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1, gigkfl ,0.87\n")
        assert pa._predictors_from_csv(csv, {"E1": "GIGKFL"}) == {"E1": {self.PRED: 0.87}}

    def test_ids_absent_from_the_database_are_ignored(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,GIGKFL,0.87\nE9,KWKLFKK,0.5\n")
        out = pa._predictors_from_csv(csv, {"E1": "GIGKFL"})
        assert set(out) == {"E1"}

    def test_ids_absent_from_the_csv_are_simply_absent(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,GIGKFL,0.87\n")
        out = pa._predictors_from_csv(csv, {"E1": "GIGKFL", "E2": "KWKLFKK"})
        assert set(out) == {"E1"}, "the caller decides what to do about E2, not this function"

    def test_first_row_wins_for_a_duplicated_id(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,GIGKFL,0.87\nE1,GIGKFL,0.11\n")
        assert pa._predictors_from_csv(csv, {"E1": "GIGKFL"})["E1"][self.PRED] == 0.87

    def test_null_is_carried_over_as_the_tools_declining(self, tmp_path: Any) -> None:
        # A blank predictor cell is the release's own record that the tool refused that sequence;
        # _predictor_values then omits the field instead of zero-filling it.
        csv = self._csv(tmp_path, "E1,GIGKFL,\n")
        out = pa._predictors_from_csv(csv, {"E1": "GIGKFL"})
        assert set(out) == {"E1"}
        assert pa._predictor_values(out["E1"]) == {}

    def test_missing_file_raises_rather_than_returning_an_empty_map(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError):
            pa._predictors_from_csv(str(tmp_path / "nope.csv"), {"E1": "GIGKFL"})

    def test_csv_without_a_sequence_column_raises(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,0.87\n", header=f"entry_id,{self.PRED}\n")
        with pytest.raises(ValueError, match="sequence"):
            pa._predictors_from_csv(csv, {"E1": "GIGKFL"})

    def test_csv_with_no_predictor_columns_at_all_raises(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,GIGKFL\n", header="entry_id,sequence\n")
        with pytest.raises(ValueError, match="predictor column"):
            pa._predictors_from_csv(csv, {"E1": "GIGKFL"})

    def test_only_known_predictor_columns_are_read(self, tmp_path: Any) -> None:
        csv = self._csv(tmp_path, "E1,GIGKFL,0.87,junk\n",
                        header=f"entry_id,sequence,{self.PRED},unrelated\n")
        assert pa._predictors_from_csv(csv, {"E1": "GIGKFL"}) == {"E1": {self.PRED: 0.87}}
