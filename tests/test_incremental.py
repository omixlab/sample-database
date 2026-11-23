"""
Tests for :mod:`sample.incremental` -- placing newly ingested peptides into the existing family map.

The distinctions these tests defend are the ones that make an *incremental* update worth having
over a full rebuild: a new sequence that resembles the corpus must **inherit** the established
family (never mint a duplicate one), several new sequences that resemble each other but nothing
existing must share **one** new family (not one each), and minted ids must continue the existing
``SAMPLE_FAM_*`` numbering so no id another table already points at is ever reused or renumbered.

The matching runs real PepMatch, like the batch clusterer it must agree with; the sequences are
short synthetic analogues chosen so a one-substitution neighbour is unambiguous.
"""
import json
from typing import Any, Dict

import pytest

from sample import incremental


# An existing corpus: one clustered family and one Pfam-anchored family, 18-mers so a single
# substitution is comfortably inside the 80% identity / 80% length-ratio family threshold.
EXISTING_SEQUENCES: Dict[str, str] = {
    "E1": "KLLKKLLKKLLKKLLKKL",   # clustered family SAMPLE_FAM_000005
    "E2": "WYWYWYWYWYWYWYWYWY",   # Pfam family PF00711
}
EXISTING_FAMILIES: Dict[str, Dict[str, Any]] = {
    "E1": {"family_id": "SAMPLE_FAM_000005", "family_name": None, "source": "cluster",
           "all_domains": []},
    "E2": {"family_id": "PF00711", "family_name": "Defensin_beta", "source": "pfam",
           "all_domains": ["PF00711"]},
}


class TestNextFamilyNumber:
    def test_empty_map_starts_at_one(self) -> None:
        assert incremental.next_family_number([]) == 1

    def test_one_past_the_highest_cluster(self) -> None:
        assert incremental.next_family_number(EXISTING_FAMILIES.values()) == 6

    def test_pfam_accessions_do_not_count(self) -> None:
        # A corpus of only Pfam families has minted no cluster ids, so numbering starts fresh.
        pfam_only = [{"family_id": "PF00711"}, {"family_id": "PF12345"}]
        assert incremental.next_family_number(pfam_only) == 1

    def test_gaps_are_not_reused(self) -> None:
        # Numbering continues past the maximum, never filling a hole a deleted family left behind.
        families = [{"family_id": "SAMPLE_FAM_000002"}, {"family_id": "SAMPLE_FAM_000009"}]
        assert incremental.next_family_number(families) == 10


class TestFormatFamilyId:
    def test_zero_padded_to_six(self) -> None:
        assert incremental.format_family_id(7) == "SAMPLE_FAM_000007"


class TestAssignFamilies:
    def test_near_identical_inherits_the_existing_cluster(self) -> None:
        # One substitution off E1 -> same family, and the match is recorded for provenance.
        result = incremental.assign_families(
            {"N1": "KLLKKLLKKLLKKLLKKA"}, EXISTING_SEQUENCES, EXISTING_FAMILIES)
        assert result["N1"]["family_id"] == "SAMPLE_FAM_000005"
        assert result["N1"]["source"] == "inherited"
        assert result["N1"]["matched_entry_id"] == "E1"

    def test_near_identical_inherits_a_pfam_family(self) -> None:
        # Near-identity to a curated-family member is evidence of that family, even without an
        # hmmsearch: the new peptide inherits the Pfam id rather than starting a cluster.
        result = incremental.assign_families(
            {"N4": "WYWYWYWYWYWYWYWYWA"}, EXISTING_SEQUENCES, EXISTING_FAMILIES)
        assert result["N4"]["family_id"] == "PF00711"
        assert result["N4"]["source"] == "inherited"

    def test_novel_sequence_mints_a_new_family(self) -> None:
        result = incremental.assign_families(
            {"N2": "CDEGCDEGCDEGCDEGCD"}, EXISTING_SEQUENCES, EXISTING_FAMILIES)
        assert result["N2"]["source"] == "new-cluster"
        # Continues past the existing maximum (SAMPLE_FAM_000005).
        assert result["N2"]["family_id"] == "SAMPLE_FAM_000006"
        assert result["N2"]["matched_entry_id"] is None

    def test_mutually_similar_novel_sequences_share_one_family(self) -> None:
        # Two new sequences one substitution apart, resembling nothing existing, must land in the
        # SAME new family -- the whole point of clustering the new batch against itself.
        result = incremental.assign_families(
            {"N2": "CDEGCDEGCDEGCDEGCD", "N3": "CDEGCDEGCDEGCDEGCA"},
            EXISTING_SEQUENCES, EXISTING_FAMILIES)
        assert result["N2"]["family_id"] == result["N3"]["family_id"]
        assert result["N2"]["source"] == "new-cluster"

    def test_full_batch_places_each_correctly(self) -> None:
        result = incremental.assign_families(
            {"N1": "KLLKKLLKKLLKKLLKKA",   # -> E1's cluster
             "N2": "CDEGCDEGCDEGCDEGCD",   # -> new family
             "N3": "CDEGCDEGCDEGCDEGCA",   # -> same new family as N2
             "N4": "WYWYWYWYWYWYWYWYWA"},  # -> E2's Pfam family
            EXISTING_SEQUENCES, EXISTING_FAMILIES)
        assert result["N1"]["family_id"] == "SAMPLE_FAM_000005"
        assert result["N4"]["family_id"] == "PF00711"
        assert result["N2"]["family_id"] == result["N3"]["family_id"] == "SAMPLE_FAM_000006"

    def test_no_existing_corpus_clusters_the_batch_alone(self) -> None:
        # The very first update, before any family map exists: new sequences still cluster among
        # themselves and start the numbering at 1.
        result = incremental.assign_families(
            {"A": "CDEGCDEGCDEGCDEGCD", "B": "CDEGCDEGCDEGCDEGCA", "C": "KLLKKLLKKLLKKLLKKL"},
            {}, {})
        assert result["A"]["family_id"] == result["B"]["family_id"] == "SAMPLE_FAM_000001"
        assert result["C"]["family_id"] == "SAMPLE_FAM_000002"
        assert result["C"]["source"] == "new-cluster"

    def test_unclusterable_sequence_still_gets_a_family(self) -> None:
        # Too short to index, but must not vanish from the update -- it becomes its own family.
        result = incremental.assign_families(
            {"TINY": "AC"}, EXISTING_SEQUENCES, EXISTING_FAMILIES)
        assert "TINY" in result
        assert result["TINY"]["source"] == "new-cluster"
        assert result["TINY"]["family_id"] == "SAMPLE_FAM_000006"

    def test_empty_input_returns_empty(self) -> None:
        assert incremental.assign_families({}, EXISTING_SEQUENCES, EXISTING_FAMILIES) == {}


class TestApplyAssignments:
    def test_merges_without_touching_existing_entries(self, tmp_path: Any) -> None:
        path = tmp_path / "families.json"
        path.write_text(json.dumps(EXISTING_FAMILIES))

        incremental.apply_assignments(
            {"N1": {"family_id": "SAMPLE_FAM_000005", "family_name": None, "source": "inherited",
                    "matched_entry_id": "E1", "all_domains": []}},
            families_json=str(path))

        merged = json.loads(path.read_text())
        # Both existing entries survive verbatim, and the new one is added in the load-sql shape.
        assert merged["E1"]["family_id"] == "SAMPLE_FAM_000005"
        assert merged["E2"]["source"] == "pfam"
        assert merged["N1"]["family_id"] == "SAMPLE_FAM_000005"
        assert merged["N1"]["matched_entry_id"] == "E1"
        # The shape matches build_families' records (evalue key present, defaulted to None).
        assert merged["N1"]["evalue"] is None

    def test_writes_a_fresh_file_when_none_exists(self, tmp_path: Any) -> None:
        path = tmp_path / "sub" / "families.json"
        incremental.apply_assignments(
            {"X": {"family_id": "SAMPLE_FAM_000001", "source": "new-cluster",
                   "family_name": None, "matched_entry_id": None, "all_domains": []}},
            families_json=str(path))
        assert json.loads(path.read_text())["X"]["family_id"] == "SAMPLE_FAM_000001"


class TestUpdateFamilies:
    def test_skips_sequences_already_in_the_family_map(self, tmp_path: Any,
                                                       monkeypatch: Any) -> None:
        # An id already carrying a family is an update of that peptide, not a new one; it keeps its
        # family and is reported as skipped rather than re-clustered.
        monkeypatch.setattr(incremental, "load_existing",
                            lambda *a, **k: (EXISTING_SEQUENCES, EXISTING_FAMILIES))
        summary = incremental.update_families(
            {"E1": "KLLKKLLKKLLKKLLKKL", "N2": "CDEGCDEGCDEGCDEGCD"},
            families_json=str(tmp_path / "families.json"), apply=False)
        assert summary["skipped_existing"] == 1
        assert "E1" not in summary["assignments"]
        assert summary["assigned"] == 1
        assert summary["new_clusters"] == 1

    def test_apply_false_writes_nothing(self, tmp_path: Any, monkeypatch: Any) -> None:
        path = tmp_path / "families.json"
        monkeypatch.setattr(incremental, "load_existing",
                            lambda *a, **k: (EXISTING_SEQUENCES, EXISTING_FAMILIES))
        summary = incremental.update_families(
            {"N2": "CDEGCDEGCDEGCDEGCD"}, families_json=str(path), apply=False)
        assert summary["applied"] is False
        assert not path.exists()
