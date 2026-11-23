"""
Tests for the table pass wired into the extraction step (:mod:`sample.pipeline_runner`).

These cover the two pure helpers -- PDF resolution and the merge -- without touching docling, a
PDF, or the ``sample-docling`` environment.

The merge is the one that matters. It runs on every document in the corpus, and a mistake there
duplicates peptides rather than losing them, which is the harder error to notice: the counts go up,
which looks like the feature working.
"""
import os
from typing import Any, Dict, List

import pytest

from sample import pipeline_runner as P


class TestPdfPathFor:
    @pytest.mark.parametrize("text_path,expected", [
        ("data/literature_data_text/pubmed/abc.txt", "data/literature_data/pubmed/abc.pdf"),
        ("data/literature_data_text/scopus/abc.txt", "data/literature_data/scopus/abc.pdf"),
        ("data/antimicrobial_databases/papers_text/dramp/x.txt",
         "data/antimicrobial_databases/papers/dramp/x.pdf"),
        ("data/patent_data_text/google_patents/p.txt", "data/patent_data/google_patents/p.pdf"),
    ])
    def test_drops_the_text_suffix_for_every_corpus(self, tmp_path: Any, monkeypatch: Any,
                                                    text_path: str, expected: str) -> None:
        # One rule rather than a per-source table: a corpus added later follows the same naming and
        # keeps working without anyone remembering to extend a mapping.
        monkeypatch.setattr(os.path, "exists", lambda p: p == expected)
        assert P._pdf_path_for(text_path) == expected

    def test_returns_none_when_the_pdf_is_absent(self) -> None:
        # A document with no PDF gets no table pass; that is not an error.
        assert P._pdf_path_for("data/literature_data_text/pubmed/does-not-exist.txt") is None

    def test_returns_none_when_the_path_has_no_text_directory(self) -> None:
        assert P._pdf_path_for("/tmp/loose/file.txt") is None


class TestMergeTableRecords:
    def _table_record(self, sequence: str, activity_type: str = "MIC") -> Dict[str, Any]:
        return {
            "peptide_name": "TL", "peptide_sequence": sequence,
            "discovery_path": "table_match",
            "biological_activities": [{"activity_type": activity_type, "value_lower": 12.0}],
            "context": "", "source_page": 6,
        }

    def test_a_peptide_in_both_prose_and_a_table_is_one_peptide(self) -> None:
        # The table usually holds the measurements and the prose the description. Emitting both as
        # separate records would double-count the molecule -- and the peptide count would RISE,
        # which reads as the feature working.
        prose: List[Dict[str, Any]] = [{
            "peptide_sequence": "FVQWFSKFLGRIL", "peptide_name": "TL",
            "discovery_path": "sequence_match",
            "biological_activities": [{"activity_type": "Hemolysis"}],
        }]
        merged, added = P._merge_table_records(prose, [self._table_record("FVQWFSKFLGRIL")])
        assert len(merged) == 1
        assert added == 0
        assert len(merged[0]["biological_activities"]) == 2
        assert merged[0]["discovery_path"] == "sequence_match", "prose provenance is not overwritten"

    def test_matching_ignores_case_and_surrounding_space(self) -> None:
        prose = [{"peptide_sequence": " fvqwfskflgril ", "biological_activities": []}]
        merged, added = P._merge_table_records(prose, [self._table_record("FVQWFSKFLGRIL")])
        assert added == 0 and len(merged) == 1

    def test_a_peptide_only_a_table_reports_is_added(self) -> None:
        merged, added = P._merge_table_records([], [self._table_record("FLPLIGRVLSGIL")])
        assert added == 1
        assert merged[0]["discovery_path"] == "table_match"

    def test_two_table_records_for_one_sequence_collapse(self) -> None:
        records = [self._table_record("FLPLIGRVLSGIL", "MIC"),
                   self._table_record("FLPLIGRVLSGIL", "Hemolysis")]
        merged, added = P._merge_table_records([], records)
        assert added == 1
        assert len(merged[0]["biological_activities"]) == 2

    def test_a_record_without_a_sequence_is_still_kept(self) -> None:
        # Papers that name peptides only by code ("H2"/"H4") yield activities with no sequence.
        # They are real measurements and must not be silently dropped for lacking an identity the
        # paper never printed.
        nameless = {"peptide_name": "H2", "peptide_sequence": None,
                    "biological_activities": [{"activity_type": "MIC"}], "context": ""}
        merged, added = P._merge_table_records([], [nameless])
        assert added == 1 and len(merged) == 1

    def test_the_original_records_are_not_reordered(self) -> None:
        prose = [{"peptide_sequence": "AAAAAAAAAA", "biological_activities": []},
                 {"peptide_sequence": "CCCCCCCCCC", "biological_activities": []}]
        merged, _ = P._merge_table_records(prose, [self._table_record("FLPLIGRVLSGIL")])
        assert [p["peptide_sequence"] for p in merged[:2]] == ["AAAAAAAAAA", "CCCCCCCCCC"]


class TestToggle:
    def test_tables_are_on_by_default(self) -> None:
        # The step must read tables unless someone deliberately turns them off: they were
        # unreachable from this path before, so at corpus scale no table was ever read.
        assert P._EXTRACT_TABLES is True or os.getenv("SAMPLE_EXTRACT_TABLES") is not None

    def test_table_records_are_labelled_distinctly(self) -> None:
        # Provenance has to survive into the record: a table peptide carries no context window, so
        # a reader needs to know why rather than inferring it from an empty field.
        assert P._TABLE_DISCOVERY_PATH == "table_match"
