"""
Tests for ATCC product links on activity rows.

DBAASP's raw `targetSpecies.name` ("Staphylococcus aureus ATCC 29213") is stored verbatim in
`activities.target_organism` (`sample/serialize.py`'s DBAASP block) -- 59,904 of 174,163 released
activities (34.4%) carry a catalog number this way, with no separate column for it. The feature
reads the number back out of that string at display time (`sample.nlp.extract_atcc_id`) and builds
the ATCC product URL (`sample.external_links.build_url("atcc", ...)`) rather than adding a new
stored column, since the source string is already there.

Real-data verification (99.80% of the 59,904 ATCC-qualified activities resolve to a link; the
remaining ~0.2% are malformed strain codes in the source data, correctly left unlinked rather than
guessed at) is recorded in `DECISIONS.md`, not re-run here -- these tests are hermetic.
"""
import sqlite3
from typing import Any

import pytest


class TestExtractAtccId:
    @pytest.mark.parametrize("organism,expected", [
        ("Staphylococcus aureus ATCC 29213", "29213"),
        ("Escherichia coli ATCC 25922", "25922"),
        ("ATCC# 6538", "6538"),
        ("Klebsiella pneumoniae (ATCC No. 43816)", "43816"),
        ("Acidipropionibacterium jensenii ATCC 4868T", "4868"),   # type-strain marker stripped
        ("Acinetobacter baumannii ATCC BAA-1605", "baa-1605"),    # BAA series, verified live
        ("Staphylococcus aureus ATCC BAA-42", "baa-42"),          # BAA series, 2 digits
        ("Candida parapsilosis ATCC MYA-4646", "mya-4646"),       # MYA series, verified live
    ])
    def test_extracts_the_catalog_number(self, organism: str, expected: str) -> None:
        from sample.nlp import extract_atcc_id

        assert extract_atcc_id(organism) == expected

    @pytest.mark.parametrize("organism", [
        "Escherichia coli",
        "Escherichia coli BL21(DE3)",
        None,
        "",
        # Non-ATCC strain codes seen beside the word "ATCC" in real DBAASP data -- not a catalog
        # number in any of ATCC's own series, and forcing a match would risk a wrong link, which
        # this project's own rule (see DECISIONS.md) treats as worse than no link.
        "Campylobacter jejuni ATCC Cj1",
        "Streptococcus mutans ATCC UA159",
    ])
    def test_returns_none_when_no_catalog_number_is_present(self, organism: Any) -> None:
        from sample.nlp import extract_atcc_id

        assert extract_atcc_id(organism) is None


class TestAtccLinkSpec:
    def test_build_url_uses_the_lowercase_id_in_the_path(self) -> None:
        from sample.external_links import build_url

        assert build_url("atcc", "29213") == "https://www.atcc.org/products/29213"
        assert build_url("atcc", "baa-1605") == "https://www.atcc.org/products/baa-1605"

    def test_case_insensitive_namespace(self) -> None:
        from sample.external_links import build_url

        assert build_url("ATCC", "29213") == build_url("atcc", "29213")


class TestWithTaxonomyLink:
    """`sample.db._with_taxonomy_link` -- extended this session to also set `target_atcc_url`
    alongside the pre-existing `target_taxonomy_url`, from the same row."""

    def test_sets_atcc_url_when_the_organism_carries_a_catalog_number(self) -> None:
        from sample.db import _with_taxonomy_link

        row = {"target_organism": "Staphylococcus aureus ATCC 29213", "target_taxid": 1280}
        result = _with_taxonomy_link(row)

        assert result["target_atcc_url"] == "https://www.atcc.org/products/29213"

    def test_atcc_url_is_none_when_no_catalog_number_is_present(self) -> None:
        from sample.db import _with_taxonomy_link

        row = {"target_organism": "Escherichia coli", "target_taxid": 562}
        result = _with_taxonomy_link(row)

        assert result["target_atcc_url"] is None

    def test_taxonomy_url_still_works_unchanged(self) -> None:
        """Regression guard: adding the ATCC field must not disturb the existing one."""
        from sample.db import _with_taxonomy_link

        row = {"target_organism": "Staphylococcus aureus ATCC 29213", "target_taxid": 1280}
        result = _with_taxonomy_link(row)

        assert result["target_taxonomy_url"] == "https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=1280"

    def test_handles_a_row_with_no_target_organism(self) -> None:
        from sample.db import _with_taxonomy_link

        result = _with_taxonomy_link({"target_organism": None, "target_taxid": None})
        assert result["target_atcc_url"] is None
        assert result["target_taxonomy_url"] is None


class TestGetPeptideActivityIncludesAtccLink:
    """End-to-end through `sample.db.get_peptide`, the actual repository function pages and the
    API both read through -- not just the helper function in isolation."""

    @pytest.fixture
    def db_path(self, tmp_path: Any) -> str:
        path = str(tmp_path / "sample.db")
        conn = sqlite3.connect(path)
        conn.executescript("""
            CREATE TABLE peptides (entry_id TEXT PRIMARY KEY, sequence TEXT NOT NULL, SMILES TEXT,
                source TEXT, is_decoy INTEGER DEFAULT 0, LITERATURE TEXT, PATENT TEXT,
                Reference TEXT, Biological_Activity TEXT);
            CREATE TABLE descriptors (entry_id TEXT PRIMARY KEY, molecular_weight REAL, logp REAL,
                hydrogen_bond_donors INTEGER, hydrogen_bond_acceptors INTEGER, charge REAL,
                sequence_length INTEGER);
            CREATE TABLE activities (id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id TEXT,
                activity_group TEXT, value REAL, unit TEXT, target_organism TEXT,
                target_taxid INTEGER, activity_type TEXT, mechanism_of_action TEXT,
                source_accession TEXT, source_reference_index TEXT);
            CREATE TABLE papers (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT,
                source_id TEXT, UNIQUE(source_type, source_id));
            CREATE TABLE peptide_references (peptide_id TEXT, paper_id INTEGER,
                UNIQUE(peptide_id, paper_id));
            CREATE VIRTUAL TABLE peptides_fts USING fts5(entry_id UNINDEXED, sequence,
                literature, patent, tokenize="unicode61");
        """)
        conn.execute(
            "INSERT INTO peptides (entry_id, sequence, is_decoy) VALUES "
            "('SAMPLE0000000001', 'GIGKFLHSAKKFGKAFVGEIMNS', 0)")
        conn.execute(
            "INSERT INTO activities (entry_id, activity_group, value, unit, target_organism, "
            "target_taxid, activity_type) VALUES ('SAMPLE0000000001', 'antibacterial', 16.0, "
            "'ug/mL', 'Staphylococcus aureus ATCC 29213', 1280, 'MIC')")
        conn.commit()
        conn.close()
        return path

    def test_activity_carries_the_atcc_link(self, db_path: str,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
        from sample import db

        monkeypatch.setenv("SAMPLE_DB", db_path)
        monkeypatch.setattr(db, "default_db_path", lambda: db_path)

        peptide = db.get_peptide("SAMPLE0000000001")
        assert peptide is not None
        assert len(peptide["activities"]) == 1
        assert peptide["activities"][0]["target_atcc_url"] == "https://www.atcc.org/products/29213"
