"""
Tests for what the peptide entry page is allowed to show.

Every case here was live on the released page. None of them raised an error or failed a test —
they rendered confidently and said something untrue, which is the failure mode this file exists to
catch: an internal filename presented as a citation, a source organism listed as a peptide's name,
a row of `N/A` presented as a measurement.
"""
import sqlite3
from typing import Any

import pytest


class TestJunkNames:
    """
    dbAMP files several kinds of thing in the column the harvest reads as a peptide name.

    All four shapes below appeared on one peptide (`SAMPLE0000000001`, dermaseptin B2) alongside
    its six real names.
    """

    @pytest.mark.parametrize("name", [
        "South American frog Phyllomedusa bicolor",      # the source organism
        "DMS2_PHYBI",                                    # a UniProt entry name
        "Adenoregulindata about MIC or HC50 is present",  # two fields run together
        "Adenoregulin (Dermasepti",                      # truncated by a fixed-width export
    ])
    def test_artifacts_are_rejected(self, name: str) -> None:
        from sample.names import looks_like_junk_name

        assert looks_like_junk_name(name) is True

    def test_a_bare_binomial_is_left_to_the_taxonomy_check(self) -> None:
        """
        The cheap filter cannot reject two-word strings, and must not try: `Dermaseptin BII` and
        `Xenopus laevis` are the same shape. Only a taxonomy lookup separates them, which is what
        :func:`sample.names.is_organism_name` does during the harvest. The rule here catches the
        descriptive form dbAMP actually files (`South American frog Phyllomedusa bicolor`), where
        the extra words make the binomial unambiguous.
        """
        from sample.names import looks_like_junk_name

        assert looks_like_junk_name("Xenopus laevis") is False
        assert looks_like_junk_name("Dermaseptin BII") is False
        assert looks_like_junk_name("frog Xenopus laevis") is True

    @pytest.mark.parametrize("name", [
        "Dermaseptin-B2", "DRS-B2", "ADENOREGULIN", "DS bII", "Dermaseptin BII",
        "Adenoregulin, Dermaseptin BII, Dermaseptin B2",  # several synonyms in one cell, but real
        "Magainin 2", "LL-37", "AS-48", "HNP1",
    ])
    def test_real_names_are_kept(self, name: str) -> None:
        from sample.names import looks_like_junk_name

        assert looks_like_junk_name(name) is False

    def test_is_usable_name_applies_the_same_rule(self) -> None:
        """The harvest and the display must agree, or a rebuild reintroduces what was filtered."""
        from sample.names import is_usable_name

        assert is_usable_name("Dermaseptin-B2") is True
        assert is_usable_name("DMS2_PHYBI") is False


class TestReferencesAreCitations:
    """`Reference` must never carry an extraction filename."""

    @pytest.mark.parametrize("entry", [
        "File: 1c273b8f-39c6-4d7e-8e73-f0ed8218b517.json",
        "pubmed_old:1c273b8f-39c6-4d7e-8e73-f0ed8218b517.json",
        "pubmed.bak:67cb6d69.json",
        "google_patents:US8394762B2.json",
    ])
    def test_internal_entries_are_recognised(self, entry: str) -> None:
        from sample.db import is_internal_reference

        assert is_internal_reference(entry) is True

    @pytest.mark.parametrize("entry", [
        "Isolation and structure of novel defensive peptides from frog skin., DOI: 10.1111/x",
        "DOI: 10.1073/pnas.86.14.5610",
        "PMID: 8306981",
    ])
    def test_citations_are_kept(self, entry: str) -> None:
        from sample.db import is_internal_reference

        assert is_internal_reference(entry) is False

    def test_page_helper_drops_filename_entries(self) -> None:
        from sample.api.pages import _real_references

        cell = ("File: a.json | Isolation and structure of novel defensive peptides., "
                "DOI: 10.1111/x | File: b.json")
        assert _real_references(cell) == [
            "Isolation and structure of novel defensive peptides., DOI: 10.1111/x"]
        assert _real_references(None) == []
        assert _real_references("File: a.json") == []


class TestSourceDocuments:
    """Links to extraction files must not point at working copies."""

    def test_backup_directories_are_skipped(self) -> None:
        from sample.api.pages import _source_documents

        cell = ("pubmed:live.json;pubmed_old:stale.json;pubmed.bak:stale.json;"
                "scopus_old:stale.json;scopus.bak:stale.json")
        docs = _source_documents(cell, None)

        assert [d["file_name"] for d in docs] == ["live.json"]
        assert docs[0]["url"] == "/extractions/view/literature/pubmed/live.json"

    def test_both_separators_are_accepted(self) -> None:
        """Rows have been written with `;` and with ` | ` depending on which merge produced them."""
        from sample.api.pages import _source_documents

        assert len(_source_documents("pubmed:a.json | scopus:b.json", None)) == 2
        assert len(_source_documents("pubmed:a.json;scopus:b.json", None)) == 2

    def test_patents_are_labelled_separately(self) -> None:
        from sample.api.pages import _source_documents

        docs = _source_documents(None, "google_patents:US8394762B2.json")
        assert docs[0]["kind"] == "patent"
        assert docs[0]["url"] == "https://patents.google.com/patent/US8394762B2"
        assert docs[0]["label"] == "Patent: US8394762B2"
        assert docs[0]["extraction_url"].startswith("/extractions/view/patent/")

    def test_literature_resolves_doi_and_journal_link(self) -> None:
        from sample.api.pages import _source_documents

        uuid = "0002a4e1-e1ea-ab73-cd37-71904e718c146b02a771"
        docs = _source_documents(f"pubmed:{uuid}.json", None)
        assert len(docs) == 1
        assert docs[0]["kind"] == "literature"
        assert docs[0]["doi"] == "10.1128/AAC.49.5.1727-1732.2005"
        assert docs[0]["url"] == "https://doi.org/10.1128/AAC.49.5.1727-1732.2005"
        assert docs[0]["label"] == "DOI: 10.1128/AAC.49.5.1727-1732.2005"
        assert docs[0]["title"] == "Immunomodulatory activities of small host defense peptides."


class TestActivityOrigin:
    """A measurement with no paper of its own still has a known provenance."""

    @pytest.mark.parametrize("marker,expected", [
        ("DBAASP database record", "DBAASP"),
        ("CAMP database record", "CAMP"),
        ("DRAMP database record", "DRAMP"),
    ])
    def test_database_markers_are_named(self, marker: str, expected: str) -> None:
        from sample.db import _activity_origin

        assert _activity_origin({"mechanism_of_action": marker,
                                 "source_accession": "X"}) == expected

    def test_rows_extracted_here_are_named_as_such(self) -> None:
        from sample.db import _activity_origin

        assert _activity_origin({"mechanism_of_action": None,
                                 "source_accession": None}) == "SAMPLE extraction"

    def test_a_real_mechanism_is_not_treated_as_provenance(self) -> None:
        from sample.db import _activity_origin

        assert _activity_origin({"mechanism_of_action": "Membrane disruption",
                                 "source_accession": "X"}) is None


class TestOptionalColumns:
    """
    A database built by an older loader must still serve every query.

    Adding `Reference` to the select list broke search outright with `no such column: p.Reference`
    — the same shape as the name subquery that once broke search against a database predating the
    names step.
    """

    def test_select_list_adapts_to_the_table(self, tmp_path: Any) -> None:
        from sample.db import _peptide_select

        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE peptides (entry_id TEXT, sequence TEXT, is_decoy INTEGER)")
        conn.commit()

        selected = _peptide_select(conn)
        assert "p.Reference" not in selected
        assert "p.entry_id" in selected and "p.sequence" in selected
        conn.close()

    def test_search_survives_a_database_without_reference(self, tmp_path: Any,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
        from sample import db

        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE peptides (entry_id TEXT, sequence TEXT, SMILES TEXT, "
                     "source TEXT, is_decoy INTEGER, LITERATURE TEXT, PATENT TEXT)")
        conn.execute("CREATE TABLE activities (entry_id TEXT, activity_group TEXT, value REAL, "
                     "unit TEXT, target_organism TEXT, activity_type TEXT, "
                     "mechanism_of_action TEXT)")
        conn.execute("CREATE TABLE peptide_references (peptide_id TEXT, paper_id INTEGER)")
        conn.execute("CREATE TABLE papers (id INTEGER, source_type TEXT, source_id TEXT)")
        conn.execute("INSERT INTO peptides VALUES ('E1', 'GIGKFLK', NULL, 'ADP', 0, NULL, NULL)")
        conn.commit()
        conn.close()

        monkeypatch.setattr(db, "default_db_path", lambda: path)
        rows, total = db.search_peptides(query_str="GIGKFLK", db_path=path)

        assert total == 1
        assert rows[0]["entry_id"] == "E1"
        assert rows[0]["Reference"] is None


class TestGetPeptideDegrades:
    """
    `get_peptide` built its select list straight from `PEPTIDE_COLUMNS`, which is the one call
    site that missed the schema-degradation fix: adding `source_organism` broke it with
    `no such column: source_organism` against a database built before that column existed.
    """

    def test_detail_view_survives_a_database_without_source_organism(
            self, tmp_path: Any) -> None:
        from sample import db

        path = str(tmp_path / "old.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE peptides (entry_id TEXT, sequence TEXT, SMILES TEXT, "
                     "source TEXT, is_decoy INTEGER, LITERATURE TEXT, PATENT TEXT)")
        conn.execute("CREATE TABLE descriptors (entry_id TEXT)")
        conn.execute("CREATE TABLE activities (entry_id TEXT)")
        conn.execute("CREATE TABLE peptide_references (peptide_id TEXT, paper_id INTEGER)")
        conn.execute("CREATE TABLE papers (id INTEGER, source_type TEXT, source_id TEXT)")
        conn.execute("INSERT INTO peptides VALUES ('E1', 'GIGKFLK', NULL, 'ADP', 0, NULL, NULL)")
        conn.commit()
        conn.close()

        peptide = db.get_peptide("E1", db_path=path)
        assert peptide is not None
        assert peptide["entry_id"] == "E1"
        assert peptide.get("source_organism") is None


class TestBiologicalActivitiesFiltering:
    """Verifies that Mechanism of Action is excluded and undefined targets are omitted."""

    def test_activities_without_target_organism_are_excluded(self) -> None:
        from fastapi.testclient import TestClient
        from sample.api.app import app
        from bs4 import BeautifulSoup

        client = TestClient(app)
        resp = client.get("/search/peptide/SAMPLE0000000001")
        assert resp.status_code == 200
        html = resp.text

        # Verify Mechanism of Action is absent from headers
        assert "<th>Mechanism of Action</th>" not in html
        assert "Mechanism of Action" not in html

        # Verify all rendered biological activity rows have non-empty target organism
        soup = BeautifulSoup(html, "html.parser")
        for tr in soup.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 6:  # Biological activities row has 6 columns (no Mechanism column)
                target_text = tds[1].get_text(strip=True)
                assert target_text != ""
                assert target_text.lower() not in ("none", "nan", "n/a", "-")

