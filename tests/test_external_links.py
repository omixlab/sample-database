"""
Tests for :mod:`sample.external_links` — identifier validation and URL resolution.

The validation half matters more than the linking half. `cross_references` is an unvalidated
passthrough of whatever the source databases wrote, and 59% of its tokens are not identifiers at
all: sentinels ("Not available"), prose ("1MAG resolved by NMR."), and stringified Python dicts.
Storing those produced a column nothing could join on. These tests pin what survives and what
does not — and the second is the important one, because a wrong accession points at a real but
different molecule, which is worse than no accession.
"""
from typing import Any, List

import pytest

from sample import external_links as E


class TestNormalizeIdentifier:
    @pytest.mark.parametrize("raw,expected", [
        ("P81941", ["P81941"]),
        ("Q9PRJ9", ["Q9PRJ9"]),
        ("P81941, P81619, Q9PRJ9", ["P81941", "P81619", "Q9PRJ9"]),
        ("  P81941  ", ["P81941"]),
    ])
    def test_accepts_valid_uniprot(self, raw: str, expected: List[str]) -> None:
        assert E.normalize_identifier("uniprot", raw) == expected

    @pytest.mark.parametrize("raw", [
        "Not available", "No entry found", "Reference ID: Ref", "", "  ", None,
    ])
    def test_rejects_uniprot_sentinels(self, raw: Any) -> None:
        assert E.normalize_identifier("uniprot", raw) == []

    def test_rejects_a_pdb_id_filed_under_uniprot(self) -> None:
        # "PDB ID: 1MAG Go to PDB" appears in the uniprot column. It is a real identifier in the
        # wrong namespace; emitting it as a UniProt accession would link to nothing.
        assert E.normalize_identifier("uniprot", "PDB ID: 1MAG Go to PDB") == []

    @pytest.mark.parametrize("raw,expected", [
        ("1AV2", ["1AV2"]),
        ("1MAG resolved by NMR.", ["1MAG"]),
        ("2n5y", ["2N5Y"]),
    ])
    def test_pdb_extraction_and_casing(self, raw: str, expected: List[str]) -> None:
        assert E.normalize_identifier("pdb", raw) == expected

    def test_pdb_rejects_sentinels(self) -> None:
        assert E.normalize_identifier("pdb", "Not available") == []

    def test_pubchem_from_stringified_dict(self) -> None:
        # DBAASP stores a repr() of a Python dict in this field.
        raw = "{'cid': '46891554', 'cidUrl': 'https://pubchem.ncbi.nlm.nih.gov/compound/46891554'}"
        assert E.normalize_identifier("pubchem", raw) == ["46891554"]

    def test_pubchem_from_bare_number(self) -> None:
        assert E.normalize_identifier("pubchem", "139589447") == ["139589447"]

    def test_chembl(self) -> None:
        assert E.normalize_identifier("chembl", "CHEMBL4300557") == ["CHEMBL4300557"]

    def test_unknown_namespace_passes_through(self) -> None:
        # Refusing to guess: an unrecognised namespace is not silently dropped, because the
        # value may be perfectly good and simply from a source not yet modelled.
        assert E.normalize_identifier("something-new", "XYZ123") == ["XYZ123"]

    def test_deduplicates(self) -> None:
        assert E.normalize_identifier("uniprot", "P81941, P81941") == ["P81941"]


class TestNormalizeCrossReferences:
    def test_parses_a_mixed_blob(self) -> None:
        blob = ("uniprot:P81941 | pdb:Not available | "
                "pubchem:{'cid': '46891554', 'cidUrl': 'x'}")
        assert E.normalize_cross_references(blob) == {
            "uniprot": ["P81941"], "pubchem": ["46891554"]}

    def test_namespace_with_nothing_valid_is_omitted(self) -> None:
        assert E.normalize_cross_references("pdb:Not available") == {}

    @pytest.mark.parametrize("blob", ["", None, "no-namespace-token"])
    def test_degenerate_input(self, blob: Any) -> None:
        assert E.normalize_cross_references(blob) == {}


class TestBuildUrl:
    @pytest.mark.parametrize("namespace,identifier,fragment", [
        ("pubmed", "12345678", "pubmed.ncbi.nlm.nih.gov/12345678"),
        ("doi", "10.1038/nature12373", "doi.org/10.1038/nature12373"),
        ("patent", "US8394762B2", "patents.google.com/patent/US8394762B2"),
        ("uniprot", "P81941", "uniprot.org/uniprotkb/P81941"),
        ("pdb", "1AV2", "rcsb.org/structure/1AV2"),
        ("pubchem", "46891554", "pubchem.ncbi.nlm.nih.gov/compound/46891554"),
        ("chembl", "CHEMBL4300557", "ebi.ac.uk/chembl"),
        ("taxonomy", "562", "wwwtax.cgi?id=562"),
        ("dbaasp", "DBAASPN_10055", "dbaasp.org/peptide-card?id=DBAASPN_10055"),
    ])
    def test_resolves(self, namespace: str, identifier: str, fragment: str) -> None:
        url = E.build_url(namespace, identifier)
        assert url and fragment in url

    def test_pubmed_tolerates_a_prefixed_id(self) -> None:
        assert E.build_url("pubmed", "PMID: 12345678").endswith("/12345678/")

    def test_a_pmid_that_travelled_through_a_float_is_not_corrupted(self) -> None:
        """
        pandas types an integer column holding any NaN as float64, so every PMID in the shipped
        release is stored as "10075760.0". Stripping non-digits alone removes the dot but keeps
        its trailing zero, yielding 100757600 -- a different, nonexistent article. The failure is
        silent: the link is well-formed, verified-looking, and points at the wrong record.
        """
        assert E.build_url("pubmed", "10075760.0").endswith("/10075760/")
        assert E.build_url("pubmed", "9843665.0").endswith("/9843665/")
        assert "100757600" not in E.build_url("pubmed", "10075760.0")

    def test_doi_extracted_from_surrounding_text(self) -> None:
        assert "10.1038/nature12373" in E.build_url("doi", "see doi:10.1038/nature12373 for more")

    @pytest.mark.parametrize("namespace,identifier", [
        ("pubmed", "not-a-number"), ("pubmed", ""), ("doi", None), ("unknown-ns", "X"),
    ])
    def test_unresolvable_is_none(self, namespace: str, identifier: Any) -> None:
        assert E.build_url(namespace, identifier) is None

    def test_known_database_without_a_pattern_falls_back_to_its_home_page(self) -> None:
        # Reaching the right database beats reaching nothing.
        assert E.build_url("dbamp", "dbAMP_00001")


class TestDescribe:
    def test_marks_verified_patterns(self) -> None:
        # `verified` means a known accession was resolved against the live resource and the right
        # record came back -- not merely that the URL follows a documented scheme.
        assert E.describe("dbaasp", "DBAASPN_1")["verified"] is True
        assert E.describe("camp", "CAMPSQ1")["verified"] is True

    def test_unknown_namespace_is_not_claimed_as_verified(self) -> None:
        assert E.describe("some-new-db", "X1")["verified"] is False

    def test_carries_a_label_and_url(self) -> None:
        record = E.describe("uniprot", "P81941")
        assert record["label"] == "UniProt"
        assert record["url"].endswith("/P81941/entry")

    def test_float_formatted_pmid_is_shown_and_linked_as_the_same_id(self) -> None:
        """The page prints the accession next to the link; if only the URL were cleaned it would
        read "PubMed: 10075760.0" beside a link to 10075760."""
        record = E.describe("pubmed", "10075760.0")
        assert record["source_id"] == "10075760"
        assert record["url"].endswith("/10075760/")

    def test_unresolvable_keeps_the_identifier_visible(self) -> None:
        record = E.describe("pubmed", "not-a-number")
        assert record["url"] is None
        assert record["source_id"] == "not-a-number"


class TestLinksFromReferenceTokens:
    def test_extracts_a_patent_number(self) -> None:
        links = E.links_from_reference_tokens("google_patents:US8394762B2.json")
        assert [l["source_id"] for l in links] == ["US8394762B2"]

    def test_ignores_literature_uuids(self) -> None:
        # Those are internal extraction ids, not public identifiers.
        assert E.links_from_reference_tokens("pubmed:0aba52bd-1234.json") == []

    def test_deduplicates_across_tokens(self) -> None:
        blob = "google_patents:US8394762B2.json;google_patents:US8394762B2.json"
        assert len(E.links_from_reference_tokens(blob)) == 1

    @pytest.mark.parametrize("blob", ["", None])
    def test_degenerate_input(self, blob: Any) -> None:
        assert E.links_from_reference_tokens(blob) == []


class TestFormTarget:
    """
    APD6 retired its addressable entry URLs. The stored template pointed at the APD2 path, which
    404s for every accession, so every ADP citation in the release was a dead link. An accession
    is now reached by submitting the site's search form.
    """

    def test_builds_the_apd_form(self) -> None:
        form = E.form_target("ADP", "AP00001")
        assert form["action"] == "https://aps.unmc.edu/database/result"
        # The form wants the digits without the AP prefix, padding intact.
        assert form["fields"]["ID"] == "00001"

    def test_carries_the_required_defaults(self) -> None:
        # Posting ID alone, or with these blank, returns "No Results Found" while still echoing
        # the id back -- which reads as a missing record rather than a malformed query.
        fields = E.form_target("ADP", "AP00001")["fields"]
        assert fields["Sort"] == "ID"
        for name in ("HydrophobicPer", "Location", "Method", "Netcharge", "Type"):
            assert fields[name] == "Any", name

    def test_is_none_for_get_namespaces(self) -> None:
        assert E.form_target("dbaasp", "DBAASPN_10055") is None

    @pytest.mark.parametrize("accession", ["", None, "not-an-adp-id"])
    def test_rejects_malformed_accessions(self, accession: Any) -> None:
        assert E.form_target("ADP", accession) is None

    def test_describe_exposes_the_form(self) -> None:
        assert E.describe("ADP", "AP00001")["form"]["fields"]["ID"] == "00001"


class TestLinkSpecTable:
    def test_every_get_template_has_the_id_placeholder(self) -> None:
        # A POST spec carries the identifier in the body, so its template is a bare form action.
        for namespace, spec in E.LINK_SPECS.items():
            if spec.method == "GET":
                assert "{id}" in spec.template, namespace

    def test_post_specs_declare_a_form_field(self) -> None:
        for namespace, spec in E.LINK_SPECS.items():
            if spec.method == "POST":
                assert spec.form_field, namespace
                assert "{id}" not in spec.template, namespace

    def test_every_spec_has_a_label(self) -> None:
        for namespace, spec in E.LINK_SPECS.items():
            assert spec.label.strip(), namespace

    def test_home_pages_cover_the_source_databases(self) -> None:
        assert {"camp", "adp", "dramp", "dbamp", "dbaasp"} <= set(E.HOME_PAGES)
