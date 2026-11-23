"""
Tests for `sample.gene_ontology`'s mapping onto the real Gene Ontology `.obo`.

The feature slices `go-basic.obo` down to the GO terms this project's peptides actually carry
(harvested from CAMP, see `TestGeneOntology` in `tests/test_families_synergy.py` for the harvest
side) plus their `is_a`/`part_of` ancestors, so the result is a connected sub-hierarchy of the real
Gene Ontology rather than an invented vocabulary. These tests build a small, synthetic `.obo`
fixture instead of the real ~32 MB `go-basic.obo` -- the suite runs offline (see `conftest.py`'s
`no_network`) and does not depend on a multi-megabyte download being present on disk.

Real-data verification (1,600 harvested GO ids, 0 unresolved, 3,686 terms in the resulting slice
after pulling in ancestors) is recorded in `DECISIONS.md` rather than re-run here.
"""
import json
import os
from typing import Any, Dict

import pytest

#: A tiny, self-contained OBO fixture with the shapes parse_obo/build_go_slice must handle: a
#: multi-level is_a chain, a part_of relationship, an alt_id, and an obsolete term.
_FIXTURE_OBO = """format-version: 1.2
ontology: fixture

[Term]
id: GO:0000001
name: root process
namespace: biological_process

[Term]
id: GO:0000002
name: child process
namespace: biological_process
is_a: GO:0000001 ! root process

[Term]
id: GO:0000003
name: grandchild process
namespace: biological_process
is_a: GO:0000002 ! child process

[Term]
id: GO:0000004
name: component whole
namespace: cellular_component

[Term]
id: GO:0000005
name: component part
namespace: cellular_component
relationship: part_of GO:0000004 ! component whole

[Term]
id: GO:0000006
name: renamed term
namespace: molecular_function
alt_id: GO:0000099

[Term]
id: GO:0000007
name: obsolete term
namespace: molecular_function
is_obsolete: true

[Typedef]
id: part_of
name: part of
"""


@pytest.fixture
def obo_fixture_path(tmp_path: Any) -> str:
    path = tmp_path / "fixture.obo"
    path.write_text(_FIXTURE_OBO, encoding="utf-8")
    return str(path)


def _write_gene_ontology_json(tmp_path: Any, terms_by_sequence: Dict[str, Any]) -> str:
    path = tmp_path / "gene_ontology.json"
    path.write_text(json.dumps({"gene_ontology": terms_by_sequence}), encoding="utf-8")
    return str(path)


class TestParseObo:
    def test_reads_id_name_namespace(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import parse_obo

        terms = parse_obo(obo_fixture_path)
        assert terms["GO:0000001"]["name"] == "root process"
        assert terms["GO:0000001"]["namespace"] == "biological_process"

    def test_is_a_strips_the_trailing_comment(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import parse_obo

        terms = parse_obo(obo_fixture_path)
        assert terms["GO:0000002"]["is_a"] == ["GO:0000001"]

    def test_relationship_part_of_is_captured_separately_from_is_a(
            self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import parse_obo

        terms = parse_obo(obo_fixture_path)
        assert terms["GO:0000005"]["part_of"] == ["GO:0000004"]
        assert terms["GO:0000005"]["is_a"] == []

    def test_alt_id_is_recorded_on_the_primary_term(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import parse_obo

        terms = parse_obo(obo_fixture_path)
        assert terms["GO:0000006"]["alt_id"] == ["GO:0000099"]
        assert "GO:0000099" not in terms, "an alt_id is not itself a resolvable primary term"

    def test_is_obsolete_is_read(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import parse_obo

        terms = parse_obo(obo_fixture_path)
        assert terms["GO:0000007"]["is_obsolete"] is True
        assert terms["GO:0000001"]["is_obsolete"] is False

    def test_typedef_stanza_is_not_mistaken_for_a_term(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import parse_obo

        terms = parse_obo(obo_fixture_path)
        assert "part_of" not in terms


class TestAncestors:
    def test_walks_the_full_is_a_chain(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import _ancestors, parse_obo

        terms = parse_obo(obo_fixture_path)
        assert _ancestors("GO:0000003", terms) == {"GO:0000002", "GO:0000001"}

    def test_root_term_has_no_ancestors(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import _ancestors, parse_obo

        terms = parse_obo(obo_fixture_path)
        assert _ancestors("GO:0000001", terms) == set()

    def test_part_of_counts_as_an_ancestor_edge(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import _ancestors, parse_obo

        terms = parse_obo(obo_fixture_path)
        assert _ancestors("GO:0000005", terms) == {"GO:0000004"}

    def test_unknown_id_returns_empty_rather_than_raising(self, obo_fixture_path: str) -> None:
        from sample.gene_ontology import _ancestors, parse_obo

        terms = parse_obo(obo_fixture_path)
        assert _ancestors("GO:9999999", terms) == set()


class TestBuildGoSlice:
    def test_used_terms_and_their_ancestors_are_both_included(
            self, tmp_path: Any, obo_fixture_path: str) -> None:
        from sample.gene_ontology import build_go_slice

        gene_ontology_json = _write_gene_ontology_json(tmp_path, {
            "SEQ1": [{"go_id": "GO:0000003", "category": "Biological process",
                     "term": "grandchild process", "evidence": "IEA"}],
        })
        output_path = str(tmp_path / "slice.obo")

        summary = build_go_slice(gene_ontology_json=gene_ontology_json,
                                 go_obo_path=obo_fixture_path, output_path=output_path)

        assert summary["used_terms"] == 1
        assert summary["ancestor_terms"] == 2  # GO:0000002 and GO:0000001
        assert summary["total_terms_in_slice"] == 3
        assert summary["unresolved"] == 0

        text = open(output_path, encoding="utf-8").read()
        for go_id in ("GO:0000001", "GO:0000002", "GO:0000003"):
            assert f"id: {go_id}" in text

    def test_only_the_directly_used_term_carries_a_usage_comment(
            self, tmp_path: Any, obo_fixture_path: str) -> None:
        from sample.gene_ontology import build_go_slice

        gene_ontology_json = _write_gene_ontology_json(tmp_path, {
            "SEQ1": [{"go_id": "GO:0000003", "category": None, "term": "x", "evidence": "IEA"}],
            "SEQ2": [{"go_id": "GO:0000003", "category": None, "term": "x", "evidence": "IDA"}],
        })
        output_path = str(tmp_path / "slice.obo")

        build_go_slice(gene_ontology_json=gene_ontology_json, go_obo_path=obo_fixture_path,
                       output_path=output_path)

        text = open(output_path, encoding="utf-8").read()
        term_block = text.split("id: GO:0000003")[1].split("[Term]")[0]
        assert "2 SAMPLE peptide annotation(s)" in term_block
        ancestor_block = text.split("id: GO:0000001")[1].split("[Term]")[0]
        assert "comment:" not in ancestor_block

    def test_obsolete_term_is_included_and_marked_not_dropped(
            self, tmp_path: Any, obo_fixture_path: str) -> None:
        """GO evolves; some of what CAMP's old scrape annotated has since been retired. An
        obsolete term must still appear (it is a real fact about what CAMP recorded) but marked,
        matching go-basic.obo's own `is_obsolete: true` convention -- not silently presented as a
        live, current classification. Found by checking real data: 102 of 1,600 harvested ids in
        the actual corpus resolve to obsolete terms (see DECISIONS.md)."""
        from sample.gene_ontology import build_go_slice

        gene_ontology_json = _write_gene_ontology_json(tmp_path, {
            "SEQ1": [{"go_id": "GO:0000007", "category": None, "term": "obsolete term",
                     "evidence": "IEA"}],
        })
        output_path = str(tmp_path / "slice.obo")

        summary = build_go_slice(gene_ontology_json=gene_ontology_json,
                                 go_obo_path=obo_fixture_path, output_path=output_path)

        assert summary["obsolete_used_terms"] == 1
        assert summary["unresolved"] == 0, "obsolete is not the same as absent from go-basic.obo"

        text = open(output_path, encoding="utf-8").read()
        term_block = text.split("id: GO:0000007")[1].split("[Term]")[0]
        assert "is_obsolete: true" in term_block

    def test_alt_id_resolves_to_the_primary_term(
            self, tmp_path: Any, obo_fixture_path: str) -> None:
        from sample.gene_ontology import build_go_slice

        # GO:0000099 is an alt_id of GO:0000006 in the fixture -- simulates CAMP having scraped a
        # since-merged accession, which real GO ids do accumulate over time.
        gene_ontology_json = _write_gene_ontology_json(tmp_path, {
            "SEQ1": [{"go_id": "GO:0000099", "category": None, "term": "x", "evidence": "IEA"}],
        })
        output_path = str(tmp_path / "slice.obo")

        summary = build_go_slice(gene_ontology_json=gene_ontology_json,
                                 go_obo_path=obo_fixture_path, output_path=output_path)

        assert summary["resolved_via_alt_id"] == 1
        assert summary["unresolved"] == 0
        text = open(output_path, encoding="utf-8").read()
        assert "id: GO:0000006" in text
        assert "id: GO:0000099" not in text, "the slice keys terms by primary id, not alt_id"

    def test_unresolvable_id_is_counted_not_silently_dropped(
            self, tmp_path: Any, obo_fixture_path: str) -> None:
        from sample.gene_ontology import build_go_slice

        gene_ontology_json = _write_gene_ontology_json(tmp_path, {
            "SEQ1": [{"go_id": "GO:9999999", "category": None, "term": "x", "evidence": "IEA"}],
        })
        output_path = str(tmp_path / "slice.obo")

        summary = build_go_slice(gene_ontology_json=gene_ontology_json,
                                 go_obo_path=obo_fixture_path, output_path=output_path)

        assert summary["unresolved"] == 1
        assert summary["used_terms"] == 0

    def test_missing_harvested_file_returns_a_zeroed_summary_not_an_exception(
            self, tmp_path: Any) -> None:
        from sample.gene_ontology import build_go_slice

        summary = build_go_slice(gene_ontology_json=str(tmp_path / "absent.json"),
                                 output_path=str(tmp_path / "slice.obo"))
        assert summary["used_terms"] == 0
        assert not (tmp_path / "slice.obo").exists()

    def test_missing_obo_and_download_disabled_returns_a_zeroed_summary(
            self, tmp_path: Any) -> None:
        from sample.gene_ontology import build_go_slice

        gene_ontology_json = _write_gene_ontology_json(tmp_path, {
            "SEQ1": [{"go_id": "GO:0000003", "category": None, "term": "x", "evidence": "IEA"}],
        })
        summary = build_go_slice(gene_ontology_json=gene_ontology_json,
                                 go_obo_path=str(tmp_path / "absent.obo"),
                                 output_path=str(tmp_path / "slice.obo"),
                                 download_if_missing=False)
        assert summary["used_terms"] == 0


class TestDownloadGoObo:
    def test_skips_download_when_the_file_already_exists(self, tmp_path: Any) -> None:
        """Must not even attempt a network call (the suite runs with sockets blocked) when the
        file is already on disk and force is not set."""
        from sample.gene_ontology import download_go_obo

        existing = tmp_path / "go-basic.obo"
        existing.write_text("format-version: 1.2\n", encoding="utf-8")

        result = download_go_obo(output_path=str(existing))
        assert result == str(existing)

    def test_a_failed_download_returns_empty_string_not_a_partial_file(
            self, tmp_path: Any) -> None:
        from sample.gene_ontology import download_go_obo

        # no_network (conftest.py) blocks the underlying socket call, so this exercises the
        # real failure path exactly as a DNS failure or a dropped connection would.
        output_path = str(tmp_path / "go-basic.obo")
        result = download_go_obo(output_path=output_path)

        assert result == ""
        assert not (tmp_path / (os.path.basename(output_path) + ".part")).exists()
