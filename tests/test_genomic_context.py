"""
Tests for :mod:`sample.genomic_context` -- tBLASTn hit parsing, the near-exact filter, taxonomic
aggregation, the Krona sunburst, and the download loop's delete-as-you-go accounting.

The load-bearing rule is the *conjunction*: a hit survives only when it clears BOTH coverage and
identity. A fragment (high identity, low coverage) and a lookalike (full coverage, low identity)
must each be dropped, because either alone would turn "this peptide's gene is present here" into a
claim the alignment does not support. The build-loop tests assert the genome file is always deleted
-- the whole reason the loop exists rather than a bulk download.
"""
import os
from typing import Any, Dict, List, Optional

import pytest

from sample import genomic_context as gc


def _line(qseqid: str, sseqid: str, pident: float, length: int, qcovhsp: float,
          evalue: str = "1e-20", bitscore: float = 80.0, sstart: int = 1, send: int = 60) -> str:
    return "\t".join(str(x) for x in
                     (qseqid, sseqid, pident, length, qcovhsp, evalue, bitscore, sstart, send))


class TestParseTblastn:
    def test_parses_percentages_into_fractions(self) -> None:
        hits = gc.parse_tblastn(_line("PEP1", "ctg", 96.0, 20, 98.0))
        assert len(hits) == 1
        assert hits[0].identity == pytest.approx(0.96)
        assert hits[0].coverage == pytest.approx(0.98)
        assert hits[0].peptide_id == "PEP1" and hits[0].contig == "ctg"

    def test_skips_malformed_and_comment_lines(self) -> None:
        text = "\n".join([_line("PEP1", "ctg", 100.0, 20, 100.0), "# a comment",
                          "not enough columns", ""])
        assert len(gc.parse_tblastn(text)) == 1


class TestThresholds:
    def test_keeps_a_near_exact_full_length_hit(self) -> None:
        hit = gc.parse_tblastn(_line("P", "c", 100.0, 20, 100.0))[0]
        assert gc.passes_thresholds(hit) is True

    def test_drops_a_high_identity_fragment(self) -> None:
        # 100% identical but only 50% of the peptide covered -- a motif match, not the gene.
        hit = gc.parse_tblastn(_line("P", "c", 100.0, 10, 50.0))[0]
        assert gc.passes_thresholds(hit) is False

    def test_drops_a_full_coverage_lookalike(self) -> None:
        # Full length but 70% identity -- a different peptide.
        hit = gc.parse_tblastn(_line("P", "c", 70.0, 20, 100.0))[0]
        assert gc.passes_thresholds(hit) is False

    def test_filter_keeps_only_survivors(self) -> None:
        text = "\n".join([
            _line("P", "keep1", 100.0, 20, 100.0),
            _line("P", "keep2", 95.0, 20, 95.0),   # exactly on both thresholds
            _line("P", "frag", 100.0, 10, 50.0),
            _line("P", "look", 70.0, 20, 100.0),
        ])
        kept = {h.contig for h in gc.filter_hits(gc.parse_tblastn(text))}
        assert kept == {"keep1", "keep2"}


class TestGenomeLineage:
    def test_binomial_strips_strain_noise(self) -> None:
        lineage = gc.genome_lineage(
            "562", "Escherichia coli str. K-12 substr. MG1655", {"562": "bacteria"})
        assert lineage == ["bacteria", "Escherichia", "Escherichia coli"]

    def test_unknown_domain_when_taxid_absent(self) -> None:
        lineage = gc.genome_lineage("999999", "Genus species", {})
        assert lineage[0] == "unknown"

    def test_genus_only_when_no_species(self) -> None:
        assert gc.genome_lineage("1", "Escherichia", {"1": "bacteria"}) == ["bacteria", "Escherichia"]


class TestKronaHierarchy:
    def _hits(self) -> List[gc.Hit]:
        base = gc.parse_tblastn(_line("P", "c", 100.0, 20, 100.0))[0]
        return [
            base._replace(taxid="562", organism="Escherichia coli"),
            base._replace(taxid="562", organism="Escherichia coli"),
            base._replace(taxid="1280", organism="Staphylococcus aureus"),
        ]

    def test_counts_nest_parent_equals_sum_of_children(self) -> None:
        h = gc.krona_hierarchy(self._hits(), {"562": "bacteria", "1280": "bacteria"})
        assert h["count"] == 3
        bacteria = h["children"]["bacteria"]
        assert bacteria["count"] == 3
        assert bacteria["children"]["Escherichia"]["count"] == 2
        assert bacteria["children"]["Staphylococcus"]["count"] == 1

    def test_empty_hits_give_an_empty_root(self) -> None:
        h = gc.krona_hierarchy([], {})
        assert h["count"] == 0 and h["children"] == {}


class TestKronaSvg:
    def test_renders_a_sunburst_for_hits(self) -> None:
        h = gc.krona_hierarchy(
            gc.parse_tblastn(_line("P", "c", 100.0, 20, 100.0)), {})
        svg = gc.krona_svg(h)
        assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
        assert "<path" in svg

    def test_empty_hierarchy_renders_a_note_not_a_crash(self) -> None:
        svg = gc.krona_svg({"count": 0, "children": {}})
        assert "no genomic hits" in svg
        assert "<path" not in svg


class TestParseAssemblySummary:
    # A trimmed assembly_summary.txt: 20 tab-separated columns, two data rows of differing category.
    def _row(self, acc: str, category: str, taxid: str, organism: str, ftp: str) -> str:
        cols = [""] * 20
        cols[0], cols[4], cols[5], cols[7], cols[19] = acc, category, taxid, organism, ftp
        return "\t".join(cols)

    def test_keeps_reference_and_representative_only(self) -> None:
        text = "\n".join([
            "# assembly_summary",
            "#assembly_accession\t...",
            self._row("GCF_1", "reference genome", "562", "Escherichia coli",
                      "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2"),
            self._row("GCF_2", "representative genome", "1280", "Staphylococcus aureus",
                      "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/013/425/GCF_000013425.1_ASM1342v1"),
            self._row("GCF_3", "na", "9606", "Homo sapiens", "https://example/GCF_x"),
        ])
        records = gc.parse_assembly_summary(text)
        accessions = {r["accession"] for r in records}
        assert accessions == {"GCF_1", "GCF_2"}  # the "na"-category row is dropped

    def test_builds_the_https_genomic_fasta_url(self) -> None:
        text = self._row(
            "GCF_1", "reference genome", "562", "Escherichia coli",
            "ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2")
        record = gc.parse_assembly_summary(text)[0]
        # ftp -> https, and NCBI's <basename>_genomic.fna.gz convention.
        assert record["url"] == (
            "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/"
            "GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz")
        assert record["taxid"] == "562" and record["organism"] == "Escherichia coli"

    def test_empty_categories_keeps_every_data_row(self) -> None:
        text = self._row("GCF_9", "na", "1", "Some organism", "https://x/GCF_9_ASM")
        assert len(gc.parse_assembly_summary(text, categories=())) == 1


class TestBuildContext:
    def test_deletes_each_genome_and_accounts_correctly(self, tmp_path: Any,
                                                        monkeypatch: Any) -> None:
        # A fake download that writes a placeholder file, and a fake search that "finds" a hit for
        # one genome and nothing for the other -- so the loop's accounting and cleanup are exercised
        # without BLAST or the network.
        created: List[str] = []

        def fake_download(record: Dict[str, str], dest: str) -> Optional[str]:
            path = os.path.join(dest, f"{record['accession']}.fna")
            with open(path, "w") as fh:
                fh.write(">c\nACGT\n")
            created.append(path)
            return path

        def fake_search(peptide_fasta: str, genome_fasta: str, work_dir: str, **kw: Any):
            if "GCF_HIT" in genome_fasta:
                base = gc.parse_tblastn(_line("P", "c", 100.0, 20, 100.0))[0]
                return [base._replace(genome_accession=kw.get("genome_accession", ""),
                                      taxid=kw.get("taxid", ""), organism=kw.get("organism", ""))]
            return []

        monkeypatch.setattr(gc, "search_genome", fake_search)

        genomes = [
            {"accession": "GCF_HIT", "taxid": "562", "organism": "Escherichia coli"},
            {"accession": "GCF_MISS", "taxid": "1280", "organism": "Staphylococcus aureus"},
        ]
        result = gc.build_context("pep.faa", genomes, fake_download, str(tmp_path),
                                  domain_lookup={"562": "bacteria"})

        assert result["genomes_searched"] == 2
        assert result["genomes_with_hits"] == 1
        assert len(result["hits"]) == 1
        assert result["krona"]["count"] == 1
        # Every downloaded genome file was deleted -- the point of the loop.
        assert all(not os.path.exists(p) for p in created)

    def test_skips_a_genome_that_fails_to_download(self, tmp_path: Any, monkeypatch: Any) -> None:
        monkeypatch.setattr(gc, "search_genome", lambda *a, **k: [])
        result = gc.build_context(
            "pep.faa", [{"accession": "X"}], lambda rec, dest: None, str(tmp_path))
        assert result["genomes_searched"] == 0
        assert result["hits"] == []
