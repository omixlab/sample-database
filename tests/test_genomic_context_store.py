"""
Tests for :mod:`sample.genomic_context_store` and the resumable, store-backed sweep.

The properties that matter for a sweep over thousands of genomes: a genome's hits and its
processed-marker are written together (so a resumed sweep never double-counts or drops one), a
zero-hit genome still counts as *searched* (so it is not re-downloaded), each genome FASTA is
deleted after it is searched (so only one sits on disk), and ``--no-resume`` re-searches everything.
The BLAST subprocess is mocked, so these run without makeblastdb/tblastn or any download.
"""
import os
from typing import Any, Dict, List, Optional

import pytest

from sample import genomic_context_store as store
from sample.genomic_context import Hit


def _hit(accession: str, identity: float = 0.98) -> Hit:
    return Hit(peptide_id="P1", contig="c1", identity=identity, coverage=0.99, evalue=1e-9,
               bitscore=120.0, sstart=1, send=30, genome_accession=accession, taxid="562",
               organism="Escherichia coli")


class TestStore:
    def test_record_and_read_back(self, tmp_path: Any) -> None:
        db = str(tmp_path / "g.sqlite")
        store.init_db(db)
        store.record_genome("P1", "GCF_1", [_hit("GCF_1"), _hit("GCF_1", 0.97)], db_path=db)
        hits = store.get_hits("P1", db_path=db)
        assert len(hits) == 2
        assert {h.genome_accession for h in hits} == {"GCF_1"}
        assert hits[0].organism == "Escherichia coli"

    def test_processed_accessions_supports_resume(self, tmp_path: Any) -> None:
        db = str(tmp_path / "g.sqlite")
        store.init_db(db)
        store.record_genome("P1", "GCF_1", [_hit("GCF_1")], db_path=db)
        store.record_genome("P1", "GCF_2", [], db_path=db)   # searched, no hits
        assert store.processed_accessions("P1", db_path=db) == {"GCF_1", "GCF_2"}

    def test_summary_counts_zero_hit_genomes_as_searched(self, tmp_path: Any) -> None:
        db = str(tmp_path / "g.sqlite")
        store.init_db(db)
        store.record_genome("P1", "GCF_1", [_hit("GCF_1")], db_path=db)
        store.record_genome("P1", "GCF_2", [], db_path=db)
        s = store.summary("P1", db_path=db)
        assert s == {"genomes_searched": 2, "genomes_with_hits": 1, "total_hits": 1}

    def test_missing_db_is_empty_not_an_error(self, tmp_path: Any) -> None:
        db = str(tmp_path / "absent.sqlite")
        assert store.get_hits("P1", db_path=db) == []
        assert store.processed_accessions("P1", db_path=db) == set()
        assert store.summary("P1", db_path=db)["genomes_searched"] == 0

    def test_clear_peptide(self, tmp_path: Any) -> None:
        db = str(tmp_path / "g.sqlite")
        store.init_db(db)
        store.record_genome("P1", "GCF_1", [_hit("GCF_1")], db_path=db)
        store.clear_peptide("P1", db_path=db)
        assert store.get_hits("P1", db_path=db) == []
        assert store.processed_accessions("P1", db_path=db) == set()


def _fake_download(record: Dict[str, str], dest_dir: str) -> Optional[str]:
    """Writes a dummy genome FASTA into dest_dir and returns its path (no network)."""
    path = os.path.join(dest_dir, f"{record['accession']}.fna")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(">contig\nACGTACGTACGT\n")
    return path


class TestSweepToStore:
    @pytest.fixture
    def genomes(self) -> List[Dict[str, str]]:
        return [{"accession": "GCF_1", "taxid": "562", "organism": "Escherichia coli"},
                {"accession": "GCF_2", "taxid": "1280", "organism": "Staphylococcus aureus"}]

    def test_persists_hits_and_deletes_each_genome(self, tmp_path: Any, genomes: Any,
                                                   monkeypatch: Any) -> None:
        from sample import genomic_context as gc

        # search_genome mocked: GCF_1 has a hit, GCF_2 has none.
        def fake_search(_pf: str, _gf: str, _wd: str, **kw: Any) -> List[Hit]:
            acc = kw.get("genome_accession", "")
            return [_hit(acc)] if acc == "GCF_1" else []

        monkeypatch.setattr(gc, "search_genome", fake_search)
        db = str(tmp_path / "g.sqlite")
        work = str(tmp_path / "work")
        result = gc.sweep_to_store("P1", "GIGKFL", genomes, _fake_download, work, store_path=db)

        assert result == {"genomes_searched": 2, "genomes_with_hits": 1, "total_hits": 1}
        # Every genome FASTA was deleted -- one genome on disk at a time is the whole point.
        assert not [f for f in os.listdir(work) if f.endswith(".fna")]

    def test_resume_skips_already_processed(self, tmp_path: Any, genomes: Any,
                                            monkeypatch: Any) -> None:
        from sample import genomic_context as gc

        calls: List[str] = []

        def fake_search(_pf: str, _gf: str, _wd: str, **kw: Any) -> List[Hit]:
            calls.append(kw.get("genome_accession", ""))
            return []

        monkeypatch.setattr(gc, "search_genome", fake_search)
        db = str(tmp_path / "g.sqlite")
        work = str(tmp_path / "work")
        gc.sweep_to_store("P1", "GIGKFL", genomes, _fake_download, work, store_path=db)
        assert set(calls) == {"GCF_1", "GCF_2"}

        # Second run: both already processed, so nothing is searched again.
        calls.clear()
        gc.sweep_to_store("P1", "GIGKFL", genomes, _fake_download, work, store_path=db)
        assert calls == []

    def test_no_resume_researches_everything(self, tmp_path: Any, genomes: Any,
                                             monkeypatch: Any) -> None:
        from sample import genomic_context as gc

        calls: List[str] = []
        monkeypatch.setattr(gc, "search_genome",
                            lambda *a, **kw: (calls.append(kw.get("genome_accession", "")) or []))
        db = str(tmp_path / "g.sqlite")
        work = str(tmp_path / "work")
        gc.sweep_to_store("P1", "GIGKFL", genomes, _fake_download, work, store_path=db)
        calls.clear()
        gc.sweep_to_store("P1", "GIGKFL", genomes, _fake_download, work, store_path=db,
                          resume=False)
        assert set(calls) == {"GCF_1", "GCF_2"}

    def test_limit_stops_after_n_new_genomes(self, tmp_path: Any, genomes: Any,
                                             monkeypatch: Any) -> None:
        from sample import genomic_context as gc

        monkeypatch.setattr(gc, "search_genome", lambda *a, **kw: [])
        db = str(tmp_path / "g.sqlite")
        work = str(tmp_path / "work")
        gc.sweep_to_store("P1", "GIGKFL", genomes, _fake_download, work, store_path=db, limit=1)
        assert store.summary("P1", db_path=db)["genomes_searched"] == 1
