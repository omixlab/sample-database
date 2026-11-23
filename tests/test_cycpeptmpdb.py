"""
Tests for :mod:`sample.cycpeptmpdb` -- indexing CycPeptMPDB as a cyclic-peptide source.

The behaviours that matter: every imported peptide is marked circular (``head_to_tail`` topology),
permeability is parsed as a number (or left ``None``, never guessed), a row with no SMILES is
dropped (structure is the identity for these non-standard cyclic peptides), and the normalised table
has exactly the declared columns. A tiny fixture CSV stands in for the 8,500-row download.
"""
import os
from typing import Any

import pytest

from sample import cycpeptmpdb


def _write_fixture(path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ID,Source,Original_Name_in_Source_Literature,SMILES,Sequence,Monomer_Length,"
                 "Permeability,PAMPA,Caco2\n")
        fh.write("1,2006_Rezai_1,Cyclosporine A,C1CCNC1,\"['meL','V']\",11,-6.6,-6.6,\n")
        fh.write("2,2011_White,Peptide2,C1CCCC1,\"['A','G']\",6,-5.1,,-6.0\n")
        fh.write("3,2012_Rand,NoStructure,,\"['X']\",4,-7.0,-7.0,\n")   # no SMILES -> dropped


class TestLoad:
    def test_marks_every_peptide_circular(self, tmp_path: Any) -> None:
        csv = str(tmp_path / "raw.csv")
        _write_fixture(csv)
        records = cycpeptmpdb.load(csv)
        assert records, "expected records"
        for r in records:
            assert r["topology"] == "head_to_tail"
            assert r["is_cyclic"] is True
            assert r["database"] == "CycPeptMPDB"

    def test_drops_rows_without_smiles(self, tmp_path: Any) -> None:
        csv = str(tmp_path / "raw.csv")
        _write_fixture(csv)
        records = cycpeptmpdb.load(csv)
        # Row 3 has no SMILES -> dropped; rows 1 and 2 kept.
        assert {r["cycpeptmpdb_id"] for r in records} == {"1", "2"}

    def test_parses_permeability_numbers_and_missing(self, tmp_path: Any) -> None:
        csv = str(tmp_path / "raw.csv")
        _write_fixture(csv)
        by_id = {r["cycpeptmpdb_id"]: r for r in cycpeptmpdb.load(csv)}
        assert by_id["1"]["permeability_logpapp"] == pytest.approx(-6.6)
        assert by_id["1"]["caco2_logpapp"] is None      # empty in the fixture
        assert by_id["2"]["caco2_logpapp"] == pytest.approx(-6.0)
        assert by_id["1"]["name"] == "Cyclosporine A"

    def test_missing_file_names_the_import_command(self, tmp_path: Any) -> None:
        with pytest.raises(FileNotFoundError, match="import-cycpeptmpdb"):
            cycpeptmpdb.load(str(tmp_path / "absent.csv"))


class TestInchikey:
    def test_same_molecule_two_spellings_one_key(self) -> None:
        # The same structure written two ways yields one InChIKey -- the property the link relies on.
        assert cycpeptmpdb.inchikey(smiles="OCC") == cycpeptmpdb.inchikey(smiles="C(O)C")

    def test_unparseable_is_none(self) -> None:
        assert cycpeptmpdb.inchikey(smiles="not a smiles") is None
        assert cycpeptmpdb.inchikey() is None


class TestLinkToRelease:
    def _release_db(self, tmp_path: Any, smiles: str) -> str:
        import sqlite3
        path = str(tmp_path / "sample.db")
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE peptides (entry_id TEXT, sequence TEXT, SMILES TEXT, "
                     "is_decoy INTEGER DEFAULT 0)")
        conn.execute("INSERT INTO peptides VALUES ('SAMPLE0000000001', 'GIGKFL', ?, 0)", (smiles,))
        conn.execute("INSERT INTO peptides VALUES ('SAMPLE_DECOY_1', 'AAAA', 'CCCC', 1)")
        conn.commit()
        conn.close()
        return path

    def test_links_same_molecule_by_structure(self, tmp_path: Any) -> None:
        # A release peptide and a CycPeptMPDB entry with the SAME SMILES must link.
        shared = "C1CCNC1"
        db = self._release_db(tmp_path, shared)
        csv = str(tmp_path / "cyc.csv")
        _write_fixture(csv)
        table = str(tmp_path / "norm.csv")
        cycpeptmpdb.build_table(csv_path=csv, output_csv=table)
        # Force the first normalised entry's SMILES to the shared structure.
        import pandas as pd
        norm = pd.read_csv(table)
        norm.loc[0, "smiles"] = shared
        norm.to_csv(table, index=False)

        out = str(tmp_path / "links.csv")
        summary = cycpeptmpdb.link_to_release(csv_path=table, db_path=db, output_csv=out)
        assert summary["links"] == 1
        assert summary["release_matched"] == 1
        linked = pd.read_csv(out)
        assert linked.iloc[0]["entry_id"] == "SAMPLE0000000001"

    def test_no_link_when_structures_differ(self, tmp_path: Any) -> None:
        db = self._release_db(tmp_path, "c1ccccc1")   # benzene, unlike any fixture entry
        csv = str(tmp_path / "cyc.csv")
        _write_fixture(csv)
        table = str(tmp_path / "norm.csv")
        cycpeptmpdb.build_table(csv_path=csv, output_csv=table)
        summary = cycpeptmpdb.link_to_release(
            csv_path=table, db_path=db, output_csv=str(tmp_path / "links.csv"))
        assert summary["links"] == 0


class TestBuildTable:
    def test_writes_the_declared_columns(self, tmp_path: Any) -> None:
        csv = str(tmp_path / "raw.csv")
        out = str(tmp_path / "normalised.csv")
        _write_fixture(csv)
        summary = cycpeptmpdb.build_table(csv_path=csv, output_csv=out)

        import pandas as pd
        table = pd.read_csv(out)
        assert list(table.columns) == cycpeptmpdb.COLUMNS
        assert summary["rows"] == 2
        assert summary["with_permeability"] == 2
        assert summary["with_caco2"] == 1
        assert bool(table["is_cyclic"].all())
