"""
Tests for :mod:`sample.pdb_homology` -- RCSB sequence search, chain resolution, and
sequence-alignment-guided CA-RMSD superposition of a ColabFold prediction onto a real PDB entry.

Hermetic throughout: the three network boundaries (`_post_json`, `_get_json`, `_get_text`) are
monkeypatched with canned responses shaped like RCSB's real API, and the superposition math is
exercised against small synthetic PDB texts built here rather than downloaded structures.
"""
from typing import Any, Dict, List, Optional

import pytest

from sample import pdb_homology as PH


def _atom_line(serial: int, resname: str, chain: str, resnum: int,
               x: float, y: float, z: float, bfactor: float = 50.0) -> str:
    return (f"ATOM  {serial:>5d}  CA  {resname:<3s} {chain}{resnum:>4d}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{1.0:>6.2f}{bfactor:>6.2f}          C")


def _backbone_pdb(coords: List[tuple], chain: str = "A", resnames: Optional[List[str]] = None) -> str:
    """A CA-only synthetic PDB: enough for `_chain_ca_atoms`, which reads only CA atoms."""
    resnames = resnames or ["ALA"] * len(coords)
    lines = [_atom_line(i + 1, resnames[i], chain, i + 1, *xyz) for i, xyz in enumerate(coords)]
    lines += ["TER", "END"]
    return "\n".join(lines) + "\n"


#: Five CA atoms 3.8 A apart along x -- an arbitrary but simple, non-degenerate backbone.
_COORDS = [(0.0, 0.0, 0.0), (3.8, 0.0, 0.0), (7.6, 0.0, 0.0), (11.4, 0.0, 0.0), (15.2, 0.0, 0.0)]


@pytest.fixture
def predicted_pdb(tmp_path: Any) -> str:
    path = tmp_path / "predicted.pdb"
    path.write_text(_backbone_pdb(_COORDS, chain="A"))
    return str(path)


class TestSearchSimilarPdb:
    def test_too_short_sequence_skips_the_network_call_entirely(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def must_not_be_called(*a: Any, **k: Any) -> Any:
            raise AssertionError("a query shorter than MIN_SEARCHABLE_LENGTH must not hit RCSB")

        monkeypatch.setattr(PH, "_post_json", must_not_be_called)
        assert PH.search_similar_pdb("GIGK") == []

    def test_parses_a_real_shaped_rcsb_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Shaped like RCSB's actual sequence-search response: result_set entries identified as
        # "<pdb_id>_<entity_id>", with identity/e-value nested under services[].nodes[].match_context.
        response = {
            "result_set": [
                {"identifier": "6M0J_1",
                 "services": [{"nodes": [{"match_context": [
                     {"sequence_identity": 0.91, "evalue": 1.2e-8}]}]}]},
                {"identifier": "1XYZ_2",
                 "services": [{"nodes": [{"match_context": [
                     {"sequence_identity": 0.75, "evalue": 3.4e-5}]}]}]},
            ]
        }
        monkeypatch.setattr(PH, "_post_json", lambda url, payload, timeout=30.0: response)

        hits = PH.search_similar_pdb("GIGKFLHSAKKFGKAFVGEIMNS")

        assert hits == [
            {"pdb_id": "6M0J", "entity_id": "1", "identity": 0.91, "evalue": 1.2e-8},
            {"pdb_id": "1XYZ", "entity_id": "2", "identity": 0.75, "evalue": 3.4e-5},
        ]

    def test_sends_the_query_sequence_and_identity_floor(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: Dict[str, Any] = {}

        def capturing_post(url: str, payload: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
            captured["payload"] = payload
            return {"result_set": []}

        monkeypatch.setattr(PH, "_post_json", capturing_post)
        PH.search_similar_pdb("GIGKFLHSAKKFGKAFVGEIMNS", min_identity=0.85)

        params = captured["payload"]["query"]["parameters"]
        assert params["value"] == "GIGKFLHSAKKFGKAFVGEIMNS"
        assert params["identity_cutoff"] == 0.85

    def test_network_failure_returns_empty_rather_than_raising(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_post(*a: Any, **k: Any) -> Any:
            raise ConnectionError("RCSB unreachable")

        monkeypatch.setattr(PH, "_post_json", failing_post)
        assert PH.search_similar_pdb("GIGKFLHSAKKFGKAFVGEIMNS") == []

    def test_malformed_response_returns_empty_rather_than_raising(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(PH, "_post_json", lambda *a, **k: {"unexpected": "shape"})
        assert PH.search_similar_pdb("GIGKFLHSAKKFGKAFVGEIMNS") == []


class TestFetchReferenceChain:
    def test_resolves_chain_and_downloads_the_entry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        entity_response = {
            "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["B", "D"]},
            "rcsb_polymer_entity": {"pdbx_description": "Some antimicrobial peptide"},
        }
        monkeypatch.setattr(PH, "_get_json", lambda url, timeout=30.0: entity_response)
        monkeypatch.setattr(PH, "_get_text", lambda url, timeout=60.0: "ATOM ... PDB TEXT")

        result = PH._fetch_reference_chain("6M0J", "1")

        assert result == {"chain": "B", "pdb_text": "ATOM ... PDB TEXT",
                          "title": "Some antimicrobial peptide"}

    def test_no_chains_in_the_entity_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(PH, "_get_json", lambda url, timeout=30.0: {
            "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": []}})
        assert PH._fetch_reference_chain("6M0J", "1") is None

    def test_network_failure_returns_none_rather_than_raising(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_get(*a: Any, **k: Any) -> Any:
            raise ConnectionError("RCSB unreachable")

        monkeypatch.setattr(PH, "_get_json", failing_get)
        assert PH._fetch_reference_chain("6M0J", "1") is None


class TestSuperpose:
    def test_recovers_near_zero_rmsd_after_a_pure_translation(self, predicted_pdb: str) -> None:
        # A structurally identical chain, just shifted in space and living in a different chain
        # letter -- exactly what a "real" PDB entry's own coordinate frame looks like relative to
        # a freshly predicted model. Superposition must remove that offset, not report it as error.
        translated = [(x + 100.0, y + 50.0, z + 20.0) for x, y, z in _COORDS]
        reference_text = _backbone_pdb(translated, chain="B")

        result = PH.superpose(predicted_pdb, reference_text, "B")

        assert result is not None
        assert result["rmsd"] == pytest.approx(0.0, abs=0.01)
        assert result["aligned_residues"] == 5
        assert result["identity"] == 1.0

    def test_amino_acid_substitution_lowers_identity_but_rmsd_is_still_computed(
            self, predicted_pdb: str) -> None:
        # Same geometry, one residue swapped from ALA to GLY: a real near-homolog, not an exact
        # match. Coordinates are unchanged, so RMSD stays ~0; only the reported identity should drop.
        resnames = ["ALA", "ALA", "GLY", "ALA", "ALA"]
        reference_text = _backbone_pdb(_COORDS, chain="B", resnames=resnames)

        result = PH.superpose(predicted_pdb, reference_text, "B")

        assert result is not None
        assert result["rmsd"] == pytest.approx(0.0, abs=0.01)
        assert result["identity"] == pytest.approx(4 / 5)

    def test_too_few_residues_returns_none_rather_than_a_meaningless_fit(
            self, tmp_path: Any) -> None:
        short_pdb = tmp_path / "short.pdb"
        short_pdb.write_text(_backbone_pdb(_COORDS[:2], chain="A"))
        reference_text = _backbone_pdb(_COORDS[:2], chain="B")

        assert PH.superpose(str(short_pdb), reference_text, "B") is None

    def test_wrong_chain_id_in_the_reference_returns_none(self, predicted_pdb: str) -> None:
        reference_text = _backbone_pdb(_COORDS, chain="B")
        assert PH.superpose(predicted_pdb, reference_text, "Z") is None


class TestHomologyForPeptide:
    def test_no_predicted_structure_short_circuits_to_no_match(self) -> None:
        result = PH.homology_for_peptide("SAMPLE0000000001", "GIGKFLHSAKKF", "/does/not/exist.pdb")
        assert result == {"has_match": False}

    def test_no_search_hits_returns_no_match(self, predicted_pdb: str,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(PH, "search_similar_pdb", lambda *a, **k: [])
        result = PH.homology_for_peptide("SAMPLE0000000001", "GIGKFLHSAKKF", predicted_pdb)
        assert result == {"has_match": False}

    def test_full_pipeline_reports_a_match(self, predicted_pdb: str,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
        translated = [(x + 5.0, y + 5.0, z + 5.0) for x, y, z in _COORDS]
        reference_text = _backbone_pdb(translated, chain="C")

        monkeypatch.setattr(PH, "search_similar_pdb", lambda *a, **k: [
            {"pdb_id": "6M0J", "entity_id": "1", "identity": 0.91, "evalue": 1e-8}])
        monkeypatch.setattr(PH, "_fetch_reference_chain", lambda pdb_id, entity_id: {
            "chain": "C", "pdb_text": reference_text, "title": "A real solved peptide"})

        result = PH.homology_for_peptide("SAMPLE0000000001", "GIGKFLHSAKKF", predicted_pdb)

        assert result["has_match"] is True
        assert result["pdb_id"] == "6M0J"
        assert result["chain"] == "C"
        assert result["title"] == "A real solved peptide"
        assert result["search_identity"] == 0.91
        assert result["rmsd"] == pytest.approx(0.0, abs=0.01)
        assert result["aligned_residues"] == 5

    def test_falls_through_to_the_next_hit_when_the_first_cannot_be_resolved(
            self, predicted_pdb: str, monkeypatch: pytest.MonkeyPatch) -> None:
        reference_text = _backbone_pdb(_COORDS, chain="C")
        fetch_calls: List[str] = []

        def fetch(pdb_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
            fetch_calls.append(pdb_id)
            if pdb_id == "1BAD":
                return None  # withdrawn entry / transient RCSB failure
            return {"chain": "C", "pdb_text": reference_text, "title": "second hit"}

        monkeypatch.setattr(PH, "search_similar_pdb", lambda *a, **k: [
            {"pdb_id": "1BAD", "entity_id": "1", "identity": 0.95, "evalue": 1e-9},
            {"pdb_id": "6M0J", "entity_id": "1", "identity": 0.80, "evalue": 1e-6}])
        monkeypatch.setattr(PH, "_fetch_reference_chain", fetch)

        result = PH.homology_for_peptide("SAMPLE0000000001", "GIGKFLHSAKKF", predicted_pdb)

        assert fetch_calls == ["1BAD", "6M0J"]
        assert result["has_match"] is True
        assert result["pdb_id"] == "6M0J"


class TestStore:
    @pytest.fixture
    def db_path(self, tmp_path: Any) -> str:
        return str(tmp_path / "pdb_homology.sqlite")

    def test_save_and_get_roundtrip(self, db_path: str) -> None:
        PH.init_db(db_path)
        result = {"has_match": True, "pdb_id": "6M0J", "rmsd": 1.2}
        PH.save("SAMPLE0000000001", result, db_path)
        assert PH.get("SAMPLE0000000001", db_path) == result

    def test_get_returns_none_before_the_db_exists(self, db_path: str) -> None:
        assert PH.get("SAMPLE0000000001", db_path) is None

    def test_get_returns_none_for_an_unstored_peptide(self, db_path: str) -> None:
        PH.init_db(db_path)
        PH.save("SAMPLE0000000001", {"has_match": False}, db_path)
        assert PH.get("SAMPLE0000000002", db_path) is None

    def test_processed_ids_reflects_what_was_saved(self, db_path: str) -> None:
        PH.init_db(db_path)
        PH.save("SAMPLE0000000001", {"has_match": False}, db_path)
        PH.save("SAMPLE0000000002", {"has_match": True}, db_path)
        assert PH.processed_ids(db_path) == {"SAMPLE0000000001", "SAMPLE0000000002"}

    def test_save_upserts_rather_than_duplicating(self, db_path: str) -> None:
        PH.init_db(db_path)
        PH.save("SAMPLE0000000001", {"has_match": False}, db_path)
        PH.save("SAMPLE0000000001", {"has_match": True, "pdb_id": "6M0J"}, db_path)
        assert PH.get("SAMPLE0000000001", db_path) == {"has_match": True, "pdb_id": "6M0J"}
        assert len(PH.processed_ids(db_path)) == 1


class TestBuildAll:
    def test_no_structures_directory_is_a_clean_no_op(self, tmp_path: Any) -> None:
        result = PH.build_all(structures_dir=str(tmp_path / "does_not_exist"),
                              store_path=str(tmp_path / "store.sqlite"))
        assert result == {"processed": 0, "matched": 0, "total_stored": 0}

    def test_only_folded_and_unprocessed_peptides_are_swept(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        structures_dir = tmp_path / "structures"
        structures_dir.mkdir()
        (structures_dir / "SAMPLE0000000001.pdb").write_text(_backbone_pdb(_COORDS))
        (structures_dir / "SAMPLE0000000002.pdb").write_text(_backbone_pdb(_COORDS))
        store_path = str(tmp_path / "store.sqlite")
        PH.init_db(store_path)
        PH.save("SAMPLE0000000002", {"has_match": False}, store_path)  # already processed

        from sample import db as release_db
        monkeypatch.setattr(release_db, "get_peptide",
                            lambda entry_id, db_path=None: {"sequence": "GIGKFLHSAKKF"})
        monkeypatch.setattr(PH, "homology_for_peptide",
                            lambda entry_id, seq, path, min_identity=0.7: {"has_match": False})

        result = PH.build_all(structures_dir=str(structures_dir), store_path=store_path,
                              sleep_seconds=0.0)

        assert result["processed"] == 1  # only the unprocessed peptide
        assert PH.processed_ids(store_path) == {"SAMPLE0000000001", "SAMPLE0000000002"}
