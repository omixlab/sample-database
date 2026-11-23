"""
Tests for embedding search (`sample.embeddings`).

The suite runs offline (`tests/conftest.py`'s `no_network`), so these tests mock
`sample.esm_enrichment.compute_esm2_embeddings` -- the boundary where a real HuggingFace model
would otherwise load and, on a cold cache, reach the network -- the same "mock the boundary"
approach the rest of this suite uses for PubMed/Ollama/UniProt. What is under test here is this
project's own logic: decoy exclusion, Chroma wiring, and how a query is turned into ranked hits --
not ESM-2's or Chroma's own correctness.

Real-data verification (37,042 real peptides indexed with the real model, 0 decoys leaked in, a
live REST + web-form query returning sensible nearest neighbors) is recorded in `DECISIONS.md`.
"""
from typing import Any, List

import numpy as np
import pandas as pd
import pytest


def _fake_embeddings(sequences: List[str], batch_size: int = 32) -> np.ndarray:
    """
    A deterministic, length-and-content-based stand-in for ESM-2: sequences that share more
    characters in common get vectors that are closer together, which is enough structure for a
    nearest-neighbor test without needing the real model.
    """
    vectors = []
    for seq in sequences:
        counts = np.zeros(26)
        for ch in seq.upper():
            idx = ord(ch) - ord("A")
            if 0 <= idx < 26:
                counts[idx] += 1
        vectors.append(counts)
    return np.array(vectors, dtype=float)


@pytest.fixture
def fake_esm2(monkeypatch: pytest.MonkeyPatch) -> None:
    import sample.esm_enrichment as esm_mod

    monkeypatch.setattr(esm_mod, "compute_esm2_embeddings", _fake_embeddings)


@pytest.fixture
def dataset_csv(tmp_path: Any) -> str:
    path = str(tmp_path / "final_dataset.csv")
    pd.DataFrame([
        {"entry_id": "SAMPLE0000000001", "sequence": "GIGKFLHSAKKFGKAFVGEIMNS", "is_decoy": 0},
        {"entry_id": "SAMPLE0000000002", "sequence": "GIGKFLHSAKKFGKAFVGEIMKS", "is_decoy": 0},
        {"entry_id": "SAMPLE0000000003", "sequence": "DDDDEEEEDDDDEEEE", "is_decoy": 0},
        {"entry_id": "SAMPLE_DECOY_00000001", "sequence": "MFGLGGAAPQISSQQKLQAAEAELDMV",
         "is_decoy": 1},
    ]).to_csv(path, index=False)
    return path


class TestBuildEmbeddingIndex:
    def test_indexes_only_real_peptides(self, dataset_csv: str, fake_esm2: None,
                                        tmp_path: Any) -> None:
        from sample.embeddings import build_embedding_index

        chroma_dir = str(tmp_path / "chroma")
        summary = build_embedding_index(dataset_csv=dataset_csv, chroma_dir=chroma_dir)

        assert summary["indexed"] == 3, "the decoy must not be counted as indexed"

        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection("sample_peptides_esm2")
        assert collection.count() == 3
        assert collection.get(ids=["SAMPLE_DECOY_00000001"])["ids"] == [], \
            "a decoy must never be a returnable embedding-search hit"

    def test_rebuilds_from_scratch_rather_than_appending(
            self, dataset_csv: str, fake_esm2: None, tmp_path: Any) -> None:
        """A second build over the same directory must not double the collection -- this feature
        deliberately has no resumable/incremental mode, unlike the family-HMM build."""
        from sample.embeddings import build_embedding_index

        chroma_dir = str(tmp_path / "chroma")
        build_embedding_index(dataset_csv=dataset_csv, chroma_dir=chroma_dir)
        build_embedding_index(dataset_csv=dataset_csv, chroma_dir=chroma_dir)

        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        collection = client.get_collection("sample_peptides_esm2")
        assert collection.count() == 3

    def test_missing_dataset_returns_a_zeroed_summary_not_an_exception(
            self, tmp_path: Any, fake_esm2: None) -> None:
        from sample.embeddings import build_embedding_index

        summary = build_embedding_index(dataset_csv=str(tmp_path / "absent.csv"),
                                        chroma_dir=str(tmp_path / "chroma"))
        assert summary == {"path": str(tmp_path / "chroma"), "indexed": 0, "skipped": 0}

    def test_limit_caps_how_many_are_indexed(self, dataset_csv: str, fake_esm2: None,
                                             tmp_path: Any) -> None:
        from sample.embeddings import build_embedding_index

        summary = build_embedding_index(dataset_csv=dataset_csv, chroma_dir=str(tmp_path / "c"),
                                        limit=1)
        assert summary["indexed"] == 1

    @pytest.mark.parametrize("batch_size", [0, -1])
    def test_non_positive_batch_size_raises_a_clear_error(
            self, dataset_csv: str, fake_esm2: None, tmp_path: Any, batch_size: int) -> None:
        """`range(0, total, 0)` raises a bare `ValueError: range() arg 3 must not be zero` --
        reachable from the CLI's `--batch-size`, so this must be caught with a message that names
        the actual mistake rather than surfacing a Python internals error."""
        from sample.embeddings import build_embedding_index

        with pytest.raises(ValueError, match="batch_size"):
            build_embedding_index(dataset_csv=dataset_csv, chroma_dir=str(tmp_path / "c"),
                                  batch_size=batch_size)


class TestEmbeddingIndexPath:
    def test_empty_string_when_not_built(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.embeddings as embeddings_mod

        monkeypatch.setattr(embeddings_mod, "CHROMA_DIR", str(tmp_path / "nope"))
        assert embeddings_mod.embedding_index_path() == ""

    def test_returns_the_path_once_a_directory_exists_there(
            self, tmp_path: Any, fake_esm2: None, dataset_csv: str,
            monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.embeddings as embeddings_mod

        chroma_dir = str(tmp_path / "chroma")
        monkeypatch.setattr(embeddings_mod, "CHROMA_DIR", chroma_dir)
        assert embeddings_mod.embedding_index_path() == "", "must not exist before the build"

        embeddings_mod.build_embedding_index(dataset_csv=dataset_csv, chroma_dir=chroma_dir)

        assert embeddings_mod.embedding_index_path() == chroma_dir


class TestRunEmbeddingSearch:
    @pytest.fixture
    def built_index(self, dataset_csv: str, fake_esm2: None, tmp_path: Any) -> str:
        from sample.embeddings import build_embedding_index

        chroma_dir = str(tmp_path / "chroma")
        build_embedding_index(dataset_csv=dataset_csv, chroma_dir=chroma_dir)
        return chroma_dir

    def test_finds_the_closer_sequence_first(self, built_index: str, fake_esm2: None) -> None:
        # Shares far more amino-acid composition with SAMPLE0000000001/2 (both K/F/G-rich) than
        # with SAMPLE0000000003 (D/E-only) -- the fake embedding is a plain letter-count vector,
        # so composition similarity is exactly what it measures.
        hits = run_embedding_search_helper("GIGKFLHSAKKFGKAFVGEIMNS", built_index)
        assert hits[0]["entry_id"] == "SAMPLE0000000001"
        assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-6)
        assert hits[0]["similarity"] == pytest.approx(1.0, abs=1e-6)

    def test_hits_are_ordered_by_ascending_distance(self, built_index: str, fake_esm2: None) -> None:
        hits = run_embedding_search_helper("GIGKFLHSAKKFGKAFVGEIMNS", built_index)
        distances = [h["distance"] for h in hits]
        assert distances == sorted(distances)

    def test_max_hits_caps_results(self, built_index: str, fake_esm2: None) -> None:
        hits = run_embedding_search_helper("GIGKFLHSAKKFGKAFVGEIMNS", built_index, max_hits=1)
        assert len(hits) == 1

    def test_decoy_never_returned(self, built_index: str, fake_esm2: None) -> None:
        hits = run_embedding_search_helper("MFGLGGAAPQISSQQKLQAAEAELDMV", built_index)
        assert all(h["entry_id"] != "SAMPLE_DECOY_00000001" for h in hits)

    def test_empty_query_returns_empty_without_touching_the_index(self) -> None:
        from sample.embeddings import run_embedding_search

        assert run_embedding_search("", "/nonexistent/path") == []

    def test_fasta_header_and_whitespace_are_stripped(self, built_index: str,
                                                       fake_esm2: None) -> None:
        hits_plain = run_embedding_search_helper("GIGKFLHSAKKFGKAFVGEIMNS", built_index)
        hits_fasta = run_embedding_search_helper(
            ">query\nGIGKFLHSAKKFGKAFVGEIMNS\n", built_index)
        assert [h["entry_id"] for h in hits_plain] == [h["entry_id"] for h in hits_fasta]

    def test_missing_collection_fails_closed_to_empty_list(self, tmp_path: Any,
                                                            fake_esm2: None) -> None:
        from sample.embeddings import run_embedding_search

        empty_dir = str(tmp_path / "empty_chroma")
        import chromadb
        chromadb.PersistentClient(path=empty_dir)  # a Chroma dir with no collection created in it
        assert run_embedding_search("GIGKFLHSAKKFGKAFVGEIMNS", empty_dir) == []


def run_embedding_search_helper(query: str, chroma_dir: str, max_hits: int = 50) -> List[Any]:
    from sample.embeddings import run_embedding_search

    return run_embedding_search(query, chroma_dir, max_hits=max_hits)


class TestEmbeddingSearchRoutes:
    """
    The REST endpoint and the web form -- mocked where each route actually bound the name (`from
    sample.api.common import ...` binds it locally, so patching `sample.api.common` itself would
    not be seen by an already-imported route module; same reasoning as `TestHmmerSearch`'s web-form
    test in `tests/test_families_synergy.py`), rather than building a real index here: what these
    verify is the wiring (status codes, response shape, template rendering), which does not depend
    on ESM-2 or Chroma being real. The real end-to-end path (real model, real 37,042-peptide index,
    live REST + web-form request) was verified manually and is recorded in `DECISIONS.md`.
    """

    @pytest.fixture
    def mocked_hit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from sample.api import pages, v1

        hits = [{"entry_id": "SAMPLE0000000001", "distance": 0.03, "similarity": 0.97}]
        for module in (pages, v1):
            monkeypatch.setattr(module, "embedding_index_path", lambda: "/fake/chroma")
            monkeypatch.setattr(module, "run_embedding_search",
                                lambda query, path, max_hits=50: hits)

    def test_rest_endpoint_returns_the_hit(self, mocked_hit: None) -> None:
        from fastapi.testclient import TestClient

        from sample.api.app import app

        client = TestClient(app)
        response = client.get("/api/v1/search/embedding", params={"sequence": "GIGKFLK"})
        assert response.status_code == 200
        assert response.json()[0]["entry_id"] == "SAMPLE0000000001"

    def test_rest_endpoint_documented_in_openapi(self, mocked_hit: None) -> None:
        from fastapi.testclient import TestClient

        from sample.api.app import app

        client = TestClient(app)
        openapi = client.get("/api/v1/openapi.json").json()
        assert "/api/v1/search/embedding" in openapi["paths"]

    def test_rest_endpoint_503s_when_index_unbuilt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from sample.api import v1
        from sample.api.app import app

        monkeypatch.setattr(v1, "embedding_index_path", lambda: "")
        client = TestClient(app)
        response = client.get("/api/v1/search/embedding", params={"sequence": "GIGKFLK"})
        assert response.status_code == 503

    def test_web_form_offers_the_mode_and_renders_the_hit(self, mocked_hit: None) -> None:
        from fastapi.testclient import TestClient

        from sample.api.app import app

        client = TestClient(app)
        body = client.get("/", params={"q": "GIGKFLK", "type": "embedding"}).text
        assert 'value="embedding"' in body
        assert "SAMPLE0000000001" in body
        assert "/search/peptide/SAMPLE0000000001" in body

    def test_web_form_reports_unavailable_index_rather_than_500(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from fastapi.testclient import TestClient

        from sample.api import pages
        from sample.api.app import app

        monkeypatch.setattr(pages, "embedding_index_path", lambda: "")
        client = TestClient(app)
        response = client.get("/", params={"q": "GIGKFLK", "type": "embedding"})
        assert response.status_code == 200
        assert "unavailable" in response.text
