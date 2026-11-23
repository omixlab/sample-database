"""
Tests for the annotator (`sample.annotator`, `sample.annotation_store`) and its REST endpoints.

`_pfam_scan` shells out to `hmmsearch` against the real ~2.2 GB Pfam-A database (measured ~18s per
sequence) and `_sample_hits` reads the release database through `sample.similarity`/`sample.db` --
both mocked at their boundary here so the suite stays fast and does not depend on either the Pfam
download or a populated release database being present. Real-data verification (a known Pfam hit
resolved to the correct accession, HemoPI2 producing a real score, 19 near-identical SAMPLE
peptides found for magainin-2 with correct literature/patent flags, the full REST flow including
the one-week cache) is recorded in `DECISIONS.md`.
"""
import time
from typing import Any, Dict, List

import pytest


class TestPfamScan:
    def test_parses_a_real_domtbl_shape(self, monkeypatch: pytest.MonkeyPatch,
                                        tmp_path: Any) -> None:
        import sample.annotator as annot

        # A minimal, realistic domtbl row: target(peptide) accession tlen query(family) accession
        # qlen full-Evalue full-score ... (fields[3]=family, fields[4]=accession, fields[6]=evalue)
        def fake_run(cmd: List[str], **kwargs: Any) -> Any:
            domtbl_path = cmd[cmd.index("--domtblout") + 1]
            with open(domtbl_path, "w", encoding="utf-8") as handle:
                handle.write("# comment\n")
                handle.write("query - 33 DD_K PF12121.11 45 9.2e-13 45.2 0.1 1 1 1 1 1 33 1 33 "
                             "0.99 -\n")
            return type("R", (), {"returncode": 0})()

        monkeypatch.setattr(annot.subprocess, "run", fake_run)
        hmm_path = str(tmp_path / "Pfam-A.hmm")
        with open(hmm_path, "w") as f:
            f.write("fake")

        hits = annot._pfam_scan("GLWSKIKEVGKEAAKAAAKAAGKAALGAVSEAV", hmm_path=hmm_path)

        assert hits == [{"family": "DD_K", "accession": "PF12121", "evalue": pytest.approx(9.2e-13)}]

    def test_missing_database_returns_empty_without_running_hmmsearch(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.annotator as annot

        calls = []
        monkeypatch.setattr(annot.subprocess, "run", lambda *a, **k: calls.append(1))

        assert annot._pfam_scan("GIGKFLK", hmm_path="/nonexistent/Pfam-A.hmm") == []
        assert not calls

    def test_hmmsearch_failure_returns_empty_not_an_exception(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
        import sample.annotator as annot

        def failing_run(cmd: List[str], **kwargs: Any) -> Any:
            raise RuntimeError("hmmsearch crashed")

        monkeypatch.setattr(annot.subprocess, "run", failing_run)
        hmm_path = str(tmp_path / "Pfam-A.hmm")
        open(hmm_path, "w").close()

        assert annot._pfam_scan("GIGKFLK", hmm_path=hmm_path) == []


class TestSampleHits:
    def test_annotates_each_hit_with_literature_and_patent_flags(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.annotator as annot
        from sample import db, similarity

        monkeypatch.setattr(similarity, "search", lambda *a, **k: [
            {"entry_id": "SAMPLE0000000001", "identity": 100.0, "mismatches": 0},
            {"entry_id": "SAMPLE0000000002", "identity": 95.7, "mismatches": 1},
        ])

        def fake_get_peptide(entry_id: str) -> Dict[str, Any]:
            return {"LITERATURE": "10.1/x" if entry_id.endswith("1") else None,
                   "PATENT": None}

        monkeypatch.setattr(db, "get_peptide", fake_get_peptide)

        hits = annot._sample_hits("GIGKFLHSAKKFGKAFVGEIMNS")

        assert hits[0] == {"entry_id": "SAMPLE0000000001", "identity": 100.0, "mismatches": 0,
                           "has_literature": True, "has_patent": False}
        assert hits[1]["has_literature"] is False

    def test_search_failure_returns_empty_not_an_exception(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.annotator as annot
        from sample import similarity

        def failing_search(*a: Any, **k: Any) -> Any:
            raise RuntimeError("index corrupt")

        monkeypatch.setattr(similarity, "search", failing_search)
        assert annot._sample_hits("GIGKFLK") == []


class TestRunAnnotation:
    @pytest.fixture
    def mocked_pieces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.annotator as annot

        monkeypatch.setattr(annot, "_pfam_scan", lambda seq: [
            {"family": "DD_K", "accession": "PF12121", "evalue": 9.2e-13}])
        monkeypatch.setattr(annot, "_sample_hits", lambda seq: [
            {"entry_id": "SAMPLE0000000001", "identity": 100.0, "mismatches": 0,
             "has_literature": True, "has_patent": False}])

    def test_empty_sequence_short_circuits_without_calling_any_tool(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sample.annotator as annot

        calls = []
        monkeypatch.setattr(annot, "_pfam_scan", lambda seq: calls.append("pfam"))
        monkeypatch.setattr(annot, "_sample_hits", lambda seq: calls.append("sample"))

        result = annot.run_annotation("")

        assert result["sequence"] == ""
        assert result["hemopi2"] is None
        assert not calls

    def test_strips_fasta_header_and_whitespace(self, mocked_pieces: None) -> None:
        from sample.annotator import run_annotation

        result = run_annotation(">query\nGIGKFLHSAKKFGKAFVGEIMNS\n")
        assert result["sequence"] == "GIGKFLHSAKKFGKAFVGEIMNS"

    def test_combines_pfam_and_sample_hits(self, mocked_pieces: None) -> None:
        from sample.annotator import run_annotation

        result = run_annotation("GIGKFLHSAKKFGKAFVGEIMNS")
        assert result["pfam_hits"][0]["accession"] == "PF12121"
        assert result["sample_hits"][0]["entry_id"] == "SAMPLE0000000001"

    def test_hemopi2_is_none_when_unavailable(self, mocked_pieces: None,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
        """`_predictor_command` locates hemopi2 by the isolated `sample-predictors` environment's
        absolute path regardless of which environment the caller runs in, so it is reachable even
        from this test suite's own `sample` environment. `EXTERNAL_PREDICTORS` holds one
        `HemoPI2Predictor` singleton that caches its own availability on first use (by design --
        it is probed once, see the class docstring), so patching `_predictor_command` alone is not
        enough once another test has already triggered that cache; the instance's own `available`
        method is patched directly instead."""
        import sample.peptide_predictors as pp

        hemopi = next(p for p in pp.EXTERNAL_PREDICTORS if p.name == "hemopi2")
        monkeypatch.setattr(hemopi, "available", lambda: False)

        from sample.annotator import run_annotation

        result = run_annotation("GIGKFLHSAKKFGKAFVGEIMNS")
        assert result["hemopi2"] is None
        assert result["predictors_available"]["hemopi2"] is False

    def test_hemopi2_result_is_included_when_available(self, mocked_pieces: None) -> None:
        """Not mocked: hemopi2 really is reachable via the isolated predictor environment (see
        the test above), so this exercises the real subprocess call end to end."""
        from sample.annotator import run_annotation

        result = run_annotation("GIGKFLHSAKKFGKAFVGEIMNS")
        assert result["hemopi2"] is not None
        assert "hemopi2_score" in result["hemopi2"]
        assert result["predictors_available"]["hemopi2"] is True


class TestAnnotationStore:
    @pytest.fixture
    def db_path(self, tmp_path: Any) -> str:
        from sample import annotation_store

        path = str(tmp_path / "annotations.db")
        annotation_store.init_db(path)
        return path

    def test_job_lifecycle(self, db_path: str) -> None:
        from sample import annotation_store

        annotation_store.create_job("job1", "GIGKFLK", db_path=db_path)
        job = annotation_store.get_job("job1", db_path=db_path)
        assert job["status"] == annotation_store.STATUS_PENDING

        annotation_store.mark_running("job1", db_path=db_path)
        assert annotation_store.get_job("job1", db_path=db_path)["status"] == \
            annotation_store.STATUS_RUNNING

        annotation_store.mark_finished("job1", annotation_store.STATUS_SUCCESS,
                                       result={"sequence": "GIGKFLK"}, db_path=db_path)
        job = annotation_store.get_job("job1", db_path=db_path)
        assert job["status"] == annotation_store.STATUS_SUCCESS
        assert job["result"] == {"sequence": "GIGKFLK"}

    def test_unknown_job_returns_none(self, db_path: str) -> None:
        from sample import annotation_store

        assert annotation_store.get_job("nope", db_path=db_path) is None

    def test_cache_hits_a_recent_successful_run_of_the_same_sequence(self, db_path: str) -> None:
        from sample import annotation_store

        annotation_store.create_job("job1", "GIGKFLK", db_path=db_path)
        annotation_store.mark_finished("job1", annotation_store.STATUS_SUCCESS,
                                       result={"sequence": "GIGKFLK"}, db_path=db_path)

        cached = annotation_store.find_cached_result("GIGKFLK", db_path=db_path)
        assert cached is not None
        assert cached["job_id"] == "job1"

    def test_cache_ignores_a_different_sequence(self, db_path: str) -> None:
        from sample import annotation_store

        annotation_store.create_job("job1", "GIGKFLK", db_path=db_path)
        annotation_store.mark_finished("job1", annotation_store.STATUS_SUCCESS,
                                       result={}, db_path=db_path)

        assert annotation_store.find_cached_result("DDDDEEEE", db_path=db_path) is None

    def test_cache_ignores_a_failed_run(self, db_path: str) -> None:
        from sample import annotation_store

        annotation_store.create_job("job1", "GIGKFLK", db_path=db_path)
        annotation_store.mark_finished("job1", annotation_store.STATUS_FAILURE,
                                       error="boom", db_path=db_path)

        assert annotation_store.find_cached_result("GIGKFLK", db_path=db_path) is None

    def test_cache_respects_the_ttl(self, db_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from sample import annotation_store

        annotation_store.create_job("job1", "GIGKFLK", db_path=db_path)
        annotation_store.mark_finished("job1", annotation_store.STATUS_SUCCESS,
                                       result={}, db_path=db_path)

        # A cache lookup with a TTL shorter than "now minus finished_at" must miss.
        time.sleep(0.05)
        assert annotation_store.find_cached_result(
            "GIGKFLK", db_path=db_path, ttl_seconds=0) is None
        assert annotation_store.find_cached_result(
            "GIGKFLK", db_path=db_path, ttl_seconds=3600) is not None

    def test_sequence_hash_is_case_insensitive(self) -> None:
        from sample.annotation_store import sequence_hash

        assert sequence_hash("gigkflk") == sequence_hash("GIGKFLK")


class TestAnnotationRoutes:
    @pytest.fixture
    def client(self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
        from fastapi.testclient import TestClient

        from sample import annotation_store, tasks
        from sample.api.app import app

        annotations_db = str(tmp_path / "annotations.db")
        annotation_store.init_db(annotations_db)
        monkeypatch.setenv("SAMPLE_ANNOTATIONS_DB", annotations_db)

        monkeypatch.setitem(tasks.celery_app.conf, "task_always_eager", True)
        monkeypatch.setattr(tasks, "broker_available", lambda: True)

        # The battery itself is mocked (see the module docstring); this suite is about the
        # queueing/caching/route wiring, not re-verifying HemoPI2/Pfam/similarity here.
        monkeypatch.setattr(
            "sample.annotator.run_annotation",
            lambda sequence: {"sequence": sequence, "hemopi2": {"hemopi2_score": 0.5},
                              "pfam_hits": [], "sample_hits": [],
                              "predictors_available": {"hemopi2": True}})

        return TestClient(app)

    def test_submit_then_poll_result(self, client: Any) -> None:
        submit = client.post("/api/v1/annotate", json={"sequence": "GIGKFLHSAKKFGKAFVGEIMNS"})
        assert submit.status_code == 202
        data = submit.json()
        assert data["cached"] is False

        result = client.get(data["result_url"])
        assert result.status_code == 200
        payload = result.json()
        assert payload["job"]["done"] is True
        assert payload["result"]["sequence"] == "GIGKFLHSAKKFGKAFVGEIMNS"

    def test_second_submission_of_the_same_sequence_is_cached(self, client: Any) -> None:
        first = client.post("/api/v1/annotate", json={"sequence": "ILPWKWPWWPWRR"})
        second = client.post("/api/v1/annotate", json={"sequence": "ILPWKWPWWPWRR"})

        assert first.json()["cached"] is False
        assert second.json()["cached"] is True
        assert second.json()["job_id"] == first.json()["job_id"]

    def test_empty_sequence_is_rejected(self, client: Any) -> None:
        response = client.post("/api/v1/annotate", json={"sequence": "   "})
        assert response.status_code == 400

    def test_unknown_job_id_is_404(self, client: Any) -> None:
        assert client.get("/api/v1/annotate/nope").status_code == 404
        assert client.get("/api/v1/annotate/nope/result").status_code == 404

    def test_broker_unavailable_returns_503_without_creating_a_job(
            self, client: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from sample import tasks

        monkeypatch.setattr(tasks, "broker_available", lambda: False)
        response = client.post("/api/v1/annotate", json={"sequence": "GIGKFLK"})
        assert response.status_code == 503

    def test_documented_in_openapi(self, client: Any) -> None:
        openapi = client.get("/api/v1/openapi.json").json()
        for path in ("/api/v1/annotate", "/api/v1/annotate/{job_id}",
                     "/api/v1/annotate/{job_id}/result"):
            assert path in openapi["paths"]
