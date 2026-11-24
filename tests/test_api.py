"""
Tests for the FastAPI web layer — HTML pages and the ``/api/v1`` JSON API.

Two things get most of the attention here, because they are the two ways this layer can be wrong
in a way nobody notices:

* **Decoys must never be served.** They share a table with real peptides and are distinguishable
  only by a flag. The Flask predecessor had no filter and returned them from search.
* **Route parity.** Every path that worked before the framework migration must still work; a
  silently-dropped route looks identical to a route nobody visited.

Predictions run with Celery in eager mode against a stub model, so the tests exercise the request
contract rather than a real estimator.
"""
import os
import pickle
import sqlite3
from typing import Any

import pytest

from sample import prediction_store


class _StubClassifier:
    """A fitted classifier stand-in; module scope so pickle can resolve it by qualified name."""

    feature_names_in_ = ["charge", "sequence_length"]

    def predict_proba(self, X: Any) -> Any:
        import numpy as np
        return np.column_stack([np.full(len(X), 0.15), np.full(len(X), 0.85)])


#: Model id matching the stub written by the `client` fixture.
MODEL_ID = "general/general/any/classification/descriptors"

#: Paths the dashboard served under Flask. Every one must still return 200 after the migration.
HTML_ROUTES = ("/", "/datasets", "/extractions", "/models", "/databases", "/analytics")


@pytest.fixture
def sample_db(tmp_path: Any) -> str:
    """
    Builds a miniature release database with the real schema, two peptides and one decoy.

    Built rather than copied from ``data/`` so the suite does not depend on a pipeline run having
    happened, and so the decoy assertions have a known decoy to find.

    :param tmp_path: pytest temporary directory.
    :returns: Path to the database.
    """
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
            activity_group TEXT, value REAL, unit TEXT, value_mM REAL, value_ug_ml REAL,
            potency REAL, target_organism TEXT, target_taxid INTEGER, activity_type TEXT,
            mechanism_of_action TEXT, source_accession TEXT, source_reference_index TEXT);
        CREATE TABLE papers (id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT,
            source_id TEXT, UNIQUE(source_type, source_id));
        CREATE TABLE peptide_references (peptide_id TEXT, paper_id INTEGER,
            UNIQUE(peptide_id, paper_id));
        CREATE VIRTUAL TABLE peptides_fts USING fts5(entry_id UNINDEXED, sequence,
            literature, patent, tokenize="unicode61");
    """)
    conn.executemany(
        "INSERT INTO peptides (entry_id, sequence, SMILES, source, is_decoy, LITERATURE, PATENT) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [("SAMPLE0000000001", "GIGKFLHSAKKFGKAFVGEIMNS", "CCO", None, 0,
          "pubmed:abc.json", None),
         ("SAMPLE0000000002", "ILPWKWPWWPWRR", None, None, 0, None,
          "google_patents:US8394762B2.json"),
         ("SAMPLE_DECOY_00000001", "MFGLGGAAPQISSQQKLQAAEAELDMV", None, "decoy_generator", 1,
          None, None)])
    conn.execute("INSERT INTO descriptors VALUES (?, ?, ?, ?, ?, ?, ?)",
                 ("SAMPLE0000000001", 2466.9, -8.1, 30, 28, 3.9, 23))
    conn.execute(
        "INSERT INTO activities (entry_id, activity_group, value, unit, value_mM, value_ug_ml, "
        "potency, target_organism, target_taxid, activity_type, mechanism_of_action) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("SAMPLE0000000001", "antibacterial", 16.0, "ug/mL", 0.006485872957963435, 16.0,
         5.188031562322899, "Escherichia coli", 562, "MIC", None))
    conn.execute(
        "INSERT INTO activities (entry_id, activity_group, value, unit, target_organism, "
        "target_taxid, activity_type, mechanism_of_action) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("SAMPLE_DECOY_00000001", "antibacterial", 99.0, "ug/mL", "Escherichia coli", 562,
         "MIC", None))
    conn.execute("INSERT INTO papers (id, source_type, source_id) VALUES (1, 'pubmed', '12345')")
    conn.execute("INSERT INTO peptide_references VALUES ('SAMPLE0000000001', 1)")
    conn.execute(
        "INSERT INTO peptides_fts (entry_id, sequence, literature, patent) "
        "SELECT entry_id, sequence, COALESCE(LITERATURE,''), COALESCE(PATENT,'') "
        "FROM peptides WHERE is_decoy = 0")
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def client(tmp_path: Any, sample_db: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    """
    :returns: A ``TestClient`` wired to the fixture database, a temp prediction store and a stub
        model, with Celery running tasks inline.
    """
    from fastapi.testclient import TestClient

    from sample import db, model_registry, tasks
    from sample.api.app import app

    monkeypatch.setenv("SAMPLE_DB", sample_db)
    monkeypatch.setattr(db, "default_db_path", lambda: sample_db)
    db.get_stats.cache_clear()

    predictions_db = str(tmp_path / "predictions.db")
    prediction_store.init_db(predictions_db)
    monkeypatch.setenv("SAMPLE_PREDICTIONS_DB", predictions_db)

    model_root = str(tmp_path / "models")
    out = os.path.join(model_root, "general", "general", "any", "classification")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "descriptors.pkl"), "wb") as fh:
        pickle.dump({"model": _StubClassifier(), "columns": ["charge", "sequence_length"],
                     "namespace": "general/general/any/classification",
                     "feature_block": "descriptors", "metrics": {}}, fh)
    monkeypatch.setenv("SAMPLE_MODEL_DIR", model_root)
    monkeypatch.setattr(model_registry, "default_model_dir", lambda: model_root)

    monkeypatch.setitem(tasks.celery_app.conf, "task_always_eager", True)
    monkeypatch.setattr(tasks, "broker_available", lambda: True)

    return TestClient(app)


class TestHtmlParity:
    """Every route that worked under Flask still works."""

    @pytest.mark.parametrize("path", HTML_ROUTES)
    def test_page_renders(self, client: Any, path: str) -> None:
        assert client.get(path).status_code == 200

    def test_peptide_detail_renders(self, client: Any) -> None:
        assert client.get("/search/peptide/SAMPLE0000000001").status_code == 200

    def test_peptide_detail_json(self, client: Any) -> None:
        payload = client.get("/search/peptide/SAMPLE0000000001/json").json()
        assert payload["entry_id"] == "SAMPLE0000000001"
        assert payload["activities"] and payload["descriptors"]

    def test_unknown_peptide_is_404(self, client: Any) -> None:
        assert client.get("/search/peptide/NOPE").status_code == 404

    def test_detail_page_shows_descriptors(self, client: Any) -> None:
        # Regression: the template reads `peptide.molecular_weight` while the old route passed
        # descriptors under a separate name, so the "no descriptors" branch always won.
        body = client.get("/search/peptide/SAMPLE0000000001").text
        assert "No computed descriptors available" not in body

    def test_detail_page_shows_admet_panel(self, client: Any) -> None:
        # The ADMET-lite panel is computed from the sequence and rendered on the peptide page,
        # including the endpoints it deliberately does not fake (membrane permeability).
        body = client.get("/search/peptide/SAMPLE0000000001").text
        assert "ADMET-lite profile" in body
        assert "membrane permeability" in body

    def test_detail_page_shows_annotation_when_present(
            self, client: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from sample import peptide_annotations as pa
        db_path = str(tmp_path / "ann.sqlite")
        monkeypatch.setenv("SAMPLE_PEPTIDE_ANNOTATIONS_DB", db_path)
        pa.init_db(db_path)
        pa.save("SAMPLE0000000001", {
            "entry_id": "SAMPLE0000000001", "is_circular": False,
            "predictors": {"macrel_amp_probability": {"label": "macrel AMP probability",
                                                       "value": 0.91}},
            "pfam_domains": ["PF00711"], "family": {"family_id": "SAMPLE_FAM_1"},
            "near_identity_hits": [{"entry_id": "OTHER1", "identity": 96.5, "mismatches": 1}],
            "hmm_hits": [{"family_id": "SAMPLE_FAM_1", "evalue": 1e-8}],
        }, db_path=db_path)
        body = client.get("/search/peptide/SAMPLE0000000001").text
        assert "macrel AMP probability" in body
        assert "OTHER1" in body

    def test_detail_page_shows_circular_caveat(
            self, client: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        from sample import peptide_annotations as pa
        db_path = str(tmp_path / "ann.sqlite")
        monkeypatch.setenv("SAMPLE_PEPTIDE_ANNOTATIONS_DB", db_path)
        pa.init_db(db_path)
        pa.save("SAMPLE0000000001", {
            "entry_id": "SAMPLE0000000001", "is_circular": True,
            "circular_caveat": pa.CIRCULAR_CAVEAT, "predictors": {},
        }, db_path=db_path)
        body = client.get("/search/peptide/SAMPLE0000000001").text
        assert "circular" in body.lower()
        assert pa.CIRCULAR_CAVEAT in body

    def test_detail_page_omits_annotation_without_a_result(
            self, client: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAMPLE_PEPTIDE_ANNOTATIONS_DB", str(tmp_path / "empty.sqlite"))
        body = client.get("/search/peptide/SAMPLE0000000001").text
        assert 'id="annotation-card"' not in body

    def test_detail_page_links_hemolysis_gap_to_the_real_annotation(
            self, client: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        # Once build-annotations has scored this peptide, the ADMET panel's hemolysis gap should
        # show the real HemoPI2 value rather than the static "run the annotator" pointer.
        from sample import peptide_annotations as pa
        db_path = str(tmp_path / "ann.sqlite")
        monkeypatch.setenv("SAMPLE_PEPTIDE_ANNOTATIONS_DB", db_path)
        pa.init_db(db_path)
        pa.save("SAMPLE0000000001", {
            "entry_id": "SAMPLE0000000001", "is_circular": False,
            "predictors": {"hemopi2_score": {"label": "HemoPI2 hemolysis score", "value": 0.42}},
        }, db_path=db_path)
        body = client.get("/search/peptide/SAMPLE0000000001").text
        assert "HemoPI2 hemolysis score = 0.420" in body
        assert "run the annotator for this peptide" not in body

    def test_dataset_download_rejects_unknown_key(self, client: Any) -> None:
        assert client.get("/datasets/download/../../etc/passwd").status_code in (404, 400)


class TestDecoyExclusion:
    """The invariant the whole repository layer exists to hold."""

    def test_collection_excludes_decoys(self, client: Any) -> None:
        payload = client.get("/api/v1/peptides?limit=500").json()
        assert payload["total"] == 2
        assert all("DECOY" not in item["entry_id"] for item in payload["items"])

    def test_is_decoy_is_not_a_public_field(self, client: Any) -> None:
        # A field that could only ever read False would imply one might sometimes be returned.
        payload = client.get("/api/v1/peptides?limit=500").json()
        assert all("is_decoy" not in item for item in payload["items"])

    def test_decoy_cannot_be_fetched_by_id(self, client: Any) -> None:
        assert client.get("/api/v1/peptides/SAMPLE_DECOY_00000001").status_code == 404

    def test_decoy_not_reachable_by_its_own_sequence(self, client: Any) -> None:
        # This is exactly what the Flask search did return.
        payload = client.get("/api/v1/peptides?q=MFGLGGAAPQISSQQ").json()
        assert payload["total"] == 0

    def test_decoy_activities_are_excluded(self, client: Any) -> None:
        payload = client.get("/api/v1/activities?limit=500").json()
        assert payload["total"] == 1
        assert payload["items"][0]["value"] == 16.0

    def test_full_text_search_excludes_decoys(self, client: Any) -> None:
        assert client.get("/api/v1/peptides/search?q=MFGLGGAAPQISSQQ").json()["total"] == 0

    def test_stats_separate_real_from_decoy(self, client: Any) -> None:
        stats = client.get("/api/v1/stats").json()
        assert stats["consensus_count"] == 2
        assert stats["decoys_count"] == 1


class TestPeptideEndpoints:
    def test_pagination_reports_totals(self, client: Any) -> None:
        payload = client.get("/api/v1/peptides?limit=1").json()
        assert len(payload["items"]) == 1
        assert payload["total"] == 2
        assert payload["limit"] == 1 and payload["offset"] == 0

    def test_offset_walks_the_collection(self, client: Any) -> None:
        first = client.get("/api/v1/peptides?limit=1&offset=0").json()["items"][0]
        second = client.get("/api/v1/peptides?limit=1&offset=1").json()["items"][0]
        assert first["entry_id"] != second["entry_id"]

    def test_offset_past_the_end_is_empty_not_an_error(self, client: Any) -> None:
        payload = client.get("/api/v1/peptides?limit=10&offset=999").json()
        assert payload["items"] == [] and payload["total"] == 2

    @pytest.mark.parametrize("limit", [0, -1, 501, 10_000])
    def test_limit_bounds_are_enforced(self, client: Any, limit: int) -> None:
        assert client.get(f"/api/v1/peptides?limit={limit}").status_code == 422

    def test_filter_by_activity_group(self, client: Any) -> None:
        assert client.get("/api/v1/peptides?activity_group=antibacterial").json()["total"] == 1
        assert client.get("/api/v1/peptides?activity_group=antifungal").json()["total"] == 0

    def test_filter_by_target_organism(self, client: Any) -> None:
        assert client.get("/api/v1/peptides?target=Escherichia").json()["total"] == 1

    def test_detail_carries_descriptors_activities_and_papers(self, client: Any) -> None:
        payload = client.get("/api/v1/peptides/SAMPLE0000000001").json()
        assert payload["descriptors"]["sequence_length"] == 23
        assert payload["activities"][0]["activity_type"] == "MIC"
        assert payload["papers"][0]["source_id"] == "12345"

    def test_reference_holds_citations_not_filenames(self, client: Any) -> None:
        """
        ``Reference`` used to be *overwritten* with ``LITERATURE`` and ``PATENT`` joined together,
        so the peptide page listed ``pubmed_old:<uuid>.json`` under "Literature & Patent
        References" and the search results showed the same under "Reference Source". Those columns
        hold extraction filenames -- provenance for the Source Documents links, never a citation.

        The patent number is still reachable: it is parsed out of the ``PATENT`` token into
        ``papers`` as a Google Patents link, which is asserted below.
        """
        payload = client.get("/api/v1/peptides/SAMPLE0000000002").json()

        reference = payload.get("Reference") or ""
        assert "US8394762B2.json" not in reference
        assert not reference.startswith("File: ")
        assert any("US8394762B2" in str(p.get("source_id")) for p in payload["papers"]), (
            "the patent must still be linkable even though it is no longer a 'reference'")


class TestFullTextSearch:
    def test_finds_a_patent_number(self, client: Any) -> None:
        payload = client.get("/api/v1/peptides/search?q=US8394762B2").json()
        assert [i["entry_id"] for i in payload["items"]] == ["SAMPLE0000000002"]

    def test_finds_a_sequence(self, client: Any) -> None:
        assert client.get(
            "/api/v1/peptides/search?q=GIGKFLHSAKKFGKAFVGEIMNS").json()["total"] == 1

    def test_boolean_operators(self, client: Any) -> None:
        assert client.get(
            "/api/v1/peptides/search?q=google_patents AND US8394762B2").json()["total"] == 1
        assert client.get(
            "/api/v1/peptides/search?q=US8394762B2 NOT google_patents").json()["total"] == 0

    def test_prefix_wildcard(self, client: Any) -> None:
        assert client.get("/api/v1/peptides/search?q=GIGKFLH*").json()["total"] >= 1

    def test_malformed_query_is_an_empty_page_not_an_error(self, client: Any) -> None:
        # This backs a search box; a stray quote must not 500.
        response = client.get('/api/v1/peptides/search?q="unclosed')
        assert response.status_code == 200
        assert response.json()["total"] == 0


class TestOpenApi:
    def test_document_is_valid(self, client: Any) -> None:
        document = client.get("/api/v1/openapi.json").json()
        assert document["openapi"].startswith("3.")
        assert document["info"]["title"] == "sAMPle API"

    def test_documents_every_api_route(self, client: Any) -> None:
        paths = client.get("/api/v1/openapi.json").json()["paths"]
        for expected in ("/api/v1/peptides", "/api/v1/peptides/{entry_id}", "/api/v1/activities",
                         "/api/v1/models", "/api/v1/predict", "/api/v1/jobs/{job_id}",
                         "/api/v1/stats", "/api/v1/search/blast"):
            assert expected in paths, f"{expected} is not documented"

    def test_html_pages_are_not_in_the_api_schema(self, client: Any) -> None:
        paths = client.get("/api/v1/openapi.json").json()["paths"]
        assert not any(p in paths for p in HTML_ROUTES)

    def test_swagger_ui_renders(self, client: Any) -> None:
        assert client.get("/api/v1/docs").status_code == 200


class TestPredictionApi:
    def test_submit_returns_202_and_a_job_id(self, client: Any) -> None:
        import uuid
        response = client.post("/api/v1/predict", json={
            "model_id": MODEL_ID, "sequences": "GIGKFLHSAKKFGKAFVGEIMNS"})
        assert response.status_code == 202
        uuid.UUID(response.json()["job_id"])

    def test_job_reaches_a_terminal_state(self, client: Any) -> None:
        job_id = client.post("/api/v1/predict", json={
            "model_id": MODEL_ID, "sequences": "GIGKFLHSAKKFGKAFVGEIMNS"}).json()["job_id"]
        status = client.get(f"/api/v1/jobs/{job_id}").json()
        assert status["done"] is True and status["status"] == prediction_store.STATUS_SUCCESS

    def test_results_carry_probabilities(self, client: Any) -> None:
        job_id = client.post("/api/v1/predict", json={
            "model_id": MODEL_ID, "peptides": ["GIGKFLHSAKK"]}).json()["job_id"]
        results = client.get(f"/api/v1/jobs/{job_id}/results").json()["results"]
        assert results[0]["probability"] == pytest.approx(0.85)
        assert results[0]["predicted_label"] == "active"

    def test_invalid_row_fails_alone(self, client: Any) -> None:
        job_id = client.post("/api/v1/predict", json={
            "model_id": MODEL_ID, "peptides": ["GIGKFLHSAKK", "ZZZZ"]}).json()["job_id"]
        results = client.get(f"/api/v1/jobs/{job_id}/results").json()["results"]
        assert sum(1 for r in results if r["error"]) == 1
        assert sum(1 for r in results if r["probability"] is not None) == 1

    def test_unknown_model_is_404(self, client: Any) -> None:
        assert client.post("/api/v1/predict", json={
            "model_id": "no/such/model", "sequences": "GIGKFLHSAKK"}).status_code == 404

    def test_no_sequences_is_400(self, client: Any) -> None:
        assert client.post("/api/v1/predict", json={
            "model_id": MODEL_ID, "sequences": ""}).status_code == 400

    def test_all_invalid_is_400(self, client: Any) -> None:
        assert client.post("/api/v1/predict", json={
            "model_id": MODEL_ID, "sequences": "ZZZZ"}).status_code == 400

    def test_oversized_job_is_400(self, client: Any) -> None:
        from sample.inference import MAX_PEPTIDES_PER_JOB
        assert client.post("/api/v1/predict", json={
            "model_id": MODEL_ID,
            "peptides": ["GIGKFLHSAKK"] * (MAX_PEPTIDES_PER_JOB + 1)}).status_code == 400

    def test_unknown_job_is_404(self, client: Any) -> None:
        assert client.get("/api/v1/jobs/not-a-job").status_code == 404
        assert client.get("/api/v1/jobs/not-a-job/results").status_code == 404

    def test_unreachable_broker_is_503(self, client: Any,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
        from sample import tasks
        monkeypatch.setattr(tasks, "broker_available", lambda: False)
        assert client.post("/api/v1/predict", json={
            "model_id": MODEL_ID, "sequences": "GIGKFLHSAKK"}).status_code == 503

    def test_models_endpoint_lists_the_stub(self, client: Any) -> None:
        assert MODEL_ID in [m["model_id"] for m in client.get("/api/v1/models").json()]


class TestLegacyPredictionRoutes:
    """The paths the shipped templates call. They must keep working after the migration."""

    def test_submit_and_poll(self, client: Any) -> None:
        response = client.post("/predict", json={
            "model_id": MODEL_ID, "sequences": "GIGKFLHSAKKFGKAFVGEIMNS"})
        assert response.status_code == 202
        payload = response.json()
        assert payload["status_url"] == f"/predict/status/{payload['job_id']}"

        status = client.get(payload["status_url"]).json()
        assert status["done"] is True

    def test_form_encoded_submission(self, client: Any) -> None:
        # models.html can post a form as well as JSON.
        response = client.post("/predict", data={
            "model_id": MODEL_ID, "peptides": "GIGKFLHSAKKFGKAFVGEIMNS"})
        assert response.status_code == 202

    def test_results_page_and_json(self, client: Any) -> None:
        job_id = client.post("/predict", json={
            "model_id": MODEL_ID, "sequences": "GIGKFLHSAKK"}).json()["job_id"]
        assert client.get(f"/predict/{job_id}").status_code == 200
        assert len(client.get(f"/predict/{job_id}/json").json()["results"]) == 1

    def test_unknown_job_page_is_404(self, client: Any) -> None:
        assert client.get("/predict/not-a-job").status_code == 404
