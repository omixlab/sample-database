"""Tests for :mod:`sample.model_registry` — discovery and loading of trained models."""
import json
import os
import pickle
from typing import Any

import pytest

from sample import model_registry as MR


class _Estimator:
    """A stand-in for a fitted estimator. Real ones are large and slow to build."""

    def __init__(self, columns: Any = None) -> None:
        if columns is not None:
            self.feature_names_in_ = list(columns)

    def predict_proba(self, X: Any) -> Any:
        return [[0.4, 0.6]] * len(X)


def _write_model(root: str, namespace: str, block: str, payload: Any) -> str:
    """
    :param root: Model directory root.
    :param namespace: Namespace path, e.g. ``antibacterial/general/mic/classification``.
    :param block: Feature block name, which is the pickle's basename.
    :param payload: Object to pickle.
    :returns: The path written.
    """
    out_dir = os.path.join(root, *namespace.split("/"))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{block}.pkl")
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    return path


@pytest.fixture
def model_dir(tmp_path: Any) -> str:
    """:returns: A model root holding two models under one namespace."""
    root = str(tmp_path / "models")
    ns = "antibacterial/general/mic/classification"
    _write_model(root, ns, "descriptors",
                 {"model": _Estimator(), "columns": ["charge", "sequence_length"],
                  "namespace": ns, "feature_block": "descriptors", "metrics": {"roc_auc": 0.9}})
    _write_model(root, ns, "esm2",
                 {"model": _Estimator(), "columns": ["esm2_dim_0", "esm2_dim_1"],
                  "namespace": ns, "feature_block": "esm2", "metrics": {"roc_auc": 0.95}})
    return root


class TestDefaultModelDir:
    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAMPLE_MODEL_DIR", "/custom/models")
        assert MR.default_model_dir() == "/custom/models"

    def test_falls_back_to_the_release_directory(self,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SAMPLE_MODEL_DIR", raising=False)
        assert MR.default_model_dir().endswith(os.path.join("sample_final_release", "models"))


class TestListModels:
    def test_finds_every_pickle(self, model_dir: str) -> None:
        assert len(MR.list_models(model_dir)) == 2

    def test_model_id_is_the_namespace_path(self, model_dir: str) -> None:
        ids = {m["model_id"] for m in MR.list_models(model_dir)}
        assert ids == {
            "antibacterial/general/mic/classification/descriptors",
            "antibacterial/general/mic/classification/esm2",
        }

    def test_parses_the_namespace_back_out_of_the_path(self, model_dir: str) -> None:
        entry = MR.get_model_entry(
            "antibacterial/general/mic/classification/descriptors", model_dir)
        assert entry["peptide_type"] == "antibacterial"
        assert entry["scope"] == "general"
        assert entry["assay_type"] == "mic"
        assert entry["learning_task"] == "classification"
        assert entry["feature_block"] == "descriptors"

    def test_sorted_for_a_stable_dropdown(self, model_dir: str) -> None:
        ids = [m["model_id"] for m in MR.list_models(model_dir)]
        assert ids == sorted(ids)

    def test_every_entry_has_a_human_label(self, model_dir: str) -> None:
        for entry in MR.list_models(model_dir):
            assert entry["label"].strip()
            assert entry["feature_block"] in entry["label"]

    def test_missing_directory_with_no_legacy_model_yields_nothing(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(MR, "_legacy_model_path",
                            lambda: str(tmp_path / "absent" / "model.pkl"))
        assert MR.list_models(str(tmp_path / "nope")) == []

    def test_falls_back_to_the_legacy_model_when_none_are_namespaced(
            self, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        # A release built before steps 45-46 were rewritten has only the single pre-namespace
        # model.pkl. Offering it keeps the webserver working instead of showing an empty
        # dropdown; it is deliberately NOT offered once real models exist.
        legacy = tmp_path / "model.pkl"
        with open(legacy, "wb") as fh:
            pickle.dump({"model": _Estimator(), "feature_columns": ["charge"]}, fh)
        monkeypatch.setattr(MR, "_legacy_model_path", lambda: str(legacy))

        models = MR.list_models(str(tmp_path / "nope"))
        assert [m["model_id"] for m in models] == [MR.LEGACY_MODEL_ID]

    def test_legacy_model_is_suppressed_once_real_models_exist(
            self, model_dir: str, tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        legacy = tmp_path / "model.pkl"
        with open(legacy, "wb") as fh:
            pickle.dump({"model": _Estimator(), "feature_columns": ["charge"]}, fh)
        monkeypatch.setattr(MR, "_legacy_model_path", lambda: str(legacy))

        assert MR.LEGACY_MODEL_ID not in {m["model_id"] for m in MR.list_models(model_dir)}

    def test_attaches_metrics_from_the_report(self, model_dir: str, tmp_path: Any) -> None:
        report = {"models": [{
            "namespace": "antibacterial/general/mic/classification",
            "feature_block": "descriptors", "n_samples": 500,
            "holdout": {"roc_auc": 0.91}, "cv_mean": {"roc_auc": 0.88},
            "learning_task": "classification"}]}
        with open(os.path.join(os.path.dirname(model_dir), "ml_report.json"),
                  "w", encoding="utf-8") as fh:
            json.dump(report, fh)

        entry = MR.get_model_entry(
            "antibacterial/general/mic/classification/descriptors", model_dir)
        assert entry["metrics"]["n_samples"] == 500
        assert entry["metrics"]["holdout"]["roc_auc"] == 0.91

    def test_unreadable_report_does_not_break_discovery(self, model_dir: str) -> None:
        # A corrupt report must cost the metrics, not the models.
        with open(os.path.join(os.path.dirname(model_dir), "ml_report.json"),
                  "w", encoding="utf-8") as fh:
            fh.write("{not json")
        assert len(MR.list_models(model_dir)) == 2


class TestGetModelEntry:
    def test_unknown_id_is_none(self, model_dir: str) -> None:
        assert MR.get_model_entry("no/such/model", model_dir) is None


class TestLoadModel:
    def test_returns_estimator_columns_and_metadata(self, model_dir: str) -> None:
        estimator, columns, entry = MR.load_model(
            "antibacterial/general/mic/classification/descriptors", model_dir)
        assert hasattr(estimator, "predict_proba")
        assert columns == ["charge", "sequence_length"]
        assert entry["feature_block"] == "descriptors"

    def test_columns_come_from_the_pickle_not_recomputed(self, model_dir: str) -> None:
        # A model must be fed its training columns in training order. Re-deriving them from the
        # current dataset would silently change the vector whenever enrichment gains a column.
        _, columns, _ = MR.load_model(
            "antibacterial/general/mic/classification/esm2", model_dir)
        assert columns == ["esm2_dim_0", "esm2_dim_1"]

    def test_unknown_id_raises(self, model_dir: str) -> None:
        with pytest.raises(FileNotFoundError):
            MR.load_model("no/such/model", model_dir)

    def test_accepts_the_legacy_feature_columns_key(self, tmp_path: Any) -> None:
        # The pre-namespace model.pkl spells it "feature_columns"; steps 45-46 write "columns".
        root = str(tmp_path / "models")
        _write_model(root, "general/general/any/classification", "legacy",
                     {"model": _Estimator(), "feature_columns": ["charge"],
                      "metrics": {}})
        _, columns, _ = MR.load_model(
            "general/general/any/classification/legacy", root)
        assert columns == ["charge"]

    def test_falls_back_to_estimator_feature_names(self, tmp_path: Any) -> None:
        root = str(tmp_path / "models")
        _write_model(root, "general/general/any/classification", "bare",
                     {"model": _Estimator(columns=["charge", "logp"])})
        _, columns, _ = MR.load_model("general/general/any/classification/bare", root)
        assert columns == ["charge", "logp"]

    def test_refuses_a_model_with_no_recoverable_columns(self, tmp_path: Any) -> None:
        # Scoring against an unknown feature order produces a confident wrong answer, so this
        # must fail rather than guess.
        root = str(tmp_path / "models")
        _write_model(root, "general/general/any/classification", "opaque",
                     {"model": _Estimator()})
        with pytest.raises(ValueError, match="feature columns"):
            MR.load_model("general/general/any/classification/opaque", root)

    def test_refuses_a_pickle_with_no_estimator(self, tmp_path: Any) -> None:
        root = str(tmp_path / "models")
        _write_model(root, "general/general/any/classification", "empty",
                     {"columns": ["charge"]})
        with pytest.raises(ValueError):
            MR.load_model("general/general/any/classification/empty", root)

    def test_scaler_and_applicability_domain_are_merged_into_entry(self, tmp_path: Any) -> None:
        root = str(tmp_path / "models")
        # A real fitted scaler, not a bare object() -- a plain object() re-constructs to a
        # non-identical, non-equal instance across a real pickle round trip, which is not the
        # thing this test wants to check.
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit([[1.0], [2.0], [3.0]])
        ad = {"k": 5, "threshold": 1.2}
        _write_model(root, "general/general/any/classification", "withad",
                     {"model": _Estimator(columns=["charge"]), "scaler": scaler,
                      "applicability_domain": ad})
        _, _, entry = MR.load_model("general/general/any/classification/withad", root)
        assert entry["scaler"] is not None
        assert entry["scaler"].mean_ == pytest.approx(scaler.mean_)
        assert entry["applicability_domain"] == ad

    def test_missing_scaler_and_ad_are_none_not_a_missing_key(self, model_dir: str) -> None:
        # A model trained before steps 45-46 carried these (the legacy model, or an older release)
        # must be usable without a KeyError, and reported as "not checked", not "out of domain".
        _, _, entry = MR.load_model(
            "antibacterial/general/mic/classification/descriptors", model_dir)
        assert entry["scaler"] is None
        assert entry["applicability_domain"] is None
