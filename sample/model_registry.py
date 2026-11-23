"""
Discovery of the trained models a user can choose between.

Steps 45-46 write one pickle per (task, feature block) into a namespace directory tree:

    <model_dir>/<peptide_type>/<scope>[/<target>]/<assay_type>/<task>/<feature_block>.pkl

The path IS the identity -- it says what kind of peptide the model predicts, tested against what,
measured how, and from which feature representation. This module turns that tree into a flat list
of ``(model_id, metadata)`` the webserver can render in a dropdown and the Celery task can load by
id, so the two never have to agree on a directory layout separately.

Only models that passed the quality gate are *written* -- a model that failed is recorded in the
report and deliberately not written -- but being on disk is not sufficient to be offered. A
rollout that dies part-way through (the last one lost 16 namespaces to `[Errno 5] Input/output
error`) leaves pickles from earlier passes behind, and those carry no holdout metrics and no
calibrated threshold: serving one means answering a prediction request with a model whose measured
performance nobody knows. So the served set is taken from the rollout's own progress file -- the
namespaces it recorded as ``written`` -- and not from the glob. See :func:`list_models`.
"""
import glob
import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("sample.model_registry")

#: Model id given to the single pre-namespace classifier from ``sample/ml.py``, when it is the
#: only thing present. Kept selectable so the webserver still works on a release built before
#: steps 45-46 were rewritten, rather than showing an empty dropdown.
LEGACY_MODEL_ID: str = "legacy/global"

#: Feature-block name the FLAML rollout (steps 45-46) writes under. Mirrors
#: ``sample.flaml_xgboost.FLAML_BLOCK_NAME`` by value rather than importing it: that module pulls
#: in FLAML and XGBoost at import time, and this one is imported by the webserver on every page.
FLAML_BLOCK_NAME: str = "flaml"

#: The one status in the rollout's progress file that means "a model was published for this
#: namespace". The others -- ``gate_failed``, ``error``, ``skipped_no_task``,
#: ``skipped_too_little_data`` -- all mean that run published nothing, whatever is on disk.
PUBLISHED_STATUS: str = "written"

#: Progress file the FLAML rollout maintains, beside the model root. It is the record of what
#: steps 45-46 actually finished, which is why it, not the directory listing, decides what is
#: served.
FLAML_PROGRESS_FILENAME: str = "flaml_rollout_progress.json"

#: Set to any non-empty value to offer every pickle on disk, including models from runs that never
#: finished. Diagnostics only -- it re-exposes models with no metrics and no calibrated threshold.
SERVE_ALL_ENV_VAR: str = "SAMPLE_SERVE_ALL_MODELS"


def default_model_dir() -> str:
    """
    :returns: Root of the namespace tree. ``SAMPLE_MODEL_DIR`` overrides it.
    """
    override = os.environ.get("SAMPLE_MODEL_DIR")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "sample_final_release", "models")


def _legacy_model_path() -> str:
    """
    :returns: Path to the pre-namespace ``model.pkl``.
    """
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "sample_final_release", "model.pkl")


def _report_metrics(model_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Reads steps 45-46's report so the dropdown can show each model's measured performance.

    :param model_dir: Namespace root; the report is expected beside it.
    :returns: ``model_id -> metrics`` mapping, empty when no report is present.
    """
    report_path = os.path.join(os.path.dirname(model_dir), "ml_report.json")
    if not os.path.exists(report_path):
        return {}
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read {report_path}: {e}")
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for record in report.get("models", []):
        namespace = record.get("namespace")
        block = record.get("feature_block")
        if namespace and block:
            out[f"{namespace}/{block}"] = {
                "n_samples": record.get("n_samples"),
                "holdout": record.get("holdout", {}),
                "holdout_ci95": record.get("holdout_ci95", {}),
                "learning_task": record.get("learning_task", "classification"),
                # The F1-calibrated decision cutoff (sample.ml_models._best_f1_threshold),
                # None for a regressor or for a model trained before calibration existed --
                # callers must fall back to the conventional 0.5 in that case.
                "threshold": record.get("threshold"),
            }
    return out


def _flaml_progress(model_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Reads the FLAML rollout's progress file: which namespaces steps 45-46 actually finished.

    :param model_dir: Namespace root; the file is expected beside it.
    :returns: ``namespace -> record``, empty when the file is absent or unreadable. Empty is the
        "do not filter" signal in :func:`list_models`, so an unreadable file costs the filtering,
        never the models.
    """
    path = os.path.join(os.path.dirname(model_dir.rstrip(os.sep)), FLAML_PROGRESS_FILENAME)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            completed = json.load(fh).get("completed")
    except (OSError, ValueError, AttributeError) as e:
        logger.warning(f"Could not read {path}: {e}")
        return {}
    return dict(completed) if isinstance(completed, dict) else {}


def _progress_metrics(progress: Dict[str, Dict[str, Any]],
                      entry: Dict[str, Any]) -> Dict[str, Any]:
    """
    The metrics for a FLAML-trained model, read from the rollout's progress file.

    ``ml_report.json`` only ever describes the ``train-ml`` feature-block matrix, so a FLAML model
    has no entry there and would otherwise show no performance at all. Its scores are taken from
    the progress file rather than from the model's own ``.dataset.json`` because that sidecar
    embeds the full train/test entry-id lists -- megabytes, parsed on every page load -- while the
    progress file is one small file covering every namespace. ``n_samples`` lives only in the
    sidecar, so it is left unset and renders as an em dash; the per-model dataset page shows the
    split in full.

    :param progress: :func:`_flaml_progress`'s mapping.
    :param entry: A registry entry.
    :returns: Metrics in the same shape :func:`_report_metrics` produces, or ``{}``.
    """
    if entry["feature_block"] != FLAML_BLOCK_NAME:
        return {}
    record = progress.get(entry["model_id"].rsplit("/", 1)[0]) or {}
    if not record.get("holdout"):
        return {}
    return {
        "n_samples": None,
        "holdout": record.get("holdout") or {},
        "holdout_ci95": record.get("holdout_ci95") or {},
        "learning_task": entry["learning_task"],
        "threshold": record.get("threshold"),
    }


def list_models(model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Enumerates every selectable model.

    Restricted to what the FLAML rollout published, whenever its progress file says anything was:
    a model is offered only if its namespace is recorded ``written`` there. That drops the pickles
    left behind by runs that errored part-way -- they have no holdout metrics and no calibrated
    threshold, so offering them would invite a prediction against a model of unmeasured quality --
    and it drops the ``train-ml`` feature-block matrix, which the Makefile keeps as a manual
    diagnostic outside the numbered pipeline rather than as something to serve.

    The filter is deliberately conditional. With no progress file (any release built before the
    rollout existed, and every test fixture) nothing is filtered, so discovery still degrades to
    "everything on disk" rather than to an empty dropdown. Set
    :data:`SERVE_ALL_ENV_VAR` to override it for diagnostics.

    :param model_dir: Namespace root; :func:`default_model_dir` when omitted.
    :returns: One dict per model with ``model_id``, ``path``, the namespace fields parsed back out
        of the path, and any metrics found in the report or the rollout's progress file. Sorted by
        id so the dropdown is stable between page loads.
    """
    root = model_dir or default_model_dir()
    metrics = _report_metrics(root)
    progress = _flaml_progress(root)
    models: List[Dict[str, Any]] = []

    for path in sorted(glob.glob(os.path.join(root, "**", "*.pkl"), recursive=True)):
        rel = os.path.relpath(path, root)
        model_id = os.path.splitext(rel)[0]
        parts = model_id.split(os.sep)
        if parts and parts[0] == "synergy":
            # Retired, and no longer under this root at all: the artifacts now live in
            # sample_final_release/models_retired/synergy, so this glob should not reach them.
            # The guard stays as a backstop, because a train-synergy run pointed at --model-dir
            # <served root> would re-create the namespace here, and serving it would be wrong
            # three times over. It is unreliable (only 2 of 18 scopes passed their held-out gate;
            # S. aureus classification scored ROC-AUC 0.482, below chance), it is unloadable
            # (train_synergy_models uses joblib.dump, so load_model's pickle.load raises
            # UnpicklingError on every one), and it is unservable (a synergy model needs a
            # peptide PLUS an antibiotic Morgan fingerprint, which
            # sample.inference.build_feature_frame cannot construct from a bare sequence).
            continue
        # <peptide_type>/<scope>[/<target>]/<assay_type>/<task>/<block> -- the optional target
        # segment is why this counts from the end rather than the start.
        entry: Dict[str, Any] = {
            "model_id": model_id.replace(os.sep, "/"),
            "path": path,
            "peptide_type": parts[0] if parts else "",
            "scope": parts[1] if len(parts) > 1 else "",
            "target": parts[2] if len(parts) > 5 else "",
            "assay_type": parts[-3] if len(parts) > 2 else "",
            "learning_task": parts[-2] if len(parts) > 1 else "classification",
            "feature_block": parts[-1],
        }
        entry["metrics"] = metrics.get(entry["model_id"]) or _progress_metrics(progress, entry)
        entry["label"] = _label(entry)
        models.append(entry)

    published = {namespace for namespace, record in progress.items()
                 if str(record.get("status")) == PUBLISHED_STATUS}
    if published and not os.environ.get(SERVE_ALL_ENV_VAR):
        served = [m for m in models
                  if m["feature_block"] == FLAML_BLOCK_NAME
                  and m["model_id"].rsplit("/", 1)[0] in published]
        # Only narrow to a non-empty set. If the progress file and the directory disagree
        # completely, the right failure is to keep offering what is on disk, not to serve nothing.
        if served:
            withheld = len(models) - len(served)
            if withheld:
                logger.info(
                    f"Serving {len(served)} published model(s); withholding {withheld} not "
                    f"recorded as '{PUBLISHED_STATUS}' by the rollout (set {SERVE_ALL_ENV_VAR}=1 "
                    f"to offer them anyway).")
            models = served

    if not models and os.path.exists(_legacy_model_path()):
        models.append({
            "model_id": LEGACY_MODEL_ID,
            "path": _legacy_model_path(),
            "peptide_type": "general", "scope": "general", "target": "",
            "assay_type": "any", "learning_task": "classification",
            "feature_block": "legacy",
            "metrics": {},
            "label": "General AMP classifier (legacy, pre-namespace)",
        })
    return models


def _label(entry: Dict[str, Any]) -> str:
    """
    :param entry: A registry entry.
    :returns: A human-readable one-line description for the dropdown.
    """
    what = entry["peptide_type"].replace("_", " ")
    against = f" vs {entry['target'].replace('_', ' ')}" if entry.get("target") else ""
    assay = entry.get("assay_type", "")
    assay_txt = "" if assay in ("any", "") else f", {assay.upper()}"
    return f"{what}{against}{assay_txt} — {entry['feature_block']} features"


def get_model_entry(model_id: str, model_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    :param model_id: Id from :func:`list_models`.
    :param model_dir: Namespace root; :func:`default_model_dir` when omitted.
    :returns: The matching registry entry, or ``None`` when the id is unknown.
    """
    for entry in list_models(model_dir):
        if entry["model_id"] == model_id:
            return entry
    return None


def get_model_dataset(model_id: str, model_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Reads the full validation report and train/test peptide split for one model.

    Steps 45-46 write ``<block>.dataset.json`` beside every kept model's ``.pkl``: the same record
    that went into the aggregate ``ml_report.json`` (holdout/CV/y-randomization scores, the
    normality-informed null comparison, the applicability domain's summary statistics, the quality
    gate outcome) plus ``train_entry_ids``/``test_entry_ids`` -- the actual peptides on each side of
    the holdout split. This is what the webserver's per-model "dataset" page and its
    ``/api/v1/models/{id}/dataset`` route serve.

    :param model_id: Id from :func:`list_models`.
    :param model_dir: Namespace root; :func:`default_model_dir` when omitted.
    :returns: The dataset record, or ``None`` when the model is unknown or predates this file (the
        legacy pre-namespace model, or a release built before steps 45-46 started writing it).
    """
    entry = get_model_entry(model_id, model_dir)
    if entry is None:
        return None
    dataset_path = os.path.splitext(entry["path"])[0] + ".dataset.json"
    if not os.path.exists(dataset_path):
        return None
    try:
        with open(dataset_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError) as e:
        logger.warning(f"Could not read {dataset_path}: {e}")
        return None


def load_model(model_id: str, model_dir: Optional[str] = None) -> Tuple[Any, List[str], Dict[str, Any]]:
    """
    Loads a model and the exact feature columns it was fitted on.

    The column list travels inside the pickle rather than being re-derived at inference time. A
    model must be fed its training columns in its training order; recomputing that from the
    current dataset would silently produce a different vector whenever the enrichment step gains
    or loses a column, and the prediction would be wrong rather than failing.

    :param model_id: Id from :func:`list_models`.
    :param model_dir: Namespace root; :func:`default_model_dir` when omitted.
    :returns: ``(estimator, feature_columns, metadata)``. `metadata` is `entry` (from
        :func:`get_model_entry`) with two keys merged in when the pickle carries them --
        ``"scaler"`` (a fitted :class:`~sklearn.preprocessing.StandardScaler`) and
        ``"applicability_domain"`` (from :func:`sample.ml_models._applicability_domain`) -- both
        ``None`` for a model trained before steps 45-46 started writing them (the legacy model, or an
        older release). Returned inside `metadata` rather than as new tuple elements so the three
        existing call sites that unpack this as `estimator, columns, entry` keep working.
    :raises FileNotFoundError: When the id is not in the registry.
    :raises ValueError: When the pickle does not carry the expected keys.
    """
    entry = get_model_entry(model_id, model_dir)
    if entry is None:
        raise FileNotFoundError(f"No model registered under id {model_id!r}")

    with open(entry["path"], "rb") as fh:
        payload = pickle.load(fh)

    estimator = payload.get("model") if isinstance(payload, dict) else payload
    if estimator is None:
        raise ValueError(f"Model {model_id!r} pickle contains no estimator.")

    # Steps 45-46 write "columns"; the legacy model.pkl writes "feature_columns". Both are checked
    # before falling back to the estimator's own record, so an older release stays usable.
    columns: List[str] = []
    if isinstance(payload, dict):
        columns = list(payload.get("columns") or payload.get("feature_columns") or [])
    if not columns:
        columns = list(getattr(estimator, "feature_names_in_", []) or [])
    if not columns:
        raise ValueError(
            f"Model {model_id!r} records no feature columns and the estimator does not expose "
            "feature_names_in_; it cannot be used for inference safely.")

    entry = dict(entry)
    entry["scaler"] = payload.get("scaler") if isinstance(payload, dict) else None
    entry["applicability_domain"] = payload.get("applicability_domain") if isinstance(payload, dict) else None
    return estimator, columns, entry
