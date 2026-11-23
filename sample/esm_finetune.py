"""
End-to-end fine-tuning of ESM-2's smallest checkpoint (``facebook/esm2_t6_8M_UR50D``, 6 layers,
8M params) for a single classification or regression task -- the full backbone is unfrozen and
trained together with a small head, rather than the ESM-2 embedding being computed once and
frozen (see :mod:`sample.enrichment`/:mod:`sample.inference`, which is what every other model in
this project trains against).

Feature contract: ``columns == ["sequence"]`` -- one raw peptide sequence string per row, not a
precomputed numeric feature. Every call site that builds a feature matrix
(:func:`sample.ml_models.train_task`, :func:`sample.benchmark._score_model_from_dataset`,
:func:`sample.inference.build_feature_frame`) has an explicit branch for this
(:func:`sample.ml_models._is_sequence_input`) rather than routing a sequence string through the
``pd.to_numeric(...).fillna(0.0)`` coercion every other block uses, which would silently zero it
out.

:class:`ESM2FineTuneEstimator` is the ``model_factory``
:func:`sample.ml_models.fit_and_validate`/``_y_randomization`` call into -- it has the same
construction signature as :func:`sample.ml_models._make_model`
(``factory(is_regression=..., **overrides)``), the same ``.fit(X, y)``/``.predict(X)``/
``.predict_proba(X)`` contract as the XGBoost/RandomForest estimators it replaces, and accepts
``scale_pos_weight`` the same way (translated to ``BCEWithLogitsLoss``'s ``pos_weight``, matching
the class-imbalance fix already validated for the XGBoost path).
"""
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("sample.esm_finetune")

#: The single ESM-2 checkpoint used everywhere in this project.
ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

#: Generous headroom over this project's own peptide length distribution (median molecular weight
#: 2,268 Da, comfortably under 100 residues) -- truncation would only ever bite an outlier, never
#: a typical sequence.
MAX_LENGTH = 256

#: A single flat learning rate for backbone and head together, rather than per-layer
#: discriminative rates -- that technique earns its complexity on much deeper backbones (650M+
#: params) where early layers are markedly more general-purpose than late ones; at 6 layers/8M
#: params there is comparatively little to differentiate, and this is the standard starting point
#: for fine-tuning an already-pretrained transformer at this scale.
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
DROPOUT = 0.15
BATCH_SIZE = 32
MAX_EPOCHS = 30
EARLY_STOP_PATIENCE = 4
#: Fraction of the TRAINING portion carved out for early stopping -- an internal slice,
#: never fit_and_validate's own outer holdout.
VAL_FRACTION = 0.1
RANDOM_STATE = 42
HEAD_HIDDEN = 64


def _extract_sequences(X: Any) -> List[str]:
    """
    :param X: Either a DataFrame with a ``"sequence"`` column
        (:func:`sample.benchmark._score_model_from_dataset`'s cached-feature path), or a 2D
        array-like whose first column holds the sequence string
        (:func:`sample.ml_models.train_task`'s ``columns == ["sequence"]`` path, a numpy object
        array).
    :returns: Plain sequence strings, in row order.
    """
    if isinstance(X, pd.DataFrame):
        return [str(s) for s in X["sequence"].tolist()]
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 2:
        return [str(s) for s in arr[:, 0]]
    return [str(s) for s in arr]


class ESM2FineTuneEstimator:
    """
    Scikit-learn-compatible wrapper fine-tuning the full ESM-2 backbone end-to-end for one
    classification or regression task.

    Constructed exactly like :func:`sample.ml_models._make_model`
    (``factory(is_regression=..., **overrides)``) so it can be passed as
    :func:`sample.ml_models.fit_and_validate`'s ``model_factory`` without any special casing at
    the call site.
    """

    def __init__(self, is_regression: bool = False,
                 scale_pos_weight: Optional[float] = None, **_ignored: Any) -> None:
        self.is_regression = is_regression
        self.scale_pos_weight = scale_pos_weight
        self._backbone_state: Optional[dict] = None
        self._head_state: Optional[dict] = None
        self._hidden_size: Optional[int] = None

    # -- training ------------------------------------------------------------------------------

    def _build_modules(self, device: "Any") -> Any:
        import torch
        from transformers import AutoModel

        backbone = AutoModel.from_pretrained(ESM2_MODEL_NAME, local_files_only=True).to(device)
        hidden = backbone.config.hidden_size
        head = torch.nn.Sequential(
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden, HEAD_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(HEAD_HIDDEN, 1),
        ).to(device)
        return backbone, head, hidden

    @staticmethod
    def _pool(last_hidden_state: "Any", attention_mask: "Any") -> "Any":
        """Masked mean-pool over non-padding tokens -- matches :mod:`sample.inference`'s
        ``_embedding_frame``, not :mod:`sample.esm_enrichment`'s unmasked ``.mean(dim=1)`` (a
        separate, pre-existing inconsistency between how the cached ``esm2_dim_*`` columns were
        computed and how a live query is embedded -- irrelevant here since this model learns its
        own representation end-to-end rather than reproducing a cached column)."""
        import torch

        mask = attention_mask.unsqueeze(-1).float()
        summed = torch.sum(last_hidden_state * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def fit(self, X: Any, y: Any) -> "ESM2FineTuneEstimator":
        import torch
        from torch.optim import AdamW
        from transformers import AutoTokenizer

        sequences = _extract_sequences(X)
        labels = np.asarray(y, dtype=np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME, do_lower_case=False, local_files_only=True)
        backbone, head, hidden = self._build_modules(device)

        # Internal early-stopping split, carved from the passed-in training portion ONLY -- the
        # caller (fit_and_validate) never passes X_test/y_test here.
        rng = np.random.default_rng(RANDOM_STATE)
        n = len(sequences)
        perm = rng.permutation(n)
        n_val = max(1, int(round(n * VAL_FRACTION)))
        val_idx, fit_idx = perm[:n_val], perm[n_val:]
        if len(fit_idx) == 0:
            fit_idx, val_idx = perm, perm[:1]

        fit_seqs = [sequences[i] for i in fit_idx]
        fit_labels = labels[fit_idx]
        val_seqs = [sequences[i] for i in val_idx]
        val_labels = labels[val_idx]

        if self.is_regression:
            loss_fn: Any = torch.nn.MSELoss()
        else:
            pos_weight = (torch.tensor(self.scale_pos_weight, dtype=torch.float32, device=device)
                         if self.scale_pos_weight else None)
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = AdamW(list(backbone.parameters()) + list(head.parameters()),
                          lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        def _forward(seqs: List[str]) -> "Any":
            enc = tokenizer(seqs, padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt").to(device)
            out = backbone(**enc)
            pooled = self._pool(out.last_hidden_state, enc["attention_mask"])
            return head(pooled).squeeze(-1)

        best_val_loss = float("inf")
        best_state = None
        patience_left = EARLY_STOP_PATIENCE
        n_fit = len(fit_seqs)

        for epoch in range(MAX_EPOCHS):
            backbone.train()
            head.train()
            epoch_perm = rng.permutation(n_fit)
            for start in range(0, n_fit, BATCH_SIZE):
                batch_idx = epoch_perm[start:start + BATCH_SIZE]
                batch_seqs = [fit_seqs[i] for i in batch_idx]
                batch_labels = torch.tensor(fit_labels[batch_idx], dtype=torch.float32,
                                            device=device)
                optimizer.zero_grad()
                logits = _forward(batch_seqs)
                loss = loss_fn(logits, batch_labels)
                loss.backward()
                optimizer.step()

            backbone.eval()
            head.eval()
            with torch.no_grad():
                val_losses = []
                for start in range(0, len(val_seqs), BATCH_SIZE):
                    chunk_seqs = val_seqs[start:start + BATCH_SIZE]
                    chunk_labels = torch.tensor(val_labels[start:start + BATCH_SIZE],
                                                dtype=torch.float32, device=device)
                    logits = _forward(chunk_seqs)
                    val_losses.append(float(loss_fn(logits, chunk_labels)) * len(chunk_seqs))
                val_loss = sum(val_losses) / max(len(val_seqs), 1)

            logger.info(f"  [esm_finetune] epoch {epoch + 1}/{MAX_EPOCHS} val_loss={val_loss:.4f}")
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state = ({k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
                             {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
                patience_left = EARLY_STOP_PATIENCE
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        if best_state is None:
            best_state = (backbone.state_dict(), head.state_dict())
        self._backbone_state, self._head_state = best_state
        self._hidden_size = hidden
        return self

    # -- inference -------------------------------------------------------------------------------

    def _load_for_inference(self) -> Any:
        import torch
        from transformers import AutoModel

        if self._backbone_state is None or self._head_state is None:
            raise RuntimeError("ESM2FineTuneEstimator.predict called before fit().")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        backbone = AutoModel.from_pretrained(ESM2_MODEL_NAME, local_files_only=True).to(device)
        backbone.load_state_dict(self._backbone_state)
        backbone.eval()
        hidden = self._hidden_size or backbone.config.hidden_size
        head = torch.nn.Sequential(
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(hidden, HEAD_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT),
            torch.nn.Linear(HEAD_HIDDEN, 1),
        ).to(device)
        head.load_state_dict(self._head_state)
        head.eval()
        return backbone, head, device

    def _raw_scores(self, X: Any) -> np.ndarray:
        import torch
        from transformers import AutoTokenizer

        sequences = _extract_sequences(X)
        backbone, head, device = self._load_for_inference()
        tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME, do_lower_case=False, local_files_only=True)

        outputs: List[float] = []
        with torch.no_grad():
            for start in range(0, len(sequences), BATCH_SIZE):
                chunk = sequences[start:start + BATCH_SIZE]
                enc = tokenizer(chunk, padding=True, truncation=True, max_length=MAX_LENGTH,
                                return_tensors="pt").to(device)
                out = backbone(**enc)
                pooled = self._pool(out.last_hidden_state, enc["attention_mask"])
                logits = head(pooled).squeeze(-1)
                outputs.extend(logits.cpu().numpy().ravel().tolist())
        return np.asarray(outputs, dtype=np.float64)

    def predict_proba(self, X: Any) -> np.ndarray:
        """
        Class probabilities for the positive and negative class.

        :returns: ``(n, 2)`` -- ``[:, 1]`` is the sigmoid of the fine-tuned model's logit, the
            same shape :func:`sample.ml_models._predict_scores` already reads from every other
            classifier this project trains.
        """
        logits = self._raw_scores(X)
        p1 = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: Any) -> np.ndarray:
        if self.is_regression:
            return self._raw_scores(X)
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    # -- pickling ----------------------------------------------------------------------------
    # State dicts, not the live nn.Module/tokenizer -- the standard, cross-version-safe pattern;
    # torch/transformers are pinned identically across sample-ml/sample-web specifically so a
    # pickle carrying this state unpickles the same way on both sides of the train/serve boundary.

    def __getstate__(self) -> dict:
        return {
            "is_regression": self.is_regression,
            "scale_pos_weight": self.scale_pos_weight,
            "hidden_size": self._hidden_size,
            "backbone_state": self._backbone_state,
            "head_state": self._head_state,
        }

    def __setstate__(self, state: dict) -> None:
        self.is_regression = state["is_regression"]
        self.scale_pos_weight = state["scale_pos_weight"]
        self._hidden_size = state["hidden_size"]
        self._backbone_state = state["backbone_state"]
        self._head_state = state["head_state"]


#: Reduced validation rigor for this architecture, applied by train_finetune_tasks() below -- an
#: XGBoost fit is cheap enough to afford 5-fold OOF threshold calibration and 5 y-randomization
#: repeats (11 fits/task); a full transformer fine-tune is not. An explicit, reported
#: compute-cost tradeoff (see sample.ml_models.fit_and_validate's own n_calibration_folds/
#: n_y_randomizations docstrings), not silently equivalent to the XGBoost path's defaults.
FINETUNE_N_CALIBRATION_FOLDS = 1
FINETUNE_N_Y_RANDOMIZATIONS = 2


def train_finetune_tasks(namespaces: List[str],
                         dataset_csv: str = "data/sample_final_release/final_dataset.csv",
                         activities_csv: str = "data/sample_raw_release/activities.csv",
                         families_json: str = "data/families/families.json",
                         model_dir: str = "data/sample_final_release/models",
                         progress_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Fine-tunes :class:`ESM2FineTuneEstimator` for exactly the given task namespaces, reusing each
    task's existing holdout split (the same anomaly-filtered dataset/family-grouping that
    :func:`sample.ml_models.train_all` used to train its XGBoost counterpart reproduces the
    identical :class:`~sklearn.model_selection.StratifiedGroupKFold` split deterministically --
    verified this session to reproduce a real task's registered holdout AUC to 3 decimal places).

    Per namespace: if the fine-tuned model passes the quality gate
    (:func:`sample.ml_models.assess_quality`), the existing ``esm2.pkl``/``.dataset.json``/
    ``.holdout.fasta`` are backed up to ``esm2_xgboost_backup.*`` (once -- a pre-existing backup
    from an earlier run is never overwritten) and REPLACED by the fine-tuned model, so it becomes
    the production model :mod:`sample.inference`/:mod:`sample.benchmark` serve for that task
    without any change on their side. If it does not pass, the XGBoost model is left untouched.

    Interruptible and resumable: progress is written incrementally to `progress_path` after every
    namespace (not just at the end), and re-calling this with the same `progress_path` skips any
    namespace already recorded there rather than re-training it.

    :param namespaces: Task namespaces to fine-tune, e.g.
        ``["antibacterial/general/any/classification"]`` -- must each resolve via
        :func:`sample.ml_models.build_tasks` against `dataset_csv`/`activities_csv`.
    :param dataset_csv: The assembled dataset (with an ``is_anomaly`` column filtered out here,
        matching :func:`sample.ml_models.train_all`'s own ``filter_anomalies=True`` convention).
    :param activities_csv: Step 19's activity table.
    :param families_json: Curated family assignment for the train/holdout grouping.
    :param model_dir: Namespace root to write/replace models under.
    :param progress_path: Where to persist per-namespace progress;
        ``<model_dir>/../finetune_rollout_progress.json`` when omitted. Shared across separate
        invocations (e.g. one CLI command for antibacterial, another for anticancer, a third for
        everything else) so each namespace is only ever trained once regardless of which
        invocation's turn it was.
    :returns: ``{"replaced": [...], "gate_failed": [...], "errors": [...]}`` -- the namespaces
        this call itself processed, not the full accumulated progress file.
    """
    import gc
    import json
    import os
    import pickle
    import shutil
    import time

    import pandas as pd

    from sample import ml_models as M
    from sample.families import write_fasta

    if progress_path is None:
        progress_path = os.path.join(os.path.dirname(model_dir.rstrip("/")),
                                     "finetune_rollout_progress.json")

    def _load_progress() -> Dict[str, Any]:
        if os.path.exists(progress_path):
            with open(progress_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"completed": {}}

    def _save_progress(progress: Dict[str, Any]) -> None:
        tmp = progress_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(progress, fh, indent=2)
        os.replace(tmp, progress_path)

    progress = _load_progress()
    done = set(progress["completed"].keys())
    remaining = [ns for ns in namespaces if ns not in done]
    this_call: Dict[str, List[str]] = {"replaced": [], "gate_failed": [], "errors": []}
    logger.info(f"{len(namespaces) - len(remaining)} of {len(namespaces)} requested namespace(s) "
               f"already done in a prior run; {len(remaining)} remaining.")
    if not remaining:
        return this_call

    logger.info("Loading dataset...")
    df = pd.read_csv(dataset_csv, low_memory=False)
    if "entry_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "entry_id"})
        df["entry_id"] = df["entry_id"].astype(str)
    if "is_anomaly" in df.columns:
        before = len(df)
        df = df[df["is_anomaly"] == 0]
        logger.info(f"  filtered {before - len(df)} anomalies -> {len(df)} rows")

    id_key = df["entry_id"].astype(str) if "entry_id" in df.columns else df.index.astype(str)
    seq_by_id = dict(zip(id_key, df["sequence"].astype(str)))

    logger.info("Building tasks...")
    tasks = M.build_tasks(df, activities_csv)
    logger.info("Clustering groups (identical to the XGBoost models' own run)...")
    groups = M._family_groups(df, families_json=families_json)

    for i, namespace in enumerate(remaining, 1):
        spec = next((t for t in tasks if t.namespace == namespace), None)
        if spec is None:
            logger.warning(f"[{namespace}] SKIPPED -- task no longer resolves via build_tasks()")
            progress["completed"][namespace] = {"status": "skipped_no_task"}
            _save_progress(progress)
            continue

        logger.info(f"{'='*70}")
        logger.info(f"=== [{i}/{len(remaining)}] {namespace} ===")
        logger.info(f"{'='*70}")
        t0 = time.time()
        try:
            record = M.train_task(
                df, spec.y, ["sequence"], spec.name, "esm2", groups,
                learning_task=spec.task, model_factory=ESM2FineTuneEstimator,
                n_calibration_folds=FINETUNE_N_CALIBRATION_FOLDS,
                n_y_randomizations=FINETUNE_N_Y_RANDOMIZATIONS)
        except Exception as e:
            logger.error(f"  FAILED: {e}", exc_info=True)
            progress["completed"][namespace] = {"status": "error", "error": str(e)}
            _save_progress(progress)
            this_call["errors"].append(namespace)
            gc.collect()
            continue
        elapsed = time.time() - t0

        if record is None:
            logger.warning("  SKIPPED (too little data)")
            progress["completed"][namespace] = {"status": "skipped_too_little_data"}
            _save_progress(progress)
            continue

        model = record.pop("_model")
        cols = record.pop("_columns")
        scaler = record.pop("_scaler")
        ad = record.pop("_applicability_domain")
        train_entry_ids = record.pop("_train_entry_ids")
        test_entry_ids = record.pop("_test_entry_ids")
        record.update({
            "peptide_type": spec.peptide_type, "scope": spec.scope,
            "assay_type": spec.assay_type, "target": spec.target,
            "learning_task": spec.task, "namespace": spec.namespace,
        })
        passed, reasons = M.assess_quality(record)
        record["quality_passed"] = passed
        record["quality_failures"] = reasons

        metric = M._primary_metric(spec.task == "regression")
        logger.info(f"  trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        logger.info(f"  {metric}={record['holdout'].get(metric, float('nan')):.3f}  "
                   f"threshold={record.get('threshold')}  "
                   f"quality_gate={'PASS' if passed else 'FAIL'}")
        if not passed:
            logger.warning(f"  reasons: {reasons}")

        out_dir = os.path.join(model_dir, *namespace.split("/"))
        os.makedirs(out_dir, exist_ok=True)
        pkl_path = os.path.join(out_dir, "esm2.pkl")
        dataset_path = os.path.join(out_dir, "esm2.dataset.json")
        fasta_path = os.path.join(out_dir, "esm2.holdout.fasta")
        backup_pkl = os.path.join(out_dir, "esm2_xgboost_backup.pkl")
        backup_dataset = os.path.join(out_dir, "esm2_xgboost_backup.dataset.json")
        backup_fasta = os.path.join(out_dir, "esm2_xgboost_backup.holdout.fasta")

        if passed:
            if os.path.exists(pkl_path) and not os.path.exists(backup_pkl):
                shutil.copy2(pkl_path, backup_pkl)
            if os.path.exists(dataset_path) and not os.path.exists(backup_dataset):
                shutil.copy2(dataset_path, backup_dataset)
            if os.path.exists(fasta_path) and not os.path.exists(backup_fasta):
                shutil.copy2(fasta_path, backup_fasta)

            with open(pkl_path, "wb") as fh:
                pickle.dump({"model": model, "columns": cols, "namespace": spec.namespace,
                            "feature_block": "esm2", "metrics": record["holdout"],
                            "scaler": scaler, "applicability_domain": ad}, fh)

            model_id = f"{spec.namespace}/esm2"
            dataset_record = dict(record)
            dataset_record.update({
                "model_id": model_id,
                "n_train": len(train_entry_ids),
                "n_test": len(test_entry_ids),
                "train_entry_ids": train_entry_ids,
                "test_entry_ids": test_entry_ids,
                "applicability_domain": {k: v for k, v in ad.items() if k != "index"},
                "architecture": "esm2_finetune",
            })
            with open(dataset_path, "w", encoding="utf-8") as fh:
                json.dump(dataset_record, fh, indent=2)

            write_fasta({eid: seq_by_id[eid] for eid in test_entry_ids if eid in seq_by_id},
                       fasta_path)
            logger.info("  WROTE production model (backed up XGBoost original to "
                       "*_xgboost_backup.*)")
            this_call["replaced"].append(namespace)
        else:
            logger.warning("  XGBoost model LEFT IN PLACE (fine-tune did not clear the "
                          "quality gate)")
            this_call["gate_failed"].append(namespace)

        progress["completed"][namespace] = {
            "status": "replaced" if passed else "gate_failed",
            "elapsed_s": elapsed, "threshold": record.get("threshold"),
            "holdout": record["holdout"], "holdout_ci95": record["holdout_ci95"],
            "quality_passed": passed, "quality_failures": reasons,
        }
        _save_progress(progress)

        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    logger.info(f"{'='*70}")
    logger.info(f"Done with this call. {len(this_call['replaced'])} replaced, "
               f"{len(this_call['gate_failed'])} kept XGBoost, {len(this_call['errors'])} errored.")
    return this_call
