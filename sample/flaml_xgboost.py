"""
Time-budgeted XGBoost hyperparameter search via FLAML, as an alternative to this project's
otherwise-fixed XGBoost configuration (:func:`sample.ml_models._make_model` -- 400 estimators,
depth 6, learning rate 0.08, deliberately not searched, so a reported score reflects one
consistent model family rather than a per-task best case; see that function's own docstring).
This module exists for exactly the comparison that decision forecloses: is a properly-tuned
XGBoost, given a real search budget, actually better on the metric that matters (recall, in
particular, has stayed the weakest metric across both the fixed-hyperparameter XGBoost and the
fine-tuned ESM-2 architecture) -- not a general replacement for the fixed-hyperparameter policy.

:class:`FlamlXGBoostEstimator` is the ``model_factory``
:func:`sample.ml_models.fit_and_validate`/``_y_randomization`` call into -- same construction
signature as :func:`sample.ml_models._make_model` (``factory(is_regression=..., **overrides)``),
same ``.fit``/``.predict``/``.predict_proba`` contract. After the search, only the plain
``xgboost.XGBClassifier``/``XGBRegressor`` FLAML found (``AutoML.model.estimator``) is kept --
the ``flaml.AutoML`` wrapper itself is discarded once fitting is done, so a model trained this way
pickles and unpickles with exactly the same dependency (`xgboost`, already present in both
``sample-ml`` and ``sample-web``) as one built by ``_make_model`` -- `flaml` is never needed to
serve a prediction, only to train one.
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger("sample.flaml_xgboost")

#: FLAML's search wall-clock per fit, in seconds. Applied to exactly ONE real search per task:
#: sample.ml_models.fit_and_validate fits this factory once (never touching X_test/y_test) and
#: reuses THAT found configuration -- via FlamlXGBoostEstimator.y_randomization_factory()'s cheap,
#: no-search FrozenXGBoostEstimator clones -- for every threshold-calibration fold, every
#: cross-validation fold, and every y-randomization repeat below. There used to be a second real
#: search here (a separate calibration-split model, before the cheap-refit hook existed to
#: calibrate off the real model's own configuration instead); it is gone, not merely cheaper.
TIME_BUDGET_SECONDS = 600


def _apply_sample_weight(is_regression: bool, scale_pos_weight: Optional[float],
                         y: Any) -> Optional[np.ndarray]:
    """
    :returns: The same class-imbalance correction :func:`sample.ml_models.fit_and_validate`
        computes (``n_neg/n_pos`` on the actual training split), applied as a sample weight rather
        than left to the estimator's own imbalance handling, so both :class:`FlamlXGBoostEstimator`
        and :class:`FrozenXGBoostEstimator` correct for imbalance identically.
    """
    if is_regression or not scale_pos_weight:
        return None
    y_arr = np.asarray(y)
    return np.where(y_arr == 1, scale_pos_weight, 1.0)


class FrozenXGBoostEstimator:
    """
    Refits an ALREADY-CHOSEN XGBoost configuration -- no FLAML search -- on a fresh set of labels.

    Exists for exactly one caller: :meth:`FlamlXGBoostEstimator.y_randomization_factory`. A
    y-randomization repeat exists to answer "does this collapse to chance," not "what is the best
    architecture for shuffled labels" -- the second question is meaningless, since a well-tuned
    search should MOST expect an untunable, high-variance response to noise. Running a fresh
    600-second FLAML search per repeat (its own default policy) answers a question nobody asked, at
    the same cost as the one search that mattered, twice over per task. Refitting the real model's
    own found configuration on shuffled labels instead keeps the null control valid -- same
    architecture and complexity as the model being validated, the property
    :func:`sample.ml_models._y_randomization`'s own docstring requires -- for the cost of a plain
    ``.fit()``.

    NOTE this removes the SEARCH cost, not the FIT cost: FLAML is free to land on a configuration
    with thousands of estimators, and fitting one of those directly is itself slow (measured at
    160s/repeat for an n_estimators=1995 config on a ~46k-row task) -- multiplied by
    :data:`FLAML_N_Y_RANDOMIZATIONS` repeats, this can still add up to a real wait. Not
    addressed here by capping ``n_estimators`` on this path, because that would silently
    contradict the "same architecture being validated" property above for whatever caller asked
    for the exact found configuration; a future caller that wants a bounded-cost null control
    should build a SEPARATE factory that clones-and-caps rather than changing this one's meaning.
    """

    def __init__(self, estimator: Any, is_regression: bool = False,
                 scale_pos_weight: Optional[float] = None, **_ignored: Any) -> None:
        # `estimator` arrives UNFITTED (a fresh sklearn.base.clone of the real model's own found
        # estimator, made by the factory closure below) -- fit() below owns fitting it, exactly
        # like every other model_factory product in this codebase.
        self._estimator = estimator
        self.is_regression = is_regression
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X: Any, y: Any) -> "FrozenXGBoostEstimator":
        t0 = time.time()
        sample_weight = _apply_sample_weight(self.is_regression, self.scale_pos_weight, y)
        self._estimator.fit(X, y, sample_weight=sample_weight)
        logger.info(f"    frozen refit (no search) done in {time.time() - t0:.1f}s")
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._estimator.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        return self._estimator.predict(X)


class FlamlXGBoostEstimator:
    """
    Scikit-learn-compatible wrapper around a FLAML-searched XGBoost model.

    Constructed exactly like :func:`sample.ml_models._make_model`
    (``factory(is_regression=..., **overrides)``) so it can be passed as
    :func:`sample.ml_models.fit_and_validate`'s ``model_factory`` without any special casing at
    the call site.
    """

    def __init__(self, is_regression: bool = False,
                 scale_pos_weight: Optional[float] = None,
                 time_budget: int = TIME_BUDGET_SECONDS, **_ignored: Any) -> None:
        self.is_regression = is_regression
        self.scale_pos_weight = scale_pos_weight
        self.time_budget = time_budget
        self._estimator: Optional[Any] = None

    def fit(self, X: Any, y: Any) -> "FlamlXGBoostEstimator":
        from flaml import AutoML

        # scale_pos_weight is applied as a sample weight rather than left to FLAML's own search:
        # this reuses the exact class-imbalance correction already validated for the fixed-
        # hyperparameter XGBoost path (sample.ml_models.fit_and_validate computes the identical
        # n_neg/n_pos value the same way), rather than having FLAML rediscover -- or fail to --
        # the same correction from scratch inside its search budget.
        sample_weight = _apply_sample_weight(self.is_regression, self.scale_pos_weight, y)

        # Classification searches for the configuration that minimizes 1-F1, not 1-ROC-AUC.
        # ROC-AUC is threshold-independent -- it rewards a model that RANKS positives above
        # negatives well at SOME threshold, with no preference for one that is usable at a
        # reasonable operating point. That is exactly how a search landed on a real production
        # run's antibacterial config: ROC-AUC 0.783, but its OOF-calibrated threshold sat at 0.055
        # with recall 0.198 -- a model whose positive-class probabilities are so compressed near
        # zero that only the most extreme ones clear any sane cutoff. F1 is computed at a fixed
        # operating point (FLAML's own default 0.5 on the search's internal holdout), so a search
        # minimizing 1-F1 is pushed toward a model that is well-separated at a conventional
        # threshold in the first place, not merely well-ranked in the abstract. Regression has no
        # F1 to target -- r2 stays what it was.
        classification_metric = "f1"
        n_rows, n_cols = np.shape(X)
        logger.info(f"  FLAML search starting: {n_rows} rows x {n_cols} cols, "
                   f"budget={self.time_budget}s, "
                   f"metric={'r2' if self.is_regression else classification_metric}")
        t0 = time.time()
        automl = AutoML()
        automl.fit(
            X_train=X, y_train=np.asarray(y),
            task="regression" if self.is_regression else "classification",
            metric="r2" if self.is_regression else classification_metric,
            estimator_list=["xgboost"],
            time_budget=self.time_budget,
            sample_weight=sample_weight,
            # >=3 turns on FLAML's own per-trial progress (one INFO line per configuration it
            # tries, via its own `flaml.automl.logger`) instead of the total silence `verbose=0`
            # produced for the whole budget -- a 600s search with zero output in between is
            # indistinguishable from a hang in a log file. FLAML installs its own stdout handler
            # (with its own `[flaml.automl.logger: ...]` prefix, since this project's
            # logging.basicConfig only attaches to the root logger, not FLAML's child logger), so
            # this shows up in the same log file with a visually distinct, clearly-labelled format.
            verbose=3,
        )
        elapsed = time.time() - t0
        logger.info(f"  FLAML search done in {elapsed:.1f}s ({elapsed/60:.1f} min). "
                   f"Best config: {automl.best_config}")
        # Keep only the plain fitted xgboost estimator -- discarding flaml.AutoML itself is what
        # keeps this model's pickle dependency-compatible with sample-web, which never installs
        # flaml (see module docstring).
        self._estimator = automl.model.estimator
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("FlamlXGBoostEstimator.predict_proba called before fit().")
        return self._estimator.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("FlamlXGBoostEstimator.predict called before fit().")
        return self._estimator.predict(X)

    def y_randomization_factory(self) -> "Callable[..., FrozenXGBoostEstimator]":
        """
        :returns: A ``model_factory``-shaped callable (``factory(is_regression=..., **params)``)
            that builds :class:`FrozenXGBoostEstimator` instances bound to THIS model's own found
            XGBoost configuration -- read by
            :func:`sample.ml_models.fit_and_validate` in place of re-running a full FLAML search
            for each y-randomization repeat. A fresh :func:`sklearn.base.clone` per call, not the
            same object reused: :func:`sample.ml_models._y_randomization` calls this factory once
            per repeat, and each repeat needs its own unfitted estimator, not one still carrying
            the previous repeat's fitted state.
        :raises RuntimeError: When called before :meth:`fit`.
        """
        if self._estimator is None:
            raise RuntimeError("y_randomization_factory called before fit().")
        found_estimator = self._estimator

        def factory(is_regression: bool = False, scale_pos_weight: Optional[float] = None,
                   **_ignored: Any) -> FrozenXGBoostEstimator:
            from sklearn.base import clone

            clone_estimator = clone(found_estimator)
            # Pinned to single-threaded, overriding whatever n_jobs FLAML's search chose (-1, in
            # every found config seen so far) -- this is the fix for a real, reproduced production
            # failure: xgboost's `hist` tree method building a GHistIndexMatrix's quantile sketch
            # with n_jobs>1 has a genuine data race across MANY rapid, back-to-back fits within one
            # long-running process (exactly this factory's own calling pattern -- calibration
            # folds, cross-validation folds, and y-randomization repeats, dozens of fits per task
            # across dozens of tasks) -- surfaced as `Check failed: count <= total_entries` deep in
            # xgboost's C++ quantile.h, killing 18 of 19 organism-specific tasks in one real
            # training run outright. Each of these fits is on one fold's worth of data, not the
            # full training set, so single-threaded cost here is small; the real model's own one-
            # time fit (FlamlXGBoostEstimator.fit, not this factory) keeps whatever n_jobs the
            # search found, since it only ever fits once per task and never hit this failure mode.
            if hasattr(clone_estimator, "set_params"):
                clone_estimator.set_params(n_jobs=1)
            return FrozenXGBoostEstimator(clone_estimator, is_regression=is_regression,
                                          scale_pos_weight=scale_pos_weight)

        return factory


#: No longer a reduced-rigor concession -- fit_and_validate's cheap-refit calibration path raises
#: whatever is passed here to at least sample.ml_models.N_SPLITS regardless (a calibration fold is
#: a FrozenXGBoostEstimator clone of the real search's found configuration, not a second
#: TIME_BUDGET_SECONDS search), so this constant is kept at N_SPLITS-equivalent rigor rather than
#: the 1 it used to need to be. Left here, rather than inlined, as the one place to change if a
#: future architecture using this module's model_factory contract genuinely needs the old,
#: reduced-rigor single-split fallback back.
FLAML_N_CALIBRATION_FOLDS = 5

#: NOT under the same constraint as the calibration folds above -- FlamlXGBoostEstimator.
#: y_randomization_factory() refits the real model's OWN found configuration directly
#: (FrozenXGBoostEstimator -- a plain XGBoost .fit(), no FLAML search), so a repeat costs no
#: TIME_BUDGET_SECONDS search. It is NOT uniformly a "couple of seconds" the way that sounds,
#: though: FLAML is free to land on a configuration with thousands of estimators (n_estimators
#: =1995 on one real run), and a plain .fit() of THAT is itself slow -- measured at 160s/repeat
#: on a ~46k-row, 586-column task. 20 rather than N_Y_RANDOMIZATIONS' 5 or the previous 100:
#: enough repeats to give _compare_to_null's real-vs-null significance test real power without
#: the multi-hour tail a big found ensemble produces at 100. See FrozenXGBoostEstimator's own
#: docstring for the option of capping estimators on this path specifically if a future found
#: config makes even 20 too slow.
FLAML_N_Y_RANDOMIZATIONS = 20

#: The block name FLAML-trained models are written under -- deliberately not "descriptors" or
#: "composition" (train_finetune_tasks's ESM-2 equivalent had to overwrite "esm2" in place to
#: replace what was actually served; this has no analogous production model to replace, since
#: feature_blocks() only ever produces "descriptors"/"composition", never "flaml" -- so it is a
#: new, additional, independently-selectable model_id sitting beside whatever train-ml already
#: produced for the same task, not a replacement for it).
FLAML_BLOCK_NAME = "flaml"


def train_flaml_tasks(namespaces: List[str],
                      dataset_csv: str = "data/sample_final_release/final_dataset.csv",
                      activities_csv: str = "data/sample_raw_release/activities.csv",
                      families_json: str = "data/families/families.json",
                      model_dir: str = "data/sample_final_release/models",
                      progress_path: Optional[str] = None,
                      resume: bool = False) -> Dict[str, Any]:
    """
    Trains :class:`FlamlXGBoostEstimator` on the union of every block
    :func:`sample.ml_models.feature_blocks` offers -- descriptors, amino-acid composition,
    dipeptide composition and CTD -- for exactly the given task namespaces, reusing each task's
    existing holdout split (same
    anomaly-filtered dataset/family-grouping :func:`sample.ml_models.train_all` used for its own
    models -- reproduces the identical split deterministically, as verified this session).

    Writes to :data:`FLAML_BLOCK_NAME` if and only if the model passes the quality gate
    (:func:`sample.ml_models.assess_quality`) -- a NEW, additional model_id, not a replacement for
    whatever ``train-ml`` already produced for the same task (see :data:`FLAML_BLOCK_NAME`'s own
    comment for why this needs no backup-then-replace step, unlike
    :func:`sample.esm_finetune.train_finetune_tasks`).

    RETRAINS (overwrites) every requested namespace by default, even one a prior call already
    finished -- a re-run is the common case (a code change, a fresh dataset, a different
    ``TIME_BUDGET_SECONDS``), and silently serving a stale model because a progress file happened
    to still be on disk would be the wrong default. Pass ``resume=True`` for the opposite,
    interrupted-run behavior: skip any namespace already recorded in `progress_path`, so a run
    killed partway through a long namespace list can continue instead of starting over. Progress is
    still written incrementally either way, so a killed ``resume=False`` run can be continued with
    ``resume=True`` afterward.

    :param namespaces: Task namespaces to train, e.g.
        ``["antibacterial/general/any/classification"]`` -- must each resolve via
        :func:`sample.ml_models.build_tasks` against `dataset_csv`/`activities_csv`.
    :param dataset_csv: The assembled dataset (with ``is_anomaly`` filtered out here, matching
        :func:`sample.ml_models.train_all`'s own ``filter_anomalies=True`` convention).
    :param activities_csv: Step 19's activity table.
    :param families_json: Curated family assignment for the train/holdout grouping.
    :param model_dir: Namespace root to write models under.
    :param progress_path: Where to persist per-namespace progress;
        ``<model_dir>/../flaml_rollout_progress.json`` when omitted.
    :param resume: ``False`` (the default) retrains every requested namespace regardless of what
        `progress_path` already records. ``True`` skips a namespace already marked completed
        there -- the old, interrupted-run-friendly behavior.
    :returns: ``{"written": [...], "gate_failed": [...], "errors": [...]}`` -- the namespaces
        this call itself processed, not the full accumulated progress file.
    """
    import gc
    import json
    import os
    import pickle
    import time

    import pandas as pd

    from sample import ml_models as M
    from sample.families import write_fasta

    if progress_path is None:
        progress_path = os.path.join(os.path.dirname(model_dir.rstrip("/")),
                                     "flaml_rollout_progress.json")

    def _load_progress() -> Any:
        if os.path.exists(progress_path):
            with open(progress_path, encoding="utf-8") as fh:
                return json.load(fh)
        return {"completed": {}}

    def _save_progress(progress: Any) -> None:
        tmp = progress_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(progress, fh, indent=2)
        os.replace(tmp, progress_path)

    progress = _load_progress()
    if resume:
        done = set(progress["completed"].keys())
        remaining = [ns for ns in namespaces if ns not in done]
        logger.info(f"{len(namespaces) - len(remaining)} of {len(namespaces)} requested "
                   f"namespace(s) already done in a prior run; {len(remaining)} remaining "
                   f"(resume=True).")
    else:
        # Retrain everything requested, overwriting whatever progress/model already exists for
        # it -- the default, per this function's own docstring.
        remaining = list(namespaces)
        already_recorded = sum(1 for ns in namespaces if ns in progress["completed"])
        if already_recorded:
            logger.info(f"{already_recorded} of {len(namespaces)} requested namespace(s) were "
                       f"already trained in a prior run; retraining anyway (resume=False, the "
                       f"default). Pass resume=True to skip them instead.")
    this_call: Any = {"written": [], "gate_failed": [], "errors": []}
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

    df = M.add_derived_feature_columns(df)
    # Every block feature_blocks() offers, concatenated -- FLAML is searching hyperparameters over
    # one fixed feature set rather than running train_all's per-block matrix, so the blocks are
    # unioned here and the search is left to weight them. Built from feature_blocks() rather than
    # by globbing columns directly so the `pep_*`/`stab_*` decoy-leakage exclusion documented
    # there applies on this path too.
    blocks = M.feature_blocks(df)
    columns = [c for block in ("descriptors", "composition", "dpc", "ctd")
              for c in blocks.get(block, [])]
    logger.info(f"  feature set: {len(columns)} columns ("
               + ", ".join(f"{b}={len(blocks.get(b, []))}"
                           for b in ("descriptors", "composition", "dpc", "ctd")) + ")")

    id_key = df["entry_id"].astype(str) if "entry_id" in df.columns else df.index.astype(str)
    seq_by_id = dict(zip(id_key, df["sequence"].astype(str)))

    logger.info("Building tasks...")
    tasks = M.build_tasks(df, activities_csv)
    logger.info("Clustering groups (identical to train-ml's own run)...")
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
                df, spec.y, columns, spec.name, FLAML_BLOCK_NAME, groups,
                learning_task=spec.task, model_factory=FlamlXGBoostEstimator,
                calibrate_threshold=False,
                n_calibration_folds=FLAML_N_CALIBRATION_FOLDS,
                n_y_randomizations=FLAML_N_Y_RANDOMIZATIONS)
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

        if passed:
            out_dir = os.path.join(model_dir, *namespace.split("/"))
            os.makedirs(out_dir, exist_ok=True)
            pkl_path = os.path.join(out_dir, f"{FLAML_BLOCK_NAME}.pkl")
            with open(pkl_path, "wb") as fh:
                pickle.dump({"model": model, "columns": cols, "namespace": spec.namespace,
                            "feature_block": FLAML_BLOCK_NAME, "metrics": record["holdout"],
                            "scaler": scaler, "applicability_domain": ad}, fh)

            dataset_path = os.path.join(out_dir, f"{FLAML_BLOCK_NAME}.dataset.json")
            dataset_record = dict(record)
            dataset_record.update({
                "model_id": f"{spec.namespace}/{FLAML_BLOCK_NAME}",
                "n_train": len(train_entry_ids), "n_test": len(test_entry_ids),
                "train_entry_ids": train_entry_ids, "test_entry_ids": test_entry_ids,
                "applicability_domain": {k: v for k, v in ad.items() if k != "index"},
            })
            with open(dataset_path, "w", encoding="utf-8") as fh:
                json.dump(dataset_record, fh, indent=2)

            write_fasta({eid: seq_by_id[eid] for eid in test_entry_ids if eid in seq_by_id},
                       os.path.join(out_dir, f"{FLAML_BLOCK_NAME}.holdout.fasta"))
            write_fasta({eid: seq_by_id[eid] for eid in train_entry_ids if eid in seq_by_id},
                       os.path.join(out_dir, f"{FLAML_BLOCK_NAME}.train.fasta"))
            logger.info(f"  WROTE {pkl_path}")
            this_call["written"].append(namespace)
        else:
            logger.warning(f"  NOT written -- reasons: {reasons}")
            this_call["gate_failed"].append(namespace)

        progress["completed"][namespace] = {
            "status": "written" if passed else "gate_failed",
            "elapsed_s": elapsed, "threshold": record.get("threshold"),
            "holdout": record["holdout"], "holdout_ci95": record["holdout_ci95"],
            "quality_passed": passed, "quality_failures": reasons,
        }
        _save_progress(progress)
        gc.collect()

    logger.info(f"{'='*70}")
    logger.info(f"Done with this call. {len(this_call['written'])} written, "
               f"{len(this_call['gate_failed'])} did not clear the gate, "
               f"{len(this_call['errors'])} errored.")
    return this_call
