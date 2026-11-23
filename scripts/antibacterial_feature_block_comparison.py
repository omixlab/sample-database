"""
FLAML-optimized comparison of amino-acid composition, physicochemical descriptors, composition +
descriptors, and cached ESM-2 embeddings -- antibacterial/general/any/classification only, by
instruction (the one case study while the FLAML-searched training path is proved out).

Uses the exact same machinery as every other training step in this pipeline -- sample.ml_models.
train_task() (same clustering via _family_groups, same holdout split, same quality gate,
same threshold calibration) and sample.flaml_xgboost.FlamlXGBoostEstimator (same 600s budget,
same reduced-rigor calibration/y-randomization constants) -- so this comparison's numbers are
directly comparable to steps 43-46's own, not a separately-tuned side channel.

Run manually (not a numbered pipeline step): a standalone script, in the same spirit as
decoy_strategy_comparison.py earlier this session.
"""
import json
import logging
import time

import pandas as pd

from sample import ml_models as M
from sample.flaml_xgboost import (FLAML_N_CALIBRATION_FOLDS, FLAML_N_Y_RANDOMIZATIONS,
                                  FlamlXGBoostEstimator)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("antibacterial_feature_block_comparison")

DATASET_CSV = "data/sample_final_release/final_dataset.csv"
ACTIVITIES_CSV = "data/sample_raw_release/activities.csv"
FAMILIES_JSON = "data/families/families.json"
TASK_NAMESPACE = "antibacterial/general/any/classification"
OUT_PATH = "data/sample_final_release/antibacterial_feature_block_comparison.json"


def main() -> None:
    logger.info("Loading dataset...")
    df = pd.read_csv(DATASET_CSV, low_memory=False)
    if "entry_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "entry_id"})
        df["entry_id"] = df["entry_id"].astype(str)
    if "is_anomaly" in df.columns:
        before = len(df)
        df = df[df["is_anomaly"] == 0]
        logger.info(f"  filtered {before - len(df)} anomalies -> {len(df)} rows")

    # Same feature computation every other step goes through -- feature_blocks() gives the
    # descriptors/composition names (and the pep_*/stab_* decoy-leakage exclusion) rather than
    # this script globbing columns a second, independently-maintained way.
    blocks_available = M.feature_blocks(df)
    descriptors = blocks_available["descriptors"]
    composition = blocks_available["composition"]
    esm2 = [c for c in df.columns if c.startswith("esm2_dim_")]
    if not esm2:
        raise RuntimeError(f"No esm2_dim_* columns in {DATASET_CSV}; step 24's embedding "
                           f"enrichment must have run for this comparison to include ESM-2.")

    blocks = {
        "composition": composition,
        "descriptors": descriptors,
        "composition_descriptors": composition + descriptors,
        "esm2": esm2,
    }
    for name, columns in blocks.items():
        logger.info(f"  block '{name}': {len(columns)} columns")

    logger.info("Building tasks...")
    tasks = M.build_tasks(df, ACTIVITIES_CSV)
    spec = next((t for t in tasks if t.namespace == TASK_NAMESPACE), None)
    if spec is None:
        raise RuntimeError(f"{TASK_NAMESPACE} did not resolve via build_tasks() against this "
                           f"dataset/activity table.")

    # Same clustering every other step uses -- _family_groups (cd-hit at GROUPING_IDENTITY,
    # unioned with the curated Pfam family assignment), computed once and reused across every
    # block the same way train_all() reuses it across every (task, block) cell.
    logger.info("Clustering groups (identical to every other training step)...")
    groups = M._family_groups(df, families_json=FAMILIES_JSON)

    results = {}
    for block_name, columns in blocks.items():
        logger.info(f"{'=' * 70}")
        logger.info(f"=== {TASK_NAMESPACE} / {block_name} ===")
        logger.info(f"{'=' * 70}")
        t0 = time.time()
        record = M.train_task(
            df, spec.y, columns, spec.name, block_name, groups, learning_task=spec.task,
            model_factory=FlamlXGBoostEstimator,
            n_calibration_folds=FLAML_N_CALIBRATION_FOLDS,
            n_y_randomizations=FLAML_N_Y_RANDOMIZATIONS)
        elapsed = time.time() - t0
        if record is None:
            logger.warning(f"  SKIPPED (too little data for {block_name})")
            continue

        record.pop("_model", None)
        record.pop("_columns", None)
        record.pop("_scaler", None)
        record.pop("_applicability_domain", None)
        record.pop("_train_entry_ids", None)
        record.pop("_test_entry_ids", None)
        passed, reasons = M.assess_quality(record)

        h = record["holdout"]
        ci = record["holdout_ci95"]
        logger.info(f"  trained in {elapsed:.1f}s ({elapsed / 60:.1f} min), "
                   f"threshold={record.get('threshold')}")
        logger.info(f"  AUC={h['roc_auc']:.3f} [{ci['roc_auc'][0]:.3f},{ci['roc_auc'][1]:.3f}]  "
                   f"prec={h['precision']:.3f}  rec={h['recall']:.3f} "
                   f"[{ci['recall'][0]:.3f},{ci['recall'][1]:.3f}]  "
                   f"f1={h['f1']:.3f}  mcc={h['mcc']:.3f}  "
                   f"quality_gate={'PASS' if passed else 'FAIL'}")
        results[block_name] = {"elapsed_s": elapsed, "quality_passed": passed,
                               "quality_failures": reasons, **record}

        with open(OUT_PATH, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info(f"  written to {OUT_PATH} (updated after every block)")

    logger.info(f"{'=' * 70}")
    logger.info("Summary:")
    for block_name, record in results.items():
        h = record["holdout"]
        logger.info(f"  {block_name:<24} AUC={h['roc_auc']:.3f}  recall={h['recall']:.3f}  "
                   f"f1={h['f1']:.3f}  gate={'PASS' if record['quality_passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
