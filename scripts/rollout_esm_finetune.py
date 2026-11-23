"""
Full-matrix rollout of sample.esm_finetune.ESM2FineTuneEstimator across every currently-kept
XGBoost+ESM2 task (per data/sample_final_release/ml_report.json). Thin CLI wrapper around
sample.esm_finetune.train_finetune_tasks() -- see that function's docstring for the actual
behavior (holdout reuse, reduced validation rigor, the backup-then-replace production rollout,
resumability).

For running just the antibacterial or anticancer task (or "everything else"), use
`sample-cli train-ml-antibacterial` / `train-ml-anticancer` / `train-ml-other` instead -- those
share this exact same underlying function and progress file, so any combination of this script
and those three commands, run in any order, never re-trains a namespace twice.

A full timestamped run log is written to logs/rollout_esm_finetune.log via a FileHandler,
appended across runs -- so resuming after an interruption keeps the earlier run's log rather than
starting a fresh, disconnected file.

Interruptible and resumable by construction: re-running this script skips any namespace already
present in the progress file, so killing it (Ctrl-C, or a `kill`) and starting it again later
picks up where it left off rather than re-training completed tasks.
"""
import json
import logging
import os

LOG_PATH = "logs/rollout_esm_finetune.log"
ML_REPORT = "data/sample_final_release/ml_report.json"


def _configure_logging() -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(LOG_PATH)  # default mode "a": appends across resumed runs
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(console)
    root.addHandler(file_handler)


def main() -> None:
    _configure_logging()
    logger = logging.getLogger("rollout_esm_finetune")
    logger.info(f"Log file: {os.path.abspath(LOG_PATH)}")

    from sample.esm_finetune import train_finetune_tasks

    with open(ML_REPORT, encoding="utf-8") as fh:
        ml_report = json.load(fh)
    namespaces = [m["namespace"] for m in ml_report["models"]]
    logger.info(f"Rolling out fine-tuning to all {len(namespaces)} currently-kept tasks.")

    train_finetune_tasks(namespaces)


if __name__ == "__main__":
    main()
