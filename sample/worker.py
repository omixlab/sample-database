"""
ML Model Worker for sAMPle.

Executes queued machine-learning bioactivity prediction jobs.
Supports running under Celery/Redis, with fallback execution for SQLite-backed jobs.
"""
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sample.worker")


def run_celery_worker():
    """Runs the Celery background worker consuming from the prediction queue."""
    from sample.tasks import celery_app, QUEUE_NAME

    logger.info(f"Starting Celery prediction worker on queue '{QUEUE_NAME}'...")
    argv = [
        "worker",
        "--loglevel=INFO",
        f"--queues={QUEUE_NAME}",
        "--concurrency=2",
        "--max-tasks-per-child=25",
    ]
    celery_app.worker_main(argv)


def run_fallback_poller(poll_interval: float = 2.0):
    """
    Fallback worker that polls data/predictions.db for jobs stuck in PENDING status
    and executes them directly if Celery/Redis is inactive or bypassed.
    """
    import json
    from sample import prediction_store
    from sample.inference import predict, validate_sequence

    logger.info("Starting fallback SQLite job poller...")
    db_path = prediction_store.default_db_path()
    prediction_store.init_db(db_path)

    while True:
        try:
            with prediction_store._connect(db_path) as conn:
                cursor = conn.execute(
                    "SELECT job_id, model_id, meta FROM jobs WHERE status = ? ORDER BY submitted_at ASC LIMIT 1",
                    (prediction_store.STATUS_PENDING,)
                )
                row = cursor.fetchone()

            if not row:
                time.sleep(poll_interval)
                continue

            job_id, model_id, meta_raw = row
            logger.info(f"[Fallback Worker] Picking up pending job {job_id} for model {model_id}")

            meta = json.loads(meta_raw or "{}")
            peptides = meta.get("peptides", [])

            if not peptides:
                # If peptides were not saved in meta, wait briefly for Celery or mark error
                time.sleep(poll_interval)
                continue

            prediction_store.mark_running(job_id, db_path=db_path)

            rows = []
            scorable = []
            for position, item in enumerate(peptides):
                seq = str(item.get("sequence", ""))
                err = validate_sequence(seq)
                if err:
                    rows.append({"position": position, "sequence": seq,
                                 "peptide_name": item.get("peptide_name"), "error": err})
                else:
                    scorable.append({"position": position, "sequence": seq,
                                     "peptide_name": item.get("peptide_name")})

            if scorable:
                scored = predict([s["sequence"] for s in scorable], model_id)
                for src, res in zip(scorable, scored):
                    rows.append({
                        "position": src["position"],
                        "peptide_name": src["peptide_name"],
                        "sequence": src["sequence"],
                        "probability": res.get("probability"),
                        "predicted_label": res.get("predicted_label"),
                        "value": res.get("value"),
                        "in_domain": res.get("in_domain"),
                        "error": None,
                    })

            rows.sort(key=lambda r: r["position"])
            prediction_store.save_predictions(job_id, rows, db_path=db_path)
            prediction_store.mark_finished(job_id, prediction_store.STATUS_SUCCESS, db_path=db_path)
            logger.info(f"[Fallback Worker] Completed job {job_id} ({len(rows)} rows)")

        except Exception as e:
            logger.exception(f"Error in fallback poller: {e}")
            time.sleep(poll_interval)


def main():
    """Main worker entry point."""
    logger.info("sAMPle ML Worker initializing...")
    from sample.tasks import broker_available

    if broker_available():
        logger.info("Redis broker is online. Launching Celery worker process.")
        try:
            run_celery_worker()
        except KeyboardInterrupt:
            logger.info("Worker stopped by user.")
    else:
        logger.warning("Redis broker unreachable. Falling back to SQLite polling worker.")
        try:
            run_fallback_poller()
        except KeyboardInterrupt:
            logger.info("Poller stopped by user.")


if __name__ == "__main__":
    main()
