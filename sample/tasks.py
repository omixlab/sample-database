"""
Celery application and the background prediction task.

Prediction is moved off the request thread because its cost is unbounded from the webserver's
point of view: loading a transformer for an ESM-2 model, or scoring a thousand sequences, takes
far longer than any sensible HTTP timeout. The route therefore accepts the work, returns a job id
immediately, and the browser polls.

Division of responsibility, which is easy to get wrong:

* Redis holds the QUEUE and Celery's own task state. It is a transport with a TTL.
* ``data/predictions.db`` holds the RESULTS. It is the record of truth and outlives the broker.

The task writes to the database at each state change, so a job's progress is visible even if the
Redis result backend has expired or been flushed. Nothing reads job outcomes back out of Celery.
"""
import logging
import os
import time
from typing import Any, Dict, List, Optional, Sequence

from celery import Celery

from sample import annotation_store, prediction_store

logger = logging.getLogger("sample.tasks")

#: Broker URL. The docker-compose stack sets this to the `redis` service; a bare local run gets
#: localhost.
BROKER_URL: str = os.environ.get("SAMPLE_BROKER_URL", "redis://localhost:6379/0")

#: Celery's own result backend, for task state only -- not for prediction results, which go to
#: :mod:`sample.prediction_store`.
RESULT_BACKEND: str = os.environ.get("SAMPLE_RESULT_BACKEND", "redis://localhost:6379/1")

#: Name of the queue the worker consumes.
QUEUE_NAME: str = "sample-predictions"

celery_app = Celery("sample", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_default_queue=QUEUE_NAME,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    # Results in Redis are only used to answer "is this task still alive"; the real record is in
    # SQLite, so they need not be kept long.
    result_expires=3600,
    # One task at a time per worker process. These tasks hold a loaded transformer and are
    # memory-bound, not IO-bound, so prefetching a queue of them into one process would serialize
    # them behind each other while other workers sat idle.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    # A single job cannot occupy a worker forever. The soft limit raises an exception the task can
    # catch and record as a failure; the hard limit kills the process if it does not.
    task_soft_time_limit=1800,
    task_time_limit=2100,
)

# Lets the webserver run without Redis -- tasks then execute inline in the request. Useful for
# development and for the test suite; never set in the docker-compose stack, where the whole point
# is that the work happens elsewhere.
if os.environ.get("SAMPLE_CELERY_EAGER", "").lower() in ("1", "true", "yes"):
    celery_app.conf.update(task_always_eager=True, task_eager_propagates=False)
    logger.warning("SAMPLE_CELERY_EAGER is set: prediction tasks run inline, not on a worker.")


def broker_available() -> bool:
    """
    Checks that the broker can be reached, so the submit route can refuse cleanly.

    Without this, dispatching to a down broker either blocks for the connection timeout or raises
    inside the request handler, and in both cases the user has a job id for a job that will never
    run.

    :returns: ``True`` when the broker answers, or when eager mode makes one unnecessary.
    """
    if celery_app.conf.get("task_always_eager"):
        return True
    try:
        conn = celery_app.connection()
        conn.ensure_connection(max_retries=1, timeout=2)
        conn.release()
        return True
    except Exception as e:
        logger.warning(f"Celery broker unreachable at {BROKER_URL}: {e}")
        return False


@celery_app.task(bind=True, name="sample.predict_peptides")
def predict_peptides(self: Any, job_id: str, model_id: str,
                     peptides: Sequence[Dict[str, str]],
                     db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Scores one to many peptides with one model and stores the results.

    :param self: Celery task instance, bound for access to the retry API and task id.
    :param job_id: The UUID the webserver already recorded and handed to the user.
    :param model_id: Registry id of the model to run.
    :param peptides: ``{"peptide_name", "sequence"}`` dicts, in submission order.
    :param db_path: Prediction database; the module default when omitted.
    :returns: A small summary dict. The full results are in the database, not here -- the return
        value goes to Redis and expires, so nothing should depend on it.
    """
    from sample.inference import predict, validate_sequence

    started = time.time()
    prediction_store.mark_running(job_id, db_path=db_path)
    logger.info(f"[{job_id}] scoring {len(peptides)} peptide(s) with {model_id}")

    try:
        # Invalid sequences are recorded as failed ROWS, not a failed job. One typo among fifty
        # sequences should not discard the other forty-nine.
        rows: List[Dict[str, Any]] = []
        scorable: List[Dict[str, Any]] = []
        for position, item in enumerate(peptides):
            sequence = str(item.get("sequence", ""))
            problem = validate_sequence(sequence)
            if problem:
                rows.append({"position": position, "sequence": sequence,
                             "peptide_name": item.get("peptide_name"), "error": problem})
            else:
                scorable.append({"position": position, "sequence": sequence,
                                 "peptide_name": item.get("peptide_name")})

        if scorable:
            scored = predict([s["sequence"] for s in scorable], model_id)
            for source, result in zip(scorable, scored):
                rows.append({
                    "position": source["position"],
                    "peptide_name": source["peptide_name"],
                    "sequence": source["sequence"],
                    "probability": result.get("probability"),
                    "predicted_label": result.get("predicted_label"),
                    "value": result.get("value"),
                    "in_domain": result.get("in_domain"),
                    "error": None,
                })

        rows.sort(key=lambda r: r["position"])
        prediction_store.save_predictions(job_id, rows, db_path=db_path)
        prediction_store.mark_finished(job_id, prediction_store.STATUS_SUCCESS, db_path=db_path)

        elapsed = time.time() - started
        logger.info(f"[{job_id}] done in {elapsed:.1f}s ({len(rows)} row(s))")
        return {"job_id": job_id, "n_rows": len(rows), "seconds": round(elapsed, 2)}

    except Exception as e:
        # Recorded rather than re-raised: the user is polling the database, and a job that dies
        # without writing its own failure looks identical to one still running.
        logger.exception(f"[{job_id}] prediction job failed")
        prediction_store.mark_finished(job_id, prediction_store.STATUS_FAILURE,
                                       error=f"{type(e).__name__}: {e}", db_path=db_path)
        return {"job_id": job_id, "error": str(e)}


@celery_app.task(bind=True, name="sample.annotate_peptide")
def annotate_peptide(self: Any, job_id: str, sequence: str,
                     db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Runs the annotator battery (`sample.annotator.run_annotation`) over one sequence and stores
    the result.

    Async for the same reason `predict_peptides` is, but a different cost shape: not "unbounded
    batch size" but "a single sequence already takes ~20s" -- a Pfam scan against the full Pfam-A
    database and a HemoPI2 subprocess call, neither fast, see `sample.annotator`'s own docstring
    for the measurements.

    :param self: Celery task instance, bound for access to the retry API and task id.
    :param job_id: The UUID the webserver already recorded and handed to the user.
    :param sequence: The submitted sequence, plain or FASTA.
    :param db_path: Annotation database; the module default when omitted.
    :returns: A small summary dict. The full result is in the database, not here -- see
        :func:`predict_peptides` for why nothing should depend on Celery's own return value.
    """
    from sample.annotator import run_annotation

    started = time.time()
    annotation_store.mark_running(job_id, db_path=db_path)
    logger.info(f"[{job_id}] annotating a {len(sequence)}-character sequence")

    try:
        result = run_annotation(sequence)
        annotation_store.mark_finished(job_id, annotation_store.STATUS_SUCCESS, result=result,
                                       db_path=db_path)
        elapsed = time.time() - started
        logger.info(f"[{job_id}] done in {elapsed:.1f}s")
        return {"job_id": job_id, "seconds": round(elapsed, 2)}
    except Exception as e:
        logger.exception(f"[{job_id}] annotation job failed")
        annotation_store.mark_finished(job_id, annotation_store.STATUS_FAILURE,
                                       error=f"{type(e).__name__}: {e}", db_path=db_path)
        return {"job_id": job_id, "error": str(e)}
