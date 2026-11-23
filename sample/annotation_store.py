"""
Persistence for background annotation jobs (`sample.annotator`).

Mirrors `sample.prediction_store`'s division of responsibility exactly, and for the same reasons:
a separate SQLite database from `data/sample.db` (a release artifact this never touches), Redis
holds only transient task state, this database holds the record of truth, and every function opens
its own short-lived connection because the worker and the webserver write and read from different
processes.

One addition prediction jobs do not need: a **one-week result cache**, keyed by the submitted
sequence. The annotator's own cost (~20s/sequence, see `sample.annotator`'s module docstring) makes
re-running an identical, recently-submitted sequence wasteful, and its inputs (HemoPI2, Pfam-A,
this release's own peptides) do not change often enough for a week-old result to likely be wrong.
"""
import hashlib
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("sample.annotation_store")

STATUS_PENDING: str = "PENDING"
STATUS_RUNNING: str = "RUNNING"
STATUS_SUCCESS: str = "SUCCESS"
STATUS_FAILURE: str = "FAILURE"

TERMINAL_STATUSES: frozenset = frozenset({STATUS_SUCCESS, STATUS_FAILURE})

#: How long a successful result answers a repeat submission of the same sequence without rerunning
#: the battery. See the module docstring for why a week is a reasonable bound.
CACHE_TTL_SECONDS: int = 7 * 24 * 60 * 60


def default_db_path() -> str:
    """
    :returns: Location of the annotation database. ``SAMPLE_ANNOTATIONS_DB`` overrides it, the
        same override shape as :func:`sample.prediction_store.default_db_path`.
    """
    override = os.environ.get("SAMPLE_ANNOTATIONS_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "annotations.db")


def sequence_hash(sequence: str) -> str:
    """:returns: A stable cache key for a cleaned, uppercased sequence."""
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Optional[str] = None) -> str:
    """
    Creates the schema if it does not exist. Safe to call on every process start.

    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The path initialized.
    """
    path = db_path or default_db_path()
    with _connect(path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                job_id          TEXT PRIMARY KEY,
                sequence        TEXT NOT NULL,
                sequence_hash   TEXT NOT NULL,
                status          TEXT NOT NULL,
                submitted_at    REAL NOT NULL,
                started_at      REAL,
                finished_at     REAL,
                error           TEXT,
                celery_task_id  TEXT,
                result          TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_annotations_hash
                ON annotations(sequence_hash, finished_at DESC);
            CREATE INDEX IF NOT EXISTS idx_annotations_submitted
                ON annotations(submitted_at DESC);
            """
        )
    return path


def create_job(job_id: str, sequence: str, celery_task_id: Optional[str] = None,
               db_path: Optional[str] = None) -> None:
    """
    Records a queued job, before the task is dispatched -- see
    :func:`sample.prediction_store.create_job` for why the ordering matters.

    :param job_id: The UUID shown to the user.
    :param sequence: The submitted sequence, uncleaned.
    :param celery_task_id: Broker task id, attached afterwards by :func:`attach_task_id`.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``None``.
    """
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO annotations (job_id, sequence, sequence_hash, status, submitted_at, "
            "celery_task_id) VALUES (?, ?, ?, ?, ?, ?)",
            (job_id, sequence, sequence_hash(sequence), STATUS_PENDING, time.time(),
             celery_task_id),
        )


def attach_task_id(job_id: str, celery_task_id: str, db_path: Optional[str] = None) -> None:
    """:param job_id: Job to update. :param celery_task_id: Id returned by ``.delay()``."""
    with _connect(db_path) as conn:
        conn.execute("UPDATE annotations SET celery_task_id = ? WHERE job_id = ?",
                     (celery_task_id, job_id))


def mark_running(job_id: str, db_path: Optional[str] = None) -> None:
    """:param job_id: Job the worker has just picked up."""
    with _connect(db_path) as conn:
        conn.execute("UPDATE annotations SET status = ?, started_at = ? WHERE job_id = ?",
                     (STATUS_RUNNING, time.time(), job_id))


def mark_finished(job_id: str, status: str, result: Optional[Dict[str, Any]] = None,
                  error: Optional[str] = None, db_path: Optional[str] = None) -> None:
    """
    :param job_id: Job that has reached a terminal state.
    :param status: :data:`STATUS_SUCCESS` or :data:`STATUS_FAILURE`.
    :param result: The annotation battery's output, from :func:`sample.annotator.run_annotation`.
    :param error: Failure message; ``None`` on success.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``None``.
    """
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE annotations SET status = ?, finished_at = ?, result = ?, error = ? "
            "WHERE job_id = ?",
            (status, time.time(), json.dumps(result) if result is not None else None, error,
             job_id))


def get_job(job_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    :param job_id: Job to look up.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The job row as a dict with ``result`` decoded, or ``None`` if no such job.
    """
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM annotations WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    try:
        job["result"] = json.loads(job["result"]) if job.get("result") else None
    except (TypeError, ValueError):
        job["result"] = None
    return job


def find_cached_result(sequence: str, db_path: Optional[str] = None,
                       ttl_seconds: int = CACHE_TTL_SECONDS) -> Optional[Dict[str, Any]]:
    """
    Looks for a successful, still-fresh annotation of this exact sequence.

    :param sequence: The sequence about to be submitted, uncleaned.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :param ttl_seconds: How recent a match must be; :data:`CACHE_TTL_SECONDS` when omitted.
    :returns: The most recent matching job, or ``None`` when nothing qualifies -- no successful
        run of this sequence exists, or the newest one is older than `ttl_seconds`.
    """
    cutoff = time.time() - ttl_seconds
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM annotations WHERE sequence_hash = ? AND status = ? "
            "AND finished_at >= ? ORDER BY finished_at DESC LIMIT 1",
            (sequence_hash(sequence), STATUS_SUCCESS, cutoff)).fetchone()
    if row is None:
        return None
    job = dict(row)
    try:
        job["result"] = json.loads(job["result"]) if job.get("result") else None
    except (TypeError, ValueError):
        job["result"] = None
    return job
