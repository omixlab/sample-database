"""
Persistence for background prediction jobs.

Deliberately a SEPARATE SQLite database from ``data/sample.db``. That file is a release artifact:
step 44 rebuilds it from the final dataset, step 52 ships it to Zenodo, and it is read-only in
normal operation. Prediction jobs are the opposite -- user-generated, written continuously by the
Celery worker, and of no interest to anyone downloading the dataset. Keeping them apart means a
pipeline re-run cannot destroy a user's results, and a busy prediction queue cannot hold write
locks on the file the webserver is serving the dataset from.

The Celery result backend (Redis) holds task state only. It is a transport detail with a TTL, and
it is not where results live: a job's outcome must survive a broker restart and stay queryable by
job id long afterwards, so the rows here are the record of truth. Redis knows whether a task is
running; this database knows what it produced.

Concurrency: the worker writes while the webserver reads, from different processes, so the
database runs in WAL mode. Every function opens its own short-lived connection rather than sharing
one, because a connection cached across Flask threads is the usual source of
``ProgrammingError: SQLite objects created in a thread…``.
"""
import json
import logging
import os
import sqlite3
import time
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger("sample.prediction_store")

#: Job has been accepted and queued, but no worker has picked it up yet.
STATUS_PENDING: str = "PENDING"
#: A worker is running the job now.
STATUS_RUNNING: str = "RUNNING"
#: The job finished and its rows are complete.
STATUS_SUCCESS: str = "SUCCESS"
#: The job stopped on an error; ``jobs.error`` explains why.
STATUS_FAILURE: str = "FAILURE"

#: Terminal states -- the poller stops when a job reaches one of these.
TERMINAL_STATUSES: frozenset = frozenset({STATUS_SUCCESS, STATUS_FAILURE})


def default_db_path() -> str:
    """
    :returns: Location of the prediction database. ``SAMPLE_PREDICTIONS_DB`` overrides it, which is
        how the docker-compose services point at the shared volume.
    """
    override = os.environ.get("SAMPLE_PREDICTIONS_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "predictions.db")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Opens a connection with the settings this database needs.

    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: An open connection with row access by name and WAL journaling enabled.
    """
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # A worker holding a write lock should make the webserver wait briefly, not fail the request.
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
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
            CREATE TABLE IF NOT EXISTS jobs (
                job_id          TEXT PRIMARY KEY,
                model_id        TEXT NOT NULL,
                status          TEXT NOT NULL,
                n_peptides      INTEGER NOT NULL DEFAULT 0,
                n_completed     INTEGER NOT NULL DEFAULT 0,
                submitted_at    REAL NOT NULL,
                started_at      REAL,
                finished_at     REAL,
                error           TEXT,
                celery_task_id  TEXT,
                meta            TEXT
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id          TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                position        INTEGER NOT NULL,
                peptide_name    TEXT,
                sequence        TEXT NOT NULL,
                probability     REAL,
                predicted_label TEXT,
                value           REAL,
                error           TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_job ON predictions(job_id);
            CREATE INDEX IF NOT EXISTS idx_jobs_submitted  ON jobs(submitted_at DESC);
            """
        )
        # Added after the table above shipped: an existing predictions.db from before the
        # applicability domain existed has no such column, and CREATE TABLE IF NOT EXISTS does not
        # retrofit one. NULL for every row already there is the correct value -- those predictions
        # really were never checked -- not an error to migrate away from.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
        if "in_domain" not in columns:
            conn.execute("ALTER TABLE predictions ADD COLUMN in_domain INTEGER")
        job_cols = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if "session_id" not in job_cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN session_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_session ON jobs(session_id)")
    return path


def create_job(job_id: str, model_id: str, n_peptides: int,
               celery_task_id: Optional[str] = None,
               meta: Optional[Dict[str, Any]] = None,
               session_id: Optional[str] = None,
               db_path: Optional[str] = None) -> None:
    """
    Records a queued job.

    Written by the webserver BEFORE the task is dispatched, so a job id handed to the user always
    resolves to a row. Dispatching first would leave a window where the browser polls an id the
    database has never heard of and the page reports a job that does exist as missing.

    :param job_id: The UUID shown to the user.
    :param model_id: Registry id of the model to run.
    :param n_peptides: How many sequences were submitted.
    :param celery_task_id: Broker task id, attached afterwards by :func:`attach_task_id`.
    :param meta: Anything else worth keeping, stored as JSON.
    :param session_id: Optional session identifier of the submitting user.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``None``.
    """
    metadata = dict(meta or {})
    if session_id and "session_id" not in metadata:
        metadata["session_id"] = session_id
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, model_id, status, n_peptides, submitted_at, "
            "celery_task_id, meta, session_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (job_id, model_id, STATUS_PENDING, int(n_peptides), time.time(),
             celery_task_id, json.dumps(metadata), session_id),
        )


def attach_task_id(job_id: str, celery_task_id: str, db_path: Optional[str] = None) -> None:
    """
    Links a job row to its broker task, once dispatch has returned an id.

    :param job_id: Job to update.
    :param celery_task_id: Id returned by ``.delay()``.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``None``.
    """
    with _connect(db_path) as conn:
        conn.execute("UPDATE jobs SET celery_task_id = ? WHERE job_id = ?",
                     (celery_task_id, job_id))


def mark_running(job_id: str, db_path: Optional[str] = None) -> None:
    """
    :param job_id: Job the worker has just picked up.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``None``.
    """
    with _connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = ?, started_at = ? WHERE job_id = ?",
                     (STATUS_RUNNING, time.time(), job_id))


def mark_finished(job_id: str, status: str, error: Optional[str] = None,
                  db_path: Optional[str] = None) -> None:
    """
    :param job_id: Job that has reached a terminal state.
    :param status: :data:`STATUS_SUCCESS` or :data:`STATUS_FAILURE`.
    :param error: Failure message; ``None`` on success.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: ``None``.
    """
    with _connect(db_path) as conn:
        conn.execute("UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE job_id = ?",
                     (status, time.time(), error, job_id))


def save_predictions(job_id: str, rows: Iterable[Dict[str, Any]],
                     db_path: Optional[str] = None) -> int:
    """
    Writes one job's result rows and updates its completed count.

    :param job_id: Owning job.
    :param rows: Dicts with ``position``, ``sequence`` and optionally ``peptide_name``,
        ``probability``, ``predicted_label``, ``value``, ``in_domain``, ``error``.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: How many rows were written.
    """
    def _in_domain_int(value: Any) -> Optional[int]:
        return None if value is None else int(bool(value))

    payload = [
        (job_id, int(r.get("position", i)), r.get("peptide_name"), str(r.get("sequence", "")),
         r.get("probability"), r.get("predicted_label"), r.get("value"),
         _in_domain_int(r.get("in_domain")), r.get("error"))
        for i, r in enumerate(rows)
    ]
    if not payload:
        return 0
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO predictions (job_id, position, peptide_name, sequence, probability, "
            "predicted_label, value, in_domain, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", payload)
        conn.execute("UPDATE jobs SET n_completed = n_completed + ? WHERE job_id = ?",
                     (len(payload), job_id))
    return len(payload)


def get_job(job_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    :param job_id: Job to look up.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The job row as a dict with ``meta`` decoded, or ``None`` if no such job.
    """
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    job = dict(row)
    try:
        job["meta"] = json.loads(job.get("meta") or "{}")
    except (TypeError, ValueError):
        job["meta"] = {}
    return job


def get_predictions(job_id: str, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    :param job_id: Job whose results are wanted.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: Result rows in submission order; empty while the job is still running.
    """
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT position, peptide_name, sequence, probability, predicted_label, value, "
            "in_domain, error FROM predictions WHERE job_id = ? ORDER BY position",
            (job_id,)).fetchall()
    results = [dict(r) for r in rows]
    for result in results:
        # Stored as SQLite's own 0/1/NULL integer; restored to a real tri-state bool/None so a
        # consumer (the API schema, the template) sees the same type predict() produced.
        result["in_domain"] = None if result["in_domain"] is None else bool(result["in_domain"])
    return results


def recent_jobs(limit: int = 25, session_id: Optional[str] = None, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    :param limit: Maximum rows to return.
    :param session_id: If specified, only jobs submitted in this user session.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The most recently submitted jobs, newest first.
    """
    with _connect(db_path) as conn:
        if session_id:
            rows = conn.execute(
                "SELECT job_id, model_id, status, n_peptides, n_completed, submitted_at, finished_at, "
                "error FROM jobs WHERE session_id = ? ORDER BY submitted_at DESC LIMIT ?",
                (str(session_id), int(limit))).fetchall()
        else:
            rows = conn.execute(
                "SELECT job_id, model_id, status, n_peptides, n_completed, submitted_at, finished_at, "
                "error FROM jobs ORDER BY submitted_at DESC LIMIT ?", (int(limit),)).fetchall()
    return [dict(r) for r in rows]
