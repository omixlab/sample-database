"""
Which pipeline steps have run, under which run, and how far each got.

Two jobs, both asked for by the operator after a 52-step run was interrupted and restarted from
step 1 by hand:

1. **Resume.** A step that already succeeded under the current run is skipped, so an interrupted
   `make` continues from where it stopped instead of rebuilding everything behind it. `make
   FROM_SCRATCH=1` suppresses that and starts a new run in which nothing is complete.
2. **Progress stats in a table, keyed by run.** Every step records its status, wall time, exit
   code and artifact into `step_progress`, tagged with the `run_id` it belongs to, and `runs`
   carries one row per run so "the last run" is a query rather than a guess.

Design constraints worth knowing before changing this module:

- **Standard library only.** The Makefile calls this between every step, through the bare
  system `python3` rather than `conda run`. Importing anything from `sample/` or from site-
  packages would force a conda environment per call -- measured at 1-2s of interpreter and
  solver startup, which across 52 steps is a minute of pure overhead on every invocation,
  including `make help`. `import sqlite3` is also known to fail intermittently in the `sample`
  env on this host (CXXABI_1.3.15, recorded in DECISIONS.md), and the system interpreter is not
  affected.
- **Its own database.** `data/pipeline_runs.sqlite`, never `data/sample.db`: that file is a
  release artifact rebuilt by step 20 and shipped by step 27, and run bookkeeping has no place
  in it.
- **A resumed invocation adopts the open run's id rather than minting one.** The alternative --
  a new run id per invocation, linked to its predecessor -- spreads one logical run's progress
  across rows that must be stitched back together to answer "did step 30 run yet?". Adoption
  keeps that a single-row lookup, and `runs.resumed_count` records how many invocations it took.
"""
import argparse
import getpass
import json
import os
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: Terminal: the step produced its artifact and exited 0.
STATUS_SUCCESS: str = "SUCCESS"
#: Terminal: the step ran and failed. Re-run on resume.
STATUS_FAILURE: str = "FAILURE"
#: Non-terminal: the step started and no end was recorded. Means killed, crashed or still going.
STATUS_RUNNING: str = "RUNNING"
#: Terminal: a RUNNING row found stale by a later invocation. Re-run on resume.
STATUS_INTERRUPTED: str = "INTERRUPTED"
#: Terminal: resume decided the step was already done, or the step range excluded it.
STATUS_SKIPPED: str = "SKIPPED"

#: Statuses that let resume skip the step. Deliberately only one: a FAILURE or an INTERRUPTED
#: step has no trustworthy artifact, and a SKIPPED one never ran.
RESUMABLE_STATUSES: frozenset = frozenset({STATUS_SUCCESS})

_SCHEMA: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id          TEXT PRIMARY KEY,
        started_at      REAL NOT NULL,
        finished_at     REAL,
        status          TEXT NOT NULL,
        from_scratch    INTEGER NOT NULL DEFAULT 0,
        start_step      INTEGER,
        end_step        INTEGER,
        resumed_count   INTEGER NOT NULL DEFAULT 0,
        git_commit      TEXT,
        host            TEXT,
        user            TEXT,
        notes           TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS step_progress (
        run_id            TEXT NOT NULL,
        step              INTEGER NOT NULL,
        step_name         TEXT,
        status            TEXT NOT NULL,
        started_at        REAL,
        finished_at       REAL,
        duration_seconds  REAL,
        exit_code         INTEGER,
        attempts          INTEGER NOT NULL DEFAULT 1,
        items_total       INTEGER,
        items_done        INTEGER,
        items_skipped     INTEGER,
        items_failed      INTEGER,
        artifact          TEXT,
        artifact_present  INTEGER,
        artifact_count    INTEGER,
        command           TEXT,
        PRIMARY KEY (run_id, step)
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_step_progress_status ON step_progress(run_id, status)",
    "CREATE INDEX IF NOT EXISTS idx_runs_started ON runs(started_at DESC)",
)


def default_db_path() -> str:
    """
    :returns: Location of the run registry. ``SAMPLE_RUN_REGISTRY_DB`` overrides it, the same
        override shape as :func:`sample.annotation_store.default_db_path`.
    """
    override = os.environ.get("SAMPLE_RUN_REGISTRY_DB")
    if override:
        return override
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "data", "pipeline_runs.sqlite")


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or default_db_path()
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    # WAL so a step writing its result never blocks a concurrent `sample-cli runs` read, and
    # NORMAL because losing the last row to a power cut costs one step's bookkeeping, not data.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db(db_path: Optional[str] = None) -> str:
    """
    Creates the schema if absent. Safe on every call, and called by every entry point here.

    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The path initialized.
    """
    path = db_path or default_db_path()
    with _connect(path) as conn:
        for statement in _SCHEMA:
            conn.execute(statement)
    return path


def new_run_id() -> str:
    """
    :returns: A sortable, unique run identifier, ``20261021T140133Z-4f2a1c``.

    Time-prefixed so an `ls`-style listing is chronological, with a random suffix because two
    runs started in the same second on one host is not worth making impossible by other means.
    """
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:6]


def _git_commit() -> Optional[str]:
    """:returns: Short HEAD hash, or ``None`` outside a repository or without git."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def open_run(db_path: Optional[str] = None) -> Optional[sqlite3.Row]:
    """
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The most recently started run still marked ``RUNNING``, or ``None``.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        return conn.execute(
            "SELECT * FROM runs WHERE status = ? ORDER BY started_at DESC LIMIT 1",
            (STATUS_RUNNING,)).fetchone()


def last_run(db_path: Optional[str] = None) -> Optional[sqlite3.Row]:
    """
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: The most recently started run whatever its status, or ``None`` when none exist.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        return conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()


def resolve_run(from_scratch: bool = False, start_step: Optional[int] = None,
                end_step: Optional[int] = None, run_id: Optional[str] = None,
                notes: Optional[str] = None, db_path: Optional[str] = None,
                step: Optional[int] = None) -> Tuple[str, bool]:
    """
    The run this invocation belongs to: the open one when resuming, otherwise a new one.

    Called once per step by the Makefile rather than once per invocation, because each recipe
    line is its own shell and there is nowhere else to keep the id. That makes idempotence the
    whole contract: the first call of a `make` creates the run, every later call in the same
    `make` finds it ``RUNNING`` and returns the same id.

    Stale ``RUNNING`` step rows are closed as ``INTERRUPTED`` here, so a step killed mid-flight
    is recorded as such and re-run rather than being mistaken for complete.

    :param from_scratch: Close any open run and start a new one in which nothing is complete.
    :param start_step: ``START_STEP`` for the record; not enforced here.
    :param end_step: ``END_STEP`` for the record; not enforced here.
    :param run_id: Identifier to use when creating a run. Generated when omitted. Ignored when
        an open run is adopted.
    :param notes: Free text stored on the run row.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :param step: The step about to run. When given, ONLY that step's stale row is closed; a
        ``RUNNING`` row belonging to any other step is left alone.
    :returns: ``(run_id, adopted)``; ``adopted`` is ``True`` when an existing run was resumed.
    """
    init_db(db_path)
    now = time.time()
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT * FROM runs WHERE status = ? ORDER BY started_at DESC LIMIT 1",
            (STATUS_RUNNING,)).fetchone()

        if existing is not None and not from_scratch:
            # Adopt, and close any step left RUNNING as INTERRUPTED.
            #
            # A surviving RUNNING row for THE STEP ABOUT TO RUN is proof that this is a new
            # invocation: within one `make`, a step is closed by `end-step` before the next
            # step's `resolve` runs, and a skipped step never opens a row at all.
            #
            # Scoped to that one step on purpose. An earlier version closed EVERY running row,
            # which misreports a concurrent step: running `make step_11_...` by hand while step 9
            # was genuinely mid-flight (1h44m elapsed, five worker subprocesses alive) recorded
            # step 9 as INTERRUPTED and bumped resumed_count, purely because another step of the
            # same run was being started in a second terminal. A long step and a quick manual one
            # alongside it is ordinary operation, not an interruption.
            #
            # An even earlier version additionally required some step to have reached a terminal
            # status before counting a resume. That was wrong too: a run interrupted during its
            # FIRST step has no closed rows, so a genuine resume recorded `resumed_count = 0`.
            if step is None:
                stale = 0
            else:
                stale = conn.execute(
                    "UPDATE step_progress SET status = ?, finished_at = ?, "
                    "duration_seconds = ? - COALESCE(started_at, ?) "
                    "WHERE run_id = ? AND step = ? AND status = ?",
                    (STATUS_INTERRUPTED, now, now, now, existing["run_id"], int(step),
                     STATUS_RUNNING)).rowcount
            if stale:
                conn.execute("UPDATE runs SET resumed_count = resumed_count + 1 WHERE run_id = ?",
                             (existing["run_id"],))
            return str(existing["run_id"]), True

        if existing is not None and from_scratch:
            conn.execute("UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                         (STATUS_INTERRUPTED, now, existing["run_id"]))
            conn.execute(
                "UPDATE step_progress SET status = ?, finished_at = ? "
                "WHERE run_id = ? AND status = ?",
                (STATUS_INTERRUPTED, now, existing["run_id"], STATUS_RUNNING))

        rid = run_id or new_run_id()
        conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, started_at, status, from_scratch, start_step, "
            "end_step, git_commit, host, user, notes) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (rid, now, STATUS_RUNNING, 1 if from_scratch else 0, start_step, end_step,
             _git_commit(), socket.gethostname(), _user(), notes))
        return rid, False


def _user() -> Optional[str]:
    try:
        return getpass.getuser()
    except Exception:          # pragma: no cover - depends on a broken passwd database
        return None


def completed_steps(run_id: str, db_path: Optional[str] = None) -> Set[int]:
    """
    :param run_id: Run to inspect.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: Step numbers resume may skip, i.e. those recorded ``SUCCESS`` under ``run_id``.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT step FROM step_progress WHERE run_id = ? AND status IN (%s)"
            % ",".join("?" * len(RESUMABLE_STATUSES)),
            (run_id, *sorted(RESUMABLE_STATUSES))).fetchall()
    return {int(r["step"]) for r in rows}


def should_skip(run_id: str, step: int, from_scratch: bool = False,
                db_path: Optional[str] = None) -> bool:
    """
    :param run_id: Run the step would be recorded under.
    :param step: Step number.
    :param from_scratch: When true, never skip.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: Whether the step already succeeded under this run and may be skipped.
    """
    if from_scratch:
        return False
    return int(step) in completed_steps(run_id, db_path)


def step_started(run_id: str, step: int, step_name: str = "", command: str = "",
                 artifact: str = "", db_path: Optional[str] = None) -> None:
    """
    Records that a step is now running, bumping ``attempts`` when it has run before.

    :param run_id: Run this step belongs to.
    :param step: Step number.
    :param step_name: Human label, as the Makefile spells it.
    :param command: Command line, for reproducing the step by hand.
    :param artifact: Path the step must produce.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        prior = conn.execute("SELECT attempts FROM step_progress WHERE run_id = ? AND step = ?",
                             (run_id, int(step))).fetchone()
        attempts = (int(prior["attempts"]) + 1) if prior else 1
        conn.execute(
            "INSERT INTO step_progress (run_id, step, step_name, status, started_at, attempts, "
            "artifact, command) VALUES (?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, step) DO UPDATE SET status = excluded.status, "
            "started_at = excluded.started_at, attempts = excluded.attempts, "
            "step_name = excluded.step_name, artifact = excluded.artifact, "
            "command = excluded.command, finished_at = NULL, duration_seconds = NULL, "
            "exit_code = NULL",
            (run_id, int(step), step_name, STATUS_RUNNING, time.time(), attempts,
             artifact, command))


def step_finished(run_id: str, step: int, status: str, exit_code: Optional[int] = None,
                  items_total: Optional[int] = None, items_done: Optional[int] = None,
                  items_skipped: Optional[int] = None, items_failed: Optional[int] = None,
                  artifact: Optional[str] = None,
                  db_path: Optional[str] = None) -> None:
    """
    Closes a step row and measures its artifact.

    ``artifact_count`` is the number of files the artifact path covers -- the entry count for a
    directory, 1 for a file, 0 when absent. It is the only per-step progress figure available
    without modifying all 52 commands to report their own, and for the file-producing steps it is
    the number that matters.

    :param run_id: Run this step belongs to.
    :param step: Step number.
    :param status: One of the module's ``STATUS_*`` values.
    :param exit_code: Exit status of the step's command.
    :param items_total: Work the step was given, when the command reports it.
    :param items_done: Work completed, when the command reports it.
    :param items_skipped: Work skipped as already done, when the command reports it.
    :param items_failed: Work that failed, when the command reports it.
    :param artifact: Artifact path; the one from :func:`step_started` when omitted.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    """
    init_db(db_path)
    now = time.time()
    with _connect(db_path) as conn:
        row = conn.execute("SELECT started_at, artifact FROM step_progress "
                           "WHERE run_id = ? AND step = ?", (run_id, int(step))).fetchone()
        started = float(row["started_at"]) if row and row["started_at"] else None
        path = artifact if artifact is not None else (row["artifact"] if row else "")
        present, count = _measure_artifact(path)
        conn.execute(
            "INSERT INTO step_progress (run_id, step, status, started_at, finished_at, "
            "duration_seconds, exit_code, items_total, items_done, items_skipped, items_failed, "
            "artifact, artifact_present, artifact_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(run_id, step) DO UPDATE SET status = excluded.status, "
            "finished_at = excluded.finished_at, duration_seconds = excluded.duration_seconds, "
            "exit_code = excluded.exit_code, items_total = excluded.items_total, "
            "items_done = excluded.items_done, items_skipped = excluded.items_skipped, "
            "items_failed = excluded.items_failed, artifact_present = excluded.artifact_present, "
            "artifact_count = excluded.artifact_count",
            (run_id, int(step), status, started, now,
             (now - started) if started else None, exit_code,
             items_total, items_done, items_skipped, items_failed,
             path, 1 if present else 0, count))


def _measure_artifact(path: Optional[str]) -> Tuple[bool, Optional[int]]:
    """
    :param path: Artifact path, possibly empty.
    :returns: ``(present, count)``; ``count`` is the entry count for a directory, 1 for a file.
    """
    if not path:
        return False, None
    if os.path.isdir(path):
        try:
            return True, sum(1 for _ in os.scandir(path))
        except OSError:
            return True, None
    if os.path.exists(path):
        return True, 1
    return False, 0


def finish_run(run_id: str, status: str = STATUS_SUCCESS,
               db_path: Optional[str] = None) -> None:
    """
    Marks a run terminal so the next invocation starts a new one instead of adopting this.

    :param run_id: Run to close.
    :param status: Terminal status to record.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        conn.execute("UPDATE runs SET status = ?, finished_at = ? WHERE run_id = ?",
                     (status, time.time(), run_id))


def step_rows(run_id: str, db_path: Optional[str] = None) -> List[sqlite3.Row]:
    """
    :param run_id: Run to report on.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: Every recorded step for the run, in step order.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        return conn.execute("SELECT * FROM step_progress WHERE run_id = ? ORDER BY step",
                            (run_id,)).fetchall()


def run_rows(limit: int = 20, db_path: Optional[str] = None) -> List[sqlite3.Row]:
    """
    :param limit: Most recent runs to return.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: Run rows, newest first.
    """
    init_db(db_path)
    with _connect(db_path) as conn:
        return conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?",
                            (int(limit),)).fetchall()


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"


def format_progress(run_id: str, db_path: Optional[str] = None) -> str:
    """
    :param run_id: Run to render.
    :param db_path: Database file; :func:`default_db_path` when omitted.
    :returns: A fixed-width table of the run's steps, for a terminal or a log.
    """
    rows = step_rows(run_id, db_path)
    out = [f"run {run_id}", ""]
    out.append(f"  {'step':>4}  {'status':12s} {'elapsed':>8s} {'artifact':>9s}  name")
    out.append(f"  {'-' * 4}  {'-' * 12} {'-' * 8} {'-' * 9}  {'-' * 34}")
    for r in rows:
        count = "-" if r["artifact_count"] is None else f"{r['artifact_count']:,}"
        out.append(f"  {r['step']:>4}  {r['status']:12s} {_fmt_duration(r['duration_seconds']):>8s} "
                   f"{count:>9s}  {(r['step_name'] or '')[:34]}")
    done = sum(1 for r in rows if r["status"] == STATUS_SUCCESS)
    out.append("")
    out.append(f"  {done} of {len(rows)} recorded step(s) succeeded")
    return "\n".join(out)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Command line used by the Makefile between steps. Deliberately terse and newline-free on
    stdout for ``resolve``/``should-skip``, because a recipe captures those in a shell variable.
    """
    parser = argparse.ArgumentParser(prog="python3 -m sample.run_registry",
                                     description=__doc__.strip().splitlines()[0])
    parser.add_argument("--db", default=None, help="Registry path; data/pipeline_runs.sqlite by default")
    sub = parser.add_subparsers(dest="action", required=True)

    p = sub.add_parser("resolve", help="Print the run id this invocation belongs to")
    p.add_argument("--from-scratch", type=int, default=0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--start-step", type=int, default=None)
    p.add_argument("--end-step", type=int, default=None)
    p.add_argument("--notes", default=None)
    p.add_argument("--step", type=int, default=None,
                   help="The step about to run. Only this step's stale row is closed as "
                        "INTERRUPTED; a concurrent step's row is left alone.")

    p = sub.add_parser("should-skip", help="Exit 0 when the step may be skipped, 1 otherwise")
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--from-scratch", type=int, default=0)

    p = sub.add_parser("begin-step", help="Record a step as RUNNING")
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--name", default="")
    p.add_argument("--command", default="")
    p.add_argument("--artifact", default="")

    p = sub.add_parser("end-step", help="Close a step row")
    p.add_argument("--run-id", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--status", required=True)
    p.add_argument("--exit-code", type=int, default=None)
    p.add_argument("--artifact", default=None)
    for name in ("items-total", "items-done", "items-skipped", "items-failed"):
        p.add_argument("--" + name, type=int, default=None)

    p = sub.add_parser("finish", help="Mark a run terminal")
    p.add_argument("--run-id", required=True)
    p.add_argument("--status", default=STATUS_SUCCESS)

    p = sub.add_parser("show", help="Print a run's progress table")
    p.add_argument("--run-id", default=None, help="The last run when omitted")
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("list", help="List recent runs")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)

    if args.action == "resolve":
        rid, adopted = resolve_run(bool(args.from_scratch), args.start_step, args.end_step,
                                   args.run_id, args.notes, args.db, step=args.step)
        sys.stdout.write(rid)
        return 0
    if args.action == "should-skip":
        return 0 if should_skip(args.run_id, args.step, bool(args.from_scratch), args.db) else 1
    if args.action == "begin-step":
        step_started(args.run_id, args.step, args.name, args.command, args.artifact, args.db)
        return 0
    if args.action == "end-step":
        step_finished(args.run_id, args.step, args.status, args.exit_code,
                      args.items_total, args.items_done, args.items_skipped, args.items_failed,
                      args.artifact, args.db)
        return 0
    if args.action == "finish":
        finish_run(args.run_id, args.status, args.db)
        return 0
    if args.action == "show":
        rid = args.run_id
        if not rid:
            row = last_run(args.db)
            if row is None:
                sys.stdout.write("no runs recorded\n")
                return 0
            rid = str(row["run_id"])
        if args.json:
            sys.stdout.write(json.dumps(
                {"run_id": rid, "steps": [dict(r) for r in step_rows(rid, args.db)]},
                indent=2) + "\n")
        else:
            sys.stdout.write(format_progress(rid, args.db) + "\n")
        return 0
    if args.action == "list":
        rows = run_rows(args.limit, args.db)
        if args.json:
            sys.stdout.write(json.dumps([dict(r) for r in rows], indent=2) + "\n")
            return 0
        if not rows:
            sys.stdout.write("no runs recorded\n")
            return 0
        sys.stdout.write(f"  {'run_id':30s} {'status':12s} {'started':20s} {'steps ok':>8s}  scratch\n")
        for r in rows:
            ok = len(completed_steps(str(r["run_id"]), args.db))
            started = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["started_at"]))
            sys.stdout.write(f"  {r['run_id']:30s} {r['status']:12s} {started:20s} {ok:>8d}  "
                             f"{'yes' if r['from_scratch'] else 'no'}\n")
        return 0
    return 2


if __name__ == "__main__":        # pragma: no cover - exercised through the CLI in tests
    raise SystemExit(_main())
