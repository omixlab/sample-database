import os
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait

import pandas as pd

from sample.llm import extract_text_from_pdf

logger = logging.getLogger(__name__)

# How many tasks to keep in flight per worker. Keeping this bounded (rather than submitting
# every task up front) means memory stays proportional to worker count, not corpus size --
# only in-flight documents' text is held in memory at once.
_SUBMISSION_WINDOW_PER_WORKER = 2

_HEARTBEAT_SECONDS = 20.0

# CHUNKING. Documents are processed in fixed-size groups, and between groups the Ollama runtime
# is returned to its start-of-run state (see sample.ollama_runtime for what "reset" means and
# why it is an unload rather than a service restart).
#
# This is a response to a measured decay, not a precaution. Throughput on the current corpus
# held at roughly 1 doc/s for the first couple of thousand documents and fell to ~0.04 docs/s
# after about 2,000 -- a 25x collapse that raised no error, logged no failure, and recovered
# completely when the server was restarted by hand. Everything on the client side is bounded
# and observably drains; the llama-server runner is the only component that spans the whole run
# while holding per-slot KV cache and an internal queue this project cannot inspect. Chunking
# replaces an unobservable cause with a bounded one: the state can accumulate for at most
# CHUNK_SIZE documents before it is thrown away.
#
# 100 is small enough that a decay cannot get far and large enough that the reset's cost (one
# model load, a few seconds for a 4B model) stays under a percent of the chunk's runtime.
# 0 disables chunking and restores the previous single-pass behaviour.
EXTRACT_CHUNK_SIZE = int(os.getenv("SAMPLE_EXTRACT_CHUNK_SIZE", "100"))

# PER-DOCUMENT WALL-CLOCK BUDGET, in seconds. 0 disables it.
#
# Not a timeout in the usual sense, and the difference matters when reading the output. It
# cannot abort a document -- no thread can be interrupted in CPython, and abandoning the future
# would leave the worker running, holding its Ollama slot, which is the backlog a timeout is
# meant to prevent. What it does is stop a document STARTING new expensive work once the budget
# is gone: each LLM call is clamped to the remainder and refused below
# sample.deadlines.MIN_USEFUL_SLICE, and the langextract subprocess (the one thing here that
# genuinely can be killed) is hard-capped.
#
# A document that hits the cap therefore completes, with fewer records than it would have had,
# and says so: metadata["budget_exceeded"] is set, alongside the seconds it actually took. That
# is deliberately a flag on a written output rather than a missing file -- a missing file is
# indistinguishable from "not reached yet" and would be silently re-attempted forever by
# --resume, while a flagged file can be counted, audited, and selectively re-run.
#
# ONE OPERATION IS DELIBERATELY NOT CLAMPED: the docling conversion, which has its own 900s
# subprocess timeout. On a scanned patent OCR runs at ~12s/page, so a long one can spend the
# whole budget before any model is called. Cutting docling to the remainder instead would kill
# the conversion and leave the document with NO text at all -- a document with text and tables
# but no LLM pass is strictly more useful than an empty one. The consequence to know about:
# in step 8, where the table pass is most of the work, a large scanned document is expected to
# be flagged budget_exceeded, and DOC_TIMEOUT=0 is the right setting for a corpus of those.
DOCUMENT_BUDGET_SECONDS = float(os.getenv("SAMPLE_DOCUMENT_BUDGET_SECONDS", "300"))

# How long a chunk barrier waits for an abandoned LLM call to return before resetting anyway.
# Bounded because such a call cannot be cancelled -- waiting indefinitely would hand one wedged
# request the power to stall the rest of the corpus, which is the opposite of the point.
_BARRIER_GRACE_SECONDS = float(os.getenv("SAMPLE_BARRIER_GRACE_SECONDS", "30"))

# One log line per extracted peptide-activity, so what the run is actually finding is visible
# while it runs rather than only by reading the JSON afterwards. Set to 0 to silence.
#
# A peptide with NO activity still gets a line, with the activity fields blank. That is not
# padding: measured over this corpus, 65.2% of LLM-derived records carry no activity at all and
# only ~24.3% carry a sequence, so "what did it find, and is any of it usable?" is answered by
# the lines that are empty as much as by the ones that are full.
LOG_ACTIVITIES = os.getenv("SAMPLE_LOG_ACTIVITIES", "1").strip().lower() not in (
    "0", "false", "off", "no", "")

#: Longest free-text field written to a log line. Names and target organisms come out of a model
#: and are occasionally a whole sentence; one of those would wrap the line and bury the rest.
_LOG_FIELD_CHARS = 60

#: Resume modes for :func:`run_parallel_pipeline`.
#:
#: ``off``      -- process every document.
#: ``exists``   -- skip a document whose output JSON exists. Right for the SCREENING pass (step
#:                 8), which is the first thing to write that file.
#: ``refined``  -- skip a document whose output JSON records that the REFINED pass already ran.
#:
#: ``refined`` exists because ``exists`` is wrong for step 9 and silently so: step 8 writes an
#: output for every document, so resuming step 9 on file existence skips the entire corpus and
#: adds no model records at all. The distinction is recorded in the output itself --
#: ``metadata.langextract_backend`` is ``"enabled"`` only when the refined pass ran, and
#: ``"disabled"`` after a screening-only pass. Measured over 4,410 outputs on disk: 2,322
#: ``enabled``, 1,983 ``disabled``, 105 absent (documents skipped before the backend was
#: consulted), with no overlap -- so the marker separates the two passes exactly.
RESUME_OFF: str = "off"
RESUME_IF_EXISTS: str = "exists"
RESUME_IF_REFINED: str = "refined"
RESUME_MODES: Tuple[str, ...] = (RESUME_OFF, RESUME_IF_EXISTS, RESUME_IF_REFINED)

#: `metadata.langextract_backend` value written when the refined pass ran for a document.
_REFINED_MARKER: str = "enabled"


def _refined_pass_done(output_path: str, text_path: Optional[str] = None) -> bool:
    """
    :param output_path: Extraction JSON for one document.
    :param text_path: The document's input text. When given and NEWER than the output, the
        document counts as not done regardless of what the output says -- the text was
        re-acquired after that output was written, so the output describes an input that no
        longer exists.
    :returns: Whether the refined pass has already dealt with this document.

    Two ways a document counts as dealt with:

    1. `metadata.langextract_backend == "enabled"` -- the model pass ran.
    2. `metadata.skipped` -- gate 1 rejected the document as not a paper at all.

    The second case is the one that cost a day of confusion. Gate 1 is
    :func:`sample.document_quality.is_unusable_document`, a pure function of the text, and
    `skipped` is written at exactly one site, so its verdict is deterministic: the refined pass
    reading the same text reaches the identical conclusion and cannot do better. Without this
    clause those documents never acquire a marker, so every resume re-queued all 509 of them --
    and because they sit early in the iteration order (139 of the first 400 queued), a restart
    spent its first several minutes re-skipping junk and reporting "0 peptides extracted so
    far". An operator watching that restarts again, and the run never reaches real work.

    A file that cannot be read or parsed counts as NOT done, so a truncated output left by a kill
    is re-extracted rather than skipped. That is the safe direction: re-running a document costs
    one document of work, while skipping a half-written one loses it from the corpus with nothing
    to indicate it.
    """
    if text_path:
        try:
            if os.path.getmtime(output_path) < os.path.getmtime(text_path):
                return False
        except OSError:
            return False
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return False
    if metadata.get("langextract_backend") == _REFINED_MARKER:
        return True
    return bool(metadata.get("skipped"))


def build_task_list(input_dir: str, output_dir: str, df: "pd.DataFrame",
                    resume_mode: str) -> Tuple[List[Tuple[str, str, str]], int, int]:
    """
    The documents an extraction run will actually process, and what it is leaving out.

    Separate from :func:`run_parallel_pipeline` so the resume decision can be tested without a
    thread pool, a model server or a corpus: which documents a mode selects is the whole of what
    ``--resume-mode`` means, and it is the part that was silently wrong before.

    :param input_dir: Directory of ``<internal_id>.txt`` source documents.
    :param output_dir: Directory of ``<internal_id>.json`` extraction outputs.
    :param df: Frame with an ``internal_id`` column naming the corpus.
    :param resume_mode: A member of :data:`RESUME_MODES`.
    :returns: ``(tasks, skipped_existing, skipped_missing)`` where each task is
        ``(internal_id, text_path, output_path)``.
    """
    tasks: List[Tuple[str, str, str]] = []
    skipped_existing = 0
    skipped_missing = 0
    for row in df.itertuples():
        internal_id = str(row.internal_id)
        text_path = os.path.join(input_dir, f"{internal_id}.txt")
        output_path = os.path.join(output_dir, f"{internal_id}.json")
        if resume_mode != RESUME_OFF and os.path.exists(output_path):
            if resume_mode == RESUME_IF_EXISTS or _refined_pass_done(output_path, text_path):
                skipped_existing += 1
                continue
        if not os.path.exists(text_path):
            skipped_missing += 1
            continue
        tasks.append((internal_id, text_path, output_path))
    return tasks, skipped_existing, skipped_missing


def _langextract_max_workers() -> int:
    """
    :returns: Threads the refined pass fans one document across, as the worker will actually use.

    Read from :mod:`sample.langextract_records` when that module is importable so the number
    cannot drift from the one in force, and from the environment otherwise. The screening pass
    does not call the model at all, but it also never reaches the warning this feeds.
    """
    try:
        from sample.langextract_records import MAX_WORKERS
        return int(MAX_WORKERS)
    except Exception:
        raw = os.environ.get("LANGEXTRACT_MAX_WORKERS", "6")
        return int(raw) if raw.isdigit() else 6


def _fit_langextract_workers(slots: int, workers: int, source: str = "") -> None:
    """
    Lowers the refined pass's per-document fan-out so it does not oversubscribe the model server.

    :param slots: Parallel slots the Ollama server is actually running (``llama-server -np``).
    :param workers: Document-level worker threads this run will use.
    :param source: Corpus name, for the log line.
    :returns: ``None``; the new value lives in :data:`sample.langextract_records.MAX_WORKERS`.

    A no-op when LANGEXTRACT_MAX_WORKERS was set explicitly -- an operator who names a number
    gets it -- and when the current value already fits.
    """
    try:
        from sample import langextract_records as _lx
    except Exception:
        return
    if _lx.MAX_WORKERS_PINNED:
        return
    current = max(1, int(_lx.MAX_WORKERS))
    if workers * current <= max(1, slots):
        return
    target = max(1, int(slots) // max(1, int(workers)))
    if target >= current:
        return
    applied = _lx.set_max_workers(target)
    logger.info(
        f"[{source}] LANGEXTRACT_MAX_WORKERS lowered {current} -> {applied} to fit the model "
        f"server: --workers {workers} x {applied} = {workers * applied} concurrent request(s) "
        f"against {slots} slot(s). Set LANGEXTRACT_MAX_WORKERS to override.")


def normalise_resume(resume: Any) -> str:
    """
    Accepts the legacy boolean and the mode strings, so existing callers keep working.

    :param resume: ``True``/``False`` (the pre-mode API), or one of :data:`RESUME_MODES`.
    :returns: A member of :data:`RESUME_MODES`.
    :raises ValueError: On an unrecognized mode string.
    """
    if resume is True:
        return RESUME_IF_EXISTS
    if resume is False or resume is None:
        return RESUME_OFF
    mode = str(resume).strip().lower()
    if mode not in RESUME_MODES:
        raise ValueError(f"resume must be one of {RESUME_MODES}, got {resume!r}")
    return mode


_DOI_INDEX: Optional[Dict[str, str]] = None
_DOI_INDEX_LOCK = threading.Lock()

#: Where each source keeps its document identity. DOI lives in the acquisition metadata, not in
#: the extraction output, so it has to be joined back in.
#:
#: The `*_downloaded.csv` files, NOT `*_metadata.csv`. Both carry an `internal_id` column, but
#: they mean different things by it: in `pubmed_metadata.csv` it is `pubmed_<pmid>`, while a
#: record's `document_id` -- and the extraction filename -- is the acquisition hash, which only
#: `pubmed_downloaded.csv` carries. Joining on the metadata file therefore matched nothing and
#: every logged DOI was `-`. Measured on the current corpus: `pubmed_downloaded.csv` resolves
#: 22,050 of 22,050 extraction stems, `pubmed_metadata.csv` resolves 0, and
#: `scopus_metadata.csv` holds a single empty row while `scopus_downloaded.csv` has 19,011.
_DOI_SOURCES = (
    ("data/literature_data/pubmed_downloaded.csv", "internal_id", "doi"),
    ("data/literature_data/scopus_downloaded.csv", "internal_id", "doi"),
)


def _doi_index() -> Dict[str, str]:
    """
    ``{internal_id: doi}`` for every acquired document, built once per process.

    :returns: The index; empty when no metadata file is readable, which costs the DOI field and
        nothing else.

    Built under a lock rather than with ``functools.lru_cache``. Several worker threads reach
    this within milliseconds of each other on the first document, and an lru_cache lets all of
    them miss and build simultaneously -- the same thundering herd that produced a ~9 GB
    transient spike elsewhere in this pipeline. Only ``internal_id`` and ``doi`` are read: the
    file also carries abstracts, and loading those for a log field would be tens of MB for
    nothing.
    """
    global _DOI_INDEX
    if _DOI_INDEX is not None:
        return _DOI_INDEX
    with _DOI_INDEX_LOCK:
        if _DOI_INDEX is not None:
            return _DOI_INDEX
        index: Dict[str, str] = {}
        for path, id_column, doi_column in _DOI_SOURCES:
            if not os.path.exists(path):
                continue
            try:
                frame = pd.read_csv(path, usecols=[id_column, doi_column], dtype=str)
            except (OSError, ValueError) as e:
                logger.debug(f"no DOI index from {path}: {type(e).__name__}: {e}")
                continue
            for internal_id, doi in zip(frame[id_column], frame[doi_column]):
                if isinstance(internal_id, str) and isinstance(doi, str) and doi.strip():
                    index.setdefault(internal_id.strip(), doi.strip())
        _DOI_INDEX = index
        logger.info(f"DOI index: {len(index)} document(s) with a DOI, for activity logging.")
        return _DOI_INDEX


def _log_field(value: Any) -> str:
    """
    :param value: Any field destined for a log line.
    :returns: A single-line, length-bounded string; ``"-"`` for anything empty.

    Newlines and tabs are replaced rather than stripped: these fields come out of document text,
    and one embedded newline would split a record across two log lines, which quietly breaks any
    later attempt to grep or count them.
    """
    text = "" if value is None else str(value)
    text = text.replace("\t", " ").replace("\n", " ").replace("\r", " ").strip()
    if not text or text.lower() in ("nan", "none"):
        return "-"
    if len(text) > _LOG_FIELD_CHARS:
        return text[:_LOG_FIELD_CHARS - 1] + "\u2026"
    return text


def _log_peptide_activities(internal_id: str, peptides: list) -> None:
    """
    Emits one line per peptide-activity for a finished document.

    :param internal_id: The document id, so a line can be traced back to its source.
    :param peptides: The records about to be written for this document.
    :returns: ``None``.

    Never raises: this is reporting, and a formatting slip on one odd record must not cost the
    document that produced it.
    """
    if not LOG_ACTIVITIES:
        return
    try:
        doi = _doi_index().get(str(internal_id), "")
        for peptide in peptides:
            if not isinstance(peptide, dict):
                continue
            name = _log_field(peptide.get("peptide_name"))
            sequence = _log_field(peptide.get("peptide_sequence"))
            activities = peptide.get("biological_activities") or []
            if not activities:
                logger.info(f"[activity] {internal_id} | name={name} | seq={sequence} | "
                            f"doi={_log_field(doi)} | activity=- | value=- | unit=- | target=-")
                continue
            for activity in activities:
                if not isinstance(activity, dict):
                    continue
                # value as well as unit: a unit with no number beside it says nothing, and the
                # raw string is kept because it carries the operator (">", "<=") that the
                # parsed bounds drop.
                value = activity.get("value_raw")
                if not value:
                    lower, upper = activity.get("value_lower"), activity.get("value_upper")
                    value = f"{lower}-{upper}" if lower is not None and upper is not None else lower
                target = activity.get("target_organism") or activity.get("target_cell_line")
                logger.info(
                    f"[activity] {internal_id} | name={name} | seq={sequence} | "
                    f"doi={_log_field(doi)} | activity={_log_field(activity.get('activity_type'))} | "
                    f"value={_log_field(value)} | unit={_log_field(activity.get('activity_unit'))} | "
                    f"target={_log_field(target)}")
    except Exception as e:                              # noqa: BLE001 -- reporting, never fatal
        logger.debug(f"[{internal_id}] could not log activities: {type(e).__name__}: {e}")


class _BudgetTally:
    """
    Counts documents that ran over the per-document budget, across worker threads.

    A count rather than a list: the ids are already in the log and in each output's metadata,
    so this exists only to put a number in the per-chunk and end-of-run summaries. Without one,
    the budget's effect on a corpus is invisible -- the only evidence would be individually
    unremarkable warning lines scattered through a multi-day log.
    """

    def __init__(self) -> None:
        """:returns: ``None``."""
        self._lock = threading.Lock()
        self.overruns = 0
        self.seconds_lost = 0.0

    def record(self, overshoot: float) -> None:
        """
        :param overshoot: Seconds this document ran beyond its budget.
        :returns: ``None``.
        """
        with self._lock:
            self.overruns += 1
            self.seconds_lost += max(overshoot, 0.0)

    def snapshot(self) -> Tuple[int, float]:
        """:returns: ``(overruns, total overshoot seconds)``."""
        with self._lock:
            return self.overruns, self.seconds_lost


class _Heartbeat:
    """
    Logs progress on a fixed wall-clock interval from a background thread, independent of
    whether any document has actually finished. A report that only fires on task completion goes
    silent for as long as the single slowest in-flight document takes -- which, for a handful of
    unusually candidate-dense documents, has been observed to be tens of seconds. A silent CLI
    is easy to mistake for a hung one; this makes "still working" visible either way.
    """

    def __init__(self, source: str, tasks_total: int, state: Dict[str, Any]) -> None:
        """
        :param source: Source name, used in the log line.
        :param tasks_total: Total documents in the run, for the progress fraction.
        :param state: Shared, mutated counters the heartbeat reads on each tick.
        """
        self._source = source
        self._tasks_total = tasks_total
        self._state = state  # dict with 'processed' and 'total_peptides', mutated by the caller
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """:returns: ``None``. Starts the daemon thread that emits periodic progress lines."""
        self._thread.start()

    def stop(self) -> None:
        """:returns: ``None``. Signals the thread to exit and waits briefly for it."""
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _run(self) -> None:
        """:returns: ``None``. Thread body: logs progress until stopped."""
        t0 = time.time()
        while not self._stop.wait(_HEARTBEAT_SECONDS):
            elapsed = time.time() - t0
            processed = self._state["processed"]
            rate = processed / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"[{self._source}] (heartbeat) {processed}/{self._tasks_total} documents processed "
                f"({rate:.2f} docs/s, {self._state['total_peptides']} peptides so far) -- still running."
            )


#: How many dropped sequences one document's record will list in full. Beyond this the count is
#: still exact and ``dropped_sequences_truncated`` is set -- the cap bounds file size, it does not
#: hide the scale of a drop.
_MAX_RECORDED_DROPS: int = 50

#: Whether each document's tables are read alongside its text. ON by default: measured over 40
#: random PubMed PDFs, 25% carry a table with peptide data (sequence / MIC / strain / activity) that
#: the text layer flattens into unusable runs of numbers, and the table path was previously
#: reachable only from a manual CLI command, so at corpus scale tables were never read at all.
#: Set SAMPLE_EXTRACT_TABLES=0 to disable, for debugging or for a text-only comparison run.
_EXTRACT_TABLES: bool = os.getenv("SAMPLE_EXTRACT_TABLES", "1").strip().lower() not in (
    "0", "false", "no", "off")

#: Marks a peptide that came from a table rather than from prose, alongside the extractor's own
#: "sequence_match"/"name_match" vocabulary. Downstream can tell the two apart, which matters
#: because a table record carries no context window (see _table_peptides).
_TABLE_DISCOVERY_PATH: str = "table_match"

#: Records recovered by the LangExtract backend, which reads the document TEXT with a local model
#: rather than parsing PDF tables. Labelled distinctly from "table_match" so precision stays
#: reportable per backend -- the two find different things and fail differently, and pooling them
#: would hide which one a regression came from.
_LANGEXTRACT_DISCOVERY_PATH: str = "langextract_match"

#: Whether LangExtract runs, and which backend wins a collision.
#:
#: BOTH backends always run and their results are unioned -- this is NOT gap-fill, and an earlier
#: version of this comment said it was, while describing a cost model the code never implemented.
#: They read different things: LangExtract reads prose with a local model and grounds every
#: extraction in a character span, docling parses the PDF's table grids. Running only one loses
#: half the evidence on a document that states a MIC in prose AND tabulates it.
#:
#: The cost is real and worth stating plainly: LangExtract is minutes per document against
#: docling's seconds (measured 101s/document average over the curation benchmark), so unioning
#: over the whole corpus is weeks of wall clock, not days.
#:
#: LANGEXTRACT_BACKEND=0             disables LangExtract (docling only).
#: LANGEXTRACT_BACKEND=1 (default)   both run, LangExtract first and winning collisions.
#: LANGEXTRACT_BACKEND=docling-first both run, docling first and winning collisions.
_LANGEXTRACT_MODE: str = os.getenv("LANGEXTRACT_BACKEND", "1").strip().lower()
USE_LANGEXTRACT_BACKEND: bool = _LANGEXTRACT_MODE not in ("0", "false", "no", "off")
#: LangExtract is PRIMARY by default: it runs first, and wins a collision. Set
#: LANGEXTRACT_BACKEND=docling-first to invert the preference (both still run), or 0 to disable it.
LANGEXTRACT_FIRST: bool = _LANGEXTRACT_MODE not in ("docling-first", "docling", "last")

#: Where docling's OCR'd document text is cached, mirroring the corpus layout.
#:
#: Cached because OCR is the most expensive thing in the pipeline by a wide margin, and because the
#: result is deterministic for a given PDF -- re-running extraction must never pay for it twice.
#: It is deliberately NOT written beside the ``.txt`` it supplements: several steps glob their text
#: directories, and a second ``.txt`` per document there would be ingested as a separate document
#: and double-count every peptide in it.
_OCR_TEXT_CACHE_DIR: str = os.getenv("OCR_TEXT_CACHE", os.path.join("data", "ocr_text"))

#: Gate-1 verdicts that OCR can legitimately overturn.
#:
#: ``empty`` and ``too_short`` are the scanned-document signature: the words are pixels, so the
#: text layer is 0 bytes. ``watermark_only`` belongs here too -- a publisher's stamp over a scanned
#: page is a text layer carrying nothing but the stamp. ``captcha_or_bot_check`` is deliberately
#: ABSENT: that text came out of the PDF, so OCR-ing the same PDF returns the same interstitial,
#: and a CAPTCHA page is junk in every representation.
#: Withhold records that have neither a sequence from the document nor a catalogue-resolved
#: identity, writing them to a sibling tree instead of the main extraction output.
#:
#: Set because such a record cannot reach the database by any route: ``sample.merge`` drops it
#: (``if not seq: continue``), so every stage that carried it was carrying something unpublishable.
#: Measured over 3,000 documents: 72.7% of peptide records and 73.4% of activities.
#:
#: ``SAMPLE_WITHHOLD_UNIDENTIFIED=0`` restores the previous behaviour, which is worth having while
#: the synonym catalogue is still growing -- the gate's output is only as good as what the external
#: sources currently know.
WITHHOLD_UNIDENTIFIED = os.getenv(
    "SAMPLE_WITHHOLD_UNIDENTIFIED", "1").strip().lower() not in ("0", "false", "off", "no", "")

#: Directory suffix for the withheld tree. Deliberately NOT one of
#: :data:`sample.config.EXTRACTION_SOURCE_DIRS`, so ``config.iter_extraction_files`` cannot reach
#: it: the allowlist is what stops a sibling directory being merged by accident.
WITHHELD_SUFFIX = "_withheld"


def _withheld_path(output_path: str) -> str:
    """
    Where a document's withheld records go.

    ``data/literature_data_json/pubmed/x.json`` ->
    ``data/literature_data_json_withheld/pubmed/x.json``, i.e. the corpus directory gains a suffix
    while the per-source subdirectory and filename are preserved, so a withheld record can always
    be traced back to the document and the run that produced it.

    :param output_path: Where the main extraction output is being written.
    :returns: The sidecar path.
    """
    source_dir, filename = os.path.split(output_path)
    corpus_dir, source = os.path.split(source_dir)
    return os.path.join(f"{corpus_dir}{WITHHELD_SUFFIX}", source, filename)


#: Where Ollama keeps its model manifests, used to report a tag's size in a diagnostic.
_OLLAMA_MANIFESTS = "/usr/share/ollama/.ollama/models/manifests/registry.ollama.ai/library"


def _model_size_gib(tag: str) -> Optional[float]:
    """
    :param tag: An Ollama tag such as ``gemma3:4b`` or ``gemma4:latest``.
    :returns: The size of its weights in GiB, or None when the manifest cannot be read (a tag that
        is not pulled, a non-default Ollama root, or a remote server).
    """
    if ":" not in tag:
        return None
    name, _, version = tag.partition(":")
    try:
        with open(os.path.join(_OLLAMA_MANIFESTS, name, version), encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    total = sum(int(layer.get("size") or 0) for layer in manifest.get("layers", [])
                if "model" in str(layer.get("mediaType", "")))
    return total / 2 ** 30 if total else None


def warn_on_model_split(source: str) -> bool:
    """
    Warn when the pipeline's LLM consumers are not all pointed at the same model.

    Seven consumers share one Ollama server here -- candidate triage, activity verification,
    organism normalisation, activity recovery, extraction validation, concentration-unit
    adjudication, and the langextract backend. On a single-GPU host two different models cannot
    both stay resident, so a split makes Ollama evict and reload one for the other continuously.

    This check exists because the fault was introduced TWICE and was invisible both times: the
    symptom is not an error but a different failure on every run, as each request lands in a
    different eviction. Measured over 20 minutes of a split configuration: 31 llama-server
    restarts, 23 "predicted to exceed available memory, evicting", the GPU reading empty whenever
    sampled, the CPU idle, and individual documents failing as a 300s timeout, a 600s timeout and
    an HTTP 500 on three consecutive runs.

    :param source: Corpus name, for the log line.
    :returns: True when a split was found and warned about.
    """
    from sample.config import DEFAULT_EXTRACT_MODEL, DEFAULT_TRIAGE_MODEL
    from sample.langextract_records import MODEL_ENV_VAR

    langextract_model = os.getenv(MODEL_ENV_VAR) or DEFAULT_EXTRACT_MODEL
    configured = {
        "extraction/validation": DEFAULT_EXTRACT_MODEL,
        "candidate triage": DEFAULT_TRIAGE_MODEL,
        "langextract backend": langextract_model,
    }
    distinct = set(configured.values())
    if len(distinct) <= 1:
        return False

    sizes = {tag: _model_size_gib(tag) for tag in distinct}
    known = [size for size in sizes.values() if size]
    combined = f"{sum(known):.1f} GiB combined" if len(known) == len(distinct) else "sizes unknown"
    logger.warning(
        "[%s] LLM consumers are pointed at %d DIFFERENT models (%s; %s). On a single GPU they "
        "cannot both stay resident, so Ollama will evict and reload one for the other on every "
        "alternating request -- throughput collapses and individual documents fail with timeouts "
        "or 500s depending on which eviction they land in. Point DEFAULT_EXTRACT_MODEL and %s at "
        "the same tag.",
        source, len(distinct),
        ", ".join(f"{role}={tag}" for role, tag in configured.items()),
        combined, MODEL_ENV_VAR)
    return True


def server_parallel_slots() -> Optional[int]:
    """
    How many requests the local Ollama server will actually run at once.

    Read from the running ``llama-server``'s own command line (``-np N``) rather than from the
    environment. The setting lives on the SERVER process, so it is absent from this process's
    environment on a correctly-configured host exactly as on a misconfigured one -- which is how a
    default of 1 serialized eight concurrent clients for 95 hours here without a single error. An
    earlier version of the check gave up at that point and printed advice; this reads the answer.

    :returns: The slot count, or None when it cannot be determined -- no model is loaded (Ollama
        starts no runner until the first request), Ollama is on another host, or ``/proc`` is not
        readable. None means "unknown", never "fine".
    """
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as handle:
                    argv = handle.read().split(b"\0")
            except OSError:
                continue
            if not argv or b"llama-server" not in argv[0]:
                continue
            for index, token in enumerate(argv):
                if token == b"-np" and index + 1 < len(argv):
                    value = argv[index + 1].decode("ascii", "ignore")
                    if value.isdigit():
                        return int(value)
    except OSError:
        return None
    return None


_OCR_RESCUABLE_REASONS = frozenset({"empty", "too_short", "watermark_only"})


def _docling_payload(internal_id: str, pdf_path: Optional[str]) -> Dict[str, Any]:
    """
    One docling conversion of one PDF: its tables and its text, together.

    :param internal_id: The document id, for logging.
    :param pdf_path: The PDF, or None.
    :returns: The payload, or ``{}`` on any failure -- docling is additive here, so a missing
        environment or a crashed conversion costs the tables and the OCR text, never the document.
    """
    if not pdf_path:
        return {}
    try:
        from sample.docling_tables import docling_document

        return docling_document(pdf_path) or {}
    except Exception as e:  # noqa: BLE001 -- additive pass; never fatal
        logger.warning(f"[{internal_id}] docling conversion unusable ({type(e).__name__}: {e}).")
        return {}


def _pdf_path_for(text_path: str) -> Optional[str]:
    """
    The PDF a converted text file came from, or None.

    Every corpus stores its text beside its PDFs under a sibling directory whose name carries a
    ``_text`` suffix: ``literature_data_text/pubmed`` beside ``literature_data/pubmed``,
    ``antimicrobial_databases/papers_text/<db>`` beside ``papers/<db>``, and
    ``patent_data_text/google_patents`` beside ``patent_data/google_patents``. Dropping the suffix is
    therefore one rule for all three rather than a per-source table that would silently miss a
    corpus added later.

    :param text_path: Path to the document's ``.txt``.
    :returns: The existing PDF path, or None when the naming does not match or the file is absent --
        a document with no PDF simply gets no table pass.
    """
    directory, filename = os.path.split(text_path)
    parts = directory.split(os.sep)
    renamed = False
    for index, part in enumerate(parts):
        if part.endswith("_text"):
            parts[index] = part[: -len("_text")]
            renamed = True
    if not renamed:
        return None
    candidate = os.path.join(os.sep.join(parts), os.path.splitext(filename)[0] + ".pdf")
    return candidate if os.path.exists(candidate) else None


def _table_peptides(internal_id: str, text_path: str,
                    payload: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> Tuple[list, Optional[str]]:
    """
    Peptides read from the document's tables, shaped like the extractor's own records.

    :param internal_id: The document id, recorded on each record as ``document_id``.
    :param text_path: Path to the document's ``.txt``; the PDF is derived from it.
    :param payload: A :func:`_docling_payload` result for this document's PDF, when the caller
        already converted it. Reused rather than reconverted.
    :param metadata: The document's metadata dict, annotated when the LangExtract gate skips the
        model. Every other gate in this stage records why it fired; this one must too, or a
        document that was never sent to the model is indistinguishable from one the model read and
        found nothing in.
    :returns: ``(records, pdf_path)``. Both empty/None when there is no PDF, when the
        ``sample-docling`` environment is not built, or when the conversion fails -- none of which
        is an extraction error, so the text-derived peptides stand on their own.
    """
    from sample.sequence import sequence_to_1_letter, sequence_to_3_letter, sequence_to_helm

    pdf_path = _pdf_path_for(text_path)

    def _docling() -> list:
        if not pdf_path:
            return []
        from sample.docling_tables import extract_tables_from_pdf

        # `or []` and not `.get("tables")`: a failed shared conversion yields {}, and passing
        # None from it would mean "no tables supplied, convert it yourself" -- re-running the one
        # operation the caller already paid for and lost. On a scan that timed out at 900s that
        # turned the document's cost into 1800s for the same nothing. An empty list correctly
        # means "the conversion produced no tables", whether it failed or the PDF simply has none.
        tables = (payload.get("tables") or []) if payload is not None else None
        return extract_tables_from_pdf(pdf_path, tables=tables)

    def _langextract() -> list:
        if not USE_LANGEXTRACT_BACKEND:
            # Recorded, because step 8 runs in two phases by default and the phases are only
            # distinguishable from the record. Without this a phase-1 record (gemma4 never
            # attempted) looks exactly like a phase-2 record where gemma4 ran and found nothing,
            # and a half-finished phase 2 becomes impossible to audit or resume correctly.
            if metadata is not None:
                metadata["langextract_backend"] = "disabled"
            return []
        if metadata is not None:
            metadata["langextract_backend"] = "enabled"
        # LangExtract reads docling's OCR output in PREFERENCE to the pdfplumber .txt. On a
        # document with no text layer the .txt is empty, so this is the difference between the
        # model reading the paper and reading nothing at all; where the text layer is healthy no
        # OCR runs and this resolves to the .txt unchanged.
        ocr_path = _ocr_text_for(internal_id, text_path, pdf_path, payload=payload)
        source_path = ocr_path or text_path
        try:
            from sample.langextract_records import (LANGEXTRACT_GATE, extract_records_from_text,
                                                    worth_running)

            # GATE, applied to the text the model would actually read -- the OCR output when there
            # is one, never the pdfplumber .txt. Gating on the .txt would skip exactly the 36.7%
            # of patents whose text layer is empty and which the OCR rescue just recovered.
            #
            # Only the EXPENSIVE pass is gated. The NLP, flashtext, catalogued-sequence and table
            # passes have already run on this document, so a gated-out document still yields
            # whatever they found; the cost is confined to peptides only the model would have seen
            # (measured: 2.4% of those at the default "unit" gate, for 8.5% fewer calls).
            source_text = _document_source_text(source_path)
            if not worth_running(source_text):
                if metadata is not None:
                    metadata["langextract_gate"] = LANGEXTRACT_GATE
                    metadata["langextract_skipped"] = True
                logger.debug(f"[{internal_id}] gate {LANGEXTRACT_GATE!r} skipped the model "
                             f"({len(source_text)} chars, no concentration unit)")
                return []
            if metadata is not None:
                metadata["langextract_gate"] = LANGEXTRACT_GATE
                metadata["langextract_skipped"] = False

            records = extract_records_from_text(source_path)
        except Exception as e:  # noqa: BLE001 -- one backend failing must never sink the others
            logger.warning(f"[{internal_id}] langextract backend unusable ({type(e).__name__}: "
                           f"{e}).")
            return []
        # Provenance: which text the model actually read. Without this an OCR-derived record is
        # indistinguishable from a text-layer one, and OCR has its own error modes -- so precision
        # has to stay reportable separately for it.
        for record in records or []:
            if isinstance(record, dict):
                record["text_source"] = "docling_ocr" if ocr_path else "pdfplumber"
        return records

    # BOTH backends run and their results are UNIONED -- they are not alternatives.
    #
    # They read different things and fail differently: langextract reads prose with a local model
    # and grounds every extraction in a character span, while docling parses the PDF's table grids,
    # where this corpus keeps most of its measurements. First-wins discarded whichever ran second,
    # so a document with both a prose-stated MIC and a results table lost half its evidence.
    # Together with the text extractor's own NLP/flashtext/LLM pass (merged by
    # _merge_table_records), every method contributes and each record keeps the provenance of the
    # method that found it, so precision stays reportable per path.
    #
    # langextract is PRIMARY: it runs first, and on a collision its record is the one kept, with
    # the other backend's activities folded in. Each backend degrades to an empty list on every
    # failure mode it has, so a missing environment or an unreachable model simply contributes
    # nothing rather than breaking the union.
    collected: list = []
    if LANGEXTRACT_FIRST:
        sources = ((_langextract, _LANGEXTRACT_DISCOVERY_PATH), (_docling, _TABLE_DISCOVERY_PATH))
    else:
        sources = ((_docling, _TABLE_DISCOVERY_PATH), (_langextract, _LANGEXTRACT_DISCOVERY_PATH))
    for produce, path in sources:
        for record in produce() or []:
            if isinstance(record, dict):
                record["discovery_path"] = path
                collected.append(record)

    records, duplicates = _dedupe_records(collected)
    if duplicates:
        logger.debug(f"[{internal_id}] merged {duplicates} record(s) found by more than one "
                     f"backend")

    if not records:
        return [], pdf_path

    from sample.preprocessing import standardize_and_filter_activities

    for record in records:
        record["document_id"] = internal_id
        # Table-derived activities go through the SAME standardisation the text path applies
        # (extractor.py), rather than straight into the record.
        #
        # They did not, and that was a real defect: every guard in that function -- unit
        # canonicalisation, the mM->uM conversion, the CFU/zone-diameter rejection, and the
        # _MAX_PLAUSIBLE_POTENCY_UM bound -- was bypassed for the table backend, which produced
        # 44% of the last run's peptides. The measured consequences were a MIC of "0.1 M" taken
        # from a buffer-composition table, a MIC in "%" taken from a percent-killing column, and
        # Polymyxin B at 1.56 mg/mL (1000x its real MIC). The text path rejects all three.
        record["biological_activities"] = standardize_and_filter_activities(
            record.get("biological_activities") or [])
        raw = record.get("peptide_sequence") or ""
        record["peptide_sequence_1letter"] = sequence_to_1_letter(raw) if raw else ""
        record["peptide_sequence_3letter"] = sequence_to_3_letter(raw) if raw else ""
        record["peptide_sequence_helm"] = sequence_to_helm(raw) if raw else ""
        for activity in record.get("biological_activities") or []:
            # value_raw and value_lower are the same number here: nothing in this path converts
            # magnitudes, so the raw unit IS the canonical one. Stating it explicitly keeps
            # preprocessing.raw_value_and_unit from falling back and pairing a converted value with
            # an unconverted unit.
            activity.setdefault("value_raw_unit", activity.get("activity_unit", ""))
            activity.setdefault("context_sentence", "")
            activity.setdefault("target_cell_line", None)
    return records, pdf_path



def _ocr_cache_path(text_path: str) -> str:
    """
    Where this document's OCR'd text is cached.

    Mirrors the document's path under :data:`_OCR_TEXT_CACHE_DIR` so a cache entry can be traced
    back to its source by eye. A document stored outside ``data/`` cannot be mirrored, so it falls
    back to a digest of its absolute path -- unreadable but collision-free, which is the property
    that actually matters for a cache key.

    :param text_path: Path to the document's ``.txt``.
    :returns: The cache file path. Not guaranteed to exist.
    """
    absolute = os.path.abspath(text_path)
    relative = os.path.relpath(absolute, os.path.abspath("data"))
    if relative.startswith(".."):
        import hashlib

        digest = hashlib.sha1(absolute.encode("utf-8")).hexdigest()
        relative = os.path.join(digest[:2], digest + ".txt")
    return os.path.join(_OCR_TEXT_CACHE_DIR, relative)


def _ocr_text_for(internal_id: str, text_path: str, pdf_path: Optional[str],
                  payload: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    A path to this document's OCR'd text, converting it first if it is not already cached.

    This is what makes a scanned document readable at all. Measured over 60 sampled patents
    (1,400 pages), 36.7% carry a text layer of exactly ZERO characters: for those the pdfplumber
    ``.txt`` is empty, every text-based extractor sees nothing, and the document contributes
    nothing no matter how good the extractor is. docling renders each page to an image and OCRs it
    with tesseract, which produces the only readable content those documents have.

    :param internal_id: The document id, for logging.
    :param text_path: Path to the document's ``.txt``; used as the cache key.
    :param pdf_path: The source PDF, or None when the document has none.
    :param payload: A :func:`_docling_payload` result for this same PDF, when the caller already
        converted it. Its text is used instead of converting again.
    :returns: Path to a non-empty OCR text file, or None when there is no PDF, when the text layer
        is already healthy, when docling is unavailable, when OCR is disabled, or when the
        conversion produced nothing. None always means "fall back to the pdfplumber text", never an
        error.
    """
    if not pdf_path:
        return None

    cache_path = _ocr_cache_path(text_path)
    try:
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            return cache_path
    except OSError:
        pass

    try:
        from sample.docling_tables import docling_text, should_ocr

        # Gate BEFORE converting. docling runs TableFormer in ACCURATE mode whether or not OCR is
        # needed, so calling it for a document whose text layer is already healthy pays a full
        # conversion to recover text the .txt already contains -- over the 124,526-paper corpus,
        # which measured 100% healthy, that is hundreds of hours bought for nothing.
        if not should_ocr(pdf_path):
            return None

        # should_ocr is consulted even when a payload is in hand. The payload's text is docling's
        # reading of the document whether or not OCR ran, and on a healthy text layer that is a
        # text-layer transcription, not OCR output -- using it here would silently relabel it
        # "docling_ocr" and switch LangExtract off the .txt the rest of the pipeline reads.
        text = payload.get("text") if payload is not None else docling_text(pdf_path)
    except Exception as e:  # noqa: BLE001 -- OCR is additive; never fatal to the text path
        logger.warning(f"[{internal_id}] docling OCR unusable ({type(e).__name__}: {e}).")
        return None
    if not text:
        return None

    # Written via a temporary file in the same directory and renamed, so a crash or a timeout
    # mid-write cannot leave a truncated cache entry that later runs would trust and reuse.
    try:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        temporary = cache_path + ".partial"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(temporary, cache_path)
    except OSError as e:
        logger.warning(f"[{internal_id}] could not cache OCR text ({e}); using it uncached.")
        return None
    logger.debug(f"[{internal_id}] cached {len(text)} chars of OCR text")
    return cache_path



def _document_source_text(path: str) -> str:
    """
    :param path: The text file the model would read.
    :returns: Its contents, or "" when unreadable -- which the gate treats as "nothing to read".
    """
    try:
        with open(path, encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except OSError:
        return ""


def _record_key(record: dict) -> Optional[str]:
    """
    The identity two records are considered the same peptide by.

    Sequence when there is one -- it is the real identity -- otherwise the normalised name. Name
    matters because the backends disagree about what they can recover: langextract's records are
    frequently name-only (``peptide_sequence: null``), while docling often reads the sequence out
    of a table column. Keying on sequence alone would emit "Polymyxin B" twice, once from each
    backend, and inflate every per-peptide count downstream.

    :param record: A peptide record.
    :returns: The key, or ``None`` when the record carries neither a sequence nor a name and
        therefore cannot be matched to anything.
    """
    sequence = (record.get("peptide_sequence") or "").strip().upper()
    if sequence:
        return f"seq:{sequence}"
    name = (record.get("peptide_name") or "").strip().lower()
    return f"name:{name}" if name else None


def _dedupe_records(records: list) -> Tuple[list, int]:
    """
    Unions records from several backends, folding duplicates together.

    The FIRST record for a key wins its identity and provenance -- callers order the backends by
    preference -- and later duplicates contribute their activities to it. A record that duplicates
    an earlier one is therefore not lost: only its peptide-level fields are, which are the fields
    the two backends agreed on anyway.

    :param records: Records from every backend, in preference order.
    :returns: ``(unioned, duplicates_merged)``.
    """
    out: list = []
    by_sequence: Dict[str, dict] = {}
    by_name: Dict[str, dict] = {}
    duplicates = 0
    for record in records:
        sequence = (record.get("peptide_sequence") or "").strip().upper()
        name = (record.get("peptide_name") or "").strip().lower()
        if not sequence and not name:
            out.append(record)
            continue
        # Looked up by sequence AND by name, not by one composite key. The backends recover
        # different fields for the same molecule -- langextract commonly returns a name with no
        # sequence while docling reads the sequence out of a table column -- so a single key made
        # those two records hash differently and "Magainin 2" survived twice, once per backend.
        existing = by_sequence.get(sequence) if sequence else None
        if existing is None and name:
            by_label = by_name.get(name)
            if by_label is not None:
                # A name match is only trusted when the two sequences do not CONTRADICT each other.
                # Generic labels are rife in this corpus -- "Peptide 1", "P1", "compound 3" -- and
                # merging on the label alone discarded a genuinely different peptide and
                # re-attributed its activities to the one that happened to be seen first. Falling
                # back to the name is still right when one side has no sequence: that is the
                # langextract-name/docling-sequence case this union exists to join.
                other = (by_label.get("peptide_sequence") or "").strip().upper()
                if not (sequence and other and sequence != other):
                    existing = by_label
        if existing is None:
            if sequence:
                by_sequence[sequence] = record
            if name:
                by_name[name] = record
            out.append(record)
            continue
        duplicates += 1
        existing.setdefault("biological_activities", []).extend(
            record.get("biological_activities") or [])
        # A sequence recovered by the second backend fills a gap the first left, which is one of
        # the concrete reasons to union rather than choose. Once filled, the record becomes
        # findable by sequence too.
        if sequence and not (existing.get("peptide_sequence") or "").strip():
            existing["peptide_sequence"] = record.get("peptide_sequence")
            by_sequence[sequence] = existing
        if name and name not in by_name:
            by_name[name] = existing
        # Record that more than one method saw it -- a corroboration signal worth keeping.
        paths = existing.setdefault("discovery_paths", [existing.get("discovery_path", "")])
        if record.get("discovery_path") and record["discovery_path"] not in paths:
            paths.append(record["discovery_path"])
    return out, duplicates


def _merge_table_records(peptides: list, table_records: list) -> Tuple[list, int]:
    """
    Folds table-derived records into the text-derived ones, by sequence.

    A peptide reported in both prose and a table is one peptide: the table usually holds the
    measurements and the prose the description, so the activities are concatenated onto the existing
    record rather than emitted as a second copy of the same molecule.

    :param peptides: Records from the text extractor.
    :param table_records: Records from the table pass.
    :returns: ``(merged, added)`` where ``added`` counts peptides the tables contributed that the
        text did not already have.
    """
    by_sequence = {}
    for peptide in peptides:
        if isinstance(peptide, dict):
            sequence = (peptide.get("peptide_sequence") or "").strip().upper()
            if sequence:
                by_sequence.setdefault(sequence, peptide)

    merged = list(peptides)
    added = 0
    for record in table_records:
        sequence = (record.get("peptide_sequence") or "").strip().upper()
        existing = by_sequence.get(sequence) if sequence else None
        if existing is not None:
            existing.setdefault("biological_activities", []).extend(
                record.get("biological_activities") or [])
            continue
        merged.append(record)
        added += 1
        if sequence:
            by_sequence[sequence] = record
    return merged, added


def _extract_document(task: Tuple[Any, ...], tally: Optional["_BudgetTally"] = None,
                      budget: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """
    Runs one document under a wall-clock budget.

    :param task: ``(internal_id, text_path, output_path)``.
    :param tally: Optional shared counter for over-budget documents, so the per-chunk and
        end-of-run summaries can report them as a number instead of leaving the only evidence
        in scattered log lines.
    :param budget: Seconds this document may spend; :data:`DOCUMENT_BUDGET_SECONDS` when
        omitted, and 0 for no budget.
    :returns: Whatever :func:`_extract_document_work` returns -- ``(internal_id, peptide_count,
        error)``.

    The budget is opened here rather than inside the worker so it covers everything the
    document does, including reading the text and the docling conversion, and so the worker
    body stays a straight line of extraction logic. See
    :data:`DOCUMENT_BUDGET_SECONDS` for what the cap does and does not guarantee; in short, it
    prevents new expensive work rather than interrupting work in progress, so a document can
    overshoot by as long as its longest already-running operation.
    """
    from sample import deadlines

    limit = DOCUMENT_BUDGET_SECONDS if budget is None else budget
    started = time.monotonic()
    with deadlines.Budget(limit):
        result = _extract_document_work(task)
    spent = time.monotonic() - started

    if limit > 0 and spent > limit:
        if tally is not None:
            tally.record(spent - limit)
        logger.warning(
            f"[{task[0]}] took {spent:.0f}s against a {limit:.0f}s budget; "
            f"its remaining model passes were skipped, so this document is extracted but "
            f"incomplete (metadata.budget_exceeded)."
        )
    return result


def _extract_document_work(task: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
    """
    One document in, one JSON file out.

    Three gates run around the extractor itself, all added from measured failures in the step-14
    curation audit. Each records WHY it fired in the output record's metadata rather than acting
    silently: an empty extraction whose reason is "this was a CAPTCHA page" is a different fact
    from an empty extraction that genuinely found nothing, and downstream evaluation depends on
    being able to tell them apart.
    """
    internal_id, text_path, output_path = task
    from sample.concentration import activity_unit_is_coherent
    from sample.document_quality import (activity_supported_by_text, is_unusable_document,
                                         looks_like_review_article,
                                         sequence_supported_by_context)
    from sample.extractor import extract_peptides_from_text

    try:
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return internal_id, 0, f"failed to read {text_path}: {e}"

    metadata: Dict[str, Any] = {"source_file": text_path}

    pdf_path = _pdf_path_for(text_path)
    # One docling conversion per document, shared by the OCR rescue immediately below and by the
    # table pass much further down. Converting a scanned PDF twice would double the single most
    # expensive operation in the pipeline (~12 s/page measured), so it is done once, lazily, and
    # only if something actually asks for it.
    _conversion: Dict[str, Any] = {}
    _converted = [False]

    def conversion() -> Dict[str, Any]:
        if not _converted[0]:
            _converted[0] = True
            _conversion.update(_docling_payload(internal_id, pdf_path))
        return _conversion

    # OCR RESCUE -- deliberately BEFORE gate 1, and it replaces the document's text outright.
    #
    # Gate 1 rejects an empty or near-empty document and returns immediately. For a scanned PDF the
    # pdfplumber text IS empty: measured over 60 sampled patents (1,400 pages), 36.7% have a text
    # layer of exactly ZERO characters, and their .txt files are 0 bytes on disk. Running the gate
    # first therefore skipped every one of them as "empty" before any PDF-based pass could run --
    # the documents that need OCR most were precisely the ones guaranteed never to get it, which is
    # why the patent corpus yielded almost nothing.
    #
    # The rescued text becomes `text`, rather than being handed only to the LangExtract backend, so
    # the NLP, flashtext and LLM passes inside extract_peptides_from_text all read it, and so
    # gate 3 below validates each sequence against the same text it was found in.
    unusable, reason = is_unusable_document(text)
    if unusable and reason in _OCR_RESCUABLE_REASONS and pdf_path:
        recovered = (conversion().get("text") or "").strip()
        if recovered:
            metadata.update({"text_source": "docling_ocr",
                             "text_layer_chars": len(text),
                             "ocr_text_chars": len(recovered)})
            logger.info(f"[{internal_id}] text layer unusable ({reason}); OCR recovered "
                        f"{len(recovered)} chars")
            text = recovered
            # Re-judged on the recovered text: OCR output can itself be junk, and a rescue that
            # bypassed the gate entirely would let a blank scan through as a real document.
            unusable, reason = is_unusable_document(text)
            metadata["ocr_rescue_succeeded"] = not unusable

    # GATE 1: not a document at all. ~2.3% of the literature corpus is Cloudflare/CAPTCHA
    # interstitial text saved as a paper. Extracting from one wastes the work and, far worse,
    # produces an empty result indistinguishable from "this real paper reports no peptide".
    if unusable:
        metadata.update({"skipped": True, "skip_reason": reason, "peptide_count": 0})
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "peptides": []}, f, indent=2, ensure_ascii=False)
        return internal_id, 0, None

    # GATE 2: a review, whose peptides belong to other papers. Marked, never dropped: the records
    # are real and useful, they simply must not be attributed to this document. Downstream decides
    # what to do with them; losing them here would be an unrecoverable guess.
    is_review, review_evidence = looks_like_review_article(text)
    if is_review:
        metadata.update({"document_type": "review", "review_evidence": review_evidence})

    try:
        peptides = extract_peptides_from_text(text, doc_id=internal_id)
    except Exception as e:
        return internal_id, 0, f"extraction failed: {e}"

    # TABLE PASS: the measurements a paper prints in its tables rather than its prose. The text
    # layer flattens a table into a run of numbers with no column structure, so a MIC table's values
    # are unattributable from text alone -- which is why this runs from the PDF. It is additive: a
    # failure here (no PDF, sample-docling not built, a conversion that crashes) leaves the
    # text-derived peptides exactly as they were, and is recorded rather than raised.
    if _EXTRACT_TABLES:
        try:
            table_records, pdf_path = _table_peptides(internal_id, text_path,
                                                      payload=conversion(),
                                                      metadata=metadata)
        except Exception as e:                                  # noqa: BLE001 - recorded, not fatal
            logger.debug(f"[{internal_id}] table extraction failed: {e}")
            table_records, pdf_path = [], None
            metadata["table_extraction_error"] = f"{type(e).__name__}: {e}"
        # Recorded whenever the pass RAN, not only when it found something. Writing these keys only
        # on a hit makes "this paper has no tables" and "the pass never executed" look identical in
        # the output, and the second is a bug that would go unnoticed across the whole corpus --
        # exactly the failure this comment previously claimed to prevent while doing the opposite.
        metadata["table_pass_ran"] = pdf_path is not None
        if pdf_path:
            metadata["table_source_pdf"] = os.path.basename(pdf_path)
            metadata["table_records_found"] = len(table_records)
        if table_records:
            peptides, added = _merge_table_records(peptides, table_records)
            metadata["table_peptides_added"] = added
            # How many of those carry a sequence, recorded separately because the two numbers can
            # differ completely and only this one predicts what reaches the database.
            # merge-extractions keys on peptide_sequence and drops any record without one, with no
            # name fallback (sample/merge.py) -- so a document whose tables are keyed by peptide
            # CODE ("H2"/"H4", "DN"/"DNB") yields records that look like a success here and are
            # discarded two stages later. Measured on the first two documents that yielded
            # anything: table_peptides_added would have reported 7 while the real, database-bound
            # peptide yield was 0.
            metadata["table_peptides_with_sequence"] = sum(
                1 for record in table_records if (record.get("peptide_sequence") or "").strip())

    # GATE 3: a sequence that does not appear in its own recorded context window was captured from
    # running prose by a regex, not read off the page. Dropped rather than flagged -- unlike a
    # review's peptides, there is no sense in which this record is correct.
    kept, dropped_records = [], []
    for pep in peptides:
        if isinstance(pep, dict) and not sequence_supported_by_context(
                pep.get("peptide_sequence", ""), pep.get("context", "")):
            # Record WHAT was dropped, not just how many. This gate performs the only irreversible
            # deletion in the extraction stage, and recording a bare count made its 743 deletions
            # across 519 documents impossible to audit -- when the drop rate was questioned, the
            # values had to be guessed at by re-running the regex, which produced a wrong answer.
            # Every other gate here records its reason (skip_reason, review_evidence); this one
            # must record its evidence too, or it cannot be checked and cannot be undone.
            dropped_records.append({
                "peptide_sequence": pep.get("peptide_sequence", ""),
                "peptide_name": pep.get("peptide_name", ""),
                "discovery_path": pep.get("discovery_path", ""),
                # Enough context to judge the call without re-reading the source document.
                "context_excerpt": (pep.get("context") or "")[:300],
            })
            continue
        if is_review and isinstance(pep, dict):
            pep["from_review_article"] = True
        kept.append(pep)
    if dropped_records:
        metadata["dropped_sequence_not_in_context"] = len(dropped_records)
        # Bounded: a pathological document must not turn its record into a dump. The count above
        # stays exact even when this list is truncated, and the truncation is stated rather than
        # silent.
        metadata["dropped_sequences"] = dropped_records[:_MAX_RECORDED_DROPS]
        if len(dropped_records) > _MAX_RECORDED_DROPS:
            metadata["dropped_sequences_truncated"] = True

    # GATE 4: a measurement whose number is nowhere in the document.
    #
    # This gate exists because the model was caught copying its own prompt into its output. The
    # LangExtract worker's few-shot example is "Peptide JG436 inhibited MCF-7 cells with an IC50
    # of 12.5 uM", and measured over one run the tuple IC50 / 12.5 uM / MCF-7 was emitted for 49
    # different documents -- 38 of which contain neither "12.5" nor "MCF" anywhere in their text.
    # The next most repeated measurement in the same corpus appeared in 3 documents, which is
    # what coincidence on a common value actually looks like.
    #
    # The activity is dropped, not the peptide: the peptide may well be real and correctly named
    # even when a measurement attributed to it was invented. Counted in the metadata, and the
    # dropped activities are recorded so the call can be audited rather than taken on trust.
    unsupported = 0
    unsupported_examples = []
    incoherent = 0
    incoherent_examples = []
    for pep in kept:
        if not isinstance(pep, dict):
            continue
        activities = pep.get("biological_activities") or []
        if not activities:
            continue
        grounded = []
        for activity in activities:
            # GATE 4b: a unit that cannot express the activity it is attached to. Checked before
            # grounding because an incoherent pairing is wrong even when both halves appear in
            # the text -- and they usually do: `MIC = 2548.2 Da` is a real number from a real
            # mass-spec table, read into the wrong column. Grounding cannot catch that; only the
            # type/unit relationship can. 6,173 of 53,107 activities on disk (11.6%) are such
            # pairings, 2,852 of them molecular weights filed as MICs.
            coherent, unit_reason = activity_unit_is_coherent(
                (activity or {}).get("activity_type"), (activity or {}).get("activity_unit"))
            if not coherent:
                incoherent += 1
                if len(incoherent_examples) < _MAX_RECORDED_DROPS:
                    incoherent_examples.append({
                        "peptide_name": pep.get("peptide_name", ""),
                        "activity_type": (activity or {}).get("activity_type", ""),
                        "value_raw": (activity or {}).get("value_raw", ""),
                        "activity_unit": (activity or {}).get("activity_unit", ""),
                        "reason": unit_reason,
                    })
                continue
            ok, reason = activity_supported_by_text(activity, text,
                                                    str(pep.get("discovery_path") or ""))
            if ok:
                grounded.append(activity)
                continue
            unsupported += 1
            if len(unsupported_examples) < _MAX_RECORDED_DROPS:
                unsupported_examples.append({
                    "peptide_name": pep.get("peptide_name", ""),
                    "activity_type": (activity or {}).get("activity_type", ""),
                    "value_raw": (activity or {}).get("value_raw", ""),
                    "activity_unit": (activity or {}).get("activity_unit", ""),
                    "target": ((activity or {}).get("target_organism")
                               or (activity or {}).get("target_cell_line") or ""),
                    "reason": reason,
                })
        pep["biological_activities"] = grounded
    if incoherent:
        metadata["dropped_activity_unit_incoherent"] = incoherent
        metadata["dropped_activity_units"] = incoherent_examples
    if unsupported:
        metadata["dropped_activity_not_in_text"] = unsupported
        metadata["dropped_activities"] = unsupported_examples

    # SYNONYM RESOLUTION, deliberately AFTER gate 3.
    #
    # Ordering is load-bearing. Gate 3 drops any sequence that does not appear in its own context
    # window, which is right for a sequence a regex lifted out of prose -- but a sequence resolved
    # from the CATALOGUE is not in the document at all, by definition, so running this before gate
    # 3 would hand it a sequence and then immediately delete the record for not containing it.
    # Gate 3 judges what was read off the page; this tier carries its own evidence trail instead.
    #
    # 81% of extracted peptides carry a name and an activity but no sequence (8,270 of 10,193 in
    # the current run), and sample.merge discards every one of them -- `if not seq: continue`. This
    # recovers the ~7% that can be resolved from evidence strong enough to keep, and RECORDS the
    # rejected attempts so the rest can be revisited without starting over.
    resolved = 0
    resolution_methods: Dict[str, int] = {}
    try:
        from sample import synonyms

        for pep in kept:
            if not isinstance(pep, dict) or (pep.get("peptide_sequence") or "").strip():
                continue
            outcome = synonyms.resolve(pep, text)
            if not outcome:
                continue
            method = str(outcome.get("method") or "unknown")
            resolution_methods[method] = resolution_methods.get(method, 0) + 1
            if synonyms.apply_resolution(pep, outcome):
                resolved += 1
    except Exception as e:  # noqa: BLE001 -- additive; a failure must not cost the document
        logger.warning(f"[{internal_id}] synonym resolution unusable ({type(e).__name__}: {e}).")
        metadata["synonym_resolution_error"] = f"{type(e).__name__}: {e}"
    if resolution_methods:
        metadata["sequences_resolved"] = resolved
        metadata["sequence_resolution_methods"] = resolution_methods

    # IDENTITY GATE. A record earns a place in the extraction output on one of two grounds:
    #
    #   1. it carries a sequence read off the page  -> a new peptide, with its activities;
    #   2. its name resolved to a peptide the external sources already know, with evidence above
    #      sample.synonyms.MIN_CONFIDENCE_TO_JOIN -> new activities for a KNOWN peptide.
    #
    # Everything else is a name and a number with no identity behind it. `sample.merge` already
    # discards those (`if not seq: continue`), so carrying them this far only inflated every
    # downstream stage: measured over 3,000 documents, 72.7% of records and 73.4% of activities
    # never reach the database. The curation review set was dominated by them too, which is why a
    # reviewer was being asked to grade records that could not be published whatever the verdict.
    #
    # Withheld, NOT deleted, and the distinction is the whole design. The catalogue grows -- a
    # name unresolvable today resolves once its source is imported -- and re-deriving these would
    # mean re-running the LLM pass, which is a multi-day job. They are written to a sibling tree
    # that config.iter_extraction_files does not read, so no downstream stage picks them up by
    # accident (the allowlist exists because an os.walk over a sibling directory once shipped a
    # database with 22.7% duplicate rows).
    # NOT when the resolver failed. A thrown resolver means "we never asked", which is
    # indistinguishable in the output from "we asked and nothing matched" -- and withholding on
    # that basis would discard records that would have resolved, for an infrastructural reason.
    # The gate needs a real negative answer, not the absence of an answer.
    if WITHHOLD_UNIDENTIFIED and not metadata.get("synonym_resolution_error"):
        from sample.sequence import validate_extracted_sequence

        identified, withheld = [], []
        reasons: Dict[str, int] = {}
        for record in kept:
            # A non-dict is never discarded here: this gate judges identity, and something that
            # is not a record is a different problem, handled where records are built.
            if not isinstance(record, dict):
                identified.append(record)
                continue
            # VALIDATED, not merely non-empty. Asking only "is the field populated?" is how
            # table labels ('N = 17'), formulas ('C2H5'), catalogue codes ('RC-160'), OCR debris,
            # nucleotide reads and compound names ('CHOLESTEROL' under the name 'Cholesterol')
            # all reached the output carrying a sequence. Measured over 1,554 logged activities
            # from one run, only 14.4% of sequence fields were plain residue strings -- and every
            # one of the rest satisfied the old check.
            valid, reason = validate_extracted_sequence(record.get("peptide_sequence"),
                                                        record.get("peptide_name") or "")
            if valid:
                identified.append(record)
                continue
            # The reason travels with the record, so a withheld peptide can be re-judged later
            # without re-deriving why it was withheld.
            record["withheld_reason"] = reason
            reasons[reason] = reasons.get(reason, 0) + 1
            withheld.append(record)
        if reasons:
            metadata["withheld_sequence_reasons"] = reasons
        if withheld:
            metadata["withheld_unidentified"] = len(withheld)
            metadata["withheld_activities"] = sum(
                len(p.get("biological_activities") or []) for p in withheld
                if isinstance(p, dict))
            withheld_path = _withheld_path(output_path)
            try:
                os.makedirs(os.path.dirname(withheld_path), exist_ok=True)
                with open(withheld_path, "w", encoding="utf-8") as handle:
                    json.dump({"metadata": {**metadata, "withheld_from": output_path},
                               "peptides": withheld}, handle, indent=2, ensure_ascii=False)
            except OSError as e:
                # Never lose the document over the sidecar: if it cannot be written, keep the
                # records in the main output rather than dropping them unrecorded.
                logger.warning(f"[{internal_id}] could not write withheld records "
                               f"({type(e).__name__}: {e}); keeping them in the main output.")
                identified = kept
        kept = identified

    metadata["peptide_count"] = len(kept)

    # Always recorded when a budget is in force, both the time taken and whether it ran out.
    # Writing these only on an overrun would make a document that finished comfortably
    # indistinguishable from one processed before budgets existed, and the whole value of the
    # flag is being able to separate "this document genuinely has few peptides" from "this
    # document was cut short", which is a question asked of the data long after the run.
    from sample import deadlines
    if deadlines.active():
        metadata["processing_seconds"] = round(deadlines.elapsed() or 0.0, 1)
        metadata["budget_seconds"] = deadlines.budget()
        if deadlines.expired():
            metadata["budget_exceeded"] = True

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "peptides": kept}, f, indent=2, ensure_ascii=False)

    # After the write, and only for the records that survived the gates -- these lines are meant
    # to show what the run is PUBLISHING, so logging withheld or gate-dropped records here would
    # overstate the yield. Those are in the _withheld sidecar and in metadata.dropped_sequences.
    _log_peptide_activities(internal_id, kept)

    return internal_id, len(kept), None


def run_parallel_pipeline(source: str, input_dir: str, output_dir: str, df: pd.DataFrame,
                          workers: int = 1, resume: Any = False,
                          chunk_size: Optional[int] = None,
                          document_budget: Optional[float] = None) -> None:
    """
    Extracts peptide data for every document in `df`, writing one `<output_dir>/<internal_id>.json`
    file per document (schema: {"metadata": {...}, "peptides": [...]}) -- the contract every
    downstream stage (validate, external-enrich, merge-extractions, the webserver) expects.

    Uses a thread pool, not a process pool. This pipeline's per-worker memory footprint is
    dominated by large read-only lookup structures (an Aho-Corasick automaton over ~2M taxonomy
    name variants, a spaCy vocab, dictionary/lexicon sets, an ONNX NLI model -- collectively
    several GB once warmed). A process pool was tried first on the assumption that fork's
    copy-on-write semantics would let worker processes share that memory; measured in practice,
    each forked worker still independently grew to 1.5-3.5GB, because CPython's reference
    counting dirties (and thus copies) the pages behind any Python object a worker actually
    touches -- and extraction touches these lookup structures on every document. That
    duplication reliably exhausted RAM and swap on real runs. Threads share one process's memory
    outright (no copying, hoped-for or otherwise): there is exactly one automaton, one vocab, one
    NLI model, regardless of worker count. The candidate-discovery and NLI-inference hot paths
    are implemented in C/C++ extensions (pyahocorasick, ONNX Runtime) that release the GIL during
    their actual work, so this still parallelizes real CPU work across cores; only the
    surrounding pure-Python glue serializes.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(df)

    resume_mode = normalise_resume(resume)
    tasks, skipped_existing, skipped_missing = build_task_list(
        input_dir, output_dir, df, resume_mode)

    logger.info(
        f"[{source}] {len(tasks)} documents queued for extraction (resume={resume_mode}: "
        f"{skipped_existing} already done, {skipped_missing} missing input text, {total} total)."
    )
    if not tasks:
        return

    effective_workers = max(1, min(workers, os.cpu_count() or 1))
    if workers > effective_workers:
        logger.warning(f"[{source}] --workers {workers} clamped to {effective_workers} (CPU core count).")

    logger.info(
        f"[{source}] Starting with {effective_workers} worker thread(s). The first document(s) "
        f"pay a one-time cost to load shared lookups (spaCy, taxonomy, known-sequence automata) "
        f"-- a silent gap of up to ~30s before the first progress line below is expected, not a hang."
    )

    # Per-document time is dominated by Ollama round trips, and the Ollama *server* decides how
    # many run concurrently (OLLAMA_NUM_PARALLEL, commonly defaulting to 1-4). Worker threads
    # beyond that just queue on the server, so a mismatch shows up as "more workers changed
    # nothing" with no error anywhere. Surfaced here because it is set on the server process, not
    # this one -- exporting it in this shell has no effect on an already-running ollama serve.
    warn_on_model_split(source)

    # Prefer what the server is actually doing over what this environment happens to say.
    #
    # The comparison must be against TOTAL concurrency, not --workers. Each worker thread holds
    # one document, and the refined pass fans that document out across LANGEXTRACT_MAX_WORKERS
    # threads of its own (default 6), so the request count is the product. An earlier version of
    # this warning compared slots against --workers alone and understated the real figure by that
    # factor: at --workers 4 it reported "2 of every 4 calls will queue" when 24 requests were
    # being issued against 2 slots and 22 were queueing.
    #
    # Its advice was wrong in the same way. Raising OLLAMA_NUM_PARALLEL is only right when there
    # is memory for the extra KV cache; on a 15GB host it is not. Measured here while 24 requests
    # were in flight against 2 slots: llama-server grew from 1,363MB to 7,833MB and available RAM
    # fell from 6,624MB to 1,086MB -- the same swap-thrash condition that OLLAMA_NUM_PARALLEL=2
    # and OLLAMA_KV_CACHE_TYPE=q8_0 were set to escape. So the recommendation leads with reducing
    # client concurrency, which costs nothing, and mentions raising slots only as the alternative.
    _slots = server_parallel_slots()
    # Fit the per-document fan-out to the server BEFORE reporting, so the warning below describes
    # what will actually be issued rather than what the default would have issued. Document-level
    # workers are left alone: a document in flight makes progress on page text, tables and gates
    # between model calls, so parallelism there is useful beyond the slot count, while a second
    # thread inside one document only splits the same call queue.
    if _slots is not None:
        _fit_langextract_workers(_slots, effective_workers, source)
    _lx_workers = max(1, _langextract_max_workers())
    _requests = effective_workers * _lx_workers
    if _slots is not None:
        # One coherent pair whose PRODUCT fits the server, not two knobs each of which is
        # individually insufficient. Document-level parallelism is spent first because a whole
        # document in flight makes progress on page text, tables and gates between model calls,
        # while a second thread inside one document only splits the same call queue.
        _suggest = (min(effective_workers, _slots), max(1, _slots // min(effective_workers, _slots)))
        if _slots < _requests:
            logger.warning(
                f"[{source}] Ollama has {_slots} parallel slot(s) (llama-server -np {_slots}) but "
                f"this run can issue {_requests} concurrent request(s): --workers "
                f"{effective_workers} x LANGEXTRACT_MAX_WORKERS {_lx_workers}. "
                f"{_requests - _slots} of every {_requests} will sit in OLLAMA_MAX_QUEUE, "
                f"consuming each document's budget while waiting and growing llama-server's "
                f"prompt-cache footprint. Prefer lowering concurrency to match the server: "
                f"WORKERS={_suggest[0]} LANGEXTRACT_MAX_WORKERS={_suggest[1]} "
                f"(= {_suggest[0] * _suggest[1]} request(s)). Raise OLLAMA_NUM_PARALLEL to "
                f"{_requests} instead only if the host has RAM for {_requests} KV caches.")
        else:
            logger.info(f"[{source}] Ollama is serving {_slots} requests in parallel "
                        f"(-np {_slots}) for {_requests} concurrent request(s) "
                        f"(--workers {effective_workers} x {_lx_workers}).")
    elif (_num_parallel := os.environ.get("OLLAMA_NUM_PARALLEL")) and _num_parallel.isdigit():
        if int(_num_parallel) < effective_workers:
            logger.warning(
                f"[{source}] OLLAMA_NUM_PARALLEL={_num_parallel} is below --workers "
                f"{effective_workers}; LLM calls will queue and the extra threads will not add "
                f"throughput. Restart the Ollama server with OLLAMA_NUM_PARALLEL>={effective_workers}."
            )
    elif effective_workers > 1:
        # WARNING, and at >1 worker rather than >4. This branch used to be an info line that only
        # appeared above four workers, and it is the reason an 8x serialization ran unnoticed for
        # 95 hours: recent Ollama defaults OLLAMA_NUM_PARALLEL to 1, the variable is set on the
        # SERVER so it is absent from this process's environment on a correctly-configured host
        # just as on a misconfigured one, and the symptom is only that throughput is flat. Measured
        # on this host with the default: the single slot was busy 97.7% of a 10-minute window at
        # 0.32 calls/s while eight client requests were in flight, so seven of eight were queued.
        #
        # It cannot be verified from here -- Ollama exposes no endpoint reporting its slot count --
        # so this says what to check rather than asserting a fault.
        logger.warning(
            f"[{source}] OLLAMA_NUM_PARALLEL is not set in this environment, so this cannot tell "
            f"how many requests the Ollama SERVER will run at once. Recent versions default to 1, "
            f"which would serialize every LLM call behind one slot and cap throughput regardless "
            f"of --workers {effective_workers}. Verify with: "
            f"`systemctl show ollama -p Environment` (and `pgrep -a llama-server` -- look for "
            f"`-np <n>`). Set it to at least {effective_workers} on the server if it is lower."
        )

    # Counters live in one dict rather than in closure locals: the heartbeat thread, the
    # progress reporter and the per-chunk summaries all read them, and they have to survive the
    # executor being torn down and rebuilt between chunks.
    budget = DOCUMENT_BUDGET_SECONDS if document_budget is None else document_budget
    state = {"processed": 0, "total_peptides": 0}
    tally = _BudgetTally()
    t0 = time.time()
    heartbeat = _Heartbeat(source, len(tasks), state)
    heartbeat.start()

    _REPORT_EVERY_N = 5

    def _report(force: bool = False) -> None:
        """
        :param force: Emit the progress line even if the reporting interval has not elapsed.
        :returns: ``None``.
        """
        processed = state["processed"]
        if force or processed % _REPORT_EVERY_N == 0 or processed == len(tasks):
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0.0
            logger.info(
                f"[{source}] {processed}/{len(tasks)} documents processed "
                f"({rate:.2f} docs/s, {state['total_peptides']} peptides extracted so far)."
            )

    def _run_chunk(chunk: list) -> None:
        """
        Processes one chunk and returns only when every document in it is finished.

        :param chunk: The ``(internal_id, text_path, output_path)`` tuples to process.
        :returns: ``None``.

        A FRESH executor per chunk, shut down with ``wait=True`` before this returns. That is
        what makes the chunk boundary a real barrier: the Ollama reset that follows must not
        land while a worker still has a request outstanding, or that document gets a hard
        failure for no reason of its own. Reusing one long-lived pool and merely waiting on
        futures would leave idle-but-alive threads and no guarantee about what they hold;
        joining them costs a few thread creations per hundred documents and buys certainty.
        The expensive shared state (spaCy, the taxonomy automaton, the NLI model) is
        module-level and lru_cached, so it is NOT reloaded -- only threads are recreated.
        """
        if effective_workers == 1:
            for task in chunk:
                internal_id, n_peptides, error = _extract_document(task, tally, budget)
                state["processed"] += 1
                state["total_peptides"] += n_peptides
                if error:
                    logger.warning(f"[{source}] {internal_id}: {error}")
                _report()
            return

        task_iter = iter(chunk)
        window = effective_workers * _SUBMISSION_WINDOW_PER_WORKER
        in_flight = {}
        executor = ThreadPoolExecutor(max_workers=effective_workers)
        try:
            def _submit_more() -> None:
                """
                Tops the in-flight set back up to the submission window.

                Work is submitted incrementally rather than all at once so that a large corpus
                does not materialize every future up front.

                :returns: ``None``.
                """
                while len(in_flight) < window:
                    task = next(task_iter, None)
                    if task is None:
                        return
                    fut = executor.submit(_extract_document, task, tally, budget)
                    in_flight[fut] = task[0]

            _submit_more()
            while in_flight:
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    internal_id = in_flight.pop(fut)
                    try:
                        _, n_peptides, error = fut.result()
                        state["total_peptides"] += n_peptides
                        if error:
                            logger.warning(f"[{source}] {internal_id}: {error}")
                    except Exception as e:
                        logger.error(f"[{source}] {internal_id}: worker crashed: {e}")
                    state["processed"] += 1
                    _report()
                _submit_more()
        finally:
            # wait=True by design: see the docstring. Every future above has already completed,
            # so this joins idle threads and returns promptly -- unless a worker is wedged, in
            # which case blocking here is correct. Proceeding would restart the model underneath
            # it, and a pool that is never joined is how threads accumulate across a long run.
            executor.shutdown(wait=True)

    def _barrier(label: str) -> None:
        """
        Confirms nothing is still running, then resets the Ollama runtime.

        :param label: Chunk position, for the log line.
        :returns: ``None``.

        Two independent checks, because they can disagree and the disagreement is the
        interesting case. The executor has been joined, so no worker thread exists; but
        :func:`sample.llm.inflight_calls` counts calls at the semaphore, and the timeout path
        in ``llm_call`` abandons a thread that outlives the caller by design. A non-zero count
        here means such a call is still running, and unloading the model under it would turn a
        slow call into a failed one -- so it is waited out, briefly, and reported either way.
        """
        from sample import llm
        from sample import ollama_runtime

        waited = 0.0
        while llm.inflight_calls() > 0 and waited < _BARRIER_GRACE_SECONDS:
            time.sleep(0.5)
            waited += 0.5
        stragglers = llm.inflight_calls()
        if stragglers:
            logger.warning(
                f"[{source}] {label}: {stragglers} LLM call(s) still in flight after "
                f"{waited:.0f}s; resetting anyway. These are calls abandoned by a timeout, "
                f"which Ollama finishes regardless -- they will fail at the unload, and the "
                f"documents that started them have already been written."
            )
        elif waited:
            logger.info(f"[{source}] {label}: waited {waited:.0f}s for the last LLM call to "
                        f"return before resetting.")

        summary = ollama_runtime.reset()
        if summary.get("error"):
            # Not fatal. The chunk that just finished is on disk; a failed reset only means the
            # next chunk runs with the accumulated state that chunking exists to avoid, which
            # is the old behaviour rather than a new fault.
            logger.warning(f"[{source}] {label}: Ollama reset incomplete "
                           f"({summary['error']}); continuing without it.")
        else:
            logger.info(f"[{source}] {label}: Ollama reset via {summary.get('method')} in "
                        f"{summary.get('seconds')}s "
                        f"(unloaded: {', '.join(summary.get('unloaded') or []) or 'nothing'}). "
                        f"The next chunk starts against a fresh runner.")

    # CHUNKS. Consecutive slices of the task list, so --resume and a Ctrl-C leave the corpus in
    # the same state they always did: a chunk boundary is a pacing decision, not a unit of work
    # anything downstream knows about.
    requested = EXTRACT_CHUNK_SIZE if chunk_size is None else chunk_size
    size = requested if requested > 0 else len(tasks)
    chunks = [tasks[i:i + size] for i in range(0, len(tasks), size)] or [[]]
    if len(chunks) > 1:
        logger.info(
            f"[{source}] Running in {len(chunks)} chunk(s) of up to {size} documents, resetting "
            f"the Ollama runtime between them; each document has a "
            f"{budget:.0f}s budget."
        )

    try:
        for index, chunk in enumerate(chunks, start=1):
            chunk_t0 = time.time()
            chunk_peptides_before = state["total_peptides"]
            _run_chunk(chunk)
            chunk_elapsed = time.time() - chunk_t0
            if len(chunks) > 1:
                overruns, _ = tally.snapshot()
                logger.info(
                    f"[{source}] chunk {index}/{len(chunks)} done: {len(chunk)} documents in "
                    f"{chunk_elapsed:.0f}s "
                    f"({len(chunk) / chunk_elapsed if chunk_elapsed > 0 else 0.0:.2f} docs/s, "
                    f"{state['total_peptides'] - chunk_peptides_before} peptides, "
                    f"{overruns} over budget run-to-date)."
                )
            # Between chunks only. After the last one there is no next chunk to protect, and
            # unloading here would just make whichever step runs next pay to load the model
            # again.
            if index < len(chunks):
                _barrier(f"chunk {index}/{len(chunks)}")
    finally:
        heartbeat.stop()

    processed = state["processed"]
    elapsed = time.time() - t0
    rate = processed / elapsed if elapsed > 0 else 0.0
    overruns, seconds_lost = tally.snapshot()
    logger.info(
        f"[{source}] Extraction complete: {processed} documents, {state['total_peptides']} peptides "
        f"in {elapsed:.1f}s ({rate:.2f} docs/s, {effective_workers} workers). Output: {output_dir}"
    )
    if overruns:
        logger.warning(
            f"[{source}] {overruns} of {processed} documents ({100.0 * overruns / max(processed, 1):.1f}%) "
            f"exceeded the {budget:.0f}s budget and were extracted incompletely, "
            f"overshooting by {seconds_lost:.0f}s in total. They are flagged with "
            f"metadata.budget_exceeded; re-run them by deleting those outputs and using --resume."
        )


def process_all_extractions(source: str, input_dir: str, output_dir: str, limit: int = 0,
                            workers: int = 1, resume: Any = False,
                            chunk_size: Optional[int] = None,
                            document_budget: Optional[float] = None) -> None:
    """Entrypoint for the extraction architecture: locates the source's document list, then runs it."""
    logger.info(f"Preparing extraction for {source} from {input_dir} (workers={workers}, resume={resume})")
    os.makedirs(output_dir, exist_ok=True)

    if source in ["google_patents", "patents"]:
        patent_csv = "data/patent_data/google_patents.csv"
        if not os.path.exists(patent_csv):
            logger.error(f"Google Patents CSV file {patent_csv} not found.")
            return
        df_raw = pd.read_csv(patent_csv, skiprows=[0])
        patent_ids = []
        for _, row in df_raw.iterrows():
            link = row.get('result link')
            if pd.notna(link) and link:
                parts = link.split('/')
                if len(parts) > 4:
                    patent_ids.append(parts[4])
        df = pd.DataFrame({"internal_id": patent_ids, "success": True})
    elif source in ["adp", "dramp", "dbamp", "dbaasp", "camp"]:
        db_csv = f"data/antimicrobial_databases/papers/{source}/{source}_downloaded.csv"
        if not os.path.exists(db_csv):
            logger.error(f"Downloaded CSV file {db_csv} not found.")
            return
        df = pd.read_csv(db_csv)
    else:
        downloaded_csv = f"data/literature_data/{source}_downloaded.csv"
        if not os.path.exists(downloaded_csv):
            logger.error(f"Downloaded CSV file {downloaded_csv} not found.")
            return
        df = pd.read_csv(downloaded_csv)

    if 'internal_id' not in df.columns:
        logger.error("CSV missing 'internal_id' column")
        return

    if limit and limit > 0:
        df = df.head(limit)

    run_parallel_pipeline(source, input_dir, output_dir, df, workers=workers, resume=resume,
                          chunk_size=chunk_size, document_budget=document_budget)


def process_single_pdf_file(pdf_path: str, output_path: str) -> None:
    """Entrypoint for extracting from a single PDF."""
    from sample.extractor import extract_peptides_from_text

    logger.info(f"Extracting from single PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        logger.error(f"Missing file: {pdf_path}")
        return

    text = extract_text_from_pdf(pdf_path)
    peptides = extract_peptides_from_text(text)
    output_data = {"metadata": {"source_file": pdf_path}, "peptides": peptides}

    if output_path == "-":
        print(json.dumps(output_data, indent=4, ensure_ascii=False))
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved JSON extraction to {output_path}")
