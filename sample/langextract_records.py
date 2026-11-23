"""
LangExtract (google/langextract) as the default structured-extraction backend, against local Ollama.

**What this is for.** The docling backend reads peptide activity records out of PDF *tables*. This
one reads them out of the document *text* with a local LLM, and returns records in the same shape,
so the two are interchangeable at the call site -- langextract is tried first and docling is the
fallback (see :func:`sample.pipeline_runner._table_records_for`).

**Why prefer it.** Every langextract extraction carries a ``char_interval`` locating it in the
source document. That is source grounding the text path never had: a record can be checked against
the characters it came from rather than trusted. The audit that drove this work turned on exactly
that gap -- a sequence recorded with a context window that did not contain it, and a provenance
question that could not be answered because nothing recorded where a record came from.

**Local only, by construction.** langextract defaults to Gemini and will call Google's API if given
a key. This module never does: ``model_url`` always points at the local Ollama server and
``model_id`` is always an Ollama tag (:data:`sample.config.DEFAULT_EXTRACT_MODEL`). No API key is
read, set or required. ``fence_output`` and ``use_schema_constraints`` are deliberately left unset
because Ollama does not support ``output_schema`` and upstream documents that the provider factory
auto-configures both.

**Run out-of-process.** Driven as a subprocess against ``sample-langextract``'s own interpreter,
resolved by absolute path, for the same reason as the docling backend: this project pins
sample-extraction to numpy<2 for spacy, and resolving another dependency tree into it moves that
pin. ``conda run`` is never used -- ``KNOWN_ISSUES.md`` records that it misroutes on this host.
"""
import functools
import hashlib
import json
import logging
import os
import random
import re
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("sample.langextract_records")

#: The isolated environment this backend runs in. See ``environments/sample-langextract.yml``.
LANGEXTRACT_ENV = "sample-langextract"

#: Local Ollama server. Overridable for a host that runs it elsewhere; never a remote provider.
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

#: Seconds before one document's extraction is abandoned. Generous because a local 8-12B model on a
#: full-length paper is slow; the caller falls back to docling rather than losing the document.
#:
#: Raised 600 -> 1200 together with REQUEST_TIMEOUT (300 -> 600). The two are coupled and must not
#: be changed independently: this one kills the subprocess, that one gives up on a single HTTP
#: request inside it. Set equal, the subprocess would be killed at the same instant the request
#: expired, so a slow document would die as an opaque subprocess timeout instead of the clean "no
#: records, fall back" path -- and the worker would never get to write its partial payload.
DEFAULT_TIMEOUT = int(os.getenv("LANGEXTRACT_DOCUMENT_TIMEOUT", "1200"))

#: Characters of a document sent to the model. Matches the text path's own bound so the two
#: backends see the same evidence.
MAX_CHARS = 200_000

#: Characters per chunk, and the Ollama context window that must hold one.
#:
#: TUNED BY MEASUREMENT, not by reasoning -- the obvious reasoning was wrong. langextract's default
#: of 1000 chars splits a paper into many small calls, so the intuitive fix is a much larger chunk.
#: Measured on one 15kB document, same model, same machine:
#:
#:     chunk/ctx     time   records  activities
#:     1000/2048      65s      2         0        (library default)
#:     3000/4096      48s      4         2        <- fastest AND better
#:     6000/8192     122s      6         5        (2.5x slower)
#:
#: Pushing to 6000/8192 made a 54kB paper go from 2m55s to 5m14s. The reason was hardware: that
#: measurement was taken with an 11.3GB model on a 12GB card, so a large KV cache had nowhere to
#: live and the per-call cost grew faster than the call count fell. 3000/4096 was the knee.
#:
#: THAT PREMISE NO LONGER HOLDS. The default model is now a ~3.3GB one (see
#: :data:`MODEL_ENV_VAR`), which leaves most of the card free, and the note above predicted that a
#: card with real headroom "would likely favour the larger setting, which finds more". That is now
#: worth re-measuring and has NOT been done -- the values are left where measurement put them
#: rather than moved on the strength of the prediction.
MAX_CHAR_BUFFER = int(os.getenv("LANGEXTRACT_MAX_CHAR_BUFFER", "3000"))
#: Derived from sample.config.DEFAULT_NUM_CTX rather than repeated as a literal: the two must be
#: equal or Ollama rebuilds the runner between this backend's calls and everything else's, and a
#: rule enforced only by a comment is a rule that drifts.
def _default_num_ctx() -> int:
    """:returns: The pinned context window, from LANGEXTRACT_NUM_CTX or the project default."""
    raw = os.getenv("LANGEXTRACT_NUM_CTX")
    if raw and raw.strip().isdigit():
        return int(raw)
    from sample.config import DEFAULT_NUM_CTX
    return DEFAULT_NUM_CTX


NUM_CTX = _default_num_ctx()

#: Concurrent chunk requests PER DOCUMENT. langextract's own default is 10.
#:
#: Raised from 2 to 6 on 2026-10-14, and the reason is not the one the previous note gave. That
#: note paired this against the server's slot count, on the assumption that the GPU was the scarce
#: resource. Measured while the step ran with 8 extraction workers and 8 server slots, it is not:
#:
#:     langextract worker subprocesses alive : 9   (aged 133-169s, each at 0.2-0.3% CPU,
#:                                                  all blocked in do_poll on an idle socket)
#:     HTTP requests Ollama received         : 31  in 20 minutes
#:     actual generation time in that window : 7 seconds
#:     CPU idle 98%, GPU 4%, no swap traffic
#:
#: The backend spawns ONE interpreter per document, each loading langextract and its dependency
#: stack. At eight extraction workers that is eight concurrent cold starts, and the per-document
#: startup -- not the model, not the GPU, not the conversion semaphore -- was the whole cost.
#: Documents then tripped REQUEST_TIMEOUT and were written with zero records.
#:
#: Raising THIS number amortises that startup across more chunks inside one subprocess, which is
#: the opposite lever to raising --workers: that multiplies the number of startups. With
#: --workers 4 this gives 24 requests in flight, which queues inside Ollama (its own queue is far
#: larger) rather than adding interpreters. Measured against this host's 2 slots that cost a 12x
#: oversubscription, so :func:`set_max_workers` now derives the value from the server's real slot
#: count unless LANGEXTRACT_MAX_WORKERS is set explicitly.
#:
#: The structural fix is to stop spawning per document -- batch N documents per subprocess, or keep
#: a persistent pool -- and this is the cheap approximation of it, not a substitute.
MAX_WORKERS = int(os.getenv("LANGEXTRACT_MAX_WORKERS", "6"))

#: Whether the operator pinned MAX_WORKERS, in which case nothing derives it away from them.
MAX_WORKERS_PINNED: bool = "LANGEXTRACT_MAX_WORKERS" in os.environ


def set_max_workers(value: int, force: bool = False) -> int:
    """
    Lowers the per-document fan-out to fit the model server, unless the operator pinned it.

    The default of 6 is wrong whenever the server has fewer slots than ``--workers x 6``, and
    silently so: the surplus requests queue inside Ollama rather than failing. Measured on this
    host with 2 slots and ``--workers 4`` -- 24 requests in flight -- documents spent their whole
    300s budget waiting, langextract subprocesses hit their own 299s request timeout and fell
    back to the non-model path, and llama-server's prompt-cache footprint grew from 1,363MB to
    9,030MB while available RAM fell from 6,624MB to 1,086MB. Throughput was 0.02 docs/s.

    Called by :func:`sample.pipeline_runner.run_parallel_pipeline` once the server's real slot
    count is known. Mutates the module global because the worker command line is built from it at
    call time, so there is one place the number comes from.

    :param value: Desired fan-out; clamped to at least 1.
    :param force: Apply even when ``LANGEXTRACT_MAX_WORKERS`` was set explicitly.
    :returns: The value now in force.
    """
    global MAX_WORKERS
    if MAX_WORKERS_PINNED and not force:
        return MAX_WORKERS
    MAX_WORKERS = max(1, int(value))
    return MAX_WORKERS

#: Model for this backend specifically. Defaults to the pipeline's own extraction model so there is
#: one convention, but is overridable: this pass is chunk-heavy, so a smaller local model can be a
#: better trade here than for the main extraction path.
MODEL_ENV_VAR = "LANGEXTRACT_MODEL"

#: Seconds Ollama is asked to hold the model loaded between requests.
#:
#: Passed explicitly because langextract sends a keep_alive of its own on every request
#: (``_DEFAULT_KEEP_ALIVE = 300``), and a per-request value overrides the server's
#: ``OLLAMA_KEEP_ALIVE`` -- so configuring 24h on the systemd unit does nothing while this backend
#: is the caller. Left at an hour: long enough that no realistic gap between documents evicts the
#: model, short enough that an abandoned run releases the GPU on its own.
KEEP_ALIVE = int(os.getenv("LANGEXTRACT_KEEP_ALIVE", "3600"))

#: Whether to let a reasoning family emit a chain of thought on the extraction call.
#:
#: Follows sample.llm.THINK_ENABLED by default so the two paths cannot disagree -- a split here
#: would mean the same model reasoning on one of this pipeline's seven Ollama consumers and not
#: the others, which is exactly the kind of inconsistency that makes a quality change
#: unattributable. Override with LANGEXTRACT_THINK.
def _think_default() -> bool:
    """:returns: The thinking setting, from LANGEXTRACT_THINK or sample.llm's."""
    raw = os.getenv("LANGEXTRACT_THINK")
    if raw is not None:
        return raw.strip().lower() not in ("0", "false", "off", "no", "")
    from sample.llm import THINK_ENABLED
    return THINK_ENABLED

#: Per-HTTP-request timeout to Ollama, in seconds.
#:
#: langextract's Ollama provider defaults this to 120s and raises InferenceRuntimeError when it
#: expires, which the worker reports as "no records" for the whole document -- so a single slow
#: chunk, or a model paying its load time inside the first request, silently costs every peptide in
#: that document. It must stay comfortably below DEFAULT_TIMEOUT, which bounds the subprocess as a
#: whole.
#:
#: Raised 300 -> 600 on 2026-10-14. At 300s two ordinary documents timed out REPEATEDLY -- the same
#: two files on consecutive runs, 6 and 15 chunks respectively, no pathological content -- and each
#: was then written with ZERO peptides. A timeout here is silent data loss, not a slow document,
#: which is what makes a longer wait the right trade.
#:
#: The cost is real and worth stating: a document that is going to fail now occupies a worker for
#: ten minutes before anyone finds out, and with --workers 4 that is a quarter of the pool. If
#: timeouts become common rather than occasional, the answer is to fix the queueing (see
#: MAX_WORKERS) rather than to raise this again.
REQUEST_TIMEOUT = int(os.getenv("LANGEXTRACT_REQUEST_TIMEOUT", "600"))

#: Extra attempts after a retryable failure. 1 is deliberate rather than conservative: measured
#: failures are ~16% and independent of the document, so one retry takes the expected loss to
#: ~2.6% while costing a second attempt on only the 16% that failed.
RETRIES = int(os.getenv("LANGEXTRACT_RETRIES", "1"))

#: Seconds before the first retry, doubling per attempt with jitter. Jitter matters because
#: several worker threads hit the same wall at the same instant; without it they would retry in
#: lockstep and re-create the contention that caused the failure.
RETRY_BACKOFF = float(os.getenv("LANGEXTRACT_RETRY_BACKOFF", "2.0"))

#: A retry is only started with at least this long left in the document's budget. A model pass
#: on a real document takes tens of seconds, so attempting one with less than this is spending a
#: server slot to guarantee a second failure.
MIN_RETRY_SECONDS = float(os.getenv("LANGEXTRACT_MIN_RETRY_SECONDS", "30"))

#: Failures worth a second attempt: a busy or wedged server, not a bad document. A 500 is here
#: because on this host it is a prompt-cache allocation failure -- transient and load-dependent,
#: so the same document succeeds moments later.
_RETRYABLE_MARKERS = ("500", "502", "503", "504", "server busy", "no slots available",
                      "maximum pending requests", "connection", "connectionerror",
                      "remote end closed", "timed out", "timeout")


def _is_retryable(error: str) -> bool:
    """
    :param error: The error text the worker reported.
    :returns: Whether a second attempt could plausibly succeed.

    Matched on the message because the worker runs in an isolated environment and can only hand
    back a string -- it cannot raise a typed exception across the process boundary.
    """
    lowered = (error or "").lower()
    return any(marker in lowered for marker in _RETRYABLE_MARKERS)


def _cleanup_payload_dir(payload_dir: str, payload_path: str) -> None:
    """
    Removes the staging directory for one document.

    :param payload_dir: The temporary directory.
    :param payload_path: The payload file inside it.
    :returns: ``None``.

    Pulled out of a ``finally`` so it can run once after a retry loop rather than once per
    attempt -- deleting the staged input between attempts would leave the retry nothing to read.
    """
    try:
        for leftover in (payload_path, os.path.join(payload_dir, "input.txt")):
            if os.path.exists(leftover):
                os.remove(leftover)
        os.rmdir(payload_dir)
    except OSError:
        pass

#: Where worker payloads are cached. Set LANGEXTRACT_CACHE="" to disable.
#:
#: This backend costs 101.4 s per document (measured over the curation benchmark), which is the
#: dominant cost in the pipeline by a wide margin -- roughly 1,301 GPU-hours for one pass over the
#: 46,189-document corpus. The extraction stage is re-run often while the stages after it are being
#: developed, and without a cache every re-run pays that in full for documents whose text and model
#: have not changed. The payload is a few kB per document.
LANGEXTRACT_CACHE_DIR = os.getenv("LANGEXTRACT_CACHE", os.path.join("data", "langextract_cache"))

#: A concentration unit anywhere in the document. The gate for the expensive pass.
_CONCENTRATION_UNIT = re.compile(
    r"(µ|u|n|m|p)\s?g\s?/\s?m?[lL]|(µ|u|n|m|p)\s?M\b|mg\s?/\s?L", re.I)

#: An assay name. Used only by the aggressive gate; see :data:`LANGEXTRACT_GATE`.
_ASSAY_NAME = re.compile(
    r"\b(MIC|MBC|MFC|IC\s?50|EC\s?50|LC\s?50|LD\s?50|GI\s?50|CC\s?50|"
    r"h(a)?emoly|cytotox|antimicrob|antibacter|antifung|bacteriostat)\b", re.I)

#: Which documents earn a gemma4 call. The CHEAP passes -- NLP, flashtext, the catalogued-sequence
#: automaton and the table grids -- always run regardless; this gates only the expensive one.
#:
#: Measured over 63 documents holding 247 LangExtract-attributable peptides, against 4,000 others:
#:
#:   gate                   calls saved    LangExtract peptides lost
#:   "none"                      0.0%                         0.0%
#:   "unit"      (default)       8.5%                         2.4%
#:   "assay+unit"               55.8%                        69.2%
#:
#: "assay+unit" is NOT the default despite saving over half the calls, because its loss is not a
#: uniform sample. LangExtract's value is concentrated in documents whose text does NOT contain
#: clean assay-and-unit strings -- scanned, OCR'd and table-heavy ones -- so that gate is
#: ANTI-CORRELATED with where the model helps, and discards roughly seven of every ten peptides
#: only it would have found. "unit" buys ~110 of the ~1,301 GPU-hours a full pass costs, for 2.4%.
LANGEXTRACT_GATE = os.getenv("LANGEXTRACT_GATE", "unit").strip().lower()

#: Strip a paper's bibliography before spending model calls on it.
#:
#: The reference list is 20-30% of a typical paper and cannot contain a bioactivity measurement,
#: but it was being chunked and sent to the model like everything else. Measured over 400 real
#: papers that pass the unit gate, removing it cuts model calls by 20.7% -- and it is the SAFE kind
#: of reduction, unlike the "assay+unit" gate above: it drops structural boilerplate rather than
#: filtering on signal, so it is not anti-correlated with the hard, OCR'd, table-heavy documents
#: where this backend earns its cost.
#:
#: Reuses :func:`sample.nlp.truncate_references`, which the candidate-discovery path already
#: applies for a related reason -- citations yield author surnames and years that come back as
#: fabricated peptide records.
TRUNCATE_REFERENCES = os.getenv(
    "LANGEXTRACT_TRUNCATE_REFERENCES", "1").strip().lower() not in ("0", "false", "off", "no", "")

#: Smallest share of a document that truncation may leave behind before it is distrusted.
#:
#: ``truncate_references`` falls back to a citation-density heuristic when it finds no explicit
#: header, and on some documents that heuristic fires early and removes nearly everything.
#: Measured over 298 papers: the median keeps 77.8%, but the 5th percentile keeps 0.0%. Rather than
#: send an empty document to the model, anything below this fraction is treated as a misfire and
#: the full text is used -- 9.7% of papers take that path, which is the price of not silently
#: extracting a paper from nothing.
MIN_KEPT_FRACTION = float(os.getenv("LANGEXTRACT_MIN_KEPT_FRACTION", "0.40"))

#: Path fragment identifying the patent corpus, where truncation is never applied.
#:
#: Not a preference -- a measurement. ``truncate_references`` is built for a manuscript whose
#: bibliography sits at the end; a patent's prior-art citations sit near the FRONT, so the heuristic
#: cuts the body instead. Measured over 198 patents: the median keeps 0.0% of the text and 75.8%
#: fall below any usable floor. Applying it to patents would have extracted three of every four
#: from an empty string.
_PATENT_PATH_MARKER = "patent_data_text"


def worth_running(text: str) -> bool:
    """
    Whether this document earns a gemma4 call.

    :param text: The text the model would ACTUALLY read -- for a scanned document that is the OCR
        output, not the empty pdfplumber ``.txt``. Testing the ``.txt`` would skip precisely the
        36.7% of patents the OCR rescue exists to recover.
    :returns: True when the document should be sent to the model. True for an unrecognised gate
        name, so a typo in the environment costs throughput rather than recall.
    """
    if LANGEXTRACT_GATE in ("none", "0", "off", ""):
        return True
    body = text or ""
    if not body.strip():
        # Nothing to read at all. The cheap passes see the same nothing, so this costs no recall.
        return False
    if LANGEXTRACT_GATE == "assay+unit":
        return bool(_ASSAY_NAME.search(body)) and bool(_CONCENTRATION_UNIT.search(body))
    if LANGEXTRACT_GATE == "unit":
        return bool(_CONCENTRATION_UNIT.search(body))
    logger.warning(f"unknown LANGEXTRACT_GATE={LANGEXTRACT_GATE!r}; running on every document.")
    return True


@functools.lru_cache(maxsize=1)
def _worker_fingerprint() -> str:
    """
    A digest of the worker script, used in the cache key.

    The prompt and the few-shot examples live in that file, so hashing it means any change to what
    the model is asked invalidates every cached payload automatically. A hand-maintained version
    constant would have to be remembered, and a stale cache after a prompt change is indistinguishable
    from a model that simply did not improve.

    :returns: A short hex digest, or "unknown" when the worker cannot be read.
    """
    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_langextract_worker.py")
    try:
        with open(worker, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:12]
    except OSError:
        return "unknown"


def _cache_path(prepared_text: str, model_id: str) -> Optional[str]:
    """
    Where this document's payload is cached, or None when caching is disabled.

    Keyed on the text the model WAS ACTUALLY SENT -- :func:`model_input_text`, not the file on disk
    -- rather than on the path. So an OCR pass that rewrites a document gets a fresh extraction, an
    unchanged document is reused even if it moved, and turning reference truncation on or off
    changes the key by itself, because it changes what the model saw. Keying on the raw file would
    have served a pre-truncation payload for a post-truncation request.

    MAX_WORKERS is deliberately NOT part of the key: it changes throughput, not output, so tuning
    concurrency must not discard the cache. Temperature is fixed at 0, so a hit is a genuine
    reproduction.

    :param prepared_text: The text handed to the worker.
    :param model_id: The resolved Ollama model tag.
    :returns: The cache file path, or None.
    """
    if not LANGEXTRACT_CACHE_DIR:
        return None
    text_digest = hashlib.sha256(prepared_text.encode("utf-8", "ignore")).hexdigest()
    key = hashlib.sha256("|".join([
        text_digest, model_id, str(MAX_CHARS), str(MAX_CHAR_BUFFER), str(NUM_CTX),
        _worker_fingerprint(),
    ]).encode("utf-8")).hexdigest()
    return os.path.join(LANGEXTRACT_CACHE_DIR, key[:2], f"{key}.json")


def langextract_python() -> Optional[List[str]]:
    """
    :returns: ``[".../envs/sample-langextract/bin/python"]``, or ``None`` when that environment has
        not been built -- which is the normal, expected signal to fall back to docling.

    Deliberately NOT ``peptide_predictors._predictor_command``. That helper falls back to
    ``shutil.which(executable)`` when the environment is missing, which is right for a TOOL like
    ``tesseract`` (any working binary will do) and wrong for an isolated INTERPRETER, where the
    whole point is that it must be that specific environment. Measured: with
    ``sample-langextract`` absent it returned ``envs/sample/bin/python``, so ``available()``
    reported True and every document would have paid for a subprocess that cannot import
    langextract before falling back.
    """
    for env_root in (os.environ.get("CONDA_ENVS_PATH"),
                     os.path.join(os.path.expanduser("~"), "miniconda3", "envs"),
                     os.path.join(os.path.expanduser("~"), "anaconda3", "envs")):
        if not env_root:
            continue
        candidate = os.path.join(env_root, LANGEXTRACT_ENV, "bin", "python")
        if os.path.exists(candidate):
            return [candidate]
    return None


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """
    :returns: Whether this backend can actually run -- the environment exists AND langextract
        imports inside it.

    The import check is not redundant with the interpreter check. A HALF-BUILT environment is a
    realistic state, and this project produced one: ``python -m venv`` succeeded, then
    ``pip install langextract`` failed partway on a slow link, leaving ``bin/python`` present with
    no langextract behind it. The path check alone reported the backend available, which would
    spend a doomed subprocess on every document before falling back. Checking what actually
    matters -- can this interpreter import the library -- costs one subprocess per process,
    cached, and makes a partial install indistinguishable from an absent one, which is what the
    caller needs.
    """
    interpreter = langextract_python()
    if not interpreter:
        return False
    try:
        completed = subprocess.run(interpreter + ["-c", "import langextract"],
                                   capture_output=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.debug(f"{LANGEXTRACT_ENV} import probe failed: {e}")
        return False
    if completed.returncode != 0:
        logger.warning(
            f"{LANGEXTRACT_ENV} exists but langextract does not import there "
            f"({(completed.stderr or b'').decode('utf-8', 'replace').strip().splitlines()[-1:] or ['']}"
            f"); treating the backend as unavailable and falling back to docling.")
        return False
    return True


def _activity(attributes: Dict[str, Any], evidence: str) -> Optional[Dict[str, Any]]:
    """
    One activity record from a langextract ``activity`` extraction.

    :param attributes: The extraction's attributes.
    :param evidence: Its ``extraction_text``, kept as ``value_raw``'s supporting quote.
    :returns: A record shaped like the docling backend's activities, or ``None`` when it carries
        no unit -- an activity with no unit is dropped for the same reason the docling backend
        drops one: ``strict_schema`` has no valid-unit entry for ``""``, so it would be silently
        discarded downstream anyway, and dropping it here makes the loss visible.
    """
    unit = str(attributes.get("activity_unit") or "").strip()
    if not unit:
        return None
    value_raw = str(attributes.get("value_raw") or "").strip()
    if not value_raw:
        return None
    # No magnitude conversion happens on this path, so the raw unit IS the canonical one. Stating
    # both keeps preprocessing.raw_value_and_unit from pairing a converted value with an
    # unconverted unit.
    return {
        "activity_type": str(attributes.get("activity_type") or "").strip() or None,
        "value_raw": value_raw,
        "value_lower": None,
        "value_upper": None,
        "activity_unit": unit,
        "value_raw_unit": unit,
        "target_organism": str(attributes.get("target_organism") or "").strip() or None,
        "target_cell_line": str(attributes.get("target_cell_line") or "").strip() or None,
        "context_sentence": evidence,
    }


def _document_text(text_path: str) -> str:
    """
    :param text_path: The document's text file.
    :returns: Its contents, or "" when unreadable. Used for the context windows attached to each
        record, on both the cached and uncached paths so a cache hit produces identical records.
    """
    try:
        with open(text_path, encoding="utf-8", errors="ignore") as handle:
            return handle.read()
    except OSError:
        return ""


def _is_patent(text_path: str) -> bool:
    """
    :param text_path: The document's text file.
    :returns: True when this document is a patent, whose reference layout makes
        :data:`TRUNCATE_REFERENCES` unsafe. See :data:`_PATENT_PATH_MARKER`.
    """
    return _PATENT_PATH_MARKER in os.path.normpath(text_path).replace(os.sep, "/")


def model_input_text(text_path: str) -> str:
    """
    The text actually sent to the model: bounded by :data:`MAX_CHARS`, with a paper's bibliography
    removed when that is safe.

    Deliberately NOT the same string as :func:`_document_text`. That one stays whole, because the
    context window attached to every record is resolved against it and a record whose context does
    not contain its own sequence is deleted downstream -- truncating the text used for context
    windows would discard records the model got right.

    :param text_path: The document's text file.
    :returns: The text to hand the worker. Falls back to the full (bounded) text whenever
        truncation is disabled, unavailable, inapplicable, or leaves less than
        :data:`MIN_KEPT_FRACTION` behind.
    """
    body = _document_text(text_path)[:MAX_CHARS]
    if not TRUNCATE_REFERENCES or not body.strip() or _is_patent(text_path):
        return body
    try:
        from sample.nlp import truncate_references
    except Exception as e:  # noqa: BLE001 -- a heavy optional import; never fail the document
        logger.debug(f"reference truncation unavailable ({type(e).__name__}: {e}); sending full text.")
        return body

    try:
        kept = truncate_references(body)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"reference truncation failed on {text_path} ({type(e).__name__}: {e}).")
        return body

    if len(kept) < MIN_KEPT_FRACTION * len(body):
        logger.debug("reference truncation kept only %.0f%% of %s; sending full text instead.",
                     len(kept) / max(1, len(body)) * 100, text_path)
        return body
    return kept


def _records_from_payload(payload: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """
    Assembles peptide records from the worker's payload.

    Activities are attached to peptides by the ``peptide_name`` the model reported on each activity.
    An activity naming a peptide that was never extracted gets its own name-only record rather than
    being dropped: the measurement is real evidence, and discarding it because the peptide
    extraction missed would lose exactly the recall this backend exists to add.

    :param payload: The worker's JSON output.
    :param text: The document text, for context windows.
    :returns: Records in the docling backend's shape.
    """
    peptides: Dict[str, Dict[str, Any]] = {}
    activities: List[Dict[str, Any]] = []

    for item in payload.get("extractions") or []:
        kind = str(item.get("extraction_class") or "").lower()
        attributes = item.get("attributes") or {}
        surface = str(item.get("extraction_text") or "").strip()
        if kind == "peptide":
            name = surface or str(attributes.get("peptide_name") or "").strip()
            if not name:
                continue
            sequence = str(attributes.get("peptide_sequence") or "").strip().upper()
            peptides.setdefault(name.lower(), {
                "peptide_name": name,
                "peptide_sequence": sequence or None,
                "biological_activities": [],
                # The grounded span, carried through so a reader can locate the claim.
                "source_char_start": item.get("start_char"),
                "source_char_end": item.get("end_char"),
            })
        elif kind == "activity":
            built = _activity(attributes, surface)
            if built:
                activities.append((str(attributes.get("peptide_name") or "").strip().lower(), built))

    for owner, activity in activities:
        record = peptides.get(owner)
        if record is None:
            if not owner:
                continue
            record = peptides.setdefault(owner, {
                "peptide_name": owner,
                "peptide_sequence": None,
                "biological_activities": [],
                "source_char_start": None,
                "source_char_end": None,
            })
        record["biological_activities"].append(activity)

    from sample.nlp import extract_context_window

    out = []
    for record in peptides.values():
        term = record.get("peptide_sequence") or record.get("peptide_name") or ""
        record["context"] = extract_context_window(text, target_term=term, token_window=1000) if term else ""
        record["source_page"] = None
        out.append(record)
    return out


def extract_records_from_text(text_path: str, model_id: Optional[str] = None,
                              timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """
    Peptide activity records for one document, via langextract against local Ollama.

    :param text_path: The document's ``.txt``.
    :param model_id: Ollama model tag; :data:`sample.config.DEFAULT_EXTRACT_MODEL` when omitted, so
        this backend uses the same local model as the rest of the pipeline rather than introducing
        a second convention.
    :param timeout: Seconds before the subprocess is abandoned.
    :returns: Records in the docling backend's shape. **Empty on every failure** -- a missing
        environment, an unreachable Ollama, a crash or a timeout all degrade to "no records from
        this backend" so the caller falls back to docling, and each is logged with its reason
        rather than passing silently.
    """
    if not os.path.exists(text_path):
        logger.error(f"Text not found: {text_path}")
        return []

    # PER-DOCUMENT BUDGET (sample.deadlines). This is the one timeout in the pipeline that is
    # genuinely hard: a subprocess can be killed, unlike a thread, so whatever the budget has
    # left is enforced exactly. Clamped rather than merely checked, because this call is
    # usually the largest single item in a document's budget -- DEFAULT_TIMEOUT alone is 1200s,
    # four times a 300s document budget, so an unclamped call would overrun the budget by
    # itself and make every other limit cosmetic.
    #
    # The cache is consulted BEFORE the budget is, further down: a cache hit costs a file read,
    # so refusing it would throw away free work for the document that most needs a cheap path.
    from sample import deadlines

    # available(), not a bare path check: a half-built environment has bin/python but no
    # langextract, and spawning a worker there wastes a subprocess per document to learn that.
    if not available():
        logger.debug(f"{LANGEXTRACT_ENV} is unavailable; skipping the langextract backend.")
        return []
    interpreter = langextract_python()

    from sample.config import DEFAULT_EXTRACT_MODEL

    resolved_model = model_id or os.getenv(MODEL_ENV_VAR) or DEFAULT_EXTRACT_MODEL

    # The cache holds the WORKER PAYLOAD, not the finished records: the model call is the expensive
    # part, while turning a payload into records is pure and cheap, so improvements to record
    # assembly take effect on cached documents without re-running anything.
    # What the model is sent, computed once: it is both the cache key and the worker's input, and
    # the two must never disagree about which text produced a payload.
    prepared_text = model_input_text(text_path)
    if not prepared_text.strip():
        logger.debug(f"nothing to send for {text_path}; skipping the langextract backend.")
        return []

    cache_path = _cache_path(prepared_text, resolved_model)
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as handle:
                cached = json.load(handle)
            logger.debug(f"langextract cache hit for {text_path}")
            return _records_from_payload(cached, _document_text(text_path))
        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"ignoring unreadable langextract cache entry {cache_path}: {e}")

    # Past this point every path costs a model call, so the budget decides whether to start.
    if deadlines.expired():
        logger.info(f"document budget exhausted ({deadlines.elapsed():.0f}s of "
                    f"{deadlines.budget():.0f}s); skipping the langextract pass for {text_path}")
        return []
    remaining = deadlines.remaining()
    if remaining is not None and remaining < timeout:
        timeout = int(remaining)
    # The worker's own per-request timeout cannot usefully exceed the time the worker has left
    # to live: leaving it higher means a request still running when the subprocess is killed,
    # which Ollama finishes anyway and charges to the next document's slot.
    request_timeout = min(REQUEST_TIMEOUT, timeout)

    worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_langextract_worker.py")
    payload_dir = tempfile.mkdtemp(prefix="langextract-")
    payload_path = os.path.join(payload_dir, "records.json")

    # The worker reads a PATH, and it runs in an environment that cannot import sample.* (numpy 2.x
    # against this project's numpy<2 pin), so the truncation cannot be done inside it. When the
    # prepared text differs from the file, it is written beside the payload and that path is passed
    # instead -- the original is never modified.
    worker_input = text_path
    if prepared_text != _document_text(text_path)[:MAX_CHARS]:
        worker_input = os.path.join(payload_dir, "input.txt")
        try:
            with open(worker_input, "w", encoding="utf-8") as handle:
                handle.write(prepared_text)
        except OSError as e:
            logger.debug(f"could not stage truncated text for {text_path} ({e}); sending the file.")
            worker_input = text_path

    command = interpreter + [
        worker, worker_input,
        "--out", payload_path,
        "--model-id", resolved_model,
        "--model-url", OLLAMA_URL,
        "--max-chars", str(MAX_CHARS),
        "--max-char-buffer", str(MAX_CHAR_BUFFER),
        "--num-ctx", str(NUM_CTX),
        "--max-workers", str(MAX_WORKERS),
        "--request-timeout", str(request_timeout),
        "--keep-alive", str(KEEP_ALIVE),
        "--think" if _think_default() else "--no-think",
    ]

    # RETRIED, because the failure this guards against is transient and expensive.
    #
    # Measured 2026-10-16 over a 12-minute window: 7 of 43 /api/generate calls -- 16.3% -- came
    # back HTTP 500, against zero failures on the 172 /api/chat calls from sample.llm in the same
    # window. The Ollama journal shows why, and it is not the document: the server is thrashing
    # its prompt cache (65 "making room for prompt cache entry, removing oldest entry" events of
    # 220-440 MiB each), and a request that cannot get its allocation 500s after ~6-22s. So the
    # same document succeeds or fails depending on what else is in flight, which is exactly the
    # shape that a retry fixes and that a per-document fallback does not.
    #
    # Without this, each 500 silently costs that document its entire model pass -- and at 16% of
    # the documents the gate lets through, that is the largest single source of data loss left in
    # the step.
    payload = None
    last_error = ""
    for attempt in range(1 + RETRIES):
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                                       check=False)
            if completed.returncode != 0:
                last_error = (f"worker exited {completed.returncode}: "
                              f"{(completed.stderr or '').strip()[:300]}")
                payload = None
            else:
                with open(payload_path, encoding="utf-8") as handle:
                    payload = json.load(handle)
                last_error = str(payload.get("error") or "")
                if not last_error:
                    break
        except subprocess.TimeoutExpired:
            # Not retried: a timeout means the work did not fit the budget, and a second attempt
            # under a now-smaller remainder cannot either. Retrying it would spend the rest of
            # the document's budget to reach the same answer.
            logger.warning(f"langextract timed out after {timeout}s on {text_path}; falling back.")
            _cleanup_payload_dir(payload_dir, payload_path)
            return []
        except (OSError, json.JSONDecodeError) as e:
            last_error = f"no readable payload: {type(e).__name__}: {e}"
            payload = None

        if attempt >= RETRIES or not _is_retryable(last_error):
            break
        # The budget decides whether there is room to try again; a retry that cannot finish is
        # worse than none, because it occupies a slot the next document needs in order to
        # produce a result nobody will read.
        #
        # Compared against an ABSOLUTE floor, not against `timeout`. `timeout` was itself clamped
        # to the remaining budget further up, so "remaining < timeout/2" is circular -- remaining
        # is always about equal to timeout, so the guard could never fire in the one case it
        # exists for. Caught by a test that asserted no retry on a 7-second budget and got one.
        remaining_now = deadlines.remaining()
        if deadlines.expired() or (remaining_now is not None
                                   and remaining_now < MIN_RETRY_SECONDS):
            logger.info(f"langextract failed on {text_path} ({last_error[:120]}); no budget left "
                        f"to retry.")
            break
        delay = RETRY_BACKOFF * (2 ** attempt) + random.uniform(0, 0.5)
        logger.info(f"langextract attempt {attempt + 1}/{1 + RETRIES} failed on {text_path} "
                    f"({last_error[:120]}); retrying in {delay:.1f}s.")
        time.sleep(delay)

    _cleanup_payload_dir(payload_dir, payload_path)

    if payload is None or last_error:
        logger.warning(f"langextract reported: {last_error or 'no payload'} ({text_path})")
        return []

    # Cached only on a clean run. A payload produced by a timeout or a half-answered document would
    # otherwise be frozen in as that document's permanent result.
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            temporary = f"{cache_path}.partial"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(temporary, cache_path)
        except OSError as e:
            logger.debug(f"could not cache langextract payload for {text_path}: {e}")

    return _records_from_payload(payload, _document_text(text_path))
