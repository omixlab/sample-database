"""
Controls the local Ollama *runtime* -- what is loaded, and whether it can be put back to a
known-clean state -- as opposed to :mod:`sample.llm`, which only makes requests to it.

Why this exists
---------------
Extraction throughput on this host decays over a long run rather than failing. Measured on the
current corpus, the rate held at ~1 doc/s early and fell to ~0.04 docs/s after roughly 2,000
documents -- a 25x collapse with no error raised anywhere, which is the signature of state
accumulating inside the serving process rather than of a bug in the client. Everything the
client can account for (its own thread pool, the submission window, the per-document
subprocesses) is bounded and verifiably drains. The llama-server runner is the one component
that lives across the entire run, holds a KV cache per slot, and has an internal request queue
that nothing in this repository can inspect.

So the remedy is structural, not diagnostic: run the corpus in chunks, and between chunks
return the runtime to the state it was in at the start of the run. A fresh runner has empty
slots, an empty queue, a cold KV cache and no fragmentation, which removes the whole class of
cause at a cost that is measurable (one model load per chunk) instead of chasing a cause that
cannot be observed from here.

How the reset is done, and why not systemctl
--------------------------------------------
Unloading the model (``keep_alive: 0``) makes Ollama terminate the llama-server subprocess that
serves it. The next request starts a new one. That is a genuine fresh runner -- new process,
new slots, new queue -- and it needs no privileges, which matters because the pipeline is run
by an unprivileged user and a reset that needs a sudo password would simply not happen at 3am.
``systemctl restart ollama`` would also work and is available via
:data:`RESET_COMMAND_ENV` for anyone who wants it, but it is not the default: it drops the
whole API server (so a concurrent reader gets a connection error rather than a short wait),
and it requires the privilege this does not.
"""

import json
import logging
import os
import subprocess
import time
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

#: Same default and same variable as every other Ollama consumer in the project.
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")

#: An optional shell command run INSTEAD of the unload (e.g. ``sudo systemctl restart ollama``).
#: Left unset by default; see the module docstring for why.
RESET_COMMAND_ENV = "SAMPLE_OLLAMA_RESET_CMD"

#: How long to wait for the runner to actually exit after the unload is accepted. The HTTP call
#: returns as soon as the unload is scheduled, not when the process is gone, so a reset that
#: did not wait would hand the next chunk a runtime still tearing down the old runner -- which
#: is the condition this is supposed to end, re-created by the fix.
DRAIN_TIMEOUT = float(os.getenv("SAMPLE_OLLAMA_DRAIN_TIMEOUT", "60"))

#: Poll interval while waiting for the runner to exit.
_POLL_SECONDS = 1.0

#: Short, because every call here is to localhost and a slow answer means the server is wedged
#: -- in which case waiting longer does not help and the caller needs to be told.
_HTTP_TIMEOUT = float(os.getenv("SAMPLE_OLLAMA_HTTP_TIMEOUT", "15"))


def loaded_models(url: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """
    What Ollama currently has in memory, from ``GET /api/ps``.

    :param url: Base URL of the Ollama server; :data:`OLLAMA_URL` when omitted.
    :returns: One entry per loaded model, ``[]`` when nothing is loaded, or ``None`` when the
        server could not be reached or answered unintelligibly. ``None`` and ``[]`` are kept
        distinct deliberately: "nothing is loaded" is the goal state of a reset, while "I could
        not ask" must never be reported as success.
    """
    base = (url or OLLAMA_URL).rstrip("/")
    try:
        response = requests.get(f"{base}/api/ps", timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as e:
        logger.debug(f"could not read {base}/api/ps: {type(e).__name__}: {e}")
        return None
    models = payload.get("models")
    if not isinstance(models, list):
        return None
    return models


def unload(model: str, url: Optional[str] = None) -> bool:
    """
    Asks Ollama to drop one model from memory.

    :param model: The model tag to unload, as ``/api/ps`` reports it.
    :param url: Base URL of the Ollama server; :data:`OLLAMA_URL` when omitted.
    :returns: ``True`` when the server accepted the request. Note this is "accepted", not
        "unloaded" -- :func:`reset` confirms the latter by polling, because the two are
        seconds apart and only the second one means the runner is gone.

    Sent with an empty prompt: ``keep_alive: 0`` with no prompt is Ollama's documented way to
    release a model, and it generates no tokens, so it costs nothing and cannot pollute the
    model's context.
    """
    base = (url or OLLAMA_URL).rstrip("/")
    try:
        response = requests.post(f"{base}/api/generate",
                                 json={"model": model, "prompt": "", "keep_alive": 0},
                                 timeout=_HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"unload of {model} refused by {base}: {type(e).__name__}: {e}")
        return False
    return True


def _run_reset_command(command: str) -> Dict[str, Any]:
    """
    Runs the operator-supplied reset command.

    :param command: A shell command line.
    :returns: A summary with ``method``, ``ok`` and, on failure, ``error``.
    """
    try:
        completed = subprocess.run(command, shell=True, capture_output=True, text=True,
                                   timeout=DRAIN_TIMEOUT, check=False)
    except subprocess.TimeoutExpired:
        return {"method": "command", "command": command, "ok": False,
                "error": f"timed out after {DRAIN_TIMEOUT:.0f}s"}
    if completed.returncode != 0:
        return {"method": "command", "command": command, "ok": False,
                "error": f"exit {completed.returncode}: {(completed.stderr or '').strip()[:300]}"}
    return {"method": "command", "command": command, "ok": True}


def reset(url: Optional[str] = None, drain_timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Never-raising wrapper around :func:`_reset`.

    :param url: Base URL of the Ollama server; :data:`OLLAMA_URL` when omitted.
    :param drain_timeout: Seconds to wait for the runner to exit.
    :returns: :func:`_reset`'s summary, or one carrying the unexpected error.

    The guarantee is worth the extra frame: this is called from inside a multi-day run, between
    chunks, and the chunk that just finished is already on disk. Letting an unexpected error --
    a library exception this does not anticipate, a transport failure with a surprising type --
    propagate would abandon a corpus mid-extraction to avoid a slowdown.
    """
    try:
        return _reset(url=url, drain_timeout=drain_timeout)
    except Exception as e:                                     # noqa: BLE001 -- reported, never fatal
        logger.warning(f"Ollama reset failed unexpectedly ({type(e).__name__}: {e}); continuing.")
        return {"method": "unload", "unloaded": [], "drained": False, "seconds": 0.0,
                "error": f"{type(e).__name__}: {e}"}


def _reset(url: Optional[str] = None, drain_timeout: Optional[float] = None) -> Dict[str, Any]:
    """
    Returns the Ollama runtime to the state it was in before the run started.

    :param url: Base URL of the Ollama server; :data:`OLLAMA_URL` when omitted.
    :param drain_timeout: Seconds to wait for the runner to exit; :data:`DRAIN_TIMEOUT` when
        omitted.
    :returns: A summary dict -- ``method``, ``unloaded`` (the model tags acted on), ``drained``
        (whether ``/api/ps`` went empty), ``seconds``, and ``error`` when something went wrong.
        Always returns; a reset that fails must not end the run, because the chunk that just
        finished is on disk and the next one will merely be as slow as the old behaviour.

    Wrapped by :func:`reset`, which is what callers use and which never raises. A failed
    reset is reported and the caller continues -- degrading to the unchunked behaviour is
    strictly better than discarding a corpus mid-run.
    """
    base = (url or OLLAMA_URL).rstrip("/")
    limit = DRAIN_TIMEOUT if drain_timeout is None else drain_timeout
    started = time.monotonic()

    command = os.getenv(RESET_COMMAND_ENV, "").strip()
    if command:
        summary = _run_reset_command(command)
        summary["seconds"] = round(time.monotonic() - started, 1)
        # Still confirmed by polling: a restart command that returns 0 has only told us the
        # command ran.
        summary["drained"] = _wait_until_empty(base, limit - (time.monotonic() - started))
        return summary

    models = loaded_models(base)
    if models is None:
        return {"method": "unload", "unloaded": [], "drained": False,
                "seconds": round(time.monotonic() - started, 1),
                "error": f"could not reach {base}/api/ps"}

    tags = [str(entry.get("name") or entry.get("model") or "").strip()
            for entry in models]
    tags = [tag for tag in tags if tag]
    if not tags:
        # Already clean. Reported as a successful reset rather than as a no-op, because from
        # the caller's point of view the postcondition it asked for holds.
        return {"method": "unload", "unloaded": [], "drained": True,
                "seconds": round(time.monotonic() - started, 1)}

    unloaded = [tag for tag in tags if unload(tag, base)]
    drained = _wait_until_empty(base, limit - (time.monotonic() - started))
    summary: Dict[str, Any] = {"method": "unload", "unloaded": unloaded, "drained": drained,
                               "seconds": round(time.monotonic() - started, 1)}
    if len(unloaded) != len(tags):
        summary["error"] = (f"{len(tags) - len(unloaded)} of {len(tags)} unload requests were "
                            f"refused")
    elif not drained:
        summary["error"] = (f"still loaded after {limit:.0f}s: "
                            f"{json.dumps(loaded_models(base) or [], default=str)[:200]}")
    return summary


def _wait_until_empty(base: str, limit: float) -> bool:
    """
    Polls ``/api/ps`` until no model is loaded.

    :param base: Base URL of the Ollama server.
    :param limit: Seconds to keep waiting; a non-positive value still checks once, so a reset
        whose unload phase consumed the whole budget reports what is actually true rather than
        defaulting to "not drained".
    :returns: ``True`` once nothing is loaded, ``False`` if the limit passed first or the
        server stopped answering.
    """
    deadline = time.monotonic() + max(limit, 0.0)
    while True:
        models = loaded_models(base)
        if models is not None and not models:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(_POLL_SECONDS)
