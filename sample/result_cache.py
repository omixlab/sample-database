"""
Redis-backed result cache for expensive, on-demand analyses.

**Currently unused.** Its only two callers were the Expression Wizard's ``/express/fold`` and
``/express/mpepe-yield`` routes, which left this repository on 2026-11-23 when the wizard was
split out into its own tool (it now carries an on-disk cache of the same shape, because a
standalone tool should not require a Redis server). This module is kept, with its tests, because
it is working and general: the next analysis that costs GPU-minutes and is deterministic for its
input wants exactly this. Nothing imports it today, so it is dead weight until something does --
delete it rather than let it rot if that stops being true.

**What it is for.** An analysis that costs minutes per call and is deterministic for a given
input: re-running an identical request is pure waste, of compute and of the request-handler
memory that holds a multi-megabyte result while it works. The two original cases were ColabFold
on a GPU and MPEPE loading TensorFlow, Theano and a 10-model ensemble from disk (~4 minutes
measured) on every call.

**Why Redis rather than the process, or a table.** Keeping results in a module-level dict would
grow the webserver's resident memory without bound and lose everything on restart, and each Celery
worker would keep its own copy. Redis holds one copy outside the application processes, expires it
on a TTL without any sweeper of this project's own, and is already a dependency of this stack (the
Celery broker and result backend, `sample.tasks`).

**This is a cache, not a record of truth** -- a deliberate distinction this project already draws:
`sample.prediction_store` and `sample.annotation_store` keep their results in SQLite precisely
because Redis holds transient state. Nothing here is authoritative: every entry can vanish (an
eviction, a restart, a flush) and the only consequence is that the next identical request pays
full price again. So a cache miss and an unreachable Redis are the same non-event, and no caller
needs to care which happened.

**Degrades silently and completely.** A missing ``redis`` package, a down server, a serialisation
failure -- each leaves the caller running its own computation exactly as if the cache did not
exist. That is the same discipline the rest of this project applies to optional infrastructure
(ColabFold, RNAfold, every external predictor): an absent tool narrows what is reported, it never
breaks the path.
"""
import hashlib
import json
import logging
import os
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("sample.result_cache")

#: Where the cache lives. Defaults to database 2 -- `sample.tasks` already uses 0 for the Celery
#: broker and 1 for its result backend, so this shares the server without colliding with either.
CACHE_URL: str = os.environ.get("SAMPLE_CACHE_URL", "redis://localhost:6379/2")

#: How long a cached result answers a repeat request. Seven days, matching
#: :data:`sample.annotation_store.CACHE_TTL_SECONDS` -- the same reasoning applies: these results
#: depend on inputs (a model checkpoint, a codon table) that do not change on a shorter cycle, and
#: a week bounds how stale a served answer can be after one does.
CACHE_TTL_SECONDS: int = int(os.environ.get("SAMPLE_CACHE_TTL", 7 * 24 * 60 * 60))

#: Key prefix, so this database stays legible next to anything else that might share the server.
KEY_PREFIX: str = "sample:result:"

#: Set once the first connection attempt has been made. ``None`` means "not tried yet"; ``False``
#: means the last attempt failed and the cache is treated as absent. Probed lazily rather than at
#: import, so importing this module never blocks on a network round trip.
_client: Any = None
_probed: bool = False


def _connect() -> Any:
    """
    :returns: A connected Redis client, or ``None`` when Redis is unavailable for any reason
        (package not installed, server down, bad URL). Probed once; the failure is logged at INFO
        rather than raised, since running without a cache is a supported mode, not an error.
    """
    global _client, _probed
    if _probed:
        return _client
    _probed = True
    try:
        import redis  # noqa: PLC0415 -- optional dependency, imported only when first used

        client = redis.Redis.from_url(CACHE_URL, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        _client = client
        logger.info(f"Result cache connected to {CACHE_URL}.")
    except Exception as e:  # noqa: BLE001 -- any failure means "no cache", never a broken caller
        logger.info(f"Result cache unavailable ({e}); analyses will run uncached.")
        _client = None
    return _client


def reset() -> None:
    """
    Forgets the cached connection so the next call re-probes.

    Exists for the tests (which point :data:`CACHE_URL` at a fake) and for a process that wants to
    retry after Redis comes back, since the probe is otherwise once-per-process.
    """
    global _client, _probed
    _client = None
    _probed = False


def cache_key(kind: str, payload: Dict[str, Any]) -> str:
    """
    Builds the key an analysis result is stored under.

    The key is a hash of the FULL input, so two requests share an entry only when every option
    that could change the answer matches -- host, carrier, linker, secretion signal and the rest.
    Hashing (rather than concatenating) keeps the key bounded no matter how long the sequence is,
    and ``sort_keys`` makes it independent of dict ordering.

    :param kind: The analysis being cached, e.g. ``"fold"`` or ``"mpepe"``. Namespaces the key so
        two analyses of the same construct never collide.
    :param payload: Every input that affects the result.
    :returns: The Redis key.
    """
    blob = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}{kind}:{digest}"


def get(kind: str, payload: Dict[str, Any]) -> Optional[Any]:
    """
    :param kind: The analysis, as in :func:`cache_key`.
    :param payload: The inputs that identify the result.
    :returns: The cached value, or ``None`` on a miss OR when the cache is unavailable -- the
        caller treats both the same way, by computing the result itself.
    """
    client = _connect()
    if client is None:
        return None
    try:
        raw = client.get(cache_key(kind, payload))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Result cache read failed: {e}")
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception as e:  # noqa: BLE001 -- a corrupt entry is a miss, not a failure
        logger.debug(f"Ignoring unreadable cache entry: {e}")
        return None


def set(kind: str, payload: Dict[str, Any], value: Any,
        ttl_seconds: Optional[int] = None) -> bool:
    """
    Stores a result under its input's key, with an expiry.

    :param kind: The analysis, as in :func:`cache_key`.
    :param payload: The inputs that identify the result.
    :param value: The result; must be JSON-serialisable.
    :param ttl_seconds: Expiry; :data:`CACHE_TTL_SECONDS` when omitted.
    :returns: Whether it was stored. ``False`` means the cache was unavailable or the value could
        not be serialised -- neither of which the caller needs to act on.
    """
    client = _connect()
    if client is None:
        return False
    try:
        blob = json.dumps(value, default=str)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Result not cacheable ({e}); skipping.")
        return False
    try:
        client.setex(cache_key(kind, payload), ttl_seconds or CACHE_TTL_SECONDS, blob)
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Result cache write failed: {e}")
        return False


def cached(kind: str, payload: Dict[str, Any], compute: Callable[[], Any],
           ttl_seconds: Optional[int] = None) -> Any:
    """
    Returns the cached result for `payload`, computing and storing it on a miss.

    The whole point of the module in one call: a caller wraps its expensive work in this and stops
    thinking about the cache. A result is only stored when it is worth storing -- see
    :func:`_worth_caching`, which refuses to persist a failed run so a transient outage does not
    get served for a week.

    :param kind: The analysis, as in :func:`cache_key`.
    :param payload: Every input that affects the result.
    :param compute: Called on a miss to produce the result.
    :param ttl_seconds: Expiry; :data:`CACHE_TTL_SECONDS` when omitted.
    :returns: The result, cached or freshly computed.
    """
    hit = get(kind, payload)
    if hit is not None:
        logger.info(f"Result cache hit for {kind}.")
        return hit
    value = compute()
    if _worth_caching(value):
        set(kind, payload, value, ttl_seconds=ttl_seconds)
    return value


def _worth_caching(value: Any) -> bool:
    """
    Whether a result should be stored.

    A failed analysis must NOT be cached: ColabFold not being installed, MPEPE's environment
    missing, a run that crashed -- these are states of the machine, not properties of the input,
    and caching one would keep serving the failure for a week after the cause was fixed. Both
    analyses report this the same way (``available``/``folded`` flags plus a note), so the check
    reads those rather than guessing.

    :param value: A computed result.
    :returns: Whether to store it.
    """
    if value is None:
        return False
    if isinstance(value, dict):
        if value.get("available") is False:
            return False
        # fold_construct reports a reached-but-unsuccessful run this way (too long for the GPU,
        # ColabFold produced nothing); the input did not change, but the outcome may next time.
        if "folded" in value and not value.get("folded"):
            return False
        # mpepe_yield_estimate is "available" whenever the tool was found, so a failed run is
        # identified by there being no number in it.
        if "probability" in value and value.get("probability") is None:
            return False
    return True
