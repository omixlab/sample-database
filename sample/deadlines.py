"""
A wall-clock budget for one unit of work, carried on the thread that is doing it.

Why thread-local instead of a parameter
---------------------------------------
A single document's processing spans the extractor, the NLP passes, several independent LLM
helpers (candidate triage, activity verification, organism normalisation, activity recovery,
extraction validation, concentration-unit adjudication) and a langextract subprocess -- call
sites across six modules, most of which neither know nor should know that a budget exists.
Threading a deadline argument through all of them would be a wide change whose correctness
depends on no caller ever forgetting to pass it, and a forgotten one fails silently by
reverting to unbounded. The extraction pool hands one document to one thread for that
document's whole lifetime, so the thread is the natural place to hang the deadline, and a
helper that takes no deadline argument still observes it.

What a budget does, and what it deliberately does not do
--------------------------------------------------------
It bounds the operations that actually consume the time: every LLM round trip is clamped to
what is left, and the langextract subprocess is killed at the deadline (a subprocess *can* be
killed, so for that one the cap is hard).

It does NOT abort the thread. CPython offers no safe way to interrupt a running thread, and
the obvious alternative -- abandoning the future and moving on -- is worse than useless here:
the abandoned thread keeps running, keeps its share of the shared lookup structures hot, and
keeps holding an Ollama slot, so "timing out" would quietly accumulate exactly the backlog a
timeout is supposed to prevent. Instead, a document that runs over budget stops being allowed
to START new expensive work. It finishes with whatever it has, writes its output, and is
recorded as having been cut short -- a degraded document that is auditable, not a lost one.

So the budget is a cap on *new* expensive work, and the overshoot is bounded by the longest
single operation already in flight rather than by the budget itself.
"""

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

#: Seconds below which a fresh LLM round trip is not worth starting. A call issued with 2
#: seconds left cannot finish, but it still occupies a server slot for its full duration (the
#: client giving up does not cancel the generation Ollama has already begun), so it costs the
#: NEXT document's throughput to produce a guaranteed failure for this one. Treating that
#: remainder as exhausted is what makes the budget a throughput protection rather than just an
#: error-reporting change.
MIN_USEFUL_SLICE = float(os.getenv("SAMPLE_MIN_USEFUL_SLICE", "5"))

_state = threading.local()


class BudgetExhausted(TimeoutError):
    """
    Raised when an operation is refused because the budget is spent, as distinct from an
    operation that was attempted and timed out.

    A subclass of :class:`TimeoutError` so existing ``except TimeoutError`` and ``except
    Exception`` handlers keep working unchanged -- every LLM layer already degrades gracefully
    on a failed call, and this must not become a new failure mode for them to learn about.

    It needs to be distinguishable, though, and for one specific reason: the circuit breakers
    in :mod:`sample.llm` open after three CONSECUTIVE failures. A single over-budget document
    refuses many calls in a row, so counting those refusals would trip the breaker and disable
    that whole verification layer -- for thirty seconds, for every other document being
    processed at the time, which is precisely the cross-document contamination the breaker was
    written to stop. A budget refusal says nothing about the server's health, so it is not a
    failure the breaker should count.
    """


def start(budget: Optional[float]) -> None:
    """
    Opens a budget on this thread.

    :param budget: Seconds of wall clock this unit of work may spend. ``None`` or a
        non-positive value clears the budget instead of setting one, so "unlimited" is
        expressed by the same call rather than by skipping it.
    :returns: ``None``.
    """
    if budget is None or budget <= 0:
        clear()
        return
    _state.deadline = time.monotonic() + float(budget)
    _state.budget = float(budget)
    _state.started = time.monotonic()


def clear() -> None:
    """
    Closes any budget on this thread.

    Pool threads are reused, so a budget left behind by one document would silently apply to
    the next one -- and, being already expired, would make that document skip every LLM call.
    Always called from a ``finally``; :class:`Budget` exists so that is not left to memory.

    :returns: ``None``.
    """
    _state.deadline = None
    _state.budget = None
    _state.started = None


def active() -> bool:
    """:returns: ``True`` when this thread is working under a budget."""
    return getattr(_state, "deadline", None) is not None


def remaining() -> Optional[float]:
    """
    :returns: Seconds left, which may be negative once the deadline has passed, or ``None``
        when no budget is open. ``None`` and ``0.0`` mean opposite things, which is why this
        does not collapse them: ``None`` is "spend what you need", ``0.0`` is "stop".
    """
    deadline = getattr(_state, "deadline", None)
    if deadline is None:
        return None
    return deadline - time.monotonic()


def elapsed() -> Optional[float]:
    """:returns: Seconds spent under the open budget, or ``None`` when none is open."""
    started = getattr(_state, "started", None)
    if started is None:
        return None
    return time.monotonic() - started


def budget() -> Optional[float]:
    """:returns: The budget this thread was given, or ``None``."""
    return getattr(_state, "budget", None)


def expired() -> bool:
    """
    :returns: ``True`` when no further expensive work should be started -- which includes the
        case where a little time is left but less than :data:`MIN_USEFUL_SLICE`. Always
        ``False`` when no budget is open, so an unbudgeted caller is never blocked.
    """
    left = remaining()
    return left is not None and left <= MIN_USEFUL_SLICE


def clamp(timeout: Optional[float]) -> Optional[float]:
    """
    Narrows one operation's timeout to what the budget still allows.

    :param timeout: The timeout the caller would have used; ``None`` for "no timeout".
    :returns: The smaller of the two, or ``timeout`` unchanged when no budget is open. Never
        returns a non-positive number: callers are expected to test :func:`expired` first, and
        a zero timeout reaching an HTTP client means something different (and worse) than a
        small one.
    """
    left = remaining()
    if left is None:
        return timeout
    left = max(left, 0.1)
    if timeout is None or timeout <= 0:
        return left
    return min(float(timeout), left)


class Budget:
    """
    Context manager form, so a budget cannot outlive its block on a reused pool thread.

    >>> with Budget(300):             # doctest: +SKIP
    ...     process_one_document()

    Nests by shortening only: an inner block never extends an outer deadline, because the
    outer one is a promise to the caller about total time. Exiting an inner block restores the
    outer deadline rather than clearing it.
    """

    def __init__(self, seconds: Optional[float]) -> None:
        """
        :param seconds: Budget for the block, or ``None`` for no budget.
        :returns: ``None``.
        """
        self._seconds = seconds
        self._saved: Optional[float] = None
        self._saved_budget: Optional[float] = None
        self._saved_started: Optional[float] = None

    def __enter__(self) -> "Budget":
        """:returns: This budget, so ``as`` has something useful to bind."""
        self._saved = getattr(_state, "deadline", None)
        self._saved_budget = getattr(_state, "budget", None)
        self._saved_started = getattr(_state, "started", None)
        if self._seconds is None or self._seconds <= 0:
            # An inner block with no budget of its own must not lift the outer one.
            if self._saved is None:
                clear()
            return self
        proposed = time.monotonic() + float(self._seconds)
        if self._saved is not None:
            proposed = min(proposed, self._saved)
        _state.deadline = proposed
        _state.budget = float(self._seconds)
        _state.started = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        """:returns: ``None`` -- exceptions propagate; a budget is not an error handler."""
        _state.deadline = self._saved
        _state.budget = self._saved_budget
        _state.started = self._saved_started
