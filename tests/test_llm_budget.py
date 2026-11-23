"""
Tests for how :func:`sample.llm.llm_call` behaves under a per-document budget, and for the
timeout that used not to fire.

A real Ollama is never contacted: ``llm_call`` does ``import ollama`` inside the function, so a
stub in ``sys.modules`` is enough, and that is also what lets these tests assert on the timeout
the client was constructed with.
"""
import sys
import time
import types

import pytest

from sample import deadlines
import sample.llm as L


class _FakeChat:
    def __init__(self, delay=0.0, record=None):
        self.delay = delay
        self.record = record if record is not None else []

    def chat(self, **kwargs):
        self.record.append(kwargs)
        if self.delay:
            time.sleep(self.delay)
        return {"message": {"content": "{}"}}


@pytest.fixture
def fake_ollama(monkeypatch):
    """Installs a stub `ollama` module and yields it, with the client timeouts it was given."""
    state = {"timeouts": [], "calls": []}
    chat = _FakeChat(record=state["calls"])

    module = types.ModuleType("ollama")

    def _client(timeout=None, **kwargs):
        state["timeouts"].append(timeout)
        return chat

    module.Client = _client
    module.chat = chat.chat
    monkeypatch.setitem(sys.modules, "ollama", module)
    state["chat"] = chat
    yield state


@pytest.fixture(autouse=True)
def _no_budget_leaks():
    """A budget left behind by a failing test would silently disable LLM calls in later ones."""
    yield
    deadlines.clear()


class TestABudgetStopsNewCalls:
    def test_an_exhausted_budget_refuses_before_the_request_is_made(self, fake_ollama):
        """
        The refusal is the point, not the error. A call issued with seconds left cannot return
        in time, but Ollama runs the generation to completion after the client gives up -- so
        it occupies a slot the NEXT document needs. Starting it would make an over-budget
        document steal throughput from documents still within theirs.
        """
        with deadlines.Budget(0.001):
            time.sleep(0.01)
            with pytest.raises(deadlines.BudgetExhausted):
                L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=90)
        assert fake_ollama["calls"] == [], "a refused call still reached the server"

    def test_the_remaining_budget_caps_the_call_timeout(self, fake_ollama):
        """
        Clamped in llm_call rather than at each of the six LLM layers, which is what makes the
        budget TOTAL instead of per-call: six helpers each honouring a 90s timeout is a 540s
        document, so a nominal 300s cap would bound nothing.
        """
        with deadlines.Budget(20):
            L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=90)
        assert fake_ollama["timeouts"], "no client was constructed"
        assert fake_ollama["timeouts"][-1] == pytest.approx(20, abs=1.5)

    def test_without_a_budget_the_callers_timeout_is_untouched(self, fake_ollama):
        deadlines.clear()
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=90)
        assert fake_ollama["timeouts"][-1] == 90


class TestTheTimeoutActuallyReturns:
    def test_a_timed_out_call_does_not_wait_for_the_call_it_abandoned(self, fake_ollama):
        """
        Regression test. The timeout used to be implemented inside a `with
        ThreadPoolExecutor(...)` block, and __exit__ calls shutdown(wait=True) -- so
        future.result(timeout=...) raised TimeoutError and then the context manager blocked
        until the very call that had timed out finished. The deadline was reported at the right
        moment and enforced at none, which is why a per-document cap built on these timeouts
        appeared to do nothing at all.
        """
        fake_ollama["chat"].delay = 2.0
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=0.2)
        elapsed = time.monotonic() - started
        assert elapsed < 1.0, (f"llm_call took {elapsed:.1f}s to honour a 0.2s timeout; it is "
                               f"still waiting for the abandoned call")


class TestBudgetRefusalsAreNotAnOutage:
    def test_the_circuit_breaker_ignores_them(self):
        """
        The breakers open after three CONSECUTIVE failures and stay open for 30s, globally. One
        over-budget document refuses several calls in a row, so counting them would disable a
        verification layer for every other document in flight -- the exact cross-document
        contamination the breaker was written to prevent.
        """
        breaker = L._CircuitBreaker("test layer", threshold=3, cooldown=60)
        for _ in range(10):
            breaker.record_failure(deadlines.BudgetExhausted("out of budget"))
        assert breaker.is_open() is False

    def test_a_real_failure_still_opens_it(self):
        breaker = L._CircuitBreaker("test layer", threshold=3, cooldown=60)
        for _ in range(3):
            breaker.record_failure(RuntimeError("connection refused"))
        assert breaker.is_open() is True

class TestPerLayerAccounting:
    """
    Which layer is spending the GPU, recorded rather than inferred.

    Measured 2026-10-16: LLM calls/minute stayed flat at 24-30 while documents/minute fell
    threefold, so cost per document rose rather than the server slowing down. That points the
    investigation at the per-layer mix -- and nothing was recording it, so the question was being
    answered by reasoning, which has been wrong three times in this project.
    """

    @pytest.fixture(autouse=True)
    def _clear(self):
        L._CALL_STATS.clear()
        yield
        L._CALL_STATS.clear()

    def test_a_successful_call_is_attributed_to_its_layer(self, fake_ollama):
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=10,
                   layer="triage")
        stats = L.call_stats()
        assert stats["triage"][0] == 1
        assert stats["triage"][1] >= 0
        assert stats["triage"][2] == 0

    def test_a_failed_call_is_recorded_as_failed_not_dropped(self, fake_ollama):
        """A layer whose calls mostly fail would otherwise look cheap in the summary."""
        with deadlines.Budget(0.001):
            time.sleep(0.01)
            with pytest.raises(deadlines.BudgetExhausted):
                L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=10,
                           layer="activity_verify")
        stats = L.call_stats()
        assert stats["activity_verify"][0] == 1
        assert stats["activity_verify"][2] == 1

    def test_an_unlabelled_caller_is_still_counted(self, fake_ollama):
        """
        A layer missing from the summary would make the mix look cleaner than it is, so an
        un-updated call site is accounted rather than ignored.
        """
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=10)
        assert L.call_stats()["unlabelled"][0] == 1

    def test_every_call_site_in_the_pipeline_names_its_layer(self):
        """
        Guards the instrumentation against erosion: a new LLM layer added without a label lands
        in "unlabelled" and the summary stops being usable for deciding what to optimise.
        """
        import re
        from pathlib import Path

        unlabelled = []
        for path in (Path("sample/llm.py"), Path("sample/llm_validator.py")):
            text = path.read_text(encoding="utf-8")
            # `def llm_call(` is the definition, not a call site.
            for match in re.finditer(r"(?<!def )(?<![_\w])llm_call\(", text):
                # the enclosing call's text, up to a balanced close paren
                depth, index = 0, match.end() - 1
                while index < len(text):
                    if text[index] == "(":
                        depth += 1
                    elif text[index] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    index += 1
                call = text[match.start():index]
                if "layer=" not in call:
                    line = text[:match.start()].count("\n") + 1
                    unlabelled.append(f"{path}:{line}")
        assert not unlabelled, f"llm_call sites with no layer= label: {unlabelled}"

class TestThinkingMode:
    """
    Whether a reasoning family may emit a chain of thought, and the budget that makes it usable.

    The measured reason this was off: these models write the trace to ``message.thinking`` and the
    answer to ``message.content``, and ``num_predict`` caps the two TOGETHER. On one verification
    prompt, thinking enabled returned ``content=""`` in 7.1s with ``done_reason="length"`` against
    the correct ``{"supported": false}`` in 0.3s with ``think=False``. That is what the "gemma4
    returns an empty completion" symptom was.
    """

    def test_thinking_reaches_the_request_for_a_reasoning_family(self, fake_ollama, monkeypatch):
        monkeypatch.setattr(L, "THINK_ENABLED", True)
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma4:latest", timeout=10,
                   max_tokens=384, layer="triage")
        assert fake_ollama["calls"][-1]["think"] is True

    def test_it_can_be_switched_off(self, fake_ollama, monkeypatch):
        monkeypatch.setattr(L, "THINK_ENABLED", False)
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma4:latest", timeout=10,
                   max_tokens=384, layer="triage")
        assert fake_ollama["calls"][-1]["think"] is False

    def test_a_non_reasoning_model_is_never_sent_the_parameter(self, fake_ollama, monkeypatch):
        """
        Sending `think` to a model without thinking support does not raise, it HANGS (verified
        with qwen2.5-coder:1.5b, which never returned), so it cannot be guarded with try/except.
        """
        monkeypatch.setattr(L, "THINK_ENABLED", True)
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma3:4b", timeout=10,
                   max_tokens=384, layer="triage")
        assert "think" not in fake_ollama["calls"][-1]

    def test_enabling_thinking_enlarges_the_completion_budget(self, fake_ollama, monkeypatch):
        """
        Without this, turning thinking on just reproduces the empty-completion bug: the trace
        spends a 384-token budget and leaves nothing for the answer.
        """
        monkeypatch.setattr(L, "THINK_ENABLED", True)
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma4:latest", timeout=10,
                   max_tokens=384, layer="triage")
        budget = fake_ollama["calls"][-1]["options"]["num_predict"]
        assert budget >= L.THINK_MIN_TOKENS
        assert budget > 384

    def test_the_budget_is_untouched_with_thinking_off(self, fake_ollama, monkeypatch):
        monkeypatch.setattr(L, "THINK_ENABLED", False)
        L.llm_call([{"role": "user", "content": "hi"}], model="gemma4:latest", timeout=10,
                   max_tokens=384, layer="triage")
        assert fake_ollama["calls"][-1]["options"]["num_predict"] == 384

    def test_the_langextract_path_follows_the_same_setting(self, monkeypatch):
        """
        A split would mean the same model reasoning on one of the seven Ollama consumers and not
        the others, making any quality change unattributable.
        """
        import sample.langextract_records as LR

        monkeypatch.delenv("LANGEXTRACT_THINK", raising=False)
        monkeypatch.setattr(L, "THINK_ENABLED", True)
        assert LR._think_default() is True
        monkeypatch.setattr(L, "THINK_ENABLED", False)
        assert LR._think_default() is False

    def test_langextract_can_override_it_explicitly(self, monkeypatch):
        import sample.langextract_records as LR

        monkeypatch.setattr(L, "THINK_ENABLED", False)
        monkeypatch.setenv("LANGEXTRACT_THINK", "1")
        assert LR._think_default() is True

