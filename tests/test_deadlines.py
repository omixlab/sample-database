"""
Tests for the per-document wall-clock budget.

Two of these exist because of what a budget is NOT. It cannot interrupt a running thread, so
its only lever is refusing to START work -- which makes "when does it refuse?" the whole
contract, and makes a budget that silently leaks onto the next document (pool threads are
reused) a bug that would quietly disable the LLM pass for a whole corpus.
"""
import time

import pytest

from sample import deadlines


class TestWhatABudgetReports:
    def test_no_budget_is_not_an_expired_budget(self):
        """
        The distinction the whole module rests on: an unbudgeted caller must never be blocked.

        Collapsing "no budget" into 0 seconds remaining would make every caller outside
        extraction -- the CLI's one-off commands, the webserver, the validators -- refuse every
        LLM call.
        """
        deadlines.clear()
        assert deadlines.active() is False
        assert deadlines.remaining() is None
        assert deadlines.expired() is False
        assert deadlines.clamp(90) == 90

    def test_clamp_narrows_to_the_remainder(self):
        with deadlines.Budget(30):
            assert deadlines.clamp(90) == pytest.approx(30, abs=1.0)
            # A smaller caller timeout wins: the budget is a ceiling, not a floor.
            assert deadlines.clamp(5) == 5

    def test_clamp_never_returns_zero_or_negative(self):
        """
        A zero timeout means something different (and worse) than a small one to an HTTP
        client, so the clamp refuses to produce one even past the deadline. Callers test
        expired() instead.
        """
        with deadlines.Budget(0.001):
            time.sleep(0.01)
            assert deadlines.remaining() < 0
            assert deadlines.clamp(90) > 0

    def test_a_remainder_too_small_to_use_counts_as_expired(self):
        """
        Below MIN_USEFUL_SLICE the budget reports exhausted even though time is left. A call
        issued with two seconds left cannot finish, but Ollama runs the generation to
        completion after the client gives up, so it costs the NEXT document a slot to produce a
        guaranteed failure for this one.
        """
        with deadlines.Budget(deadlines.MIN_USEFUL_SLICE / 2):
            assert deadlines.remaining() > 0
            assert deadlines.expired() is True


class TestTheBudgetDoesNotOutliveItsBlock:
    def test_exit_restores_the_absence_of_a_budget(self):
        """
        Pool threads are reused. A budget left behind would apply to the next document and,
        being already spent, would make it skip every LLM call -- a silently degraded corpus
        with no error anywhere.
        """
        with deadlines.Budget(10):
            assert deadlines.active() is True
        assert deadlines.active() is False

    def test_exit_restores_the_budget_even_when_the_body_raises(self):
        with pytest.raises(ValueError):
            with deadlines.Budget(10):
                raise ValueError("boom")
        assert deadlines.active() is False

    def test_nesting_shortens_but_never_extends(self):
        """
        An inner block cannot buy more time than the caller was promised; an outer budget is a
        commitment about total time, so a 1000s inner block inside a 10s outer one stays at 10s.
        """
        with deadlines.Budget(10):
            outer = deadlines.remaining()
            with deadlines.Budget(1000):
                assert deadlines.remaining() == pytest.approx(outer, abs=1.0)
            assert deadlines.remaining() == pytest.approx(outer, abs=1.0)

    def test_an_unbudgeted_inner_block_does_not_lift_the_outer_one(self):
        with deadlines.Budget(10):
            with deadlines.Budget(None):
                assert deadlines.active() is True
                assert deadlines.remaining() == pytest.approx(10, abs=1.0)


class TestBudgetExhaustedIsDistinguishable:
    def test_it_is_a_timeout_error(self):
        """
        So every existing `except TimeoutError` / `except Exception` handler in the LLM layers
        keeps degrading gracefully without being taught about a new exception type.
        """
        assert issubclass(deadlines.BudgetExhausted, TimeoutError)

    def test_but_not_the_same_as_one(self):
        """
        The circuit breakers open after three CONSECUTIVE failures. One over-budget document
        refuses several calls in a row, so counting them as failures would disable a whole
        verification layer for every document in flight beside it.
        """
        assert deadlines.BudgetExhausted is not TimeoutError
        assert not isinstance(TimeoutError("x"), deadlines.BudgetExhausted)
