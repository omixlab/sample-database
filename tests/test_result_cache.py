"""
Tests for :mod:`sample.result_cache` -- the Redis result cache for the expensive analyses.

The behaviours that actually matter here are the failure modes, not the happy path: an absent or
broken Redis must be indistinguishable from a cache miss (the caller just computes), and a FAILED
analysis must never be stored, or a transient outage gets served for a week after it is fixed.

Uses a fake client rather than a real Redis, so the suite has no service dependency -- the same
reason the predictor tests mock their tools rather than requiring every environment to be built.
"""
from typing import Any, Dict

import pytest

from sample import result_cache


class FakeRedis:
    """A minimal stand-in: just enough of the interface result_cache actually calls."""

    def __init__(self, fail: bool = False) -> None:
        self.store: Dict[str, Any] = {}
        self.ttls: Dict[str, int] = {}
        self.fail = fail

    def ping(self) -> bool:
        if self.fail:
            raise ConnectionError("down")
        return True

    def get(self, key: str) -> Any:
        if self.fail:
            raise ConnectionError("down")
        return self.store.get(key)

    def setex(self, key: str, ttl: int, value: Any) -> None:
        if self.fail:
            raise ConnectionError("down")
        self.store[key] = value
        self.ttls[key] = ttl


@pytest.fixture(autouse=True)
def _reset_cache() -> Any:
    """Each test re-probes, so one test's fake never leaks into the next."""
    result_cache.reset()
    yield
    result_cache.reset()


def _install(monkeypatch: pytest.MonkeyPatch, client: Any) -> None:
    """Points the module at `client` without needing a real redis package or server."""
    monkeypatch.setattr(result_cache, "_connect", lambda: client)


class TestCacheKey:
    def test_same_inputs_give_the_same_key(self) -> None:
        a = result_cache.cache_key("fold", {"protein": "MGIGKFL"})
        b = result_cache.cache_key("fold", {"protein": "MGIGKFL"})
        assert a == b

    def test_key_is_independent_of_dict_ordering(self) -> None:
        a = result_cache.cache_key("x", {"one": 1, "two": 2})
        b = result_cache.cache_key("x", {"two": 2, "one": 1})
        assert a == b

    def test_different_inputs_give_different_keys(self) -> None:
        a = result_cache.cache_key("fold", {"protein": "MGIGKFL"})
        b = result_cache.cache_key("fold", {"protein": "MGIGKFW"})
        assert a != b

    def test_kind_namespaces_the_key(self) -> None:
        # Two analyses of the same construct must not collide.
        assert (result_cache.cache_key("fold", {"x": 1})
                != result_cache.cache_key("mpepe", {"x": 1}))

    def test_key_length_is_bounded_regardless_of_input_size(self) -> None:
        short = result_cache.cache_key("fold", {"protein": "M"})
        long = result_cache.cache_key("fold", {"protein": "M" * 100_000})
        assert len(short) == len(long)


class TestRoundTrip:
    def test_set_then_get_returns_the_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, FakeRedis())
        payload = {"protein": "MGIGKFL"}
        result_cache.set("fold", payload, {"folded": True, "pdb": "ATOM..."})
        assert result_cache.get("fold", payload)["pdb"] == "ATOM..."

    def test_miss_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, FakeRedis())
        assert result_cache.get("fold", {"protein": "NEVER-STORED"}) is None

    def test_stores_with_a_seven_day_ttl_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.set("fold", {"p": 1}, {"folded": True})
        assert list(client.ttls.values()) == [7 * 24 * 60 * 60]

    def test_explicit_ttl_is_honoured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.set("fold", {"p": 1}, {"folded": True}, ttl_seconds=60)
        assert list(client.ttls.values()) == [60]


class TestDegradesWithoutRedis:
    def test_get_returns_none_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, None)
        assert result_cache.get("fold", {"p": 1}) is None

    def test_set_reports_false_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, None)
        assert result_cache.set("fold", {"p": 1}, {"folded": True}) is False

    def test_cached_still_computes_when_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The whole point: no Redis must be a non-event for the caller.
        _install(monkeypatch, None)
        calls = []
        value = result_cache.cached("fold", {"p": 1},
                                    lambda: calls.append(1) or {"folded": True})
        assert value == {"folded": True}
        assert len(calls) == 1

    def test_a_broken_server_is_treated_as_a_miss(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, FakeRedis(fail=True))
        assert result_cache.get("fold", {"p": 1}) is None
        assert result_cache.set("fold", {"p": 1}, {"folded": True}) is False

    def test_a_corrupt_entry_is_a_miss_not_a_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        client.store[result_cache.cache_key("fold", {"p": 1})] = b"{not json"
        assert result_cache.get("fold", {"p": 1}) is None


class TestCachedComputesOnce:
    def test_second_call_is_served_from_the_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, FakeRedis())
        calls = []

        def compute() -> Dict[str, Any]:
            calls.append(1)
            return {"folded": True, "pdb": "ATOM"}

        payload = {"protein": "MGIGKFL"}
        first = result_cache.cached("fold", payload, compute)
        second = result_cache.cached("fold", payload, compute)
        assert first == second
        assert len(calls) == 1, "the second call must not recompute"

    def test_a_different_input_is_computed_separately(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install(monkeypatch, FakeRedis())
        calls = []
        result_cache.cached("fold", {"protein": "A"}, lambda: calls.append(1) or {"folded": True})
        result_cache.cached("fold", {"protein": "B"}, lambda: calls.append(1) or {"folded": True})
        assert len(calls) == 2


class TestFailuresAreNotCached:
    """The most important behaviour here: a failed run reflects the MACHINE, not the input."""

    def test_unavailable_tool_is_not_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.cached("mpepe", {"dna": "ATG"},
                            lambda: {"available": False, "note": "not installed"})
        assert client.store == {}, "an uninstalled tool must not be cached for a week"

    def test_failed_fold_is_not_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.cached("fold", {"protein": "M"},
                            lambda: {"available": True, "folded": False, "note": "too long"})
        assert client.store == {}

    def test_mpepe_run_without_a_number_is_not_stored(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.cached("mpepe", {"dna": "ATG"},
                            lambda: {"available": True, "probability": None, "note": "failed"})
        assert client.store == {}

    def test_none_is_not_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.cached("fold", {"protein": "M"}, lambda: None)
        assert client.store == {}

    def test_a_successful_result_is_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.cached("fold", {"protein": "M"},
                            lambda: {"available": True, "folded": True, "pdb": "ATOM"})
        assert len(client.store) == 1

    def test_a_successful_mpepe_result_is_stored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = FakeRedis()
        _install(monkeypatch, client)
        result_cache.cached("mpepe", {"dna": "ATG"},
                            lambda: {"available": True, "probability": 0.71, "std": 0.18})
        assert len(client.store) == 1
