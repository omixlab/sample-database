"""
Tests for the Ollama runtime reset that runs between extraction chunks.

The one behaviour worth more than a line of code here is the refusal to report success it has
not confirmed. ``POST /api/generate`` with ``keep_alive: 0`` returns when the unload is
SCHEDULED, seconds before the llama-server runner actually exits, so a reset that trusted the
HTTP status would hand the next chunk a runtime still tearing down the old runner -- re-creating
the condition chunking exists to end, while logging that it had been fixed.
"""
import json

import pytest

from sample import ollama_runtime as runtime


class _Response:
    def __init__(self, payload=None, status=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


class TestLoadedModels:
    def test_unreachable_server_is_none_not_empty(self, monkeypatch):
        """
        `None` and `[]` mean opposite things and the reset branches on the difference: `[]` is
        the goal state, `None` is "I could not ask" -- which must never be reported as a
        successful reset.
        """
        import requests

        def _boom(*a, **k):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(runtime.requests, "get", _boom)
        assert runtime.loaded_models() is None

    def test_a_malformed_body_is_also_none(self, monkeypatch):
        monkeypatch.setattr(runtime.requests, "get",
                            lambda *a, **k: _Response({"models": "gemma3:4b"}))
        assert runtime.loaded_models() is None

    def test_nothing_loaded_is_an_empty_list(self, monkeypatch):
        monkeypatch.setattr(runtime.requests, "get", lambda *a, **k: _Response({"models": []}))
        assert runtime.loaded_models() == []


class TestReset:
    def test_an_already_clean_runtime_is_a_successful_reset(self, monkeypatch):
        """
        Reported as success, not as a no-op: the caller asked for a postcondition, and it holds.
        """
        monkeypatch.setattr(runtime, "loaded_models", lambda *a, **k: [])
        summary = runtime.reset()
        assert summary["drained"] is True
        assert summary["unloaded"] == []
        assert "error" not in summary

    def test_it_unloads_every_loaded_model_and_waits_for_the_runner_to_go(self, monkeypatch):
        """
        The poll is the point. The unload is confirmed by /api/ps going empty, not by the POST
        returning 200.
        """
        states = [[{"name": "gemma3:4b"}], [{"name": "gemma3:4b"}], []]
        monkeypatch.setattr(runtime, "loaded_models",
                            lambda *a, **k: states.pop(0) if states else [])
        unloaded = []
        monkeypatch.setattr(runtime, "unload",
                            lambda model, base=None: unloaded.append(model) or True)
        monkeypatch.setattr(runtime.time, "sleep", lambda s: None)

        summary = runtime.reset(drain_timeout=10)
        assert unloaded == ["gemma3:4b"]
        assert summary["drained"] is True
        assert "error" not in summary

    def test_a_model_that_will_not_unload_is_reported_not_swallowed(self, monkeypatch):
        """
        A reset that cannot be completed must say so. The caller continues either way -- the
        finished chunk is on disk -- but a silent failure would leave the decay it exists to
        prevent, with a log line claiming otherwise.
        """
        monkeypatch.setattr(runtime, "loaded_models", lambda *a, **k: [{"name": "gemma3:4b"}])
        monkeypatch.setattr(runtime, "unload", lambda model, base=None: True)
        monkeypatch.setattr(runtime.time, "sleep", lambda s: None)

        summary = runtime.reset(drain_timeout=0)
        assert summary["drained"] is False
        assert "still loaded" in summary["error"]

    def test_an_unreachable_server_is_an_error_not_a_drain(self, monkeypatch):
        monkeypatch.setattr(runtime, "loaded_models", lambda *a, **k: None)
        summary = runtime.reset()
        assert summary["drained"] is False
        assert "api/ps" in summary["error"]

    def test_it_never_raises(self, monkeypatch):
        """
        A failed reset must not end a multi-day run: the chunk just finished is already on disk,
        and the worst case of continuing is the old, unchunked behaviour. So even an error this
        module does not anticipate comes back as a summary rather than as an exception.
        """
        def _boom(*a, **k):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(runtime, "loaded_models", lambda *a, **k: [{"name": "m"}])
        monkeypatch.setattr(runtime, "unload", _boom)
        summary = runtime.reset()
        assert summary["drained"] is False
        assert "RuntimeError" in summary["error"]

    def test_the_model_tag_is_read_from_either_field(self, monkeypatch):
        """
        /api/ps reports both `name` and `model`; a reset keyed on only one would silently
        unload nothing on a server version that omits it.
        """
        monkeypatch.setattr(runtime, "loaded_models",
                            lambda *a, **k: [{"model": "gemma3:4b"}])
        seen = []
        monkeypatch.setattr(runtime, "unload",
                            lambda model, base=None: seen.append(model) or True)
        monkeypatch.setattr(runtime, "_wait_until_empty", lambda base, limit: True)
        runtime.reset()
        assert seen == ["gemma3:4b"]
