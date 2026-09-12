"""GOLDEN RULE: the OTel OpenAI stream accumulator must never crash the
customer's `for chunk in stream:` loop.

Root cause: opentelemetry-instrumentation-openai's ChatStream.__next__ /
__anext__ call self._process_item(chunk) OUTSIDE their try block, and
_process_item -> module-level _accumulate_stream_items carries no @dont_throw
(unlike _process_complete_response / _build_from_streaming_response). Any
exception raised while accumulating a chunk escapes straight into the
customer's iteration loop. Live crash shapes from OpenAI-compatible endpoints
(Gemini and other gateways):

  * choice {"index": None} -> `len(choices) <= None` TypeError (all versions)
  * chunk with `choices` absent or None -> iteration TypeError
  * tool-call delta {"index": None} -> int(None) TypeError (<0.54.0; closed
    by the pyproject floor bump — the shim also assigns synthetic indexes so
    distinct calls don't collapse into slot 0 on >=0.54.0)

Fix: telemetry._patch_openai_otel_stream_accumulator replaces the module
attribute with a shim that pre-normalizes the telemetry-side dict and swallows
any remaining Exception (warn once), after OpenAIInstrumentor is activated.
"""
from __future__ import annotations

import sys

import pytest

# Skip the whole module if the bundled openai instrumentor is not installed
# (base deps include it, but keep CI/unit envs that strip extras green).
pytest.importorskip("opentelemetry.instrumentation.openai")

from opentelemetry.instrumentation.openai.shared import chat_wrappers  # noqa: E402

from token_police import telemetry  # noqa: E402


class _RecordingSpan:
    """Minimal span accepted by ChatStream / accumulator paths."""

    def __init__(self):
        self.events = []
        self.attrs = {}
        self.end_calls = 0
        self.status = None

    def is_recording(self):
        return True

    def add_event(self, name=None, *a, **k):
        self.events.append(name)

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def set_status(self, status, *a, **k):
        self.status = status

    def end(self):
        self.end_calls += 1

    def record_exception(self, *a, **k):
        pass


class _FakeSyncStream:
    """Minimal stand-in for openai Stream (sync iterator, dict chunks)."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        pass


def _raw_accumulator():
    """The unshimmed upstream function, regardless of current patch state."""
    fn = chat_wrappers._accumulate_stream_items
    while getattr(fn, "_tp_stream_guard", False):
        fn = fn.__wrapped__
    return fn


def _fresh_cr():
    """A complete_response dict shaped like ChatStream initializes it."""
    return {"choices": [], "model": ""}


@pytest.fixture(autouse=True)
def _restore_accumulator_state():
    """Leave chat_wrappers module state + warn-once flag as we found them."""
    saved_fn = chat_wrappers._accumulate_stream_items
    saved_warned = telemetry._openai_stream_accum_warned
    yield
    chat_wrappers._accumulate_stream_items = saved_fn
    telemetry._openai_stream_accum_warned = saved_warned


def _install_patch():
    chat_wrappers._accumulate_stream_items = _raw_accumulator()
    telemetry._openai_stream_accum_warned = False
    telemetry._patch_openai_otel_stream_accumulator(log_errors=True)


# ---------------------------------------------------------------------------
# Install mechanics
# ---------------------------------------------------------------------------

def test_patch_installs_marker_and_preserves_original():
    raw = _raw_accumulator()
    _install_patch()
    shim = chat_wrappers._accumulate_stream_items
    assert shim is not raw
    assert getattr(shim, "_tp_stream_guard", False) is True
    assert shim.__wrapped__ is raw


def test_patch_idempotent():
    _install_patch()
    shim1 = chat_wrappers._accumulate_stream_items
    telemetry._patch_openai_otel_stream_accumulator(log_errors=True)
    shim2 = chat_wrappers._accumulate_stream_items
    assert shim1 is shim2  # no stacking


def test_patch_failopen_on_import_error(monkeypatch):
    """Missing instrumentor must not raise out of the patcher."""
    real_import = __import__

    def boom(name, *a, **k):
        if name.startswith("opentelemetry.instrumentation.openai"):
            raise ImportError("simulated missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", boom)
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k.startswith("opentelemetry.instrumentation.openai")
    }
    try:
        telemetry._patch_openai_otel_stream_accumulator(log_errors=True)  # must not raise
    finally:
        sys.modules.update(saved)


def test_auto_instrument_calls_patch(monkeypatch):
    """_auto_instrument wires the guard when OpenAIInstrumentor activates."""
    import importlib

    calls = []

    class FakeInstr:
        is_instrumented_by_opentelemetry = False

        def instrument(self):
            pass

    class FakeMod:
        OpenAIInstrumentor = FakeInstr

    def fake_import(path, package=None):
        if path == "opentelemetry.instrumentation.openai":
            return FakeMod
        raise ImportError(path)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(telemetry, "_maybe_warn_missing_instrumentor", lambda *a, **k: None)
    monkeypatch.setattr(
        telemetry, "_unwrap_openai_responses_hooks", lambda log_errors=False: None
    )
    monkeypatch.setattr(
        telemetry,
        "_patch_openai_otel_stream_accumulator",
        lambda log_errors=False: calls.append(log_errors),
    )

    telemetry._auto_instrument(log_errors=True)
    assert calls == [True]


# ---------------------------------------------------------------------------
# Escape-proof regression: the bug exists unpatched, is gone patched
# ---------------------------------------------------------------------------

def test_unpatched_none_choice_index_raises_typeerror():
    """Documents the upstream defect our shim guards against."""
    raw = _raw_accumulator()
    with pytest.raises(TypeError):
        raw({"choices": [{"index": None, "delta": {"content": "x"}}]}, _fresh_cr())


def test_patched_none_choice_index_accumulates():
    _install_patch()
    cr = _fresh_cr()
    chat_wrappers._accumulate_stream_items(
        {"model": "m", "choices": [{"index": None, "delta": {"content": "Hel"}}]}, cr
    )
    chat_wrappers._accumulate_stream_items(
        {"model": "m", "choices": [{"index": None, "delta": {"content": "lo"}}]}, cr
    )
    assert cr["choices"][0]["message"]["content"] == "Hello"


def test_patched_missing_and_none_choices_do_not_raise():
    _install_patch()
    cr = _fresh_cr()
    # Bare usage-only terminal frame with no choices key at all.
    chat_wrappers._accumulate_stream_items({"model": "m", "id": "c1"}, cr)
    # Explicit null choices.
    chat_wrappers._accumulate_stream_items({"model": "m", "choices": None}, cr)
    assert cr["model"] == "m"


def test_patched_tool_call_none_index_gets_distinct_slots():
    """Gemini shape: full id/type/function in one delta, index omitted (None).
    Two id-bearing deltas must land in DISTINCT accumulated slots; an id-less
    fragment continues the last opened slot."""
    _install_patch()
    cr = _fresh_cr()

    def tc_chunk(tool_call):
        return {
            "model": "m",
            "choices": [{"index": 0, "delta": {"tool_calls": [tool_call]}}],
        }

    chat_wrappers._accumulate_stream_items(
        tc_chunk({"index": None, "id": "call_a", "type": "function",
                  "function": {"name": "f1", "arguments": '{"x":'}}),
        cr,
    )
    # id-less continuation fragment -> appends to call_a's arguments.
    chat_wrappers._accumulate_stream_items(
        tc_chunk({"index": None, "function": {"arguments": "1}"}}),
        cr,
    )
    chat_wrappers._accumulate_stream_items(
        tc_chunk({"index": None, "id": "call_b", "type": "function",
                  "function": {"name": "f2", "arguments": "{}"}}),
        cr,
    )

    tool_calls = cr["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["id"] == "call_a"
    assert tool_calls[0]["function"]["name"] == "f1"
    assert tool_calls[0]["function"]["arguments"] == '{"x":1}'
    assert tool_calls[1]["id"] == "call_b"
    assert tool_calls[1]["function"]["name"] == "f2"


def test_patched_usage_preserved_when_accumulation_crashes(monkeypatch):
    """usage is captured BEFORE the choices loop; a swallowed crash later in
    the same chunk must keep it. A sparse valid index (5 into an empty list)
    is a shape normalization leaves alone and upstream IndexErrors on."""
    _install_patch()
    warnings = []
    monkeypatch.setattr(
        telemetry.logger, "warning", lambda *a, **k: warnings.append(a)
    )
    cr = _fresh_cr()
    crashing = {
        "model": "m",
        "usage": {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10},
        "choices": [{"index": 5, "delta": {"content": "x"}}],
    }
    chat_wrappers._accumulate_stream_items(crashing, cr)  # must not raise
    assert cr["usage"] == {
        "prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10,
    }
    assert len(warnings) == 1
    # Warn-once: a second swallowed failure stays silent.
    chat_wrappers._accumulate_stream_items(dict(crashing), _fresh_cr())
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# Through a real ChatStream: customer loop survives, chunks untouched
# ---------------------------------------------------------------------------

def test_real_chatstream_survives_malformed_chunks():
    _install_patch()
    chunk1 = {"model": "m", "choices": [{"index": None, "delta": {"content": "Hi"}}]}
    chunk2 = {  # usage-only terminal frame, no choices key
        "model": "m",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }
    span = _RecordingSpan()
    stream = chat_wrappers.ChatStream(
        span,
        _FakeSyncStream([chunk1, chunk2]),
        instance=None,
        start_time=0.0,
        request_kwargs={"model": "m"},
    )

    got = list(stream)  # the customer loop — must not raise

    # Customer receives the ORIGINAL chunk objects, unmutated by our shim.
    assert got[0] is chunk1
    assert got[1] is chunk2
    assert chunk1["choices"][0]["index"] is None  # deepcopy, not in-place
    # Telemetry still accumulated content + usage.
    assert stream._complete_response["choices"][0]["message"]["content"] == "Hi"
    assert stream._complete_response["usage"]["total_tokens"] == 3


def test_real_chatstream_unpatched_baseline_raises():
    """Documents the escape path: without the shim the same stream kills the
    customer loop with a TypeError."""
    chat_wrappers._accumulate_stream_items = _raw_accumulator()
    chunk = {"model": "m", "choices": [{"index": None, "delta": {"content": "Hi"}}]}
    stream = chat_wrappers.ChatStream(
        _RecordingSpan(),
        _FakeSyncStream([chunk]),
        instance=None,
        start_time=0.0,
        request_kwargs={"model": "m"},
    )
    with pytest.raises(TypeError):
        list(stream)
