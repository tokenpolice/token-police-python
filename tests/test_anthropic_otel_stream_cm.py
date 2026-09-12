"""GOLDEN RULE: OTel AnthropicAsyncStream must support async with.

Regression for support_agent_pydantic_ai_python --provider anthropic --streaming
(and any customer code that does async with on create(stream=True)).

Root cause: opentelemetry-instrumentation-anthropic wraps streaming responses
in AnthropicAsyncStream (wrapt.ObjectProxy). On wrapt < 2.0 ObjectProxy has
sync CM dunders but not async ones; Python looks up dunders on the type, so
customer-correct `async with stream:` raises TypeError under TokenPolice. On
wrapt >= 2.0 ObjectProxy grew a delegating __aenter__, so `async with` binds
the RAW wrapped stream instead — instrumentation bypassed, call unmetered.

Fix: telemetry._patch_anthropic_otel_stream_cm installs type-level
__aenter__/__aexit__ after AnthropicInstrumentor is activated.
"""
from __future__ import annotations

import asyncio
import sys

import pytest

# Skip the whole module if the bundled anthropic instrumentor is not installed
# (base deps include it, but keep CI/unit envs that strip extras green).
pytest.importorskip("opentelemetry.instrumentation.anthropic")

from opentelemetry.instrumentation.anthropic.streaming import (  # noqa: E402
    AnthropicAsyncStream,
)
from token_police import telemetry  # noqa: E402


class _RecordingSpan:
    """Recording span that counts end() calls and captures the final status."""

    def __init__(self):
        self.end_calls = 0
        self.status = None
        self.attrs = {}

    def is_recording(self):
        return self.end_calls == 0

    def end(self):
        self.end_calls += 1

    def set_status(self, status, *a, **k):
        self.status = status

    def set_attribute(self, key, value):
        self.attrs[key] = value

    def record_exception(self, *a, **k):
        pass


class _DummySpan:
    def is_recording(self):
        return False

    def end(self):
        pass

    def set_status(self, *a, **k):
        pass

    def set_attribute(self, *a, **k):
        pass

    def record_exception(self, *a, **k):
        pass


class _FakeAsyncStream:
    """Minimal stand-in for anthropic.AsyncStream (async-iter + async CM + close)."""

    def __init__(self, chunks=None):
        self._chunks = list(chunks if chunks is not None else [1, 2, 3])
        # Pre-build iterator: OTel AnthropicAsyncStream.__anext__ calls
        # wrapped.__anext__ without first calling wrapped.__aiter__.
        self._it = iter(self._chunks)
        self.closed = False
        self.close_calls = 0

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        await self.close()
        return False

    async def close(self):
        self.close_calls += 1
        self.closed = True


def _make_proxy(wrapped=None, span=None):
    return AnthropicAsyncStream(
        span if span is not None else _DummySpan(),
        wrapped if wrapped is not None else _FakeAsyncStream(),
        object(),  # instance
        0.0,  # start_time
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _restore_cm_state():
    """Leave AnthropicAsyncStream class state as we found it after each test."""
    had_aenter = "__aenter__" in AnthropicAsyncStream.__dict__
    had_aexit = "__aexit__" in AnthropicAsyncStream.__dict__
    had_flag = getattr(AnthropicAsyncStream, "_tp_cm_patched", False)
    saved_aenter = AnthropicAsyncStream.__dict__.get("__aenter__")
    saved_aexit = AnthropicAsyncStream.__dict__.get("__aexit__")
    yield
    # Tear down any methods/flags we may have added.
    if not had_aenter and "__aenter__" in AnthropicAsyncStream.__dict__:
        delattr(AnthropicAsyncStream, "__aenter__")
    elif had_aenter and saved_aenter is not None:
        AnthropicAsyncStream.__aenter__ = saved_aenter
    if not had_aexit and "__aexit__" in AnthropicAsyncStream.__dict__:
        delattr(AnthropicAsyncStream, "__aexit__")
    elif had_aexit and saved_aexit is not None:
        AnthropicAsyncStream.__aexit__ = saved_aexit
    if had_flag:
        AnthropicAsyncStream._tp_cm_patched = had_flag
    elif hasattr(AnthropicAsyncStream, "_tp_cm_patched"):
        delattr(AnthropicAsyncStream, "_tp_cm_patched")


def _strip_tp_patch():
    """Remove TP's CM methods if present so we can assert the OTel baseline."""
    if getattr(AnthropicAsyncStream, "_tp_cm_patched", False):
        if "__aenter__" in AnthropicAsyncStream.__dict__:
            delattr(AnthropicAsyncStream, "__aenter__")
        if "__aexit__" in AnthropicAsyncStream.__dict__:
            delattr(AnthropicAsyncStream, "__aexit__")
        delattr(AnthropicAsyncStream, "_tp_cm_patched")


def test_unpatched_async_with_is_broken_upstream():
    """Documents the upstream baseline our patch repairs — on both wrapt eras.

    wrapt < 2.0: ``ObjectProxy`` has no type-level async CM at all, so
    ``async with`` raises TypeError (loud failure).
    wrapt >= 2.0: ``ObjectProxy`` grew an (undocumented) ``__aenter__`` that
    delegates to ``__wrapped__``, so ``async with`` binds the RAW stream —
    the instrumented ``__anext__`` is bypassed and the span is never ended
    (silent metering loss). Worse symptom, same fix.
    """
    _strip_tp_patch()
    # Only assert while upstream OTel itself still lacks its own __aenter__.
    if "__aenter__" in AnthropicAsyncStream.__dict__:
        pytest.skip("upstream AnthropicAsyncStream already defines __aenter__")
    wrapped = _FakeAsyncStream([1, 2])
    proxy = _make_proxy(wrapped)

    if not hasattr(AnthropicAsyncStream, "__aenter__"):
        # wrapt < 2 era: no async CM anywhere on the MRO → hard TypeError.
        async def _body():
            async with proxy:
                pass

        with pytest.raises(TypeError, match="asynchronous context manager"):
            _run(_body())
    else:
        # wrapt >= 2 era: the inherited ObjectProxy.__aenter__ delegates to
        # __wrapped__, so the as-target is the raw stream, not the proxy.
        async def _body():
            async with proxy as s:
                return s

        assert _run(_body()) is wrapped


def test_patched_async_with_succeeds_and_closes():
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm(log_errors=True)
    wrapped = _FakeAsyncStream([10, 20])
    proxy = _make_proxy(wrapped)

    async def _body():
        async with proxy as s:
            assert s is proxy  # stay on the instrumented proxy
            got = []
            async for item in s:
                got.append(item)
            return got

    got = _run(_body())
    assert got == [10, 20]
    assert wrapped.closed is True
    assert wrapped.close_calls >= 1


def test_early_exit_ends_span_and_closes():
    """Breaking out of `async with` before exhaustion must still end the OTel
    span (meter the call) and close the underlying HTTP stream."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    span = _RecordingSpan()
    wrapped = _FakeAsyncStream([10, 20, 30])
    proxy = _make_proxy(wrapped, span)

    async def _body():
        async with proxy as s:
            async for _ in s:
                break  # early exit — __anext__ never sees StopAsyncIteration

    _run(_body())
    assert wrapped.closed is True
    assert span.end_calls == 1
    assert proxy._instrumentation_completed is True


def test_body_exception_propagates_and_ends_span():
    """An exception raised in the `async with` body must propagate unchanged,
    still close the underlying stream, and end the span with error status."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    span = _RecordingSpan()
    wrapped = _FakeAsyncStream([10, 20])
    proxy = _make_proxy(wrapped, span)

    async def _body():
        async with proxy as s:
            async for _ in s:
                raise RuntimeError("customer body error")

    with pytest.raises(RuntimeError, match="customer body error"):
        _run(_body())
    assert wrapped.closed is True
    assert span.end_calls == 1
    from opentelemetry.trace.status import StatusCode
    assert span.status is not None
    assert span.status.status_code is StatusCode.ERROR
    assert proxy._instrumentation_completed is True


def test_body_exception_hostile_str_still_ends_span():
    """A body exception whose __str__ raises must still propagate unchanged AND
    still end the span (a hostile __str__ must not un-meter the call)."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    span = _RecordingSpan()
    wrapped = _FakeAsyncStream([10, 20])
    proxy = _make_proxy(wrapped, span)

    class _HostileError(Exception):
        def __str__(self):
            raise RuntimeError("hostile __str__")

    async def _body():
        async with proxy as s:
            async for _ in s:
                raise _HostileError()

    with pytest.raises(_HostileError):
        _run(_body())
    assert wrapped.closed is True
    assert span.end_calls == 1
    from opentelemetry.trace.status import StatusCode
    assert span.status is not None
    assert span.status.status_code is StatusCode.ERROR
    assert proxy._instrumentation_completed is True


def test_full_iteration_does_not_double_end_span():
    """Fully-consumed stream: __anext__ already ended the span; __aexit__ must
    be a no-op for instrumentation (no second end())."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    span = _RecordingSpan()
    wrapped = _FakeAsyncStream([10, 20])
    proxy = _make_proxy(wrapped, span)

    async def _body():
        async with proxy as s:
            async for _ in s:
                pass

    _run(_body())
    assert wrapped.closed is True
    assert span.end_calls == 1


def test_patch_idempotent():
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    aenter1 = AnthropicAsyncStream.__dict__.get("__aenter__")
    telemetry._patch_anthropic_otel_stream_cm()
    aenter2 = AnthropicAsyncStream.__dict__.get("__aenter__")
    assert aenter1 is aenter2
    assert getattr(AnthropicAsyncStream, "_tp_cm_patched", False) is True


def test_patch_failopen_on_import_error(monkeypatch):
    """Missing instrumentor must not raise out of the patcher."""
    real_import = __import__

    def boom(name, *a, **k):
        if name.startswith("opentelemetry.instrumentation.anthropic"):
            raise ImportError("simulated missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", boom)
    # Also clear sys.modules so import goes through our boom path.
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k.startswith("opentelemetry.instrumentation.anthropic")
    }
    try:
        telemetry._patch_anthropic_otel_stream_cm(log_errors=True)  # must not raise
    finally:
        sys.modules.update(saved)


def test_auto_instrument_calls_patch(monkeypatch):
    """_auto_instrument wires the CM patch when AnthropicInstrumentor activates."""
    import importlib

    calls = []

    class FakeInstr:
        is_instrumented_by_opentelemetry = False

        def instrument(self):
            pass

    class FakeMod:
        AnthropicInstrumentor = FakeInstr

    def fake_import(path, package=None):
        if path == "opentelemetry.instrumentation.anthropic":
            return FakeMod
        raise ImportError(path)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    monkeypatch.setattr(telemetry, "_maybe_warn_missing_instrumentor", lambda *a, **k: None)
    monkeypatch.setattr(
        telemetry,
        "_patch_anthropic_otel_stream_cm",
        lambda log_errors=False: calls.append(log_errors),
    )
    monkeypatch.setattr(
        telemetry, "_unwrap_openai_responses_hooks", lambda log_errors=False: None
    )

    # Non-anthropic modules raise ImportError → skip; anthropic path patches CM.
    telemetry._auto_instrument(log_errors=True)
    assert calls == [True]
