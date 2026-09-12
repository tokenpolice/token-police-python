"""OTel AnthropicStream (sync) must keep `with` on the instrumented proxy.

Sync twin of test_anthropic_otel_stream_cm.py. Unlike the async pair there is
no wrapt-era split: wrapt.ObjectProxy has ALWAYS supplied inherited delegating
__enter__/__exit__, and anthropic.Stream.__enter__ returns self — so on every
wrapt version an unpatched `with proxy as s` binds the RAW stream, iteration
bypasses the instrumented __next__, the span never ends, and the call is
silently unmetered.

Live victim: `with client.beta.messages.create(..., stream=True) as s` (and
the Bedrock beta twin) — the beta seam is wrapped only by OTel, while the
non-beta path is additionally shielded by TP's own _ModeAStreamProxy.

Fix: telemetry._patch_anthropic_otel_stream_cm installs class-own
__enter__/__exit__ on AnthropicStream (a class-own method overrides the
inherited ObjectProxy delegate).
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("opentelemetry.instrumentation.anthropic")

from opentelemetry.instrumentation.anthropic.streaming import (  # noqa: E402
    AnthropicAsyncStream,
    AnthropicStream,
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


class _FakeStream:
    """Minimal stand-in for anthropic.Stream (sync iter + sync CM + close)."""

    def __init__(self, chunks=None):
        self._chunks = list(chunks if chunks is not None else [1, 2, 3])
        # Pre-build iterator: OTel AnthropicStream.__next__ calls
        # wrapped.__next__ without first calling wrapped.__iter__.
        self._it = iter(self._chunks)
        self.closed = False
        self.close_calls = 0

    def __iter__(self):
        self._it = iter(self._chunks)
        return self

    def __next__(self):
        return next(self._it)

    def __enter__(self):
        # Mirrors anthropic.Stream.__enter__ — returns the raw stream. This is
        # exactly what ObjectProxy's inherited delegate hands the customer.
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def close(self):
        self.close_calls += 1
        self.closed = True


def _make_proxy(wrapped=None, span=None):
    return AnthropicStream(
        span if span is not None else _DummySpan(),
        wrapped if wrapped is not None else _FakeStream(),
        object(),  # instance
        0.0,  # start_time
    )


def _snapshot_cls(cls, names):
    return {n: (n in cls.__dict__, cls.__dict__.get(n)) for n in names}


def _restore_cls(cls, snap):
    for name, (had, saved) in snap.items():
        if not had and name in cls.__dict__:
            delattr(cls, name)
        elif had and saved is not None:
            setattr(cls, name, saved)


@pytest.fixture(autouse=True)
def _restore_cm_state():
    """Leave BOTH proxy classes' state as we found it after each test — the
    shared patcher touches AnthropicAsyncStream too."""
    names = ("__enter__", "__exit__", "__aenter__", "__aexit__", "_tp_cm_patched")
    sync_snap = _snapshot_cls(AnthropicStream, names)
    async_snap = _snapshot_cls(AnthropicAsyncStream, names)
    yield
    _restore_cls(AnthropicStream, sync_snap)
    _restore_cls(AnthropicAsyncStream, async_snap)


def _strip_tp_patch():
    """Remove TP's sync CM methods if present to assert the OTel baseline."""
    if getattr(AnthropicStream, "_tp_cm_patched", False):
        if "__enter__" in AnthropicStream.__dict__:
            delattr(AnthropicStream, "__enter__")
        if "__exit__" in AnthropicStream.__dict__:
            delattr(AnthropicStream, "__exit__")
        delattr(AnthropicStream, "_tp_cm_patched")


def test_unpatched_with_binds_raw_stream():
    """Documents the upstream baseline the patch repairs — every wrapt era:
    the inherited ObjectProxy.__enter__ delegates to __wrapped__, so the
    as-target is the RAW stream and instrumentation is bypassed."""
    _strip_tp_patch()
    # Only assert while upstream OTel itself still lacks its own __enter__.
    if "__enter__" in AnthropicStream.__dict__:
        pytest.skip("upstream AnthropicStream already defines __enter__")
    span = _RecordingSpan()
    wrapped = _FakeStream([1, 2])
    proxy = _make_proxy(wrapped, span)

    with proxy as s:
        assert s is wrapped  # raw stream, not the proxy
        for _ in s:
            pass
    assert span.end_calls == 0  # span never ended → call unmetered


def test_patched_with_returns_proxy_and_meters():
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm(log_errors=True)
    span = _RecordingSpan()
    wrapped = _FakeStream([10, 20])
    proxy = _make_proxy(wrapped, span)

    got = []
    with proxy as s:
        assert s is proxy  # stay on the instrumented proxy
        for item in s:
            got.append(item)
    assert got == [10, 20]
    assert wrapped.closed is True
    assert wrapped.close_calls >= 1
    assert span.end_calls == 1


def test_early_break_ends_span_and_closes():
    """Breaking out of `with` before exhaustion must still end the OTel span
    (meter the call) and close the underlying HTTP stream."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    span = _RecordingSpan()
    wrapped = _FakeStream([10, 20, 30])
    proxy = _make_proxy(wrapped, span)

    with proxy as s:
        for _ in s:
            break  # early exit — __next__ never sees StopIteration

    assert wrapped.closed is True
    assert span.end_calls == 1
    assert proxy._instrumentation_completed is True


def test_body_exception_propagates_and_ends_span():
    """An exception raised in the `with` body must propagate unchanged, still
    close the underlying stream, and end the span with error status."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    span = _RecordingSpan()
    wrapped = _FakeStream([10, 20])
    proxy = _make_proxy(wrapped, span)

    with pytest.raises(RuntimeError, match="customer body error"):
        with proxy as s:
            for _ in s:
                raise RuntimeError("customer body error")

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
    wrapped = _FakeStream([10, 20])
    proxy = _make_proxy(wrapped, span)

    class _HostileError(Exception):
        def __str__(self):
            raise RuntimeError("hostile __str__")

    with pytest.raises(_HostileError):
        with proxy as s:
            for _ in s:
                raise _HostileError()

    assert wrapped.closed is True
    assert span.end_calls == 1
    from opentelemetry.trace.status import StatusCode
    assert span.status is not None
    assert span.status.status_code is StatusCode.ERROR
    assert proxy._instrumentation_completed is True


def test_full_iteration_does_not_double_end_span():
    """Fully-consumed stream: __next__ already ended the span; __exit__ must
    be a no-op for instrumentation (no second end())."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    span = _RecordingSpan()
    wrapped = _FakeStream([10, 20])
    proxy = _make_proxy(wrapped, span)

    with proxy as s:
        for _ in s:
            pass

    assert wrapped.closed is True
    assert span.end_calls == 1


def test_patch_idempotent_and_independent_of_async():
    """Re-patch keeps the same function object, and the sync install must not
    be short-circuited by the async class already being patched (per-class
    flags — the classes are siblings)."""
    _strip_tp_patch()
    telemetry._patch_anthropic_otel_stream_cm()
    enter1 = AnthropicStream.__dict__.get("__enter__")
    telemetry._patch_anthropic_otel_stream_cm()
    enter2 = AnthropicStream.__dict__.get("__enter__")
    assert enter1 is enter2
    assert getattr(AnthropicStream, "_tp_cm_patched", False) is True

    # Independence: strip ONLY the sync patch (async class stays flagged as
    # patched from the calls above) and re-run — the sync pair must reinstall
    # even though AnthropicAsyncStream._tp_cm_patched is already True.
    _strip_tp_patch()
    assert "__enter__" not in AnthropicStream.__dict__
    assert getattr(AnthropicAsyncStream, "_tp_cm_patched", False) is True
    telemetry._patch_anthropic_otel_stream_cm()
    assert "__enter__" in AnthropicStream.__dict__
    assert getattr(AnthropicStream, "_tp_cm_patched", False) is True


def test_patch_failopen_on_import_error(monkeypatch):
    """Missing instrumentor must not raise out of the patcher."""
    real_import = __import__

    def boom(name, *a, **k):
        if name.startswith("opentelemetry.instrumentation.anthropic"):
            raise ImportError("simulated missing")
        return real_import(name, *a, **k)

    monkeypatch.setattr("builtins.__import__", boom)
    saved = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k.startswith("opentelemetry.instrumentation.anthropic")
    }
    try:
        telemetry._patch_anthropic_otel_stream_cm(log_errors=True)  # must not raise
    finally:
        sys.modules.update(saved)
