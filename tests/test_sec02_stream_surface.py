"""Regression: Mode-A streaming must hand the customer an object at
least as capable as the provider's original stream (context-manager protocol,
`.response`, arbitrary attribute passthrough, a `close()` that closes the
underlying connection) — NOT a bare generator — and early abandonment / CM exit
/ explicit close must close-through to BOTH the underlying provider stream (leak
fix) AND the retained metering generator (its `finally` still flushes partial
telemetry). Every close step is fail-open: nothing escapes into customer
code. Mid-stream provider errors are re-raised verbatim.

These tests fail against the pre-fix bare-generator implementation (no proxy
type, no `.response`, `TypeError` on `with`, underlying never closed on
abandonment) and pass post-fix. Encodes contract assertions 2-21.
"""
from __future__ import annotations

import asyncio
import types
from unittest.mock import MagicMock

import pytest

from token_police import enforcer


# ─── fakes ───────────────────────────────────────────────────────────
class FakeSyncStream:
    """Mimics openai.Stream: iterable + context manager + .response + close()."""

    def __init__(self, chunks=None):
        self._chunks = list(chunks if chunks is not None else [
            {"choices": [{"delta": {"content": "hi"}}]},
        ])
        self._it = iter(self._chunks)
        self.response = object()
        self.sentinel = object()
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def close(self):
        self.closed = True


class FakeAsyncStream:
    """Mimics openai.AsyncStream: async-iterable + .response + coroutine aclose()."""

    def __init__(self, chunks=None):
        self._chunks = list(chunks if chunks is not None else [
            {"choices": [{"delta": {"content": "hi"}}]},
        ])
        self.response = object()
        self.sentinel = object()
        self.closed = False

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class SyncOnlyStream:
    """A plain sync generator-like stream."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class NoCloseStream:
    """A stream missing `close`/`__exit__` entirely (hostile input)."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class NoAcloseAsyncStream:
    """Async stream missing aclose/close entirely."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _Sess:
    """Minimal session — the metering generator only reads/sets attributes."""

    def __init__(self):
        self._defer_telemetry = False


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _drain_async(proxy):
    out = []
    async for c in proxy:
        out.append(c)
    return out


@pytest.fixture(autouse=True)
def _noop_flush(monkeypatch):
    """Default: neutralize the real flush (OTel side effects). Tests that need to
    spy the flush override this with their own MagicMock."""
    monkeypatch.setattr(enforcer, "_flush_deferred_spans", MagicMock())


# ─── surface restoration ─────────────────────────────────────────────
def test_returns_proxy_not_bare_generator():                       # assertion 2
    returned = enforcer._wrap_mode_a_sync_stream(
        FakeSyncStream(), "openai", _Sess(), False)
    assert not isinstance(returned, types.GeneratorType)
    assert type(returned).__name__ == "_ModeAStreamProxy"


def test_response_delegation_sync():                               # assertion 3
    fake = FakeSyncStream()
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    assert hasattr(wrapped, "response")
    assert wrapped.response is fake.response


def test_response_delegation_async():                             # assertion 4
    fake = FakeAsyncStream()
    wrapped = enforcer._wrap_mode_a_async_stream(fake, "openai", _Sess(), False)
    assert hasattr(wrapped, "response")
    assert wrapped.response is fake.response


def test_arbitrary_attribute_passthrough():                       # assertion 5
    fake = FakeSyncStream()
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    assert wrapped.sentinel is fake.sentinel


# ─── context-manager protocol (dunders on the type) ─────────────────
def test_sync_context_manager_returns_self():                     # assertion 6
    fake = FakeSyncStream()
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    with wrapped as s:
        assert s is wrapped


def test_async_context_manager_returns_self():                    # assertion 7
    fake = FakeAsyncStream()
    wrapped = enforcer._wrap_mode_a_async_stream(fake, "openai", _Sess(), False)

    async def go():
        async with wrapped as s:
            assert s is wrapped

    _run(go())


def test_sync_exit_closes_underlying():                           # assertion 8
    fake = FakeSyncStream()
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    with wrapped:
        pass
    assert fake.closed is True


def test_async_exit_closes_underlying():                          # assertion 9
    fake = FakeAsyncStream()
    wrapped = enforcer._wrap_mode_a_async_stream(fake, "openai", _Sess(), False)

    async def go():
        async with wrapped:
            pass

    _run(go())
    assert fake.closed is True


# ─── close-through on early abandonment (leak fix) ──────────────────
def test_sync_early_abandonment_closes_underlying():             # assertion 10
    fake = FakeSyncStream([{"choices": [{"delta": {"content": "a"}}]}] * 3)
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    first = next(wrapped)                    # __next__ required (repro line 48)
    assert first == {"choices": [{"delta": {"content": "a"}}]}
    wrapped.close()
    assert fake.closed is True


def test_async_early_abandonment_closes_underlying():            # assertion 11
    fake = FakeAsyncStream([{"choices": [{"delta": {"content": "a"}}]}] * 3)
    wrapped = enforcer._wrap_mode_a_async_stream(fake, "openai", _Sess(), False)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()

    _run(go())
    assert fake.closed is True


# ─── close never raises — ANY step ────────────────
def test_sync_close_swallows_underlying_close_raise():           # assertion 12a
    class RaisingClose(FakeSyncStream):
        def close(self):
            raise RuntimeError("underlying close boom")

    fake = RaisingClose()
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    next(wrapped)
    wrapped.close()  # must NOT raise


def test_sync_close_swallows_metering_gen_close_raise():          # assertion 12b
    def _gen_raise_on_close():
        try:
            yield 1
        finally:
            raise RuntimeError("metering gen finally boom")

    g = _gen_raise_on_close()
    next(g)  # suspend at yield so close() drives the finally
    proxy = enforcer._ModeAStreamProxy(FakeSyncStream(), g)
    proxy.close()  # must NOT raise despite the gen's finally raising


def test_async_close_swallows_underlying_aclose_raise():          # assertion 12a-async
    class RaisingAclose(FakeAsyncStream):
        async def aclose(self):
            raise RuntimeError("underlying aclose boom")

    fake = RaisingAclose()
    wrapped = enforcer._wrap_mode_a_async_stream(fake, "openai", _Sess(), False)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()  # must NOT raise

    _run(go())


def test_async_close_swallows_metering_gen_close_raise():         # assertion 12b-async
    async def _agen_raise_on_close():
        try:
            yield 1
        finally:
            raise RuntimeError("async metering gen finally boom")

    async def go():
        ag = _agen_raise_on_close()
        await ag.__anext__()  # suspend at yield
        proxy = enforcer._ModeAAsyncStreamProxy(FakeAsyncStream(), ag)
        await proxy.aclose()  # must NOT raise

    _run(go())


def test_sync_close_on_stream_missing_close():                   # assertion 13
    fake = NoCloseStream([{"choices": [{"delta": {"content": "a"}}]}] * 2)
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    next(wrapped)
    wrapped.close()          # no AttributeError escapes
    with wrapped:            # __exit__ path also safe
        pass


def test_async_close_on_stream_missing_aclose():                 # assertion 13-async
    fake = NoAcloseAsyncStream([{"choices": [{"delta": {"content": "a"}}]}] * 2)
    wrapped = enforcer._wrap_mode_a_async_stream(fake, "openai", _Sess(), False)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()  # no AttributeError escapes

    _run(go())


# ─── close-through drives the metering generator's finally (flush) ──
def test_sync_abandonment_flush_fires_once(monkeypatch):         # assertion 14
    spy = MagicMock()
    monkeypatch.setattr(enforcer, "_flush_deferred_spans", spy)
    fake = FakeSyncStream([{"choices": [{"delta": {"content": "a"}}]}] * 3)
    wrapped = enforcer._wrap_mode_a_sync_stream(fake, "openai", _Sess(), False)
    next(wrapped)          # partially consumed
    wrapped.close()        # GeneratorExit -> metering gen finally -> flush
    assert spy.call_count == 1
    assert fake.closed is True   # BOTH: metering flush AND underlying close


def test_async_abandonment_flush_fires_once(monkeypatch):        # assertion 15
    spy = MagicMock()
    monkeypatch.setattr(enforcer, "_flush_deferred_spans", spy)
    fake = FakeAsyncStream([{"choices": [{"delta": {"content": "a"}}]}] * 3)
    wrapped = enforcer._wrap_mode_a_async_stream(fake, "openai", _Sess(), False)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()

    _run(go())
    assert spy.call_count == 1
    assert fake.closed is True   # BOTH


# ─── metering preserved (must-preserve behavior) ────────────────────
def test_full_drain_flushes_once(monkeypatch):                   # assertion 16
    spy = MagicMock()
    monkeypatch.setattr(enforcer, "_flush_deferred_spans", spy)
    chunks = [{"choices": [{"delta": {"content": c}}]} for c in ("a", "b", "c")]
    wrapped = enforcer._wrap_mode_a_sync_stream(
        FakeSyncStream(chunks), "openai", _Sess(), False)
    assert list(wrapped) == chunks
    assert spy.call_count == 1


def test_suppress_usage_chunk_under_context_manager():           # assertion 17
    content = {"choices": [{"delta": {"content": "hi"}}]}
    usage_only = {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                  "choices": []}
    wrapped = enforcer._wrap_mode_a_sync_stream(
        FakeSyncStream([content, usage_only]), "openai", _Sess(), False,
        suppress_usage_chunk=True)
    seen = []
    with wrapped as s:
        for c in s:
            seen.append(c)
    assert content in seen
    assert usage_only not in seen     # usage-only terminal chunk stripped


def test_bug_b_sync_underlying_async_consumer_bridge():          # assertion 18
    wrapped = enforcer._wrap_mode_a_async_stream(
        SyncOnlyStream(["1", "2"]), "cohere", _Sess(), False)
    assert _run(_drain_async(wrapped)) == ["1", "2"]


def test_tap_failopen_preserved(monkeypatch):                    # assertion 19
    def _boom(*a, **k):
        raise RuntimeError("chunk shape drifted")

    monkeypatch.setattr(enforcer, "_mode_a_accumulate", _boom)
    chunks = [{"choices": [{"delta": {"content": c}}]} for c in ("p", "q", "r")]
    wrapped = enforcer._wrap_mode_a_sync_stream(
        FakeSyncStream(chunks), "openai", _Sess(), False)
    assert list(wrapped) == chunks   # no RuntimeError escapes


# ─── mid-stream provider error (hot-area) ───────────────────────────
def test_sync_mid_stream_error_reraised_verbatim(monkeypatch):   # assertion 20
    emit = MagicMock()
    monkeypatch.setattr(enforcer, "_emit_call_failure_log", emit)
    sentinel = ValueError("provider stream died")

    class MidError:
        def __init__(self):
            self._it = iter([{"choices": [{"delta": {"content": "a"}}]}])

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise sentinel

    sess = _Sess()
    wrapped = enforcer._wrap_mode_a_sync_stream(MidError(), "openai", sess, False)
    with pytest.raises(ValueError) as ei:
        list(wrapped)
    assert ei.value is sentinel           # verbatim, not SDK-wrapped
    emit.assert_called_once()
    assert sess._defer_telemetry is False  # restored to prev_defer


def test_async_mid_stream_error_reraised_verbatim(monkeypatch):  # assertion 21
    emit = MagicMock()
    monkeypatch.setattr(enforcer, "_emit_call_failure_log", emit)
    sentinel = ValueError("async provider stream died")

    class MidErrorAsync:
        def __init__(self):
            self._n = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            self._n += 1
            if self._n == 1:
                return {"choices": [{"delta": {"content": "a"}}]}
            raise sentinel

    sess = _Sess()
    wrapped = enforcer._wrap_mode_a_async_stream(MidErrorAsync(), "openai", sess, False)
    with pytest.raises(ValueError) as ei:
        _run(_drain_async(wrapped))
    assert ei.value is sentinel
    emit.assert_called_once()
    assert sess._defer_telemetry is False
