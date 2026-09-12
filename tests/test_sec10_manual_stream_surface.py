"""Regression: the MANUAL-path streaming wrappers `_wrap_sync_stream` /
`_wrap_async_stream` (mistral / litellm / cohere / groq / together / openrouter /
xai / cerebras / huggingface) must hand the customer an object at least as capable
as the provider's original stream — context-manager protocol, `.response`,
arbitrary attribute passthrough, and a `close()`/`aclose()` that closes the
underlying connection — NOT a bare generator. Early abandonment / CM exit /
explicit close must close-through to BOTH the underlying provider stream (leak
fix) AND the retained metering generator (its `finally` still runs the litellm
guard reset + conditional `_log_manual`). Every close step is fail-open:
nothing escapes into customer code. Mid-stream provider errors are re-raised
verbatim.

This is the defect class on the manual-path twins.
The fix is two return-line swaps reusing the proxies
(`_ModeAStreamProxy` / `_ModeAAsyncStreamProxy`); the `_gen()`/`_agen()` bodies
(tap, metering finally, litellm guard set/reset, `_log_manual`, bridge) are
unchanged. These tests fail against the pre-fix bare-generator implementation
(`generator`/`async_generator`, no `.response`, `TypeError` on `with`, underlying
never closed on abandonment) and pass post-fix. Encodes contract assertions 4-22.

Manual-path gotcha vs the metering `finally` calls **`_log_manual`**
(iteration-conditional on `last is not None`), NOT `_flush_deferred_spans`, and
resets the **`_in_litellm`** guard — so the flush/guard assertions spy those
symbols.
"""
from __future__ import annotations

import asyncio
import types
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from token_police import enforcer


# ─── chunk helpers (`_chunk_usage` reads getattr(chunk, "usage")) ─────
def _content_chunk(text="hi"):
    return SimpleNamespace(content=text)


def _usage_chunk(p=1, c=1):
    return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c))


# ─── fakes (mistral EventStream / litellm CustomStreamWrapper shape) ─
class FakeSyncStream:
    """Iterable + context manager + .response + close() — like a Mistral
    EventStream / litellm CustomStreamWrapper."""

    def __init__(self, chunks=None):
        self._chunks = list(chunks if chunks is not None else [_content_chunk()])
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
    """Async-iterable + .response + coroutine aclose()."""

    def __init__(self, chunks=None):
        self._chunks = list(chunks if chunks is not None else [_content_chunk()])
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
    """A plain sync stream ( underlying shape — sync underlying, async
    consumer)."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class NoCloseStream:
    """A stream missing close/__exit__ entirely (hostile input)."""

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
    def __init__(self):
        self._defer_telemetry = False
        self._call_outcome = None


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


_TS = datetime.now(timezone.utc)


def _wrap_sync(stream, provider="mistral", framework=None):
    return enforcer._wrap_sync_stream(
        stream, provider, _Sess(), {}, 0, "span", _TS,
        framework=framework, req_start_mono=None)


def _wrap_async(stream, provider="mistral", framework=None):
    return enforcer._wrap_async_stream(
        stream, provider, _Sess(), {}, 0, "span", _TS,
        framework=framework, req_start_mono=None)


@pytest.fixture(autouse=True)
def _reset_litellm_guard():
    """Clean baseline so guard-leak assertions observe only this test's writes."""
    token = enforcer._in_litellm.set(False)
    yield
    try:
        enforcer._in_litellm.reset(token)
    except Exception:
        enforcer._in_litellm.set(False)


@pytest.fixture
def _log_spy(monkeypatch):
    """Spy `_log_manual` and neutralize composition capture (its OTel side
    effects run BEFORE `_log_manual` in the same try-block, so it must not
    raise or the log call would be skipped)."""
    spy = MagicMock()
    monkeypatch.setattr(enforcer, "_log_manual", spy)
    monkeypatch.setattr(enforcer, "_capture_composition_at", MagicMock())
    return spy


# ─── surface restoration (defect leg A) ──────────────────────────────
def test_sync_returns_proxy_not_bare_generator():                  # assertion 4
    returned = _wrap_sync(FakeSyncStream())
    assert not isinstance(returned, types.GeneratorType)
    assert type(returned).__name__ == "_ModeAStreamProxy"


def test_async_returns_proxy_not_bare_generator():                 # assertion 4
    returned = _wrap_async(FakeAsyncStream())
    assert not isinstance(returned, types.AsyncGeneratorType)
    assert type(returned).__name__ == "_ModeAAsyncStreamProxy"


def test_response_delegation_sync():                               # assertion 5
    fake = FakeSyncStream()
    wrapped = _wrap_sync(fake)
    assert hasattr(wrapped, "response")
    assert wrapped.response is fake.response


def test_response_delegation_async():                              # assertion 5
    fake = FakeAsyncStream()
    wrapped = _wrap_async(fake)
    assert hasattr(wrapped, "response")
    assert wrapped.response is fake.response


def test_arbitrary_attribute_passthrough_sync():                  # assertion 6
    fake = FakeSyncStream()
    assert _wrap_sync(fake).sentinel is fake.sentinel


def test_arbitrary_attribute_passthrough_async():                 # assertion 6
    fake = FakeAsyncStream()
    assert _wrap_async(fake).sentinel is fake.sentinel


# ─── context-manager protocol restored (defect leg B) ───────────────
def test_sync_context_manager_returns_self():                     # assertion 7
    fake = FakeSyncStream()
    wrapped = _wrap_sync(fake)
    with wrapped as s:
        assert s is wrapped


def test_async_context_manager_returns_self():                    # assertion 8
    fake = FakeAsyncStream()
    wrapped = _wrap_async(fake)

    async def go():
        async with wrapped as s:
            assert s is wrapped

    _run(go())


def test_sync_exit_closes_underlying():                           # assertion 9
    fake = FakeSyncStream()
    wrapped = _wrap_sync(fake)
    with wrapped:
        pass
    assert fake.closed is True


def test_async_exit_closes_underlying():                          # assertion 10
    fake = FakeAsyncStream()
    wrapped = _wrap_async(fake)

    async def go():
        async with wrapped:
            pass

    _run(go())
    assert fake.closed is True


# ─── close-through on early abandonment (defect leg C, leak fix) ─────
def test_sync_early_abandonment_closes_underlying():             # assertion 11
    fake = FakeSyncStream([_content_chunk("a")] * 3)
    wrapped = _wrap_sync(fake)
    first = next(wrapped)
    assert first == _content_chunk("a")
    wrapped.close()
    assert fake.closed is True


def test_async_early_abandonment_closes_underlying():            # assertion 12
    fake = FakeAsyncStream([_content_chunk("a")] * 3)
    wrapped = _wrap_async(fake)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()

    _run(go())
    assert fake.closed is True


# ─── close on a stream missing close/aclose (hostile input, ) ──
def test_sync_close_on_stream_missing_close():                   # assertion 13
    fake = NoCloseStream([_content_chunk("a")] * 2)
    wrapped = _wrap_sync(fake)
    next(wrapped)
    wrapped.close()          # no AttributeError escapes
    with wrapped:            # __exit__ path also safe
        pass


def test_async_close_on_stream_missing_aclose():                # assertion 13
    fake = NoAcloseAsyncStream([_content_chunk("a")] * 2)
    wrapped = _wrap_async(fake)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()   # no AttributeError escapes

    _run(go())


# ─── abandonment metering drives the manual finally (spy _log_manual) ─
def test_sync_abandonment_logs_manual_once(_log_spy):            # assertion 14
    # usage chunk FIRST so `last is not None` after one next() -> finally logs.
    fake = FakeSyncStream([_usage_chunk(), _content_chunk("a"), _content_chunk("b")])
    wrapped = _wrap_sync(fake)
    next(wrapped)            # consume past the usage chunk (last set)
    wrapped.close()          # GeneratorExit -> manual finally -> _log_manual
    assert _log_spy.call_count == 1
    assert fake.closed is True    # BOTH: manual finally AND underlying close


def test_async_abandonment_logs_manual_once(_log_spy):          # assertion 15
    fake = FakeAsyncStream([_usage_chunk(), _content_chunk("a"), _content_chunk("b")])
    wrapped = _wrap_async(fake)

    async def go():
        async for _c in wrapped:
            break               # consumed the usage chunk (last set)
        await wrapped.aclose()

    _run(go())
    assert _log_spy.call_count == 1
    assert fake.closed is True    # BOTH


def test_sync_abandonment_before_usage_logs_nothing(_log_spy):  # assertion 14 (accepted)
    # Behavior-preserving: abandon BEFORE any usage chunk -> last is None ->
    # no _log_manual (matches the pre-fix bare-gen semantics). The finally still
    # ran (close-through reached it); it just had nothing to log.
    fake = FakeSyncStream([_content_chunk("a"), _content_chunk("b")])
    wrapped = _wrap_sync(fake)
    next(wrapped)
    wrapped.close()
    assert _log_spy.call_count == 0
    assert fake.closed is True


# ─── _in_litellm guard-leak on abandon (enforcement gap, ) ─────
def test_sync_litellm_guard_reset_on_abandon(_log_spy):         # assertion 16
    fake = FakeSyncStream([_usage_chunk(), _content_chunk("a")])
    wrapped = _wrap_sync(fake, framework="litellm")
    next(wrapped)                       # body ran -> _in_litellm.set(True)
    assert enforcer._in_litellm.get() is True   # guard held during iteration
    wrapped.close()                     # GeneratorExit -> finally -> set(False)
    assert enforcer._in_litellm.get() is False  # guard NOT leaked
    assert fake.closed is True


def test_async_litellm_guard_reset_on_abandon(_log_spy):        # assertion 16
    fake = FakeAsyncStream([_usage_chunk(), _content_chunk("a")])
    wrapped = _wrap_async(fake, framework="litellm")

    async def go():
        # observe INSIDE the task context (run_until_complete copies context,
        # so the async-gen's set()/reset() are only visible here).
        async for _c in wrapped:
            break
        held = enforcer._in_litellm.get()
        await wrapped.aclose()
        after = enforcer._in_litellm.get()
        return held, after

    held, after = _run(go())
    assert held is True                 # guard held during iteration
    assert after is False               # guard NOT leaked
    assert fake.closed is True


# ─── close never raises — ANY step ────────────────
def test_sync_close_swallows_underlying_close_raise():           # assertion 17
    class RaisingClose(FakeSyncStream):
        def close(self):
            raise RuntimeError("underlying close boom")

    fake = RaisingClose()
    wrapped = _wrap_sync(fake)
    next(wrapped)
    wrapped.close()          # must NOT raise


def test_sync_close_swallows_metering_gen_close_raise():         # assertion 17
    def _gen_raise_on_close():
        try:
            yield 1
        finally:
            raise RuntimeError("metering gen finally boom")

    g = _gen_raise_on_close()
    next(g)
    proxy = enforcer._ModeAStreamProxy(FakeSyncStream(), g)
    proxy.close()            # must NOT raise despite the gen's finally raising


def test_async_close_swallows_underlying_aclose_raise():         # assertion 17
    class RaisingAclose(FakeAsyncStream):
        async def aclose(self):
            raise RuntimeError("underlying aclose boom")

    fake = RaisingAclose()
    wrapped = _wrap_async(fake)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()   # must NOT raise

    _run(go())


def test_async_close_swallows_metering_gen_close_raise():        # assertion 17
    async def _agen_raise_on_close():
        try:
            yield 1
        finally:
            raise RuntimeError("async metering gen finally boom")

    async def go():
        ag = _agen_raise_on_close()
        await ag.__anext__()
        proxy = enforcer._ModeAAsyncStreamProxy(FakeAsyncStream(), ag)
        await proxy.aclose()     # must NOT raise

    _run(go())


# ─── double-close fires _log_manual exactly once (idempotency) ──────
def test_sync_double_close_logs_manual_once(_log_spy):          # assertion 18
    fake = FakeSyncStream([_usage_chunk(), _content_chunk("a")])
    wrapped = _wrap_sync(fake)
    next(wrapped)
    wrapped.close()
    wrapped.close()          # second close: gen already finalized -> no-op
    assert _log_spy.call_count == 1
    assert fake.closed is True


def test_async_double_close_logs_manual_once(_log_spy):         # assertion 18
    fake = FakeAsyncStream([_usage_chunk(), _content_chunk("a")])
    wrapped = _wrap_async(fake)

    async def go():
        async for _c in wrapped:
            break
        await wrapped.aclose()
        await wrapped.aclose()   # second aclose: no-op

    _run(go())
    assert _log_spy.call_count == 1
    assert fake.closed is True


def test_sync_drain_then_close_logs_manual_once(_log_spy):      # assertion 18
    fake = FakeSyncStream([_content_chunk("a"), _usage_chunk()])
    wrapped = _wrap_sync(fake)
    assert len(list(wrapped)) == 2   # full drain -> finally -> _log_manual once
    wrapped.close()                  # already exhausted -> no second log
    assert _log_spy.call_count == 1


# ─── close-before-iteration is clean (no leak, no spurious log) ─────
def test_sync_close_before_iteration(_log_spy):                # assertion 19
    fake = FakeSyncStream([_usage_chunk(), _content_chunk("a")])
    wrapped = _wrap_sync(fake, framework="litellm")
    wrapped.close()                  # never iterated
    assert _log_spy.call_count == 0          # body never entered, last unset
    assert enforcer._in_litellm.get() is False   # set(True) never ran -> no leak
    assert fake.closed is True               # underlying still closed


def test_async_close_before_iteration(_log_spy):               # assertion 19
    fake = FakeAsyncStream([_usage_chunk(), _content_chunk("a")])
    wrapped = _wrap_async(fake, framework="litellm")

    async def go():
        await wrapped.aclose()       # never iterated
        return enforcer._in_litellm.get()

    leaked = _run(go())
    assert _log_spy.call_count == 0
    assert leaked is False
    assert fake.closed is True


# ─── metering preserved (must-preserve behavior) ────────────────────
def test_full_drain_delivers_all_chunks_sync():                # assertion 20
    chunks = [_content_chunk("a"), _content_chunk("b"), _usage_chunk()]
    wrapped = _wrap_sync(FakeSyncStream(chunks))
    assert list(wrapped) == chunks


def test_tap_failopen_preserved_sync(monkeypatch):             # assertion 20
    def _boom(*a, **k):
        raise RuntimeError("chunk shape drifted")

    monkeypatch.setattr(enforcer, "_accumulate_stream_chunk", _boom)
    chunks = [_content_chunk("p"), _content_chunk("q"), _content_chunk("r")]
    wrapped = _wrap_sync(FakeSyncStream(chunks))
    assert list(wrapped) == chunks   # no RuntimeError escapes


def test_bug_b_sync_underlying_async_consumer_bridge():        # assertion 20
    wrapped = _wrap_async(SyncOnlyStream(["1", "2"]), provider="cohere")
    assert _run(_drain_async(wrapped)) == ["1", "2"]


# ─── mid-stream provider error re-raised verbatim (hot-area) ────────
def test_sync_mid_stream_error_reraised_verbatim(monkeypatch):  # assertion 21
    emit = MagicMock()
    monkeypatch.setattr(enforcer, "_emit_call_failure_log", emit)
    sentinel = ValueError("provider stream died")

    class MidError:
        def __init__(self):
            self._it = iter([_content_chunk("a")])

        def __iter__(self):
            return self

        def __next__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise sentinel

    wrapped = _wrap_sync(MidError())
    with pytest.raises(ValueError) as ei:
        list(wrapped)
    assert ei.value is sentinel          # verbatim, not SDK-wrapped
    emit.assert_called_once()


def test_async_mid_stream_error_reraised_verbatim(monkeypatch): # assertion 21
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
                return _content_chunk("a")
            raise sentinel

    wrapped = _wrap_async(MidErrorAsync())
    with pytest.raises(ValueError) as ei:
        _run(_drain_async(wrapped))
    assert ei.value is sentinel
    emit.assert_called_once()


# ─── _ResponsesRawStreamProxy.parse inherits the async proxy for free ─
def test_responses_raw_parse_inherits_proxy():                 # assertion 22
    inner_stream = FakeAsyncStream([_content_chunk("a")])

    class FakeRawResponse:
        def __init__(self):
            self.marker = object()

        async def parse(self, *a, **k):
            return inner_stream

    raw = FakeRawResponse()
    proxy = enforcer._ResponsesRawStreamProxy(
        raw, "openai_responses", _Sess(), {}, 0, "span", _TS, None,
        req_start_mono=None)

    async def go():
        parsed = await proxy.parse()
        # parsed is now a _ModeAAsyncStreamProxy, not a bare async_generator
        assert type(parsed).__name__ == "_ModeAAsyncStreamProxy"
        assert not isinstance(parsed, types.AsyncGeneratorType)
        assert parsed.response is inner_stream.response      # surface restored
        async with parsed as s:                              # CM restored
            async for _c in s:
                break
        assert inner_stream.closed is True                   # close-through

    _run(go())
