"""Async Anthropic `.stream()` pre-flight check must run on the event loop.

Invariant: the budget/anomaly pre-flight for the async
`client.messages.stream(...)` context manager runs inside the wrapper's
`__aenter__` (awaited on the caller's event loop), NOT as a blocking synchronous
HTTP round-trip on the running loop thread. A denied call raises
TokenPoliceBlockedError when the stream is entered (`async with`), and the
underlying provider manager is never entered — so the provider request never
fires. Every other failure fails open. On a blocked entry, no half-initialized
state is left behind (the OTel-suppression flag is set only after the check
passes).

All fakes — no anthropic SDK involved. The instrumentation entry point
(`_instrument_anthropic_stream`) is exercised end-to-end against a fake
AsyncMessages patched into sys.modules, so the patched `.stream()` call itself is
under test.
"""
import asyncio
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police.exceptions import TokenPoliceBlockedError
from token_police.context import (
    _anthropic_stream_span_window,
    _current_session,
    TPSession,
)
from token_police.telemetry import TokenPoliceSpanProcessor


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — asyncio.run
    unsets the main-thread event loop, breaking later get_event_loop()-based
    tests in the same pytest process."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _events():
    return [
        NS(type="content_block_start"),
        NS(type="content_block_delta", delta=NS(text="Hel")),
        NS(type="content_block_delta", delta=NS(text="lo")),
        NS(type="message_stop"),
    ]


class _FakeAsyncStream:
    """Stand-in for anthropic AsyncMessageStream: async-iterable + final msg."""

    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def _agen():
            for ev in self._events:
                yield ev
        return _agen()

    async def get_final_message(self):
        return NS(model="claude-3-5-sonnet", content=[], usage=None)

    @property
    def text_stream(self):
        for ev in self._events:
            t = getattr(getattr(ev, "delta", None), "text", None)
            if t:
                yield t


class _SpyAsyncMgr:
    """Underlying manager whose __aenter__ MUST NOT run on a blocked call."""

    def __init__(self, stream):
        self._stream = stream
        self.enter_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return self._stream

    async def __aexit__(self, *a):
        return False


def _install_fake_anthropic(mgr):
    """Register a fake `anthropic.resources.messages` module with fresh
    Messages/AsyncMessages classes, run `_instrument_anthropic_stream()` to patch
    them, and return (async_client_instance, cleanup). The async client's
    `.stream(...)` yields `mgr` through the original (unpatched) body."""

    class Messages:
        def stream(self, *a, **k):
            # Sync path is not exercised here; must exist so the instrumenter
            # doesn't early-return before patching AsyncMessages.
            return NS()

    class AsyncMessages:
        def stream(self, *a, **k):
            return self._mgr

    messages_mod = types.ModuleType("anthropic.resources.messages")
    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages
    resources_mod = types.ModuleType("anthropic.resources")
    resources_mod.messages = messages_mod
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.resources = resources_mod

    names = ("anthropic", "anthropic.resources", "anthropic.resources.messages")
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules["anthropic"] = anthropic_mod
    sys.modules["anthropic.resources"] = resources_mod
    sys.modules["anthropic.resources.messages"] = messages_mod

    enforcer._instrument_anthropic_stream()

    client = AsyncMessages()
    client._mgr = mgr

    def cleanup():
        enforcer._originals.pop((Messages, "stream"), None)
        enforcer._originals.pop((AsyncMessages, "stream"), None)
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v

    return client, cleanup


_KW = dict(model="claude-3-5-sonnet",
           messages=[{"role": "user", "content": "hi"}])


class _Scope:
    def __init__(self, name):
        self.name = name


class _FakeSpan:
    """Writable attributes so on_start's set_attribute is observable —
    mirrors tests/test_anthropic_stream_enter_failure.py's fixture."""

    def __init__(self, scope="opentelemetry.instrumentation.anthropic"):
        self.attributes = {}
        self.name = "anthropic.chat"
        self.instrumentation_scope = _Scope(scope)
        self.context = NS(trace_id=0x1111, span_id=0x2222)
        self.parent = None
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestAnthropicAsyncStreamCheck(unittest.TestCase):

    # ── 1. blocked verdict: .stream() returns cleanly; entry raises; the
    # underlying manager is never entered (no provider request) ──
    def test_blocked_raises_on_enter_underlying_mgr_never_entered(self):
        mgr = _SpyAsyncMgr(_FakeAsyncStream(_events()))
        client, cleanup = _install_fake_anthropic(mgr)

        # The wrapper now threads kwargs/provider/serving_unverified into
        # the pre-flight, so the stub must accept them.
        async def _blocked(**_kw):
            raise TokenPoliceBlockedError("budget exhausted")

        try:
            with mock.patch.object(enforcer, "_run_async_check", new=_blocked):
                async def _run():
                    m = client.stream(**_KW)
                    # Constructing the manager must NOT raise.
                    self.assertIsInstance(
                        m, enforcer._AnthropicAsyncStreamMgrWrapper)
                    with self.assertRaises(TokenPoliceBlockedError):
                        async with m:
                            pass
                _run_coro(_run())
            self.assertEqual(mgr.enter_count, 0)
        finally:
            cleanup()

    # ── 2. allowed verdict: stream works end-to-end; async check awaited
    # exactly once; the blocking sync check is NEVER called (event-loop
    # regression pin). Pre-fix, the wrapper called _run_sync_check on the
    # loop thread and never awaited _run_async_check — so this test fails
    # before the fix on BOTH assertions. ──
    def test_allowed_uses_async_check_not_sync_check(self):
        mgr = _SpyAsyncMgr(_FakeAsyncStream(_events()))
        client, cleanup = _install_fake_anthropic(mgr)

        awaited = []

        async def _allowed(**kw):
            awaited.append(kw)

        try:
            with mock.patch.object(enforcer, "_run_async_check", new=_allowed), \
                    mock.patch.object(enforcer, "_run_sync_check") as sync_spy:
                async def _run():
                    got = []
                    async with client.stream(**_KW) as stream:
                        async for ev in stream:
                            got.append(ev)
                    return got
                got = _run_coro(_run())

            self.assertEqual(len(got), len(_events()))
            self.assertEqual(mgr.enter_count, 1)
            self.assertEqual(len(awaited), 1)
            # Real call context reaches the pre-flight (was bare).
            self.assertEqual(awaited[0]["kwargs"], _KW)
            self.assertEqual(awaited[0]["provider"], "anthropic")
            sync_spy.assert_not_called()
        finally:
            cleanup()

    # ── 3. non-block failure → fail open (stream proceeds normally) ──
    def test_non_block_exception_fails_open(self):
        mgr = _SpyAsyncMgr(_FakeAsyncStream(_events()))
        client, cleanup = _install_fake_anthropic(mgr)

        async def _boom(**_kw):
            raise RuntimeError("collector unreachable")

        try:
            with mock.patch.object(enforcer, "_run_async_check", new=_boom):
                async def _run():
                    got = []
                    async with client.stream(**_KW) as stream:
                        async for ev in stream:
                            got.append(ev)
                    return got
                got = _run_coro(_run())

            self.assertEqual(len(got), len(_events()))
            self.assertEqual(mgr.enter_count, 1)
        finally:
            cleanup()

    # ── 4. blocked entry leaves no stale suppression window on the session
    # (armed only after the check passes, else it would eat a later span) ──
    #
    # NOTE ON THE REWRITE: this test used to assert
    # ``getattr(sess, "_suppress_anthropic_otel_stream", False) is False`` —
    # the deleted one-shot session flag. That assertion now passes VACUOUSLY
    # (the attribute doesn't exist at all any more), so it is rewritten
    # against the replacement mechanism — the per-call
    # ``_anthropic_stream_span_window`` ContextVar (token_police/context.py)
    # — preserving the original intent: a blocked/failed entry must never
    # eat the NEXT anthropic span's telemetry.
    def test_blocked_does_not_leave_suppression_window(self):
        mgr = _SpyAsyncMgr(_FakeAsyncStream(_events()))
        client, cleanup = _install_fake_anthropic(mgr)

        sess = TPSession()
        token = _current_session.set(sess)

        # The wrapper now threads kwargs/provider/serving_unverified into
        # the pre-flight, so the stub must accept them.
        async def _blocked(**_kw):
            raise TokenPoliceBlockedError("budget exhausted")

        try:
            with mock.patch.object(enforcer, "_run_async_check", new=_blocked):
                async def _run():
                    with self.assertRaises(TokenPoliceBlockedError):
                        async with client.stream(**_KW):
                            pass
                _run_coro(_run())

            # The check raises before the W1 arm statement is even reached —
            # no window survives a blocked entry at all.
            self.assertIsNone(_anthropic_stream_span_window.get())

            # Driving the real span processor with a fresh fake anthropic
            # span right now must NOT stamp tp.suppress — nothing is armed
            # to claim it.
            span = _FakeSpan()
            TokenPoliceSpanProcessor().on_start(span)
            self.assertNotIn("tp.suppress", span.attributes)

            # The REAL behavioural proof, not just internal state: the VERY
            # NEXT call on the same client (now allowed) must still go
            # through the vendor manager and yield its real content — not be
            # silently swallowed the way a leaked suppress flag would have
            # eaten its telemetry.
            async def _allowed(**_kw):
                return None

            with mock.patch.object(enforcer, "_run_async_check", new=_allowed):
                async def _run2():
                    got = []
                    async with client.stream(**_KW) as stream:
                        async for ev in stream:
                            got.append(ev)
                    return got
                got = _run_coro(_run2())
            self.assertEqual(len(got), len(_events()))
            # The blocked call never entered the vendor manager; this is its
            # first and only successful entry.
            self.assertEqual(mgr.enter_count, 1)
        finally:
            _current_session.reset(token)
            cleanup()

    def test_control_armed_window_really_does_suppress(self):
        """Proves the negative assertions above can fail — i.e. they measure
        the window mechanism, not an unrelated no-op. This is the new
        mechanism's equivalent of the tripwire control in
        tests/test_anthropic_stream_enter_failure.py."""
        handle = enforcer.arm_anthropic_stream_span_window()
        try:
            span = _FakeSpan()
            TokenPoliceSpanProcessor().on_start(span)
            self.assertTrue(span.attributes.get("tp.suppress"))
        finally:
            enforcer.disarm_anthropic_stream_span_window(handle)
        self.assertIsNone(_anthropic_stream_span_window.get())


if __name__ == "__main__":
    unittest.main()
