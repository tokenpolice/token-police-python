"""Anthropic `.stream()` finalizers must forward the verbatim usage shape.

Audit S1 (HIGH, under-count): the sync/async stream context-manager finalizers
folded cache WRITES (cache_creation_input_tokens) into cached_tokens and sent
NO usage= kwarg, so client.log_sync synthesised an openai_compatible shape. The
server then priced cache writes at the (~10x cheaper) cache-READ rate and
subtracted cached from prompt — under-billing cache writes ~4-10x.

Fix: when the usage object serializes, forward
`usage={"shape": "anthropic_messages", "raw": <verbatim>}` and set
cached_tokens = reads-only (writes priced via raw). Gated: if serialization
fails, fall back to EXACTLY the prior behavior (cached_tokens = read + write,
no usage=), because an anthropic_messages shape with empty raw maps to
all-zeros at the server — worse than today.

All fakes — no anthropic SDK involved. Mirrors the harness in
test_latency_metrics.py (TestAnthropicStreamMgrLatency).
"""
import asyncio
import json
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import context, enforcer


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — asyncio.run
    unsets the main-thread event loop, breaking later get_event_loop()-based
    tests in the same pytest process."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeAnthropicStream:
    """Stand-in for anthropic MessageStream: iterable events + final message."""

    def __init__(self, events, final_msg):
        self._events = events
        self._final = final_msg

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final

    @property
    def text_stream(self):
        for ev in self._events:
            t = getattr(getattr(ev, "delta", None), "text", None)
            if t:
                yield t


class _FakeAnthropicAsyncStream(_FakeAnthropicStream):
    def __aiter__(self):
        async def _agen():
            for ev in self._events:
                yield ev
        return _agen()

    async def get_final_message(self):
        return self._final


class _FakeMgr:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *a):
        return False


class _FakeAsyncMgr(_FakeMgr):
    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *a):
        return False


def _anthropic_events():
    return [
        NS(type="message_start"),
        NS(type="content_block_start"),
        NS(type="content_block_delta", delta=NS(text="Hel")),
        NS(type="content_block_delta", delta=NS(text="lo")),
        NS(type="message_stop"),
    ]


def _cache_priming_final_message():
    """Cache-priming turn: input_tokens EXCLUDES cache (Anthropic semantics),
    cache_creation_input_tokens carries the write, and a nested `cache_creation`
    object splits the 5m/1h ephemeral buckets."""
    return NS(
        model="claude-3-5-sonnet",
        content=[NS(type="text", text="Hello")],
        usage=NS(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=200,
            cache_creation_input_tokens=1000,
            cache_creation=NS(
                ephemeral_5m_input_tokens=1000,
                ephemeral_1h_input_tokens=0,
            ),
        ),
    )


class _CaptureTP:
    """Captures the kwargs of the single log_sync the finalizer emits."""

    def __init__(self):
        self.captured = {}

    def log_sync(self, **kw):
        self.captured.update(kw)


class TestAnthropicStreamUsageShape(unittest.TestCase):

    # ── 1. sync cache-priming forwards anthropic_messages ──
    def test_sync_forwards_anthropic_messages_shape(self):
        tp = _CaptureTP()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(_anthropic_events(),
                                          _cache_priming_final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                list(stream)

        kw = tp.captured
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        raw = kw["usage"]["raw"]
        self.assertEqual(raw["input_tokens"], 100)
        self.assertEqual(raw["cache_creation_input_tokens"], 1000)
        self.assertEqual(raw["cache_read_input_tokens"], 200)
        # Nested 5m/1h split survives the json round-trip.
        self.assertEqual(raw["cache_creation"]["ephemeral_5m_input_tokens"], 1000)
        # cached_tokens carries READS only — writes are priced via raw.
        self.assertEqual(kw["cached_tokens"], 200)
        self.assertEqual(kw["input_tokens"], 100)
        self.assertEqual(kw["output_tokens"], 50)
        # The forwarded payload must be JSON-serializable end to end.
        json.dumps(kw["usage"])

    # ── 2. async variant of (1) ──
    def test_async_forwards_anthropic_messages_shape(self):
        tp = _CaptureTP()
        wrapper = enforcer._AnthropicAsyncStreamMgrWrapper(
            _FakeAsyncMgr(_FakeAnthropicAsyncStream(_anthropic_events(),
                                                    _cache_priming_final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )

        async def _run():
            async with wrapper as stream:
                async for _ in stream:
                    pass

        with mock.patch.object(enforcer, "get_client", return_value=tp):
            _run_coro(_run())

        kw = tp.captured
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        raw = kw["usage"]["raw"]
        self.assertEqual(raw["input_tokens"], 100)
        self.assertEqual(raw["cache_creation_input_tokens"], 1000)
        self.assertEqual(raw["cache_creation"]["ephemeral_5m_input_tokens"], 1000)
        self.assertEqual(kw["cached_tokens"], 200)
        json.dumps(kw["usage"])

    # ── 3. serialization failure → gated fallback to today's behavior ──
    def test_serialization_failure_falls_back_to_prior_behavior(self):
        tp = _CaptureTP()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(_anthropic_events(),
                                          _cache_priming_final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        # Force the helper's serialization to raise → it must return None →
        # finalizer must NOT emit an anthropic_messages shape (empty raw would
        # map to all-zeros at the server).
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "_as_dict",
                                  side_effect=RuntimeError("boom")):
            with wrapper as stream:
                list(stream)

        kw = tp.captured
        # log_sync still fired (row not lost).
        self.assertIn("model", kw)
        # Fallback: no usage shape forwarded.
        self.assertIsNone(kw.get("usage"))
        # Fallback: writes folded back into cached_tokens (read + write).
        self.assertEqual(kw["cached_tokens"], 1200)

    # ── 4. Golden Rule: usage=None must not raise, logs fallback semantics ──
    def test_usage_none_does_not_raise(self):
        tp = _CaptureTP()
        final = NS(model="claude-3-5-sonnet",
                   content=[NS(type="text", text="Hello")],
                   usage=None)
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(_anthropic_events(), final)),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                list(stream)

        kw = tp.captured
        self.assertIn("model", kw)
        self.assertIsNone(kw.get("usage"))
        self.assertEqual(kw["cached_tokens"], 0)
        self.assertEqual(kw["input_tokens"], 0)
        self.assertEqual(kw["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()


def _tool_use_final_message(tool_id="toolu_stream_1", name="escalate"):
    """Final message of a streamed turn that requested a tool call."""
    content = [NS(type="text", text="Hello")]
    if tool_id:
        content.append(NS(type="tool_use", id=tool_id, name=name, input={}))
    return NS(
        model="claude-3-5-sonnet",
        content=content,
        usage=NS(input_tokens=10, output_tokens=5,
                 cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class _RaisingFinalStream(_FakeAnthropicStream):
    def get_final_message(self):
        raise RuntimeError("stream died")


class _RaisingFinalAsyncStream(_FakeAnthropicAsyncStream):
    async def get_final_message(self):
        raise RuntimeError("stream died")


class TestAnthropicStreamPendingToolIds(unittest.TestCase):
    """`client.messages.stream(...)` bypasses .create and logs by hand, so
    neither pending-id producer ever ran — 0/31 streamed tool rows in the
    2026-07-26 run carried an id while the non-streamed arm was 23/23."""

    def setUp(self):
        self.session = context.TPSession()
        self._token = context._current_session.set(self.session)
        self.addCleanup(context._current_session.reset, self._token)

    def _sync_wrapper(self, stream):
        return enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(stream),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )

    def _async_wrapper(self, stream):
        return enforcer._AnthropicAsyncStreamMgrWrapper(
            _FakeAsyncMgr(stream),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )

    def test_sync_stashes_tool_call_ids(self):
        tp = _CaptureTP()
        wrapper = self._sync_wrapper(
            _FakeAnthropicStream(_anthropic_events(), _tool_use_final_message()))
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                list(stream)
        self.assertEqual(
            context._pop_pending_tool_call_id("escalate"), "toolu_stream_1")

    def test_async_stashes_tool_call_ids(self):
        tp = _CaptureTP()
        wrapper = self._async_wrapper(
            _FakeAnthropicAsyncStream(_anthropic_events(),
                                      _tool_use_final_message("toolu_async_1")))

        async def _run():
            async with wrapper as stream:
                async for _ in stream:
                    pass

        with mock.patch.object(enforcer, "get_client", return_value=tp):
            _run_coro(_run())
        self.assertEqual(
            context._pop_pending_tool_call_id("escalate"), "toolu_async_1")

    def test_no_tool_streamed_turn_clears_stale_ids(self):
        """REPLACE semantics — matches the non-streamed path."""
        tp = _CaptureTP()
        context.set_pending_tool_calls([{"id": "stale", "name": "escalate"}])
        wrapper = self._sync_wrapper(
            _FakeAnthropicStream(_anthropic_events(),
                                 _tool_use_final_message(tool_id=None)))
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                list(stream)
        self.assertEqual(context._pop_pending_tool_call_id("escalate"), "")

    def test_get_final_message_raising_leaves_stash_untouched_and_never_throws(self):
        """Golden rule: a broken stream must not throw at `with`-exit, and must
        not clobber the stash (the early return sits above the stash)."""
        tp = _CaptureTP()
        context.set_pending_tool_calls([{"id": "prior", "name": "escalate"}])
        wrapper = self._sync_wrapper(
            _RaisingFinalStream(_anthropic_events(), None))
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                list(stream)
        self.assertEqual(context._pop_pending_tool_call_id("escalate"), "prior")

    def test_async_get_final_message_raising_never_throws(self):
        tp = _CaptureTP()
        wrapper = self._async_wrapper(
            _RaisingFinalAsyncStream(_anthropic_events(), None))

        async def _run():
            async with wrapper as stream:
                async for _ in stream:
                    pass

        with mock.patch.object(enforcer, "get_client", return_value=tp):
            _run_coro(_run())  # must not raise

    def test_tool_row_carries_the_streamed_id_end_to_end(self):
        """Producer → stash → @tp.tool row, the path the run measured."""
        import token_police as tp_mod
        from token_police import state

        logged = []

        class _FakeToolClient:
            def log_sync(self, **kwargs):
                logged.append(kwargs)

        tp = _CaptureTP()
        wrapper = self._sync_wrapper(
            _FakeAnthropicStream(_anthropic_events(),
                                 _tool_use_final_message("toolu_e2e", "escalate")))
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                list(stream)

        with mock.patch.object(state, "get_client", return_value=_FakeToolClient()):
            @tp_mod.tool()
            def escalate(reason):
                return "ok"

            escalate("angry customer")

        self.assertEqual(logged[0]["tool"]["call_id"], "toolu_e2e")
