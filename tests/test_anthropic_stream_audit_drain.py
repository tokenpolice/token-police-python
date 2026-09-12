"""Anthropic `.stream()` finalizers must forward the firewall audit payload.

`_AnthropicStreamMgrWrapper._finalize` / `_AnthropicAsyncStreamMgrWrapper.
_afinalize` hand-roll their /log call and passed NEITHER `observations=` nor
`local_decision=`. Consequence: `reroute_rejected` observations pushed by the
pre-flight were stranded in the global queue (zero REROUTE_REJECTED audit rows
for anthropic-protocol streamed calls) and an applied local reroute's
`session._local_decision` never reached /log (silent unaudited
REQUEST_REROUTED).

Fix: the canonical `_log_manual` drain block, placed AFTER the early returns so
a broken stream never destroys observations without logging them.

All fakes — no anthropic SDK involved. Mirrors the harness in
test_anthropic_stream_usage_shape.py.
"""
import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import context, enforcer
from token_police import state


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


class _RaisingFinalStream(_FakeAnthropicStream):
    def get_final_message(self):
        raise RuntimeError("stream died")


class _RaisingFinalAsyncStream(_FakeAnthropicAsyncStream):
    async def get_final_message(self):
        raise RuntimeError("stream died")


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
        NS(type="content_block_delta", delta=NS(text="Hi")),
        NS(type="message_stop"),
    ]


def _final_message():
    return NS(
        model="claude-3-5-sonnet",
        content=[NS(type="text", text="Hi")],
        usage=NS(input_tokens=10, output_tokens=5,
                 cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class _CaptureTP:
    """Captures the kwargs of every log_sync the finalizer emits."""

    def __init__(self):
        self.calls = []

    def log_sync(self, **kw):
        self.calls.append(kw)

    @property
    def captured(self):
        return self.calls[-1] if self.calls else {}


_REJECTED_OBS = {
    "type": "reroute_rejected",
    "rule_id": "rule_f9",
    "reason": "target_model_unavailable",
    "from_model": "claude-3-5-sonnet",
    "to_model": "claude-3-5-haiku",
}

_LOCAL_DECISION = {
    "action": "REROUTE",
    "rule_id": "rule_f9",
    "from_model": "claude-3-5-sonnet",
    "to_model": "claude-3-5-haiku",
    "applied": True,
}


class TestAnthropicStreamAuditDrain(unittest.TestCase):

    def setUp(self):
        # Open a real session so `get_current_session()` inside __enter__
        # resolves to the SAME object the pre-flight would stash on.
        self.session = context.TPSession()
        self._token = context._current_session.set(self.session)
        self.addCleanup(context._current_session.reset, self._token)
        # Never inherit/leak queue state across tests. _observations_set also
        # clears the per-call obs key contextvar, so this test's manual
        # push_observation() calls queue UNTAGGED entries — claimed by the
        # wrapper's keyed drain regardless of which key its own check mints.
        state._observations_set([])
        self.addCleanup(state.drain_observations)

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

    def _drive_sync(self, wrapper, tp):
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                list(stream)

    def _drive_async(self, wrapper, tp):
        async def _run():
            async with wrapper as stream:
                async for _ in stream:
                    pass
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            _run_coro(_run())

    # ── 1. sync: audit payload reaches /log and the sources are cleared ──
    def test_sync_forwards_observations_and_local_decision(self):
        tp = _CaptureTP()
        state.push_observation(dict(_REJECTED_OBS))

        # The sync manager does NOT re-mint its obs key in __enter__ (its
        # check ran immediately before construction, in the same context —
        # see the wrapper's __init__ comment); construct first, then seed
        # under its captured key (None here, per setUp's reset) — the keyed
        # equivalent of the old `session._local_decision = {...}` seed.
        wrapper = self._sync_wrapper(
            _FakeAnthropicStream(_anthropic_events(), _final_message()))
        enforcer._stash_local_decision_entry(
            self.session, dict(_LOCAL_DECISION), wrapper._obs_key)

        self._drive_sync(wrapper, tp)

        kw = tp.captured
        self.assertEqual(kw.get("observations"), [_REJECTED_OBS])
        self.assertEqual(kw.get("local_decision"), _LOCAL_DECISION)
        # Both sources drained — nothing can leak onto a later row.
        self.assertEqual(state.drain_observations(), [])
        self.assertFalse(getattr(self.session, "_local_decisions", None))

    # ── 2. async variant of (1) ──
    def test_async_forwards_observations_and_local_decision(self):
        tp = _CaptureTP()
        state.push_observation(dict(_REJECTED_OBS))

        wrapper = self._async_wrapper(
            _FakeAnthropicAsyncStream(_anthropic_events(), _final_message()))

        # Unlike the sync manager, __aenter__ RE-CAPTURES self._obs_key (its
        # check runs there, not at construction) — so the seed must happen
        # AFTER entering, once the real key is known, and BEFORE the drive
        # consumes the stream (which runs the finalize/claim).
        async def _run():
            async with wrapper as stream:
                enforcer._stash_local_decision_entry(
                    self.session, dict(_LOCAL_DECISION), wrapper._obs_key)
                async for _ in stream:
                    pass

        with mock.patch.object(enforcer, "get_client", return_value=tp):
            _run_coro(_run())

        kw = tp.captured
        self.assertEqual(kw.get("observations"), [_REJECTED_OBS])
        self.assertEqual(kw.get("local_decision"), _LOCAL_DECISION)
        self.assertEqual(state.drain_observations(), [])
        self.assertFalse(getattr(self.session, "_local_decisions", None))

    # ── 3. clean call → None, not [] (`pending_observations or None`) ──
    def test_sync_no_observations_sends_none_not_empty_list(self):
        tp = _CaptureTP()
        wrapper = self._sync_wrapper(
            _FakeAnthropicStream(_anthropic_events(), _final_message()))
        self._drive_sync(wrapper, tp)

        kw = tp.captured
        self.assertIn("observations", kw)
        self.assertIsNone(kw["observations"])
        self.assertIsNone(kw["local_decision"])

    def test_async_no_observations_sends_none_not_empty_list(self):
        tp = _CaptureTP()
        wrapper = self._async_wrapper(
            _FakeAnthropicAsyncStream(_anthropic_events(), _final_message()))
        self._drive_async(wrapper, tp)

        kw = tp.captured
        self.assertIn("observations", kw)
        self.assertIsNone(kw["observations"])
        self.assertIsNone(kw["local_decision"])

    # ── 4. early return → no log AND no destructive drain ──
    def test_sync_get_final_message_failure_preserves_observations(self):
        """The drain sits BELOW the early returns: a stream that dies must not
        swallow the observation without logging it — it stays queued for the
        next /log. Must also never throw at `with`-exit (golden rule).

        NOTE: `session._local_decision` (the old flat slot) is never written
        by any code path any more, so asserting equality against it would be
        vacuously true no matter what the finalize actually does — assert
        against the keyed store (`session._local_decisions`) instead, which
        is the thing the finalize would have claimed from had it not
        early-returned.
        """
        tp = _CaptureTP()
        state.push_observation(dict(_REJECTED_OBS))

        wrapper = self._sync_wrapper(
            _RaisingFinalStream(_anthropic_events(), None))
        enforcer._stash_local_decision_entry(
            self.session, dict(_LOCAL_DECISION), wrapper._obs_key)

        self._drive_sync(wrapper, tp)

        self.assertEqual(tp.calls, [])
        self.assertEqual(state.drain_observations(), [_REJECTED_OBS])
        self.assertEqual(len(self.session._local_decisions), 1)
        self.assertEqual(self.session._local_decisions[-1]["ld"], _LOCAL_DECISION)

    def test_async_get_final_message_failure_preserves_observations(self):
        tp = _CaptureTP()
        state.push_observation(dict(_REJECTED_OBS))

        wrapper = self._async_wrapper(
            _RaisingFinalAsyncStream(_anthropic_events(), None))

        async def _run():
            async with wrapper as stream:
                enforcer._stash_local_decision_entry(
                    self.session, dict(_LOCAL_DECISION), wrapper._obs_key)
                async for _ in stream:
                    pass

        with mock.patch.object(enforcer, "get_client", return_value=tp):
            _run_coro(_run())

        self.assertEqual(tp.calls, [])
        self.assertEqual(state.drain_observations(), [_REJECTED_OBS])
        self.assertEqual(len(self.session._local_decisions), 1)
        self.assertEqual(self.session._local_decisions[-1]["ld"], _LOCAL_DECISION)

    # ── 5. golden rule: a hostile session attribute must not break the call ──
    def test_drain_failure_never_throws_and_still_logs(self):
        tp = _CaptureTP()
        wrapper = self._sync_wrapper(
            _FakeAnthropicStream(_anthropic_events(), _final_message()))

        with mock.patch.object(enforcer._state, "drain_observations",
                               side_effect=RuntimeError("boom")):
            self._drive_sync(wrapper, tp)

        kw = tp.captured
        self.assertEqual(kw.get("model"), "claude-3-5-sonnet")
        self.assertIsNone(kw["observations"])
        self.assertIsNone(kw["local_decision"])


if __name__ == "__main__":
    unittest.main()
