"""P1 / P2 / P4(part) / P6 / P7 — LlamaIndex STREAMING success emitters (Node
twin: tests/rerouteRejectedSuccessEmission.test.ts) must ship the
`reroute_rejected` observation `_apply_reroute` mints for an unappliable
(body-less) call shape — not just mint it into the module-global queue and
leave it there.

Per FIX_PLAN §2.2: the non-stream `chat`/`achat` arms already drain via
`_flush_deferred_spans` (run #20 data: the 26 silent rows were 100%
streaming) — out of scope here, untouched by the fix. This file covers ONLY
the streaming call sites the fix touched:

  - `_guard_sync_stream_li` (installed as `stream_chat`, kind="stream")
  - `li_async`'s inner `_drain()` (installed as `astream_chat`, kind="async"
    — the LIVE astream_chat path, `hasattr(result, "__aiter__")`)

(The old kind="astream" guard was never registered by any target and has been
deleted — real `astream_chat` is `async def`, so the "async" kind above is the
only astream_chat path.)

The suite-wide blind spot this bug exposed: every PRE-EXISTING rejection test
asserts against `state.drain_observations()` directly with `client.log_sync`
mocked to a no-op — minted, never verified SHIPPED. Every assertion below
instead reads `observations` off the CAPTURED `log_sync(**kwargs)` call.

Harness mirrors tests/test_llamaindex_stream_latency.py's `_install_sync` /
`_install_async` pattern, combined with tests/test_reroute_unappliable_shape.py's
`_init_client` (`tp.init()` + stubbed `check`/`check_sync`) so
`_run_sync_check` / `_run_async_check` run for REAL and `_apply_reroute`
actually mints — nothing here neutralizes the pre-flight the way
test_llamaindex_stream_latency.py does.
"""
import asyncio
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as _state
from token_police.context import TPSession, _current_session, _in_llamaindex


REROUTE_RESULT = {
    "status": "allowed",
    "reroute": {
        "mode": "enforce",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "rule_id": "rule_rr",
    },
}


def _init_client(firewall, check_result):
    """Init a client in `firewall` mode with check_sync + check (async)
    mocked to return `check_result`. Mirrors
    test_reroute_unappliable_shape.py's `_init_client`."""
    client = tp.init(api_key="tp_sk_test_rr_li_stream", firewall=firewall,
                     deployment="serverless")
    client.check_sync = MagicMock(return_value=check_result)
    client.check = AsyncMock(return_value=check_result)
    client.log_sync = MagicMock()
    return client


def _chunk(delta="hi"):
    return NS(delta=delta, message=NS(additional_kwargs={}), raw=None)


@contextmanager
def _pinned_session(**kwargs):
    """Pin one session directly on the contextvar — NOT `tp.session()`, which
    opens its own real OTel "agent" structural span and logs a SECOND row
    when the `with` block exits (unrelated to the call under test). Mirrors
    tests/test_llamaindex_stream_latency.py's `_env` fixture."""
    sess = TPSession(**kwargs)
    sess._deferred_spans = []
    sess._defer_telemetry = False
    tok = _current_session.set(sess)
    guard = _in_llamaindex.set(False)
    try:
        yield sess
    finally:
        try:
            _in_llamaindex.reset(guard)
        except Exception:
            _in_llamaindex.set(False)
        _current_session.reset(tok)


class _Base(unittest.TestCase):
    def setUp(self):
        _state.reset_pack()
        try:
            _state.drain_observations()
        except Exception:
            pass

    def tearDown(self):
        _state.reset_pack()
        tp.uninstrument()


# ── P1. Sync stream_chat success ships the rejection ─────────────

class TestP1SyncStreamSuccess(_Base):
    def test_stream_chat_success_ships_reroute_rejected_observation(self):
        client = _init_client("enforce", REROUTE_RESULT)

        def stream_chat(self, *a, **k):
            yield _chunk("Hel")
            yield _chunk("lo")

        cls = type("FakeOpenAI", (), {"stream_chat": stream_chat, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")

        with _pinned_session(workflow_name="wf"):
            out = list(cls().stream_chat())

        self.assertEqual(len(out), 2)  # the customer's stream is untouched
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")
        self.assertEqual(obs[0]["rule_id"], "rule_rr")
        self.assertNotEqual(payload.get("call_outcome", {}) or {}, {"status": "failed"})


# ── P2. Async astream_chat success ships the rejection (li_async _drain) ──

class TestP2AsyncStreamSuccess(_Base):
    def test_astream_chat_success_ships_reroute_rejected_observation(self):
        client = _init_client("enforce", REROUTE_RESULT)

        class _AsyncStream:
            def __aiter__(self):
                async def _gen():
                    yield _chunk("Hel")
                    await asyncio.sleep(0)
                    yield _chunk("lo")
                return _gen()

        async def astream_chat(self, *a, **k):
            return _AsyncStream()

        cls = type("FakeOpenAIAsync", (), {"astream_chat": astream_chat, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(cls, "astream_chat", cls.astream_chat, "async")

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                gen = await cls().astream_chat()
                return [c async for c in gen]

        out = asyncio.run(_drive())

        self.assertEqual(len(out), 2)
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")


# ── N3-equivalent — sibling isolation across two sessions ────────

class TestSiblingIsolation(_Base):
    def test_two_concurrent_streams_on_different_sessions_never_cross_ship(self):
        """Mirrors the Node N3 case: an in-flight stream (paused mid-flight)
        and a second call that completes fully while the first is paused —
        each row must ship only its OWN observation."""
        client = _init_client("enforce", REROUTE_RESULT)

        gate = asyncio.Event()

        class _AsyncStreamA:
            def __aiter__(self):
                async def _gen():
                    yield _chunk("a1")
                    await gate.wait()
                    yield _chunk("a2")
                return _gen()

        async def astream_chat_a(self, *a, **k):
            return _AsyncStreamA()

        class _AsyncStreamB:
            def __aiter__(self):
                async def _gen():
                    yield _chunk("b1")
                return _gen()

        async def astream_chat_b(self, *a, **k):
            return _AsyncStreamB()

        ClsA = type("FakeAnthropicA", (), {"astream_chat": astream_chat_a, "model": "claude-3-haiku"})
        ClsB = type("FakeOpenAIB", (), {"astream_chat": astream_chat_b, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(ClsA, "astream_chat", ClsA.astream_chat, "async")
        _enforcer._set_llamaindex_wrapper(ClsB, "astream_chat", ClsB.astream_chat, "async")

        async def _drive():
            async def run_a():
                with _pinned_session(workflow_name="wf-a"):
                    gen = await ClsA().astream_chat()
                    return [c async for c in gen]

            async def run_b():
                with _pinned_session(workflow_name="wf-b"):
                    gen = await ClsB().astream_chat()
                    return [c async for c in gen]

            task_a = asyncio.ensure_future(run_a())
            # Let stream A reach its first yield (paused on `gate`).
            await asyncio.sleep(0.01)
            # Run B to full completion while A is still paused.
            out_b = await run_b()
            gate.set()
            out_a = await task_a
            return out_a, out_b

        out_a, out_b = asyncio.run(_drive())

        self.assertEqual(len(out_a), 2)
        self.assertEqual(len(out_b), 1)
        self.assertEqual(client.log_sync.call_count, 2)

        by_model = {c.kwargs["model"]: c.kwargs for c in client.log_sync.call_args_list}
        self.assertIn("claude-3-haiku", by_model)
        self.assertIn("gpt-4o", by_model)
        obs_a = by_model["claude-3-haiku"].get("observations")
        obs_b = by_model["gpt-4o"].get("observations")
        self.assertEqual(len(obs_a), 1)
        self.assertEqual(len(obs_b), 1)


# ── P4 (part) — mid-stream failure: one row, one copy of the observation ──

class TestP4FailurePathUnchanged(_Base):
    def test_mid_stream_failure_emits_exactly_one_row_one_copy(self):
        client = _init_client("enforce", REROUTE_RESULT)
        boom = RuntimeError("upstream 500")

        def stream_chat(self, *a, **k):
            yield _chunk("Hel")
            raise boom

        cls = type("FakeOpenAI", (), {"stream_chat": stream_chat, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")

        caught = None
        with _pinned_session(workflow_name="wf"):
            try:
                for _ in cls().stream_chat():
                    pass
            except RuntimeError as e:
                caught = e

        self.assertIs(caught, boom)
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertEqual(payload["call_outcome"]["status"], "failed")
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")

    def test_zero_pull_synchronous_failure_emits_exactly_one_row_one_copy(self):
        """The li_stream wrapper's OWN except (not _guard_sync_stream_li's) —
        a provider that raises before ever returning a stream."""
        client = _init_client("enforce", REROUTE_RESULT)
        boom = RuntimeError("auth boom")

        def stream_chat(self, *a, **k):
            raise boom

        cls = type("FakeOpenAI", (), {"stream_chat": stream_chat, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")

        caught = None
        with _pinned_session(workflow_name="wf"):
            try:
                cls().stream_chat()
            except RuntimeError as e:
                caught = e

        self.assertIs(caught, boom)
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertEqual(payload["call_outcome"]["status"], "failed")
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)


# ── P6. local_decision fence — observations-only, never a claim ──

class TestP6LocalDecisionFence(_Base):
    def test_untagged_decision_ships_nowhere_and_survives_the_call(self):
        client = _init_client("enforce", REROUTE_RESULT)

        def stream_chat(self, *a, **k):
            yield _chunk("hi")

        cls = type("FakeOpenAI", (), {"stream_chat": stream_chat, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")

        untagged = {
            "outcome": "rerouted",
            "rule_id": "someone_elses_rule",
            "reroute": {"from": {"model": "x"}, "to": {"model": "y"}},
        }
        with _pinned_session(workflow_name="wf") as sess:
            # Simulates a genuinely concurrent degraded-path sibling's stash
            # that never got tagged to its own key (mirrors
            # test_local_decision_keyed_store.py's untagged-fallback tests).
            _enforcer._stash_local_decision_entry(sess, untagged, None)

            list(cls().stream_chat())

            self.assertEqual(client.log_sync.call_count, 1)
            payload = client.log_sync.call_args.kwargs
            # Observations-only: the mint still ships…
            self.assertEqual(len(payload.get("observations") or []), 1)
            # …but no local_decision at all — this emitter never claims one.
            self.assertIsNone(payload.get("local_decision"))

            # Untouched — still claimable by whoever actually owns it.
            claimed = _enforcer._claim_local_decision(sess, "any-later-key")
            self.assertEqual(claimed, untagged)


# ── P7. No-op reroute — nothing minted, nothing shipped ───────────

class TestP7NoopReroute(_Base):
    def test_dry_run_directive_never_reaches_appliability(self):
        client = _init_client("enforce", {
            "status": "allowed",
            "reroute": {"mode": "dry_run", "model": "gpt-4o-mini",
                        "provider": "openai", "rule_id": "rule_rr"},
        })

        def stream_chat(self, *a, **k):
            yield _chunk("hi")

        cls = type("FakeOpenAI", (), {"stream_chat": stream_chat, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")

        with _pinned_session(workflow_name="wf"):
            list(cls().stream_chat())

        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertFalse(payload.get("observations"))
        self.assertEqual(_state.drain_observations(), [])

    def test_no_reroute_directive_at_all(self):
        client = _init_client("enforce", {"status": "allowed"})

        def stream_chat(self, *a, **k):
            yield _chunk("hi")

        cls = type("FakeOpenAI", (), {"stream_chat": stream_chat, "model": "gpt-4o"})
        _enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")

        with _pinned_session(workflow_name="wf"):
            list(cls().stream_chat())

        self.assertEqual(client.log_sync.call_count, 1)
        self.assertFalse(client.log_sync.call_args.kwargs.get("observations"))


if __name__ == "__main__":
    unittest.main()
