"""P3 / P4(part) — pydantic_ai's manual `_log_pydantic_ai` emitter (the SOLE
carrier for this seam — there is no `_emit_call_failure_log` fallback here,
see FIX_PLAN §2.2(b)) must ship the `reroute_rejected` observation
`_apply_reroute` mints for an unappliable (body-less) call shape, on BOTH the
`request` (non-stream) and `request_stream` (async context manager) arms —
neither captured an obs key at all pre-fix.

Fix under test (FIX_PLAN §2.2(b)):
  1. `_make_pydantic_ai_request_wrapper`'s wrapper captures
     `_obs_key = _state.get_current_obs_key()` right after the check.
  2. `_PydanticAIAsyncStreamMgr.__aenter__` captures `self._obs_key` the same
     way (mirrors `_AnthropicStreamMgrWrapper`).
  3. `_log_pydantic_ai` gained `observations=`/`obs_key=` params, resolves the
     key via the `_OBS_KEY_CURRENT` sentinel + `_resolve_obs_key`, and drains
     INSIDE the function (own try/except, fail-open) — both directions
     (success AND failure) ride this one emitter, so the internal drain fixes
     both at once and a destructive drain makes double emission structurally
     impossible (FIX_PLAN §2.2(b)4).

Every assertion reads `observations` off the CAPTURED `log_sync(**kwargs)`
call — not `state.drain_observations()` directly.

Harness mirrors tests/test_pydantic_ai_is_streaming.py (`_PydanticAIAsyncStreamMgr`
/ `_make_pydantic_ai_request_wrapper` direct invocation, `_capture_pydantic_ai_*`
mocked out) combined with tests/test_reroute_unappliable_shape.py's
`_init_client` (`tp.init()` + stubbed `check`) so `_run_async_check` runs for
REAL and `_apply_reroute` actually mints.
"""
import asyncio
import unittest
from contextlib import contextmanager
from types import SimpleNamespace as NS
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as _state
from token_police.context import TPSession, _current_session, _in_pydantic_ai


REROUTE_RESULT = {
    "status": "allowed",
    "reroute": {
        "mode": "enforce",
        "model": "claude-haiku-4-5-20251001",
        "provider": "anthropic",
        "rule_id": "rule_rr",
    },
}


def _init_client(firewall, check_result):
    client = tp.init(api_key="tp_sk_test_rr_pydantic_ai", firewall=firewall,
                     deployment="serverless")
    client.check_sync = MagicMock(return_value=check_result)
    client.check = AsyncMock(return_value=check_result)
    client.log_sync = MagicMock()
    return client


@contextmanager
def _pinned_session(**kwargs):
    """See test_reroute_rejected_success_emission_llamaindex.py's twin."""
    sess = TPSession(**kwargs)
    sess._deferred_spans = []
    sess._defer_telemetry = False
    tok = _current_session.set(sess)
    guard = _in_pydantic_ai.set(False)
    try:
        yield sess
    finally:
        try:
            _in_pydantic_ai.reset(guard)
        except Exception:
            _in_pydantic_ai.set(False)
        _current_session.reset(tok)


def _fake_response():
    return NS(
        model_name="claude-haiku-4-5-20251001",
        provider_name="anthropic",
        usage=NS(input_tokens=803, output_tokens=124),
    )


def _fake_model():
    return NS(model_name="claude-haiku-4-5-20251001", system="anthropic")


def _capture_patches():
    """Prompt/response composition parsing needs real pydantic_ai message
    shapes we don't build here — neutered exactly like
    test_pydantic_ai_is_streaming.py does."""
    return (
        mock.patch.object(_enforcer, "_capture_pydantic_ai_prompt_at"),
        mock.patch.object(_enforcer, "_capture_pydantic_ai_response_at"),
    )


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

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ── P3a. request() (non-stream) success ships the rejection ──────

class TestP3RequestSuccess(_Base):
    def test_request_success_ships_reroute_rejected_observation(self):
        client = _init_client("enforce", REROUTE_RESULT)
        response = _fake_response()
        model = _fake_model()

        async def _original(_self, *a, **k):
            return response

        wrapper = _enforcer._make_pydantic_ai_request_wrapper(_original)

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                p1, p2 = _capture_patches()
                with p1, p2:
                    return await wrapper(model)

        out = self._run(_drive())

        self.assertIs(out, response)
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")
        self.assertEqual(obs[0]["rule_id"], "rule_rr")
        self.assertNotEqual((payload.get("call_outcome") or {}).get("status"), "failed")


# ── P3b. request_stream() (__aenter__/__aexit__ mgr) success ships ─

class TestP3StreamSuccess(_Base):
    def _make_mgr(self, stream, model=None):
        class _Mgr:
            async def __aenter__(_self):
                return stream

            async def __aexit__(_self, *a):
                return False

        return _enforcer._PydanticAIAsyncStreamMgr(_Mgr(), model or _fake_model(), messages=[])

    def test_stream_success_ships_reroute_rejected_observation(self):
        client = _init_client("enforce", REROUTE_RESULT)

        class _FakeStream:
            def get(self):
                return _fake_response()

        wrapper = self._make_mgr(_FakeStream())

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                p1, p2 = _capture_patches()
                with p1, p2:
                    async with wrapper as s:
                        self.assertIsInstance(s, _FakeStream)

        self._run(_drive())

        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")

    # ── Sibling isolation: keyed by the captured self._obs_key ────
    def test_sibling_stream_keyed_observation_is_not_claimed(self):
        """Two concurrent stream managers on DIFFERENT sessions, each under
        its own live reroute directive — each `self._obs_key` must claim only
        its OWN mint; neither steals the other's."""
        client = _init_client("enforce", REROUTE_RESULT)

        gate = asyncio.Event()

        class _FakeStreamA:
            def get(self):
                return NS(model_name="claude-haiku-4-5-20251001",
                          provider_name="anthropic",
                          usage=NS(input_tokens=10, output_tokens=5))

        class _FakeStreamB:
            def get(self):
                return NS(model_name="gpt-4o", provider_name="openai",
                          usage=NS(input_tokens=20, output_tokens=8))

        class _MgrA:
            async def __aenter__(_self):
                await gate.wait()
                return _FakeStreamA()

            async def __aexit__(_self, *a):
                return False

        class _MgrB:
            async def __aenter__(_self):
                return _FakeStreamB()

            async def __aexit__(_self, *a):
                return False

        model_a = NS(model_name="claude-haiku-4-5-20251001", system="anthropic")
        model_b = NS(model_name="gpt-4o", system="openai")
        wrapper_a = _enforcer._PydanticAIAsyncStreamMgr(_MgrA(), model_a, messages=[])
        wrapper_b = _enforcer._PydanticAIAsyncStreamMgr(_MgrB(), model_b, messages=[])

        async def run_a():
            with _pinned_session(workflow_name="wf-a"):
                p1, p2 = _capture_patches()
                with p1, p2:
                    async with wrapper_a:
                        pass

        async def run_b():
            with _pinned_session(workflow_name="wf-b"):
                p1, p2 = _capture_patches()
                with p1, p2:
                    async with wrapper_b:
                        pass

        async def _drive():
            # Start A's __aenter__ — it captures its own obs key from ITS
            # check, then blocks on the gate before returning the stream, so
            # B's own check+capture+emit runs fully in between.
            task_a = asyncio.ensure_future(run_a())
            await asyncio.sleep(0.01)
            await run_b()
            gate.set()
            await task_a

        self._run(_drive())

        self.assertEqual(client.log_sync.call_count, 2)
        by_model = {c.kwargs["model"]: c.kwargs for c in client.log_sync.call_args_list}
        self.assertIn("claude-haiku-4-5-20251001", by_model)
        self.assertIn("gpt-4o", by_model)
        obs_a = by_model["claude-haiku-4-5-20251001"].get("observations")
        obs_b = by_model["gpt-4o"].get("observations")
        self.assertEqual(len(obs_a), 1)
        self.assertEqual(len(obs_b), 1)


# ── P4 (part) — pydantic_ai failure paths: one row, one copy ─────

class TestP4FailurePathUnchanged(_Base):
    def test_request_failure_emits_exactly_one_row_one_copy(self):
        """pydantic_ai's request() failure rides the SAME `_log_pydantic_ai`
        emitter as success — there is no separate `_emit_call_failure_log`
        fallback on this seam (FIX_PLAN §2.2(b)4), so double emission is
        structurally impossible: one call, one emitter invocation."""
        client = _init_client("enforce", REROUTE_RESULT)
        boom = RuntimeError("provider 500")
        model = _fake_model()

        async def _original(_self, *a, **k):
            raise boom

        wrapper = _enforcer._make_pydantic_ai_request_wrapper(_original)

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                p1, p2 = _capture_patches()
                with p1, p2:
                    with self.assertRaises(RuntimeError):
                        await wrapper(model)

        self._run(_drive())

        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertEqual(payload["call_outcome"]["status"], "failed")
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")

    def test_stream_failure_emits_exactly_one_row_one_copy(self):
        client = _init_client("enforce", REROUTE_RESULT)

        class _FakeStream:
            def get(self):
                return _fake_response()

        class _Mgr:
            async def __aenter__(_self):
                return _FakeStream()

            async def __aexit__(_self, *a):
                return False

        wrapper = _enforcer._PydanticAIAsyncStreamMgr(_Mgr(), _fake_model(), messages=[])

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                p1, p2 = _capture_patches()
                with p1, p2:
                    with self.assertRaises(ValueError):
                        async with wrapper:
                            raise ValueError("customer stream consumption blew up")

        self._run(_drive())

        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertEqual(payload["call_outcome"]["status"], "failed")
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)


# ── P7-equivalent — no-op reroute: nothing minted, nothing shipped ─

class TestNoopReroute(_Base):
    def test_dry_run_directive_ships_no_observations(self):
        client = _init_client("enforce", {
            "status": "allowed",
            "reroute": {"mode": "dry_run", "model": "claude-haiku-4-5-20251001",
                        "provider": "anthropic", "rule_id": "rule_rr"},
        })
        response = _fake_response()
        model = _fake_model()

        async def _original(_self, *a, **k):
            return response

        wrapper = _enforcer._make_pydantic_ai_request_wrapper(_original)

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                p1, p2 = _capture_patches()
                with p1, p2:
                    return await wrapper(model)

        self._run(_drive())

        self.assertEqual(client.log_sync.call_count, 1)
        self.assertFalse(client.log_sync.call_args.kwargs.get("observations"))
        self.assertEqual(_state.drain_observations(), [])


if __name__ == "__main__":
    unittest.main()
