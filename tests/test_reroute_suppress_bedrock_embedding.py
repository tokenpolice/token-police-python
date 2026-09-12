"""On the Bedrock-embedding InvokeModel path the pre-flight ``/check`` is fed
a THROWAWAY ``{"model": ...}`` kwargs dict; the real InvokeModel re-reads ``*args``
unchanged, so a reroute swap could never reach the provider. The two handlers now
pass ``can_reroute=False`` to ``_run_sync_check`` / ``_run_async_check``; the whole
rerouted branch header is gated on ``can_reroute`` so BOTH the ``_apply_reroute``
call AND the separate rerouted ``_stash_local_decision`` are suppressed (no phantom
``_tp_routing``). BLOCK/entity-block on the same path still enforces.

Mirrors token-police-node/tests/rerouteSuppressBedrockEmbedding.test.ts.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as tp_state
from token_police.context import get_current_session
from token_police.exceptions import TokenPoliceBlockedError


# Same-provider (bedrock) reroute so the cross-provider guard does NOT fire —
# isolates the can_reroute gate as the sole reason a swap is suppressed.
_BEDROCK_REROUTE_SNAPSHOT = {
    "schema_version": 1, "type": "snapshot", "version": 1,
    "tenant_id": "t", "project_id": "p", "ttl_seconds": 600, "loop_blocks": [],
    "directives": [
        {
            "id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
            "selector": {"match": None, "group_by": []},
            "reroute": {"from": {}, "to": {"provider": "bedrock",
                                           "model": "amazon.titan-embed-text-v2:0"}},
        }
    ],
}

_BLOCK_SNAPSHOT = {
    "schema_version": 1, "type": "snapshot", "version": 1,
    "tenant_id": "t", "project_id": "p", "ttl_seconds": 600, "loop_blocks": [],
    "directives": [
        {
            "id": "b1", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
            "selector": {"match": None, "group_by": []},
        }
    ],
}


def _init(firewall, check_result):
    """Init a client in `firewall` mode with check_sync + check (async) + log_sync
    mocked to return `check_result`. Returns the client."""
    client = tp.init(api_key="tp_sk_test_f26", firewall=firewall)
    client.check_sync = MagicMock(return_value=check_result)
    client.check = AsyncMock(return_value=check_result)
    client.log_sync = MagicMock()
    return client


class TestRerouteSuppressBedrockEmbedding(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()

    def tearDown(self):
        tp_state.reset_pack()
        tp.uninstrument()

    # ── A2: reroute suppressed, kwargs unchanged, no _tp_routing ──────────
    def test_state_a_reroute_suppressed_sync(self):
        _init("enforce", {"status": "allowed"})
        tp_state.apply_snapshot(_BEDROCK_REROUTE_SNAPSHOT)
        kwargs = {"model": "amazon.titan-embed-text-v1"}
        with tp.session(name="t"):
            _enforcer._run_sync_check(kwargs=kwargs, provider="bedrock",
                                      intent={"kind": "embedding"}, can_reroute=False)
            self.assertEqual(kwargs["model"], "amazon.titan-embed-text-v1")  # unchanged
            self.assertNotIn("_tp_routing", get_current_session().metadata)

    def test_state_b_reroute_suppressed_sync(self):
        # No pack → State-B fallback; reroute rides the /check result.
        _init("enforce", {"status": "allowed",
                          "reroute": {"mode": "enforce",
                                      "model": "amazon.titan-embed-text-v2:0",
                                      "provider": "bedrock", "rule_id": "rr"}})
        kwargs = {"model": "amazon.titan-embed-text-v1"}
        with tp.session(name="t"):
            _enforcer._run_sync_check(kwargs=kwargs, provider="bedrock",
                                      intent={"kind": "embedding"}, can_reroute=False)
            self.assertEqual(kwargs["model"], "amazon.titan-embed-text-v1")
            self.assertNotIn("_tp_routing", get_current_session().metadata)

    def test_state_a_reroute_suppressed_async(self):
        _init("enforce", {"status": "allowed"})
        tp_state.apply_snapshot(_BEDROCK_REROUTE_SNAPSHOT)
        kwargs = {"model": "amazon.titan-embed-text-v1"}

        async def _run():
            with tp.session(name="t"):
                await _enforcer._run_async_check(kwargs=kwargs, provider="bedrock",
                                                 intent={"kind": "embedding"}, can_reroute=False)
                self.assertEqual(kwargs["model"], "amazon.titan-embed-text-v1")
                self.assertNotIn("_tp_routing", get_current_session().metadata)

        asyncio.run(_run())

    # ── A4: BLOCK on the same path STILL enforces with can_reroute=False ──
    def test_block_still_raises_sync_state_a(self):
        _init("enforce", {"status": "blocked", "reason": "budget exceeded"})
        tp_state.apply_snapshot(_BLOCK_SNAPSHOT)
        with tp.session(name="t"):
            with self.assertRaises(TokenPoliceBlockedError):
                _enforcer._run_sync_check(kwargs={"model": "amazon.titan-embed-text-v1"},
                                          provider="bedrock", intent={"kind": "embedding"},
                                          can_reroute=False)

    def test_block_still_raises_sync_state_b(self):
        _init("enforce", {"status": "blocked", "reason": "budget exceeded"})
        with tp.session(name="t"):
            with self.assertRaises(TokenPoliceBlockedError):
                _enforcer._run_sync_check(kwargs={"model": "amazon.titan-embed-text-v1"},
                                          provider="bedrock", intent={"kind": "embedding"},
                                          can_reroute=False)

    def test_block_still_raises_async_state_b(self):
        _init("enforce", {"status": "blocked", "reason": "budget exceeded"})

        async def _run():
            with tp.session(name="t"):
                with self.assertRaises(TokenPoliceBlockedError):
                    await _enforcer._run_async_check(kwargs={"model": "amazon.titan-embed-text-v1"},
                                                     provider="bedrock", intent={"kind": "embedding"},
                                                     can_reroute=False)

        asyncio.run(_run())

    def test_reroute_local_but_check_blocked_still_raises(self):
        # Local decision rerouted, but /check says blocked → still blocks even with
        # can_reroute=False (falls through to State B block).
        _init("enforce", {"status": "blocked", "reason": "budget exceeded"})
        tp_state.apply_snapshot(_BEDROCK_REROUTE_SNAPSHOT)
        with tp.session(name="t"):
            with self.assertRaises(TokenPoliceBlockedError):
                _enforcer._run_sync_check(kwargs={"model": "amazon.titan-embed-text-v1"},
                                          provider="bedrock", intent={"kind": "embedding"},
                                          can_reroute=False)

    # ── A6: default can_reroute=True STILL reroutes (anti-over-suppression) ──
    def test_default_can_reroute_applies_sync_state_a(self):
        _init("enforce", {"status": "allowed"})
        tp_state.apply_snapshot(_BEDROCK_REROUTE_SNAPSHOT)
        kwargs = {"model": "amazon.titan-embed-text-v1"}
        with tp.session(name="t"):
            _enforcer._run_sync_check(kwargs=kwargs, provider="bedrock")  # default True
            self.assertEqual(kwargs["model"], "amazon.titan-embed-text-v2:0")  # swapped
            self.assertIn("_tp_routing", get_current_session().metadata)

    def test_default_can_reroute_applies_sync_state_b(self):
        _init("enforce", {"status": "allowed",
                          "reroute": {"mode": "enforce", "model": "gpt-4o-mini",
                                      "provider": "openai", "rule_id": "rr"}})
        kwargs = {"model": "gpt-4o"}
        with tp.session(name="t"):
            _enforcer._run_sync_check(kwargs=kwargs, provider="openai")  # default True
            self.assertEqual(kwargs["model"], "gpt-4o-mini")
            self.assertIn("_tp_routing", get_current_session().metadata)

    # ── A10 golden rule: can_reroute=False never raises on an allowed decision ──
    def test_can_reroute_false_never_raises_on_allowed(self):
        _init("enforce", {"status": "allowed"})
        tp_state.apply_snapshot(_BEDROCK_REROUTE_SNAPSHOT)
        with tp.session(name="t"):
            # Must NOT raise.
            _enforcer._run_sync_check(kwargs={"model": "amazon.titan-embed-text-v1"},
                                      provider="bedrock", intent={"kind": "embedding"},
                                      can_reroute=False)


if __name__ == "__main__":
    unittest.main()
