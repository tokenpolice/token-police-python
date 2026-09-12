"""A live ENFORCE REROUTE directive reaching ``_apply_reroute`` on a call
shape with no appliable request body (framework/hint-only paths: LangChain/
LlamaIndex, framework embeddings, Bedrock Converse envelope, native
OpenRouter envelope) used to silently return "noop" — no observation, so the
audit trail had REROUTE_DIRECTIVE_ISSUED with no resolution.

``_apply_reroute`` now runs an appliability check FIRST (before the
serving_unverified / cross-provider guards): kwargs that are not a dict
carrying a top-level "model" key pushes a ``reroute_rejected`` observation
with ``rejection_reason: "unappliable_call_shape"`` and returns "rejected".
The ``model_hint`` param supplies ``reroute.from.model`` on that observation
(never a swap target — kwargs are never mutated).

Mirrors token-police-node/tests/rerouteUnappliableShape.test.ts.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as tp_state
from token_police.enforcer import _apply_reroute


_OMIT = object()  # sentinel: "don't set the rule_name key at all"


def _directive(model, provider="openai", rule_name=_OMIT):
    """Build a `{"reroute": {...}}` result. `rule_name=_OMIT` (default)
    means "omit the key entirely"; pass None/""/a string to set it
    explicitly."""
    reroute = {
        "mode": "enforce",
        "model": model,
        "provider": provider,
        "rule_id": "rule_rr",
    }
    if rule_name is not _OMIT:
        reroute["rule_name"] = rule_name
    return {"reroute": reroute}


class TestApplyRerouteUnappliableShape(unittest.TestCase):
    def setUp(self):
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    # ── 1/2: None kwargs ────────────────────────────────────────────
    def test_none_kwargs_with_hint_rejected_one_observation(self):
        status = _apply_reroute(
            _directive("gpt-4o-mini"), None, "openai", model_hint="gpt-4o"
        )
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        o = obs[0]
        self.assertEqual(o["outcome"], "reroute_rejected")
        self.assertEqual(o["rejection_reason"], "unappliable_call_shape")
        self.assertEqual(o["reroute"]["from"]["model"], "gpt-4o")
        self.assertEqual(o["reroute"]["to"]["model"], "gpt-4o-mini")
        self.assertEqual(o["reroute"]["to"]["provider"], "openai")

    def test_none_kwargs_no_hint_from_model_empty_never_raises(self):
        status = _apply_reroute(_directive("gpt-4o-mini"), None, "openai")
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["reroute"]["from"]["model"], "")

    # ── 3: dict kwargs without a top-level "model" key ─────────────
    def test_bedrock_converse_like_kwargs_without_model_key(self):
        kwargs = {"modelId": "anthropic.claude-3-haiku", "input": {"messages": []}}
        snapshot = dict(kwargs)
        status = _apply_reroute(
            _directive("gpt-4o-mini"),
            kwargs,
            "openai",
            model_hint="anthropic.claude-3-haiku",
        )
        self.assertEqual(status, "rejected")
        self.assertEqual(kwargs, snapshot)  # no mutation, no invented "model" key
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["reroute"]["from"]["model"], "anthropic.claude-3-haiku")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")

    # ── 5: appliability runs before the cross-provider guard ───────
    def test_appliability_runs_before_cross_provider_guard(self):
        status = _apply_reroute(
            _directive("claude-3-haiku", provider="anthropic"),
            None,
            "openai",
            model_hint="gpt-4o",
        )
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")

    # ── 7: per-rule DRY_RUN + None kwargs → noop, no observation ───
    def test_dry_run_directive_with_none_kwargs_is_noop(self):
        status = _apply_reroute(
            {"reroute": {"mode": "dry_run", "model": "gpt-4o-mini",
                         "provider": "openai", "rule_id": "rule_rr"}},
            None,
            "openai",
            model_hint="gpt-4o",
        )
        self.assertEqual(status, "noop")
        self.assertEqual(tp_state.drain_observations(), [])

    # ── 8b: golden rule — non-dict kwargs never raises ─────────────
    def test_non_dict_kwargs_string_rejected_without_raising(self):
        status = _apply_reroute(
            _directive("gpt-4o-mini"), "not-a-dict", "openai", model_hint="gpt-4o"
        )
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")
        # A non-dict kwargs has no `.get`, so from.model can only be the hint.
        self.assertEqual(obs[0]["reroute"]["from"]["model"], "gpt-4o")

    def test_non_dict_kwargs_int_rejected_without_raising(self):
        status = _apply_reroute(_directive("gpt-4o-mini"), 42, "openai")
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")

    def test_none_kwargs_never_raises(self):
        try:
            status = _apply_reroute(_directive("gpt-4o-mini"), None, "openai")
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"_apply_reroute raised: {exc}")
        self.assertEqual(status, "rejected")


class TestUnappliableShapeRejectionRuleName(unittest.TestCase):
    """Omit-when-empty convention (never `None`) mirrors the appliable-path
    reroute_rejected coverage in test_reroute_rule_name.py."""

    def setUp(self):
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    def test_carries_rule_name_when_present(self):
        status = _apply_reroute(
            _directive("gpt-4o-mini", rule_name="Send free users to a cheaper model"),
            None,
            "openai",
            model_hint="gpt-4o",
        )
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(obs[0]["rule_name"], "Send free users to a cheaper model")
        self.assertEqual(obs[0]["rule_id"], "rule_rr")

    def test_omits_rule_name_when_absent(self):
        _apply_reroute(_directive("gpt-4o-mini"), None, "openai", model_hint="gpt-4o")
        obs = tp_state.drain_observations()
        self.assertNotIn("rule_name", obs[0])
        self.assertEqual(obs[0]["rule_id"], "rule_rr")

    def test_omits_rule_name_when_empty_string(self):
        _apply_reroute(
            _directive("gpt-4o-mini", rule_name=""), None, "openai", model_hint="gpt-4o"
        )
        obs = tp_state.drain_observations()
        self.assertNotIn("rule_name", obs[0])

    def test_omits_rule_name_when_none(self):
        _apply_reroute(
            _directive("gpt-4o-mini", rule_name=None), None, "openai", model_hint="gpt-4o"
        )
        obs = tp_state.drain_observations()
        self.assertNotIn("rule_name", obs[0])


def _init_client(firewall, check_result):
    """Init a client in `firewall` mode with check_sync + check (async)
    mocked to return `check_result`. Returns the client."""
    client = tp.init(api_key="tp_sk_test_unappliable", firewall=firewall,
                     deployment="serverless")
    client.check_sync = MagicMock(return_value=check_result)
    client.check = AsyncMock(return_value=check_result)
    client.log_sync = MagicMock()
    return client


class TestGateRemovalRunCheckWithNoneKwargs(unittest.TestCase):
    """Proves the six now-unconditional `_apply_reroute` call sites inside
    `_run_sync_check` / `_run_async_check` actually reach `_apply_reroute`
    with kwargs=None (the `isinstance(kwargs, dict)` gates around those call
    sites were removed) instead of short-circuiting before the reject."""

    def setUp(self):
        tp_state.reset_pack()
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    def tearDown(self):
        tp_state.reset_pack()
        tp.uninstrument()

    def test_run_sync_check_kwargs_none_pushes_one_reroute_rejected(self):
        _init_client("enforce", {
            "status": "allowed",
            "reroute": {"mode": "enforce", "model": "gpt-4o-mini",
                        "provider": "openai", "rule_id": "rule_rr"},
        })
        with tp.session(name="t"):
            _enforcer._run_sync_check(kwargs=None, provider="openai", model_hint="gpt-4o")
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")
        self.assertEqual(obs[0]["reroute"]["from"]["model"], "gpt-4o")

    def test_run_async_check_kwargs_none_pushes_one_reroute_rejected(self):
        _init_client("enforce", {
            "status": "allowed",
            "reroute": {"mode": "enforce", "model": "gpt-4o-mini",
                        "provider": "openai", "rule_id": "rule_rr"},
        })

        async def _run():
            with tp.session(name="t"):
                await _enforcer._run_async_check(kwargs=None, provider="openai",
                                                  model_hint="gpt-4o")

        asyncio.run(_run())
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")

    def test_run_sync_check_can_reroute_false_kwargs_none_zero_observations(self):
        # B7 regression fence: can_reroute=False must never reach
        # _apply_reroute at all, so no reroute_rejected observation either.
        _init_client("enforce", {
            "status": "allowed",
            "reroute": {"mode": "enforce", "model": "gpt-4o-mini",
                        "provider": "openai", "rule_id": "rule_rr"},
        })
        with tp.session(name="t"):
            _enforcer._run_sync_check(kwargs=None, provider="openai",
                                      can_reroute=False, model_hint="gpt-4o")
        self.assertEqual(tp_state.drain_observations(), [])


if __name__ == "__main__":
    unittest.main()
