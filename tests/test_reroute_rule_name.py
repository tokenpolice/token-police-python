"""``rule_name`` must flow from the directive pack through the local
evaluator and synthetic reroute into ``_tp_routing``, and be omitted when
absent/empty (Node/Python shape parity — never ``null``).
"""

import unittest

import token_police as tp
from token_police import state as tp_state
from token_police.context import get_current_session
from token_police.enforcer import _apply_reroute
from token_police.local_evaluator import evaluate


def _reroute_pack(name=None, include_name_key=True):
    d = {
        "id": "rule_rr",
        "kind": "REROUTE",
        "mode": "enforce",
        "selector": {"match": None, "group_by": []},
        "reroute": {
            "from": None,
            "to": {"provider": "openai", "model": "gpt-4o-mini"},
        },
    }
    if include_name_key:
        d["name"] = name
    return {"directives": [d]}


class TestRerouteRuleName(unittest.TestCase):
    def test_local_eval_carries_rule_name_when_pack_has_name(self):
        res = evaluate(
            _reroute_pack("Send free users to a cheaper model"),
            {"paid_plan": "free"},
            {"provider": "openai", "model": "gpt-4o"},
        )
        self.assertEqual(res["decision"]["status"], "rerouted")
        self.assertEqual(res["decision"]["rule_id"], "rule_rr")
        self.assertEqual(
            res["decision"]["rule_name"], "Send free users to a cheaper model"
        )

    def test_local_eval_omits_rule_name_when_pack_has_no_name(self):
        res = evaluate(
            _reroute_pack(include_name_key=False),
            {},
            {"provider": "openai", "model": "gpt-4o"},
        )
        self.assertEqual(res["decision"]["status"], "rerouted")
        self.assertNotIn("rule_name", res["decision"])

    # Observations, not just the enforce decision: a DRY_RUN rule never produces
    # a decision, so the observation was the only carrier of the name — and it
    # shipped without one, which is why the routing feed rendered rule UUIDs.
    def test_would_reroute_observation_carries_rule_name(self):
        pack = _reroute_pack("Send free users to a cheaper model")
        pack["directives"][0]["mode"] = "dry_run"
        res = evaluate(pack, {"paid_plan": "free"}, {"provider": "openai", "model": "gpt-4o"})
        self.assertEqual(res["decision"]["status"], "allowed")
        self.assertEqual(len(res["observations"]), 1)
        self.assertEqual(res["observations"][0]["outcome"], "would_reroute")
        self.assertEqual(
            res["observations"][0]["rule_name"], "Send free users to a cheaper model"
        )

    def test_would_reroute_observation_omits_rule_name_when_absent(self):
        pack = _reroute_pack(include_name_key=False)
        pack["directives"][0]["mode"] = "dry_run"
        res = evaluate(pack, {}, {"provider": "openai", "model": "gpt-4o"})
        self.assertEqual(len(res["observations"]), 1)
        self.assertNotIn("rule_name", res["observations"][0])

    def test_reroute_rejected_observation_carries_rule_name(self):
        # Target provider != observed provider -> rejected, not applied.
        res = evaluate(
            _reroute_pack("Send free users to a cheaper model"),
            {},
            {"provider": "anthropic", "model": "claude-haiku-4-5"},
        )
        self.assertEqual(len(res["observations"]), 1)
        self.assertEqual(res["observations"][0]["outcome"], "reroute_rejected")
        self.assertEqual(
            res["observations"][0]["rule_name"], "Send free users to a cheaper model"
        )

    def test_would_block_observation_carries_rule_name(self):
        res = evaluate(
            {
                "directives": [
                    {
                        "id": "rule_b",
                        "kind": "UNCONDITIONAL_BLOCK",
                        "mode": "dry_run",
                        "name": "Block everything",
                        "selector": {"match": None, "group_by": []},
                    }
                ]
            },
            {},
            {"provider": "openai", "model": "gpt-4o"},
        )
        self.assertEqual(res["decision"]["status"], "allowed")
        self.assertEqual(len(res["observations"]), 1)
        self.assertEqual(res["observations"][0]["outcome"], "would_block")
        self.assertEqual(res["observations"][0]["rule_name"], "Block everything")

    def test_local_eval_omits_rule_name_when_pack_name_empty(self):
        res = evaluate(
            _reroute_pack(""),
            {},
            {"provider": "openai", "model": "gpt-4o"},
        )
        self.assertEqual(res["decision"]["status"], "rerouted")
        self.assertNotIn("rule_name", res["decision"])

    def test_apply_reroute_stashes_rule_name_when_present(self):
        kwargs = {"model": "gpt-4o"}
        with tp.session(name="t"):
            _apply_reroute(
                {
                    "reroute": {
                        "mode": "enforce",
                        "model": "gpt-4o-mini",
                        "provider": "openai",
                        "rule_id": "rule_rr",
                        "rule_name": "Send free users to a cheaper model",
                        "original": {"provider": "openai", "model": "gpt-4o"},
                    }
                },
                kwargs,
                "openai",
            )
            routing = get_current_session().metadata.get("_tp_routing")
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertIsNotNone(routing)
        self.assertEqual(routing["rule_id"], "rule_rr")
        self.assertEqual(routing["rule_name"], "Send free users to a cheaper model")
        self.assertEqual(routing["actual_model"], "gpt-4o-mini")

    def test_apply_reroute_omits_rule_name_when_absent(self):
        kwargs = {"model": "gpt-4o"}
        with tp.session(name="t"):
            _apply_reroute(
                {
                    "reroute": {
                        "mode": "enforce",
                        "model": "gpt-4o-mini",
                        "provider": "openai",
                        "rule_id": "rule_rr",
                        "original": {"provider": "openai", "model": "gpt-4o"},
                    }
                },
                kwargs,
                "openai",
            )
            routing = get_current_session().metadata.get("_tp_routing")
        self.assertIsNotNone(routing)
        self.assertEqual(routing["rule_id"], "rule_rr")
        self.assertNotIn("rule_name", routing)

    def test_apply_reroute_omits_rule_name_when_none(self):
        kwargs = {"model": "gpt-4o"}
        with tp.session(name="t"):
            _apply_reroute(
                {
                    "reroute": {
                        "mode": "enforce",
                        "model": "gpt-4o-mini",
                        "provider": "openai",
                        "rule_id": "rule_rr",
                        "rule_name": None,
                        "original": {"provider": "openai", "model": "gpt-4o"},
                    }
                },
                kwargs,
                "openai",
            )
            routing = get_current_session().metadata.get("_tp_routing")
        self.assertIsNotNone(routing)
        self.assertNotIn("rule_name", routing)

    def test_apply_reroute_omits_rule_name_when_empty_string(self):
        kwargs = {"model": "gpt-4o"}
        with tp.session(name="t"):
            _apply_reroute(
                {
                    "reroute": {
                        "mode": "enforce",
                        "model": "gpt-4o-mini",
                        "provider": "openai",
                        "rule_id": "rule_rr",
                        "rule_name": "",
                        "original": {"provider": "openai", "model": "gpt-4o"},
                    }
                },
                kwargs,
                "openai",
            )
            routing = get_current_session().metadata.get("_tp_routing")
        self.assertIsNotNone(routing)
        self.assertNotIn("rule_name", routing)


class TestRerouteRejectedRuleNameStateB(unittest.TestCase):
    """State B: the reroute directive comes from the /check HTTP response, so the
    local evaluator never runs and the observation is ``_apply_reroute``'s own.
    It shipped without ``rule_name``, so the collector's customer-visible message
    fell back to the raw rule UUID.
    """

    def setUp(self):
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    def _rejected_obs(self):
        for o in tp_state.drain_observations():
            if o.get("outcome") == "reroute_rejected":
                return o
        return None

    def test_cross_provider_rejected_observation_carries_rule_name(self):
        kwargs = {"model": "gpt-4o"}
        status = _apply_reroute(
            {
                "reroute": {
                    "mode": "enforce",
                    "model": "claude-haiku-4-5",
                    "provider": "anthropic",  # != call provider -> cross-provider reject
                    "rule_id": "rule_rr",
                    "rule_name": "Send free users to a cheaper model",
                    "original": {"provider": "openai", "model": "gpt-4o"},
                }
            },
            kwargs,
            "openai",
        )
        self.assertEqual(status, "rejected")
        obs = self._rejected_obs()
        self.assertIsNotNone(obs)
        self.assertEqual(obs["outcome"], "reroute_rejected")
        self.assertEqual(obs["rejection_reason"], "cross_provider_unsupported")
        self.assertEqual(obs["rule_name"], "Send free users to a cheaper model")
        # The collector falls back to rule_id, so it must still be present.
        self.assertEqual(obs["rule_id"], "rule_rr")

    def test_serving_unverified_rejected_observation_carries_rule_name(self):
        kwargs = {"model": "gpt-4o"}
        status = _apply_reroute(
            {
                "reroute": {
                    "mode": "enforce",
                    "model": "gpt-4o-mini",
                    "provider": "openai",  # same provider — only the flag rejects
                    "rule_id": "rule_rr",
                    "rule_name": "Send free users to a cheaper model",
                    "original": {"provider": "openai", "model": "gpt-4o"},
                }
            },
            kwargs,
            "openai",
            serving_unverified=True,
        )
        self.assertEqual(status, "rejected")
        obs = self._rejected_obs()
        self.assertIsNotNone(obs)
        self.assertEqual(obs["rejection_reason"], "serving_unverified")
        self.assertEqual(obs["rule_name"], "Send free users to a cheaper model")

    def test_rejected_observation_omits_rule_name_when_empty_string(self):
        kwargs = {"model": "gpt-4o"}
        _apply_reroute(
            {
                "reroute": {
                    "mode": "enforce",
                    "model": "claude-haiku-4-5",
                    "provider": "anthropic",
                    "rule_id": "rule_rr",
                    "rule_name": "",
                    "original": {"provider": "openai", "model": "gpt-4o"},
                }
            },
            kwargs,
            "openai",
        )
        obs = self._rejected_obs()
        self.assertIsNotNone(obs)
        self.assertEqual(obs["rule_id"], "rule_rr")
        self.assertNotIn("rule_name", obs)

    def test_rejected_observation_omits_rule_name_when_absent(self):
        kwargs = {"model": "gpt-4o"}
        _apply_reroute(
            {
                "reroute": {
                    "mode": "enforce",
                    "model": "claude-haiku-4-5",
                    "provider": "anthropic",
                    "rule_id": "rule_rr",
                    "original": {"provider": "openai", "model": "gpt-4o"},
                }
            },
            kwargs,
            "openai",
        )
        obs = self._rejected_obs()
        self.assertIsNotNone(obs)
        self.assertEqual(obs["rule_id"], "rule_rr")
        self.assertNotIn("rule_name", obs)

    def test_rejected_observation_omits_rule_name_when_none(self):
        kwargs = {"model": "gpt-4o"}
        _apply_reroute(
            {
                "reroute": {
                    "mode": "enforce",
                    "model": "claude-haiku-4-5",
                    "provider": "anthropic",
                    "rule_id": "rule_rr",
                    "rule_name": None,
                    "original": {"provider": "openai", "model": "gpt-4o"},
                }
            },
            kwargs,
            "openai",
        )
        obs = self._rejected_obs()
        self.assertIsNotNone(obs)
        self.assertEqual(obs["rule_id"], "rule_rr")
        self.assertNotIn("rule_name", obs)


if __name__ == "__main__":
    unittest.main()
