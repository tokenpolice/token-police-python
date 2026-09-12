"""A REROUTE rule targeting an alias (``anthropic/claude-haiku-4-5``) fires
against apps that pin the dated snapshot of the SAME model
(``claude-haiku-4-5-20251001``). The swap was applied blindly — phantom
REQUEST_REROUTED events with zero cost delta, and the customer's pinned
snapshot silently unpinned. Both apply paths must now treat "target resolves to
the model already requested" as a NO-OP: no rewrite, no ``_tp_routing``, no
local decision, no observation."""

import unittest

import token_police as tp
from token_police import state as tp_state
from token_police.context import get_current_session
from token_police.enforcer import _apply_reroute
from token_police.local_evaluator import evaluate
from token_police.reroute_noop import is_noop_reroute, strip_date_suffix


def _reroute_result(model, provider="anthropic"):
    return {
        "reroute": {
            "mode": "enforce",
            "model": model,
            "provider": provider,
            "rule_id": "rule_rr",
            "rule_name": "downgrade to haiku",
        }
    }


def _reroute_pack(target_model, target_provider="anthropic"):
    return {
        "directives": [
            {
                "id": "rule_rr",
                "kind": "REROUTE",
                "mode": "enforce",
                "name": "downgrade to haiku",
                "selector": {"match": {"field": "model", "operator": "EXISTS"}},
                "reroute": {"to": {"provider": target_provider, "model": target_model}},
            }
        ]
    }


class TestStripDateSuffix(unittest.TestCase):
    def test_strips_anthropic_form(self):
        self.assertEqual(
            strip_date_suffix("claude-haiku-4-5-20251001"), "claude-haiku-4-5"
        )

    def test_strips_openai_form(self):
        self.assertEqual(strip_date_suffix("gpt-4o-2024-08-13"), "gpt-4o")

    def test_strips_vertex_form(self):
        self.assertEqual(strip_date_suffix("claude-3-sonnet@20240229"), "claude-3-sonnet")

    def test_strips_only_one_suffix(self):
        self.assertEqual(
            strip_date_suffix("claude-haiku-20240101-20251001"),
            "claude-haiku-20240101",
        )

    def test_leaves_non_date_suffixes(self):
        for model in (
            "claude-3-5-sonnet-latest",
            "gemini-1.5-pro-002",
            "anthropic.claude-3-sonnet-v1:0",
            "gpt-4o",
            "claude-haiku-4-5",
            "gpt-4o-1024-08-13",
            "claude-haiku-4-5-2025100",
        ):
            self.assertEqual(strip_date_suffix(model), model)

    def test_non_str_inputs_pass_through_safely(self):
        self.assertEqual(strip_date_suffix(None), None)
        self.assertEqual(strip_date_suffix(123), 123)
        self.assertEqual(strip_date_suffix(""), "")


class TestIsNoopReroute(unittest.TestCase):
    def test_identical_models(self):
        self.assertTrue(is_noop_reroute("claude-haiku-4-5", "claude-haiku-4-5"))

    def test_case_and_whitespace_insensitive(self):
        self.assertTrue(is_noop_reroute("  Claude-Haiku-4-5 ", "claude-haiku-4-5"))
        self.assertTrue(
            is_noop_reroute(" CLAUDE-HAIKU-4-5-20251001 ", " claude-haiku-4-5 ")
        )

    def test_dated_request_matches_alias_target(self):
        self.assertTrue(
            is_noop_reroute("claude-haiku-4-5-20251001", "claude-haiku-4-5")
        )
        self.assertTrue(is_noop_reroute("gpt-4o-2024-08-13", "gpt-4o"))
        self.assertTrue(is_noop_reroute("claude-3-sonnet@20240229", "claude-3-sonnet"))

    def test_alias_request_to_dated_target_is_not_noop(self):
        # Direction-sensitive: an explicitly dated TARGET is a deliberate pin.
        self.assertFalse(
            is_noop_reroute("claude-haiku-4-5", "claude-haiku-4-5-20251001")
        )

    def test_dated_to_different_dated_is_not_noop(self):
        self.assertFalse(
            is_noop_reroute("claude-haiku-4-5-20250101", "claude-haiku-4-5-20251001")
        )

    def test_genuine_downgrade_is_not_noop(self):
        self.assertFalse(is_noop_reroute("claude-sonnet-4-5", "claude-haiku-4-5"))
        self.assertFalse(is_noop_reroute("gpt-4o", "gpt-4o-mini"))

    def test_non_date_suffixes_are_not_stripped(self):
        self.assertFalse(is_noop_reroute("claude-3-5-sonnet-latest", "claude-3-5-sonnet"))
        self.assertFalse(is_noop_reroute("gemini-1.5-pro-002", "gemini-1.5-pro"))

    def test_bad_inputs_are_never_noop(self):
        for requested, target in (
            (None, "claude-haiku-4-5"),
            ("claude-haiku-4-5", None),
            (None, None),
            ("", "claude-haiku-4-5"),
            ("claude-haiku-4-5", ""),
            ("   ", "claude-haiku-4-5"),
            (123, "claude-haiku-4-5"),
            ("claude-haiku-4-5", {"model": "claude-haiku-4-5"}),
            (["claude-haiku-4-5"], ["claude-haiku-4-5"]),
        ):
            self.assertFalse(is_noop_reroute(requested, target))


class TestApplyRerouteNoop(unittest.TestCase):
    def setUp(self):
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    def test_noop_directive_returns_noop_and_leaves_call_alone(self):
        kwargs = {"model": "claude-haiku-4-5-20251001"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("claude-haiku-4-5"), kwargs, "anthropic"
            )
            self.assertEqual(status, "noop")
            self.assertEqual(kwargs["model"], "claude-haiku-4-5-20251001")
            self.assertNotIn("_tp_routing", get_current_session().metadata)
        self.assertEqual(tp_state.drain_observations(), [])

    def test_identical_model_directive_is_noop(self):
        kwargs = {"model": "claude-haiku-4-5"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("claude-haiku-4-5"), kwargs, "anthropic"
            )
            self.assertEqual(status, "noop")
            self.assertNotIn("_tp_routing", get_current_session().metadata)
        self.assertEqual(tp_state.drain_observations(), [])

    def test_genuine_reroute_still_applies(self):
        # Regression guard — the guard must not disarm real downgrades.
        kwargs = {"model": "claude-sonnet-4-5"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("claude-haiku-4-5"), kwargs, "anthropic"
            )
            self.assertEqual(status, "applied")
            self.assertEqual(kwargs["model"], "claude-haiku-4-5")
            routing = get_current_session().metadata.get("_tp_routing")
            self.assertIsInstance(routing, dict)
            self.assertEqual(routing["original_model"], "claude-sonnet-4-5")
            self.assertEqual(routing["actual_model"], "claude-haiku-4-5")
        self.assertEqual(tp_state.drain_observations(), [])

    def test_alias_request_to_dated_target_still_applies(self):
        kwargs = {"model": "claude-haiku-4-5"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("claude-haiku-4-5-20251001"), kwargs, "anthropic"
            )
            self.assertEqual(status, "applied")
            self.assertEqual(kwargs["model"], "claude-haiku-4-5-20251001")

    def test_cross_provider_rejection_wins_over_noop(self):
        # The reject checks run first, so a same-model cross-provider directive
        # is still audited as rejected rather than silently swallowed.
        kwargs = {"model": "claude-haiku-4-5"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("claude-haiku-4-5", provider="anthropic"),
                kwargs,
                "openai",
            )
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertTrue(
            any(o.get("outcome") == "reroute_rejected" for o in obs)
        )


class TestLocalEvaluatorRerouteNoop(unittest.TestCase):
    def test_noop_directive_yields_no_decision_or_observation(self):
        res = evaluate(
            _reroute_pack("claude-haiku-4-5"),
            None,
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"},
        )
        self.assertEqual(res["decision"]["status"], "allowed")
        self.assertEqual(res["observations"], [])

    def test_identical_model_directive_yields_no_decision(self):
        res = evaluate(
            _reroute_pack("claude-haiku-4-5"),
            None,
            {"provider": "anthropic", "model": "claude-haiku-4-5"},
        )
        self.assertEqual(res["decision"]["status"], "allowed")
        self.assertEqual(res["observations"], [])

    def test_noop_directive_does_not_shadow_a_later_genuine_reroute(self):
        pack = {
            "directives": [
                _reroute_pack("claude-haiku-4-5")["directives"][0],
                {
                    "id": "rule_rr2",
                    "kind": "REROUTE",
                    "mode": "enforce",
                    "selector": {"match": {"field": "model", "operator": "EXISTS"}},
                    "reroute": {
                        "to": {"provider": "anthropic", "model": "claude-haiku-3"}
                    },
                },
            ]
        }
        res = evaluate(
            pack, None, {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
        )
        self.assertEqual(res["decision"]["status"], "rerouted")
        self.assertEqual(res["decision"]["rule_id"], "rule_rr2")

    def test_genuine_directive_unchanged(self):
        res = evaluate(
            _reroute_pack("claude-haiku-4-5"),
            None,
            {"provider": "anthropic", "model": "claude-sonnet-4-5"},
        )
        self.assertEqual(res["decision"]["status"], "rerouted")
        self.assertEqual(
            res["decision"]["reroute"]["to"]["model"], "claude-haiku-4-5"
        )
        self.assertEqual(res["observations"], [])

    def test_dry_run_noop_directive_emits_no_would_reroute(self):
        pack = _reroute_pack("claude-haiku-4-5")
        pack["directives"][0]["mode"] = "dry_run"
        res = evaluate(
            pack, None, {"provider": "anthropic", "model": "claude-haiku-4-5-20251001"}
        )
        self.assertEqual(res["decision"]["status"], "allowed")
        self.assertEqual(res["observations"], [])

    def test_absent_payload_model_still_reroutes(self):
        # Nothing to compare against → guard must stay out of the way.
        pack = _reroute_pack("claude-haiku-4-5")
        pack["directives"][0]["selector"]["match"] = {
            "field": "provider",
            "operator": "EQ",
            "value": "anthropic",
        }
        res = evaluate(pack, None, {"provider": "anthropic"})
        self.assertEqual(res["decision"]["status"], "rerouted")


if __name__ == "__main__":
    unittest.main()
