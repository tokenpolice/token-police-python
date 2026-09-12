"""A cross-provider REROUTE directive is rejected by the local fast path
(State A, ``cross_provider_unsupported``) but was applied unconditionally by
``_apply_reroute`` (State B, /check response). The guard makes the two paths
agree: ``_apply_reroute`` now skips the model swap when the reroute's target
provider is present and differs from the call provider, using the SAME
``effective_provider()`` semantics the evaluator uses."""

import unittest

import token_police as tp
from token_police import state as tp_state
from token_police.context import get_current_session
from token_police.enforcer import _apply_reroute
from token_police.local_evaluator import evaluate, effective_provider


def _reroute_result(provider, model="target-model"):
    return {
        "reroute": {
            "mode": "enforce",
            "model": model,
            "provider": provider,
            "rule_id": "rule_rr",
            "rule_name": "reroute rule",
        }
    }


class TestRerouteCrossProvider(unittest.TestCase):
    def setUp(self):
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    def test_same_provider_reroute_applies(self):
        # Assertion 2
        kwargs = {"model": "gpt-4o"}
        status = _apply_reroute(
            {"reroute": {"mode": "enforce", "model": "gpt-4o-mini", "provider": "openai"}},
            kwargs,
            "openai",
        )
        self.assertEqual(status, "applied")
        self.assertEqual(kwargs["model"], "gpt-4o-mini")

    def test_cross_provider_reroute_skipped(self):
        # Assertion 4 — model untouched AND no _tp_routing written to session.metadata
        kwargs = {"model": "gpt-4o"}
        with tp.session(name="t"):
            status = _apply_reroute(
                {
                    "reroute": {
                        "mode": "enforce",
                        "model": "claude-3-haiku",
                        "provider": "anthropic",
                    }
                },
                kwargs,
                "openai",
            )
            self.assertEqual(status, "rejected")
            self.assertEqual(kwargs["model"], "gpt-4o")  # unchanged
            self.assertNotIn("_tp_routing", get_current_session().metadata)
            obs = tp_state.drain_observations()
            self.assertTrue(any(
                o.get("outcome") == "reroute_rejected"
                and o.get("rejection_reason") == "cross_provider_unsupported"
                for o in obs
            ))

    def test_b20_serving_unverified_rejected(self):
        kwargs = {"model": "gpt-4o"}
        status = _apply_reroute(
            {"reroute": {"mode": "enforce", "model": "gpt-4o-mini", "provider": "openai"}},
            kwargs,
            "openai",
            serving_unverified=True,
        )
        self.assertEqual(status, "rejected")
        self.assertEqual(kwargs["model"], "gpt-4o")
        obs = tp_state.drain_observations()
        self.assertTrue(any(
            o.get("outcome") == "reroute_rejected"
            and o.get("rejection_reason") == "serving_unverified"
            for o in obs
        ))

    def test_b20_missing_kwargs_noop(self):
        # Superseded: a live ENFORCE directive on a kwargs-less call shape now
        # reports the refusal instead of resolving silently — "rejected" with
        # a reroute_rejected/unappliable_call_shape observation, not a silent
        # noop. See tests/test_reroute_unappliable_shape.py for full coverage.
        status = _apply_reroute(
            {"reroute": {"mode": "enforce", "model": "gpt-4o-mini", "provider": "openai"}},
            None,
            "openai",
        )
        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")

    def test_state_a_and_b_agree_cross_provider(self):
        # Assertion 5 — State A rejects, State B leaves body unchanged.
        pack = {
            "directives": [
                {
                    "id": "rule_rr",
                    "kind": "REROUTE",
                    "mode": "enforce",
                    "selector": {"match": {"field": "model", "operator": "EXISTS"}},
                    "reroute": {"to": {"provider": "anthropic", "model": "claude-3-haiku"}},
                }
            ]
        }
        res = evaluate(pack, None, {"provider": "openai", "model": "gpt-4o"})
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[0]["rejection_reason"], "cross_provider_unsupported")

        kwargs = {"model": "gpt-4o"}
        _apply_reroute(_reroute_result("anthropic", "claude-3-haiku"), kwargs, "openai")
        self.assertEqual(kwargs["model"], "gpt-4o")

    def test_mixed_case_provider_applies(self):
        # Assertion 6 — effective_provider("OpenAI") == "openai" so it still applies.
        kwargs = {"model": "gpt-4o"}
        _apply_reroute(
            {"reroute": {"mode": "enforce", "model": "gpt-4o-mini", "provider": "openai"}},
            kwargs,
            "OpenAI",
        )
        self.assertEqual(kwargs["model"], "gpt-4o-mini")

    def test_absent_target_provider_applies(self):
        # Assertion 7 — falsy/absent target provider preserves the fallback.
        kwargs = {"model": "gpt-4o"}
        _apply_reroute(
            {"reroute": {"mode": "enforce", "model": "gpt-4o-mini"}},
            kwargs,
            "openai",
        )
        self.assertEqual(kwargs["model"], "gpt-4o-mini")

    def test_falsy_provider_present_target_skips(self):
        # Assertion 8 — falsy call provider + present target → guard fires.
        for call_provider in ("", None):
            kwargs = {"model": "gpt-4o"}
            with tp.session(name="t"):
                _apply_reroute(
                    _reroute_result("anthropic", "claude-3-haiku"), kwargs, call_provider
                )
                self.assertEqual(kwargs["model"], "gpt-4o")
                self.assertNotIn("_tp_routing", get_current_session().metadata)

    def test_effective_provider_trims_lowercases_and_canonicalizes_aliases(self):
        # Trim + lowercase, then canonicalize alias slugs onto their canonical
        # serving-provider slug so both sides of a comparison agree.
        self.assertEqual(effective_provider("openai"), "openai")
        self.assertEqual(effective_provider(" OpenAI "), "openai")
        self.assertEqual(effective_provider(""), "")
        # Alias canonicalization.
        self.assertEqual(effective_provider("together_ai"), "together")
        self.assertEqual(effective_provider("together"), "together")
        self.assertEqual(effective_provider(" Together_AI "), "together")
        # Responses pseudo-provider is runtime-only (not shared price map).
        self.assertEqual(effective_provider("openai_responses"), "openai")
        self.assertEqual(effective_provider("OpenAI_Responses"), "openai")

    def test_b16_openai_responses_entity_block_uses_openai_group_tag(self):
        # /log arms entities as openai; local eval must look up the same tag
        # when the enforcer still observes the Responses parse pseudo-provider.
        pack = {
            "version": 1,
            "tenant_id": "t",
            "project_id": "p",
            "directives": [
                {
                    "id": "r-prov",
                    "kind": "ENTITY_BLOCK",
                    "mode": "enforce",
                    "priority": 10,
                    "selector": {
                        "match": {"field": "model", "operator": "EXISTS"},
                        "group_by": ["provider"],
                    },
                    "entities": ["openai"],
                }
            ],
        }
        result = evaluate(
            pack,
            None,
            {"model": "gpt-4o-mini", "provider": "openai_responses"},
        )
        self.assertEqual(result["decision"]["status"], "blocked")
        self.assertEqual(result["decision"]["rule_id"], "r-prov")

    def test_alias_target_matches_observed_reroute_applies_state_a(self):
        # Directive target "together_ai" vs observed "together" — same provider
        # once aliases canonicalize, so the local evaluator reroutes (not rejects).
        pack = {
            "version": 1, "tenant_id": "t", "project_id": "p",
            "directives": [
                {
                    "id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
                    "selector": {"match": {"field": "model", "operator": "EXISTS"}},
                    "reroute": {"to": {"provider": "together_ai", "model": "target-model"}},
                }
            ],
        }
        res = evaluate(pack, None, {"provider": "together", "model": "llama-3"})
        self.assertEqual(res["decision"]["status"], "rerouted")
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertFalse(rejected)

    def test_alias_target_matches_observed_reverse_reroute_applies_state_a(self):
        # Reverse: directive "together" vs observed "together_ai".
        pack = {
            "version": 1, "tenant_id": "t", "project_id": "p",
            "directives": [
                {
                    "id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
                    "selector": {"match": {"field": "model", "operator": "EXISTS"}},
                    "reroute": {"to": {"provider": "together", "model": "target-model"}},
                }
            ],
        }
        res = evaluate(pack, None, {"provider": "together_ai", "model": "llama-3"})
        self.assertEqual(res["decision"]["status"], "rerouted")
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertFalse(rejected)

    def test_alias_provider_apply_reroute_state_b(self):
        # State B (_apply_reroute) must agree: alias target + observed provider
        # canonicalize equal → the model swap applies.
        kwargs = {"model": "llama-3"}
        with tp.session(name="t"):
            _apply_reroute(_reroute_result("together_ai", "target-model"), kwargs, "together")
            self.assertEqual(kwargs["model"], "target-model")

    def test_never_throws_on_malformed_input(self):
        # Assertions 11/12 — golden rule: never throws.
        _apply_reroute(None, None, None)
        _apply_reroute({"reroute": None}, {}, "openai")
        _apply_reroute(_reroute_result("anthropic"), {"model": "gpt-4o"}, "openai")


# ── serving-provider from base_url drives the cross-provider guard ──
from token_police.enforcer import (
    _effective_provider,
    _resolve_serving_provider,
    resolve_serving_from_base_url,
    _match_host_to_provider,
    _extract_host,
)


class _FakeBoundMethodSelf:
    """Mimics resource binding (messages.create) with a nested _client.base_url."""

    def __init__(self, base_url):
        self._client = type("C", (), {"base_url": base_url})()


_ANTHROPIC_TO_HAIKU_PACK = {
    "directives": [
        {
            "id": "rule_rr",
            "kind": "REROUTE",
            "mode": "enforce",
            "selector": {"match": {"field": "model", "operator": "EXISTS"}},
            "reroute": {"to": {"provider": "anthropic", "model": "claude-haiku-4-5"}},
        }
    ]
}

_OPENAI_TO_MINI_PACK = {
    "directives": [
        {
            "id": "rule_rr",
            "kind": "REROUTE",
            "mode": "enforce",
            "selector": {"match": {"field": "model", "operator": "EXISTS"}},
            "reroute": {"to": {"provider": "openai", "model": "gpt-4o-mini"}},
        }
    ]
}


class TestB01ServingProviderHostMap(unittest.TestCase):
    def test_extract_host(self):
        self.assertEqual(_extract_host("https://api.minimax.io/anthropic/v1"), "api.minimax.io")
        self.assertEqual(
            _extract_host("https://user:pass@api.anthropic.com:443/v1"), "api.anthropic.com"
        )
        self.assertEqual(_extract_host(""), "")

    def test_match_known_hosts(self):
        self.assertEqual(_match_host_to_provider("api.minimax.io"), "minimax")
        self.assertEqual(_match_host_to_provider("api.minimaxi.com"), "minimax")
        self.assertEqual(_match_host_to_provider("api.anthropic.com"), "anthropic")
        self.assertEqual(_match_host_to_provider("openrouter.ai"), "openrouter")
        self.assertEqual(
            _match_host_to_provider("my-resource.openai.azure.com"), "azure-openai"
        )
        self.assertIsNone(_match_host_to_provider("llm.corp.example"))

    def test_resolve_serving_three_way(self):
        self.assertEqual(resolve_serving_from_base_url(""), {"kind": "absent"})
        self.assertEqual(
            resolve_serving_from_base_url("https://api.minimax.io/anthropic"),
            {"kind": "recognized", "provider": "minimax"},
        )
        self.assertEqual(
            resolve_serving_from_base_url("https://openrouter.ai/api/v1"),
            {"kind": "recognized", "provider": "openrouter"},
        )
        self.assertEqual(
            resolve_serving_from_base_url("https://llm.corp.example/v1"),
            {"kind": "unrecognized"},
        )


class TestB01EffectiveProvider(unittest.TestCase):
    def test_a_anthropic_minimax_host(self):
        args = (_FakeBoundMethodSelf("https://api.minimax.io/anthropic/v1"),)
        self.assertEqual(_effective_provider("anthropic", args), "minimax")

    def test_b_anthropic_default_host(self):
        args = (_FakeBoundMethodSelf("https://api.anthropic.com"),)
        self.assertEqual(_effective_provider("anthropic", args), "anthropic")

    def test_b2_absent_base_url_keeps_module(self):
        args = (_FakeBoundMethodSelf(""),)
        self.assertEqual(_effective_provider("anthropic", args), "anthropic")
        self.assertEqual(_effective_provider("anthropic", None), "anthropic")
        self.assertEqual(_effective_provider("anthropic", ()), "anthropic")

    def test_c_openai_minimax_host(self):
        args = (_FakeBoundMethodSelf("https://api.minimax.io/v1"),)
        self.assertEqual(_effective_provider("openai", args), "minimax")

    def test_d_unknown_custom_base_url_keeps_module_plus_unverified(self):
        # Provider stays anthropic for match/groupBy; REROUTE refuse via flag.
        args = (_FakeBoundMethodSelf("https://llm.corp.example/v1"),)
        self.assertEqual(_effective_provider("anthropic", args), "anthropic")
        self.assertEqual(
            _resolve_serving_provider("anthropic", args),
            {"provider": "anthropic", "serving_unverified": True},
        )

    def test_e_openrouter_regression(self):
        args = (_FakeBoundMethodSelf("https://openrouter.ai/api/v1"),)
        self.assertEqual(_effective_provider("openai", args), "openrouter")
        self.assertEqual(
            _resolve_serving_provider("openai", args),
            {"provider": "openrouter", "serving_unverified": False},
        )

    def test_never_throws(self):
        class Boom:
            @property
            def _client(self):
                raise RuntimeError("boom")

        self.assertEqual(_effective_provider("openai", (Boom(),)), "openai")


class TestB01RerouteGuardServingProvider(unittest.TestCase):
    def test_a_anthropic_minimax_rejects_anthropic_target(self):
        serving = _effective_provider(
            "anthropic", (_FakeBoundMethodSelf("https://api.minimax.io/anthropic/v1"),)
        )
        self.assertEqual(serving, "minimax")

        res = evaluate(
            _ANTHROPIC_TO_HAIKU_PACK, None, {"provider": serving, "model": "MiniMax-M2.5"}
        )
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[0]["rejection_reason"], "cross_provider_unsupported")
        self.assertEqual(res["decision"]["status"], "allowed")

        kwargs = {"model": "MiniMax-M2.5"}
        with tp.session(name="t"):
            _apply_reroute(_reroute_result("anthropic", "claude-haiku-4-5"), kwargs, serving)
            self.assertEqual(kwargs["model"], "MiniMax-M2.5")

    def test_b_default_anthropic_applies(self):
        serving = _effective_provider(
            "anthropic", (_FakeBoundMethodSelf("https://api.anthropic.com"),)
        )
        self.assertEqual(serving, "anthropic")

        res = evaluate(
            _ANTHROPIC_TO_HAIKU_PACK,
            None,
            {"provider": serving, "model": "claude-sonnet-4"},
        )
        self.assertEqual(res["decision"]["status"], "rerouted")
        self.assertEqual(res["decision"]["reroute"]["to"]["model"], "claude-haiku-4-5")

        kwargs = {"model": "claude-sonnet-4"}
        with tp.session(name="t"):
            _apply_reroute(_reroute_result("anthropic", "claude-haiku-4-5"), kwargs, serving)
            self.assertEqual(kwargs["model"], "claude-haiku-4-5")

    def test_c_openai_minimax_rejects_openai_target(self):
        serving = _effective_provider(
            "openai", (_FakeBoundMethodSelf("https://api.minimax.io/v1"),)
        )
        self.assertEqual(serving, "minimax")

        res = evaluate(
            _OPENAI_TO_MINI_PACK, None, {"provider": serving, "model": "MiniMax-M2.5"}
        )
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(res["decision"]["status"], "allowed")

        kwargs = {"model": "MiniMax-M2.5"}
        with tp.session(name="t"):
            _apply_reroute(_reroute_result("openai", "gpt-4o-mini"), kwargs, serving)
            self.assertEqual(kwargs["model"], "MiniMax-M2.5")

    def test_d_unknown_custom_base_rejects_same_module_target(self):
        resolved = _resolve_serving_provider(
            "anthropic", (_FakeBoundMethodSelf("https://llm.corp.example/v1"),)
        )
        self.assertEqual(resolved, {"provider": "anthropic", "serving_unverified": True})

        res = evaluate(
            _ANTHROPIC_TO_HAIKU_PACK,
            None,
            {
                "provider": resolved["provider"],
                "model": "claude-sonnet-4",
                "serving_unverified": True,
            },
        )
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[0]["rejection_reason"], "serving_unverified")
        self.assertEqual(res["decision"]["status"], "allowed")

        # Without the flag, same-module target would apply.
        would = evaluate(
            _ANTHROPIC_TO_HAIKU_PACK,
            None,
            {"provider": "anthropic", "model": "claude-sonnet-4"},
        )
        self.assertEqual(would["decision"]["status"], "rerouted")

        kwargs = {"model": "claude-sonnet-4"}
        with tp.session(name="t"):
            _apply_reroute(
                _reroute_result("anthropic", "claude-haiku-4-5"),
                kwargs,
                resolved["provider"],
                serving_unverified=True,
            )
            self.assertEqual(kwargs["model"], "claude-sonnet-4")

    def test_d2_serving_unverified_wins_over_cross_provider(self):
        # Observed provider ("anthropic") and the directive's target ("openai")
        # are genuinely different modules, so cross_provider_unsupported would
        # also fire on its own — but serving_unverified must win, mirroring
        # _apply_reroute's precedence (B4/B5 parity).
        res = evaluate(
            _OPENAI_TO_MINI_PACK,
            None,
            {
                "provider": "anthropic",
                "model": "claude-sonnet-4",
                "serving_unverified": True,
            },
        )
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[0]["rejection_reason"], "serving_unverified")
        self.assertEqual(res["decision"]["status"], "allowed")

    def test_e_openrouter_still_applies(self):
        serving = _effective_provider(
            "openai", (_FakeBoundMethodSelf("https://openrouter.ai/api/v1"),)
        )
        self.assertEqual(serving, "openrouter")

        pack = {
            "directives": [
                {
                    "id": "rule_rr",
                    "kind": "REROUTE",
                    "mode": "enforce",
                    "selector": {"match": {"field": "model", "operator": "EXISTS"}},
                    "reroute": {
                        "to": {"provider": "openrouter", "model": "openai/gpt-4o-mini"}
                    },
                }
            ]
        }
        res = evaluate(pack, None, {"provider": serving, "model": "openai/gpt-4o"})
        self.assertEqual(res["decision"]["status"], "rerouted")

        kwargs = {"model": "openai/gpt-4o"}
        with tp.session(name="t"):
            _apply_reroute(
                _reroute_result("openrouter", "openai/gpt-4o-mini"), kwargs, serving
            )
            self.assertEqual(kwargs["model"], "openai/gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
