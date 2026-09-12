"""B2 — LiteLLM `provider/model` prefix normalization.

LiteLLM addresses non-OpenAI vendors as ``"<vendor>/<model>"``
(``gemini/gemini-2.5-flash``), but every other surface a rule/budget can be
authored against — the dashboard, the /log row (LiteLLM echoes the bare
name back), the collector's EQ match — carries the BARE name. Before this
fix, the litellm manual-telemetry seam forwarded the prefixed request string
verbatim into ``/check``'s ``model.name``, so a rule authored for the
dashboard-visible bare name could never match (silent non-enforcement), the
cross-provider REROUTE gate rejected every vendor target (it compared against
the framework slug "litellm"), and a rewritten model would have been written
back bare (breaking the customer's litellm route).

The fix resolves a per-call route context ``{request_model, bare_model,
route_head, vendor}`` once per call via ``litellm.get_llm_provider`` — see
``enforcer._litellm_route_context`` — and threads it through the seam's
matching/reroute/telemetry sites. It is gated on the SEAM REGISTRATION
(``framework == "litellm"`` or the registration's module root is
"litellm"), never on the provider label — OpenRouter's ``openai/gpt-4o-mini``
and HuggingFace's ``org/model`` are the model IDENTITY on their own
(non-litellm) seams and must stay byte-untouched.

Covers (see b2_fix_plan.md §4, items 1-9, and the write-back guard cases
from review):
  1. /check payload bare (sync + async + embedding).
  2. OpenRouter (openai-module base_url seam AND native openrouter module):
     /check model stays PREFIXED, byte-identical.
  3. Reroute applied end-to-end: kwargs["model"] re-prefixed with the
     REQUEST's own head, from-model/_tp_routing bare.
  4. Cross-provider gate: vendor-aware apply + genuine cross-vendor reject.
  5. No-op: same model under the route head → no write-back.
  6. Bare model + `custom_llm_provider=` kwarg: matches bare, stays bare.
  7. Resolver failure (unattributable model / litellm absent / raises):
     verbatim everywhere, never raises.
  8. Failure path: failed call's stashed model is bare (see also the updated
     assertion in test_litellm_original_provider.py).
  9. Write-back guards: no double-prefix, prefixed-target-equals-request is a
     clean no-op, bare request writes verbatim, a bare-name PREFIX without a
     slash still gets prefixed, gateway-style two-level requests keep only
     their own head.

Fixture conventions mirror tests/test_litellm_original_provider.py (fake
litellm module injection via sys.modules, manual-wrapper installation via
_set_manual_wrapper/_install_and_call) and tests/test_reroute_cross_provider.py
(_apply_reroute driven directly with hand-built check-response dicts).
"""
from __future__ import annotations

import asyncio
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import token_police as tp
from token_police import enforcer
from token_police import state as tp_state
from token_police import local_evaluator as _local_evaluator
from token_police.context import TPSession, get_current_session
from token_police.enforcer import _apply_reroute
from token_police.local_evaluator import evaluate, effective_provider


# ── Fake litellm ────────────────────────────────────────────────────────────
# Mirrors litellm.get_llm_provider's contract (verified against the real
# library in tests/test_litellm_original_provider.py::TestRealLitellmContract):
# returns a 4-tuple (stripped_model, provider_slug, api_key, api_base) and
# RAISES on a model it cannot attribute. Any "vendor/model" (or two-slash
# "vendor/org/model") id resolves generically by splitting on the FIRST "/" —
# this is what the real library does for huggingface-style ids too. A bare
# model with a `custom_llm_provider=` kwarg resolves via that kwarg, exactly
# as litellm does (it wins over any inference from the model string).
class _FakeBadRequestError(Exception):
    pass


def _fake_get_llm_provider(model, *args, custom_llm_provider=None, **kwargs):
    if custom_llm_provider:
        return (model, custom_llm_provider, None, None)
    if "/" in model:
        head, tail = model.split("/", 1)
        return (tail, head, None, None)
    raise _FakeBadRequestError(f"LLM Provider NOT provided for model={model}")


def _fake_litellm_module(get_provider=_fake_get_llm_provider):
    mod = types.ModuleType("litellm")
    mod.get_llm_provider = get_provider
    return mod


def _with_litellm(get_provider=_fake_get_llm_provider):
    """Install a fake `litellm` in sys.modules for the duration of a `with`."""
    return mock.patch.dict(
        "sys.modules", {"litellm": _fake_litellm_module(get_provider)}
    )


def _route_ctx_for(model, custom_llm_provider=None, framework="litellm",
                    module_path="litellm", get_provider=_fake_get_llm_provider):
    """Resolve a real route context via the fake resolver — exercises the
    actual _litellm_route_context implementation rather than hand-building
    the dict, so downstream tests still catch a resolution regression."""
    kwargs = {"model": model}
    if custom_llm_provider:
        kwargs["custom_llm_provider"] = custom_llm_provider
    with _with_litellm(get_provider):
        return enforcer._litellm_route_context(framework, module_path, kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# _litellm_route_context — resolution + seam gating (D2, D3)
# ═══════════════════════════════════════════════════════════════════════════
class TestLitellmRouteContextResolution(unittest.TestCase):
    def test_gemini_prefixed_model(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm", {"model": "gemini/gemini-2.5-flash"})
        self.assertEqual(ctx, {
            "request_model": "gemini/gemini-2.5-flash",
            "bare_model": "gemini-2.5-flash",
            "route_head": "gemini",
            "vendor": "gemini",
        })

    def test_anthropic_prefixed_model(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm",
                {"model": "anthropic/claude-haiku-4-5-20251001"})
        self.assertEqual(ctx["bare_model"], "claude-haiku-4-5-20251001")
        self.assertEqual(ctx["route_head"], "anthropic")
        self.assertEqual(ctx["vendor"], "anthropic")

    def test_two_slash_huggingface_id(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm",
                {"model": "huggingface/my-org/my-model"})
        # Only the FIRST slash is the route head — the rest is the bare id.
        self.assertEqual(ctx["bare_model"], "my-org/my-model")
        self.assertEqual(ctx["route_head"], "huggingface")
        self.assertEqual(ctx["vendor"], "huggingface")

    def test_bare_model_with_custom_llm_provider_kwarg(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm",
                {"model": "gemini-2.5-flash", "custom_llm_provider": "gemini"})
        self.assertEqual(ctx["bare_model"], "gemini-2.5-flash")
        self.assertEqual(ctx["vendor"], "gemini")
        # No literal prefix on the request → nothing to re-apply on write-back.
        self.assertIsNone(ctx["route_head"])


class TestLitellmSeamGating(unittest.TestCase):
    """D2 — gated on the REGISTRATION, never on the provider label."""

    def test_registered_via_framework_litellm(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                "litellm", None, {"model": "gemini/gemini-2.5-flash"})
        self.assertIsNotNone(ctx)

    def test_protect_litellm_module_path_without_framework(self):
        # tp.protect("litellm", ...) supplies no framework and may name any
        # provider slug (e.g. "openai_responses") — module_path alone must
        # still be recognized.
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                None, "litellm", {"model": "gemini/gemini-2.5-flash"})
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["vendor"], "gemini")

    def test_dotted_module_path_root_matches(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                None, "litellm.utils", {"model": "gemini/gemini-2.5-flash"})
        self.assertIsNotNone(ctx)

    def test_openai_module_base_url_openrouter_seam_not_gated(self):
        # OpenRouter reached via the openai module + base_url seam — the
        # prefixed id IS the model identity there, must never be touched.
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                None, "openai", {"model": "openai/gpt-4o-mini"})
        self.assertIsNone(ctx)

    def test_native_openrouter_module_not_gated(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                None, "openrouter", {"model": "anthropic/claude-3-haiku"})
        self.assertIsNone(ctx)

    def test_unrelated_module_not_gated(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                None, "huggingface_hub", {"model": "org/model"})
        self.assertIsNone(ctx)


class TestLitellmRouteContextFailsSafe(unittest.TestCase):
    """D3 — any resolution failure yields None, never guesses, never raises."""

    def test_unattributable_bare_model_raises_inside_resolver(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm", {"model": "totally-unknown-model"})
        self.assertIsNone(ctx)

    def test_resolver_raising_unexpected_error(self):
        def _boom(model, *a, **k):
            raise RuntimeError("resolver exploded")

        with _with_litellm(_boom):
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm", {"model": "gemini/gemini-2.5-flash"})
        self.assertIsNone(ctx)

    def test_litellm_module_absent(self):
        with mock.patch.dict("sys.modules", {"litellm": None}):
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm", {"model": "gemini/gemini-2.5-flash"})
        self.assertIsNone(ctx)

    def test_missing_model(self):
        with _with_litellm():
            self.assertIsNone(enforcer._litellm_route_context(
                "litellm", "litellm", {}))
            self.assertIsNone(enforcer._litellm_route_context(
                "litellm", "litellm", None))

    def test_non_string_model(self):
        with _with_litellm():
            ctx = enforcer._litellm_route_context(
                "litellm", "litellm", {"model": 42})
        self.assertIsNone(ctx)

    def test_never_throws_on_hostile_kwargs(self):
        with _with_litellm():
            enforcer._litellm_route_context("litellm", "litellm", "not-a-dict")
            enforcer._litellm_route_context("litellm", "litellm", {"model": ""})


# ═══════════════════════════════════════════════════════════════════════════
# _litellm_bare — matching-side helper
# ═══════════════════════════════════════════════════════════════════════════
class TestLitellmBareHelper(unittest.TestCase):
    def _ctx(self, request_model, bare_model, route_head, vendor="gemini"):
        return {"request_model": request_model, "bare_model": bare_model,
                "route_head": route_head, "vendor": vendor}

    def test_maps_request_string_to_resolved_bare(self):
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertEqual(
            enforcer._litellm_bare(ctx, "gemini/gemini-2.5-flash"),
            "gemini-2.5-flash")

    def test_maps_post_swap_value_carrying_the_same_head(self):
        # Needed for no-op detection: a REROUTE target authored with this
        # route's head must bare the same way the request did.
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertEqual(
            enforcer._litellm_bare(ctx, "gemini/gemini-2.5-flash-lite"),
            "gemini-2.5-flash-lite")

    def test_none_context_is_passthrough(self):
        self.assertEqual(
            enforcer._litellm_bare(None, "gemini/gemini-2.5-flash"),
            "gemini/gemini-2.5-flash")

    def test_unrelated_string_is_passthrough(self):
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertEqual(enforcer._litellm_bare(ctx, "totally-unrelated"),
                         "totally-unrelated")

    def test_non_string_and_empty_are_passthrough(self):
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertIsNone(enforcer._litellm_bare(ctx, None))
        self.assertEqual(enforcer._litellm_bare(ctx, ""), "")
        self.assertEqual(enforcer._litellm_bare(ctx, 42), 42)


# ═══════════════════════════════════════════════════════════════════════════
# _litellm_route_target — write-back helper (D4, plus review guard cases)
# ═══════════════════════════════════════════════════════════════════════════
class TestLitellmRouteTargetWriteBack(unittest.TestCase):
    def _ctx(self, request_model, bare_model, route_head, vendor="gemini"):
        return {"request_model": request_model, "bare_model": bare_model,
                "route_head": route_head, "vendor": vendor}

    def test_prefixed_request_reprefixes_bare_target(self):
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertEqual(
            enforcer._litellm_route_target(ctx, "gemini-2.5-flash-lite"),
            "gemini/gemini-2.5-flash-lite")

    def test_guard_i_target_already_prefixed_no_double_prefix(self):
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertEqual(
            enforcer._litellm_route_target(ctx, "gemini/gemini-2.5-flash-lite"),
            "gemini/gemini-2.5-flash-lite")

    def test_guard_iii_bare_request_writes_target_verbatim(self):
        # route_head None: vendor came from custom_llm_provider= or the
        # litellm registry, not a literal request prefix.
        ctx = self._ctx("gemini-2.5-flash", "gemini-2.5-flash", None)
        self.assertEqual(
            enforcer._litellm_route_target(ctx, "gemini-2.5-pro"),
            "gemini-2.5-pro")

    def test_guard_iv_bare_name_prefix_without_slash_still_gets_prefixed(self):
        # Target merely STARTS WITH the head as a bare-name prefix (no "/") —
        # must not be mistaken for "already carries the route head".
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertEqual(
            enforcer._litellm_route_target(ctx, "gemini-x"), "gemini/gemini-x")

    def test_guard_v_gateway_style_request_keeps_only_its_own_head(self):
        ctx = self._ctx("openrouter/openai/gpt-4o-mini", "openai/gpt-4o-mini",
                        "openrouter", vendor="openrouter")
        self.assertEqual(
            enforcer._litellm_route_target(ctx, "anthropic/claude-3-haiku"),
            "openrouter/anthropic/claude-3-haiku")

    def test_none_context_is_passthrough(self):
        self.assertEqual(
            enforcer._litellm_route_target(None, "claude-3-haiku"),
            "claude-3-haiku")

    def test_no_route_head_and_no_context_never_prefix(self):
        ctx = self._ctx("gemini-2.5-flash", "gemini-2.5-flash", None)
        self.assertEqual(enforcer._litellm_route_target(ctx, "gemini-2.5-pro"),
                         "gemini-2.5-pro")

    def test_non_string_and_empty_target_passthrough(self):
        ctx = self._ctx("gemini/gemini-2.5-flash", "gemini-2.5-flash", "gemini")
        self.assertIsNone(enforcer._litellm_route_target(ctx, None))
        self.assertEqual(enforcer._litellm_route_target(ctx, ""), "")


# ═══════════════════════════════════════════════════════════════════════════
# _apply_reroute — matching, cross-provider gate, no-op, write-back (D4)
# Mirrors tests/test_reroute_cross_provider.py's direct-call style.
# ═══════════════════════════════════════════════════════════════════════════
def _reroute_result(provider, model, rule_id="rr1", rule_name=None, original=None):
    d = {
        "reroute": {
            "mode": "enforce",
            "model": model,
            "provider": provider,
            "rule_id": rule_id,
        }
    }
    if rule_name:
        d["reroute"]["rule_name"] = rule_name
    if original:
        d["reroute"]["original"] = original
    return d


class TestApplyRerouteLitellmSeam(unittest.TestCase):
    def setUp(self):
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    # ── item 3: reroute applied end-to-end ──
    def test_reroute_applied_writes_back_with_request_route_head(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("gemini", "gemini-2.5-flash-lite"),
                kwargs, "litellm", route_ctx=route_ctx,
            )
            self.assertEqual(status, "applied")
            # Written back PREFIXED with the request's own head — never bare
            # (bare would misroute through litellm's OpenAI default).
            self.assertEqual(kwargs["model"], "gemini/gemini-2.5-flash-lite")
            routing = get_current_session().metadata["_tp_routing"]
            # Telemetry/markers stay bare everywhere.
            self.assertEqual(routing["original_model"], "gemini-2.5-flash")
            self.assertEqual(routing["actual_model"], "gemini-2.5-flash-lite")

    def test_rejection_observation_from_model_is_bare(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        with tp.session(name="t"):
            _apply_reroute(
                _reroute_result("anthropic", "claude-3-haiku"),
                kwargs, "litellm", route_ctx=route_ctx,
            )
        obs = tp_state.drain_observations()
        rejected = [o for o in obs if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[0]["reroute"]["from"]["model"], "gemini-2.5-flash")

    # ── item 4: cross-provider gate is vendor-aware ──
    def test_cross_provider_gate_applies_when_target_matches_underlying_vendor(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        status = _apply_reroute(
            _reroute_result("google", "gemini-2.5-flash-lite"),
            kwargs, "litellm", route_ctx=route_ctx,
        )
        self.assertEqual(status, "applied")
        self.assertEqual(kwargs["model"], "gemini/gemini-2.5-flash-lite")

    def test_cross_provider_gate_rejects_genuine_cross_vendor_target(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("anthropic", "claude-3-haiku"),
                kwargs, "litellm", route_ctx=route_ctx,
            )
            self.assertEqual(status, "rejected")
            self.assertEqual(kwargs["model"], "gemini/gemini-2.5-flash")  # unchanged
            self.assertNotIn("_tp_routing", get_current_session().metadata)
        obs = tp_state.drain_observations()
        self.assertTrue(any(
            o.get("outcome") == "reroute_rejected"
            and o.get("rejection_reason") == "cross_provider_unsupported"
            for o in obs
        ))

    def test_without_route_ctx_every_vendor_target_would_reject(self):
        # Regression pin for the ORIGINAL bug: with no route context (as if
        # the seam gate had not fired), the framework slug "litellm" can
        # never equal a vendor target — every reroute rejects.
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        with tp.session(name="t"):
            status = _apply_reroute(
                _reroute_result("google", "gemini-2.5-flash-lite"),
                kwargs, "litellm", route_ctx=None,
            )
            self.assertEqual(status, "rejected")
            self.assertEqual(kwargs["model"], "gemini/gemini-2.5-flash")

    # ── item 5: no-op ──
    def test_noop_when_target_equals_bare_request_model(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        status = _apply_reroute(
            _reroute_result("gemini", "gemini-2.5-flash"),
            kwargs, "litellm", route_ctx=route_ctx,
        )
        self.assertEqual(status, "noop")
        self.assertEqual(kwargs["model"], "gemini/gemini-2.5-flash")  # untouched
        self.assertEqual(tp_state.drain_observations(), [])

    # ── item 6: bare model + custom_llm_provider kwarg ──
    def test_bare_request_with_custom_llm_provider_kwarg(self):
        route_ctx = _route_ctx_for("gemini-2.5-flash", custom_llm_provider="gemini")
        kwargs = {"model": "gemini-2.5-flash", "custom_llm_provider": "gemini"}
        status = _apply_reroute(
            _reroute_result("gemini", "gemini-2.5-pro"),
            kwargs, "litellm", route_ctx=route_ctx,
        )
        self.assertEqual(status, "applied")
        # Write-back stays bare (no route head to re-apply)...
        self.assertEqual(kwargs["model"], "gemini-2.5-pro")
        # ...and the kwarg itself is never touched.
        self.assertEqual(kwargs["custom_llm_provider"], "gemini")

    # ── review guard cases (adjustment B) ──
    def test_guard_i_target_already_prefixed_same_head_no_double_prefix(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        status = _apply_reroute(
            _reroute_result("gemini", "gemini/gemini-2.0-flash-lite"),
            kwargs, "litellm", route_ctx=route_ctx,
        )
        self.assertEqual(status, "applied")
        self.assertEqual(kwargs["model"], "gemini/gemini-2.0-flash-lite")

    def test_guard_ii_target_equal_to_prefixed_request_is_clean_noop(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        kwargs = {"model": "gemini/gemini-2.5-flash"}
        status = _apply_reroute(
            _reroute_result("gemini", "gemini/gemini-2.5-flash"),
            kwargs, "litellm", route_ctx=route_ctx,
        )
        self.assertEqual(status, "noop")
        self.assertEqual(kwargs["model"], "gemini/gemini-2.5-flash")
        self.assertEqual(tp_state.drain_observations(), [])

    def test_guard_v_gateway_style_request_and_target(self):
        route_ctx = _route_ctx_for("openrouter/openai/gpt-4o-mini")
        kwargs = {"model": "openrouter/openai/gpt-4o-mini"}
        status = _apply_reroute(
            _reroute_result("openrouter", "anthropic/claude-3-haiku"),
            kwargs, "litellm", route_ctx=route_ctx,
        )
        self.assertEqual(status, "applied")
        self.assertEqual(kwargs["model"], "openrouter/anthropic/claude-3-haiku")

    def test_never_throws_on_malformed_input_with_route_ctx(self):
        route_ctx = _route_ctx_for("gemini/gemini-2.5-flash")
        _apply_reroute(None, None, None, route_ctx=route_ctx)
        _apply_reroute({"reroute": None}, {}, "litellm", route_ctx=route_ctx)
        _apply_reroute(_reroute_result("gemini", "x"), {"model": "gemini/gemini-2.5-flash"},
                       "litellm", route_ctx="not-a-dict")


# ═══════════════════════════════════════════════════════════════════════════
# local_evaluator PASS 2 — SDK-only route_vendor gate (State A parity)
# ═══════════════════════════════════════════════════════════════════════════
class TestLocalEvaluatorRouteVendorGate(unittest.TestCase):
    def _reroute_pack(self, target_provider, target_model="gemini-2.5-flash-lite"):
        return {
            "version": 1, "tenant_id": "t", "project_id": "p",
            "directives": [
                {
                    "id": "rr1", "kind": "REROUTE", "mode": "enforce", "priority": 10,
                    "selector": {"match": {"field": "model", "operator": "EXISTS"}},
                    "reroute": {"to": {"provider": target_provider, "model": target_model}},
                }
            ],
        }

    def test_route_vendor_present_applies_matching_vendor_target(self):
        # item 4: gemini request + target provider "google" applies.
        res = evaluate(
            self._reroute_pack("google"), None,
            {"provider": "litellm", "model": "gemini-2.5-flash", "route_vendor": "gemini"},
        )
        self.assertEqual(res["decision"]["status"], "rerouted")
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertFalse(rejected)

    def test_route_vendor_present_still_rejects_genuine_cross_vendor(self):
        # item 4: gemini request + target provider "anthropic" rejects.
        res = evaluate(
            self._reroute_pack("anthropic", "claude-3-haiku"), None,
            {"provider": "litellm", "model": "gemini-2.5-flash", "route_vendor": "gemini"},
        )
        self.assertEqual(res["decision"]["status"], "allowed")
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(rejected[0]["rejection_reason"], "cross_provider_unsupported")

    def test_route_vendor_absent_falls_back_to_framework_slug(self):
        # Regression pin: without the SDK-only route_vendor hint, the gate
        # compares against the framework slug "litellm" — no vendor target
        # could ever match (the original bug).
        res = evaluate(
            self._reroute_pack("google"), None,
            {"provider": "litellm", "model": "gemini-2.5-flash"},
        )
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)
        self.assertEqual(res["decision"]["status"], "allowed")

    def test_route_vendor_ignored_when_not_a_string(self):
        res = evaluate(
            self._reroute_pack("anthropic", "claude-3-haiku"), None,
            {"provider": "litellm", "model": "gemini-2.5-flash", "route_vendor": 42},
        )
        rejected = [o for o in res["observations"] if o.get("outcome") == "reroute_rejected"]
        self.assertTrue(rejected)


# ═══════════════════════════════════════════════════════════════════════════
# _stash_attempt_context / _emit_call_failure_log — item 8
# ═══════════════════════════════════════════════════════════════════════════
class TestStashAttemptContextRouteVendor(unittest.TestCase):
    def test_bares_model_and_stashes_resolved_vendor(self):
        route_ctx = _route_ctx_for("anthropic/claude-haiku-4-5-20251001")
        session = TPSession()
        enforcer._stash_attempt_context(
            session, "litellm", "litellm", (),
            {"model": "anthropic/claude-haiku-4-5-20251001"}, route_ctx=route_ctx)
        self.assertEqual(session._attempted_model, "claude-haiku-4-5-20251001")
        self.assertEqual(session._attempted_route_vendor, "anthropic")

    def test_non_litellm_call_clears_route_vendor(self):
        session = TPSession()
        session._attempted_route_vendor = "stale-from-a-previous-litellm-call"
        enforcer._stash_attempt_context(
            session, "openai", "", (), {"model": "gpt-4o-mini"}, route_ctx=None)
        self.assertIsNone(session._attempted_route_vendor)
        self.assertEqual(session._attempted_model, "gpt-4o-mini")

    def test_no_route_ctx_leaves_model_verbatim(self):
        session = TPSession()
        enforcer._stash_attempt_context(
            session, "litellm", "litellm", (),
            {"model": "anthropic/claude-haiku-4-5-20251001"}, route_ctx=None)
        self.assertEqual(session._attempted_model, "anthropic/claude-haiku-4-5-20251001")
        self.assertIsNone(session._attempted_route_vendor)


class TestEmitCallFailureLogPrefersStashedVendor(unittest.TestCase):
    def _base_session(self):
        session = TPSession()
        session.user_id = "u"
        session.paid_plan = "free"
        session.workflow_name = "wf"
        session.session_id = "s"
        session.metadata = {}
        session._call_outcome = {"error_kind": "server_error", "http_status": 500}
        return session

    def test_prefers_attempted_route_vendor_over_resolver(self):
        captured = {}

        class _FakeTP:
            def log_sync(self, **kw):
                captured.update(kw)

        session = self._base_session()
        # Model is ALREADY bare (as _stash_attempt_context now stores it) —
        # the request-prefix fallback inside _resolve_litellm_original_provider
        # has nothing to read, so only the stash can supply the hint.
        session._attempted_model = "claude-haiku-4-5-20251001"
        session._attempted_provider = "litellm"
        session._attempted_route_vendor = "anthropic"

        def _wrong_if_consulted(model, *a, **k):
            return (model, "WRONG_VENDOR", None, None)

        with _with_litellm(_wrong_if_consulted):
            enforcer._emit_call_failure_log(_FakeTP(), session)
        self.assertEqual(captured.get("model"), "claude-haiku-4-5-20251001")
        self.assertEqual(
            (captured.get("model_extras") or {}).get("original_provider"), "anthropic")

    def test_falls_back_to_resolver_when_vendor_not_stashed(self):
        captured = {}

        class _FakeTP:
            def log_sync(self, **kw):
                captured.update(kw)

        session = self._base_session()
        session._attempted_model = "mystery-1"
        session._attempted_provider = "litellm"
        session._attempted_route_vendor = None

        def _resolver(model, *a, **k):
            return (model, "newvendor", None, None)

        with _with_litellm(_resolver):
            enforcer._emit_call_failure_log(_FakeTP(), session)
        self.assertEqual(
            (captured.get("model_extras") or {}).get("original_provider"), "newvendor")


# ═══════════════════════════════════════════════════════════════════════════
# Full manual-wrapper install — end-to-end wiring through _set_manual_wrapper.
# Mirrors tests/test_litellm_original_provider.py::_install_and_call.
# ═══════════════════════════════════════════════════════════════════════════
class _FakeRouteClient:
    """Fake `tp` client exposing check_sync/check/log_sync, capturing every
    call for assertion."""

    def __init__(self, check_result=None, firewall="enforce", deployment="cloud"):
        self.firewall = firewall
        self.deployment = deployment
        self.check_calls = []
        self.log_calls = []
        self._check_result = check_result if check_result is not None else {"status": "allowed"}

    def check_sync(self, **kw):
        self.check_calls.append(kw)
        return dict(self._check_result)

    async def check(self, **kw):
        self.check_calls.append(kw)
        return dict(self._check_result)

    def log_sync(self, **kw):
        self.log_calls.append(kw)


def _chat_response(model):
    return NS(model=model, usage=NS(prompt_tokens=10, completion_tokens=5))


def _install_and_call_success(*, provider="litellm", module_path="litellm",
                              framework="litellm", model="gemini/gemini-2.5-flash",
                              operation=None, is_async=False, check_result=None,
                              custom_llm_provider=None):
    """Install a real manual litellm wrapper (_set_manual_wrapper) and drive
    ONE successful call through it, capturing:
      - the /check payload the fake client's check(_sync) received,
      - the kwargs the underlying provider call actually received (post any
        reroute write-back),
      - the session (for _tp_routing / _local_decision assertions).
    """
    fake_tp = _FakeRouteClient(check_result=check_result)
    received_kwargs = {}

    if is_async:
        class _FakeTarget:
            async def completion(self, *args, **kwargs):
                received_kwargs.update(kwargs)
                return _chat_response(kwargs.get("model"))
    else:
        class _FakeTarget:
            def completion(self, *args, **kwargs):
                received_kwargs.update(kwargs)
                return _chat_response(kwargs.get("model"))

    original = _FakeTarget.completion
    enforcer._set_manual_wrapper(
        _FakeTarget, "completion", original,
        provider=provider, is_async=is_async,
        framework=framework, operation=operation, module_path=module_path,
    )

    session = TPSession()
    session.user_id = "u"
    session.paid_plan = "free"
    session.workflow_name = "f4"
    session.session_id = "s"
    session.metadata = {}

    call_kwargs = {"model": model}
    if custom_llm_provider:
        call_kwargs["custom_llm_provider"] = custom_llm_provider

    try:
        with mock.patch("token_police.enforcer.get_current_session", return_value=session), \
             mock.patch("token_police.enforcer.get_client", return_value=fake_tp), \
             mock.patch("token_police.enforcer.in_langchain", return_value=False), \
             mock.patch("token_police.enforcer.in_litellm", return_value=False), \
             mock.patch("token_police.enforcer.in_llamaindex", return_value=False), \
             mock.patch("token_police.enforcer.in_pydantic_ai", return_value=False), \
             mock.patch("token_police.enforcer.in_agno", return_value=False), \
             mock.patch("token_police.enforcer.maybe_register_openai_agents_tracing"), \
             mock.patch("token_police.enforcer.consume_pending_span_name", return_value=None), \
             mock.patch("token_police.enforcer._capture_composition_at"), \
             mock.patch("token_police.enforcer._build_intent", return_value=None):
            if is_async:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(_FakeTarget().completion(**call_kwargs))
                finally:
                    loop.close()
            else:
                _FakeTarget().completion(**call_kwargs)
    finally:
        _FakeTarget.completion = original
        enforcer._originals.pop((_FakeTarget, "completion"), None)

    return fake_tp, received_kwargs, session


class TestFullWrapperCheckPayload(unittest.TestCase):
    """item 1 — /check payload bare, sync/async/embedding."""

    def test_sync_chat_check_payload_is_bare(self):
        with _with_litellm():
            fake_tp, _, _ = _install_and_call_success(model="anthropic/claude-haiku-4-5-20251001")
        self.assertEqual(len(fake_tp.check_calls), 1)
        self.assertEqual(fake_tp.check_calls[0]["model"], "claude-haiku-4-5-20251001")
        self.assertEqual(fake_tp.check_calls[0]["provider"], "litellm")

    def test_async_chat_check_payload_is_bare(self):
        with _with_litellm():
            fake_tp, _, _ = _install_and_call_success(
                model="gemini/gemini-2.5-flash", is_async=True)
        self.assertEqual(fake_tp.check_calls[0]["model"], "gemini-2.5-flash")

    def test_embedding_check_payload_is_bare(self):
        with _with_litellm():
            fake_tp, _, _ = _install_and_call_success(
                model="gemini/text-embedding-004", operation="embedding")
        self.assertEqual(fake_tp.check_calls[0]["model"], "text-embedding-004")
        self.assertEqual(fake_tp.check_calls[0]["provider"], "litellm")


class TestFullWrapperOpenRouterUnchanged(unittest.TestCase):
    """item 2 — non-litellm seams keep the prefixed model byte-identical."""

    def test_openai_module_base_url_seam(self):
        with _with_litellm():
            fake_tp, received, _ = _install_and_call_success(
                provider="openrouter", module_path="openai", framework=None,
                model="openai/gpt-4o-mini")
        self.assertEqual(fake_tp.check_calls[0]["model"], "openai/gpt-4o-mini")
        self.assertEqual(received.get("model"), "openai/gpt-4o-mini")

    def test_native_openrouter_module_seam(self):
        with _with_litellm():
            fake_tp, received, _ = _install_and_call_success(
                provider="openrouter", module_path="openrouter", framework=None,
                model="anthropic/claude-3-haiku")
        self.assertEqual(fake_tp.check_calls[0]["model"], "anthropic/claude-3-haiku")
        self.assertEqual(received.get("model"), "anthropic/claude-3-haiku")


class TestFullWrapperRerouteEndToEnd(unittest.TestCase):
    """item 3 — full wiring: seam gate -> bare /check -> apply -> re-prefixed
    write-back -> bare telemetry, never raises."""

    def _reroute_check_result(self):
        return {
            "status": "allowed",
            "reroute": {
                "mode": "enforce",
                "model": "gemini-2.5-flash-lite",
                "provider": "gemini",
                "rule_id": "rr1",
                "rule_name": "cheap-gemini",
                "original": {"provider": "litellm", "model": "gemini-2.5-flash"},
            },
        }

    def test_sync_reroute_applied_end_to_end(self):
        with _with_litellm():
            fake_tp, received, session = _install_and_call_success(
                model="gemini/gemini-2.5-flash", check_result=self._reroute_check_result())
        self.assertEqual(fake_tp.check_calls[0]["model"], "gemini-2.5-flash")
        # The underlying litellm.completion call actually received the
        # PREFIXED rewrite — bare would misroute through litellm's default.
        self.assertEqual(received.get("model"), "gemini/gemini-2.5-flash-lite")
        routing = session.metadata.get("_tp_routing")
        self.assertIsNotNone(routing)
        self.assertEqual(routing["original_model"], "gemini-2.5-flash")
        self.assertEqual(routing["actual_model"], "gemini-2.5-flash-lite")
        # The success /log drains + clears session._local_decision onto the
        # log row itself (see _log_manual) — assert on the logged row.
        self.assertEqual(len(fake_tp.log_calls), 1)
        logged_decision = fake_tp.log_calls[0]["local_decision"]
        self.assertEqual(logged_decision["reroute"]["from"]["model"], "gemini-2.5-flash")
        self.assertEqual(logged_decision["reroute"]["to"]["model"], "gemini-2.5-flash-lite")

    def test_async_reroute_applied_end_to_end(self):
        with _with_litellm():
            fake_tp, received, session = _install_and_call_success(
                model="gemini/gemini-2.5-flash", is_async=True,
                check_result=self._reroute_check_result())
        self.assertEqual(fake_tp.check_calls[0]["model"], "gemini-2.5-flash")
        self.assertEqual(received.get("model"), "gemini/gemini-2.5-flash-lite")


class TestFullWrapperResolverFailureFallback(unittest.TestCase):
    """item 7 — resolver failure degrades to verbatim everywhere, never raises."""

    def test_litellm_absent_check_payload_verbatim(self):
        with mock.patch.dict("sys.modules", {"litellm": None}):
            fake_tp, received, _ = _install_and_call_success(
                model="newvendor/mystery-model")
        self.assertEqual(fake_tp.check_calls[0]["model"], "newvendor/mystery-model")
        self.assertEqual(received.get("model"), "newvendor/mystery-model")

    def test_unattributable_bare_model_check_payload_verbatim(self):
        with _with_litellm():
            fake_tp, received, _ = _install_and_call_success(
                model="totally-unknown-model")
        self.assertEqual(fake_tp.check_calls[0]["model"], "totally-unknown-model")
        self.assertEqual(received.get("model"), "totally-unknown-model")


if __name__ == "__main__":
    unittest.main()
