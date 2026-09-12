"""A7 — Bedrock Python chat pre-flight reached ``/check`` with NO model at all.

botocore invokes ``BaseClient._make_api_call(self, operation_name, api_params)``
POSITIONALLY, so inside the enforcer's botocore/aiobotocore wrappers ``kwargs``
is ALWAYS empty and the model id only exists at ``args[2]["modelId"]``. Both
chat pre-flight call sites (``enforcer.py`` async wrapper ~:2755, sync wrapper
~:2942) used to pass a bare ``kwargs=kwargs`` into ``_run_async_check`` /
``_run_sync_check``, so ``target_model = (kwargs or {}).get("model")`` was
always ``None``: a model-scoped BLOCK could never match, so an ENFORCE BLOCK
silently failed open and the call was billed; a model-scoped REROUTE rejected
with a blank ``reroute.from.model`` (sibling report item A6); and budget
group-bys keyed on model bucketed the traffic to unknown.

Same failure class as B12 (``test_b12_lc_google_models_prefix.py``) — there a
WRONG model reached ``/check``, here it was NO model at all.

Fix, both in ``token_police/enforcer.py``:

1. A new module-level helper ``_bedrock_model_hint(module_path, args)``
   (~:408-452) plus a ``_BEDROCK_MODELED_OPS`` tuple — the same four-op
   allowlist ``_stash_attempt_context`` already uses for failure rows
   (``Converse``, ``ConverseStream``, ``InvokeModel``,
   ``InvokeModelWithResponseStream``). Gated on ``module_path`` FIRST — the
   two chat pre-flight call sites are shared by EVERY provider (openai,
   anthropic, google, ...), whose ``args`` mean something else entirely, so a
   leak through that gate would corrupt every other provider's ``/check``
   payload.
2. Its result threaded as ``model_hint=`` into both chat pre-flight call
   sites (async ~:2755, sync ~:2942).

``model_hint`` is the SDK's existing, purpose-built mechanism (see
``test_f7_preflight_context.py``, ``test_reroute_unappliable_shape.py``):
matching + audit only, NEVER merged into kwargs. ``_make_api_call`` accepts no
``model`` keyword, so writing one back would raise ``TypeError`` inside the
customer's call; handing the check a throwaway ``{"model": ...}`` body would
let ``_apply_reroute`` "apply" a swap that never reaches AWS (phantom
``_tp_routing`` + false ``REQUEST_REROUTED``). So the chat path keeps
rejecting an enforce-mode REROUTE honestly as ``unappliable_call_shape`` —
only ``reroute.from.model`` on that rejection observation stops being blank
(closes A6).

Node already has this: ``token-police-node/src/enforcer.ts:1915-1931`` sets
``modelHint = reqBody.modelId``. This ticket brings Python to parity; Node is
untouched.

Fixture shapes below deliberately mirror the sibling bedrock test files:

* ``_args(...)`` mirrors ``test_bedrock_body_restore.py``'s ``_args`` and
  ``test_bedrock_failure_single_emitter.py``'s ``_stash`` helper — botocore's
  own positional call shape ``(self, operation_name, api_params)``.
* Group 2's hostile fakes mirror ``test_f7_preflight_context.py``'s ``_Boom``
  (every attribute raises) and ``test_b12_lc_google_models_prefix.py``'s
  house style of one small fake per failure mode.
* Group 3's check-payload harness mirrors ``test_reroute_unappliable_shape.py``'s
  ``_init_client`` (a ``deployment="serverless"`` client, so the pre-flight always
  falls through to the inline ``/check`` mock instead of a warm daemon pack's
  local evaluator possibly deciding "allowed" first and skipping the round trip).
* Group 4 reuses ``test_reroute_unappliable_shape.py``'s ``_directive`` +
  ``drain_observations`` idiom verbatim rather than reinventing it.

``boto3``/``botocore`` are NOT imported and need not be installed — every
call shape below is a plain tuple of stand-in objects, exactly like the
existing bedrock test files in this suite.
"""
from __future__ import annotations

import asyncio
import types as _types
import unittest
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer
from token_police import state as tp_state
from token_police.client import TokenPolice
from token_police.enforcer import _apply_reroute, _bedrock_model_hint, _BEDROCK_MODELED_OPS


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — mirrors
    test_f7_preflight_context.py's rationale (asyncio.run() unsets the
    main-thread event loop, breaking later get_event_loop()-based tests in
    the same pytest process)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _args(op, api_params, self_obj="client-placeholder"):
    """botocore's own positional call shape: ``_make_api_call(self,
    operation_name, api_params)``. Mirrors _args() in
    test_bedrock_body_restore.py / the _stash() tuple in
    test_bedrock_failure_single_emitter.py."""
    return (self_obj, op, api_params)


class _Boom:
    """Every attribute read explodes (mirrors tests/test_f7_preflight_context.py)."""

    def __getattr__(self, name):
        raise RuntimeError("boom")


class _BoomLen:
    """len() and __getitem__ both explode."""

    def __len__(self):
        raise RuntimeError("boom-len")

    def __getitem__(self, idx):
        raise RuntimeError("boom-getitem")


class _BoomEq:
    """Comparing this object with == always raises. Hashable, so it can also
    stand in for "op-slot object" without touching hashing."""

    def __eq__(self, other):
        raise RuntimeError("boom-eq")

    def __hash__(self):
        return id(self)


class _UnhashableOp:
    """No __hash__ at all — membership test on a tuple must not require it."""

    __hash__ = None

    def __eq__(self, other):
        return False


class _BoomDict(dict):
    """A dict subclass (so isinstance(..., dict) is True) whose .get() raises."""

    def get(self, *a, **k):
        raise RuntimeError("boom-get")


class _BoomModulePath:
    """A module_path whose == raises — the very first comparison the helper
    makes."""

    def __eq__(self, other):
        raise RuntimeError("boom-module-eq")

    def __hash__(self):
        return id(self)


# ═══════════════════════════════════════════════════════════════════════
# Group 1 — _bedrock_model_hint gating
# ═══════════════════════════════════════════════════════════════════════
class TestBedrockModelHintGating(unittest.TestCase):

    def test_converse_returns_model_id(self):
        args = _args("Converse", {"modelId": "amazon.nova-lite-v1:0",
                                   "messages": []})
        self.assertEqual(_bedrock_model_hint("botocore.client", args),
                         "amazon.nova-lite-v1:0")

    def test_converse_stream_returns_model_id(self):
        args = _args("ConverseStream", {"modelId": "amazon.nova-lite-v1:0"})
        self.assertEqual(_bedrock_model_hint("botocore.client", args),
                         "amazon.nova-lite-v1:0")

    def test_invoke_model_returns_model_id(self):
        args = _args("InvokeModel", {"modelId": "amazon.titan-embed-text-v1",
                                      "body": b"{}"})
        self.assertEqual(_bedrock_model_hint("botocore.client", args),
                         "amazon.titan-embed-text-v1")

    def test_invoke_model_with_response_stream_returns_model_id(self):
        args = _args("InvokeModelWithResponseStream",
                     {"modelId": "amazon.nova-pro-v1:0"})
        self.assertEqual(_bedrock_model_hint("botocore.client", args),
                         "amazon.nova-pro-v1:0")

    def test_all_four_modeled_ops_are_covered(self):
        # Pins _BEDROCK_MODELED_OPS itself so a future edit that silently
        # drops/renames one of the four ops is caught here, not just via the
        # four tests above happening to still pass.
        self.assertEqual(set(_BEDROCK_MODELED_OPS),
                         {"Converse", "ConverseStream", "InvokeModel",
                          "InvokeModelWithResponseStream"})

    def test_aiobotocore_module_path_works_identically(self):
        args = _args("Converse", {"modelId": "amazon.nova-lite-v1:0"})
        self.assertEqual(_bedrock_model_hint("aiobotocore.client", args),
                         "amazon.nova-lite-v1:0")
        args2 = _args("InvokeModel", {"modelId": "cohere.embed-english-v3"})
        self.assertEqual(_bedrock_model_hint("aiobotocore.client", args2),
                         "cohere.embed-english-v3")

    def test_apply_guardrail_non_llm_op_returns_none(self):
        # ApplyGuardrail carries no modelId — must keep reaching /check with
        # none, exactly as before this fix.
        args = _args("ApplyGuardrail", {"guardrailIdentifier": "g-1"})
        self.assertIsNone(_bedrock_model_hint("botocore.client", args))

    def test_list_foundation_models_non_llm_op_returns_none(self):
        args = _args("ListFoundationModels", {})
        self.assertIsNone(_bedrock_model_hint("botocore.client", args))

    def test_non_botocore_module_path_returns_none_even_bedrock_shaped(self):
        # Load-bearing gate: the two chat pre-flight call sites are shared by
        # every provider. Bedrock-shaped args reaching this helper from a
        # different module_path must still yield None, or a leak here would
        # corrupt every other provider's /check payload.
        bedrock_shaped_args = _args("Converse", {"modelId": "amazon.nova-lite-v1:0"})
        for module_path in ("openai", "anthropic", "google.genai",
                            "openai.resources.chat.completions"):
            self.assertIsNone(
                _bedrock_model_hint(module_path, bedrock_shaped_args), module_path)

    def test_args_too_short_returns_none(self):
        self.assertIsNone(_bedrock_model_hint("botocore.client", ()))
        self.assertIsNone(_bedrock_model_hint("botocore.client",
                                              ("self-only",)))
        self.assertIsNone(_bedrock_model_hint("botocore.client",
                                              ("self", "Converse")))

    def test_api_params_not_dict_returns_none(self):
        for bad_params in ("not-a-dict", ["modelId", "x"], None):
            args = _args("Converse", bad_params)
            self.assertIsNone(_bedrock_model_hint("botocore.client", args),
                             repr(bad_params))

    def test_model_id_missing_returns_none(self):
        args = _args("Converse", {"messages": []})
        self.assertIsNone(_bedrock_model_hint("botocore.client", args))

    def test_model_id_empty_string_returns_none(self):
        args = _args("Converse", {"modelId": ""})
        self.assertIsNone(_bedrock_model_hint("botocore.client", args))

    def test_model_id_non_string_returns_none(self):
        for bad_id in (7, None, object(), ["amazon.nova-lite-v1:0"]):
            args = _args("Converse", {"modelId": bad_id})
            self.assertIsNone(_bedrock_model_hint("botocore.client", args),
                             repr(bad_id))

    def test_cross_region_inference_profile_id_returned_verbatim(self):
        # /log records the full vendor-prefixed id, so the hint must match it
        # exactly — no stripping, no normalization.
        profile_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        args = _args("Converse", {"modelId": profile_id})
        self.assertEqual(_bedrock_model_hint("botocore.client", args), profile_id)


# ═══════════════════════════════════════════════════════════════════════
# Group 2 — totality / GOLDEN RULE: never raises, never mutates
# ═══════════════════════════════════════════════════════════════════════
class TestBedrockModelHintTotality(unittest.TestCase):

    def test_args_none_returns_none(self):
        self.assertIsNone(_bedrock_model_hint("botocore.client", None))

    def test_args_len_and_getitem_raise_returns_none(self):
        try:
            result = _bedrock_model_hint("botocore.client", _BoomLen())
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"_bedrock_model_hint raised: {exc}")
        self.assertIsNone(result)

    def test_op_eq_raises_returns_none(self):
        args = ("self", _BoomEq(), {"modelId": "x"})
        try:
            result = _bedrock_model_hint("botocore.client", args)
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"_bedrock_model_hint raised: {exc}")
        self.assertIsNone(result)

    def test_op_unhashable_returns_none_without_raising(self):
        args = ("self", _UnhashableOp(), {"modelId": "x"})
        try:
            result = _bedrock_model_hint("botocore.client", args)
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"_bedrock_model_hint raised: {exc}")
        self.assertIsNone(result)

    def test_api_params_dict_subclass_get_raises_returns_none(self):
        args = ("self", "Converse", _BoomDict({"modelId": "x"}))
        try:
            result = _bedrock_model_hint("botocore.client", args)
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"_bedrock_model_hint raised: {exc}")
        self.assertIsNone(result)

    def test_module_path_eq_raises_returns_none(self):
        args = ("self", "Converse", {"modelId": "x"})
        try:
            result = _bedrock_model_hint(_BoomModulePath(), args)
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"_bedrock_model_hint raised: {exc}")
        self.assertIsNone(result)

    def test_boom_attribute_object_as_args_returns_none(self):
        try:
            result = _bedrock_model_hint("botocore.client", _Boom())
        except Exception as exc:  # pragma: no cover - defensive
            self.fail(f"_bedrock_model_hint raised: {exc}")
        self.assertIsNone(result)

    def test_does_not_mutate_api_params(self):
        api_params = {"modelId": "amazon.nova-lite-v1:0", "messages": []}
        snapshot = dict(api_params)
        args = _args("Converse", api_params)

        result = _bedrock_model_hint("botocore.client", args)

        self.assertEqual(result, "amazon.nova-lite-v1:0")
        self.assertEqual(api_params, snapshot)


# ═══════════════════════════════════════════════════════════════════════
# Group 3 — the hint actually reaches the /check payload
# ═══════════════════════════════════════════════════════════════════════
def _init_client(firewall="enforce", check_result=None):
    """Constructs a ``TokenPolice`` client directly and registers it via
    ``tp_state.set_client`` — mirrors ``test_f7_preflight_context.py``'s
    ``_PreflightBase._arm`` rather than ``test_reroute_unappliable_shape.py``'s
    ``tp.init()``-based ``_init_client``. Deliberately NOT ``tp.init()``: that
    call auto-instruments every REAL installed provider OTel instrumentor
    (anthropic/openai/bedrock/...) as a process-wide side effect, and
    ``tp.uninstrument()`` does not fully undo it — a LATER test file that
    drives the real installed ``opentelemetry-instrumentation-anthropic``
    package (``test_anthropic_stream_cm_double_log.py``) then observes
    stale/duplicate patches and extra check/log calls. Constructing the
    client directly keeps these tests hermetic and leaves zero cross-file
    global state.

    ``deployment="serverless"`` is deliberate too: ``_local_evaluate`` short
    circuits immediately for any non-"daemon" deployment
    (``tp.deployment != "daemon"``), so every call below is guaranteed to
    fall through to the inline ``/check`` mock instead of a warm daemon
    pack's local evaluator possibly deciding "allowed" on an empty
    directive set and skipping the round trip entirely."""
    client = TokenPolice(api_key="tp_sk_test_a7", firewall=firewall,
                         deployment="serverless")
    tp_state.set_client(client)
    result = check_result if check_result is not None else {"status": "allowed"}
    client.check_sync = MagicMock(return_value=result)
    client.check = AsyncMock(return_value=result)
    client.log_sync = MagicMock()
    return client


class TestBedrockHintReachesCheckPayload(unittest.TestCase):

    def setUp(self):
        tp_state.reset_pack()
        tp_state.drain_observations()

    def tearDown(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        try:
            tp_state.set_client(None)
        except Exception:
            pass

    def test_sync_check_uses_hint_when_kwargs_carries_no_model(self):
        # kwargs={} is the botocore reality: _make_api_call is invoked
        # positionally, so the wrapper's kwargs is always empty.
        client = _init_client()

        with tp.session(name="t"):
            enforcer._run_sync_check(kwargs={}, provider="bedrock",
                                     model_hint="amazon.nova-lite-v1:0")

        self.assertEqual(client.check_sync.call_args.kwargs["model"],
                         "amazon.nova-lite-v1:0")

    def test_async_check_uses_hint_when_kwargs_carries_no_model(self):
        client = _init_client()

        async def _run():
            with tp.session(name="t"):
                await enforcer._run_async_check(kwargs={}, provider="bedrock",
                                                model_hint="amazon.nova-lite-v1:0")

        _run_coro(_run())

        self.assertEqual(client.check.call_args.kwargs["model"],
                         "amazon.nova-lite-v1:0")

    def test_sync_check_kwargs_model_wins_over_the_hint(self):
        # Precedence pin: this guarantees no other provider's payload can be
        # perturbed by the bedrock hint threaded into the shared call sites.
        client = _init_client()

        with tp.session(name="t"):
            enforcer._run_sync_check(kwargs={"model": "gpt-4o"}, provider="openai",
                                     model_hint="amazon.nova-lite-v1:0")

        self.assertEqual(client.check_sync.call_args.kwargs["model"], "gpt-4o")

    def test_async_check_kwargs_model_wins_over_the_hint(self):
        client = _init_client()

        async def _run():
            with tp.session(name="t"):
                await enforcer._run_async_check(kwargs={"model": "gpt-4o"},
                                                provider="openai",
                                                model_hint="amazon.nova-lite-v1:0")

        _run_coro(_run())

        self.assertEqual(client.check.call_args.kwargs["model"], "gpt-4o")


# ═══════════════════════════════════════════════════════════════════════
# Group 4 — no mutation leak (regression guard on the worst failure mode)
# ═══════════════════════════════════════════════════════════════════════
_OMIT = object()


def _directive(model, provider="bedrock", rule_name=_OMIT):
    """Mirrors test_reroute_unappliable_shape.py's _directive verbatim."""
    reroute = {
        "mode": "enforce",
        "model": model,
        "provider": provider,
        "rule_id": "rule_rr",
    }
    if rule_name is not _OMIT:
        reroute["rule_name"] = rule_name
    return {"reroute": reroute}


class TestBedrockHintNeverLeaksIntoKwargs(unittest.TestCase):

    def setUp(self):
        tp_state.reset_pack()
        tp_state.drain_observations()

    def tearDown(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        try:
            tp_state.set_client(None)
        except Exception:
            pass

    def test_sync_check_never_writes_model_into_kwargs(self):
        # The botocore call shape must stay kwargs-less: _make_api_call takes
        # no `model` keyword, so injecting one would TypeError inside the
        # customer's call.
        _init_client()
        kwargs = {}

        with tp.session(name="t"):
            enforcer._run_sync_check(kwargs=kwargs, provider="bedrock",
                                     model_hint="amazon.nova-lite-v1:0")

        self.assertNotIn("model", kwargs)
        self.assertEqual(kwargs, {})

    def test_async_check_never_writes_model_into_kwargs(self):
        _init_client()
        kwargs = {}

        async def _run():
            with tp.session(name="t"):
                await enforcer._run_async_check(kwargs=kwargs, provider="bedrock",
                                                model_hint="amazon.nova-lite-v1:0")

        _run_coro(_run())

        self.assertNotIn("model", kwargs)
        self.assertEqual(kwargs, {})


class TestApplyRerouteBedrockHintPopulatesFromModel(unittest.TestCase):
    """A6: reroute.from.model on the 18 REROUTE_REJECTED rows stops being
    blank once the bedrock hint threads through. Uses _apply_reroute's own
    unit-level idiom from test_reroute_unappliable_shape.py rather than the
    full check() round trip."""

    def setUp(self):
        try:
            tp_state.drain_observations()
        except Exception:
            pass

    def test_botocore_empty_kwargs_rejected_with_hint_populated_from_model(self):
        kwargs = {}  # botocore-shaped: _make_api_call's kwargs is always empty
        snapshot = dict(kwargs)

        status = _apply_reroute(
            _directive("amazon.nova-pro-v1:0"),
            kwargs,
            "bedrock",
            model_hint="amazon.nova-lite-v1:0",
        )

        self.assertEqual(status, "rejected")
        self.assertEqual(kwargs, snapshot)  # no mutation, no invented "model" key
        obs = tp_state.drain_observations()
        self.assertEqual(len(obs), 1)
        o = obs[0]
        self.assertEqual(o["outcome"], "reroute_rejected")
        self.assertEqual(o["rejection_reason"], "unappliable_call_shape")
        self.assertEqual(o["reroute"]["from"]["model"], "amazon.nova-lite-v1:0")
        self.assertEqual(o["reroute"]["to"]["model"], "amazon.nova-pro-v1:0")

    def test_without_a_hint_from_model_stays_blank_as_before(self):
        # Control: proves the populated from.model above comes from the hint,
        # not from some other change to _apply_reroute's rejection path.
        status = _apply_reroute(_directive("amazon.nova-pro-v1:0"), {}, "bedrock")

        self.assertEqual(status, "rejected")
        obs = tp_state.drain_observations()
        self.assertEqual(obs[0]["reroute"]["from"]["model"], "")


# ═══════════════════════════════════════════════════════════════════════
# Group 5 — end-to-end through the REAL wrapper: this is the coverage that
# actually guards the fix. Groups 3/4 above call _run_sync_check /
# _run_async_check / _apply_reroute DIRECTLY with a hand-supplied
# model_hint= kwarg, so they stay green even if the two real call sites
# (async_wrapper ~:2755, sync_wrapper ~:2942) never compute or forward
# _bedrock_model_hint(...) at all. These tests instead drive the installed
# `_make_api_call` wrapper itself — the exact code the fix touches — via
# enforcer._wrap_method(override_module=...), mirroring
# test_bedrock_failure_single_emitter.py's TestWrapperSingleEmitter._install.
# ═══════════════════════════════════════════════════════════════════════
def _e2e_response():
    """A normal (non-error, non-stream) bedrock-shaped response so the
    wrapper's success path runs to completion."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
    }


def _install_fake_boto_client(is_async, service, make_api_call):
    """Builds a FRESH client class per call (so _wrap_method's `_originals`
    guard, keyed on (cls, method_name), can never skip re-patching across
    tests) and installs the real `_make_api_call` pre-flight wrapper on it
    via enforcer._wrap_method — same idiom as
    test_bedrock_failure_single_emitter.py's TestWrapperSingleEmitter._install,
    generalized to both the sync (botocore.client) and async
    (aiobotocore.client) wrapper configs."""

    class _Meta:
        def __init__(self):
            self.service_model = _types.SimpleNamespace(service_name=service)

    if is_async:
        class FakeAioClient:
            def __init__(self):
                self.meta = _Meta()

            async def _make_api_call(self, operation_name, api_params):
                return make_api_call(operation_name, api_params)

        cls = FakeAioClient
    else:
        class FakeBotoClient:
            def __init__(self):
                self.meta = _Meta()

            def _make_api_call(self, operation_name, api_params):
                return make_api_call(operation_name, api_params)

        cls = FakeBotoClient

    enforcer._wrap_method(
        {"module": "aiobotocore.client" if is_async else "botocore.client",
         "object": "", "method": "_make_api_call", "async": is_async},
        override_module=cls,
    )
    return cls


class TestWrapperThreadsHintIntoCheck(unittest.TestCase):
    """Drives the REAL sync + async botocore/aiobotocore `_make_api_call`
    pre-flight wrapper end to end and asserts the model the mocked `/check`
    call actually received. `firewall="enforce"` (never "off" —
    `_run_sync_check`/`_run_async_check` return early on `tp.firewall ==
    "off"`, which would make every assertion here vacuous)."""

    def setUp(self):
        tp_state.reset_pack()
        tp_state.drain_observations()

    def tearDown(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        try:
            tp_state.set_client(None)
        except Exception:
            pass

    def test_sync_wrapper_converse_threads_model_into_check(self):
        client = _init_client(firewall="enforce")
        cls = _install_fake_boto_client(
            False, "bedrock-runtime", lambda op, params: _e2e_response())
        boto_client = cls()

        with tp.session(name="t"):
            result = boto_client._make_api_call(
                "Converse", {"modelId": "amazon.nova-lite-v1:0", "messages": []})

        self.assertEqual(result, _e2e_response())
        self.assertTrue(client.check_sync.called)
        self.assertEqual(client.check_sync.call_args.kwargs["model"],
                         "amazon.nova-lite-v1:0")

    def test_async_wrapper_converse_stream_threads_model_into_check(self):
        client = _init_client(firewall="enforce")
        cls = _install_fake_boto_client(
            True, "bedrock-runtime", lambda op, params: _e2e_response())
        boto_client = cls()

        async def _run():
            with tp.session(name="t"):
                return await boto_client._make_api_call(
                    "ConverseStream", {"modelId": "amazon.nova-lite-v1:0"})

        result = _run_coro(_run())

        self.assertEqual(result, _e2e_response())
        self.assertTrue(client.check.called)
        self.assertEqual(client.check.call_args.kwargs["model"],
                         "amazon.nova-lite-v1:0")

    def test_apply_guardrail_check_still_called_with_none_model(self):
        # Non-LLM bedrock-runtime op: /check is still reached (unlike before
        # this fix, when the model was always None for EVERY op — so this
        # much was already the behavior), but the model stays None. Pins
        # that the fix did not widen the model to ops that carry none.
        client = _init_client(firewall="enforce")
        cls = _install_fake_boto_client(
            False, "bedrock-runtime",
            lambda op, params: {"guardrailId": "g-1"})
        boto_client = cls()

        with tp.session(name="t"):
            boto_client._make_api_call("ApplyGuardrail",
                                       {"guardrailIdentifier": "g-1"})

        self.assertTrue(client.check_sync.called)
        self.assertIsNone(client.check_sync.call_args.kwargs["model"])

    def test_non_bedrock_client_passes_straight_through_no_check(self):
        client = _init_client(firewall="enforce")
        calls = {"n": 0}

        def _make_api_call(op, params):
            calls["n"] += 1
            return {"Buckets": []}

        cls = _install_fake_boto_client(False, "s3", _make_api_call)
        boto_client = cls()

        with tp.session(name="t"):
            result = boto_client._make_api_call("ListBuckets", {})

        self.assertEqual(result, {"Buckets": []})
        self.assertEqual(calls["n"], 1)
        client.check_sync.assert_not_called()
        client.check.assert_not_called()

    def test_customer_api_params_dict_untouched_no_model_key_injected(self):
        # GOLDEN RULE: _make_api_call accepts no `model` keyword — if a
        # mutation bug ever injected kwargs={"model": ...} into the call,
        # this fake's real 2-positional-arg signature would TypeError on the
        # round trip below, failing this test outright.
        client = _init_client(firewall="enforce")
        cls = _install_fake_boto_client(
            False, "bedrock-runtime", lambda op, params: _e2e_response())
        boto_client = cls()
        api_params = {"modelId": "amazon.nova-lite-v1:0", "messages": []}
        snapshot = dict(api_params)

        with tp.session(name="t"):
            boto_client._make_api_call("Converse", api_params)

        self.assertEqual(api_params, snapshot)
        self.assertNotIn("model", api_params)


if __name__ == "__main__":
    unittest.main()
