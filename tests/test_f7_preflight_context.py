"""Pre-flight checks must carry the call's real model + provider.

Many surfaces keep the model somewhere other than the call kwargs — xai-sdk on
``self._proto``, every framework wrapper (agno / pydantic_ai / LangChain /
LlamaIndex) on the bound instance — so their pre-flights used to evaluate with
``model=""`` and often ``provider=""``. Consequences: model/provider-scoped
BLOCK rules never matched, ``groupBy: [model|provider]`` budgets bucketed under
"unknown", REROUTE was always rejected, and /check audit rows carried empty
from_model/from_provider.

The fix threads a NEW ``model_hint`` argument (never merged into kwargs) plus
the derived provider. Two behaviours are asserted together:

  * matching + audit context is now populated — a model-conditioned BLOCK on
    these paths genuinely blocks in enforce mode (the sanctioned raise); and
  * reroute APPLY-ability is unchanged — a hint is not an appliable body, so a
    matching same-provider REROUTE still ends as a noop with no ``_tp_routing``
    and no "applied" local_decision. A phantom "applied" audit on a call the
    wire never saw would be worse than no reroute at all.

All fakes — no provider SDK or framework is installed.
"""
import asyncio
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police import state as tp_state
from token_police.client import TokenPolice
from token_police.context import _current_session, TPSession
from token_police.exceptions import TokenPoliceBlockedError


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — asyncio.run
    unsets the main-thread event loop, breaking later get_event_loop()-based
    tests in the same pytest process."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _snapshot(directives):
    return {
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "t", "project_id": "p", "ttl_seconds": 600,
        "loop_blocks": [], "directives": directives,
    }


def _reroute_rule(target_provider, target_model):
    return {
        "id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
        "selector": {"match": None, "group_by": []},
        "reroute": {"from": {}, "to": {"provider": target_provider,
                                       "model": target_model}},
    }


def _model_block_rule(model):
    return {
        "id": "blk", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce",
        "priority": 10,
        "selector": {"match": {"field": "model", "operator": "EQ",
                               "value": model},
                     "group_by": []},
    }


class _PreflightBase(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        self._session = TPSession()
        self._token = _current_session.set(self._session)
        self._teardowns = []
        self._async_checks = []

    def tearDown(self):
        for fn in reversed(self._teardowns):
            try:
                fn()
            except Exception:
                pass
        _current_session.reset(self._token)
        tp_state.reset_pack()
        tp_state.drain_observations()
        try:
            tp_state.set_client(None)
        except Exception:
            pass

    def _arm(self, directives=(), check_result=None, firewall="enforce"):
        """Install a daemon client with a warm pack and a stubbed /check."""
        client = TokenPolice(
            api_key="tp_sk_test123", base_url="http://127.0.0.1:59999",
            timeout=0.1, firewall=firewall, deployment="daemon",
        )
        tp_state.set_client(client)
        self.assertTrue(tp_state.apply_snapshot(_snapshot(list(directives))))
        result = check_result if check_result is not None else {"status": "allowed"}

        async def _acheck(**kw):
            self._async_checks.append(kw)
            return result

        patches = [
            mock.patch.object(client, "check_sync", return_value=result),
            mock.patch.object(client, "check", new=_acheck),
            mock.patch.object(client, "log_sync"),
        ]
        for p in patches:
            p.start()
            self._teardowns.append(p.stop)
        return client

    def _spy_evaluator(self):
        """Record every ctx the local evaluator is handed; always fall through
        to the inline /check so the check payload is observable too."""
        seen = []

        def _evaluate(_pack, _session, ctx, *a, **k):
            seen.append(dict(ctx))
            return {"decision": None, "observations": []}

        p = mock.patch.object(enforcer._local_evaluator, "evaluate", _evaluate)
        p.start()
        self._teardowns.append(p.stop)
        return seen


# ── 1. model_hint plumbing ────────────────────────────────────────────
class TestModelHintPlumbing(_PreflightBase):

    def test_hint_reaches_the_evaluator_ctx_and_the_check_payload(self):
        client = self._arm()
        ctxs = self._spy_evaluator()

        enforcer._run_sync_check(model_hint="m1", provider="p1")

        self.assertEqual(ctxs[0]["model"], "m1")
        self.assertEqual(ctxs[0]["provider"], "p1")
        self.assertEqual(client.check_sync.call_args.kwargs["model"], "m1")
        self.assertEqual(client.check_sync.call_args.kwargs["provider"], "p1")

    def test_kwargs_model_wins_over_the_hint(self):
        client = self._arm()
        ctxs = self._spy_evaluator()

        enforcer._run_sync_check(kwargs={"model": "from-kwargs"},
                                 model_hint="from-hint", provider="p1")

        self.assertEqual(ctxs[0]["model"], "from-kwargs")
        self.assertEqual(client.check_sync.call_args.kwargs["model"], "from-kwargs")

    def test_hint_fills_in_when_kwargs_model_is_empty(self):
        self._arm()
        ctxs = self._spy_evaluator()

        enforcer._run_sync_check(kwargs={"model": ""}, model_hint="m1")

        self.assertEqual(ctxs[0]["model"], "m1")

    def test_non_string_hint_is_ignored(self):
        client = self._arm()
        ctxs = self._spy_evaluator()

        enforcer._run_sync_check(model_hint=object())

        self.assertEqual(ctxs[0]["model"], "")
        self.assertIsNone(client.check_sync.call_args.kwargs["model"])

    def test_async_twin_plumbs_the_hint(self):
        self._arm()
        ctxs = self._spy_evaluator()

        _run_coro(enforcer._run_async_check(model_hint="m1", provider="p1"))

        self.assertEqual(ctxs[0]["model"], "m1")
        self.assertEqual(ctxs[0]["provider"], "p1")

    def test_hint_never_lands_in_kwargs(self):
        """The scope invariant: a hint must not become an appliable body."""
        self._arm()
        self._spy_evaluator()
        kwargs = {"messages": []}

        enforcer._run_sync_check(kwargs=kwargs, model_hint="m1")

        self.assertNotIn("model", kwargs)

    def test_hint_only_ctx_matches_a_model_conditioned_block(self):
        """Intended behaviour change — the sanctioned raise."""
        self._arm([_model_block_rule("m1")], check_result={"status": "blocked"})

        with self.assertRaises(TokenPoliceBlockedError):
            enforcer._run_sync_check(model_hint="m1", provider="openai")

    def test_hint_only_reroute_match_never_claims_an_apply(self):
        """No-phantom-apply: the rule matches on hint ctx, but kwargs carries no
        "model" key, so nothing may be rewritten or audited as rerouted."""
        self._arm([_reroute_rule("openai", "gpt-4o-mini")])

        enforcer._run_sync_check(model_hint="gpt-4o", provider="openai")

        self.assertNotIn("_tp_routing", self._session.metadata or {})
        # `_local_decision` (old flat slot) is never written by any code
        # path any more, so this would be vacuously true regardless of a
        # phantom-apply bug; assert against the keyed store instead.
        self.assertFalse(getattr(self._session, "_local_decisions", None))


# ── 2. xai-sdk (model on the proto / positional args) ─────────────────
class _FakeXaiChat:
    def __init__(self, model):
        self._proto = NS(model=model, messages=[])

    def sample(self, *a, **k):
        return NS(usage=None)


class _FakeXaiImageClient:
    def sample(self, prompt, model, *a, **k):
        return NS(usage=None)


# _build_xai_pseudo_kwargs recognizes the image surface by class name + module.
_FakeXaiImageClient.__name__ = "Client"
_FakeXaiImageClient.__qualname__ = "Client"
_FakeXaiImageClient.__module__ = "xai_sdk.image"


class TestXaiPreflightContext(_PreflightBase):

    def _wrap(self, cls, method):
        original = getattr(cls, method)
        enforcer._set_manual_wrapper(cls, method, original, "xai", False)
        self._teardowns.append(lambda: setattr(cls, method, original))

    def test_chat_proto_model_reaches_the_check(self):
        client = self._arm()
        self._spy_evaluator()
        self._wrap(_FakeXaiChat, "sample")

        _FakeXaiChat("grok-4").sample()

        self.assertEqual(client.check_sync.call_args.kwargs["model"], "grok-4")
        self.assertEqual(client.check_sync.call_args.kwargs["provider"], "xai")

    def test_image_positional_model_reaches_the_check(self):
        client = self._arm()
        self._spy_evaluator()
        self._wrap(_FakeXaiImageClient, "sample")

        _FakeXaiImageClient().sample("a cat", "grok-2-image")

        self.assertEqual(client.check_sync.call_args.kwargs["model"],
                         "grok-2-image")

    def test_hint_is_not_written_into_the_call_kwargs(self):
        """xai kwargs carry no model — a REROUTE must stay a noop."""
        self._arm([_reroute_rule("xai", "grok-3-mini")])
        self._wrap(_FakeXaiChat, "sample")

        _FakeXaiChat("grok-4").sample()

        self.assertNotIn("_tp_routing", self._session.metadata or {})

    def test_garbage_args_degrade_to_no_hint(self):
        self.assertIsNone(enforcer._xai_model_hint(None))
        self.assertIsNone(enforcer._xai_model_hint(()))
        self.assertIsNone(enforcer._xai_model_hint((object(),)))
        self.assertIsNone(enforcer._xai_model_hint((NS(_proto=NS(model="")),)))
        self.assertIsNone(enforcer._xai_model_hint((NS(_proto=None),)))
        # Non-str proto models are stringified by _build_xai_pseudo_kwargs — the
        # hint stays consistent with what the logged row reports.
        self.assertEqual(enforcer._xai_model_hint((NS(_proto=NS(model=7)),)), "7")


# ── 3. agno ───────────────────────────────────────────────────────────
class _Boom:
    """Every attribute read explodes."""

    def __getattr__(self, name):
        raise RuntimeError("boom")


class TestAgnoCheckCtx(_PreflightBase):

    def test_declared_provider_string_is_mapped(self):
        agent = NS(model=NS(id="gpt-5.4-mini", provider="OpenAI"))
        self.assertEqual(enforcer._agno_check_ctx(agent),
                         ("gpt-5.4-mini", "openai"))

    def test_class_name_fallback_when_provider_is_synthesized(self):
        # agno's base Model turns a missing provider into "<name> (<id>)".
        class Claude:
            id = "claude-sonnet-4-5"
            provider = "Claude (claude-sonnet-4-5)"

        self.assertEqual(enforcer._agno_check_ctx(NS(model=Claude())),
                         ("claude-sonnet-4-5", "anthropic"))

    def test_composite_openai_wrappers_are_not_mislabeled(self):
        """CerebrasOpenAI/LlamaOpenAI are NOT openai — exact match, not substring."""
        class CerebrasOpenAI:
            id = "llama-3.3-70b"
            provider = "CerebrasOpenAI"

        self.assertEqual(enforcer._agno_check_ctx(NS(model=CerebrasOpenAI())),
                         ("llama-3.3-70b", None))

    def test_unknown_model_class_keeps_the_id_and_drops_the_provider(self):
        class WhoKnows:
            id = "some-model"

        self.assertEqual(enforcer._agno_check_ctx(NS(model=WhoKnows())),
                         ("some-model", None))

    def test_hostile_objects_never_raise(self):
        self.assertEqual(enforcer._agno_check_ctx(_Boom()), (None, None))
        self.assertEqual(enforcer._agno_check_ctx(NS(model=_Boom())), (None, None))
        self.assertEqual(enforcer._agno_check_ctx(None), (None, None))
        self.assertEqual(enforcer._agno_check_ctx(NS(model=None)), (None, None))

    def test_run_wrapper_threads_the_ctx(self):
        client = self._arm()
        self._spy_evaluator()

        def _run(self, *a, **k):
            return "ok"

        agent = NS(model=NS(id="claude-sonnet-4-5", provider="Anthropic"),
                   stream=False)
        wrapper = enforcer._make_agno_run_wrapper(_run)

        self.assertEqual(wrapper(agent), "ok")
        self.assertEqual(client.check_sync.call_args.kwargs["model"],
                         "claude-sonnet-4-5")
        self.assertEqual(client.check_sync.call_args.kwargs["provider"],
                         "anthropic")

    def test_hostile_agent_still_runs_the_check_bare(self):
        client = self._arm()
        self._spy_evaluator()

        def _run(self, *a, **k):
            return "ok"

        wrapper = enforcer._make_agno_run_wrapper(_run)

        self.assertEqual(wrapper(_Boom()), "ok")
        self.assertTrue(client.check_sync.called)
        self.assertIsNone(client.check_sync.call_args.kwargs["model"])
        self.assertIsNone(client.check_sync.call_args.kwargs["provider"])


# ── 4. pydantic_ai ────────────────────────────────────────────────────
class TestPydanticAiCheckCtx(_PreflightBase):

    def test_model_name_and_system_populate_the_ctx(self):
        self.assertEqual(enforcer._pa_check_ctx(NS(model_name="gpt-4o",
                                                   system="openai")),
                         ("gpt-4o", "openai"))
        self.assertEqual(enforcer._pa_check_ctx(NS(model_name="claude-sonnet-4-5",
                                                   system="anthropic")),
                         ("claude-sonnet-4-5", "anthropic"))

    def test_framework_fallback_provider_is_dropped(self):
        """"pydantic_ai" is not a serving vendor — omit rather than mislabel."""
        self.assertEqual(enforcer._pa_check_ctx(NS(model_name="m", system="")),
                         ("m", None))

    def test_garbage_never_raises(self):
        self.assertEqual(enforcer._pa_check_ctx(None), (None, None))
        self.assertEqual(enforcer._pa_check_ctx(_Boom()), (None, None))

    def test_request_wrapper_threads_the_ctx(self):
        self._arm()
        self._spy_evaluator()

        async def _original(self, *a, **k):
            return NS(usage=None, model_name="gpt-4o", parts=[])

        wrapper = enforcer._make_pydantic_ai_request_wrapper(_original)
        model = NS(model_name="gpt-4o", system="openai")
        _run_coro(wrapper(model, []))

        self.assertEqual(self._async_checks[0]["model"], "gpt-4o")
        self.assertEqual(self._async_checks[0]["provider"], "openai")


# ── 5. LangChain / LlamaIndex chat ────────────────────────────────────
def _chat_cls(name, module):
    cls = type(name, (), {"model": "m-x"})
    cls.__module__ = module
    return cls


class TestLangChainCheckCtx(_PreflightBase):

    def test_module_root_map(self):
        for module, expected in (
            ("langchain_openai.chat_models.base", "openai"),
            ("langchain_anthropic.chat_models", "anthropic"),
            ("langchain_google_genai.chat_models", "google"),
            ("langchain_google_vertexai.chat_models", "vertex-ai"),
            ("langchain_mistralai.chat_models", "mistral"),
            ("langchain_cohere.chat_models", "cohere"),
            ("langchain_groq.chat_models", "groq"),
            ("langchain_aws.chat_models", "bedrock"),
        ):
            inst = _chat_cls("ChatX", module)()
            self.assertEqual(enforcer._lc_provider_from_instance(inst), expected,
                             module)

    def test_class_name_fallback_for_out_of_tree_subclasses(self):
        inst = _chat_cls("ChatAnthropicVendored", "acme.llms")()
        self.assertEqual(enforcer._lc_provider_from_instance(inst), "anthropic")

    def test_underivable_provider_is_none_never_langchain(self):
        inst = _chat_cls("ChatMystery", "acme.llms")()
        self.assertIsNone(enforcer._lc_provider_from_instance(inst))
        self.assertEqual(enforcer._lc_check_ctx((inst,)), ("m-x", None))

    def test_model_name_alias_is_used_when_model_is_absent(self):
        cls = type("ChatOpenAI", (), {"model_name": "gpt-4o"})
        cls.__module__ = "langchain_openai.chat_models.base"
        self.assertEqual(enforcer._lc_check_ctx((cls(),)), ("gpt-4o", "openai"))

    def test_garbage_never_raises(self):
        self.assertEqual(enforcer._lc_check_ctx(()), (None, None))
        self.assertEqual(enforcer._lc_check_ctx((None,)), (None, None))
        self.assertEqual(enforcer._lc_check_ctx((_Boom(),))[0], None)
        self.assertIsNone(enforcer._lc_provider_from_instance(None))

    def test_sync_wrapper_threads_the_ctx(self):
        client = self._arm()
        self._spy_evaluator()

        cls = type("ChatOpenAI", (), {"model": "gpt-4o"})
        cls.__module__ = "langchain_openai.chat_models.base"

        def _generate(self, messages, **k):
            return NS(generations=[])

        enforcer._set_langchain_wrapper(cls, "_generate", _generate, "sync")
        cls()._generate([])

        self.assertEqual(client.check_sync.call_args.kwargs["model"], "gpt-4o")
        self.assertEqual(client.check_sync.call_args.kwargs["provider"], "openai")

    def test_matching_reroute_stays_a_noop_on_the_chat_path(self):
        """Chat wrappers pass no kwargs dict — nothing to rewrite, so the
        directive must not produce an "applied" audit."""
        self._arm([_reroute_rule("openai", "gpt-4o-mini")])

        cls = type("ChatOpenAI", (), {"model": "gpt-4o"})
        cls.__module__ = "langchain_openai.chat_models.base"

        def _generate(self, messages, **k):
            return NS(generations=[])

        enforcer._set_langchain_wrapper(cls, "_generate", _generate, "sync")
        cls()._generate([])

        self.assertNotIn("_tp_routing", self._session.metadata or {})
        # `_local_decision` (old flat slot) is never written by any code
        # path any more, so this would be vacuously true regardless of a
        # phantom-apply bug; assert against the keyed store instead.
        self.assertFalse(getattr(self._session, "_local_decisions", None))


class TestLlamaIndexCheckCtx(_PreflightBase):

    def test_provider_uses_the_existing_normalizer_with_openai_default(self):
        anthr = type("Anthropic", (), {"model": "claude-sonnet-4-5"})()
        self.assertEqual(enforcer._li_check_ctx((anthr,)),
                         ("claude-sonnet-4-5", "anthropic"))
        mystery = type("Mystery", (), {"model": "m-x"})()
        self.assertEqual(enforcer._li_check_ctx((mystery,)), ("m-x", "openai"))

    def test_garbage_never_raises(self):
        self.assertEqual(enforcer._li_check_ctx(()), (None, None))
        self.assertEqual(enforcer._li_check_ctx((None,)), (None, None))
        self.assertEqual(enforcer._li_check_ctx((_Boom(),))[0], None)

    def test_stream_wrapper_threads_ctx_and_session(self):
        client = self._arm()
        self._spy_evaluator()

        cls = type("OpenAI", (), {"model": "gpt-4o"})
        cls.__module__ = "llama_index.llms.openai"

        def _stream_chat(self, messages, **k):
            return iter(())

        enforcer._set_llamaindex_wrapper(cls, "stream_chat", _stream_chat, "stream")
        list(cls().stream_chat([]))

        self.assertEqual(client.check_sync.call_args.kwargs["model"], "gpt-4o")
        self.assertEqual(client.check_sync.call_args.kwargs["provider"], "openai")


# ── 6. framework embeddings ───────────────────────────────────────────
def _install_fake_lc_embeddings():
    for key in list(sys.modules):
        if key == "langchain_core" or key.startswith("langchain_core."):
            del sys.modules[key]

    class Embeddings:
        def embed_query(self, text):
            return [0.1]

    Embeddings.__module__ = "langchain_openai.embeddings.base"
    lc_core = types.ModuleType("langchain_core")
    lc_emb = types.ModuleType("langchain_core.embeddings")
    lc_emb.Embeddings = Embeddings
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.embeddings"] = lc_emb
    for key in list(enforcer._originals):
        cls = key[0] if isinstance(key, tuple) else None
        if getattr(cls, "__name__", None) == "Embeddings":
            enforcer._originals.pop(key, None)
    enforcer._instrument_langchain_embeddings()
    return Embeddings


def _install_fake_li_embeddings():
    for key in list(sys.modules):
        if key == "llama_index" or key.startswith("llama_index."):
            del sys.modules[key]

    class BaseEmbedding:
        def get_query_embedding(self, text):
            return [0.1]

    BaseEmbedding.__module__ = "llama_index.embeddings.openai"
    li_root = types.ModuleType("llama_index")
    li_core = types.ModuleType("llama_index.core")
    li_emb = types.ModuleType("llama_index.core.embeddings")
    li_emb.BaseEmbedding = BaseEmbedding
    sys.modules["llama_index"] = li_root
    sys.modules["llama_index.core"] = li_core
    sys.modules["llama_index.core.embeddings"] = li_emb
    for key in list(enforcer._originals):
        cls = key[0] if isinstance(key, tuple) else None
        if getattr(cls, "__name__", None) == "BaseEmbedding":
            enforcer._originals.pop(key, None)
    enforcer._instrument_llama_index_embeddings()
    return BaseEmbedding


class TestEmbeddingPreflightContext(_PreflightBase):
    """The hint computation was BELOW the check; hoisting it is the fix. The
    `method_name` last resort stays logging-only — a method name is not a
    model and must never reach the check."""

    def _spy_sync_check(self):
        seen = []

        def _check(**kw):
            seen.append(kw)

        p = mock.patch.object(enforcer, "_run_sync_check", _check)
        p.start()
        self._teardowns.append(p.stop)
        return seen

    def test_langchain_hint_is_hoisted_above_the_check(self):
        seen = self._spy_sync_check()
        cls = _install_fake_lc_embeddings()
        inst = cls()
        inst.model = "text-embedding-3-small"

        inst.embed_query("hi")

        self.assertEqual(seen[0]["model_hint"], "text-embedding-3-small")
        self.assertEqual(seen[0]["provider"], "openai")

    def test_langchain_method_name_is_never_used_as_the_hint(self):
        seen = self._spy_sync_check()
        cls = _install_fake_lc_embeddings()

        cls().embed_query("hi")

        self.assertIsNone(seen[0]["model_hint"])

    def test_llamaindex_hint_is_hoisted_above_the_check(self):
        seen = self._spy_sync_check()
        cls = _install_fake_li_embeddings()
        inst = cls()
        inst.model_name = "text-embedding-ada-002"

        inst.get_query_embedding("hi")

        self.assertEqual(seen[0]["model_hint"], "text-embedding-ada-002")
        self.assertEqual(seen[0]["provider"], "openai")

    def test_llamaindex_method_name_is_never_used_as_the_hint(self):
        seen = self._spy_sync_check()
        cls = _install_fake_li_embeddings()

        cls().get_query_embedding("hi")

        self.assertIsNone(seen[0]["model_hint"])


# ── 7. golden rule: every helper survives garbage ─────────────────────
class TestHelpersNeverRaise(unittest.TestCase):

    def test_instance_model_attr(self):
        self.assertIsNone(enforcer._instance_model_attr(None, "model"))
        self.assertIsNone(enforcer._instance_model_attr(_Boom(), "model"))
        self.assertIsNone(enforcer._instance_model_attr(NS(model=7), "model"))
        self.assertIsNone(enforcer._instance_model_attr(NS(model=""), "model"))
        self.assertEqual(
            enforcer._instance_model_attr(NS(model_name="m"), "model", "model_name"),
            "m")


if __name__ == "__main__":
    unittest.main()
