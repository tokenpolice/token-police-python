"""Failed framework embedding must emit a failed embedding row.

When LangChain / LlamaIndex embedding wrappers raise at the provider, the
exception must re-raise unchanged (GOLDEN RULE) AND `_emit_call_failure_log`
must still ship one /log with operation=embedding + classified call_outcome.

Optional deps (langchain_core / llama_index) are injected as fake modules so
the unit suite does not require them installed.
"""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession


class _StatusError(Exception):
    def __init__(self, msg="Internal Server Error", status_code=500):
        super().__init__(msg)
        self.status_code = status_code


def _session():
    s = TPSession()
    s.user_id = "u"
    s.paid_plan = "free"
    s.workflow_name = "b21"
    s.session_id = "s"
    s.metadata = {}
    return s


class _FakeTP:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kw):
        self.calls.append(kw)


def _install_fake_langchain():
    """Inject a minimal langchain_core.embeddings.Embeddings and instrument it."""
    # Drop any prior fake so re-instrument re-patches cleanly.
    for key in list(sys.modules):
        if key == "langchain_core" or key.startswith("langchain_core."):
            del sys.modules[key]

    class Embeddings:
        model = "text-embedding-3-small"

        def embed_documents(self, texts):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [[0.1, 0.2] for _ in texts]

        def embed_query(self, text):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [0.1, 0.2]

        async def aembed_documents(self, texts):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [[0.1, 0.2] for _ in texts]

        async def aembed_query(self, text):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [0.1, 0.2]

    lc_core = types.ModuleType("langchain_core")
    lc_emb = types.ModuleType("langchain_core.embeddings")
    lc_emb.Embeddings = Embeddings
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.embeddings"] = lc_emb

    # Clear prior originals for this class so instrument re-wraps.
    for key in list(enforcer._originals):
        if key[0] is Embeddings or (
            isinstance(key, tuple) and getattr(key[0], "__name__", "") == "Embeddings"
            and getattr(key[0], "__module__", "") == "__main__"
        ):
            enforcer._originals.pop(key, None)
    # Also clear by name match from previous test runs (class identity changes).
    for key in list(enforcer._originals):
        cls = key[0] if isinstance(key, tuple) else None
        if (
            cls is not None
            and getattr(cls, "__name__", None) == "Embeddings"
            and any(
                m in getattr(cls, "__dict__", {})
                for m in ("embed_documents", "embed_query", "aembed_documents", "aembed_query")
            )
        ):
            # Only drop fakes we installed (module is langchain_core.embeddings after instrument
            # may reassign). Safer: drop any Embeddings whose methods we know.
            if getattr(cls, "__module__", "") in (
                "langchain_core.embeddings",
                __name__,
            ) or cls is Embeddings:
                enforcer._originals.pop(key, None)

    enforcer._instrument_langchain_embeddings()
    return Embeddings


def _install_fake_llamaindex():
    for key in list(sys.modules):
        if key == "llama_index" or key.startswith("llama_index."):
            del sys.modules[key]

    class BaseEmbedding:
        model_name = "text-embedding-ada-002"

        def get_text_embedding(self, text):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [0.1, 0.2]

        def get_query_embedding(self, text):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [0.1, 0.2]

        def get_text_embedding_batch(self, texts):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [[0.1, 0.2] for _ in texts]

        async def aget_text_embedding(self, text):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [0.1, 0.2]

        async def aget_query_embedding(self, text):
            err = getattr(self, "_err", None)
            if err is not None:
                raise err
            return [0.1, 0.2]

    li_root = types.ModuleType("llama_index")
    li_core = types.ModuleType("llama_index.core")
    li_emb = types.ModuleType("llama_index.core.embeddings")
    li_emb.BaseEmbedding = BaseEmbedding
    sys.modules["llama_index"] = li_root
    sys.modules["llama_index.core"] = li_core
    sys.modules["llama_index.core.embeddings"] = li_emb

    for key in list(enforcer._originals):
        cls = key[0] if isinstance(key, tuple) else None
        if cls is not None and getattr(cls, "__name__", None) == "BaseEmbedding":
            enforcer._originals.pop(key, None)

    enforcer._instrument_llama_index_embeddings()
    return BaseEmbedding


def _call_with_mocks(fn, session, tp, *, is_async=False):
    """Run fn under the same session/client/guard mocks the production wrappers need."""
    raised = None
    result = None
    check_patch = (
        mock.patch("token_police.enforcer._run_async_check", new=mock.AsyncMock(return_value=None))
        if is_async
        else mock.patch("token_police.enforcer._run_sync_check", return_value=None)
    )
    with mock.patch("token_police.enforcer.get_current_session", return_value=session), \
         mock.patch("token_police.enforcer.get_client", return_value=tp), \
         check_patch, \
         mock.patch("token_police.enforcer.in_langchain", return_value=False), \
         mock.patch("token_police.enforcer.in_litellm", return_value=False), \
         mock.patch("token_police.enforcer.in_llamaindex", return_value=False), \
         mock.patch("token_police.enforcer.in_pydantic_ai", return_value=False), \
         mock.patch("token_police.enforcer.in_agno", return_value=False), \
         mock.patch("token_police.enforcer.consume_pending_span_name", return_value=None):
        try:
            if is_async:
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(fn())
                finally:
                    loop.close()
            else:
                result = fn()
        except Exception as exc:
            raised = exc
    return result, raised


class TestLangChainEmbeddingFailureLog(unittest.TestCase):
    def setUp(self):
        self.Embeddings = _install_fake_langchain()

    def test_sync_failure_logs_embedding_row_and_reraise(self):
        tp = _FakeTP()
        session = _session()
        err = _StatusError(status_code=500)
        inst = self.Embeddings()
        inst._err = err

        _, raised = _call_with_mocks(
            lambda: inst.embed_documents(["hello"]),
            session,
            tp,
        )
        self.assertIs(raised, err)
        self.assertEqual(len(tp.calls), 1)
        call = tp.calls[0]
        self.assertEqual(call.get("operation"), "embedding")
        self.assertEqual(call.get("provider"), "langchain")
        self.assertEqual(call.get("model"), "text-embedding-3-small")
        outcome = call.get("call_outcome") or {}
        self.assertEqual(outcome.get("status"), "failed")
        self.assertEqual(outcome.get("http_status"), 500)
        self.assertEqual(outcome.get("error_kind"), "server_error")
        # Failure path ships zero tokens.
        self.assertEqual(call.get("input_tokens", 0) or 0, 0)
        self.assertEqual(call.get("output_tokens", 0) or 0, 0)
        # N4 — an embedding failure must carry an embed usage shape, not the
        # client's openai_compatible_chat synth. Framework providers resolve to
        # openai_embeddings, matching their successful siblings.
        usage = call.get("usage")
        self.assertIsInstance(usage, dict)
        self.assertEqual(usage.get("shape"), "openai_embeddings")
        self.assertEqual(usage.get("raw"), {"prompt_tokens": 0, "total_tokens": 0})

    def test_async_failure_logs_embedding_row_and_reraise(self):
        tp = _FakeTP()
        session = _session()
        err = _StatusError(status_code=429)
        inst = self.Embeddings()
        inst._err = err

        _, raised = _call_with_mocks(
            lambda: inst.aembed_query("q"),
            session,
            tp,
            is_async=True,
        )
        self.assertIs(raised, err)
        self.assertEqual(len(tp.calls), 1)
        call = tp.calls[0]
        self.assertEqual(call.get("operation"), "embedding")
        outcome = call.get("call_outcome") or {}
        self.assertEqual(outcome.get("status"), "failed")
        self.assertEqual(outcome.get("http_status"), 429)
        self.assertEqual(outcome.get("error_kind"), "rate_limited")

    def test_success_path_unchanged(self):
        tp = _FakeTP()
        session = _session()
        inst = self.Embeddings()
        inst._err = None

        result, raised = _call_with_mocks(
            lambda: inst.embed_query("hi"),
            session,
            tp,
        )
        self.assertIsNone(raised)
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(len(tp.calls), 1)
        call = tp.calls[0]
        self.assertEqual(call.get("operation"), "embedding")
        # Success path does not attach a failed call_outcome.
        outcome = call.get("call_outcome")
        if outcome is not None:
            self.assertNotEqual(outcome.get("status"), "failed")
        # Approx tokens from composition (non-zero for non-empty input).
        self.assertGreater(call.get("input_tokens") or 0, 0)

    def test_emit_failure_cannot_mask_customer_exception(self):
        # Real path: client.log_sync raises inside @fail_safe emit → still
        # re-raises customer err (logging must never mask provider failure).
        tp = _FakeTP()
        session = _session()
        err = _StatusError(status_code=503)
        inst = self.Embeddings()
        inst._err = err

        def boom_log(**kw):
            raise RuntimeError("log exploded")

        tp.log_sync = boom_log  # type: ignore[method-assign]
        _, raised = _call_with_mocks(
            lambda: inst.embed_documents(["x"]),
            session,
            tp,
        )
        self.assertIs(raised, err)


class TestLlamaIndexEmbeddingFailureLog(unittest.TestCase):
    def setUp(self):
        self.BaseEmbedding = _install_fake_llamaindex()

    def test_sync_failure_logs_embedding_row_and_reraise(self):
        tp = _FakeTP()
        session = _session()
        err = _StatusError(status_code=500)
        inst = self.BaseEmbedding()
        inst._err = err

        _, raised = _call_with_mocks(
            lambda: inst.get_text_embedding("hello"),
            session,
            tp,
        )
        self.assertIs(raised, err)
        self.assertEqual(len(tp.calls), 1)
        call = tp.calls[0]
        self.assertEqual(call.get("operation"), "embedding")
        self.assertEqual(call.get("provider"), "llamaindex")
        self.assertEqual(call.get("model"), "text-embedding-ada-002")
        outcome = call.get("call_outcome") or {}
        self.assertEqual(outcome.get("status"), "failed")
        self.assertEqual(outcome.get("http_status"), 500)
        self.assertEqual(outcome.get("error_kind"), "server_error")
        self.assertEqual(call.get("input_tokens", 0) or 0, 0)
        self.assertEqual(call.get("output_tokens", 0) or 0, 0)
        # N4 — see the LangChain twin above.
        usage = call.get("usage")
        self.assertIsInstance(usage, dict)
        self.assertEqual(usage.get("shape"), "openai_embeddings")
        self.assertEqual(usage.get("raw"), {"prompt_tokens": 0, "total_tokens": 0})

    def test_async_failure_logs_embedding_row_and_reraise(self):
        tp = _FakeTP()
        session = _session()
        err = _StatusError(status_code=401)
        inst = self.BaseEmbedding()
        inst._err = err

        _, raised = _call_with_mocks(
            lambda: inst.aget_query_embedding("q"),
            session,
            tp,
            is_async=True,
        )
        self.assertIs(raised, err)
        self.assertEqual(len(tp.calls), 1)
        call = tp.calls[0]
        self.assertEqual(call.get("operation"), "embedding")
        outcome = call.get("call_outcome") or {}
        self.assertEqual(outcome.get("status"), "failed")
        self.assertEqual(outcome.get("http_status"), 401)
        self.assertEqual(outcome.get("error_kind"), "auth_error")

    def test_success_path_unchanged(self):
        tp = _FakeTP()
        session = _session()
        inst = self.BaseEmbedding()
        inst._err = None

        result, raised = _call_with_mocks(
            lambda: inst.get_query_embedding("hi"),
            session,
            tp,
        )
        self.assertIsNone(raised)
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(len(tp.calls), 1)
        call = tp.calls[0]
        self.assertEqual(call.get("operation"), "embedding")
        outcome = call.get("call_outcome")
        if outcome is not None:
            self.assertNotEqual(outcome.get("status"), "failed")
        self.assertGreater(call.get("input_tokens") or 0, 0)


if __name__ == "__main__":
    unittest.main()


class TestLlamaIndexEmbeddingOriginalProvider(unittest.TestCase):
    """LlamaIndex embedding rows must attribute the underlying vendor.

    The LangChain sibling emits model_extras.original_provider so the server
    can price against the real deployer's table; the LlamaIndex wrapper
    hardcoded {"framework": "llamaindex"}, silently losing the attribution.
    Only BaseEmbedding is patched (no concrete-subclass loop), so the provider
    is derived at call time from the concrete class's module path.
    """

    def setUp(self):
        self.BaseEmbedding = _install_fake_llamaindex()

    def _subclass_in_module(self, module_name):
        """A concrete embedding class as it would live in a provider subpackage."""
        cls = type("FakeEmbedding", (self.BaseEmbedding,), {})
        cls.__module__ = module_name
        return cls

    def _extras_for_module(self, module_name):
        tp = _FakeTP()
        inst = self._subclass_in_module(module_name)()
        inst._err = None
        _, raised = _call_with_mocks(
            lambda: inst.get_query_embedding("hi"), _session(), tp)
        self.assertIsNone(raised)
        self.assertEqual(len(tp.calls), 1)
        return tp.calls[0].get("model_extras")

    def test_openai_subpackage_attributes_openai(self):
        extras = self._extras_for_module("llama_index.embeddings.openai")
        self.assertEqual(extras.get("framework"), "llamaindex")
        self.assertEqual(extras.get("original_provider"), "openai")

    def test_nested_module_path_still_matches(self):
        extras = self._extras_for_module("llama_index.embeddings.openai.base")
        self.assertEqual(extras.get("original_provider"), "openai")

    def test_every_mapped_subpackage(self):
        """Slugs must stay identical to the LangChain table — a wrong slug
        silently breaks server-side price resolution."""
        expected = {
            "openai": "openai",
            "cohere": "cohere",
            "mistralai": "mistral",
            "google_genai": "gemini",
            "gemini": "gemini",
            "huggingface": "huggingface",
            "voyageai": "voyage",
            "together": "together_ai",
            "bedrock": "bedrock",
        }
        for segment, slug in expected.items():
            with self.subTest(segment=segment):
                extras = self._extras_for_module(f"llama_index.embeddings.{segment}")
                self.assertEqual(extras.get("original_provider"), slug)

    def test_unknown_module_is_byte_identical_to_before(self):
        """No guessing: an unrecognised subpackage emits exactly today's extras."""
        extras = self._extras_for_module("llama_index.embeddings.some_new_vendor")
        self.assertEqual(extras, {"framework": "llamaindex"})

    def test_base_class_instance_emits_no_original_provider(self):
        tp = _FakeTP()
        inst = self.BaseEmbedding()
        inst._err = None
        _, raised = _call_with_mocks(
            lambda: inst.get_query_embedding("hi"), _session(), tp)
        self.assertIsNone(raised)
        self.assertEqual(tp.calls[0].get("model_extras"), {"framework": "llamaindex"})

    def test_failed_call_still_reraises_unchanged(self):
        """GOLDEN RULE: provider derivation must not alter failure behavior."""
        tp = _FakeTP()
        inst = self._subclass_in_module("llama_index.embeddings.openai")()
        inst._err = _StatusError()
        _, raised = _call_with_mocks(
            lambda: inst.get_query_embedding("hi"), _session(), tp)
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(len(tp.calls), 1)
        self.assertEqual(tp.calls[0].get("operation"), "embedding")


class TestEmbeddingSuccessEmitFailOpen(unittest.TestCase):
    """GOLDEN RULE: the SUCCESS emitter (`_log_embedding_call`) is @fail_safe.

    The success log runs after the provider already returned the customer's
    embeddings; a raise inside it (here: `log_sync` itself exploding) must be
    swallowed and the provider result returned unchanged — for all four
    wrappers (LangChain sync/async, LlamaIndex sync/async). The failure-path
    twin is covered by test_emit_failure_cannot_mask_customer_exception above.
    """

    def _boom_tp(self):
        tp = _FakeTP()
        tp.boom_calls = 0

        def boom_log(**kw):
            tp.boom_calls += 1
            raise RuntimeError("log exploded")

        tp.log_sync = boom_log  # type: ignore[method-assign]
        return tp

    def test_langchain_sync_success_survives_log_failure(self):
        Embeddings = _install_fake_langchain()
        tp = self._boom_tp()
        inst = Embeddings()
        inst._err = None

        result, raised = _call_with_mocks(
            lambda: inst.embed_documents(["hello"]), _session(), tp)
        self.assertIsNone(raised)
        self.assertEqual(result, [[0.1, 0.2]])
        # The emitter must actually have been driven into the exploding log —
        # a green assert off a never-called emitter would prove nothing.
        self.assertEqual(tp.boom_calls, 1)

    def test_langchain_async_success_survives_log_failure(self):
        Embeddings = _install_fake_langchain()
        tp = self._boom_tp()
        inst = Embeddings()
        inst._err = None

        result, raised = _call_with_mocks(
            lambda: inst.aembed_query("q"), _session(), tp, is_async=True)
        self.assertIsNone(raised)
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(tp.boom_calls, 1)

    def test_llamaindex_sync_success_survives_log_failure(self):
        BaseEmbedding = _install_fake_llamaindex()
        tp = self._boom_tp()
        inst = BaseEmbedding()
        inst._err = None

        result, raised = _call_with_mocks(
            lambda: inst.get_text_embedding("hello"), _session(), tp)
        self.assertIsNone(raised)
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(tp.boom_calls, 1)

    def test_llamaindex_async_success_survives_log_failure(self):
        BaseEmbedding = _install_fake_llamaindex()
        tp = self._boom_tp()
        inst = BaseEmbedding()
        inst._err = None

        result, raised = _call_with_mocks(
            lambda: inst.aget_text_embedding("hello"), _session(), tp,
            is_async=True)
        self.assertIsNone(raised)
        self.assertEqual(result, [0.1, 0.2])
        self.assertEqual(tp.boom_calls, 1)
