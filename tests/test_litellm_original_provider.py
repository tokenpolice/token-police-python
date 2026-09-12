"""LiteLLM rows must carry the underlying provider as `original_provider`.

The Python SDK instruments LiteLLM via MANUAL wrapper targets, so every
completion/embedding row logs `provider="litellm"`. The server types
"litellm" as a *framework*, not a provider, and therefore honestly leaves
`original_provider` blank (39/39 blank in the 2026-07-27 verification run) —
even though the deployer is trivially derivable inside the wrapper from the
request model (`anthropic/claude-...`, `gemini/...`, bare `gpt-4o-mini`) or
from the response's `_hidden_params["custom_llm_provider"]`.

Asserts, across all four LiteLLM log paths:
  - non-streaming completion, streaming (latched final usage chunk) and
    embedding rows all ship `model_extras["original_provider"]`
  - the slug is forwarded VERBATIM ("gemini", never canonicalized to "google")
  - `_hidden_params` wins over the model-string resolution
  - the FAILURE path (provider raises) still logs an error row, now carrying
    the hint, and re-raises the original exception (GOLDEN RULE)
  - a hard resolver failure (no litellm module / `get_llm_provider` raises /
    bare unknown model) silently degrades to no hint and never throws
  - non-LiteLLM providers' payloads stay byte-identical (no stray extras)
"""
from __future__ import annotations

import asyncio
import types
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession


# ── Fake litellm ────────────────────────────────────────────────────────────
# Mirrors the real `litellm.get_llm_provider` contract, verified against the
# installed litellm: returns a 4-tuple (stripped_model, provider_slug, api_key,
# api_base) and RAISES (BadRequestError) on a model it cannot attribute.
# TestRealLitellmContract below pins that behavior against the real library.
_REAL_SLUGS = {
    "gpt-4o-mini": "openai",
    "text-embedding-3-small": "openai",
    "anthropic/claude-haiku-4-5-20251001": "anthropic",
    "gemini/gemini-2.5-flash": "gemini",
}


class _FakeBadRequestError(Exception):
    pass


def _fake_get_llm_provider(model, *args, **kwargs):
    slug = _REAL_SLUGS.get(model)
    if slug is None:
        if "/" in model:
            head, tail = model.split("/", 1)
            return (tail, head, None, None)
        raise _FakeBadRequestError(f"LLM Provider NOT provided for model={model}")
    stripped = model.split("/", 1)[1] if "/" in model else model
    return (stripped, slug, None, None)


def _fake_litellm_module(get_provider=_fake_get_llm_provider):
    mod = types.ModuleType("litellm")
    mod.get_llm_provider = get_provider
    return mod


def _with_litellm(get_provider=_fake_get_llm_provider):
    """Install a fake `litellm` in sys.modules for the duration of a `with`."""
    return mock.patch.dict(
        "sys.modules", {"litellm": _fake_litellm_module(get_provider)}
    )


# ── _log_manual harness (mirrors tests/test_gateway_attribution.py) ─────────
def _run_log_manual(provider, result, kwargs=None, args=None, **log_kwargs):
    """Drive _log_manual with a fresh session, capturing log_sync kwargs."""
    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    session = TPSession()
    with mock.patch.object(enforcer, "get_client", return_value=_FakeTP()):
        enforcer._log_manual(provider, session, kwargs or {}, result, 0, None,
                             datetime.now(timezone.utc), args=args, **log_kwargs)
    return captured


def _chat_response(model, hidden=None):
    resp = NS(model=model, usage=NS(prompt_tokens=10, completion_tokens=5))
    if hidden is not None:
        resp._hidden_params = hidden
    return resp


def _final_stream_chunk(model, hidden=None):
    """What the stream wrapper latches and hands to _log_manual as `result`."""
    chunk = NS(model=model,
               usage=NS(prompt_tokens=21, completion_tokens=7),
               choices=[NS(delta=NS(content=None), finish_reason="stop")])
    if hidden is not None:
        chunk._hidden_params = hidden
    return chunk


def _embedding_response(model, hidden=None):
    resp = NS(model=model,
              usage=NS(prompt_tokens=8, total_tokens=8),
              data=[NS(embedding=[0.1, 0.2])])
    if hidden is not None:
        resp._hidden_params = hidden
    return resp


class TestLitellmSuccessPaths(unittest.TestCase):
    def _orig(self, captured):
        return (captured.get("model_extras") or {}).get("original_provider")

    def test_bare_openai_model_resolved_via_get_llm_provider(self):
        with _with_litellm():
            captured = _run_log_manual(
                "litellm", _chat_response("gpt-4o-mini"),
                kwargs={"model": "gpt-4o-mini"})
        self.assertEqual(self._orig(captured), "openai")
        # Reported provider is untouched — the server still sees the framework.
        self.assertEqual(captured["provider"], "litellm")
        # Tokens unaffected.
        self.assertEqual(captured["input_tokens"], 10)
        self.assertEqual(captured["output_tokens"], 5)

    def test_anthropic_prefixed_model(self):
        with _with_litellm():
            captured = _run_log_manual(
                "litellm", _chat_response("claude-haiku-4-5-20251001"),
                kwargs={"model": "anthropic/claude-haiku-4-5-20251001"})
        self.assertEqual(self._orig(captured), "anthropic")

    def test_gemini_slug_forwarded_verbatim_not_canonicalized(self):
        # The SDK stays thin: "gemini" -> "google" is the server's job.
        with _with_litellm():
            captured = _run_log_manual(
                "litellm", _chat_response("gemini-2.5-flash"),
                kwargs={"model": "gemini/gemini-2.5-flash"})
        self.assertEqual(self._orig(captured), "gemini")
        self.assertNotEqual(self._orig(captured), "google")

    def test_streaming_final_chunk_gets_hint(self):
        with _with_litellm():
            captured = _run_log_manual(
                "litellm", _final_stream_chunk("claude-haiku-4-5-20251001"),
                kwargs={"model": "anthropic/claude-haiku-4-5-20251001"})
        self.assertEqual(self._orig(captured), "anthropic")
        self.assertEqual(captured["input_tokens"], 21)
        self.assertEqual(captured["output_tokens"], 7)

    def test_embedding_path_gets_hint(self):
        with _with_litellm():
            captured = _run_log_manual(
                "litellm", _embedding_response("text-embedding-3-small"),
                kwargs={"model": "text-embedding-3-small"},
                operation="embedding", shape_override="openai_embeddings")
        self.assertEqual(self._orig(captured), "openai")
        self.assertEqual(captured["operation"], "embedding")
        self.assertEqual(captured["usage"]["shape"], "openai_embeddings")
        self.assertEqual(captured["input_tokens"], 8)

    def test_hidden_params_wins_over_model_string(self):
        # LiteLLM stamps the provider it actually dispatched to; it beats any
        # inference from the request model (e.g. azure deployments, routers).
        with _with_litellm():
            captured = _run_log_manual(
                "litellm", _chat_response("gpt-4o-mini",
                                          hidden={"custom_llm_provider": "azure"}),
                kwargs={"model": "gpt-4o-mini"})
        self.assertEqual(self._orig(captured), "azure")

    def test_hidden_params_used_when_litellm_module_absent(self):
        with mock.patch.dict("sys.modules", {"litellm": None}):
            captured = _run_log_manual(
                "litellm", _final_stream_chunk("some-model",
                                               hidden={"custom_llm_provider": "vertex_ai"}),
                kwargs={"model": "some-model"})
        # Verbatim, un-canonicalized ("vertex_ai", not "vertex-ai").
        self.assertEqual(self._orig(captured), "vertex_ai")

    def test_vendor_head_fallback_when_get_llm_provider_raises(self):
        def _boom(model, *a, **k):
            raise RuntimeError("resolver exploded")

        with _with_litellm(_boom):
            captured = _run_log_manual(
                "litellm", _chat_response("mystery-1"),
                kwargs={"model": "newvendor/mystery-1"})
        self.assertEqual(self._orig(captured), "newvendor")


class TestLitellmResolverDegradesSilently(unittest.TestCase):
    def test_hard_failure_omits_hint_and_never_throws(self):
        # No litellm module, no hidden params, bare unattributable model →
        # step (d): the hint is simply absent, exactly as before.
        with mock.patch.dict("sys.modules", {"litellm": None}):
            captured = _run_log_manual(
                "litellm", _chat_response("totally-unknown-model"),
                kwargs={"model": "totally-unknown-model"})
        self.assertIsNone(captured.get("model_extras"))
        # The row is still logged in full.
        self.assertEqual(captured["provider"], "litellm")
        self.assertEqual(captured["input_tokens"], 10)
        self.assertEqual(captured["output_tokens"], 5)

    def test_get_llm_provider_raising_on_bare_model_omits_hint(self):
        with _with_litellm():  # fake raises for unmapped bare models
            captured = _run_log_manual(
                "litellm", _chat_response("no-such-model"),
                kwargs={"model": "no-such-model"})
        self.assertIsNone(captured.get("model_extras"))
        self.assertEqual(captured["input_tokens"], 10)

    def test_hostile_hidden_params_is_fail_safe(self):
        class _Hostile:
            model = "gpt-4o-mini"
            usage = NS(prompt_tokens=3, completion_tokens=1)

            @property
            def _hidden_params(self):
                raise RuntimeError("nope")

        with _with_litellm():
            captured = _run_log_manual("litellm", _Hostile(),
                                       kwargs={"model": "gpt-4o-mini"})
        # Falls through to get_llm_provider; tokens still logged.
        self.assertEqual(
            (captured.get("model_extras") or {}).get("original_provider"), "openai")
        self.assertEqual(captured["input_tokens"], 3)

    def test_missing_request_model_omits_hint(self):
        with _with_litellm():
            captured = _run_log_manual("litellm",
                                       NS(usage=NS(prompt_tokens=2, completion_tokens=1)),
                                       kwargs={})
        self.assertIsNone(captured.get("model_extras"))

    def test_non_string_request_model_omits_hint(self):
        with _with_litellm():
            captured = _run_log_manual("litellm", _chat_response("x"),
                                       kwargs={"model": 42})
        self.assertIsNone(captured.get("model_extras"))


class TestNonLitellmUnchanged(unittest.TestCase):
    def test_plain_openai_payload_byte_identical(self):
        with _with_litellm():
            captured = _run_log_manual(
                "openai", _chat_response("gpt-4.1-mini"),
                kwargs={"model": "gpt-4.1-mini"})
        self.assertIsNone(captured.get("model_extras"))
        self.assertEqual(captured["model"], "gpt-4.1-mini")
        self.assertEqual(captured["provider"], "openai")

    def test_prefixed_model_on_non_litellm_provider_gets_no_litellm_hint(self):
        # A vendor-prefixed slug alone must NOT trigger the litellm branch.
        with _with_litellm():
            captured = _run_log_manual(
                "cerebras", _chat_response("meta/llama-3.3-70b"),
                kwargs={"model": "meta/llama-3.3-70b"})
        self.assertIsNone(captured.get("model_extras"))

    def test_resolver_not_consulted_for_non_litellm_provider(self):
        calls = []

        def _spy(model, *a, **k):
            calls.append(model)
            return (model, "openai", None, None)

        with _with_litellm(_spy):
            _run_log_manual("openai", _chat_response("gpt-4o-mini"),
                            kwargs={"model": "gpt-4o-mini"})
        self.assertEqual(calls, [])


# ── Failure path (provider raises) ─────────────────────────────────────────
class _StatusError(Exception):
    def __init__(self, msg="Unauthorized", status_code=401):
        super().__init__(msg)
        self.status_code = status_code


def _install_and_call(*, provider="litellm", model="anthropic/claude-haiku-4-5-20251001",
                      operation=None, is_async=False):
    """Install a one-shot manual litellm wrapper on a throwaway target, call it
    once (the underlying raises), capture the failure-path log_sync kwargs.
    Mirrors tests/test_failed_embedding_usage_shape.py::_install_and_call."""
    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    if is_async:
        class _FakeTarget:
            async def completion(self, *args, **kwargs):
                raise _StatusError()
    else:
        class _FakeTarget:
            def completion(self, *args, **kwargs):
                raise _StatusError()

    original = _FakeTarget.completion
    enforcer._set_manual_wrapper(
        _FakeTarget, "completion", original,
        provider=provider, is_async=is_async,
        framework="litellm", operation=operation,
    )

    session = TPSession()
    session.user_id = "u"
    session.paid_plan = "free"
    session.workflow_name = "f4"
    session.session_id = "s"
    session.metadata = {}

    raised = None
    check_patch = (
        mock.patch("token_police.enforcer._run_async_check",
                   new=mock.AsyncMock(return_value=None))
        if is_async
        else mock.patch("token_police.enforcer._run_sync_check", return_value=None)
    )
    try:
        with mock.patch("token_police.enforcer.get_current_session", return_value=session), \
             mock.patch("token_police.enforcer.get_client", return_value=_FakeTP()), \
             check_patch, \
             mock.patch("token_police.enforcer.in_langchain", return_value=False), \
             mock.patch("token_police.enforcer.in_litellm", return_value=False), \
             mock.patch("token_police.enforcer.in_llamaindex", return_value=False), \
             mock.patch("token_police.enforcer.in_pydantic_ai", return_value=False), \
             mock.patch("token_police.enforcer.in_agno", return_value=False), \
             mock.patch("token_police.enforcer.maybe_register_openai_agents_tracing"), \
             mock.patch("token_police.enforcer.consume_pending_span_name", return_value=None), \
             mock.patch("token_police.enforcer._capture_composition_at"), \
             mock.patch("token_police.enforcer._build_intent", return_value=None):
            try:
                if is_async:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(_FakeTarget().completion(model=model))
                    finally:
                        loop.close()
                else:
                    _FakeTarget().completion(model=model)
            except Exception as exc:
                raised = exc
    finally:
        _FakeTarget.completion = original
        enforcer._originals.pop((_FakeTarget, "completion"), None)

    return captured, raised


class TestLitellmFailurePath(unittest.TestCase):
    def _orig(self, captured):
        return (captured.get("model_extras") or {}).get("original_provider")

    def test_failed_litellm_call_logs_row_with_hint_and_reraises(self):
        with _with_litellm():
            captured, raised = _install_and_call()
        # GOLDEN RULE: the customer's exception propagates unchanged.
        self.assertIsInstance(raised, _StatusError)
        # The error row is still logged...
        self.assertEqual(captured.get("provider"), "litellm")
        # B2: the failure row now carries the BARE model — _stash_attempt_context
        # resolves the LiteLLM route context and strips the request prefix before
        # stashing session._attempted_model, so a failed call lands on the same
        # model row as its successful siblings (which already log the bare
        # provider echo). original_provider below still proves the vendor is
        # recoverable — now via session._attempted_route_vendor, since the
        # prefix that used to carry it is gone from _attempted_model.
        self.assertEqual(captured.get("model"), "claude-haiku-4-5-20251001")
        self.assertIsNotNone(captured.get("call_outcome"))
        # ...and now carries the deployer hint.
        self.assertEqual(self._orig(captured), "anthropic")

    def test_failed_bare_openai_model(self):
        with _with_litellm():
            captured, raised = _install_and_call(model="gpt-4o-mini")
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(self._orig(captured), "openai")

    def test_failed_gemini_slug_verbatim(self):
        with _with_litellm():
            captured, _ = _install_and_call(model="gemini/gemini-2.5-flash")
        self.assertEqual(self._orig(captured), "gemini")

    def test_failed_async_litellm_call(self):
        with _with_litellm():
            captured, raised = _install_and_call(is_async=True)
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(self._orig(captured), "anthropic")

    def test_failed_embedding_keeps_shape_and_gains_hint(self):
        with _with_litellm():
            captured, _ = _install_and_call(model="text-embedding-3-small",
                                            operation="embedding")
        self.assertEqual(captured.get("operation"), "embedding")
        self.assertEqual((captured.get("usage") or {}).get("shape"),
                         "openai_embeddings")
        self.assertEqual(self._orig(captured), "openai")

    def test_failed_unresolvable_model_omits_hint(self):
        with mock.patch.dict("sys.modules", {"litellm": None}):
            captured, raised = _install_and_call(model="totally-unknown-model")
        self.assertIsInstance(raised, _StatusError)
        self.assertIsNone(captured.get("model_extras"))
        self.assertIsNotNone(captured.get("call_outcome"))

    def test_failed_non_litellm_call_ships_no_extras(self):
        with _with_litellm():
            captured, _ = _install_and_call(provider="openai", model="gpt-4o-mini")
        self.assertIsNone(captured.get("model_extras"))


class TestRealLitellmContract(unittest.TestCase):
    """Pins the real library's contract the fake above mirrors. Skipped when
    litellm isn't installed (it is an optional customer dependency)."""

    def test_real_get_llm_provider_slugs(self):
        try:
            import litellm  # noqa: F401
        except Exception:
            self.skipTest("litellm not installed")
        for model, expected in _REAL_SLUGS.items():
            with self.subTest(model=model):
                self.assertEqual(litellm.get_llm_provider(model)[1], expected)


if __name__ == "__main__":
    unittest.main()
