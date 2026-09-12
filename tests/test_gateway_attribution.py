"""OpenAI-compatible gateway attribution on the manual-telemetry log path.

Fix: `_log_manual` never sent `model_extras`, so an OpenRouter-routed call
(OpenAI SDK with base_url=openrouter.ai, or the native openrouter SDK) lost its
gateway identity — the server saw provider="openai", no api_base, and no
deployer hint. The fix is narrowly gated: model_extras is built ONLY when the
call is detectably gateway-routed (stashed provider override, gateway base_url,
or the native openrouter provider); every other call's payload is unchanged.

The model string must stay FULLY vendor-prefixed ("google/gemini-2.5-flash-lite")
and original_provider must carry the vendor slug head ("google").
"""
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession


class _FakeHttpClient:
    def __init__(self, base_url):
        self.base_url = base_url


class _FakeBoundSelf:
    """Mimics the SDK resource instance whose ._client carries base_url."""

    def __init__(self, base_url):
        self._client = _FakeHttpClient(base_url)


def _openrouter_response(model="google/gemini-2.5-flash-lite"):
    return NS(model=model,
              usage=NS(prompt_tokens=10, completion_tokens=5))


def _run_log_manual(provider, result, kwargs=None, args=None, stash=None):
    """Drive _log_manual with a fresh session, capturing log_sync kwargs."""
    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    session = TPSession()
    if stash is not None:
        session._pending_compositions = {f"{session.trace_id}:0": dict(stash)}
    with mock.patch.object(enforcer, "get_client", return_value=_FakeTP()):
        enforcer._log_manual(provider, session, kwargs or {}, result, 0, None,
                             datetime.now(timezone.utc), args=args)
    return captured


class TestGatewayAttribution(unittest.TestCase):
    def test_openai_sdk_via_openrouter_base_url(self):
        args = (_FakeBoundSelf("https://openrouter.ai/api/v1"),)
        captured = _run_log_manual("openai", _openrouter_response(), args=args)
        extras = captured.get("model_extras")
        self.assertIsNotNone(extras)
        self.assertEqual(extras["original_provider"], "google")
        self.assertIn("openrouter.ai", extras["api_base"])
        # Model stays FULLY vendor-prefixed — never stripped.
        self.assertEqual(captured["model"], "google/gemini-2.5-flash-lite")
        # Reported provider is unchanged (server remaps via api_base).
        self.assertEqual(captured["provider"], "openai")

    def test_stashed_provider_override_used(self):
        stash = {"provider": "openrouter",
                 "api_base": "https://openrouter.ai/api/v1"}
        captured = _run_log_manual("openai", _openrouter_response(), stash=stash)
        extras = captured.get("model_extras")
        self.assertIsNotNone(extras)
        self.assertEqual(extras["original_provider"], "google")
        self.assertEqual(extras["api_base"], "https://openrouter.ai/api/v1")

    def test_native_openrouter_sdk_gets_original_provider(self):
        # Native openrouter SDK: no _client/base_url, gate (c) engages on the
        # provider itself; original_provider derives from the model slug.
        captured = _run_log_manual(
            "openrouter", _openrouter_response("anthropic/claude-3.5-haiku"))
        extras = captured.get("model_extras")
        self.assertIsNotNone(extras)
        self.assertEqual(extras["original_provider"], "anthropic")
        self.assertEqual(captured["model"], "anthropic/claude-3.5-haiku")

    def test_plain_openai_call_unchanged(self):
        # Non-gateway base_url → model_extras must stay absent (None).
        args = (_FakeBoundSelf("https://api.openai.com/v1"),)
        captured = _run_log_manual(
            "openai", NS(model="gpt-4.1-mini",
                         usage=NS(prompt_tokens=4, completion_tokens=2)),
            args=args)
        self.assertIsNone(captured.get("model_extras"))
        self.assertEqual(captured["model"], "gpt-4.1-mini")

    def test_non_openai_provider_without_gateway_unchanged(self):
        captured = _run_log_manual(
            "cerebras", NS(model="llama-3.3-70b",
                           usage=NS(prompt_tokens=4, completion_tokens=2)))
        self.assertIsNone(captured.get("model_extras"))

    def test_gateway_detection_failure_is_fail_open(self):
        # A hostile client object must not break logging — extras just stay off.
        class _Boom:
            @property
            def _client(self):
                raise RuntimeError("nope")

        captured = _run_log_manual("openai", _openrouter_response(),
                                   args=(_Boom(),))
        # Tokens still logged.
        self.assertEqual(captured["input_tokens"], 10)
        self.assertEqual(captured["output_tokens"], 5)

    def test_tokens_and_usage_block_unaffected_by_extras(self):
        args = (_FakeBoundSelf("https://openrouter.ai/api/v1"),)
        captured = _run_log_manual("openai", _openrouter_response(), args=args)
        self.assertEqual(captured["input_tokens"], 10)
        self.assertEqual(captured["output_tokens"], 5)
        self.assertEqual(captured["usage"]["shape"], "openai_chat")


def _fake_otel_span(attrs, status=None):
    """Minimal finished OTel span for TokenPoliceSpanProcessor.on_end."""
    return NS(
        attributes=attrs,
        name="openai.chat",
        instrumentation_scope=NS(name="opentelemetry.instrumentation.openai"),
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        status=status,
        start_time=0,
        end_time=1,
    )


def _run_on_end(session, attrs, status=None):
    """Drive telemetry on_end against `session`, capturing log_sync kwargs."""
    from token_police.telemetry import TokenPoliceSpanProcessor

    captured = {}

    class _FakeClient:
        def log_sync(self, **kw):
            captured.update(kw)

    with mock.patch("token_police.state.get_client",
                    return_value=_FakeClient()), \
         mock.patch("token_police.context.get_current_session",
                    return_value=session):
        TokenPoliceSpanProcessor().on_end(_fake_otel_span(attrs, status=status))
    return captured


class TestInstrumentedGatewayAttribution(unittest.TestCase):
    """Round 2: openai-SDK-via-OpenRouter calls log via telemetry.py's span
    on_end (NOT _log_manual), and Traceloop strips the vendor prefix from
    gen_ai.request.model before we see it ('openai/gpt-4.1-nano' ->
    'gpt-4.1-nano'). The enforcer's gateway detection now ALSO stashes the
    verbatim slug + original_provider; on_end must restore the full slug and
    forward model_extras = {original_provider, api_base}. Non-gateway spans
    stay byte-identical."""

    def _llm_attrs(self, session, model="gpt-4.1-nano"):
        return {
            "gen_ai.system": "openai",
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 5,
            "tp.trace_id": session.trace_id,
            "tp.span_order": 0,
        }

    def test_gateway_stash_restores_full_slug_and_extras(self):
        session = TPSession()
        # Simulate the Mode-A wrapper's pre-call stashes on gateway detection.
        enforcer._stash_provider_override("openrouter", session)
        enforcer._stash_gateway_request_model(
            session, {"model": "openai/gpt-4.1-nano"})
        enforcer._stash_api_base(session, "https://openrouter.ai/api/v1")
        captured = _run_on_end(session, self._llm_attrs(session))
        # Full vendor-prefixed slug restored over the stripped attr value.
        self.assertEqual(captured["model"], "openai/gpt-4.1-nano")
        self.assertEqual(captured["provider"], "openrouter")
        extras = captured.get("model_extras")
        self.assertIsNotNone(extras)
        self.assertEqual(extras["original_provider"], "openai")
        self.assertIn("openrouter.ai", extras["api_base"])
        # Tokens unaffected.
        self.assertEqual(captured["input_tokens"], 10)
        self.assertEqual(captured["output_tokens"], 5)

    def test_unprefixed_slug_omits_original_provider(self):
        session = TPSession()
        enforcer._stash_provider_override("openrouter", session)
        enforcer._stash_gateway_request_model(session, {"model": "gpt-4.1-nano"})
        enforcer._stash_api_base(session, "https://openrouter.ai/api/v1")
        captured = _run_on_end(session, self._llm_attrs(session))
        extras = captured.get("model_extras")
        self.assertIsNotNone(extras)
        self.assertNotIn("original_provider", extras)
        self.assertIn("openrouter.ai", extras["api_base"])

    def test_non_gateway_span_byte_identical(self):
        # No gateway stash → no model_extras key, model stays as the attr.
        session = TPSession()
        captured = _run_on_end(session, self._llm_attrs(session))
        self.assertEqual(captured["model"], "gpt-4.1-nano")
        self.assertEqual(captured["provider"], "openai")
        self.assertNotIn("model_extras", captured)

    def test_stash_helper_is_fail_open(self):
        # Hostile inputs must never raise (GOLDEN RULE) and never stash.
        session = TPSession()
        enforcer._stash_gateway_request_model(session, None)
        enforcer._stash_gateway_request_model(session, {})
        enforcer._stash_gateway_request_model(session, {"model": 42})
        self.assertEqual(getattr(session, "_pending_compositions", {}) or {}, {})


if __name__ == "__main__":
    unittest.main()
