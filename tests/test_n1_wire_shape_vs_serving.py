"""N1 — usage.shape follows the WIRE surface, provider follows the SERVING host.

Made the emitted ``provider`` the serving slug (an Anthropic SDK client
pointed at api.minimax.io reports "minimax"). The Mode-A shape derivation,
however, kept keying off that same serving slug — "minimax" is not in the
shape table, so the span went out as ``openai_compatible_chat`` even though the
bytes are Anthropic-shaped. The server then ran the OpenAI mapper over
Anthropic fields: ``cache_creation_input_tokens`` fell into extra_units (cache
writes billed $0) and the already cache-EXCLUSIVE ``input_tokens`` had cache
reads subtracted a second time.

These assert the two axes at the emitted-payload level:
    shape = wire surface (gen_ai.system / instrumentation scope)
    provider = serving slug (enforcer stash override) must still hold

The gate is anthropic-ONLY: every non-anthropic wire keeps the exact shape it
produced before the fix.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import context as tp_context
from token_police.context import TPSession
from token_police.telemetry import TokenPoliceSpanProcessor


def _fake_span(system=None, scope=None, name="anthropic.chat",
               model="MiniMax-M2", trace_id="", extra_attrs=None):
    attrs = {
        "gen_ai.request.model": model,
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 50,
        "tp.trace_id": trace_id,
        "tp.span_order": 0,
    }
    if system is not None:
        attrs["gen_ai.system"] = system
    if extra_attrs:
        attrs.update(extra_attrs)
    return NS(
        attributes=attrs,
        name=name,
        instrumentation_scope=NS(name=scope) if scope else None,
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        start_time=0,
        end_time=1,
    )


def _emit(override=None, **span_kwargs):
    """Drive on_end with a bound session, return the log_sync kwargs."""
    captured = {}

    class _FakeClient:
        def log_sync(self, **kwargs):
            captured.update(kwargs)

    session = TPSession()
    if override:
        session._pending_compositions = {
            f"{session.trace_id}:0": {"provider": override}
        }
    span = _fake_span(trace_id=session.trace_id, **span_kwargs)

    token = tp_context._current_session.set(session)
    try:
        proc = TokenPoliceSpanProcessor()
        with mock.patch("token_police.state.get_client",
                        return_value=_FakeClient()):
            proc.on_end(span)
    finally:
        tp_context._current_session.reset(token)
    return captured


class TestN1WireShapeVsServing(unittest.TestCase):
    def test_1_anthropic_wire_minimax_serving(self):
        kw = _emit(system="Anthropic", override="minimax")
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        # The emitted provider is the SERVING slug, not the wire vendor.
        self.assertEqual(kw["provider"], "minimax")

    def test_2_anthropic_wire_no_remap_unchanged(self):
        kw = _emit(system="Anthropic")
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        self.assertEqual(kw["provider"], "anthropic")

    def test_3_openai_wire_minimax_serving_stays_compatible_chat(self):
        kw = _emit(system="openai", name="openai.chat", override="minimax")
        self.assertEqual(kw["usage"]["shape"], "openai_compatible_chat")
        self.assertEqual(kw["provider"], "minimax")

    def test_4_openai_wire_openrouter_stays_routed(self):
        kw = _emit(system="openai", name="openai.chat", override="openrouter")
        self.assertEqual(kw["usage"]["shape"], "openrouter_routed")
        self.assertEqual(kw["provider"], "openrouter")

    def test_5_scope_only_signal(self):
        # No gen_ai.system at all — the instrumentation scope is the only wire
        # evidence available.
        kw = _emit(scope="opentelemetry.instrumentation.anthropic",
                   override="minimax")
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        self.assertEqual(kw["provider"], "minimax")

    def test_5b_node_scope_literal_also_accepted(self):
        kw = _emit(scope="@traceloop/instrumentation-anthropic",
                   override="minimax")
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")

    def test_6_genuine_openai_unchanged(self):
        kw = _emit(system="openai", name="openai.chat",
                   scope="opentelemetry.instrumentation.openai",
                   model="gpt-4.1-mini")
        self.assertEqual(kw["usage"]["shape"], "openai_chat")
        self.assertEqual(kw["provider"], "openai")

    def test_7_langchain_gemini_keeps_google_genai(self):
        kw = _emit(system="Google", name="langchain.chat",
                   scope="opentelemetry.instrumentation.langchain",
                   model="models/gemini-2.5-flash")
        self.assertEqual(kw["usage"]["shape"], "google_genai")
        self.assertEqual(kw["provider"], "google")

    def test_8_hostile_scope_name_is_fail_open(self):
        # A scope object whose name stringifies explosively: the wire-signal
        # block must swallow it and leave the pre-existing provider lookup in
        # charge — never drop or crash the log.
        class _BoomName:
            def __str__(self):
                raise RuntimeError("nope")

        captured = {}

        class _FakeClient:
            def log_sync(self, **kwargs):
                captured.update(kwargs)

        session = TPSession()
        span = _fake_span(name="cerebras.chat", trace_id=session.trace_id)
        span.instrumentation_scope = NS(name=_BoomName())
        token = tp_context._current_session.set(session)
        try:
            with mock.patch("token_police.state.get_client",
                            return_value=_FakeClient()):
                TokenPoliceSpanProcessor().on_end(span)
        finally:
            tp_context._current_session.reset(token)
        # Fail-open: the pre-existing provider lookup still decides, and the
        # span is still logged.
        self.assertEqual(captured["usage"]["shape"], "openai_compatible_chat")

    def test_9_anthropic_cache_write_on_native_key(self):
        kw = _emit(
            system="Anthropic",
            override="minimax",
            extra_attrs={
                "gen_ai.usage.cache_read.input_tokens": 200,
                "gen_ai.usage.cache_creation_input_tokens": 1000,
            },
        )
        usage = kw["usage"]
        self.assertEqual(usage["shape"], "anthropic_messages")
        # The server's anthropic mapper reads these two keys; under the old
        # openai_compatible_chat shape the creation tokens were unpriced extras.
        self.assertEqual(usage["raw"]["cache_creation_input_tokens"], 1000)
        self.assertEqual(usage["raw"]["cache_read_input_tokens"], 200)


if __name__ == "__main__":
    unittest.main()
