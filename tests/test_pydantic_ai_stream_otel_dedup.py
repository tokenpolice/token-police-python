"""pydantic_ai streaming must not double-log the inner Anthropic OTel span.

Regression for real-apps support_agent_pydantic_ai_python --provider anthropic
--streaming: each real Model.request_stream step produced TWO llm /log rows —
one good (manual pydantic_ai: user_id + composition) and one ghost (OTel
AnthropicAsyncStream: end_user_id=anonymous, empty composition, workflow=default).

Root cause: on_start early-returned under `_in_pydantic_ai` without marking the
span suppressed; OTel ends the stream span after request_stream __aexit__
cleared the guard, so on_end's in_pydantic_ai() check missed and emitted the
ghost. Fix: stamp tp.suppress on on_start under the guard (survives contextvar
timing) + keep the guard through the underlying manager __aexit__.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import context as ctx
from token_police.context import TPSession, _in_pydantic_ai
from token_police.state import set_client
from token_police import enforcer
from token_police.telemetry import TokenPoliceSpanProcessor


class _Scope:
    def __init__(self, name):
        self.name = name


class _FakeSpan:
    """Writable attributes so on_start set_attribute is observable."""

    def __init__(self, attributes=None, name="anthropic.chat",
                 scope="opentelemetry.instrumentation.anthropic"):
        self.attributes = dict(attributes or {})
        self.name = name
        self.instrumentation_scope = _Scope(scope)
        self.context = NS(trace_id=0x1111, span_id=0x2222)
        self.parent = None
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


def _anthropic_usage_attrs():
    return {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-haiku-4-5-20251001",
        "gen_ai.usage.input_tokens": 803,
        "gen_ai.usage.output_tokens": 124,
    }


class TestPydanticAiStreamOtelDedup(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient()
        set_client(self.client)
        self.session = TPSession(
            user_id="usr_pro_001",
            paid_plan="pro",
            workflow_name="support_desk_agent_pydantic_ai_python",
            trace_id="a" * 32,
            root_span_id="b" * 16,
            session_id="sess-1",
        )
        self._token = ctx._current_session.set(self.session)
        self.proc = TokenPoliceSpanProcessor()

    def tearDown(self):
        ctx._current_session.reset(self._token)
        set_client(None)
        # Never leave the framework guard stuck True across tests.
        try:
            _in_pydantic_ai.set(False)
        except Exception:
            pass

    def test_on_start_stamps_suppress_under_pydantic_ai_guard(self):
        span = _FakeSpan(_anthropic_usage_attrs())
        tok = _in_pydantic_ai.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_pydantic_ai.reset(tok)
        self.assertTrue(
            span.attributes.get("tp.suppress"),
            "on_start under pydantic_ai must stamp tp.suppress so late on_end drops",
        )
        # Must not reserve/consume a span_order for a suppressed inner span.
        self.assertNotIn("tp.span_order", span.attributes)
        self.assertNotIn("tp.user_id", span.attributes)

    def test_on_end_drops_suppressed_even_after_guard_cleared(self):
        """The streaming race: guard already False when OTel finishes the span."""
        span = _FakeSpan({
            **_anthropic_usage_attrs(),
            "tp.suppress": True,
        })
        self.assertFalse(_in_pydantic_ai.get())
        self.proc.on_end(span)
        self.assertEqual(
            self.client.calls, [],
            "tp.suppress must drop the ghost even when in_pydantic_ai() is False",
        )

    def test_on_end_without_suppress_still_logs_when_guard_clear(self):
        """Outside pydantic_ai, the same OTel span remains a real telemetry row."""
        span = _FakeSpan(_anthropic_usage_attrs())
        # Enrich as a normal on_start would (no framework guard).
        self.proc.on_start(span)
        self.proc.on_end(span)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["user_id"], "usr_pro_001")

    def test_aexit_keeps_guard_until_after_mgr_exit(self):
        """Guard must still be True while the underlying stream manager exits
        (where AnthropicAsyncStream often ends its OTel span)."""
        guard_at_mgr_exit = []

        class _Mgr:
            async def __aenter__(self):
                return NS(get=lambda: None)

            async def __aexit__(self, *a):
                guard_at_mgr_exit.append(_in_pydantic_ai.get())
                return False

        model = NS(
            model_name="claude-haiku-4-5-20251001",
            system="anthropic",
        )
        # Bypass check / composition; drive enter+exit with the real wrapper.
        wrapper = enforcer._PydanticAIAsyncStreamMgr(_Mgr(), model, messages=[])

        async def _run():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_log_pydantic_ai"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                async with wrapper:
                    self.assertTrue(_in_pydantic_ai.get())
                # After full exit the guard must be cleared.
                self.assertFalse(_in_pydantic_ai.get())

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()

        self.assertEqual(guard_at_mgr_exit, [True])

    def test_end_to_end_streaming_race_yields_single_row(self):
        """Full lifecycle of the actual regression: the OTel span starts inside
        the guarded window (during the provider create call) but ends only
        AFTER the wrapper has fully exited and cleared the guard. Exactly one
        priced row must survive — the manual pydantic_ai one."""
        span = _FakeSpan(_anthropic_usage_attrs())
        proc = self.proc

        class _Mgr:
            async def __aenter__(_self):
                # The provider SDK call runs inside the guard → OTel on_start.
                proc.on_start(span)
                return NS(get=lambda: None)

            async def __aexit__(_self, *a):
                return False

        model = NS(model_name="claude-haiku-4-5-20251001", system="anthropic")
        wrapper = enforcer._PydanticAIAsyncStreamMgr(_Mgr(), model, messages=[])
        manual_log = mock.MagicMock()

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_log_pydantic_ai", new=manual_log), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                async with wrapper:
                    pass

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_drive())
        finally:
            loop.close()

        # OTel ends the stream span late — the guard is already cleared.
        self.assertFalse(_in_pydantic_ai.get())
        proc.on_end(span)

        manual_log.assert_called_once()  # the single real row
        self.assertEqual(
            self.client.calls, [],
            "late OTel on_end must not emit a second (ghost) /log row",
        )

    @unittest.expectedFailure  # T-P3 / G3 — stamp-only; no registry backstop (see design)
    def test_lost_at_start_contextvar_false_both_sides_drops_ghost(self):
        """Invariant under adversarial timing: contextvar False at on_start AND
        on_end must still drop the inner OTel span when a pydantic_ai call is
        conceptually in flight.

        Stamp-only design cannot satisfy this (same class as pre-registry ).
        Marked expectedFailure until a registry (or equivalent) lands — the red
        documents the latent gap rather than encoding a false green.
        """
        self.assertFalse(_in_pydantic_ai.get())
        span = _FakeSpan(_anthropic_usage_attrs())
        # No guard signal at start → no stamp.
        self.proc.on_start(span)
        self.assertFalse(
            span.attributes.get("tp.suppress"),
            "without contextvar, stamp-only cannot mark suppress",
        )
        # Still no guard at end — stamp-only logs a ghost row today.
        self.assertFalse(_in_pydantic_ai.get())
        self.proc.on_end(span)
        self.assertEqual(
            self.client.calls, [],
            "G3 invariant: ghost must be dropped even when contextvar is "
            "False throughout (needs registry backstop, not stamp-only)",
        )


if __name__ == "__main__":
    unittest.main()
