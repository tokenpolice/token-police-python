"""Tests for litellm chat-span de-duplication.

`litellm.completion()` calls the provider SDK (OpenAI) internally, so the bundled
opentelemetry-instrumentation-openai emits an `openai_chat` span for that inner
call WHILE TokenPolice's litellm manual wrapper separately logs the authoritative
`openai_compatible_chat` row via `_log_manual` → ~2x cost. The litellm wrapper
holds the `_in_litellm` guard while the inner span finishes, so the SpanProcessor
drops the inner span when `in_litellm()` is True (mirrors the existing pydantic_ai
drop). langchain/llamaindex must NOT be dropped — they defer + flush the inner
span as their single canonical row.
"""
import unittest

from token_police import context as ctx
from token_police.context import TPSession, _in_litellm, _in_pydantic_ai
from token_police.state import set_client
from token_police.telemetry import TokenPoliceSpanProcessor


class _Scope:
    def __init__(self, name):
        self.name = name


class _FakeSpan:
    """Minimal stand-in for an OpenLLMetry-emitted openai chat ReadableSpan."""

    def __init__(self, attributes, name="openai.chat",
                 scope="opentelemetry.instrumentation.openai"):
        self.attributes = attributes
        self.name = name
        self.instrumentation_scope = _Scope(scope)
        self.context = None
        self.parent = None
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


def _openai_chat_span():
    return _FakeSpan({
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.system": "openai",
        "gen_ai.usage.input_tokens": 207,
        "gen_ai.usage.output_tokens": 52,
    })


class TestLitellmChatDedup(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient()
        set_client(self.client)
        self.session = TPSession(
            user_id="u1", paid_plan="pro", workflow_name="wf",
            trace_id="a" * 32, root_span_id="b" * 16,
        )
        self._token = ctx._current_session.set(self.session)
        self.proc = TokenPoliceSpanProcessor()

    def tearDown(self):
        ctx._current_session.reset(self._token)
        set_client(None)

    def test_inner_chat_span_dropped_under_litellm_guard(self):
        # While litellm runs, the inner openai chat span is a duplicate
        # (litellm logs the authoritative row itself) and must be dropped.
        tok = _in_litellm.set(True)
        try:
            self.proc.on_end(_openai_chat_span())
        finally:
            _in_litellm.reset(tok)
        self.assertEqual(self.client.calls, [], "inner litellm span should be dropped")

    def test_direct_chat_span_is_logged(self):
        # Regression guard: outside any litellm guard the SAME span IS logged —
        # we must not suppress direct openai chat telemetry.
        self.assertFalse(_in_litellm.get())
        self.proc.on_end(_openai_chat_span())
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["provider"], "openai")

    def test_pydantic_ai_guard_still_drops(self):
        # The pre-existing pydantic_ai drop must keep working after the change.
        tok = _in_pydantic_ai.set(True)
        try:
            self.proc.on_end(_openai_chat_span())
        finally:
            _in_pydantic_ai.reset(tok)
        self.assertEqual(self.client.calls, [])

    @unittest.expectedFailure  # T-P3 / G3 — litellm is contextvar-only at on_end (see design)
    def test_lost_at_start_contextvar_false_both_sides_drops_inner(self):
        """Invariant: when the litellm wrapper is conceptually in flight but the
        contextvar is already False at on_end (lost-at-start / lost throughout),
        the inner openai chat span must still be dropped.

        Today's guard only consults ``in_litellm()`` at on_end — no stamp, no
        registry. expectedFailure documents the latent G3 gap.
        """
        self.assertFalse(_in_litellm.get())
        # No contextvar signal for the whole lifecycle.
        self.proc.on_end(_openai_chat_span())
        self.assertEqual(
            self.client.calls, [],
            "G3 invariant: inner litellm-duplicate span must drop even when "
            "contextvar is False throughout (needs stamp/registry backstop)",
        )


if __name__ == "__main__":
    unittest.main()
