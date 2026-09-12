"""I1 — phantom parent / missing anchor for unscoped calls.

Unscoped get_current_session() mints a throwaway TPSession with a random
root_span_id that is never logged as an agent/chain row. Manual, Mode A,
tools, and hardcode bypasses must parent with "" in that case — not the
throwaway root. Scoped sessions set session._anchored when the structural
span binds real OTel ids; post-scope finalize holding that object must still
parent to the agent root.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from opentelemetry import trace

from token_police.context import (
    TPSession,
    get_current_session,
    manual_span_ids,
    _session_parent_span_id,
    _structural_span,
)
from token_police import telemetry
from token_police.telemetry import TokenPoliceSpanProcessor
from token_police import context as ctx


class TestSessionParentHelper(unittest.TestCase):
    def test_unanchored_throwaway_parent_empty(self):
        s = TPSession()
        self.assertFalse(s._anchored)
        self.assertEqual(_session_parent_span_id(s), "")
        self.assertEqual(manual_span_ids(s)["parent_span_id"], "")

    def test_get_current_session_throwaway_unscoped(self):
        s = get_current_session()
        self.assertFalse(getattr(s, "_anchored", False))
        self.assertEqual(manual_span_ids(s)["parent_span_id"], "")

    def test_anchored_session_fallback_is_root(self):
        s = TPSession(root_span_id="c" * 16)
        s._anchored = True
        self.assertEqual(_session_parent_span_id(s), "c" * 16)
        # No live OTel span → fallback to anchored root.
        self.assertEqual(manual_span_ids(s)["parent_span_id"], "c" * 16)

    def test_post_scope_held_session_still_parents(self):
        """Flag lives on the object, not 'is contextvar set now'."""
        held = None

        def body():
            nonlocal held
            held = get_current_session()
            self.assertTrue(held._anchored)
            return held.root_span_id

        # Open a real structural span via session/workflow-style path.
        import token_police as tp

        tp.init(api_key="tp_sk_test_i1", base_url="http://localhost:59999")
        try:
            with tp.session(name="post_scope_probe") as s:
                root = s.root_span_id
                held = s
                self.assertTrue(s._anchored)
            # Context exited — contextvar unset.
            self.assertIs(ctx._current_session.get(), None)
            # Held object still anchored → parent = agent root.
            self.assertIsNotNone(held)
            self.assertTrue(held._anchored)
            self.assertEqual(manual_span_ids(held)["parent_span_id"], root)
            self.assertEqual(_session_parent_span_id(held), root)
        finally:
            tp.uninstrument()


class TestModeAUnscopedParent(unittest.TestCase):
    """Mode A on_start stamps "" when unanchored; on_end parent is not phantom."""

    def setUp(self):
        telemetry._span_parent.clear()
        telemetry._kept_spans.clear()
        self.proc = TokenPoliceSpanProcessor()

    def test_unscoped_on_start_stamps_empty_root(self):
        span = NS(
            attributes={},
            name="ChatOpenAI.chat",
            context=NS(trace_id=0xABCD, span_id=0x1111),
            parent=None,
            instrumentation_scope=NS(name="opentelemetry.instrumentation.openai"),
        )
        stamped = {}

        def set_attribute(k, v):
            stamped[k] = v

        span.set_attribute = set_attribute
        # Outside any session → throwaway unanchored.
        self.proc.on_start(span)
        self.assertEqual(stamped.get("tp.root_span_id"), "")

    def test_scoped_on_start_stamps_real_root(self):
        import token_police as tp

        tp.init(api_key="tp_sk_test_i1_scoped_start", base_url="http://localhost:59999")
        try:
            with tp.session(name="scoped_stamp") as s:
                self.assertTrue(s._anchored)
                span = NS(
                    attributes={},
                    name="ChatOpenAI.chat",
                    context=NS(trace_id=0xABCD, span_id=0x2222),
                    parent=None,
                    instrumentation_scope=NS(
                        name="opentelemetry.instrumentation.openai"
                    ),
                )
                stamped = {}

                def set_attribute(k, v):
                    stamped[k] = v

                span.set_attribute = set_attribute
                self.proc.on_start(span)
                self.assertEqual(stamped.get("tp.root_span_id"), s.root_span_id)
                self.assertTrue(stamped.get("tp.root_span_id"))
        finally:
            tp.uninstrument()

    def test_unscoped_on_end_parent_empty_no_otel_parent(self):
        """LLM row with empty stamped root + no OTel parent → parent_span_id ""."""
        L = 0x4444
        span = NS(
            context=NS(trace_id=0x1234, span_id=L),
            parent=None,
            attributes={
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 5,
                "gen_ai.usage.output_tokens": 3,
                "tp.root_span_id": "",  # unanchored stamp
                "tp.trace_id": "a" * 32,
                "tp.user_id": "anonymous",
                "tp.workflow_name": "default_workflow",
            },
            name="ChatOpenAI.chat",
            instrumentation_scope=NS(name="opentelemetry.instrumentation.openai"),
            start_time=1,
            end_time=2,
            status=None,
        )
        captured = {}

        class _FakeClient:
            def log_sync(self, **kwargs):
                captured.update(kwargs)

        with mock.patch("token_police.state.get_client", return_value=_FakeClient()):
            self.proc.on_end(span)

        self.assertIn("span", captured)
        parent = captured["span"]["parent_span_id"]
        self.assertEqual(parent, "")

    def test_resolve_kept_parent_empty_root_stays_empty(self):
        self.assertEqual(telemetry._resolve_kept_parent("", ""), "")
        self.assertEqual(telemetry._resolve_kept_parent("", "deadbeefdeadbeef"), "deadbeefdeadbeef")


class TestBedrockAndAgentsBypass(unittest.TestCase):
    def test_bedrock_uses_session_parent_helper(self):
        from token_police.context import _session_parent_span_id

        unscoped = TPSession()
        self.assertEqual(_session_parent_span_id(unscoped), "")
        scoped = TPSession(root_span_id="d" * 16)
        scoped._anchored = True
        self.assertEqual(_session_parent_span_id(scoped), "d" * 16)

    def test_bedrock_embed_log_parent_empty_when_unanchored(self):
        """Wire-level: _log_bedrock_embedding must not parent to throwaway root."""
        from token_police.state import set_client
        from token_police.enforcer import _log_bedrock_embedding

        class _FakeClient:
            def __init__(self):
                self.calls = []

            def log_sync(self, **kwargs):
                self.calls.append(kwargs)

        client = _FakeClient()
        set_client(client)
        # Throwaway unanchored session in context.
        s = TPSession(root_span_id="e" * 16)  # non-empty root, but not anchored
        self.assertFalse(s._anchored)
        token = ctx._current_session.set(s)
        try:
            _log_bedrock_embedding(
                "amazon.titan-embed-text-v1",
                {"body": b"{}"},
                {},
                "embed",
                None,
            )
        finally:
            ctx._current_session.reset(token)

        self.assertGreaterEqual(len(client.calls), 1)
        parent = client.calls[0]["span"]["parent_span_id"]
        self.assertEqual(parent, "")
        self.assertNotEqual(parent, "e" * 16)

    def test_openai_agents_unscoped_tool_parent_empty(self):
        from token_police.state import set_client
        from token_police.openai_agents import _handle_span_end

        class _FakeClient:
            def __init__(self):
                self.calls = []

            def log_sync(self, **kwargs):
                self.calls.append(kwargs)

        class _SD:
            type = "function"
            name = "probe"
            input = "{}"
            output = "ok"

        class _Span:
            span_data = _SD()
            started_at = "2026-06-03T10:00:00Z"
            ended_at = "2026-06-03T10:00:01Z"
            error = None

        client = _FakeClient()
        set_client(client)
        # No context session → throwaway unanchored.
        token = ctx._current_session.set(None)
        try:
            _handle_span_end(_Span())
        finally:
            ctx._current_session.reset(token)

        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["span"]["parent_span_id"], "")


class TestStructuralSpanSetsAnchored(unittest.TestCase):
    def test_structural_span_sets_anchored_when_otel_binds(self):
        import token_police as tp

        tp.init(api_key="tp_sk_test_i1b", base_url="http://localhost:59999")
        try:
            with tp.session(name="anchor_probe") as s:
                self.assertTrue(s._anchored)
                self.assertTrue(s.root_span_id)
                ids = manual_span_ids(s)
                # Active OTel structural span is preferred parent.
                self.assertEqual(ids["parent_span_id"], s.root_span_id)
        finally:
            tp.uninstrument()


if __name__ == "__main__":
    unittest.main()
