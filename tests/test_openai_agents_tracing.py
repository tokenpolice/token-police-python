"""Tests for OpenAI Agents SDK tool-span capture.

The Agents SDK records tool executions in its own (non-OTel) tracing as
`FunctionSpanData` spans. TokenPolice registers a TracingProcessor that emits a
`tool` row for those, correlated to the current session's trace. These tests
exercise the emission + registration logic with fakes — no real `agents`
dependency (which isn't installed in this SDK's venv).
"""
import unittest

from token_police import context as ctx
from token_police.context import TPSession
from token_police.state import set_client
from token_police.openai_agents import (
    _handle_span_end,
    register_openai_agents_tracing,
    maybe_register_openai_agents_tracing,
)


class _FakeSpanData:
    def __init__(self, type_, name=None, input=None, output=None):
        self.type = type_
        self.name = name
        self.input = input
        self.output = output


class _FakeSpan:
    def __init__(self, span_data, started_at="2026-06-03T10:00:00Z",
                 ended_at="2026-06-03T10:00:01Z", error=None):
        self.span_data = span_data
        self.span_id = "span_abc"
        self.trace_id = "trace_xyz"
        self.parent_id = "span_parent"
        self.started_at = started_at
        self.ended_at = ended_at
        self.error = error


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


class TestAgentsToolSpan(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient()
        set_client(self.client)
        self.session = TPSession(
            user_id="u1", paid_plan="pro", workflow_name="wf",
            trace_id="a" * 32, root_span_id="b" * 16,
        )
        # Fixture models a scoped session with a real structural root (I1).
        self.session._anchored = True
        self._token = ctx._current_session.set(self.session)
        ctx.set_pending_tool_calls([])  # isolate I7 pending-pop tests

    def tearDown(self):
        ctx._current_session.reset(self._token)

    def test_function_span_emits_tool_row(self):
        span = _FakeSpan(_FakeSpanData(
            "function", name="get_customer_info",
            input='{"customer_id": 42}', output="Premium customer",
        ))
        _handle_span_end(span)

        self.assertEqual(len(self.client.calls), 1)
        kw = self.client.calls[0]
        self.assertEqual(kw["span"]["span_kind"], "tool")
        self.assertEqual(kw["span"]["span_name"], "get_customer_info")
        self.assertEqual(kw["span"]["trace_id"], "a" * 32)
        self.assertEqual(kw["span"]["parent_span_id"], "b" * 16)
        self.assertEqual(kw["tool"]["name"], "get_customer_info")
        self.assertEqual(kw["tool"]["type"], "function")
        # No pending stash → call_id stays empty (never invent).
        self.assertEqual(kw["tool"]["call_id"], "")
        # arg/result are hashed, not stored raw.
        self.assertTrue(kw["tool"]["param_hash"])
        self.assertTrue(kw["tool"]["param_length"] > 0)
        self.assertTrue(kw["tool"]["result_hash"])
        self.assertEqual(kw["call_outcome"]["status"], "success")
        # No raw input/output anywhere in the payload.
        flat = repr(kw)
        self.assertNotIn("customer_id", flat)
        self.assertNotIn("Premium customer", flat)

    def test_function_span_pops_pending_call_id_by_name(self):
        # I7 PR-B: Agents FunctionSpanData has no call_id — pop from pending
        # stash filled by the preceding Responses/chat capture.
        ctx.set_pending_tool_calls([
            {"id": "call_agents_42", "name": "get_customer_info"},
            {"id": "call_other", "name": "other_tool"},
        ])
        span = _FakeSpan(_FakeSpanData(
            "function", name="get_customer_info",
            input="{}", output="ok",
        ))
        _handle_span_end(span)
        self.assertEqual(self.client.calls[0]["tool"]["call_id"], "call_agents_42")
        # Name-exact: unmatched entry remains.
        self.assertEqual(ctx._pop_pending_tool_call_id("other_tool"), "call_other")

    def test_concurrent_same_name_function_spans_get_distinct_call_ids(self):
        # OpenAI Agents runs same-turn tools in concurrent Tasks; both
        # FunctionSpanData rows must carry distinct provider tool_call_ids.
        import asyncio
        ctx.set_pending_tool_calls([
            {"id": "call_A", "name": "get_customer_info"},
            {"id": "call_B", "name": "get_customer_info"},
        ])

        def _emit():
            _handle_span_end(_FakeSpan(_FakeSpanData(
                "function", name="get_customer_info",
                input="{}", output="ok",
            )))

        async def _one():
            await asyncio.sleep(0)
            _emit()

        async def _run():
            await asyncio.gather(_one(), _one())

        # Private loop — do not asyncio.run (unsets main-thread loop).
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_run())
        finally:
            loop.close()
        self.assertEqual(len(self.client.calls), 2)
        ids = {c["tool"]["call_id"] for c in self.client.calls}
        self.assertEqual(ids, {"call_A", "call_B"})

    def test_function_span_name_mismatch_leaves_call_id_empty(self):
        ctx.set_pending_tool_calls([{"id": "call_x", "name": "other"}])
        span = _FakeSpan(_FakeSpanData("function", name="get_customer_info", input="{}", output="ok"))
        _handle_span_end(span)
        self.assertEqual(self.client.calls[0]["tool"]["call_id"], "")
        # Stash not drained on mismatch.
        self.assertEqual(ctx._pop_pending_tool_call_id("other"), "call_x")

    def test_function_span_explicit_call_id_wins_over_pending(self):
        # Future FunctionSpanData fields (or camelCase) win; stash not drained.
        ctx.set_pending_tool_calls([{"id": "stashed", "name": "get_customer_info"}])
        sd = _FakeSpanData("function", name="get_customer_info", input="{}", output="ok")
        sd.call_id = "from_span"
        _handle_span_end(_FakeSpan(sd))
        self.assertEqual(self.client.calls[0]["tool"]["call_id"], "from_span")
        self.assertEqual(ctx._pop_pending_tool_call_id("get_customer_info"), "stashed")

    def test_duration_ms_derived_from_timestamps(self):
        # §4-py-p: duration_ms is computed from the span's ISO-8601
        # started_at/ended_at (1s apart in the fake → 1000ms), not hardcoded 0.
        span = _FakeSpan(
            _FakeSpanData("function", name="slow_tool", input="{}", output="ok"),
            started_at="2026-06-03T10:00:00Z",
            ended_at="2026-06-03T10:00:01.500Z",
        )
        _handle_span_end(span)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["call_outcome"]["duration_ms"], 1500)

    def test_duration_ms_zero_when_timestamps_unparseable(self):
        # Fail-open: a garbage timestamp degrades duration to 0 (row still emitted).
        span = _FakeSpan(
            _FakeSpanData("function", name="t", input="{}", output="ok"),
            started_at="not-a-date", ended_at=None,
        )
        _handle_span_end(span)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["call_outcome"]["duration_ms"], 0)

    def test_non_function_span_is_ignored(self):
        span = _FakeSpan(_FakeSpanData("response"))
        _handle_span_end(span)
        self.assertEqual(self.client.calls, [])

    def test_missing_span_data_is_ignored(self):
        class _Bare:
            span_data = None
        _handle_span_end(_Bare())
        self.assertEqual(self.client.calls, [])

    def test_failed_span_marks_failure(self):
        span = _FakeSpan(
            _FakeSpanData("function", name="escalate", input="{}", output=None),
            error={"message": "boom", "data": None},
        )
        _handle_span_end(span)
        self.assertEqual(len(self.client.calls), 1)
        co = self.client.calls[0]["call_outcome"]
        self.assertEqual(co["status"], "failed")
        # By default the raw error string no longer ships; a SHA-256 hash
        # does instead (the _FakeClient has no error_detail → redacted).
        import hashlib
        self.assertNotIn("error_message", co)
        self.assertEqual(
            co["error_message_hash"],
            hashlib.sha256("boom".encode("utf-8")).hexdigest(),
        )

    def test_emit_never_raises_on_bad_session_or_client(self):
        # Even with a broken client, _handle_span_end must swallow the error.
        class _BoomClient:
            def log_sync(self, **kwargs):
                raise RuntimeError("network down")
        set_client(_BoomClient())
        span = _FakeSpan(_FakeSpanData("function", name="t", input="x", output="y"))
        _handle_span_end(span)  # must not raise

    def test_registration_is_safe_when_agents_not_imported(self):
        # `agents` is not installed in this venv → registration returns False,
        # and the cheap guard is a no-op without raising.
        self.assertFalse(register_openai_agents_tracing())
        maybe_register_openai_agents_tracing()  # must not raise


if __name__ == "__main__":
    unittest.main()
