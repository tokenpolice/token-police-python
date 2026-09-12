"""To_duration_ms: positive sub-ms deltas report 1, true-zero stays 0.

Multi-ms keeps Python's historical int floor. Wired through tool/structural
span emitters so harness tools that finish in <1ms no longer land duration_ms=0.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police._classify import to_duration_ms
from token_police.telemetry import TokenPoliceSpanProcessor


class TestToDurationMs(unittest.TestCase):
    def test_positive_sub_ms_ceil_to_one(self):
        self.assertEqual(to_duration_ms(0.3), 1)
        self.assertEqual(to_duration_ms(0.001), 1)
        self.assertEqual(to_duration_ms(0.999), 1)

    def test_true_zero_negative_non_finite(self):
        self.assertEqual(to_duration_ms(0), 0)
        self.assertEqual(to_duration_ms(-1), 0)
        self.assertEqual(to_duration_ms(float("nan")), 0)
        self.assertEqual(to_duration_ms(float("inf")), 0)
        self.assertEqual(to_duration_ms("nope"), 0)  # type: ignore[arg-type]
        self.assertEqual(to_duration_ms(True), 0)  # bool is int subclass

    def test_multi_ms_keeps_floor(self):
        self.assertEqual(to_duration_ms(5), 5)
        self.assertEqual(to_duration_ms(5.2), 5)
        self.assertEqual(to_duration_ms(1.9), 1)
        self.assertEqual(to_duration_ms(1500), 1500)


NS_START = 1_000_000_000  # 1s in ns


def _fake_tool_span(start_ns, end_ns):
    return NS(
        attributes={
            "traceloop.span.kind": "tool",
            "gen_ai.tool.name": "lookup_tool",
            "gen_ai.tool.type": "function",
            "tp.user_id": "u-b09",
            "tp.paid_plan": "free",
            "tp.workflow_name": "wf",
            "tp.kind": "tool",
        },
        name="lookup_tool.tool",
        instrumentation_scope=NS(name="tokenpolice"),
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        start_time=start_ns,
        end_time=end_ns,
        status=None,
    )


def _run_tool_log(span):
    captured = {}

    class _FakeClient:
        def log_sync(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    proc = TokenPoliceSpanProcessor()
    with mock.patch("token_police.state.get_client", return_value=_FakeClient()):
        proc._log_tool_span(span, dict(span.attributes), span.name)
    return captured


class TestToolSpanDurationMs(unittest.TestCase):
    def test_sub_ms_otel_delta_reports_one(self):
        # 300_000 ns = 0.3 ms → must not floor to 0
        payload = _run_tool_log(_fake_tool_span(NS_START, NS_START + 300_000))
        self.assertEqual(payload.get("span", {}).get("span_kind"), "tool")
        self.assertEqual(payload["call_outcome"]["duration_ms"], 1)

    def test_zero_delta_reports_zero(self):
        payload = _run_tool_log(_fake_tool_span(NS_START, NS_START))
        self.assertEqual(payload["call_outcome"]["duration_ms"], 0)

    def test_multi_ms_unchanged(self):
        # 5_000_000 ns = 5 ms
        payload = _run_tool_log(_fake_tool_span(NS_START, NS_START + 5_000_000))
        self.assertEqual(payload["call_outcome"]["duration_ms"], 5)


class TestStructuralSubMs(unittest.TestCase):
    def test_agent_sub_ms_reports_one(self):
        captured = {}

        class _FakeClient:
            def log_sync(self, **kwargs):
                captured.clear()
                captured.update(kwargs)

        span = NS(
            attributes={
                "tp.kind": "agent",
                "tp.workflow_name": "wf",
                "tp.user_id": "u-b09",
                "tp.paid_plan": "free",
            },
            name="agent",
            instrumentation_scope=NS(name="tokenpolice"),
            context=NS(trace_id=0x1234, span_id=0xABCD),
            parent=None,
            start_time=NS_START,
            end_time=NS_START + 300_000,
            status=None,
        )
        proc = TokenPoliceSpanProcessor()
        with mock.patch("token_police.state.get_client", return_value=_FakeClient()):
            proc._log_agent_span(span, dict(span.attributes), "agent")
        self.assertEqual(captured["call_outcome"]["duration_ms"], 1)


class TestOpenAIAgentsSubMs(unittest.TestCase):
    def test_sub_ms_iso_timestamps(self):
        from token_police.openai_agents import _tool_duration_ms

        span = NS(
            started_at="2026-06-03T10:00:00.000Z",
            ended_at="2026-06-03T10:00:00.0003Z",  # +0.3 ms
            span_data=NS(type="function", name="t", input="{}", output="ok"),
            error=None,
        )
        self.assertEqual(_tool_duration_ms(span), 1)

    def test_zero_when_same_timestamp(self):
        from token_police.openai_agents import _tool_duration_ms

        span = NS(
            started_at="2026-06-03T10:00:00Z",
            ended_at="2026-06-03T10:00:00Z",
            span_data=NS(type="function", name="t"),
            error=None,
        )
        self.assertEqual(_tool_duration_ms(span), 0)


if __name__ == "__main__":
    unittest.main()
