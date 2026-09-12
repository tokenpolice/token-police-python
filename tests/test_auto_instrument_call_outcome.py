"""Python auto-instrument LLM path (TokenPoliceSpanProcessor.on_end)
must forward `call_outcome`, `observations`, and `latency` on the IMMEDIATE
(non-deferred) `client.log_sync(**payload)` branch — parity with Node
(`telemetry.ts:1210-1219`) and with what the deferred `_flush_deferred_spans`
path already ships.

Gate rules (AGREED contract ):
  - call_outcome: FAILED span → ALWAYS attach (mirror tool-span idiom
    telemetry.py:1028-1039, never gated on duration) with the scrub so no
    raw error text leaks; SUCCESS span → attach only when duration_ms > 0
    (mirror Node telemetry.ts:1215-1217).
  - observations: drain_observations() attached only when non-empty, drained
    exactly once (immediate/deferred branches are mutually exclusive per call).
  - latency: comp_data.get("latency") when present — NameError-safe via the
    comp_data={} default.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from opentelemetry.trace import StatusCode

from token_police import state
from token_police.telemetry import TokenPoliceSpanProcessor


# 1 ns == 1e-6 ms; 1e9 ns == 1000 ms.
NS_START = 1_000_000_000
NS_END = 2_000_000_000  # 1e9 ns after start -> duration_ms == 1000


def _fake_span(attrs, *, start_time=NS_START, end_time=NS_END,
               error=False, description=""):
    status = None
    if error:
        status = NS(status_code=StatusCode.ERROR, description=description)
    return NS(
        attributes=attrs,
        name="anthropic.chat",
        instrumentation_scope=NS(name="opentelemetry.instrumentation.anthropic"),
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        start_time=start_time,
        end_time=end_time,
        status=status,
    )


def _usage_attrs(**extra):
    a = {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-3-5-sonnet-20241022",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 50,
    }
    a.update(extra)
    return a


class _FakeSession:
    """Minimal session: only carries the attributes on_end reads."""


def _run_on_end(span, session=None):
    """Run on_end over `span` with a controlled session, capturing the kwargs
    passed to client.log_sync (and the number of calls)."""
    captured = {}
    calls = {"n": 0}

    class _FakeClient:
        def log_sync(self, **kwargs):
            calls["n"] += 1
            captured.clear()
            captured.update(kwargs)

    if session is None:
        session = _FakeSession()  # no _defer_telemetry, no _pending_compositions

    proc = TokenPoliceSpanProcessor()
    with mock.patch("token_police.state.get_client", return_value=_FakeClient()), \
         mock.patch("token_police.context.get_current_session", return_value=session):
        proc.on_end(span)
    return captured, calls["n"]


class TestCallOutcome(unittest.TestCase):
    def setUp(self):
        state._observations_set([])  # isolate the global queue per test

    def tearDown(self):
        state._observations_set([])

    # 1
    def test_success_call_outcome(self):
        payload, n = _run_on_end(_fake_span(_usage_attrs()))
        self.assertEqual(n, 1)
        self.assertEqual(
            payload["call_outcome"], {"status": "success", "duration_ms": 1000}
        )

    # 2
    def test_failed_call_outcome_scrubbed(self):
        payload, _ = _run_on_end(
            _fake_span(_usage_attrs(), error=True, description="boom secret")
        )
        co = payload["call_outcome"]
        self.assertEqual(co["status"], "failed")
        self.assertEqual(co["duration_ms"], 1000)
        # Default 'redacted' ships a SHA-256 hash, never raw text.
        self.assertIn("error_message_hash", co)
        self.assertEqual(len(co["error_message_hash"]), 64)
        for v in co.values():
            self.assertNotIn("boom secret", str(v))

    # 2b
    def test_failed_call_outcome_shipped_when_duration_zero(self):
        payload, _ = _run_on_end(
            _fake_span(
                _usage_attrs(), start_time=NS_START, end_time=NS_START,
                error=True, description="boom secret",
            )
        )
        co = payload["call_outcome"]  # STILL present on a 0-duration failure
        self.assertEqual(co["status"], "failed")
        self.assertEqual(co["duration_ms"], 0)
        self.assertIn("error_message_hash", co)
        for v in co.values():
            self.assertNotIn("boom secret", str(v))

    # 5b
    def test_success_call_outcome_omitted_when_duration_zero(self):
        payload, _ = _run_on_end(
            _fake_span(_usage_attrs(), start_time=NS_START, end_time=NS_START)
        )
        self.assertNotIn("call_outcome", payload)

    # 3
    def test_observations_shipped_on_success(self):
        obs1 = {"kind": "would_block", "rule": "r1"}
        obs2 = {"kind": "would_reroute", "rule": "r2"}
        state.push_observation(obs1)
        state.push_observation(obs2)
        payload, _ = _run_on_end(_fake_span(_usage_attrs()))
        self.assertEqual(payload["observations"], [obs1, obs2])

    # 4
    def test_observations_drained_once(self):
        state.push_observation({"kind": "would_block"})
        payload, n = _run_on_end(_fake_span(_usage_attrs()))
        self.assertEqual(n, 1)
        self.assertIn("observations", payload)
        # Queue emptied by the on_end drain — a second drain yields nothing.
        self.assertEqual(state.drain_observations(), [])

    # 5
    def test_observations_empty_omitted(self):
        payload, _ = _run_on_end(_fake_span(_usage_attrs()))
        self.assertNotIn("observations", payload)

    # 6
    def test_latency_forwarded_from_comp_data(self):
        lat = {"total_ms": 1234, "ttft_ms": 50}
        session = _FakeSession()
        session._pending_compositions = {"t1:0": {"latency": lat}}
        span = _fake_span(_usage_attrs(**{"tp.trace_id": "t1", "tp.span_order": 0}))
        payload, _ = _run_on_end(span, session=session)
        self.assertEqual(payload["latency"], lat)

    # 7
    def test_latency_safe_when_no_composition(self):
        # No _pending_compositions entry for this key: comp_data must default to
        # {} (no NameError), log_sync still fires exactly once, no latency key,
        # and the on_end error-swallow warning must NOT be triggered.
        session = _FakeSession()  # no _pending_compositions at all
        span = _fake_span(_usage_attrs(**{"tp.trace_id": "nope", "tp.span_order": 0}))
        with mock.patch("token_police.telemetry.logger") as log:
            payload, n = _run_on_end(span, session=session)
            log.warning.assert_not_called()
        self.assertEqual(n, 1)
        self.assertNotIn("latency", payload)

    # 8
    def test_deferred_path_untouched(self):
        session = _FakeSession()
        session._defer_telemetry = True
        state.push_observation({"kind": "would_block"})
        span = _fake_span(_usage_attrs())

        captured = {}
        calls = {"n": 0}

        class _FakeClient:
            def log_sync(self, **kwargs):
                calls["n"] += 1
                captured.update(kwargs)

        proc = TokenPoliceSpanProcessor()
        with mock.patch("token_police.state.get_client", return_value=_FakeClient()), \
             mock.patch("token_police.context.get_current_session", return_value=session):
            proc.on_end(span)

        # Deferred: immediate log_sync NOT called, observations NOT drained.
        self.assertEqual(calls["n"], 0)
        self.assertEqual(len(session._deferred_spans), 1)
        self.assertEqual(state.drain_observations(), [{"kind": "would_block"}])
        # And the deferred payload itself carries none of the immediate extras.
        deferred = session._deferred_spans[0]
        self.assertNotIn("call_outcome", deferred)
        self.assertNotIn("observations", deferred)
        self.assertNotIn("latency", deferred)


if __name__ == "__main__":
    unittest.main()
