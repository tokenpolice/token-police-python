"""T3 — structural root (workflow/agent/chain) reports status='failed' when the
decorated body raises uncaught, instead of the server's default 'success'.

Two halves:
  A. Emitter unit (`_log_agent_span` via `on_end`): given an anchor span whose
     OTel status is ERROR, the zero-usage structural row carries
     call_outcome={'status':'failed', ...} with the scrub (no raw text);
     an UNSET/None status → 'success'. Mirrors the tool-span idiom.
  B. End-to-end (real tracer): raising inside `tp.session()` / `tp.chain()`
     auto-stamps the anchor span ERROR (start_as_current_span default
     set_status_on_exception=True), so the emitted row is 'failed', while the
     customer's ORIGINAL exception propagates unchanged. Nested cases: an inner
     failure caught by the outer body → inner 'failed' / outer 'success'; an
     inner failure that propagates through the outer → both 'failed'.

Golden rule: the SDK only READS the span status (inside the already-swallowed
on_end) — no control-flow change; the exception the customer raised is the same
object that leaves the `with` block.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from opentelemetry.trace import StatusCode

from token_police import state
from token_police.telemetry import TokenPoliceSpanProcessor


NS_START = 1_000_000_000
NS_END = 2_000_000_000  # +1e9 ns → duration_ms == 1000


def _fake_anchor_span(kind, *, error=False, description="",
                      start_time=NS_START, end_time=NS_END):
    status = None
    if error:
        status = NS(status_code=StatusCode.ERROR, description=description)
    return NS(
        attributes={
            "tp.kind": kind,
            "tp.workflow_name": "wf" if kind == "agent" else "pipeline",
            "tp.user_id": "u-t3",
            "tp.paid_plan": "pro",
        },
        name=kind,
        instrumentation_scope=NS(name="tokenpolice"),
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        start_time=start_time,
        end_time=end_time,
        status=status,
    )


def _run_on_end(span):
    captured = {}
    calls = {"n": 0}

    class _FakeClient:
        def log_sync(self, **kwargs):
            calls["n"] += 1
            captured.clear()
            captured.update(kwargs)

    proc = TokenPoliceSpanProcessor()
    with mock.patch("token_police.state.get_client", return_value=_FakeClient()):
        proc.on_end(span)
    return captured, calls["n"]


class TestAgentSpanCallOutcome(unittest.TestCase):
    """A. emitter maps OTel status → call_outcome on the structural row."""

    def test_chain_error_marks_failed(self):
        payload, n = _run_on_end(_fake_anchor_span("chain", error=True,
                                                   description="voyage 429 secret"))
        self.assertEqual(n, 1)
        self.assertEqual(payload["span"]["span_kind"], "chain")
        co = payload["call_outcome"]
        self.assertEqual(co["status"], "failed")
        self.assertEqual(co["duration_ms"], 1000)
        # Default 'redacted' ships a SHA-256 hash, never the raw text.
        self.assertIn("error_message_hash", co)
        self.assertEqual(len(co["error_message_hash"]), 64)
        for v in co.values():
            self.assertNotIn("voyage 429 secret", str(v))

    def test_agent_unset_marks_success(self):
        payload, n = _run_on_end(_fake_anchor_span("agent"))  # status=None
        self.assertEqual(n, 1)
        self.assertEqual(payload["span"]["span_kind"], "agent")
        self.assertEqual(
            payload["call_outcome"], {"status": "success", "duration_ms": 1000}
        )

    def test_failed_shipped_even_when_duration_zero(self):
        payload, _ = _run_on_end(
            _fake_anchor_span("chain", error=True, description="boom",
                              start_time=NS_START, end_time=NS_START)
        )
        co = payload["call_outcome"]
        self.assertEqual(co["status"], "failed")
        self.assertEqual(co["duration_ms"], 0)


class TestStructuralEndToEnd(unittest.TestCase):
    """B. real tracer: raising inside the wrapper flips the emitted row failed,
    the original exception is untouched, and nested status is independent.

    Note on isolation: production calls tp.init() once, but re-calling it per
    test piggybacks an extra span processor onto the (do-once) global tracer
    provider, so one span may emit several IDENTICAL rows. We therefore key
    captured rows by span_name and assert the STATUS SET per name — duplicates
    collapse and any inconsistency would still be caught.
    """

    def _statuses_by_name(self, fn):
        """init → drive fn() through the real tracer → {span_name: {statuses}}."""
        import token_police as tp
        from token_police.state import set_client

        rows = []

        class _CapturingClient:
            def log_sync(self, **kwargs):
                rows.append(kwargs)

        tp.init(api_key="tp_sk_test_t3", base_url="http://localhost:59999")
        set_client(_CapturingClient())
        try:
            fn(tp)
        finally:
            tp.uninstrument()

        out = {}
        for r in rows:
            span = r.get("span", {})
            if span.get("span_kind") not in ("agent", "chain"):
                continue
            out.setdefault(span.get("span_name"), set()).add(
                r.get("call_outcome", {}).get("status")
            )
        return out

    def test_session_body_raises_marks_root_failed_and_reraises_same_error(self):
        sentinel = RuntimeError("workflow blew up")
        caught = {}

        def body(tp):
            try:
                with tp.session(name="failing_wf"):
                    raise sentinel
            except RuntimeError as e:
                caught["err"] = e

        statuses = self._statuses_by_name(body)
        # Golden rule: the exact object the customer raised propagated unchanged.
        self.assertIs(caught["err"], sentinel)
        self.assertEqual(statuses.get("failing_wf"), {"failed"})

    def test_session_body_returns_marks_root_success(self):
        def body(tp):
            with tp.session(name="ok_wf"):
                pass

        statuses = self._statuses_by_name(body)
        self.assertEqual(statuses.get("ok_wf"), {"success"})

    def test_nested_inner_caught_inner_failed_outer_success(self):
        def body(tp):
            with tp.session(name="outer_wf"):
                try:
                    with tp.chain(name="inner_chain"):
                        raise ValueError("inner boom")
                except ValueError:
                    pass  # outer body recovers → outer must stay success

        statuses = self._statuses_by_name(body)
        self.assertEqual(statuses.get("inner_chain"), {"failed"})
        self.assertEqual(statuses.get("outer_wf"), {"success"})

    def test_nested_inner_propagates_both_failed(self):
        def body(tp):
            try:
                with tp.session(name="outer_wf2"):
                    with tp.chain(name="inner_chain2"):
                        raise KeyError("inner propagates")
            except KeyError:
                pass

        statuses = self._statuses_by_name(body)
        self.assertEqual(statuses.get("inner_chain2"), {"failed"})
        self.assertEqual(statuses.get("outer_wf2"), {"failed"})


if __name__ == "__main__":
    unittest.main()
