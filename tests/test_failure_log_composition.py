"""Failed-row prompt_composition (enforcer._emit_call_failure_log).

Manual-path wrappers reserve their span order via next_span_order() BEFORE
stashing the prompt composition, so when the LLM call raises, _span_counter has
already advanced one past the stash key — the failure log previously looked up
`{trace_id}:{_span_counter}`, missed, and the failed row shipped without its
prompt composition. The fix adds a narrowly-gated one-order-back fallback that
only engages when the primary key held no prompt.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession

_PROMPT = [{"role": "user", "type": "text", "length": 5, "hash": "abcd1234"}]
_OUTCOME = {"status": "failed", "duration_ms": 7, "error_kind": "client_error",
            "http_status": 400, "error_message": "bad request"}


def _emit(session):
    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    enforcer._emit_call_failure_log(_FakeTP(), session)
    return captured


class TestFailureLogPromptComposition(unittest.TestCase):
    def test_manual_path_stash_found_via_fallback(self):
        # Manual wrapper: order 0 consumed, prompt stashed at:0, counter now 1.
        session = TPSession()
        order = session.next_span_order()
        session._pending_compositions = {
            f"{session.trace_id}:{order}": {"prompt": list(_PROMPT)},
        }
        session._call_outcome = dict(_OUTCOME)
        captured = _emit(session)
        self.assertEqual(captured.get("prompt_composition"), _PROMPT)
        # Stash consumed — must not leak into the next call.
        self.assertEqual(session._pending_compositions, {})

    def test_primary_key_still_wins(self):
        # Mode-A path: prompt stashed at the CURRENT counter (on_start not yet
        # run). The original lookup must keep working, no fallback involved.
        session = TPSession()
        session._pending_compositions = {
            f"{session.trace_id}:{session._span_counter}": {"prompt": list(_PROMPT)},
        }
        session._call_outcome = dict(_OUTCOME)
        captured = _emit(session)
        self.assertEqual(captured.get("prompt_composition"), _PROMPT)

    def test_no_stash_stays_empty(self):
        session = TPSession()
        session.next_span_order()
        session._call_outcome = dict(_OUTCOME)
        captured = _emit(session)
        self.assertIsNone(captured.get("prompt_composition"))

    def test_outcome_and_model_context_unchanged(self):
        session = TPSession()
        order = session.next_span_order()
        session._pending_compositions = {
            f"{session.trace_id}:{order}": {"prompt": list(_PROMPT)},
        }
        session._call_outcome = dict(_OUTCOME)
        session._attempted_model = "gpt-4.1-mini"
        session._attempted_provider = "openai"
        captured = _emit(session)
        self.assertEqual(captured["call_outcome"]["http_status"], 400)
        self.assertEqual(captured["model"], "gpt-4.1-mini")
        self.assertEqual(captured["provider"], "openai")


def _run_on_end(session, span):
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
        TokenPoliceSpanProcessor().on_end(span)
    return captured


def _instrumented_span(session, order, status=None):
    return NS(
        attributes={
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4.1-mini",
            "tp.trace_id": session.trace_id,
            "tp.span_order": order,
        },
        name="openai.chat",
        instrumentation_scope=NS(name="opentelemetry.instrumentation.openai"),
        context=NS(trace_id=0x1, span_id=0x2),
        parent=None,
        status=status,
        start_time=0,
        end_time=1,
    )


class TestErroredSpanKeepsStashForFailureLog(unittest.TestCase):
    """Instrumented (Traceloop) path: the ERRORED span's on_end fires DURING
    the failing call and previously popped the `_pending_compositions` entry,
    so `_emit_call_failure_log` (which runs right after) found nothing and the
    failed row shipped without prompt_composition. on_end must PEEK (not pop)
    when the span ended with ERROR status; the failure logger pops it.
    Successful spans must keep popping (no leak regression).

    Tradeoff (noted, accepted): if an errored span is ever NOT followed by a
    failure log, the peeked entry lingers on the session dict until the
    session object is released — bounded, and safer than clearing the whole
    trace's stash (which could nuke a concurrent sibling call's entry)."""

    def _error_status(self):
        from opentelemetry.trace import StatusCode
        return NS(status_code=StatusCode.ERROR, description="401 unauthorized")

    def test_errored_span_leaves_stash_for_failure_log(self):
        session = TPSession()
        # Mode-A wrapper stashes the prompt BEFORE the call (counter == 0)...
        session._pending_compositions = {
            f"{session.trace_id}:0": {"prompt": list(_PROMPT)},
        }
        # ...then the instrumentor span consumes order 0 during the failing call.
        order = session.next_span_order()
        _run_on_end(session, _instrumented_span(session, order,
                                                status=self._error_status()))
        # The errored span must NOT have consumed the stash entry.
        self.assertIn(f"{session.trace_id}:0", session._pending_compositions)
        # The failure log (runs right after in the wrapper's except) finds it
        # via the prev-order fallback and pops it.
        session._call_outcome = dict(_OUTCOME)
        captured = _emit(session)
        self.assertEqual(captured.get("prompt_composition"), _PROMPT)
        self.assertEqual(session._pending_compositions, {})

    def test_errored_span_payload_still_carries_peeked_composition(self):
        session = TPSession()
        session._pending_compositions = {
            f"{session.trace_id}:0": {"prompt": list(_PROMPT)},
        }
        order = session.next_span_order()
        captured = _run_on_end(session, _instrumented_span(
            session, order, status=self._error_status()))
        # The (zero-usage) errored span row keeps its composition too.
        self.assertEqual(captured.get("prompt_composition"), _PROMPT)

    def test_successful_span_still_pops_stash(self):
        # Non-errored spans keep the destructive pop — no entry leak.
        session = TPSession()
        session._pending_compositions = {
            f"{session.trace_id}:0": {"prompt": list(_PROMPT)},
        }
        order = session.next_span_order()
        captured = _run_on_end(session, _instrumented_span(session, order,
                                                           status=None))
        self.assertEqual(captured.get("prompt_composition"), _PROMPT)
        self.assertEqual(session._pending_compositions, {})


if __name__ == "__main__":
    unittest.main()
