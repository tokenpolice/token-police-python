"""Regression — TokenPoliceSpanProcessor.on_end must still emit the telemetry
row when session resolution fails.

`session` is only bound inside a best-effort `get_current_session()` try; if the
`.context` import fails or `get_current_session()` raises, `session` was left
unbound. A later `getattr(session, '_defer_telemetry', ...)` on the immediate
log path then raised NameError, which the outer on_end handler swallowed as a
warning — silently DROPPING the whole span row instead of logging it.

Fix: `session = None` default before the try, so a failed resolution simply
omits the session-derived enrichment and the row is still emitted.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import state
from token_police.telemetry import TokenPoliceSpanProcessor


NS_START = 1_000_000_000
NS_END = 2_000_000_000  # duration_ms == 1000


def _fake_span(attrs):
    return NS(
        attributes=attrs,
        name="anthropic.chat",
        instrumentation_scope=NS(name="opentelemetry.instrumentation.anthropic"),
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        start_time=NS_START,
        end_time=NS_END,
        status=None,
    )


def _usage_attrs():
    return {
        "gen_ai.system": "anthropic",
        "gen_ai.request.model": "claude-3-5-sonnet-20241022",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 50,
    }


class TestOnEndSessionUnbound(unittest.TestCase):
    def setUp(self):
        state._observations_set([])

    def tearDown(self):
        state._observations_set([])

    def _run(self, raiser):
        """Drive on_end with get_current_session patched to `raiser`; capture
        log_sync kwargs, call count, and whether the swallow-warning fired."""
        captured = {}
        calls = {"n": 0}

        class _FakeClient:
            def log_sync(self, **kwargs):
                calls["n"] += 1
                captured.update(kwargs)

        proc = TokenPoliceSpanProcessor()
        span = _fake_span(_usage_attrs())
        # Patch at the import SOURCE (`token_police.context`) — on_end does
        # `from .context import get_current_session` inside the try.
        with mock.patch("token_police.state.get_client", return_value=_FakeClient()), \
             mock.patch("token_police.context.get_current_session", side_effect=raiser), \
             mock.patch("token_police.telemetry.logger") as log:
            proc.on_end(span)
            warned = log.warning.called
        return captured, calls["n"], warned

    def test_row_emitted_when_get_current_session_raises(self):
        payload, n, warned = self._run(RuntimeError("no session context"))
        # The row IS emitted (not dropped) despite session resolution failing.
        self.assertEqual(n, 1)
        # It carries the span-derived essentials, just without session enrichment.
        self.assertEqual(payload["model"], "claude-3-5-sonnet-20241022")
        self.assertEqual(int(payload["input_tokens"]), 100)
        self.assertEqual(int(payload["output_tokens"]), 50)
        # No NameError bubbled into the outer swallow-warning handler.
        self.assertFalse(warned)

    def test_row_emitted_when_context_lookup_returns_none(self):
        # A benign None (no active session) must also emit — the immediate path,
        # not the deferred branch.
        payload, n, warned = self._run(lambda: None)
        self.assertEqual(n, 1)
        self.assertFalse(warned)


if __name__ == "__main__":
    unittest.main()
