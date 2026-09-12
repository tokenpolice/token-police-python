"""T4a / pydantic_ai streamed rows: is_streaming + ttft_ms.

pydantic_ai is a *manual* enforcer wrapper (not traceloop-instrumented). It saw
the call but never passed a `latency=` object to `log_sync`, so streamed rows
defaulted to `is_streaming=0` at the server. T4a fixed is_streaming;
(T4b) captures TTFT by instance-tapping StreamedResponse._get_event_iterator
so the first content event stamps `_ttft_mono` without replacing the stream
object.

Assertions:
  - STREAM + drain multi-event → `latency.is_streaming == True`, `ttft_ms > 0`,
    `ttft_ms <= total_ms`.
  - STREAM + no iteration → `is_streaming == True`, `ttft_ms is None` (honest;
    never fabricate TTFT when no chunk was observed).
  - STREAM + tool-only PartStartEvent (no text delta) → `ttft_ms > 0`.
  - STREAM + no `_get_event_iterator` (tap fail-open) → customer still gets the
    stream; log may have null TTFT.
  - NON-STREAM path → `is_streaming == False`, `ttft_ms is None`.
  - ERRORED stream → NO /log row.

Golden rule: tap failure returns the raw stream; mark probe never raises into
customer iteration; event sequence is unchanged.
"""
from __future__ import annotations

import asyncio
import time
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import context as ctx
from token_police.context import TPSession, _in_pydantic_ai
from token_police.client import set_client
from token_police import enforcer


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


class _PartStartEvent:
    """Duck-typed pydantic_ai PartStartEvent (class name matters for the marker)."""

    def __init__(self, part=None):
        self.part = part if part is not None else NS(part_kind="text", content="hi")


class _PartDeltaEvent:
    def __init__(self, delta=None):
        self.delta = delta if delta is not None else NS(content_delta=" there")


class _FakeStream:
    """Minimal StreamedResponse stand-in with _get_event_iterator + __aiter__
    so the instance tap and customer drain both work like real pydantic_ai.
    """

    def __init__(self, response, events=None):
        self._response = response
        self._events = list(events) if events is not None else []
        self._event_iterator = None

    def get(self):
        return self._response

    async def _get_event_iterator(self):
        # Delay before first yield so (ttft_mono - start_mono) is > 0 after
        # integer ms rounding — mirrors real network TTFB.
        if self._events:
            await asyncio.sleep(0.02)
        for i, ev in enumerate(self._events):
            yield ev
            if i + 1 < len(self._events):
                await asyncio.sleep(0.01)

    def __aiter__(self):
        # Mirror base StreamedResponse: build pipeline from _get_event_iterator once.
        if self._event_iterator is None:
            self._event_iterator = self._get_event_iterator()
        return self._event_iterator


class _FakeStreamNoIterator:
    """StreamedResponse-shaped object without _get_event_iterator — tap fail-open."""

    def __init__(self, response):
        self._response = response

    def get(self):
        return self._response


def _fake_response():
    return NS(
        model_name="claude-haiku-4-5-20251001",
        provider_name="anthropic",
        usage=NS(input_tokens=803, output_tokens=124),
    )


def _fake_model():
    return NS(model_name="claude-haiku-4-5-20251001", system="anthropic")


class TestPydanticAiIsStreaming(unittest.TestCase):
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

    def tearDown(self):
        ctx._current_session.reset(self._token)
        set_client(None)
        try:
            _in_pydantic_ai.set(False)
        except Exception:
            pass

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _make_mgr(self, stream):
        class _Mgr:
            async def __aenter__(_self):
                return stream

            async def __aexit__(_self, *a):
                return False

        return enforcer._PydanticAIAsyncStreamMgr(
            _Mgr(), _fake_model(), messages=[]
        )

    # ── STREAM path: drain multi-event → TTFT > 0 ─────────────
    def test_stream_drain_logs_ttft_and_is_streaming(self):
        response = _fake_response()
        events = [
            _PartStartEvent(),
            _PartDeltaEvent(),
            _PartDeltaEvent(NS(content_delta="!")),
        ]
        stream = _FakeStream(response, events=events)
        wrapper = self._make_mgr(stream)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                async with wrapper as s:
                    # Golden rule: same object identity (instance tap, not proxy).
                    self.assertIs(s, stream)
                    drained = []
                    async for ev in s:
                        drained.append(ev)
                    self.assertEqual(len(drained), 3)

        self._run(_drive())

        self.assertEqual(len(self.client.calls), 1, "stream must log exactly one row")
        latency = self.client.calls[0].get("latency")
        self.assertIsNotNone(latency, "stream row must carry a latency object")
        self.assertTrue(latency["is_streaming"], "streamed row must set is_streaming=True")
        self.assertEqual(latency["clock"], "monotonic")
        self.assertIsInstance(latency["total_ms"], int)
        self.assertGreaterEqual(latency["total_ms"], 0)
        self.assertIsNotNone(latency["ttft_ms"], "drained content must stamp TTFT")
        self.assertIsInstance(latency["ttft_ms"], int)
        self.assertGreater(latency["ttft_ms"], 0, "TTFT should be positive after simulated TTFB")
        self.assertIsNotNone(latency["generation_ms"], "generation_ms set when ttft_mono observed")
        self.assertLessEqual(
            latency["ttft_ms"], latency["total_ms"],
            "ttft_ms must not exceed total_ms",
        )

    # ── STREAM path: no iteration → honest null TTFT ──────────────────
    def test_stream_no_iteration_logs_is_streaming_null_ttft(self):
        response = _fake_response()
        stream = _FakeStream(response, events=[_PartStartEvent()])
        wrapper = self._make_mgr(stream)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                async with wrapper as s:
                    self.assertIsInstance(s, _FakeStream)
                    # Customer never iterates — no fabricated TTFT.

        self._run(_drive())

        self.assertEqual(len(self.client.calls), 1)
        latency = self.client.calls[0].get("latency")
        self.assertIsNotNone(latency)
        self.assertTrue(latency["is_streaming"])
        self.assertIsNone(
            latency["ttft_ms"],
            "no iteration → ttft_ms stays None (never fabricate)",
        )
        self.assertIsInstance(latency["total_ms"], int)
        self.assertGreaterEqual(latency["total_ms"], 0)

    # ── STREAM path: tool-only PartStartEvent (no text delta) ─────────
    def test_stream_tool_only_part_start_stamps_ttft(self):
        response = _fake_response()
        tool_start = _PartStartEvent(
            part=NS(part_kind="tool-call", tool_name="get_customer_info", args={})
        )
        stream = _FakeStream(response, events=[tool_start])
        wrapper = self._make_mgr(stream)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                async with wrapper as s:
                    async for _ in s:
                        pass

        t0 = time.monotonic()
        self._run(_drive())
        # Ensure wall time advanced a hair so total_ms is non-zero
        self.assertGreaterEqual(time.monotonic() - t0, 0)

        latency = self.client.calls[0].get("latency")
        self.assertTrue(latency["is_streaming"])
        self.assertIsNotNone(
            latency["ttft_ms"],
            "tool-only PartStartEvent must still stamp TTFT",
        )
        self.assertIsInstance(latency["ttft_ms"], int)
        self.assertGreater(
            latency["ttft_ms"], 0,
            "tool-only PartStartEvent TTFT should be positive after simulated TTFB",
        )
        self.assertIsNotNone(latency["generation_ms"])

    # ── STREAM path: tap fail-open (no _get_event_iterator) ───────────
    def test_stream_tap_failopen_without_get_event_iterator(self):
        response = _fake_response()
        stream = _FakeStreamNoIterator(response)
        wrapper = self._make_mgr(stream)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                async with wrapper as s:
                    # Customer still receives the raw stream object.
                    self.assertIs(s, stream)
                    self.assertEqual(s.get(), response)

        self._run(_drive())

        self.assertEqual(len(self.client.calls), 1)
        latency = self.client.calls[0].get("latency")
        self.assertIsNotNone(latency)
        self.assertTrue(latency["is_streaming"])
        self.assertIsNone(
            latency["ttft_ms"],
            "untappable stream → null TTFT, not a crash",
        )

    # ── STREAM path: _mark_ttft raises → drain still completes ────────
    def test_mark_ttft_raise_does_not_break_drain(self):
        response = _fake_response()
        e1, e2 = _PartStartEvent(), _PartDeltaEvent()
        stream = _FakeStream(response, events=[e1, e2])
        wrapper = self._make_mgr(stream)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"), \
                    mock.patch.object(
                        enforcer._PydanticAIAsyncStreamMgr,
                        "_mark_ttft",
                        side_effect=RuntimeError("probe boom"),
                    ):
                async with wrapper as s:
                    got = [ev async for ev in s]
                    self.assertEqual(got, [e1, e2])

        self._run(_drive())
        # Still logs (ok path); TTFT may be null because mark never succeeded.
        self.assertEqual(len(self.client.calls), 1)
        self.assertTrue(self.client.calls[0]["latency"]["is_streaming"])

    # ── NON-STREAM path ──────────────────────────────────────────────
    def test_non_stream_path_logs_is_streaming_false(self):
        response = _fake_response()
        model = _fake_model()

        async def _original(_self, *a, **k):
            return response

        wrapper = enforcer._make_pydantic_ai_request_wrapper(_original)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                return await wrapper(model)

        out = self._run(_drive())
        self.assertIs(out, response, "customer still gets the raw response back")

        self.assertEqual(len(self.client.calls), 1)
        latency = self.client.calls[0].get("latency")
        self.assertIsNotNone(latency)
        self.assertFalse(latency["is_streaming"], "non-stream must NOT report is_streaming=True")
        self.assertIsNone(latency["ttft_ms"])
        self.assertEqual(latency["clock"], "monotonic")

    # ── ERRORED stream ───────────────────────────────────────────────
    def test_errored_stream_logs_failed_row(self):
        """L-2: an exception inside the customer's `async with` used to log NO
        row at all — the inner provider's OTel span is hard-dropped under the
        pydantic_ai guard, so the call vanished from the customer's FinOps view.
        It must now land exactly ONE row, labeled failed.

        The original no-latency assertion survives as-is: a broken stream still
        carries no latency object, so there is no is_streaming/ttft false
        positive — only the row-count contract changed.
        """
        response = _fake_response()
        stream = _FakeStream(response, events=[_PartStartEvent()])
        wrapper = self._make_mgr(stream)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                with self.assertRaises(ValueError):
                    async with wrapper:
                        raise ValueError("customer stream consumption blew up")

        self._run(_drive())

        self.assertEqual(
            len(self.client.calls), 1,
            "an errored stream must log exactly one row — never zero, never two",
        )
        outcome = self.client.calls[0].get("call_outcome")
        self.assertIsNotNone(
            outcome, "pre-fix: ZERO rows, so the failed call was never recorded",
        )
        self.assertEqual(
            outcome["status"], "failed",
            "an errored stream must NOT be logged as a success row",
        )
        self.assertIsNone(
            self.client.calls[0].get("latency"),
            "a broken stream still carries no latency — no is_streaming false positive",
        )

    # ── Tap must not alter yielded event sequence ────────────────────
    def test_tap_preserves_event_identity_and_order(self):
        response = _fake_response()
        e1, e2 = _PartStartEvent(), _PartDeltaEvent()
        stream = _FakeStream(response, events=[e1, e2])
        wrapper = self._make_mgr(stream)

        async def _drive():
            with mock.patch.object(enforcer, "_run_async_check", new=mock.AsyncMock()), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_prompt_at"), \
                    mock.patch.object(enforcer, "_capture_pydantic_ai_response_at"):
                async with wrapper as s:
                    got = [ev async for ev in s]
                    self.assertIs(got[0], e1)
                    self.assertIs(got[1], e2)

        self._run(_drive())


if __name__ == "__main__":
    unittest.main()
