"""Latency capture (TTFT + streaming throughput) for the manual-mode stream wrapper.

These cover both the metric math (`_build_stream_latency`, `_stream_acc_has_content`)
and the fail-safety risk register for the streaming tap (A1-A8 in
latency_metrics_design.md): a probe failure, an early consumer break, a mid-stream
error, and a clock anomaly must never break the customer's stream, alter the chunk
sequence, or fabricate a metric.

All fakes — no provider SDKs involved.
"""
import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police.enforcer import (
    _build_non_stream_latency,
    _build_stream_latency,
    _stream_acc_has_content,
    _new_stream_accumulator,
    _accumulate_stream_chunk,
    _wrap_sync_stream,
)


def _content_chunk(text):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=None))], usage=None)


def _usage_chunk(inp, out):
    # Final chunk: no content delta, carries usage so the wrapper latches `last`.
    return NS(choices=[NS(delta=NS(content=None, tool_calls=None))],
              usage=NS(prompt_tokens=inp, completion_tokens=out))


class _FakeSession:
    """Minimal stand-in; the wrapper only reads/sets ad-hoc attributes."""
    pass


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — asyncio.run
    unsets the main-thread event loop, breaking later get_event_loop()-based
    tests in the same pytest process."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestLatencyMath(unittest.TestCase):
    def test_build_latency_basic(self):
        lat = _build_stream_latency(100.0, 100.4, 101.8)
        self.assertTrue(lat["is_streaming"])
        self.assertEqual(lat["ttft_ms"], 400)
        self.assertEqual(lat["total_ms"], 1800)
        self.assertEqual(lat["generation_ms"], 1400)
        self.assertEqual(lat["clock"], "monotonic")
        self.assertIsNone(lat["output_tokens"])  # server supplies the count

    def test_build_latency_no_content_chunk_yields_null_ttft(self):
        # ttft_mono None (no content ever seen) → ttft/generation null, total set.
        lat = _build_stream_latency(100.0, None, 101.0)
        self.assertIsNone(lat["ttft_ms"])
        self.assertIsNone(lat["generation_ms"])
        self.assertEqual(lat["total_ms"], 1000)

    def test_build_latency_clamps_clock_anomaly(self):
        # A6: a backwards clock (NTP step) must clamp to 0, never go negative.
        lat = _build_stream_latency(100.0, 99.0, 98.0)
        self.assertEqual(lat["ttft_ms"], 0)
        self.assertEqual(lat["total_ms"], 0)
        self.assertEqual(lat["generation_ms"], 0)

    def test_build_latency_missing_anchor_returns_none(self):
        self.assertIsNone(_build_stream_latency(None, 1.0, 2.0))

    def test_acc_has_content_flips_on_first_text(self):
        acc = _new_stream_accumulator("openai")
        self.assertFalse(_stream_acc_has_content("openai", acc))  # empty
        _accumulate_stream_chunk("openai", acc, _content_chunk("hi"))
        self.assertTrue(_stream_acc_has_content("openai", acc))

    def test_acc_has_content_none_acc_is_best_effort_true(self):
        # Provider not accumulated → can't introspect → treat first chunk as content.
        self.assertTrue(_stream_acc_has_content("some_unaccumulated_provider", None))


class TestStreamWrapperLatency(unittest.TestCase):
    def _run(self, chunks, req_start=1000.0):
        """Drive _wrap_sync_stream over `chunks`, capturing the latency passed to
        _log_manual. Returns (yielded_chunks, captured_latency)."""
        captured = {}

        # `**_kw` keeps the double signature-transparent (obs_key etc.): a
        # TypeError here is swallowed by the fail-open caller, so the fake would
        # silently never record.
        def _fake_log_manual(provider, session, kwargs, last, order, span_name,
                             start_time, latency=None, **_kw):
            captured["latency"] = latency

        session = _FakeSession()
        with mock.patch.object(enforcer, "_log_manual", _fake_log_manual), \
             mock.patch.object(enforcer, "_capture_composition_at", lambda *a, **k: None):
            gen = _wrap_sync_stream(iter(chunks), "openai", session, {}, 0,
                                    "span", None, req_start_mono=req_start)
            yielded = list(gen)
        return yielded, captured.get("latency")

    def test_happy_path_emits_ttft_and_total(self):
        chunks = [_content_chunk("Hello"), _content_chunk(" world"), _usage_chunk(10, 5)]
        # T-P5 / G4: capture that a usage-bearing stream dispatches exactly one log.
        log_calls = []

        def _fake_log_manual(provider, session, kwargs, last, order, span_name,
                             start_time, latency=None, **_kw):
            log_calls.append({
                "provider": provider,
                "input_tokens": getattr(getattr(last, "usage", None), "prompt_tokens", 0) or 0,
                "output_tokens": getattr(getattr(last, "usage", None), "completion_tokens", 0) or 0,
                "latency": latency,
            })

        session = _FakeSession()
        with mock.patch.object(enforcer, "_log_manual", _fake_log_manual), \
             mock.patch.object(enforcer, "_capture_composition_at", lambda *a, **k: None):
            gen = _wrap_sync_stream(iter(chunks), "openai", session, {}, 0,
                                    "span", None, req_start_mono=1000.0)
            yielded = list(gen)
        self.assertEqual(yielded, chunks)
        from _stream_presence import assert_streamed_log_present
        rows = assert_streamed_log_present(log_calls, provider="openai", min_tokens=1)
        latency = rows[0]["latency"]
        self.assertIsNotNone(latency)
        self.assertTrue(latency["is_streaming"])
        self.assertIsNotNone(latency["ttft_ms"])
        self.assertGreaterEqual(latency["total_ms"], latency["ttft_ms"])

    def test_A1_probe_exception_never_breaks_stream(self):
        # A1: the content probe raising must degrade to "no TTFT", not abort.
        chunks = [_content_chunk("a"), _content_chunk("b"), _usage_chunk(10, 5)]
        with mock.patch.object(enforcer, "_stream_acc_has_content",
                               side_effect=RuntimeError("boom")):
            yielded, latency = self._run(chunks)
        self.assertEqual(yielded, chunks)            # stream survived
        self.assertIsNotNone(latency)                 # still logged
        self.assertIsNone(latency["ttft_ms"])         # probe never succeeded

    def test_A3_early_consumer_break_still_logs_without_throwing(self):
        # Consumer breaks after the first chunk; the finally must run, log once,
        # and never raise. No usage chunk seen → `last` is None → no _log_manual.
        captured = {"called": False}

        def _fake_log_manual(*a, **k):
            captured["called"] = True

        session = _FakeSession()
        with mock.patch.object(enforcer, "_log_manual", _fake_log_manual), \
             mock.patch.object(enforcer, "_capture_composition_at", lambda *a, **k: None):
            gen = _wrap_sync_stream(iter([_content_chunk("x"), _usage_chunk(1, 1)]),
                                    "openai", session, {}, 0, "span", None,
                                    req_start_mono=1000.0)
            first = next(gen)            # pull one chunk
            gen.close()                  # early break → GeneratorExit into finally
        self.assertIsNotNone(first)      # no throw
        # `last` was never set (usage chunk not reached) → wrapper skips _log_manual.
        self.assertFalse(captured["called"])

    def test_A8_mid_stream_error_propagates_and_logs_no_latency(self):
        # A8: an error from the underlying stream re-raises verbatim and the
        # success-path latency log never fires.
        def _boom_stream():
            yield _content_chunk("partial")
            raise ValueError("network drop")

        captured = {"latency_logged": False}

        def _fake_log_manual(*a, **k):
            captured["latency_logged"] = True

        session = _FakeSession()
        with mock.patch.object(enforcer, "_log_manual", _fake_log_manual), \
             mock.patch.object(enforcer, "_emit_call_failure_log", lambda *a, **k: None), \
             mock.patch.object(enforcer, "build_call_outcome", lambda *a, **k: {}):
            gen = _wrap_sync_stream(_boom_stream(), "openai", session, {}, 0,
                                    "span", None, req_start_mono=1000.0)
            with self.assertRaises(ValueError):
                list(gen)
        self.assertFalse(captured["latency_logged"])  # no success-path log


class TestNonStreamLatencyMath(unittest.TestCase):
    """Fix: manual NON-streaming calls previously logged with no latency at all
    (duration_ms lost). The non-stream builder mirrors _build_stream_latency's
    key set with the streaming-only fields null."""

    def test_build_non_stream_latency_keys(self):
        lat = _build_non_stream_latency(100.0, 100.25)
        self.assertEqual(lat, {
            "is_streaming": False,
            "ttft_ms": None,
            "total_ms": 250,
            "generation_ms": None,
            "output_tokens": None,
            "clock": "monotonic",
        })

    def test_missing_anchor_returns_none(self):
        self.assertIsNone(_build_non_stream_latency(None, 100.0))
        self.assertIsNone(_build_non_stream_latency(100.0, None))

    def test_clock_anomaly_clamps_to_zero(self):
        self.assertEqual(_build_non_stream_latency(100.0, 99.0)["total_ms"], 0)


class TestManualNonStreamWrapperLatency(unittest.TestCase):
    """The manual sync/async wrappers must pass a non-streaming latency dict to
    _log_manual; every other argument stays as before."""

    def _install(self, is_async):
        captured = {}

        # `**_kw` absorbs current + future keyword args (obs_key etc.) so the
        # double stays signature-transparent; see the note in _run above.
        def fake_log_manual(provider, session, kwargs, result, order, span_name,
                            start_time, operation="chat", shape_override=None,
                            args=None, latency=None, **_kw):
            captured["provider"] = provider
            captured["latency"] = latency

        class FakeAPI:
            pass

        if is_async:
            async def create(self, **kwargs):
                return NS(model="m-1", usage=NS(prompt_tokens=3, completion_tokens=2))
        else:
            def create(self, **kwargs):
                return NS(model="m-1", usage=NS(prompt_tokens=3, completion_tokens=2))

        FakeAPI.create = create
        patches = [
            mock.patch.object(enforcer, "_log_manual", fake_log_manual),
            mock.patch.object(enforcer, "_capture_composition_at", lambda *a, **k: None),
            mock.patch.object(enforcer, "maybe_register_openai_agents_tracing", lambda: None),
            mock.patch.object(enforcer, "_run_sync_check", lambda **k: None),
            mock.patch.object(enforcer, "_run_async_check", self._noop_async),
        ]
        return FakeAPI, create, captured, patches

    @staticmethod
    async def _noop_async(**kwargs):
        return None

    def test_sync_manual_wrapper_passes_non_stream_latency(self):
        FakeAPI, orig, captured, patches = self._install(is_async=False)
        with patches[0], patches[1], patches[2], patches[3]:
            enforcer._set_manual_wrapper(FakeAPI, "create", orig, "cerebras", False)
            FakeAPI().create(model="m-1")
        lat = captured["latency"]
        self.assertIsNotNone(lat)
        self.assertFalse(lat["is_streaming"])
        self.assertIsNone(lat["ttft_ms"])
        self.assertIsNone(lat["generation_ms"])
        self.assertGreaterEqual(lat["total_ms"], 0)

    def test_async_manual_wrapper_passes_non_stream_latency(self):
        FakeAPI, orig, captured, patches = self._install(is_async=True)
        with patches[0], patches[1], patches[2], patches[4]:
            enforcer._set_manual_wrapper(FakeAPI, "create", orig, "cerebras", True)
            _run_coro(FakeAPI().create(model="m-1"))
        lat = captured["latency"]
        self.assertIsNotNone(lat)
        self.assertFalse(lat["is_streaming"])
        self.assertGreaterEqual(lat["total_ms"], 0)


class _FakeAnthropicStream:
    """Stand-in for anthropic MessageStream: iterable events + final message."""

    def __init__(self, events, final_msg):
        self._events = events
        self._final = final_msg

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final

    @property
    def text_stream(self):
        for ev in self._events:
            t = getattr(getattr(ev, "delta", None), "text", None)
            if t:
                yield t


class _FakeAnthropicAsyncStream(_FakeAnthropicStream):
    def __aiter__(self):
        async def _agen():
            for ev in self._events:
                yield ev
        return _agen()

    async def get_final_message(self):
        return self._final


class _FakeMgr:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *a):
        return False


class _FakeAsyncMgr(_FakeMgr):
    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *a):
        return False


def _anthropic_events():
    return [
        NS(type="message_start"),
        NS(type="content_block_start"),
        NS(type="content_block_delta", delta=NS(text="Hel")),
        NS(type="content_block_delta", delta=NS(text="lo")),
        NS(type="message_stop"),
    ]


def _final_message():
    return NS(
        model="claude-3-5-sonnet",
        content=[NS(type="text", text="Hello")],
        usage=NS(input_tokens=10, output_tokens=5,
                 cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


def _anthropic_tool_only_events():
    """A tool-call-only turn: tool_use content_block_start + input_json_delta
    deltas — NO text deltas, so the SDK's text_stream yields zero chunks."""
    return [
        NS(type="message_start"),
        NS(type="content_block_start",
           content_block=NS(type="tool_use", name="lookup_invoice")),
        NS(type="content_block_delta",
           delta=NS(type="input_json_delta", partial_json='{"invoice_id":')),
        NS(type="content_block_delta",
           delta=NS(type="input_json_delta", partial_json='"INV-42"}')),
        NS(type="content_block_stop"),
        NS(type="message_stop"),
    ]


def _tool_only_final_message():
    return NS(
        model="claude-3-5-sonnet",
        content=[NS(type="tool_use", name="lookup_invoice",
                    input={"invoice_id": "INV-42"})],
        usage=NS(input_tokens=10, output_tokens=5,
                 cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class TestAnthropicStreamMgrLatency(unittest.TestCase):
    """Fix: the anthropic `messages.stream()` context-manager wrapper logged
    tokens but NO latency. The wrapper must anchor _req_start_mono at enter,
    mark TTFT on the first content event, and pass the stream latency dict."""

    def _capture_log(self):
        captured = {}

        class FakeTP:
            def log_sync(self, **kw):
                captured.update(kw)

        return captured, FakeTP()

    def test_sync_wrapper_emits_latency_with_ttft(self):
        events = _anthropic_events()
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(events, _final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                seen = list(stream)
        # Chunk passthrough is untouched, in order.
        self.assertEqual(seen, events)
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertTrue(lat["is_streaming"])
        self.assertIsNotNone(lat["ttft_ms"])
        self.assertGreaterEqual(lat["total_ms"], lat["ttft_ms"])
        # Tokens keep flowing as before.
        self.assertEqual(captured["input_tokens"], 10)
        self.assertEqual(captured["output_tokens"], 5)

    def test_sync_wrapper_no_iteration_still_logs_total_only(self):
        # Customer enters the CM and only calls get_final_message() — no
        # iteration means no observed first-content event: ttft stays None,
        # total_ms still recorded.
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(_anthropic_events(), _final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                stream.get_final_message()
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertIsNone(lat["ttft_ms"])
        self.assertGreaterEqual(lat["total_ms"], 0)

    def test_sync_wrapper_text_stream_marks_ttft(self):
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(_anthropic_events(), _final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                text = "".join(stream.text_stream)
        self.assertEqual(text, "Hello")
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertIsNotNone(lat["ttft_ms"])

    def test_async_wrapper_emits_latency_with_ttft(self):
        events = _anthropic_events()
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicAsyncStreamMgrWrapper(
            _FakeAsyncMgr(_FakeAnthropicAsyncStream(events, _final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )

        async def _run():
            seen = []
            async with wrapper as stream:
                async for ev in stream:
                    seen.append(ev)
            return seen

        with mock.patch.object(enforcer, "get_client", return_value=tp):
            seen = _run_coro(_run())
        self.assertEqual(seen, events)
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertTrue(lat["is_streaming"])
        self.assertIsNotNone(lat["ttft_ms"])
        self.assertGreaterEqual(lat["total_ms"], lat["ttft_ms"])

    def test_sync_wrapper_text_stream_tool_only_turn_marks_ttft(self):
        # minimax_anthropic stream repro: the app drains stream.text_stream,
        # but the turn is tool-calls only — the SDK text filter yields ZERO
        # chunks, so the old output-tap never observed the first served token
        # and ttft was lost. text_stream now rides the EVENT iteration, which
        # marks TTFT on the tool_use content_block_start / input_json_delta
        # events while yielding the same (empty) text.
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(_anthropic_tool_only_events(),
                                          _tool_only_final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "refund INV-42"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                text = "".join(stream.text_stream)
        # No text chunks fabricated — customer-visible output unchanged.
        self.assertEqual(text, "")
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertIsNotNone(lat["ttft_ms"])
        self.assertGreaterEqual(lat["total_ms"], lat["ttft_ms"])

    def test_sync_wrapper_event_iteration_tool_only_turn_marks_ttft(self):
        # Event-iteration consumers must mark TTFT on ANY content_block_*
        # event regardless of block type.
        events = _anthropic_tool_only_events()
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(events, _tool_only_final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "refund INV-42"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                seen = list(stream)
        self.assertEqual(seen, events)
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertIsNotNone(lat["ttft_ms"])

    def test_text_stream_chunks_unchanged_on_text_turn(self):
        # The event-driven text filter must reproduce the SDK's text output
        # exactly on a normal text turn (chunk boundaries included).
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(_FakeAnthropicStream(_anthropic_events(), _final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                chunks = list(stream.text_stream)
        self.assertEqual(chunks, ["Hel", "lo"])
        self.assertIsNotNone(captured.get("latency"))

    def test_proxy_failure_degrades_to_raw_stream(self):
        # If proxy construction blows up, the customer gets the raw stream.
        raw = _FakeAnthropicStream(_anthropic_events(), _final_message())
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _FakeMgr(raw), {"model": "m"})
        with mock.patch.object(enforcer, "_AnthropicMessageStreamProxy",
                               side_effect=RuntimeError("boom")):
            with mock.patch.object(enforcer, "get_client", return_value=None):
                with wrapper as stream:
                    self.assertIs(stream, raw)

    def test_proxy_forwards_attributes(self):
        raw = _FakeAnthropicStream(_anthropic_events(), _final_message())
        proxy = enforcer._AnthropicMessageStreamProxy(raw, lambda e: None)
        self.assertIs(proxy.get_final_message(), raw.get_final_message())

    def test_b06_slow_enter_span_start_includes_handshake(self):
        # Path #2: when mgr.__enter__ sleeps (slow gateway handshake),
        # span.start_time must be stamped BEFORE enter so span wall duration
        # covers the handshake window that TTFT also includes.
        import time
        from datetime import datetime, timedelta, timezone

        class _SlowMgr(_FakeMgr):
            def __enter__(self):
                time.sleep(0.06)
                return self._stream

        events = _anthropic_events()
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicStreamMgrWrapper(
            _SlowMgr(_FakeAnthropicStream(events, _final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )
        t_before = datetime.now(timezone.utc)
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            with wrapper as stream:
                seen = list(stream)

        self.assertEqual(seen, events)
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertIsNotNone(lat["ttft_ms"])
        self.assertGreaterEqual(lat["total_ms"], lat["ttft_ms"])

        span = captured.get("span") or {}
        start_s = span.get("start_time")
        end_s = span.get("end_time")
        self.assertIsNotNone(start_s)
        self.assertIsNotNone(end_s)
        start_dt = datetime.fromisoformat(start_s)
        end_dt = datetime.fromisoformat(end_s)
        # start stamped at request start (pre-handshake), not after enter sleep
        self.assertLessEqual(start_dt, t_before + timedelta(milliseconds=30))
        span_dur_ms = int((end_dt - start_dt).total_seconds() * 1000)
        # Handshake ~60ms is inside the span; wall must exceed pure post-enter generation
        self.assertGreaterEqual(span_dur_ms, 50)
        # Server will use max(span, total_ms); invariant holds locally too
        self.assertGreaterEqual(max(span_dur_ms, lat["total_ms"]), lat["ttft_ms"])

    def test_b06_async_slow_aenter_span_start_includes_handshake(self):
        from datetime import datetime, timedelta, timezone

        class _SlowAsyncMgr(_FakeAsyncMgr):
            async def __aenter__(self):
                await asyncio.sleep(0.06)
                return self._stream

        events = _anthropic_events()
        captured, tp = self._capture_log()
        wrapper = enforcer._AnthropicAsyncStreamMgrWrapper(
            _SlowAsyncMgr(_FakeAnthropicAsyncStream(events, _final_message())),
            {"model": "claude-3-5-sonnet",
             "messages": [{"role": "user", "content": "hi"}]},
        )

        async def _run():
            seen = []
            async with wrapper as stream:
                async for ev in stream:
                    seen.append(ev)
            return seen

        t_before = datetime.now(timezone.utc)
        with mock.patch.object(enforcer, "get_client", return_value=tp):
            seen = _run_coro(_run())
        self.assertEqual(seen, events)
        lat = captured.get("latency")
        self.assertIsNotNone(lat)
        self.assertIsNotNone(lat["ttft_ms"])
        span = captured.get("span") or {}
        start_dt = datetime.fromisoformat(span["start_time"])
        end_dt = datetime.fromisoformat(span["end_time"])
        self.assertLessEqual(start_dt, t_before + timedelta(milliseconds=30))
        span_dur_ms = int((end_dt - start_dt).total_seconds() * 1000)
        self.assertGreaterEqual(span_dur_ms, 50)
        self.assertGreaterEqual(max(span_dur_ms, lat["total_ms"]), lat["ttft_ms"])


if __name__ == "__main__":
    unittest.main()
