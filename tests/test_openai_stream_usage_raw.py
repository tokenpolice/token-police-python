"""G3-14-1 — verbatim final-chunk usage forwarding for OpenAI-wire STREAMS.

The OpenLLMetry openai instrumentor never sets ``gen_ai.usage.reasoning_tokens``
on STREAMED spans, so the Mode-A deferred payload's attribute-synthesized
``usage.raw`` silently dropped ``completion_tokens_details.reasoning_tokens``.
On xAI/Grok (openai driver + custom base_url) ``completion_tokens`` EXCLUDES
reasoning, so the loss is a silent UNDERCHARGE.

Fix under test: the Mode-A stream wrappers keep a last-write-wins reference to
each chunk's ``usage`` (openai wire only), serialize it verbatim at clean stream
end via ``_serialize_openai_stream_usage``, hand it to
``_capture_response_composition(usage_raw=...)`` which stashes it under the same
reserved-order ``comp_key``, and the pre-existing ``_flush_deferred_spans`` merge
replaces ``payload["usage"]["raw"]`` wholesale.

Golden rule: every path here is fail-open. Hostile chunks, garbage usage and
mid-stream provider errors must leave the customer's chunk sequence and
exceptions byte-identical — the SDK never raises into customer code from this
seam.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import TPSession, _current_session

# Captured at import time, BEFORE the autouse no-op fixture swaps the module
# attribute out, so the end-to-end merge tests can drive the real thing.
_REAL_FLUSH = enforcer._flush_deferred_spans


# ─────────────────────────────────────────────────────────────────────────
# The verbatim xAI/Grok usage payload the collector's shape mapper expects:
# nested details + provider extras, no renaming / netting / re-derivation.
# ─────────────────────────────────────────────────────────────────────────
XAI_USAGE = {
    "prompt_tokens": 11,
    "completion_tokens": 1,
    "total_tokens": 369,
    "prompt_tokens_details": {"cached_tokens": 4, "text_tokens": 7},
    "completion_tokens_details": {"reasoning_tokens": 357},
    "cost_in_usd_ticks": 42,
}


# The openai-shaped Mode-A accumulator is getattr-based (typed SDK chunk
# objects), so content chunks must be objects for a synthetic response — and
# hence the composition capture — to be produced at all.
def _content(text, usage=None):
    return NS(choices=[NS(delta=NS(content=text, tool_calls=None), index=0)],
              usage=usage)


def _usage_only(usage):
    """The terminal usage-only chunk include_usage appends (empty choices)."""
    return NS(choices=[], usage=usage)


def _usage_only_dict(usage):
    """Same terminal chunk as a plain dict — exercises the wrapper's
    ``chunk.get("usage")`` fallback (some wire shims hand back dicts)."""
    return {"choices": [], "usage": usage}


# ─── fakes (mirrors tests/test_sec02_stream_surface.py) ──────────────────
class FakeSyncStream:
    def __init__(self, chunks):
        self._it = iter(list(chunks))

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def close(self):
        pass


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        pass


class RaisingSyncStream:
    """Yields ``chunks`` then raises ``exc`` (mid-stream provider error)."""

    def __init__(self, chunks, exc):
        self._it = iter(list(chunks))
        self._exc = exc

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise self._exc


class RaisingAsyncStream:
    def __init__(self, chunks, exc):
        self._chunks = list(chunks)
        self._exc = exc

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise self._exc


class HostileUsageChunk:
    """A chunk whose ``usage`` attribute access blows up."""

    choices = []

    @property
    def usage(self):
        raise RuntimeError("usage property exploded")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _drain_async(proxy):
    out = []
    async for c in proxy:
        out.append(c)
    return out


# ─── fixtures ────────────────────────────────────────────────────────────
@pytest.fixture()
def sess():
    """A real TPSession installed in the contextvar — the wrappers pass the
    session explicitly but ``_capture_response_composition`` resolves it via
    ``get_current_session()``, so both must be the same object."""
    s = TPSession(trace_id="a" * 32)
    s._span_counter = 0
    s._deferred_spans = []
    s._defer_telemetry = False
    s._pending_compositions = {}
    tok = _current_session.set(s)
    try:
        yield s
    finally:
        _current_session.reset(tok)


@pytest.fixture(autouse=True)
def _noop_flush(monkeypatch):
    """Neutralize the real end-of-stream flush so the stash stays observable
    (the flush POPS the comp_key). The end-to-end merge tests below drive
    ``_flush_deferred_spans`` directly instead."""
    monkeypatch.setattr(enforcer, "_flush_deferred_spans", MagicMock())


@pytest.fixture(autouse=True)
def _noop_failure_log(monkeypatch):
    monkeypatch.setattr(enforcer, "_emit_call_failure_log", MagicMock())


def _stash(session, order):
    return session._pending_compositions.get(f"{session.trace_id}:{order}", {})


# ─────────────────────────────────────────────────────────────────────────
# 0. The code under test really is the worktree copy.
# ─────────────────────────────────────────────────────────────────────────
def test_enforcer_module_is_this_worktree():
    assert Path(enforcer.__file__).resolve() == (
        ROOT / "token_police" / "enforcer.py").resolve()


# ─────────────────────────────────────────────────────────────────────────
# 1. _serialize_openai_stream_usage — helper unit tests
# ─────────────────────────────────────────────────────────────────────────
class TestSerializeOpenAIStreamUsage:
    def test_verbatim_pass_through(self):
        src = {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "completion_tokens_details": {"reasoning_tokens": 357},
            "cost_in_usd_ticks": 42,
        }
        out = enforcer._serialize_openai_stream_usage(src)
        # EXACT verbatim: nested details + provider extras preserved, no
        # renaming, no netting reasoning into completion_tokens.
        assert out == {
            "prompt_tokens": 5,
            "completion_tokens": 1,
            "completion_tokens_details": {"reasoning_tokens": 357},
            "cost_in_usd_ticks": 42,
        }
        assert out["completion_tokens"] == 1          # NOT 358
        assert out["completion_tokens_details"]["reasoning_tokens"] == 357
        assert out["cost_in_usd_ticks"] == 42

    def test_full_xai_shape_round_trips_unchanged(self):
        assert enforcer._serialize_openai_stream_usage(XAI_USAGE) == XAI_USAGE

    def test_zero_tokens_dict_is_none(self):
        assert enforcer._serialize_openai_stream_usage(
            {"prompt_tokens": 0, "completion_tokens": 0}) is None

    def test_empty_dict_is_none(self):
        assert enforcer._serialize_openai_stream_usage({}) is None

    def test_none_is_none(self):
        assert enforcer._serialize_openai_stream_usage(None) is None

    @pytest.mark.parametrize("garbage", ["nope", 42, 3.5, True, [1, 2, 3], object()])
    def test_non_dict_garbage_is_none(self, garbage):
        assert enforcer._serialize_openai_stream_usage(garbage) is None

    def test_object_with_dunder_dict_positive_tokens(self):
        usage = NS(
            prompt_tokens=11,
            completion_tokens=1,
            completion_tokens_details=NS(reasoning_tokens=357),
            cost_in_usd_ticks=42,
        )
        out = enforcer._serialize_openai_stream_usage(usage)
        assert isinstance(out, dict)
        assert out["prompt_tokens"] == 11
        assert out["completion_tokens_details"]["reasoning_tokens"] == 357
        assert out["cost_in_usd_ticks"] == 42

    def test_object_with_zero_tokens_is_none(self):
        assert enforcer._serialize_openai_stream_usage(
            NS(prompt_tokens=0, completion_tokens=0)) is None

    def test_only_completion_tokens_positive_still_kept(self):
        out = enforcer._serialize_openai_stream_usage({"completion_tokens": 7})
        assert out == {"completion_tokens": 7}

    def test_hostile_object_returns_none_never_raises(self):
        class Boom:
            def __getattr__(self, name):
                raise RuntimeError("boom")

            def __repr__(self):
                raise RuntimeError("boom repr")

        assert enforcer._serialize_openai_stream_usage(Boom()) is None

    def test_non_numeric_token_counts_fail_closed_to_none(self):
        # int("abc") raises inside the guard → fail-open → None, never raises.
        assert enforcer._serialize_openai_stream_usage(
            {"prompt_tokens": "abc", "completion_tokens": "xyz"}) is None


# ─────────────────────────────────────────────────────────────────────────
# 2. Wrapper → stash integration (sync + async)
# ─────────────────────────────────────────────────────────────────────────
class TestStreamStashSync:
    def test_terminal_usage_chunk_stashed_verbatim_at_explicit_order(self, sess):
        chunks = [_content("hi"), _content(" there"), _usage_only(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=3)
        out = list(wrapped)

        assert out == chunks                      # customer sees everything
        assert _stash(sess, 3)["usage_raw"] == XAI_USAGE
        # keyed to the RESERVED order, not the _span_counter-1 fallback
        assert "usage_raw" not in _stash(sess, -1)
        assert f"{sess.trace_id}:-1" not in sess._pending_compositions

    def test_suppressed_usage_chunk_still_stashes(self, sess):
        chunks = [_content("hi"), _usage_only(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False,
            suppress_usage_chunk=True, order=0)
        out = list(wrapped)

        # SDK-injected include_usage: the customer must NOT see the usage chunk
        assert out == [chunks[0]]
        assert all(c.choices != [] for c in out)
        # ...but the metering still captured it
        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_unsuppressed_usage_chunk_yielded_and_stashed(self, sess):
        chunks = [_content("hi"), _usage_only(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False,
            suppress_usage_chunk=False, order=0)
        out = list(wrapped)

        assert out == chunks
        assert out[-1] is chunks[-1]              # same object, untouched
        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_usage_on_last_content_chunk_captured(self, sess):
        """No usage-only terminal chunk: usage rides the final content chunk."""
        chunks = [_content("hel"), _content("lo", usage=XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)
        out = list(wrapped)

        assert out == chunks
        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_interim_cumulative_usage_final_wins(self, sess):
        interim_a = {"prompt_tokens": 11, "completion_tokens": 1,
                     "completion_tokens_details": {"reasoning_tokens": 20}}
        interim_b = {"prompt_tokens": 11, "completion_tokens": 1,
                     "completion_tokens_details": {"reasoning_tokens": 120}}
        chunks = [
            _content("a", usage=interim_a),
            _content("b", usage=interim_b),
            _usage_only(XAI_USAGE),
        ]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)
        list(wrapped)

        stashed = _stash(sess, 0)["usage_raw"]
        assert stashed == XAI_USAGE               # last write wins
        assert stashed["completion_tokens_details"]["reasoning_tokens"] == 357

    def test_typed_usage_object_serialized(self, sess):
        """A typed-SDK usage OBJECT (not a dict) still forwards verbatim."""
        chunks = [
            _content("hi"),
            _usage_only(NS(prompt_tokens=11, completion_tokens=1,
                           completion_tokens_details=NS(reasoning_tokens=357))),
        ]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)
        list(wrapped)

        stashed = _stash(sess, 0)["usage_raw"]
        assert stashed["completion_tokens_details"]["reasoning_tokens"] == 357
        assert stashed["completion_tokens"] == 1

    def test_dict_chunk_usage_key_fallback(self, sess):
        """Chunk is a plain dict → the wrapper's ``chunk.get('usage')``
        fallback (getattr returns None on a dict) must still capture."""
        chunks = [_content("hi"), _usage_only_dict(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)
        list(wrapped)

        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_zero_usage_leaves_no_stash(self, sess):
        chunks = [_content("hi"),
                  _usage_only({"prompt_tokens": 0, "completion_tokens": 0})]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)
        list(wrapped)

        assert "usage_raw" not in _stash(sess, 0)

    def test_absent_usage_leaves_no_stash(self, sess):
        chunks = [_content("hi"), _content(" there")]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)
        list(wrapped)

        # payload keeps its OTel-synth raw untouched
        assert "usage_raw" not in _stash(sess, 0)


class TestStreamStashAsync:
    def test_terminal_usage_chunk_stashed_verbatim_at_explicit_order(self, sess):
        chunks = [_content("hi"), _content(" there"), _usage_only(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "openai", sess, False, order=3)
        out = _run(_drain_async(wrapped))

        assert out == chunks
        assert _stash(sess, 3)["usage_raw"] == XAI_USAGE
        assert f"{sess.trace_id}:-1" not in sess._pending_compositions

    def test_suppressed_usage_chunk_still_stashes(self, sess):
        chunks = [_content("hi"), _usage_only(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "openai", sess, False,
            suppress_usage_chunk=True, order=0)
        out = _run(_drain_async(wrapped))

        assert out == [chunks[0]]
        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_unsuppressed_usage_chunk_yielded_and_stashed(self, sess):
        chunks = [_content("hi"), _usage_only(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "openai", sess, False,
            suppress_usage_chunk=False, order=0)
        out = _run(_drain_async(wrapped))

        assert out == chunks
        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_usage_on_last_content_chunk_captured(self, sess):
        chunks = [_content("hel"), _content("lo", usage=XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "openai", sess, False, order=0)
        _run(_drain_async(wrapped))

        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_interim_cumulative_usage_final_wins(self, sess):
        chunks = [
            _content("a", usage={"prompt_tokens": 11, "completion_tokens": 1,
                                 "completion_tokens_details": {"reasoning_tokens": 20}}),
            _usage_only(XAI_USAGE),
        ]
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "openai", sess, False, order=0)
        _run(_drain_async(wrapped))

        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE

    def test_zero_usage_leaves_no_stash(self, sess):
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream([_content("hi"),
                             _usage_only({"prompt_tokens": 0, "completion_tokens": 0})]),
            "openai", sess, False, order=0)
        _run(_drain_async(wrapped))

        assert "usage_raw" not in _stash(sess, 0)

    def test_absent_usage_leaves_no_stash(self, sess):
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream([_content("hi")]), "openai", sess, False, order=0)
        _run(_drain_async(wrapped))

        assert "usage_raw" not in _stash(sess, 0)

    def test_sync_underlying_iterator_bridged(self, sess):
        """The async wrapper also drives a plain sync underlying stream."""
        chunks = [_content("hi"), _usage_only(XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)
        out = _run(_drain_async(wrapped))

        assert out == chunks
        assert _stash(sess, 0)["usage_raw"] == XAI_USAGE


# ─────────────────────────────────────────────────────────────────────────
# 3. Provider gate — per-provider taps read only their own wire attr
# (G3-O1: google now forwards too, via `usage_metadata` — see
# tests/test_google_verbatim_usage.py; the openai tap still reads `.usage`)
# ─────────────────────────────────────────────────────────────────────────
def _anthropic_chunks():
    usage = {"input_tokens": 11, "output_tokens": 358}
    return [
        NS(type="content_block_start", index=0, content_block=NS(type="text"),
           usage=usage),
        NS(type="content_block_delta", index=0,
           delta=NS(type="text_delta", text="hi"), usage=usage),
        NS(type="message_delta", delta=NS(stop_reason="end_turn"), usage=usage),
    ]


def _google_chunks():
    usage = {"prompt_token_count": 11, "candidates_token_count": 358}
    return [
        NS(candidates=[NS(content=NS(parts=[NS(text="hi", function_call=None)]))],
           usage=usage),
        NS(candidates=[NS(content=NS(parts=[NS(text=" there", function_call=None)]))],
           usage=usage),
    ]


class TestProviderGate:
    """Pin the per-provider tap boundaries.

    Since G3-O1 google streams DO forward usage — but only from the chunk's
    native ``usage_metadata`` attr (covered in
    tests/test_google_verbatim_usage.py). The google fixtures here carry an
    OPENAI-style ``usage`` attr instead, so these cases pin that the openai
    tap never picks up another provider's chunks and the google tap never
    reads the openai wire attr. Anthropic remains fully gated (cumulative
    semantics, no stream-usage forward at all)."""

    @pytest.mark.parametrize("provider,chunks", [
        ("anthropic", _anthropic_chunks()),
        ("google", _google_chunks()),
    ])
    def test_non_openai_streams_never_stash_usage_raw_sync(self, sess, provider, chunks):
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), provider, sess, False, order=0)
        out = list(wrapped)

        assert out == chunks                       # customer sequence unchanged
        assert "usage_raw" not in _stash(sess, 0)
        # the composition capture itself still ran (proves we reached the seam
        # and the GATE — not an early bail — is what suppressed usage_raw)
        assert _stash(sess, 0).get("response")

    @pytest.mark.parametrize("provider,chunks", [
        ("anthropic", _anthropic_chunks()),
        ("google", _google_chunks()),
    ])
    def test_non_openai_streams_never_stash_usage_raw_async(self, sess, provider, chunks):
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), provider, sess, False, order=0)
        out = _run(_drain_async(wrapped))

        assert out == chunks
        assert "usage_raw" not in _stash(sess, 0)
        assert _stash(sess, 0).get("response")


# ─────────────────────────────────────────────────────────────────────────
# 4. Failure paths — the golden rule (never raise into customer code)
# ─────────────────────────────────────────────────────────────────────────
class TestFailurePaths:
    def test_mid_stream_error_no_stash_and_reraised_verbatim_sync(self, sess):
        sentinel = ValueError("provider dropped the connection")
        chunks = [_content("hi"), _content("lo", usage=XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            RaisingSyncStream(chunks, sentinel), "openai", sess, False, order=0)

        seen = []
        with pytest.raises(ValueError) as ei:
            for c in wrapped:
                seen.append(c)

        assert ei.value is sentinel                # verbatim, not wrapped
        assert seen == chunks                      # chunks before the error kept
        # failed stream → composition/usage capture is skipped entirely
        assert "usage_raw" not in _stash(sess, 0)

    def test_mid_stream_error_no_stash_and_reraised_verbatim_async(self, sess):
        sentinel = ValueError("provider dropped the connection")
        chunks = [_content("hi"), _content("lo", usage=XAI_USAGE)]
        wrapped = enforcer._wrap_mode_a_async_stream(
            RaisingAsyncStream(chunks, sentinel), "openai", sess, False, order=0)

        seen = []

        async def go():
            async for c in wrapped:
                seen.append(c)

        with pytest.raises(ValueError) as ei:
            _run(go())

        assert ei.value is sentinel
        assert seen == chunks
        assert "usage_raw" not in _stash(sess, 0)

    def test_hostile_usage_attribute_never_breaks_iteration_sync(self, sess):
        hostile = HostileUsageChunk()
        chunks = [_content("hi"), hostile]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "openai", sess, False, order=0)

        out = list(wrapped)                        # must NOT raise

        assert out[0] == chunks[0]
        assert out[1] is hostile                   # delivered unchanged
        assert "usage_raw" not in _stash(sess, 0)

    def test_hostile_usage_attribute_never_breaks_iteration_async(self, sess):
        hostile = HostileUsageChunk()
        chunks = [_content("hi"), hostile]
        wrapped = enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "openai", sess, False, order=0)

        out = _run(_drain_async(wrapped))           # must NOT raise

        assert out[0] == chunks[0]
        assert out[1] is hostile
        assert "usage_raw" not in _stash(sess, 0)

    def test_hostile_usage_attribute_with_suppress_on(self, sess):
        """suppress_usage_chunk also walks the hostile chunk — still no crash."""
        hostile = HostileUsageChunk()
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream([_content("hi"), hostile]), "openai", sess, False,
            suppress_usage_chunk=True, order=0)

        out = list(wrapped)

        assert hostile in out
        assert "usage_raw" not in _stash(sess, 0)

    def test_garbage_usage_value_leaves_no_stash(self, sess):
        """A chunk whose usage is a non-dict scalar → serializer returns None."""
        wrapped = enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream([_content("hi"), _usage_only("not-a-usage-object")]),
            "openai", sess, False, order=0)

        out = list(wrapped)

        assert len(out) == 2
        assert "usage_raw" not in _stash(sess, 0)


# ─────────────────────────────────────────────────────────────────────────
# 5. End-to-end merge in _flush_deferred_spans
#    (mirrors tests/test_gemini_tts_operation.py::test_flush_injects_*)
# ─────────────────────────────────────────────────────────────────────────
def _openai_stream_payload(session, span_order=0):
    """A Mode-A deferred payload as on_end builds it for a STREAMED openai call:
    OTel-attribute-synthesized usage.raw with NO completion_tokens_details —
    exactly the shape that silently drops reasoning tokens."""
    return {
        "user_id": "u",
        "paid_plan": "free",
        "workflow_name": "w",
        "session_id": "",
        "model": "grok-4-fast-reasoning",
        "provider": "openai",
        "input_tokens": 11,
        "output_tokens": 1,
        "cached_tokens": 0,
        "metadata": {},
        "span": {
            "trace_id": session.trace_id,
            "span_id": "d" * 16,
            "parent_span_id": "",
            "span_kind": "llm",
            "span_name": "m",
            "span_order": span_order,
        },
        "usage": {
            "shape": "openai_chat",
            "raw": {"prompt_tokens": 11, "completion_tokens": 1, "total_tokens": 12},
        },
    }


def _flush_capturing(session):
    logged = []

    class FakeClient:
        def log_sync(self, **kwargs):
            logged.append(kwargs)

    with mock.patch("token_police.enforcer.get_client", return_value=FakeClient()):
        _REAL_FLUSH(session)
    return logged


class TestFlushMerge:
    def test_flush_replaces_synth_raw_with_verbatim_usage(self):
        session = TPSession(trace_id="c" * 32)
        session._deferred_spans = [_openai_stream_payload(session, 0)]
        session._pending_compositions = {
            f"{session.trace_id}:0": {
                "response": [{"role": "assistant", "type": "text"}],
                "usage_raw": dict(XAI_USAGE),
            }
        }

        logged = _flush_capturing(session)

        assert len(logged) == 1
        raw = logged[0]["usage"]["raw"]
        assert raw == XAI_USAGE                              # verbatim, wholesale
        assert raw["completion_tokens_details"]["reasoning_tokens"] == 357
        assert raw["prompt_tokens_details"]["cached_tokens"] == 4
        assert raw["cost_in_usd_ticks"] == 42
        # the shape (which mapper the collector picks) must NOT change
        assert logged[0]["usage"]["shape"] == "openai_chat"

    def test_flush_without_usage_raw_leaves_synth_raw_byte_identical(self):
        session = TPSession(trace_id="e" * 32)
        payload = _openai_stream_payload(session, 0)
        original_raw = dict(payload["usage"]["raw"])
        session._deferred_spans = [payload]
        session._pending_compositions = {
            f"{session.trace_id}:0": {
                "response": [{"role": "assistant", "type": "text"}],
            }
        }

        logged = _flush_capturing(session)

        assert len(logged) == 1
        assert logged[0]["usage"]["raw"] == original_raw
        assert logged[0]["usage"]["shape"] == "openai_chat"
        assert "completion_tokens_details" not in logged[0]["usage"]["raw"]

    def test_flush_merges_onto_the_matching_span_only(self):
        """Two deferred spans, usage_raw reserved for span_order=1."""
        session = TPSession(trace_id="f" * 32)
        p0 = _openai_stream_payload(session, 0)
        p1 = _openai_stream_payload(session, 1)
        original_raw = dict(p0["usage"]["raw"])
        session._deferred_spans = [p0, p1]
        session._pending_compositions = {
            f"{session.trace_id}:1": {"usage_raw": dict(XAI_USAGE)},
        }

        logged = _flush_capturing(session)

        assert len(logged) == 2
        assert logged[0]["usage"]["raw"] == original_raw      # sibling untouched
        assert logged[1]["usage"]["raw"] == XAI_USAGE
