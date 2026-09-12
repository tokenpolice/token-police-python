"""G3-O1 (Fix A) — verbatim google-genai ``usage_metadata`` forwarding.

The OpenLLMetry google instrumentor only stamps flat prompt/completion token
attrs, so the Mode-A deferred payload's attribute-synthesized ``usage.raw``
dropped ``thoughts_token_count`` and ``cached_content_token_count``. Because
Google's ``candidates_token_count`` EXCLUDES thoughts (the collector's
google_genai mapper adds reasoning ON TOP of text output), every Gemini
thinking token was silently UNBILLED on the Python SDK — a systematic
UNDERCHARGE, and a mismatch with the Node SDK.

Fix under test, two halves:
  A1 non-stream: ``_stash_google_verbatim_usage`` (called unconditionally from
     ``_capture_response_composition`` for google/gemini) stashes the verbatim
     ``usage_metadata`` under the call's ``comp_key`` so the pre-existing
     ``_flush_deferred_spans`` merge replaces ``payload["usage"]["raw"]``.
  A2 stream: the Mode-A sync/async stream taps keep a LAST-WRITE-WINS reference
     to each chunk's ``usage_metadata`` (Google puts the COMPLETE usage on the
     terminal chunk — never accumulated, never de-cumulated),
     ``_serialize_google_stream_usage`` serializes it behind the same positivity
     guard, and ``_serialize_mode_a_stream_usage`` routes per provider. The
     forward is deliberately NOT gated on ``_mode_a_synthetic_response``
     returning non-None: a parts-less google turn yields no synthetic response
     but its terminal-chunk usage must still land.

Load-bearing invariants pinned here:
  * POSITIVITY GUARD — a zero/absent usage block must NEVER clobber the good
    OTel-synth counts, and it structurally excludes LangChain results (whose
    ``usage_metadata`` is keyed input_tokens/output_tokens, not
    prompt_token_count/candidates_token_count).
  * PRECEDENCE — an existing ``usage_raw`` stash (the stream wrappers'
    explicit ``usage_raw=`` forward, G3-14-1) is never overwritten.
  * GOLDEN RULE — hostile/None/non-serializable results never raise into
    customer code and never leave a partial stash.
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
# The verbatim Gemini usage payload the collector's google_genai mapper
# expects: thoughts + cached + tool-use prompt counts, no renaming/netting.
# ─────────────────────────────────────────────────────────────────────────
GOOGLE_USAGE = {
    "prompt_token_count": 1103,
    "candidates_token_count": 96,
    "thoughts_token_count": 512,
    "cached_content_token_count": 800,
    "tool_use_prompt_token_count": 40,
    "total_token_count": 1711,
}

# What the OTel synth produces today for the same call — no thoughts, no cache.
SYNTH_RAW = {
    "prompt_tokens": 1103,
    "completion_tokens": 96,
    "input_tokens": 1103,
    "output_tokens": 96,
}


def _google_response(usage_metadata, text="hi"):
    """A typed google-genai GenerateContentResponse-like object."""
    return NS(
        candidates=[NS(content=NS(parts=[NS(text=text, function_call=None)]))],
        usage_metadata=usage_metadata,
    )


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


def _content_chunk(text, usage_metadata=None):
    return NS(
        candidates=[NS(content=NS(parts=[NS(text=text, function_call=None)]))],
        usage_metadata=usage_metadata,
    )


def _partsless_chunk(usage_metadata=None):
    """A terminal google chunk with no candidates/parts (safety-blocked or
    usage-only turn). ``_mode_a_synthetic_response`` returns None for this
    accumulator — the usage forward must survive that."""
    return NS(candidates=[], usage_metadata=usage_metadata)


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
    (the flush POPS the comp_key). The merge tests drive the real flush."""
    monkeypatch.setattr(enforcer, "_flush_deferred_spans", MagicMock())


@pytest.fixture(autouse=True)
def _noop_failure_log(monkeypatch):
    monkeypatch.setattr(enforcer, "_emit_call_failure_log", MagicMock())


def _slot(session, order):
    return session._pending_compositions.get(f"{session.trace_id}:{order}", {})


# ─────────────────────────────────────────────────────────────────────────
# 0. The code under test really is the worktree copy.
# ─────────────────────────────────────────────────────────────────────────
def test_enforcer_module_is_this_worktree():
    assert Path(enforcer.__file__).resolve() == (
        ROOT / "token_police" / "enforcer.py").resolve()


# ─────────────────────────────────────────────────────────────────────────
# 1. A1 — _stash_google_verbatim_usage (non-stream)
# ─────────────────────────────────────────────────────────────────────────
class TestStashGoogleVerbatimUsage:
    def test_object_response_stashed_verbatim(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0, result=_google_response(NS(**GOOGLE_USAGE)))

        raw = _slot(sess, 0)["usage_raw"]
        # Every key survives — thoughts + cache are exactly what the synth drops.
        assert raw["thoughts_token_count"] == 512
        assert raw["cached_content_token_count"] == 800
        assert raw["tool_use_prompt_token_count"] == 40
        assert raw["prompt_token_count"] == 1103
        # candidates stays EXCLUSIVE of thoughts — never netted or summed here.
        assert raw["candidates_token_count"] == 96
        assert raw["candidates_token_count"] != 96 + 512

    def test_dict_response_stashed_verbatim(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0, result={"usage_metadata": dict(GOOGLE_USAGE)})
        assert _slot(sess, 0)["usage_raw"] == GOOGLE_USAGE

    def test_camel_usage_metadata_key(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0, result={"usageMetadata": dict(GOOGLE_USAGE)})
        assert _slot(sess, 0)["usage_raw"] == GOOGLE_USAGE

    def test_gemini_provider_alias_accepted(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "Gemini", 0, result=_google_response(NS(**GOOGLE_USAGE)))
        assert _slot(sess, 0)["usage_raw"]["thoughts_token_count"] == 512

    def test_existing_usage_raw_is_never_overwritten(self, sess):
        """The stream wrappers' explicit ``usage_raw=`` forward (G3-14-1) has
        precedence — this stash must not clobber it."""
        sess._pending_compositions[f"{sess.trace_id}:0"] = {
            "usage_raw": {"prompt_token_count": 1, "candidates_token_count": 2}}

        enforcer._stash_google_verbatim_usage(
            sess, "google", 0, result=_google_response(NS(**GOOGLE_USAGE)))

        assert _slot(sess, 0)["usage_raw"] == {
            "prompt_token_count": 1, "candidates_token_count": 2}

    # ── POSITIVITY GUARD: a zero block never clobbers good synth counts ──
    def test_all_zero_usage_metadata_never_stashes(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0,
            result=_google_response({"prompt_token_count": 0,
                                     "candidates_token_count": 0,
                                     "total_token_count": 0}))
        assert "usage_raw" not in _slot(sess, 0)

    def test_thoughts_only_block_without_positive_counts_skipped(self, sess):
        # A degenerate block carrying ONLY thoughts is not trustworthy enough to
        # replace the synth raw wholesale (the merge is a REPLACE, not a patch).
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0,
            result=_google_response({"thoughts_token_count": 512}))
        assert "usage_raw" not in _slot(sess, 0)

    def test_only_prompt_positive_is_enough(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0,
            result=_google_response({"prompt_token_count": 7,
                                     "candidates_token_count": 0}))
        assert _slot(sess, 0)["usage_raw"] == {"prompt_token_count": 7,
                                               "candidates_token_count": 0}

    def test_only_candidates_positive_is_enough(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0,
            result=_google_response({"candidates_token_count": 5}))
        assert _slot(sess, 0)["usage_raw"] == {"candidates_token_count": 5}

    def test_absent_usage_metadata_no_stash(self, sess):
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0,
            result=NS(candidates=[], usage_metadata=None))
        assert "usage_raw" not in _slot(sess, 0)

    # ── LangChain results are structurally excluded by the same guard ──
    @pytest.mark.parametrize("lc_result", [
        # AIMessage-shaped: LC-standard usage_metadata keys only.
        NS(usage_metadata={"input_tokens": 1103, "output_tokens": 96,
                           "total_tokens": 1199,
                           "output_token_details": {"reasoning": 512}}),
        # dict form of the same
        {"usage_metadata": {"input_tokens": 10, "output_tokens": 2}},
    ])
    def test_langchain_usage_metadata_never_taken(self, sess, lc_result):
        # Even if a caller mislabels an LC result as provider "google", the
        # positivity guard (which only knows google's native key names) refuses.
        enforcer._stash_google_verbatim_usage(sess, "google", 0, result=lc_result)
        assert "usage_raw" not in _slot(sess, 0)

    # ── provider gate ──
    @pytest.mark.parametrize("provider", ["openai", "anthropic", "xai", "", None,
                                          "langchain", "vertexai"])
    def test_non_google_providers_never_stash(self, sess, provider):
        enforcer._stash_google_verbatim_usage(
            sess, provider, 0, result=_google_response(NS(**GOOGLE_USAGE)))
        assert "usage_raw" not in _slot(sess, 0)

    def test_none_result_creates_no_slot(self, sess):
        enforcer._stash_google_verbatim_usage(sess, "google", 0, result=None)
        assert f"{sess.trace_id}:0" not in sess._pending_compositions

    def test_none_session_never_raises(self):
        assert enforcer._stash_google_verbatim_usage(
            None, "google", 0, result=_google_response(NS(**GOOGLE_USAGE))) is None

    def test_order_none_falls_back_to_span_counter(self, sess):
        sess._span_counter = 4
        enforcer._stash_google_verbatim_usage(
            sess, "google", None, result=_google_response(NS(**GOOGLE_USAGE)))
        assert _slot(sess, 4)["usage_raw"]["thoughts_token_count"] == 512


# ── GOLDEN RULE: nothing at this seam may raise into customer code ──
class TestStashGoldenRule:
    def test_hostile_usage_metadata_property_never_raises(self, sess):
        class Boom:
            @property
            def usage_metadata(self):
                raise RuntimeError("boom")

            @property
            def usageMetadata(self):  # noqa: N802 - mirrors the SDK camel alias
                raise RuntimeError("boom")

        enforcer._stash_google_verbatim_usage(sess, "google", 0, result=Boom())
        assert "usage_raw" not in _slot(sess, 0)

    def test_non_serializable_usage_never_raises(self, sess):
        class Unserializable:
            __slots__ = ()          # no __dict__ → the json default falls to str

            def __repr__(self):
                raise RuntimeError("repr exploded")

        enforcer._stash_google_verbatim_usage(
            sess, "google", 0, result=NS(usage_metadata=Unserializable()))
        assert "usage_raw" not in _slot(sess, 0)

    def test_non_numeric_counts_fail_closed_to_no_stash(self, sess):
        # int("abc") raises inside the guard → swallowed → no stash, no raise.
        enforcer._stash_google_verbatim_usage(
            sess, "google", 0,
            result=_google_response({"prompt_token_count": "abc",
                                     "candidates_token_count": "xyz"}))
        assert "usage_raw" not in _slot(sess, 0)

    @pytest.mark.parametrize("weird", [42, "nope", [1, 2], True, object()])
    def test_weird_result_objects_never_raise(self, sess, weird):
        enforcer._stash_google_verbatim_usage(sess, "google", 0, result=weird)
        assert "usage_raw" not in _slot(sess, 0)

    def test_hostile_session_never_raises(self):
        """@fail_safe is the only thing standing between a broken session
        object and the customer's call."""
        class BoomSession:
            @property
            def trace_id(self):
                raise RuntimeError("boom")

        assert enforcer._stash_google_verbatim_usage(
            BoomSession(), "google", 0,
            result=_google_response(NS(**GOOGLE_USAGE))) is None


# ─────────────────────────────────────────────────────────────────────────
# 2. A1 wired through _capture_response_composition (the real call site)
# ─────────────────────────────────────────────────────────────────────────
class TestCaptureResponseCompositionWiring:
    def test_google_non_stream_capture_stashes_usage_raw(self, sess):
        enforcer._capture_response_composition(
            "google", _google_response(NS(**GOOGLE_USAGE)), order=2)

        slot = _slot(sess, 2)
        assert slot["usage_raw"]["thoughts_token_count"] == 512
        assert slot["usage_raw"]["cached_content_token_count"] == 800
        assert slot.get("response")          # composition still captured

    def test_explicit_usage_raw_argument_wins(self, sess):
        explicit = {"prompt_token_count": 9, "candidates_token_count": 9}
        enforcer._capture_response_composition(
            "google", _google_response(NS(**GOOGLE_USAGE)), order=2,
            usage_raw=explicit)
        assert _slot(sess, 2)["usage_raw"] == explicit

    def test_openai_capture_unaffected(self, sess):
        enforcer._capture_response_composition(
            "openai",
            NS(choices=[NS(message=NS(content="hi", tool_calls=None))]),
            order=1)
        assert "usage_raw" not in _slot(sess, 1)

    def test_response_none_builds_no_composition(self, sess):
        """G3-O1 changed ``_capture_response_composition`` to skip composition
        building when response is None — it must never fabricate a Tier-3
        complete_response entry for a stream that produced no parts."""
        enforcer._capture_response_composition(
            "google", None, order=3, usage_raw=dict(GOOGLE_USAGE))

        slot = _slot(sess, 3)
        assert "response" not in slot
        assert slot["usage_raw"] == GOOGLE_USAGE

    def test_response_none_without_usage_raw_never_raises(self, sess):
        enforcer._capture_response_composition("google", None, order=3)
        assert "response" not in _slot(sess, 3)


# ─────────────────────────────────────────────────────────────────────────
# 3. A2 — the stream serializers
# ─────────────────────────────────────────────────────────────────────────
class TestSerializeGoogleStreamUsage:
    def test_verbatim_pass_through(self):
        assert enforcer._serialize_google_stream_usage(
            dict(GOOGLE_USAGE)) == GOOGLE_USAGE

    def test_object_with_dunder_dict(self):
        out = enforcer._serialize_google_stream_usage(NS(**GOOGLE_USAGE))
        assert out["thoughts_token_count"] == 512
        assert out["cached_content_token_count"] == 800

    def test_nested_modality_details_survive(self):
        src = {
            "prompt_token_count": 10,
            "candidates_token_count": 4,
            "candidates_tokens_details": [{"modality": "TEXT", "token_count": 4}],
        }
        assert enforcer._serialize_google_stream_usage(src) == src

    def test_zero_counts_is_none(self):
        assert enforcer._serialize_google_stream_usage(
            {"prompt_token_count": 0, "candidates_token_count": 0}) is None

    def test_thoughts_only_is_none(self):
        assert enforcer._serialize_google_stream_usage(
            {"thoughts_token_count": 500}) is None

    def test_empty_dict_and_none(self):
        assert enforcer._serialize_google_stream_usage({}) is None
        assert enforcer._serialize_google_stream_usage(None) is None

    @pytest.mark.parametrize("garbage", ["nope", 42, 3.5, True, [1, 2, 3], object()])
    def test_non_dict_garbage_is_none(self, garbage):
        assert enforcer._serialize_google_stream_usage(garbage) is None

    def test_openai_shaped_usage_rejected_by_the_positivity_guard(self):
        # An openai-wire block has neither google key → guard refuses it, so a
        # provider mislabel can never forward the wrong shape.
        assert enforcer._serialize_google_stream_usage(
            {"prompt_tokens": 11, "completion_tokens": 358}) is None

    def test_langchain_shaped_usage_rejected(self):
        assert enforcer._serialize_google_stream_usage(
            {"input_tokens": 11, "output_tokens": 358}) is None

    def test_hostile_object_returns_none_never_raises(self):
        class Boom:
            def __getattr__(self, name):
                raise RuntimeError("boom")

            def __repr__(self):
                raise RuntimeError("boom repr")

        assert enforcer._serialize_google_stream_usage(Boom()) is None

    def test_non_numeric_counts_fail_closed_to_none(self):
        assert enforcer._serialize_google_stream_usage(
            {"prompt_token_count": "abc", "candidates_token_count": "xyz"}) is None


class TestSerializeModeAStreamUsageRouter:
    def test_openai_routes_to_openai_serializer(self):
        src = {"prompt_tokens": 11, "completion_tokens": 1,
               "completion_tokens_details": {"reasoning_tokens": 357}}
        assert enforcer._serialize_mode_a_stream_usage("openai", src) == src

    @pytest.mark.parametrize("provider", ["google", "gemini"])
    def test_google_routes_to_google_serializer(self, provider):
        assert enforcer._serialize_mode_a_stream_usage(
            provider, dict(GOOGLE_USAGE)) == GOOGLE_USAGE

    def test_google_block_under_openai_provider_is_rejected(self):
        # Cross-routing must not smuggle a google block through the openai
        # serializer (its positivity guard reads prompt_tokens/completion_tokens).
        assert enforcer._serialize_mode_a_stream_usage(
            "openai", dict(GOOGLE_USAGE)) is None

    def test_openai_block_under_google_provider_is_rejected(self):
        assert enforcer._serialize_mode_a_stream_usage(
            "google", {"prompt_tokens": 11, "completion_tokens": 1}) is None

    @pytest.mark.parametrize("provider", ["anthropic", "xai", "cohere", "bedrock",
                                          "langchain", "", None])
    def test_other_providers_are_none(self, provider):
        assert enforcer._serialize_mode_a_stream_usage(
            provider, dict(GOOGLE_USAGE)) is None

    def test_none_usage_is_none(self):
        for provider in ("openai", "google", "gemini", "anthropic"):
            assert enforcer._serialize_mode_a_stream_usage(provider, None) is None

    def test_hostile_usage_never_raises(self):
        class Boom:
            def __getattr__(self, name):
                raise RuntimeError("boom")

        for provider in ("openai", "google"):
            assert enforcer._serialize_mode_a_stream_usage(provider, Boom()) is None


# ─────────────────────────────────────────────────────────────────────────
# 4. A2 — stream wrapper → stash (sync + async)
# ─────────────────────────────────────────────────────────────────────────
class TestGoogleStreamStashSync:
    def test_terminal_chunk_usage_metadata_stashed_verbatim(self, sess):
        chunks = [_content_chunk("he"), _content_chunk("llo"),
                  _partsless_chunk(dict(GOOGLE_USAGE))]
        out = list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=3))

        assert out == chunks                       # customer sequence unchanged
        assert _slot(sess, 3)["usage_raw"] == GOOGLE_USAGE
        assert f"{sess.trace_id}:-1" not in sess._pending_compositions

    def test_usage_on_last_content_chunk_captured(self, sess):
        chunks = [_content_chunk("he"), _content_chunk("llo", dict(GOOGLE_USAGE))]
        list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0))

        assert _slot(sess, 0)["usage_raw"] == GOOGLE_USAGE
        assert _slot(sess, 0).get("response")      # composition still built

    def test_last_write_wins_never_accumulates(self, sess):
        """Gemini restates a CUMULATIVE usage block on every chunk. Summing or
        de-cumulating would corrupt the counts — the terminal block is final."""
        interim = {"prompt_token_count": 1103, "candidates_token_count": 40,
                   "thoughts_token_count": 200}
        chunks = [_content_chunk("a", interim),
                  _content_chunk("b", dict(GOOGLE_USAGE))]
        list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0))

        raw = _slot(sess, 0)["usage_raw"]
        assert raw == GOOGLE_USAGE
        assert raw["thoughts_token_count"] == 512       # NOT 200, NOT 712
        assert raw["candidates_token_count"] == 96      # NOT 136

    def test_partsless_stream_still_forwards_usage(self, sess):
        """CRITICAL: ``_mode_a_synthetic_response`` returns None when no parts
        were accumulated (safety-blocked / usage-only turn). Pre-fix the usage
        forward was nested under that result and the whole block was lost."""
        chunks = [_partsless_chunk(dict(GOOGLE_USAGE))]
        out = list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0))

        assert out == chunks
        slot = _slot(sess, 0)
        assert slot["usage_raw"] == GOOGLE_USAGE
        # ...and no fabricated composition for a response that had no parts.
        assert "response" not in slot

    def test_dict_chunk_usage_metadata_fallback(self, sess):
        chunks = [{"candidates": [], "usage_metadata": dict(GOOGLE_USAGE)}]
        list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0))
        assert _slot(sess, 0)["usage_raw"] == GOOGLE_USAGE

    def test_dict_chunk_camel_usage_metadata_fallback(self, sess):
        chunks = [{"candidates": [], "usageMetadata": dict(GOOGLE_USAGE)}]
        list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0))
        assert _slot(sess, 0)["usage_raw"] == GOOGLE_USAGE

    def test_zero_usage_leaves_no_stash(self, sess):
        chunks = [_content_chunk("hi"),
                  _partsless_chunk({"prompt_token_count": 0,
                                    "candidates_token_count": 0})]
        list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0))
        assert "usage_raw" not in _slot(sess, 0)

    def test_absent_usage_leaves_no_stash(self, sess):
        chunks = [_content_chunk("hi"), _content_chunk(" there")]
        list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0))
        assert "usage_raw" not in _slot(sess, 0)
        assert _slot(sess, 0).get("response")

    def test_anthropic_stream_still_never_stashes(self, sess):
        """The google branch must not widen the gate for other providers."""
        usage = {"input_tokens": 11, "output_tokens": 358}
        chunks = [
            NS(type="content_block_start", index=0,
               content_block=NS(type="text"), usage_metadata=usage),
            NS(type="content_block_delta", index=0,
               delta=NS(type="text_delta", text="hi"), usage_metadata=usage),
        ]
        out = list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream(chunks), "anthropic", sess, False, order=0))

        assert out == chunks
        assert "usage_raw" not in _slot(sess, 0)


class TestGoogleStreamStashAsync:
    def test_terminal_chunk_usage_metadata_stashed_verbatim(self, sess):
        chunks = [_content_chunk("he"), _partsless_chunk(dict(GOOGLE_USAGE))]
        out = _run(_drain_async(enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "google", sess, False, order=3)))

        assert out == chunks
        assert _slot(sess, 3)["usage_raw"] == GOOGLE_USAGE

    def test_last_write_wins_never_accumulates(self, sess):
        interim = {"prompt_token_count": 1103, "candidates_token_count": 40,
                   "thoughts_token_count": 200}
        chunks = [_content_chunk("a", interim),
                  _content_chunk("b", dict(GOOGLE_USAGE))]
        _run(_drain_async(enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "google", sess, False, order=0)))

        assert _slot(sess, 0)["usage_raw"]["thoughts_token_count"] == 512

    def test_partsless_stream_still_forwards_usage(self, sess):
        chunks = [_partsless_chunk(dict(GOOGLE_USAGE))]
        _run(_drain_async(enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream(chunks), "google", sess, False, order=0)))

        slot = _slot(sess, 0)
        assert slot["usage_raw"] == GOOGLE_USAGE
        assert "response" not in slot

    def test_zero_usage_leaves_no_stash(self, sess):
        _run(_drain_async(enforcer._wrap_mode_a_async_stream(
            FakeAsyncStream([_partsless_chunk({"prompt_token_count": 0})]),
            "google", sess, False, order=0)))
        assert "usage_raw" not in _slot(sess, 0)

    def test_sync_underlying_iterator_bridged(self, sess):
        chunks = [_content_chunk("hi"), _partsless_chunk(dict(GOOGLE_USAGE))]
        out = _run(_drain_async(enforcer._wrap_mode_a_async_stream(
            FakeSyncStream(chunks), "google", sess, False, order=0)))

        assert out == chunks
        assert _slot(sess, 0)["usage_raw"] == GOOGLE_USAGE


# ── GOLDEN RULE at the stream seam ──
class TestGoogleStreamGoldenRule:
    def test_hostile_usage_metadata_attribute_never_breaks_iteration(self, sess):
        class HostileChunk:
            candidates = []

            @property
            def usage_metadata(self):
                raise RuntimeError("usage_metadata exploded")

        hostile = HostileChunk()
        out = list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream([_content_chunk("hi"), hostile]), "google", sess,
            False, order=0))

        assert out[1] is hostile                   # delivered unchanged
        assert "usage_raw" not in _slot(sess, 0)

    def test_mid_stream_error_reraised_verbatim_no_stash(self, sess):
        sentinel = ValueError("provider dropped the connection")
        chunks = [_content_chunk("hi", dict(GOOGLE_USAGE))]
        wrapped = enforcer._wrap_mode_a_sync_stream(
            RaisingSyncStream(chunks, sentinel), "google", sess, False, order=0)

        seen = []
        with pytest.raises(ValueError) as ei:
            for c in wrapped:
                seen.append(c)

        assert ei.value is sentinel                # identity, not wrapped
        assert seen == chunks
        assert "usage_raw" not in _slot(sess, 0)

    def test_garbage_usage_metadata_value_leaves_no_stash(self, sess):
        out = list(enforcer._wrap_mode_a_sync_stream(
            FakeSyncStream([_partsless_chunk("not-a-usage-object")]),
            "google", sess, False, order=0))

        assert len(out) == 1
        assert "usage_raw" not in _slot(sess, 0)


# ─────────────────────────────────────────────────────────────────────────
# 5. End-to-end merge in _flush_deferred_spans
# ─────────────────────────────────────────────────────────────────────────
def _google_payload(session, span_order=0, raw=None):
    """A Mode-A deferred payload as on_end builds it for a google call: an
    attribute-synthesized usage.raw with NO thoughts/cache keys."""
    return {
        "user_id": "u",
        "paid_plan": "free",
        "workflow_name": "w",
        "session_id": "",
        "model": "gemini-2.5-pro",
        "provider": "google",
        "input_tokens": 1103,
        "output_tokens": 96,
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
        "usage": {"shape": "google_genai",
                  "raw": dict(SYNTH_RAW) if raw is None else raw},
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
    def test_verbatim_usage_replaces_synth_raw(self):
        session = TPSession(trace_id="c" * 32)
        session._deferred_spans = [_google_payload(session, 0)]
        session._pending_compositions = {
            f"{session.trace_id}:0": {"usage_raw": dict(GOOGLE_USAGE)}}

        logged = _flush_capturing(session)

        assert len(logged) == 1
        raw = logged[0]["usage"]["raw"]
        assert raw == GOOGLE_USAGE
        # These are the two keys the OTel synth drops — the whole point.
        assert raw["thoughts_token_count"] == 512
        assert raw["cached_content_token_count"] == 800
        # Shape (i.e. which collector mapper runs) must NOT change.
        assert logged[0]["usage"]["shape"] == "google_genai"

    def test_no_stash_leaves_synth_raw_byte_identical(self):
        session = TPSession(trace_id="e" * 32)
        payload = _google_payload(session, 0)
        original = dict(payload["usage"]["raw"])
        session._deferred_spans = [payload]
        session._pending_compositions = {f"{session.trace_id}:0": {}}

        logged = _flush_capturing(session)

        assert logged[0]["usage"]["raw"] == original
        assert "thoughts_token_count" not in logged[0]["usage"]["raw"]

    def test_merge_hits_only_the_matching_span(self):
        session = TPSession(trace_id="f" * 32)
        p0 = _google_payload(session, 0)
        p1 = _google_payload(session, 1)
        original = dict(p0["usage"]["raw"])
        session._deferred_spans = [p0, p1]
        session._pending_compositions = {
            f"{session.trace_id}:1": {"usage_raw": dict(GOOGLE_USAGE)}}

        logged = _flush_capturing(session)

        assert logged[0]["usage"]["raw"] == original     # sibling untouched
        assert logged[1]["usage"]["raw"] == GOOGLE_USAGE


# ─────────────────────────────────────────────────────────────────────────
# 6. Full non-stream path: capture → flush (the customer-visible outcome)
# ─────────────────────────────────────────────────────────────────────────
def test_non_stream_capture_then_flush_forwards_thoughts(sess):
    sess._deferred_spans = [_google_payload(sess, 0)]

    enforcer._capture_response_composition(
        "google", _google_response(NS(**GOOGLE_USAGE)), order=0)
    logged = _flush_capturing(sess)

    assert len(logged) == 1
    raw = logged[0]["usage"]["raw"]
    assert raw["thoughts_token_count"] == 512
    assert raw["candidates_token_count"] == 96
    assert raw["cached_content_token_count"] == 800


def test_non_stream_zero_usage_leaves_synth_raw_untouched(sess):
    payload = _google_payload(sess, 0)
    original = dict(payload["usage"]["raw"])
    sess._deferred_spans = [payload]

    enforcer._capture_response_composition(
        "google",
        _google_response({"prompt_token_count": 0, "candidates_token_count": 0}),
        order=0)
    logged = _flush_capturing(sess)

    assert logged[0]["usage"]["raw"] == original
