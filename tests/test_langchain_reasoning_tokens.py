"""G3-O1 (Fix D) — LangChain / LangGraph reasoning tokens (Python).

The OpenLLMetry LangChain instrumentor never sets
``gen_ai.usage.reasoning_tokens``, and the LC wrappers extracted no usage of
their own, so the Mode-A attribute-synthesized ``usage.raw`` for every LC /
LangGraph call landed with NO reasoning breakdown: o-series / gpt-5 and Gemini
thinking traffic through LangChain reported ``reasoning_output_tokens = 0``.

The data IS available — LangChain standardizes it at
``AIMessage.usage_metadata["output_token_details"]["reasoning"]`` (both
``langchain_openai`` and ``langchain_google_genai``).

Fix under test:
  D1 ``_lc_reasoning_from_llmresult`` reads it off the LLMResult in the
     lc_sync / lc_async generate path (FIRST usage-bearing candidate per outer
     generations list — n>1 candidates carry the same duplicated full-call
     usage — SUMMED across outer lists, one per provider call).
  D2 ``_finalize_langchain_stream`` reads it off the concat'd AIMessageChunk
     (LC's concat has already summed it; we must NOT re-derive).
  D3 ``_stash_lc_reasoning_tokens`` files the scalar on the pending-composition
     slot, and ``_flush_deferred_spans`` merges it into the payload's raw under
     the SHAPE'S NATIVE KEY:
       * openai_chat / openai_compatible_chat → ``completion_tokens_details``
         (SUBSET semantics: mapper does text = completion − reasoning);
       * google_genai → ``thoughts_token_count`` + a recomputed
         ``candidates_token_count = completion − reasoning`` (ADDITIVE
         semantics: the google mapper treats candidates as EXCLUSIVE of
         thoughts, so emitting only thoughts would INFLATE total output).

Load-bearing invariants pinned here:
  * TOTAL OUTPUT NEVER MOVES on either shape.
  * A real provider usage block already carrying google keys (or a nonzero
    reasoning value) is never overwritten.
  * Reasoning is clamped to ``completion_tokens`` — an over-summed concat can
    never bill more reasoning than output.
  * Shapes we don't understand are left completely alone.
  * GOLDEN RULE — every new read is guarded; nothing raises into customer code.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock
from unittest.mock import MagicMock

import pytest

# Make the local (worktree) SDK importable ahead of any editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import TPSession, _current_session, _in_langchain

_REAL_FLUSH = enforcer._flush_deferred_spans


# ─────────────────────────────────────────────────────────────────────────
# Fakes — LangChain shapes, no langchain installed.
# ─────────────────────────────────────────────────────────────────────────
def _um(input_tokens=100, output_tokens=400, reasoning=None):
    """A LangChain-standard ``usage_metadata`` dict."""
    um = {"input_tokens": input_tokens, "output_tokens": output_tokens,
          "total_tokens": input_tokens + output_tokens}
    if reasoning is not None:
        um["output_token_details"] = {"reasoning": reasoning}
    return um


def _gen(usage_metadata, text="hi"):
    """A LangChain Generation carrying an AIMessage with usage_metadata."""
    return NS(text=text, message=NS(content=text, usage_metadata=usage_metadata))


def _llm_result(generations):
    return NS(generations=generations, llm_output={})


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture()
def sess():
    s = TPSession(user_id="u", paid_plan="pro", workflow_name="wf",
                  trace_id="a" * 32, root_span_id="b" * 16)
    s._span_counter = 0
    s._deferred_spans = []
    s._defer_telemetry = False
    s._pending_compositions = {}
    s._mode_a_prompt_order = None
    tok = _current_session.set(s)
    guard = _in_langchain.set(False)
    try:
        yield s
    finally:
        try:
            _in_langchain.reset(guard)
        except Exception:
            _in_langchain.set(False)
        _current_session.reset(tok)


def _slot(session, order):
    return session._pending_compositions.get(f"{session.trace_id}:{order}", {})


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────────────────
# 0. The code under test really is the worktree copy.
# ─────────────────────────────────────────────────────────────────────────
def test_enforcer_module_is_this_worktree():
    assert Path(enforcer.__file__).resolve() == (
        ROOT / "token_police" / "enforcer.py").resolve()


# ─────────────────────────────────────────────────────────────────────────
# 1. _lc_reasoning_from_usage_metadata
# ─────────────────────────────────────────────────────────────────────────
class TestReasoningFromUsageMetadata:
    def test_reads_lc_standard_key(self):
        assert enforcer._lc_reasoning_from_usage_metadata(
            _um(reasoning=357)) == 357

    def test_missing_output_token_details_is_zero(self):
        assert enforcer._lc_reasoning_from_usage_metadata(_um()) == 0

    def test_zero_and_none_are_zero(self):
        assert enforcer._lc_reasoning_from_usage_metadata(_um(reasoning=0)) == 0
        assert enforcer._lc_reasoning_from_usage_metadata(
            {"output_token_details": {"reasoning": None}}) == 0

    def test_negative_clamped_to_zero(self):
        assert enforcer._lc_reasoning_from_usage_metadata(
            {"output_token_details": {"reasoning": -9}}) == 0

    def test_sibling_detail_keys_ignored(self):
        # LC also reports audio / accepted_prediction under the same object.
        assert enforcer._lc_reasoning_from_usage_metadata(
            {"output_token_details": {"audio": 40}}) == 0

    @pytest.mark.parametrize("um", [None, "nope", 42, [1, 2], object(),
                                    {"output_token_details": "nope"},
                                    {"output_token_details": None},
                                    {"output_token_details": {"reasoning": "abc"}},
                                    {"output_token_details": {"reasoning": object()}}])
    def test_junk_is_zero_never_raises(self, um):
        assert enforcer._lc_reasoning_from_usage_metadata(um) == 0

    def test_hostile_mapping_is_zero(self):
        class _Hostile(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom")

        assert enforcer._lc_reasoning_from_usage_metadata(_Hostile()) == 0


# ─────────────────────────────────────────────────────────────────────────
# 2. _lc_reasoning_from_llmresult (D1)
# ─────────────────────────────────────────────────────────────────────────
class TestReasoningFromLLMResult:
    def test_single_call_single_candidate(self):
        assert enforcer._lc_reasoning_from_llmresult(
            _llm_result([[_gen(_um(reasoning=357))]])) == 357

    def test_n_gt_1_candidates_counted_once(self):
        """LC stamps the SAME full-call usage_metadata on every candidate —
        summing within a list would double-bill an n=2 call."""
        um = _um(reasoning=357)
        assert enforcer._lc_reasoning_from_llmresult(
            _llm_result([[_gen(um, "a"), _gen(um, "b"), _gen(um, "c")]])) == 357

    def test_summed_across_outer_lists(self):
        """One outer list per provider call in a batch generate."""
        assert enforcer._lc_reasoning_from_llmresult(_llm_result([
            [_gen(_um(reasoning=10))],
            [_gen(_um(reasoning=20))],
        ])) == 30

    def test_first_usage_bearing_candidate_per_list(self):
        assert enforcer._lc_reasoning_from_llmresult(_llm_result([
            # first candidate has no usage → the second is read
            [_gen(None, "a"), _gen(_um(reasoning=10), "b")],
            [_gen(_um(reasoning=20), "c"), _gen(_um(reasoning=20), "d")],
        ])) == 30

    def test_bare_generation_inner_entry_normalized(self):
        assert enforcer._lc_reasoning_from_llmresult(
            _llm_result([_gen(_um(reasoning=5))])) == 5

    def test_no_reasoning_anywhere_is_zero(self):
        assert enforcer._lc_reasoning_from_llmresult(
            _llm_result([[_gen(_um())]])) == 0

    def test_candidate_without_message_skipped(self):
        assert enforcer._lc_reasoning_from_llmresult(
            _llm_result([[NS(text="hi"), _gen(_um(reasoning=7))]])) == 7

    @pytest.mark.parametrize("result", [
        None, 42, "nope",
        NS(generations=None),
        NS(generations="nope"),
        NS(generations=[]),
        NS(generations=[[]]),
        NS(generations=[[None, None]]),
        NS(generations=[[NS(message=None)]]),
        NS(generations=[[NS(message=NS(usage_metadata="lots"))]]),
    ])
    def test_junk_results_are_zero_never_raise(self, result):
        assert enforcer._lc_reasoning_from_llmresult(result) == 0

    def test_raising_generations_property_is_zero(self):
        class Boom:
            @property
            def generations(self):
                raise RuntimeError("boom")

        assert enforcer._lc_reasoning_from_llmresult(Boom()) == 0

    def test_raising_usage_metadata_property_is_zero(self):
        class BoomMsg:
            @property
            def usage_metadata(self):
                raise RuntimeError("boom")

        assert enforcer._lc_reasoning_from_llmresult(
            NS(generations=[[NS(message=BoomMsg())]])) == 0


# ─────────────────────────────────────────────────────────────────────────
# 3. _stash_lc_reasoning_tokens
# ─────────────────────────────────────────────────────────────────────────
class TestStashLcReasoningTokens:
    def test_explicit_order(self, sess):
        enforcer._stash_lc_reasoning_tokens(sess, 357, order=4)
        assert _slot(sess, 4)["lc_reasoning_tokens"] == 357

    def test_mailbox_order_fallback(self, sess):
        """Must key onto the SAME slot ``_capture_response_composition`` uses —
        hence the mailbox-mirroring fallback, read BEFORE the capture clears it."""
        sess._mode_a_prompt_order = 2
        sess._span_counter = 9          # deliberately different
        enforcer._stash_lc_reasoning_tokens(sess, 357)
        assert _slot(sess, 2)["lc_reasoning_tokens"] == 357
        assert "lc_reasoning_tokens" not in _slot(sess, 8)

    def test_span_counter_fallback_when_mailbox_empty(self, sess):
        sess._mode_a_prompt_order = None
        sess._span_counter = 3
        enforcer._stash_lc_reasoning_tokens(sess, 357)
        assert _slot(sess, 2)["lc_reasoning_tokens"] == 357

    def test_does_not_disturb_other_slot_keys(self, sess):
        sess._pending_compositions[f"{sess.trace_id}:0"] = {
            "prompt": [{"role": "user"}], "usage_raw": {"prompt_tokens": 1}}
        enforcer._stash_lc_reasoning_tokens(sess, 42, order=0)
        slot = _slot(sess, 0)
        assert slot["lc_reasoning_tokens"] == 42
        assert slot["prompt"] == [{"role": "user"}]
        assert slot["usage_raw"] == {"prompt_tokens": 1}

    @pytest.mark.parametrize("reasoning", [0, None, -5, False])
    def test_non_positive_never_stashes(self, sess, reasoning):
        enforcer._stash_lc_reasoning_tokens(sess, reasoning, order=0)
        assert "lc_reasoning_tokens" not in _slot(sess, 0)

    def test_none_session_never_raises(self):
        assert enforcer._stash_lc_reasoning_tokens(None, 357, order=0) is None

    @pytest.mark.parametrize("reasoning", ["abc", object(), [1]])
    def test_junk_reasoning_never_raises(self, sess, reasoning):
        assert enforcer._stash_lc_reasoning_tokens(
            sess, reasoning, order=0) is None
        assert "lc_reasoning_tokens" not in _slot(sess, 0)

    def test_hostile_session_never_raises(self):
        class BoomSession:
            @property
            def trace_id(self):
                raise RuntimeError("boom")

        assert enforcer._stash_lc_reasoning_tokens(
            BoomSession(), 357, order=0) is None


# ─────────────────────────────────────────────────────────────────────────
# 4. _flush_deferred_spans merge (D3) — the customer-visible outcome
# ─────────────────────────────────────────────────────────────────────────
def _payload(session, shape, raw, span_order=0):
    """A Mode-A deferred payload as on_end builds it for an LC span."""
    return {
        "user_id": "u",
        "paid_plan": "pro",
        "workflow_name": "wf",
        "session_id": "",
        "model": "m",
        "provider": "langchain",
        "input_tokens": int(raw.get("prompt_tokens") or 0),
        "output_tokens": int(raw.get("completion_tokens") or 0),
        "cached_tokens": 0,
        "metadata": {},
        "span": {
            "trace_id": session.trace_id,
            "span_id": "d" * 16,
            "parent_span_id": "",
            "span_kind": "llm",
            "span_name": "langchain.chat",
            "span_order": span_order,
        },
        "usage": {"shape": shape, "raw": raw},
    }


def _flush_capturing(session):
    client = _FakeClient()
    with mock.patch("token_police.enforcer.get_client", return_value=client):
        _REAL_FLUSH(session)
    return client.calls


def _openai_raw(prompt=100, completion=400):
    return {"prompt_tokens": prompt, "completion_tokens": completion,
            "input_tokens": prompt, "output_tokens": completion}


class TestFlushMergeOpenAIShapes:
    @pytest.mark.parametrize("shape", ["openai_chat", "openai_compatible_chat"])
    def test_reasoning_injected_as_subset(self, sess, shape):
        sess._deferred_spans = [_payload(sess, shape, _openai_raw())]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]

        assert raw["completion_tokens_details"]["reasoning_tokens"] == 357
        # SUBSET: completion is unchanged; the mapper derives text = 400 − 357.
        assert raw["completion_tokens"] == 400
        assert raw["prompt_tokens"] == 100

    def test_clamped_to_completion_tokens(self, sess):
        sess._deferred_spans = [
            _payload(sess, "openai_chat", _openai_raw(completion=400))]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 9000}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]

        assert raw["completion_tokens_details"]["reasoning_tokens"] == 400
        # text_output = completion − reasoning must never go negative.
        assert (raw["completion_tokens"]
                - raw["completion_tokens_details"]["reasoning_tokens"]) >= 0

    def test_existing_nonzero_reasoning_never_overwritten(self, sess):
        raw_in = _openai_raw()
        raw_in["completion_tokens_details"] = {"reasoning_tokens": 111}
        sess._deferred_spans = [_payload(sess, "openai_chat", raw_in)]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]
        assert raw["completion_tokens_details"]["reasoning_tokens"] == 111

    def test_existing_zero_reasoning_is_filled(self, sess):
        raw_in = _openai_raw()
        raw_in["completion_tokens_details"] = {"reasoning_tokens": 0,
                                               "audio_tokens": 5}
        sess._deferred_spans = [_payload(sess, "openai_chat", raw_in)]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]
        assert raw["completion_tokens_details"]["reasoning_tokens"] == 357
        assert raw["completion_tokens_details"]["audio_tokens"] == 5

    def test_zero_completion_tokens_injects_nothing(self, sess):
        sess._deferred_spans = [
            _payload(sess, "openai_chat", _openai_raw(completion=0))]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]
        assert "completion_tokens_details" not in raw


class TestFlushMergeGoogleShape:
    def test_thoughts_split_keeps_total_output_identical(self, sess):
        raw_in = {"prompt_tokens": 1103, "completion_tokens": 500,
                  "input_tokens": 1103, "output_tokens": 500}
        sess._deferred_spans = [_payload(sess, "google_genai", raw_in)]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 120}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]

        assert raw["thoughts_token_count"] == 120
        # ADDITIVE mapper: total output = candidates + thoughts. LC's
        # output_tokens was thoughts-INCLUSIVE, so candidates must be netted.
        assert raw["candidates_token_count"] == 380
        assert (raw["candidates_token_count"]
                + raw["thoughts_token_count"]) == 500
        # completion/prompt keys are left alone (mapper prefers candidates).
        assert raw["completion_tokens"] == 500
        assert raw["prompt_tokens"] == 1103

    def test_clamped_to_completion_tokens(self, sess):
        raw_in = {"prompt_tokens": 10, "completion_tokens": 100}
        sess._deferred_spans = [_payload(sess, "google_genai", raw_in)]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 5000}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]
        assert raw["thoughts_token_count"] == 100
        assert raw["candidates_token_count"] == 0

    @pytest.mark.parametrize("existing_key", [
        "thoughts_token_count", "thoughtsTokenCount",
        "candidates_token_count", "candidatesTokenCount",
    ])
    def test_real_google_usage_block_never_rewritten(self, sess, existing_key):
        """When the raw already carries native google keys it came from the
        provider (Fix A's verbatim forward) — an LC-derived guess must not
        touch it."""
        raw_in = {"prompt_tokens": 10, "completion_tokens": 100, existing_key: 77}
        sess._deferred_spans = [_payload(sess, "google_genai", dict(raw_in))]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 40}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]
        assert raw == raw_in

    def test_zero_completion_tokens_injects_nothing(self, sess):
        raw_in = {"prompt_tokens": 10}
        sess._deferred_spans = [_payload(sess, "google_genai", dict(raw_in))]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 40}}

        raw = _flush_capturing(sess)[0]["usage"]["raw"]
        assert raw == raw_in


class TestFlushMergeInertCases:
    @pytest.mark.parametrize("shape", ["anthropic_messages", "openai_responses",
                                       "cohere_chat", "bedrock_converse",
                                       "openrouter_routed", "", None])
    def test_other_shapes_untouched(self, sess, shape):
        raw_in = _openai_raw()
        sess._deferred_spans = [_payload(sess, shape, dict(raw_in))]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        assert _flush_capturing(sess)[0]["usage"]["raw"] == raw_in

    def test_no_stash_leaves_payload_byte_identical(self, sess):
        raw_in = _openai_raw()
        sess._deferred_spans = [_payload(sess, "openai_chat", dict(raw_in))]
        sess._pending_compositions = {f"{sess.trace_id}:0": {}}

        assert _flush_capturing(sess)[0]["usage"]["raw"] == raw_in

    @pytest.mark.parametrize("stashed", [0, None, "abc", -1])
    def test_non_positive_or_junk_stash_is_inert(self, sess, stashed):
        raw_in = _openai_raw()
        sess._deferred_spans = [_payload(sess, "openai_chat", dict(raw_in))]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": stashed}}

        assert _flush_capturing(sess)[0]["usage"]["raw"] == raw_in

    def test_raw_not_a_dict_is_inert(self, sess):
        payload = _payload(sess, "openai_chat", _openai_raw())
        payload["usage"]["raw"] = "not-a-dict"
        sess._deferred_spans = [payload]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        assert _flush_capturing(sess)[0]["usage"]["raw"] == "not-a-dict"

    def test_usage_absent_is_inert(self, sess):
        payload = _payload(sess, "openai_chat", _openai_raw())
        payload.pop("usage")
        sess._deferred_spans = [payload]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        logged = _flush_capturing(sess)
        assert len(logged) == 1
        assert "usage" not in logged[0]

    def test_hostile_details_object_never_raises(self, sess):
        # completion_tokens_details is a non-dict → the injection raises inside
        # the guarded block; the row must still be logged, unmutated.
        raw_in = _openai_raw()
        raw_in["completion_tokens_details"] = "nope"
        sess._deferred_spans = [_payload(sess, "openai_chat", dict(raw_in))]
        sess._pending_compositions = {
            f"{sess.trace_id}:0": {"lc_reasoning_tokens": 357}}

        logged = _flush_capturing(sess)
        assert len(logged) == 1
        assert logged[0]["usage"]["raw"] == raw_in

    def test_merge_hits_only_the_matching_span(self, sess):
        raw_in = _openai_raw()
        sess._deferred_spans = [
            _payload(sess, "openai_chat", dict(raw_in), span_order=0),
            _payload(sess, "openai_chat", dict(raw_in), span_order=1),
        ]
        sess._pending_compositions = {
            f"{sess.trace_id}:1": {"lc_reasoning_tokens": 357}}

        logged = _flush_capturing(sess)
        assert logged[0]["usage"]["raw"] == raw_in         # sibling untouched
        assert logged[1]["usage"]["raw"][
            "completion_tokens_details"]["reasoning_tokens"] == 357


# ─────────────────────────────────────────────────────────────────────────
# 5. D2 — _finalize_langchain_stream stashes from the concat'd chunk
# ─────────────────────────────────────────────────────────────────────────
class TestFinalizeLangchainStream:
    @pytest.fixture(autouse=True)
    def _noop_flush(self, monkeypatch):
        monkeypatch.setattr(enforcer, "_flush_deferred_spans", MagicMock())

    def test_reasoning_stashed_from_accumulated_chunk(self, sess):
        acc = NS(content="hello", usage_metadata=_um(reasoning=357))
        enforcer._finalize_langchain_stream(sess, False, acc, order=0)
        assert _slot(sess, 0)["lc_reasoning_tokens"] == 357

    def test_stash_keys_onto_the_threaded_order(self, sess):
        acc = NS(content="hello", usage_metadata=_um(reasoning=357))
        enforcer._finalize_langchain_stream(sess, False, acc, order=5)
        assert _slot(sess, 5)["lc_reasoning_tokens"] == 357
        assert "lc_reasoning_tokens" not in _slot(sess, 0)

    def test_concat_sum_is_not_re_derived(self, sess):
        """LC's concat already summed reasoning across chunks — the finalize
        must forward the accumulated value verbatim, not recompute it."""
        acc = NS(content="hello", usage_metadata=_um(reasoning=512))
        enforcer._finalize_langchain_stream(sess, False, acc, order=0)
        assert _slot(sess, 0)["lc_reasoning_tokens"] == 512

    def test_no_reasoning_leaves_no_stash(self, sess):
        acc = NS(content="hello", usage_metadata=_um())
        enforcer._finalize_langchain_stream(sess, False, acc, order=0)
        assert "lc_reasoning_tokens" not in _slot(sess, 0)

    def test_acc_none_is_inert(self, sess):
        enforcer._finalize_langchain_stream(sess, False, None, order=0)
        assert "lc_reasoning_tokens" not in _slot(sess, 0)

    def test_hostile_acc_never_raises(self, sess):
        class Boom:
            @property
            def usage_metadata(self):
                raise RuntimeError("boom")

        enforcer._finalize_langchain_stream(sess, False, Boom(), order=0)
        assert "lc_reasoning_tokens" not in _slot(sess, 0)

    def test_defer_flag_still_restored(self, sess):
        """The stash sits inside the finalize's try block — it must not disturb
        the defer-state restore the whole streaming path depends on."""
        acc = NS(content="hello", usage_metadata=_um(reasoning=357))
        sess._defer_telemetry = True
        enforcer._finalize_langchain_stream(sess, False, acc, order=0)
        assert sess._defer_telemetry is False


# ─────────────────────────────────────────────────────────────────────────
# 6. D1 wiring — end-to-end through the real lc_sync / lc_async wrappers
#    (harness mirrors tests/test_eager_failure_outcomes.py)
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def lc_env(sess, monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(enforcer, "_run_sync_check", lambda *a, **k: None)

    async def _noop_async_check(*a, **k):
        return None

    monkeypatch.setattr(enforcer, "_run_async_check", _noop_async_check)
    monkeypatch.setattr(enforcer, "get_client", lambda: client)
    return sess, client


def _install_lc_sync(invoke_fn, model="gpt-5"):
    cls = type("FakeChatOpenAI", (), {"invoke": invoke_fn, "model": model})
    enforcer._set_langchain_wrapper(cls, "invoke", cls.invoke, "sync")
    return cls


def _install_lc_async(ainvoke_fn, model="gpt-5"):
    cls = type("FakeChatOpenAI", (), {"ainvoke": ainvoke_fn, "model": model})
    enforcer._set_langchain_wrapper(cls, "ainvoke", cls.ainvoke, "async")
    return cls


class TestLcWrapperWiring:
    def test_lc_sync_flushed_row_carries_reasoning(self, lc_env):
        sess, client = lc_env
        result = _llm_result([[_gen(_um(100, 400, reasoning=357))]])

        def invoke(self, prompt="p", *a, **k):
            # The instrumentor's deferred span for this call (span_order 0 —
            # `_capture_langchain_prompt` snapshots `_span_counter` == 0).
            sess._deferred_spans.append(
                _payload(sess, "openai_chat", _openai_raw(), span_order=0))
            return result

        cls = _install_lc_sync(invoke)
        assert cls().invoke("hi") is result       # customer result unchanged

        assert len(client.calls) == 1
        raw = client.calls[0]["usage"]["raw"]
        assert raw["completion_tokens_details"]["reasoning_tokens"] == 357
        assert raw["completion_tokens"] == 400    # total output unmoved

    def test_lc_async_flushed_row_carries_reasoning(self, lc_env):
        sess, client = lc_env
        result = _llm_result([[_gen(_um(100, 400, reasoning=357))]])

        async def ainvoke(self, prompt="p", *a, **k):
            sess._deferred_spans.append(
                _payload(sess, "openai_chat", _openai_raw(), span_order=0))
            return result

        cls = _install_lc_async(ainvoke)
        assert _run(cls().ainvoke("hi")) is result

        assert len(client.calls) == 1
        assert client.calls[0]["usage"]["raw"][
            "completion_tokens_details"]["reasoning_tokens"] == 357

    def test_lc_sync_without_reasoning_is_byte_identical(self, lc_env):
        sess, client = lc_env
        raw_in = _openai_raw()

        def invoke(self, prompt="p", *a, **k):
            sess._deferred_spans.append(
                _payload(sess, "openai_chat", dict(raw_in), span_order=0))
            return _llm_result([[_gen(_um(100, 400))]])

        cls = _install_lc_sync(invoke)
        cls().invoke("hi")

        assert client.calls[0]["usage"]["raw"] == raw_in

    def test_lc_sync_hostile_result_never_raises(self, lc_env):
        """GOLDEN RULE: a result the reasoning reader can't parse must still
        return normally to the customer."""
        sess, client = lc_env

        class BoomResult:
            @property
            def generations(self):
                raise RuntimeError("boom")

        boom = BoomResult()

        def invoke(self, prompt="p", *a, **k):
            return boom

        cls = _install_lc_sync(invoke)
        assert cls().invoke("hi") is boom
