"""Regression suite — interleaved LangChain STREAMS must key each stream's
RESPONSE-composition fingerprint onto ITS OWN LLM span, not a sibling's.

Finding: the response-composition fingerprint was attributed via
the single shared session slot `_mode_a_prompt_order` (one-slot mailbox,
last-writer-wins). Under two same-thread interleaved LangChain-native streaming
chains, stream B's `_capture_langchain_prompt` clobbers the mailbox before stream
A finalizes, so A's response fingerprint is filed under B's span_order and A's own
LLM span is left with a prompt but NO response composition. That empty-response
row (with output tokens) trips the backend `bad_composition` health predicate
(`healthDetectors.js:183`/`:392`) and a false `TOOL_SPAN_ORPHAN`
(`integrationVerifier.js:523-529`/`:548`).

Fix (5 touch points, all in enforcer.py): the two stream guards snapshot their own
LLM-span order at prompt-capture time (`_capture_langchain_prompt` now RETURNS it)
and thread it through `_finalize_langchain_stream` into the SHARED
`_capture_response_composition`, which gained an optional `order=None`. The mailbox
read-and-clear stays UNCONDITIONAL (never leaves a dirty last-writer order); only
the USE of the mailbox value is gated (`if order is None`). Extending the shared
helper (not swapping to `_capture_response_composition_at`) preserves the
`service_tier` filing + `set_pending_tool_calls` side-effects. `order=None` default
keeps all ~10 non-stream callers byte-for-byte unchanged.

These tests FAIL on pre-fix enforcer.py (A mis-keyed onto B / A empty) and PASS
post-fix. Assertion numbers map to the security regression contract.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import (
    _current_session,
    _in_langchain,
    in_langchain,
    TPSession,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _fresh_session_and_guard():
    sess = TPSession()
    sess.trace_id = "trace-sec11"
    sess._deferred_spans = []
    sess._defer_telemetry = False
    sess._span_counter = 0
    tok = _current_session.set(sess)
    guard = _in_langchain.set(False)
    try:
        yield sess
    finally:
        try:
            _in_langchain.reset(guard)
        except Exception:
            _in_langchain.set(False)
        _current_session.reset(tok)


@pytest.fixture(autouse=True)
def _stub_composition_builders(monkeypatch):
    """Return identifiable markers so we can see WHICH stream's fingerprint landed
    on WHICH span. Response carries a tool_call entry (for the TOOL_SPAN_ORPHAN
    mapping)."""
    def fake_prompt(provider, kwargs):
        return [{"role": "user", "type": "text", "name": f"prompt::{kwargs.get('messages')}"}]

    def fake_resp(provider, response):
        return [
            {"role": "assistant", "type": "text", "name": f"resp::{response}"},
            {"role": "assistant", "type": "tool_call", "name": f"tool_of::{response}"},
        ]

    monkeypatch.setattr(enforcer, "build_prompt_composition", fake_prompt)
    monkeypatch.setattr(enforcer, "build_response_composition", fake_resp)
    # Quiet the checks by default (individual tests may re-spy).
    monkeypatch.setattr(enforcer, "_run_sync_check", lambda *a, **k: None)

    async def _achk(*a, **k):
        return None
    monkeypatch.setattr(enforcer, "_run_async_check", _achk)


def _instrument_sync(model_cls):
    cls = type("FakeSyncS11", (model_cls,), {})
    enforcer._set_langchain_wrapper(cls, "stream", cls.stream, "stream")
    return cls


def _instrument_async(model_cls):
    cls = type("FakeAsyncS11", (model_cls,), {})
    enforcer._set_langchain_wrapper(cls, "astream", cls.astream, "astream")
    return cls


def _fake_client(monkeypatch):
    logged = []

    class _FakeTP:
        def log_sync(self, **payload):
            logged.append(payload)

    monkeypatch.setattr(enforcer, "get_client", lambda: _FakeTP())
    return logged


def _seed_two_spans(sess):
    """Two deferred LLM spans, self-keyed by their OWN span_order, each with output
    tokens and an empty on_end response_composition fallback (as a streamed
    langchain span arrives)."""
    sess._deferred_spans = [
        {"span": {"trace_id": sess.trace_id, "span_order": 0},
         "usage": {"output_tokens": 10}, "span_kind": "llm", "status": "success",
         "response_composition": []},
        {"span": {"trace_id": sess.trace_id, "span_order": 1},
         "usage": {"output_tokens": 12}, "span_kind": "llm", "status": "success",
         "response_composition": []},
    ]


def _names(comp):
    return [e.get("name") for e in comp] if isinstance(comp, list) else comp


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _SyncModel:
    def stream(self, prompt="p", *a, **k):
        yield f"{prompt}1"
        yield f"{prompt}2"


class _SyncModelErr:
    def stream(self, prompt="p", *a, **k):
        yield f"{prompt}1"
        raise RuntimeError(f"boom-{prompt}")


class _AsyncModel:
    async def astream(self, prompt="p", *a, **k):
        yield f"{prompt}1"
        yield f"{prompt}2"


class _AsyncModelErr:
    async def astream(self, prompt="p", *a, **k):
        yield f"{prompt}1"
        raise RuntimeError(f"boom-{prompt}")


# ═════════════════════════════════════════════════════════════════════════
# Assertion 2 — order=None preserves the existing caller path, BOTH branches
# ═════════════════════════════════════════════════════════════════════════
def test_assert2a_order_none_uses_mailbox_hit_and_clears(_fresh_session_and_guard):
    """2a mailbox-HIT branch: an explicit-order-less call keys onto the STASHED
    mailbox order (not _span_counter-1) and clears the mailbox."""
    sess = _current_session.get()
    sess._mode_a_prompt_order = 7          # stashed by a prior prompt capture
    sess._span_counter = 99                # deliberately different
    enforcer._capture_response_composition("openai", object())  # no order
    assert f"{sess.trace_id}:7" in sess._pending_compositions       # keyed on mailbox 7
    assert f"{sess.trace_id}:98" not in sess._pending_compositions  # NOT _span_counter-1
    assert sess._mode_a_prompt_order is None                        # consumed/cleared


def test_assert2b_order_none_falls_back_to_span_counter_minus_1(_fresh_session_and_guard):
    """2b fallback branch: no mailbox, no explicit order → legacy _span_counter-1."""
    sess = _current_session.get()
    sess._mode_a_prompt_order = None
    sess._span_counter = 5
    enforcer._capture_response_composition("openai", object())  # no order
    assert f"{sess.trace_id}:4" in sess._pending_compositions   # _span_counter-1


def test_assert2_explicit_order_bypasses_mailbox_and_still_clears(_fresh_session_and_guard):
    """Explicit order wins over the mailbox, AND the mailbox is still cleared
    unconditionally (change 1 — no dirty last-writer lingering)."""
    sess = _current_session.get()
    sess._mode_a_prompt_order = 1          # poisoned (a sibling's order)
    sess._span_counter = 50
    enforcer._capture_response_composition("langchain", object(), order=0)
    assert f"{sess.trace_id}:0" in sess._pending_compositions   # explicit order used
    assert f"{sess.trace_id}:1" not in sess._pending_compositions
    assert sess._mode_a_prompt_order is None                    # UNCONDITIONAL clear


# ═════════════════════════════════════════════════════════════════════════
# Assertion 5 — _capture_langchain_prompt returns a non-None int == A's span_order
# ═════════════════════════════════════════════════════════════════════════
def test_assert5_capture_prompt_returns_stashed_order_int(_fresh_session_and_guard):
    sess = _current_session.get()
    sess._span_counter = 3
    order = enforcer._capture_langchain_prompt((object(), "A"))
    assert isinstance(order, int) and order is not None
    assert order == 3                                  # == the LLM-span order it stashed
    assert sess._mode_a_prompt_order == 3              # stashed at:4554
    assert f"{sess.trace_id}:3" in sess._pending_compositions   # prompt filed on that key


def test_assert5_capture_prompt_short_args_returns_none(_fresh_session_and_guard):
    # len(args) < 2 → early return None (unknown order → guards fall back). No raise.
    assert enforcer._capture_langchain_prompt((object(),)) is None


# ═════════════════════════════════════════════════════════════════════════
# Assertion 3 — SYNC interleave: A keyed to A's own span, A not empty, B keeps own
# Assertion 6 — bad_composition predicate cleared (payload tuple)
# Assertion 7 — TOOL_SPAN_ORPHAN cleared (A's tool_call on A's span)
# ═════════════════════════════════════════════════════════════════════════
def _drive_sync_interleave(monkeypatch, sess):
    logged = _fake_client(monkeypatch)
    _seed_two_spans(sess)
    model = _instrument_sync(_SyncModel)()

    sess._span_counter = 0
    gA = model.stream("A")
    next(gA)                       # A starts (order 0), mailbox <- 0
    sess._span_counter = 1
    gB = model.stream("B")
    next(gB)                       # B starts (order 1), mailbox CLOBBERED to 1
    sess._span_counter = 2
    # A left with the clobbered mailbox → mailbox is 1 here (poisoned).
    assert sess._mode_a_prompt_order == 1
    list(gA)                       # A finalizes (inner branch) with explicit order 0
    list(gB)                       # B finalizes (outermost) + flush
    return logged


def test_assert3_6_7_sync_interleave_keys_correctly(monkeypatch, _fresh_session_and_guard):
    sess = _current_session.get()
    logged = _drive_sync_interleave(monkeypatch, sess)

    by_order = {p["span"]["span_order"]: p for p in logged}
    a, b = by_order[0], by_order[1]

    # Assertion 3: A's response on A's OWN span, not empty; B keeps its own.
    assert _names(a["response_composition"]) == ["resp::A1A2", "tool_of::A1A2"]
    assert _names(b["response_composition"]) == ["resp::B1B2", "tool_of::B1B2"]
    # no cross-inheritance
    assert "resp::A1A2" not in (_names(b["response_composition"]) or [])

    # Assertion 3 (mailbox clean): after both streams, mailbox not left dirty.
    assert sess._mode_a_prompt_order is None

    # Assertion 6: healthDetectors bad_composition — verbatim disjunct
    # status='success' AND span_kind='llm' AND (output_tokens>0 AND response IN ('','[]'))
    # A's payload carries the FULL tuple the predicate reads; the empty-response
    # disjunct is FALSE (response non-empty).
    assert a["span_kind"] == "llm"
    assert a["status"] == "success"
    assert a["usage"]["output_tokens"] > 0
    assert a["response_composition"] not in ("", "[]", [])   # non-empty → predicate FALSE

    # Assertion 7: TOOL_SPAN_ORPHAN — A's response has a tool_call entry w/ a name
    # (integrationVerifier.js:525 `type==='tool_call'` → request on A's span).
    a_tool = [e for e in a["response_composition"]
              if e.get("type") == "tool_call" or e.get("role") == "tool_call"]
    assert a_tool and a_tool[0].get("name") == "tool_of::A1A2"
    # B does not inherit A's tool_call
    b_tool_names = [e.get("name") for e in b["response_composition"]
                    if e.get("type") == "tool_call"]
    assert "tool_of::A1A2" not in b_tool_names


# ═════════════════════════════════════════════════════════════════════════
# Assertion 4 — ASYNC same-Task interleave: same guarantees
# ═════════════════════════════════════════════════════════════════════════
def test_assert4_async_interleave_keys_correctly(monkeypatch, _fresh_session_and_guard):
    async def go():
        sess = _current_session.get()
        logged = _fake_client(monkeypatch)
        _seed_two_spans(sess)
        model = _instrument_async(_AsyncModel)()

        sess._span_counter = 0
        agA = model.astream("A")
        await agA.__anext__()          # A's prompt capture + order 0 snapshot
        sess._span_counter = 1
        agB = model.astream("B")
        await agB.__anext__()          # B clobbers mailbox to 1
        sess._span_counter = 2
        assert sess._mode_a_prompt_order == 1     # poisoned before A finalizes

        async for _ in agA:
            pass
        async for _ in agB:
            pass

        by_order = {p["span"]["span_order"]: p for p in logged}
        a, b = by_order[0], by_order[1]
        assert _names(a["response_composition"]) == ["resp::A1A2", "tool_of::A1A2"]
        assert _names(b["response_composition"]) == ["resp::B1B2", "tool_of::B1B2"]
        assert sess._mode_a_prompt_order is None

    _run(go())


# ═════════════════════════════════════════════════════════════════════════
# Assertion 8 — side-effects PRESERVED on the STREAM path (positive spy).
# set_pending_tool_calls INVOKED with A's ids + usage.tier filed at explicit order.
# (A _capture_response_composition_at swap would fail BOTH.)
# ═════════════════════════════════════════════════════════════════════════
def test_assert8_stream_path_invokes_set_pending_tool_calls(monkeypatch, _fresh_session_and_guard):
    sess = _current_session.get()
    spy = []
    monkeypatch.setattr(enforcer, "set_pending_tool_calls", lambda tc: spy.append(tc))
    monkeypatch.setattr(enforcer, "extract_pending_tool_calls",
                        lambda provider, response: [{"id": "call_A", "name": f"tool::{response}"}])
    _fake_client(monkeypatch)
    sess._deferred_spans = [
        {"span": {"trace_id": sess.trace_id, "span_order": 0},
         "usage": {"output_tokens": 10}, "span_kind": "llm", "status": "success",
         "response_composition": []},
    ]
    model = _instrument_sync(_SyncModel)()
    sess._span_counter = 0
    list(model.stream("A"))        # single stream, full drain → finalize path

    # POSITIVE spy: set_pending_tool_calls was actually CALLED on the stream path,
    # with A's extracted tool ids — not merely a non-empty end state.
    assert spy, "set_pending_tool_calls was never invoked on the langchain-stream path"
    assert spy[-1] == [{"id": "call_A", "name": "tool::A1A2"}]


def test_assert8_stream_path_files_service_tier_at_order(monkeypatch, _fresh_session_and_guard):
    sess = _current_session.get()
    monkeypatch.setattr(enforcer, "_extract_service_tier", lambda r: "priority")
    _fake_client(monkeypatch)
    sess._deferred_spans = [
        {"span": {"trace_id": sess.trace_id, "span_order": 0},
         "usage": {"output_tokens": 10}, "span_kind": "llm", "status": "success",
         "response_composition": []},
    ]
    # Capture directly on the stream path via the shared helper with an explicit order.
    enforcer._capture_response_composition("langchain", object(), order=0)
    assert sess._pending_compositions[f"{sess.trace_id}:0"]["service_tier"] == "priority"


# ═════════════════════════════════════════════════════════════════════════
# Assertion 13 — tier filed on A's key under the CLOBBERED mailbox (future
# cost-mis-attribution scenario: explicit order overrides poisoned mailbox).
# ═════════════════════════════════════════════════════════════════════════
def test_assert13_tier_lands_on_A_key_under_clobbered_mailbox(monkeypatch, _fresh_session_and_guard):
    sess = _current_session.get()
    monkeypatch.setattr(enforcer, "_extract_service_tier", lambda r: "priority")
    sess._mode_a_prompt_order = 1          # B was last writer (poisoned to order_B=1)
    sess._span_counter = 9
    # A finalizes with its OWN explicit order 0 while the mailbox says 1.
    enforcer._capture_response_composition("langchain", object(), order=0)
    assert sess._pending_compositions[f"{sess.trace_id}:0"]["service_tier"] == "priority"
    assert "service_tier" not in sess._pending_compositions.get(f"{sess.trace_id}:1", {})
    assert sess._mode_a_prompt_order is None


# ═════════════════════════════════════════════════════════════════════════
# Assertion 11 streaming edges on the THREADED-ORDER path
# ═════════════════════════════════════════════════════════════════════════
def test_assert11a_sync_early_abandonment_no_crash_guard_restored(monkeypatch, _fresh_session_and_guard):
    sess = _current_session.get()
    _fake_client(monkeypatch)
    _seed_two_spans(sess)
    model = _instrument_sync(_SyncModel)()
    gA = model.stream("A")
    next(gA)
    gA.close()                          # abandon early (triggers finally / order threading)
    assert in_langchain() is False      # guard restored, no crash
    # A subsequent stream still works + keys correctly.
    sess._span_counter = 1
    gC = model.stream("C")
    assert next(gC) == "C1"
    gC.close()


def test_assert11a_async_early_abandonment_no_crash(monkeypatch, _fresh_session_and_guard):
    async def go():
        sess = _current_session.get()
        _fake_client(monkeypatch)
        _seed_two_spans(sess)
        model = _instrument_async(_AsyncModel)()
        agA = model.astream("A")
        await agA.__anext__()
        await agA.aclose()
        assert in_langchain() is False
    _run(go())


def test_assert11b_sync_midstream_error_verbatim_guard_restored(monkeypatch, _fresh_session_and_guard):
    sess = _current_session.get()
    _fake_client(monkeypatch)
    sess._deferred_spans = []
    model = _instrument_sync(_SyncModelErr)()
    gA = model.stream("A")
    assert next(gA) == "A1"
    with pytest.raises(RuntimeError) as ei:
        next(gA)
    assert str(ei.value) == "boom-A"        # provider error propagates VERBATIM
    assert type(ei.value) is RuntimeError
    assert in_langchain() is False          # guard restored on error unwind


def test_assert11b_async_midstream_error_verbatim(monkeypatch, _fresh_session_and_guard):
    async def go():
        sess = _current_session.get()
        _fake_client(monkeypatch)
        sess._deferred_spans = []
        model = _instrument_async(_AsyncModelErr)()
        agA = model.astream("A")
        assert await agA.__anext__() == "A1"
        with pytest.raises(RuntimeError) as ei:
            await agA.__anext__()
        assert str(ei.value) == "boom-A"
        assert in_langchain() is False
    _run(go())


def test_assert11c_failopen_when_prompt_capture_raises(monkeypatch, _fresh_session_and_guard):
    """A hostile build_prompt_composition making _capture_langchain_prompt raise
    internally → @fail_safe returns None → order=None → mailbox fallback, and the
    customer still receives ALL chunks (no escape)."""
    sess = _current_session.get()
    _fake_client(monkeypatch)
    _seed_two_spans(sess)

    def _boom(provider, kwargs):
        raise ValueError("hostile prompt builder")
    monkeypatch.setattr(enforcer, "build_prompt_composition", _boom)

    model = _instrument_sync(_SyncModel)()
    sess._span_counter = 0
    got = list(model.stream("A"))       # must NOT raise into customer code
    assert got == ["A1", "A2"]          # all chunks delivered (fail-open)
    assert in_langchain() is False


def test_assert11c_failopen_when_response_builder_raises(monkeypatch, _fresh_session_and_guard):
    sess = _current_session.get()
    _fake_client(monkeypatch)
    _seed_two_spans(sess)

    def _boom(provider, response):
        raise ValueError("hostile response builder")
    monkeypatch.setattr(enforcer, "build_response_composition", _boom)

    model = _instrument_sync(_SyncModel)()
    sess._span_counter = 0
    got = list(model.stream("A"))
    assert got == ["A1", "A2"]
    assert in_langchain() is False
