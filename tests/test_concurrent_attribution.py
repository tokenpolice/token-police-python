"""Concurrent same-session attribution — two calls racing on ONE session must
key each call's prompt/response composition and service_tier onto ITS OWN LLM
span, never a sibling's.

Mechanism under test: an instrumented provider wrapper reserves its LLM-span
order UP FRONT (before invoking the provider) via ``reserve_span_order`` and
threads that order into every pre-call stash and the post-call response capture.
Telemetry ``on_start`` CONSUMES that reservation (``consume_reserved_span_order``)
instead of allocating its own, so the stash keys and the span's ``tp.span_order``
always agree. Because the reservation lives in a per-thread / per-task contextvar
and ``next_span_order`` is lock-guarded, two calls interleaved on one session get
distinct, non-crossing orders.

Before the fix the wrapper only PEEKED ``_span_counter`` pre-call and the post-
call capture recomputed it from the mutable counter / one-slot mailbox, so two
calls that both peek the same counter value before either span starts collide on
one key: one span inherits the other's prompt while its own is dropped.

The tests drive the real order lifecycle (reserve -> pre-stash -> on_start
consume -> reset -> post-capture) across real threads / asyncio tasks, with a
2-party barrier forcing the racy interleave (both pre-stash before either
consumes). They pass on the fixed code and fail if the reserved order is not
threaded through both the stash and the capture.
"""
from __future__ import annotations

import asyncio
import contextvars
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import (
    _current_session,
    TPSession,
    reserve_span_order,
    reset_span_order,
    consume_reserved_span_order,
)


# ─────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────
@pytest.fixture()
def sess():
    s = TPSession()
    s.trace_id = "trace-concurrent"
    s._span_counter = 0
    s._deferred_spans = []
    s._defer_telemetry = False
    tok = _current_session.set(s)
    try:
        yield s
    finally:
        _current_session.reset(tok)


@pytest.fixture(autouse=True)
def _stub_builders(monkeypatch):
    """Identifiable markers so we can see WHICH call's fingerprint landed on
    WHICH span."""
    def fake_prompt(provider, kwargs, operation=None):
        return [{"role": "user", "type": "text", "name": f"prompt::{kwargs.get('tag')}"}]

    def fake_resp(provider, response, usage_shape=None):
        return [{"role": "assistant", "type": "text", "name": f"resp::{response}"}]

    monkeypatch.setattr(enforcer, "build_prompt_composition", fake_prompt)
    monkeypatch.setattr(enforcer, "build_response_composition", fake_resp)


def _fake_client(monkeypatch):
    logged = []

    class _FakeTP:
        def log_sync(self, **payload):
            logged.append(payload)

    monkeypatch.setattr(enforcer, "get_client", lambda: _FakeTP())
    return logged


def _seed_two_spans(session):
    """Two deferred LLM spans self-keyed by their OWN span_order (as on_end would
    build them once the reservation is consumed)."""
    session._deferred_spans = [
        {"span": {"trace_id": session.trace_id, "span_order": 0},
         "usage": {"output_tokens": 10}, "span_kind": "llm", "status": "success",
         "response_composition": []},
        {"span": {"trace_id": session.trace_id, "span_order": 1},
         "usage": {"output_tokens": 12}, "span_kind": "llm", "status": "success",
         "response_composition": []},
    ]


def _names(comp):
    return [e.get("name") for e in comp] if isinstance(comp, list) else comp


# A single call's order lifecycle, split so a caller can interleave the phases
# of two concurrent calls exactly as the generic Mode-A wrapper + telemetry
# on_start do. Each phase mirrors the real wrapper code path.
def _reserve():
    session = enforcer.get_current_session()
    order = session.next_span_order()
    token = reserve_span_order(order)
    return order, token


def _prestash(tag, order):
    # Mirrors the wrapper's pre-call stashes threading the reserved order.
    enforcer._capture_prompt_composition("openai", {"tag": tag}, order=order)
    enforcer._stash_api_base(enforcer.get_current_session(), "https://api.example.com", order=order)


def _on_start_consume():
    # Mirrors telemetry on_start: consume the reservation, else allocate.
    o = consume_reserved_span_order()
    if o is None:
        o = enforcer.get_current_session().next_span_order()
    return o


def _post(tag, order, tier):
    enforcer._capture_response_composition("openai", f"{tag}-resp", service_tier=tier, order=order)


# ═════════════════════════════════════════════════════════════════════════
# 1. Two THREADED concurrent calls on one session — no crossing, no loss.
# ═════════════════════════════════════════════════════════════════════════
def test_threaded_concurrent_no_cross_attribution(monkeypatch, sess):
    logged = _fake_client(monkeypatch)
    _seed_two_spans(sess)

    barrier = threading.Barrier(2)
    span_of = {}
    errors = []

    def call(tag, tier):
        # A fresh thread starts with a default contextvar context, so each call
        # gets its OWN reservation / session binding — exactly the isolation the
        # fix relies on.
        tok = _current_session.set(sess)
        try:
            order, rtoken = _reserve()
            _prestash(tag, order)
            barrier.wait()          # both prompts stashed + orders reserved
            span_order = _on_start_consume()
            reset_span_order(rtoken)
            _post(tag, order, tier)
            span_of[tag] = span_order
        except Exception as e:       # pragma: no cover - surfaced via errors
            errors.append(e)
        finally:
            _current_session.reset(tok)

    tA = threading.Thread(target=call, args=("A", "priority"))
    tB = threading.Thread(target=call, args=("B", "batch"))
    tA.start(); tB.start()
    tA.join(); tB.join()

    assert not errors, errors
    # Distinct spans, no double-allocation.
    assert set(span_of.values()) == {0, 1}

    enforcer._flush_deferred_spans(sess)
    by_order = {p["span"]["span_order"]: p for p in logged}

    for tag in ("A", "B"):
        so = span_of[tag]
        p = by_order[so]
        # prompt AND response AND tier all land on THIS call's own span.
        assert _names(p["prompt_composition"]) == [f"prompt::{tag}"], p["prompt_composition"]
        assert _names(p["response_composition"]) == [f"resp::{tag}-resp"], p["response_composition"]
        expected_tier = "priority" if tag == "A" else "batch"
        assert p["usage"]["tier"] == expected_tier
    # No cross-inheritance between the two spans.
    assert span_of["A"] != span_of["B"]


# ═════════════════════════════════════════════════════════════════════════
# 2. Two concurrent ASYNC calls (asyncio.gather) — same guarantees.
# ═════════════════════════════════════════════════════════════════════════
def test_async_concurrent_no_cross_attribution(monkeypatch, sess):
    logged = _fake_client(monkeypatch)
    _seed_two_spans(sess)

    class _ABarrier:
        """2-party rendezvous (asyncio.Barrier is 3.11+; this works on 3.8+)."""
        def __init__(self):
            self._n = 0
            self._ev = asyncio.Event()

        async def wait(self):
            self._n += 1
            if self._n >= 2:
                self._ev.set()
            await self._ev.wait()

    async def go():
        barrier = _ABarrier()
        span_of = {}

        async def call(tag, tier):
            # gather runs each coroutine in its own Task, which copies the
            # current context → per-task reservation isolation.
            order, rtoken = _reserve()
            _prestash(tag, order)
            await barrier.wait()
            span_order = _on_start_consume()
            reset_span_order(rtoken)
            _post(tag, order, tier)
            span_of[tag] = span_order

        await asyncio.gather(call("A", "priority"), call("B", "batch"))
        return span_of

    span_of = asyncio.new_event_loop().run_until_complete(go())
    assert set(span_of.values()) == {0, 1}

    enforcer._flush_deferred_spans(sess)
    by_order = {p["span"]["span_order"]: p for p in logged}
    for tag in ("A", "B"):
        p = by_order[span_of[tag]]
        assert _names(p["prompt_composition"]) == [f"prompt::{tag}"]
        assert _names(p["response_composition"]) == [f"resp::{tag}-resp"]
        assert p["usage"]["tier"] == ("priority" if tag == "A" else "batch")


# ═════════════════════════════════════════════════════════════════════════
# 3. Reservation hygiene — a completed wrapped call leaves NO stale reservation;
# a subsequent un-wrapped span start allocates fresh.
# ═════════════════════════════════════════════════════════════════════════
def test_reservation_reset_leaves_no_stale(sess):
    order, rtoken = _reserve()
    _prestash("A", order)
    consumed = _on_start_consume()
    reset_span_order(rtoken)
    _post("A", order, "")
    assert consumed == order == 0

    # A later span start (no wrapper reservation) must NOT consume a stale one.
    assert consume_reserved_span_order() is None
    # …and would allocate fresh via next_span_order.
    assert sess.next_span_order() == 1


def test_double_consume_is_one_shot(sess):
    order, rtoken = _reserve()
    assert consume_reserved_span_order() == order      # first consumer wins
    assert consume_reserved_span_order() is None        # already consumed
    reset_span_order(rtoken)


# ═════════════════════════════════════════════════════════════════════════
# 4. Fallback intactness — with NO explicit order the mailbox-then-`-1` path is
# unchanged (dormant fallback layer preserved).
# ═════════════════════════════════════════════════════════════════════════
def test_fallback_uses_mailbox_when_no_order(sess):
    sess._mode_a_prompt_order = 7
    sess._span_counter = 99
    enforcer._capture_response_composition("openai", object())  # no order
    assert f"{sess.trace_id}:7" in sess._pending_compositions
    assert f"{sess.trace_id}:98" not in sess._pending_compositions
    assert sess._mode_a_prompt_order is None


def test_fallback_uses_span_counter_minus_one_when_no_mailbox(sess):
    sess._mode_a_prompt_order = None
    sess._span_counter = 5
    enforcer._capture_response_composition("openai", object())  # no order
    assert f"{sess.trace_id}:4" in sess._pending_compositions
