"""Anthropic `.stream()` context-manager exit must drop ONLY its own deferred
entries — never wipe telemetry that other concurrent calls queued on the same
session.

Bug (state-observed): both `__exit__` (sync) and `__aexit__` (async) did
`session._deferred_spans.clear()` — wiping EVERY queued span, including those
belonging to other interleaved streams or third-party instrumented calls
sharing the session. Their telemetry was permanently lost.

Fix: at exit, drop only (a) this call's reserved span_order and (b) the
usage-less inner-instrumentor "dud" span queued in this call's window on this
call's trace. Everything else — real-usage spans, other orders/traces — is kept.

Empirical dud shape (confirmed by driving the real span processor): when the
inner OpenLLMetry anthropic stream span is SUPPRESSED it never lands; when
suppression is MISSED it lands as one payload (provider="anthropic") carrying
span.trace_id == session.trace_id and an EARLIER span_order (!= this call's
reserved order). Whether that payload carries gen_ai.usage.* depends on the
instrumentor version (0.60 populated none, 0.61 does) — never branch on it;
the scoped drop here only ever removes the usage-LESS variant, and the fakes
below model that variant (input/output/cached == 0), appended by the manager
during `__exit__`/`__aexit__` — exactly when the real on_end fires.

All fakes — no anthropic SDK involved. Harness style mirrors
tests/test_anthropic_stream_usage_shape.py and
tests/test_sec06_langchain_interleaved.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import TPSession, _current_session
from token_police.exceptions import TokenPoliceBlockedError


# ─────────────────────────────────────────────────────────────────────────
# Fakes
# ─────────────────────────────────────────────────────────────────────────
def _dud(session, order=99):
    """A usage-less anthropic dud, shaped like the inner instrumentor's payload
    (empty tokens, span on this session's trace, an unrelated span_order)."""
    return {
        "tp_tag": "DUD",
        "provider": "anthropic",
        "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
        "span": {"trace_id": session.trace_id, "span_order": order},
    }


def _real(tag, session, order):
    """A real-usage deferred payload from some OTHER call (must survive)."""
    return {
        "tp_tag": tag,
        "provider": "openai",
        "input_tokens": 100, "output_tokens": 50, "cached_tokens": 0,
        "span": {"trace_id": session.trace_id, "span_order": order},
    }


class _FakeStream:
    def __init__(self, events=(), final=None):
        self._events = list(events)
        self._final = final

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class _FakeAsyncStream(_FakeStream):
    def __aiter__(self):
        async def _g():
            for e in self._events:
                yield e
        return _g()

    async def get_final_message(self):
        return self._final


class _DudMgr:
    """Fake sync MessageStreamManager. On __exit__ it appends a usage-less dud
    to the session's deferred buffer — mimicking the inner OpenLLMetry on_end
    firing INSIDE the manager's __exit__ (the missed-suppression flow)."""

    def __init__(self, stream, session, append_dud=True):
        self._stream = stream
        self._session = session
        self._append_dud = append_dud
        self._entered = False
        self.dud = None

    def __enter__(self):
        self._entered = True
        return self._stream

    def __exit__(self, *a):
        # The inner instrumentor's on_end only fires for a call that actually
        # ran — mirror that (a blocked-at-enter call never queues a dud).
        if self._append_dud and self._entered:
            self.dud = _dud(self._session)
            self._session._deferred_spans.append(self.dud)
        return False


class _AsyncDudMgr(_DudMgr):
    async def __aenter__(self):
        self._entered = True
        return self._stream

    async def __aexit__(self, *a):
        if self._append_dud and self._entered:
            self.dud = _dud(self._session)
            self._session._deferred_spans.append(self.dud)
        return False


class _CaptureTP:
    def __init__(self):
        self.logged = []

    def log_sync(self, **payload):
        self.logged.append(payload)


@pytest.fixture(autouse=True)
def _session_and_client(monkeypatch):
    """Pin one shared session (so interleaves share it) and a capturing client.
    `get_final_message()` returns None in every stream, so `_finalize` never
    emits a manual row — the only rows are the ones a later flush dispatches,
    which keeps these tests focused on the buffer contents."""
    sess = TPSession()
    sess._deferred_spans = []
    sess._defer_telemetry = False
    tok = _current_session.set(sess)
    tp = _CaptureTP()
    monkeypatch.setattr(enforcer, "get_client", lambda: tp)
    try:
        yield sess, tp
    finally:
        _current_session.reset(tok)


def _kwargs():
    return {"model": "claude-3-5-sonnet",
            "messages": [{"role": "user", "content": "hi"}]}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _tags(session):
    return [p.get("tp_tag") for p in session._deferred_spans]


# ─────────────────────────────────────────────────────────────────────────
# 2. Interleaving: a third-party real-usage span queued while A and B are both
# open SURVIVES both exits, and a later flush dispatches it.
# ─────────────────────────────────────────────────────────────────────────
def test_interleaved_third_party_span_survives_both_exits(_session_and_client):
    sess, tp = _session_and_client

    wA = enforcer._AnthropicStreamMgrWrapper(_DudMgr(_FakeStream(), sess), _kwargs())
    wB = enforcer._AnthropicStreamMgrWrapper(_DudMgr(_FakeStream(), sess), _kwargs())

    wA.__enter__()
    wB.__enter__()                       # both open, both deferring
    # Third-party instrumented call queues a REAL-usage span while both open.
    third = _real("THIRD_PARTY", sess, order=7)
    sess._deferred_spans.append(third)

    wB.__exit__(None, None, None)        # B exits first (non-LIFO)
    assert "THIRD_PARTY" in _tags(sess)  # pre-fix: whole buffer cleared → gone
    assert "DUD" not in _tags(sess)      # B's own dud dropped
    assert sess._deferred_spans == [third]

    wA.__exit__(None, None, None)        # A exits
    assert sess._deferred_spans == [third]   # still there; A's dud dropped
    assert sess._defer_telemetry is False    # A restored the baseline

    # A later flush (gate now open) dispatches the survivor exactly once.
    enforcer._flush_deferred_spans(sess)
    assert [p.get("tp_tag") for p in tp.logged] == ["THIRD_PARTY"]
    assert sess._deferred_spans == []


def test_interleaved_third_party_span_survives_async(_session_and_client):
    sess, tp = _session_and_client

    async def go():
        wA = enforcer._AnthropicAsyncStreamMgrWrapper(
            _AsyncDudMgr(_FakeAsyncStream(), sess), _kwargs())
        wB = enforcer._AnthropicAsyncStreamMgrWrapper(
            _AsyncDudMgr(_FakeAsyncStream(), sess), _kwargs())
        await wA.__aenter__()
        await wB.__aenter__()
        third = _real("THIRD_PARTY", sess, order=7)
        sess._deferred_spans.append(third)

        await wB.__aexit__(None, None, None)
        assert "THIRD_PARTY" in _tags(sess)
        assert "DUD" not in _tags(sess)

        await wA.__aexit__(None, None, None)
        assert sess._deferred_spans == [third]

        enforcer._flush_deferred_spans(sess)
        assert [p.get("tp_tag") for p in tp.logged] == ["THIRD_PARTY"]

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# 3. Dud IS still dropped — no duplicate empty row leaks to a later flush.
# ─────────────────────────────────────────────────────────────────────────
def test_own_dud_dropped_no_duplicate_empty_row(_session_and_client):
    sess, tp = _session_and_client
    mgr = _DudMgr(_FakeStream(), sess)
    w = enforcer._AnthropicStreamMgrWrapper(mgr, _kwargs())

    with w as s:
        list(s)

    assert mgr.dud is not None                    # the dud WAS queued at exit
    assert sess._deferred_spans == []             # ...and then dropped
    enforcer._flush_deferred_spans(sess)
    assert tp.logged == []                         # no empty row dispatched


def test_own_dud_dropped_async(_session_and_client):
    sess, tp = _session_and_client

    async def go():
        mgr = _AsyncDudMgr(_FakeAsyncStream(), sess)
        w = enforcer._AnthropicAsyncStreamMgrWrapper(mgr, _kwargs())
        async with w as s:
            async for _ in s:
                pass
        assert mgr.dud is not None
        assert sess._deferred_spans == []
        enforcer._flush_deferred_spans(sess)
        assert tp.logged == []

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# 4. Nested (LIFO): an OUTER defer scope with its own queued span + one
# anthropic stream inside → outer span intact, dud gone, defer restored to
# the OUTER True (not blindly cleared).
# ─────────────────────────────────────────────────────────────────────────
def test_nested_inside_outer_defer_scope(_session_and_client):
    sess, tp = _session_and_client
    # Simulate being inside an outer defer scope that already queued a span.
    sess._defer_telemetry = True
    outer = _real("OUTER", sess, order=0)
    sess._deferred_spans.append(outer)

    mgr = _DudMgr(_FakeStream(), sess)
    w = enforcer._AnthropicStreamMgrWrapper(mgr, _kwargs())
    with w as s:
        list(s)

    # Outer scope's queued span survives; the inner dud is dropped.
    assert sess._deferred_spans == [outer]
    # Defer restored to the OUTER scope's value, so the outer owner still
    # controls when the buffer flushes.
    assert sess._defer_telemetry is True


# ─────────────────────────────────────────────────────────────────────────
# 5. Blocked-at-enter: the async pre-flight check raises → the manager is never
# entered, no buffer mutation, no crash — and a stray __aexit__ with unset
# attributes still tolerates gracefully.
# ─────────────────────────────────────────────────────────────────────────
def test_blocked_at_enter_no_buffer_mutation(_session_and_client, monkeypatch):
    sess, tp = _session_and_client
    pre = _real("PRE_EXISTING", sess, order=0)
    sess._deferred_spans.append(pre)

    async def _blocked_check(*a, **k):
        raise TokenPoliceBlockedError("budget exceeded")

    monkeypatch.setattr(enforcer, "_run_async_check", _blocked_check)

    async def go():
        w = enforcer._AnthropicAsyncStreamMgrWrapper(
            _AsyncDudMgr(_FakeAsyncStream(), sess), _kwargs())
        with pytest.raises(TokenPoliceBlockedError):
            async with w as s:      # __aenter__ raises before entering the mgr
                async for _ in s:   # never reached
                    pass
        # Buffer untouched, defer untouched — the blocked call never deferred.
        assert sess._deferred_spans == [pre]
        assert sess._defer_telemetry is False
        # Even a stray exit (attrs never set past the block) must not crash or
        # mutate the buffer: _enter_deferred_ids is None → drop is skipped.
        await w.__aexit__(None, None, None)
        assert sess._deferred_spans == [pre]

    _run(go())
