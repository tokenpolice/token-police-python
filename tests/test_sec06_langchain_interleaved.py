"""Regression suite — interleaved LangChain streams must NOT skip
enforcement + telemetry for the second stream.

Finding (MIRROR of the guard-flag leak, different trigger): the LangChain
sync/async stream guards (`_guard_sync_stream` / `_guard_async_stream`) held the
`_in_langchain` re-entrancy ContextVar True across the WHOLE customer iteration
(set on the first pull, cleared only in the outer `finally` at exhaustion/close).
A plain sync generator shares the caller's `contextvars.Context`, so that hold
LEAKS into the caller and stays True while stream A is alive-but-suspended at a
`yield`. A second stream B opened in that window saw `in_langchain()==True` at the
`lc_stream`/`lc_astream` gate (enforcer.py:4672/:4682) and took the pass-through
branch — skipping its pre-flight `_run_sync_check()` (UNDER-enforcement) and never
being wrapped by the guard (partial metering loss).

Fix (two coordinated parts, both confined to the two guard bodies):
  (A) per-step guard scope — hold `_in_langchain` True ONLY around each inner pull
      and restore the TRUE prior value (`set(prev)`, which cannot raise) BEFORE the
      customer-facing `yield`.
  (B) stream-local defer coordination — a depth counter (`_lc_stream_depth`) + a
      one-time baseline stash (`_lc_stream_defer_prev`) keep `_defer_telemetry`
      truthy across the interleave and flush the shared deferred-span buffer ONCE,
      at the outermost (depth-0) unwind, so a non-LIFO drain neither clobbers the
      other stream's in-flight span nor pins `_defer_telemetry` (no cascade).

These tests are written to FAIL on pre-fix `enforcer.py` (guard leaks True → B
short-circuits at the gate) and PASS post-fix. The mid-stream-error / early-
abandonment / fail-open tests are the (golden-rule) guards.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make the local SDK importable without a prior editable install.
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
    """Install a fresh shared session + reset the guard around every test so the
    two guards in an interleave share one session and nothing leaks between
    cases. `get_current_session()` returns a NEW default TPSession when none is
    set, so we MUST pin one for the interleave state to be shared/observable."""
    sess = TPSession()
    sess._deferred_spans = []
    sess._defer_telemetry = False
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


def _instrument_sync(model_cls):
    cls = type("FakeSync", (model_cls,), {})
    enforcer._set_langchain_wrapper(cls, "stream", cls.stream, "stream")
    return cls


def _instrument_async(model_cls):
    cls = type("FakeAsync", (model_cls,), {})
    enforcer._set_langchain_wrapper(cls, "astream", cls.astream, "astream")
    return cls


def _spy_sync_check(monkeypatch):
    calls = []
    monkeypatch.setattr(enforcer, "_run_sync_check",
                        lambda *a, **k: calls.append(1))
    return calls


def _spy_async_check(monkeypatch):
    calls = []

    async def spy(*a, **k):
        calls.append(1)

    monkeypatch.setattr(enforcer, "_run_async_check", spy)
    return calls


def _spy_flush(monkeypatch):
    """Wrap the REAL _flush_deferred_spans so real dispatch (log_sync + buffer
    clear) still happens, while counting invocations. _finalize_langchain_stream
    references it as a module global, so this is observed by the fix too."""
    real = enforcer._flush_deferred_spans
    calls = []

    # Signature-transparent: forward every keyword (e.g. obs_key) so the spy
    # survives param additions instead of raising TypeError into a fail-open
    # caller, which would silently swallow the flush.
    def wrapper(session, **kw):
        calls.append(1)
        return real(session, **kw)

    monkeypatch.setattr(enforcer, "_flush_deferred_spans", wrapper)
    return calls


def _fake_client(monkeypatch):
    """Give _flush_deferred_spans a client so it actually dispatches + clears the
    buffer. Records every payload log_sync receives."""
    logged = []

    class _FakeTP:
        def log_sync(self, **payload):
            logged.append(payload)

    monkeypatch.setattr(enforcer, "get_client", lambda: _FakeTP())
    return logged


def _tagged(tag, order=0):
    """A minimally-valid deferred-span payload (shape mirrors telemetry.py's
    deferred payloads: a dict `**`-expandable into log_sync with a `span` sub-dict
    carrying trace_id/span_order). `tp_tag` uniquely identifies it."""
    return {"tp_tag": tag, "span": {"trace_id": "t", "span_order": order}}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# Fake models — `stream`/`astream` yield chunks tagged with the prompt so each
# stream's fold (acc) is distinguishable.
class _SyncModel:
    def stream(self, prompt="p", *a, **k):
        for i in range(3):
            yield f"{prompt}-{i}"


class _SyncModelErr:
    def stream(self, prompt="p", *a, **k):
        yield f"{prompt}-0"
        raise RuntimeError(f"boom-{prompt}")


class _AsyncModel:
    async def astream(self, prompt="p", *a, **k):
        for i in range(3):
            yield f"{prompt}-{i}"


class _AsyncModelErr:
    async def astream(self, prompt="p", *a, **k):
        yield f"{prompt}-0"
        raise RuntimeError(f"boom-{prompt}")


def _is_sync_guard(obj):
    return getattr(getattr(obj, "gi_code", None), "co_name", "") == "_guard_sync_stream"


def _is_async_guard(obj):
    return getattr(getattr(obj, "ag_code", None), "co_name", "") == "_guard_async_stream"


# ─────────────────────────────────────────────────────────────────────────
# Assertions 2, 3 — B ENFORCED (spy check fires) + goes through the guard
# ─────────────────────────────────────────────────────────────────────────
def test_sync_interleaved_B_enforced_and_wrapped(monkeypatch):        # assert 2
    calls = _spy_sync_check(monkeypatch)
    model = _instrument_sync(_SyncModel)()

    gA = model.stream("A")
    next(gA)                                # A starts, guard set + restored
    assert calls == [1]                     # A enforced

    gB = model.stream("B")                  # opened while A suspended
    assert calls == [1, 1]                  # B ENFORCED (pre-fix: still [1])
    assert _is_sync_guard(gB)               # B wrapped by _guard_sync_stream

    for _ in gA:
        pass
    for _ in gB:
        pass


def test_async_interleaved_B_enforced_and_wrapped(monkeypatch):       # assert 3
    async def go():
        calls = _spy_async_check(monkeypatch)
        model = _instrument_async(_AsyncModel)()

        agA = model.astream("A")
        await agA.__anext__()               # A starts (check A fires)
        assert calls == [1]

        agB = model.astream("B")            # same-Task interleave
        assert _is_async_guard(agB)         # pre-fix: raw async iter, not the guard
        await agB.__anext__()               # B's check fires here
        assert calls == [1, 1]              # B ENFORCED (pre-fix: never wrapped)

        async for _ in agA:
            pass
        async for _ in agB:
            pass

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 2(iii) / 6 — B's deferred SPAN LANDS exactly once (non-LIFO drain);
# A-finalizes-first does NOT flush/clear (B's in-flight span survives)
# ─────────────────────────────────────────────────────────────────────────
def test_sync_B_span_lands_once_no_premature_clear(monkeypatch):      # assert 2/6
    _spy_sync_check(monkeypatch)
    logged = _fake_client(monkeypatch)
    flushes = _spy_flush(monkeypatch)
    sess = _current_session.get()
    model = _instrument_sync(_SyncModel)()

    gA = model.stream("A")
    next(gA)
    sess._deferred_spans.append(_tagged("A", 0))    # simulate span processor for A

    gB = model.stream("B")
    next(gB)
    sess._deferred_spans.append(_tagged("B", 1))    # B's in-flight span

    # Drain A FIRST (non-LIFO). depth is still >0 (B open) → A must NOT flush.
    for _ in gA:
        pass
    assert flushes == []                            # (6.i) A did NOT flush
    tags = {p["tp_tag"] for p in sess._deferred_spans}
    assert tags == {"A", "B"}                       # buffer intact, B survived

    # Drain B (outermost) → flush ONCE, dispatch BOTH.
    for _ in gB:
        pass
    assert flushes == [1]                            # (6.ii) exactly one flush
    logged_tags = sorted(p["tp_tag"] for p in logged)
    assert logged_tags == ["A", "B"]                # both dispatched once each
    assert sess._deferred_spans == []               # (6.iii) buffer empty
    assert in_langchain() is False


def test_async_B_span_lands_once_no_premature_clear(monkeypatch):     # assert 2/6
    async def go():
        _spy_async_check(monkeypatch)
        logged = _fake_client(monkeypatch)
        flushes = _spy_flush(monkeypatch)
        sess = _current_session.get()
        model = _instrument_async(_AsyncModel)()

        agA = model.astream("A")
        await agA.__anext__()
        sess._deferred_spans.append(_tagged("A", 0))

        agB = model.astream("B")
        await agB.__anext__()
        sess._deferred_spans.append(_tagged("B", 1))

        async for _ in agA:                          # A first (non-LIFO)
            pass
        assert flushes == []
        assert {p["tp_tag"] for p in sess._deferred_spans} == {"A", "B"}

        async for _ in agB:                          # outermost → flush once
            pass
        assert flushes == [1]
        assert sorted(p["tp_tag"] for p in logged) == ["A", "B"]
        assert sess._deferred_spans == []

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertions 4, 5 — A stays enforced AND its fold is complete/unbroken
# ─────────────────────────────────────────────────────────────────────────
def test_sync_A_unbroken_by_interleave(monkeypatch):                  # assert 4/5
    checks = _spy_sync_check(monkeypatch)
    comps = []
    real_cap = enforcer._capture_response_composition
    monkeypatch.setattr(enforcer, "_capture_response_composition",
                        lambda provider, resp, *a, **k: comps.append(resp))
    model = _instrument_sync(_SyncModel)()

    gA = model.stream("A")
    got_a = [next(gA)]
    gB = model.stream("B")
    got_b = [c for c in gB]
    got_a += [c for c in gA]

    # A yields ALL its chunks in order
    assert got_a == ["A-0", "A-1", "A-2"]
    assert got_b == ["B-0", "B-1", "B-2"]
    # exactly one check attributable to A (and one to B) — A enforced once
    assert checks == [1, 1]
    # A's fold reflects EVERY A chunk (not truncated)
    assert "A-0A-1A-2" in comps
    assert "B-0B-1B-2" in comps


def test_async_A_unbroken_by_interleave(monkeypatch):                 # assert 4/5
    async def go():
        checks = _spy_async_check(monkeypatch)
        comps = []
        monkeypatch.setattr(enforcer, "_capture_response_composition",
                            lambda provider, resp, *a, **k: comps.append(resp))
        model = _instrument_async(_AsyncModel)()

        agA = model.astream("A")
        got_a = [await agA.__anext__()]
        agB = model.astream("B")
        got_b = [c async for c in agB]
        got_a += [c async for c in agA]

        assert got_a == ["A-0", "A-1", "A-2"]
        assert got_b == ["B-0", "B-1", "B-2"]
        assert checks == [1, 1]
        assert "A-0A-1A-2" in comps
        assert "B-0B-1B-2" in comps

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 7 — guard NOT leaked across the yield (the crux)
# ─────────────────────────────────────────────────────────────────────────
def test_sync_guard_not_leaked_across_yield(monkeypatch):             # assert 7
    _spy_sync_check(monkeypatch)
    model = _instrument_sync(_SyncModel)()
    gA = model.stream("A")
    assert next(gA) == "A-0"
    # customer now holds A suspended at the yield, between chunks
    assert in_langchain() is False          # pre-fix: True (leaked)
    next(gA)
    assert in_langchain() is False
    for _ in gA:
        pass
    assert in_langchain() is False


def test_async_guard_not_leaked_across_yield(monkeypatch):            # assert 7
    async def go():
        _spy_async_check(monkeypatch)
        model = _instrument_async(_AsyncModel)()
        agA = model.astream("A")
        assert await agA.__anext__() == "A-0"
        assert in_langchain() is False      # pre-fix: True
        await agA.__anext__()
        assert in_langchain() is False
        async for _ in agA:
            pass
        assert in_langchain() is False

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 15 — NO CASCADE: after BOTH drain, defer==baseline + depth==0, and a
# SUBSEQUENT stream is still enforced + metered. Both non-LIFO drain orders.
# ─────────────────────────────────────────────────────────────────────────
def _sync_cascade_body(monkeypatch, a_first):
    checks = _spy_sync_check(monkeypatch)
    logged = _fake_client(monkeypatch)
    flushes = _spy_flush(monkeypatch)
    sess = _current_session.get()
    model = _instrument_sync(_SyncModel)()

    gA = model.stream("A"); next(gA)
    sess._deferred_spans.append(_tagged("A", 0))
    gB = model.stream("B"); next(gB)
    sess._deferred_spans.append(_tagged("B", 1))

    first, second = (gA, gB) if a_first else (gB, gA)
    for _ in first:
        pass
    for _ in second:
        pass

    # cascade guards: state returned to baseline
    assert sess._defer_telemetry is False
    assert getattr(sess, "_lc_stream_depth", 0) == 0
    assert sess._deferred_spans == []
    assert flushes == [1]
    assert sorted(p["tp_tag"] for p in logged) == ["A", "B"]

    # subsequent stream still enforced + metered + its own span flushed once
    checks.clear(); logged.clear(); flushes.clear()
    gC = model.stream("C")
    assert checks == [1]                    # C ENFORCED
    assert _is_sync_guard(gC)               # C METERED (guarded)
    next(gC)
    sess._deferred_spans.append(_tagged("C", 2))
    for _ in gC:
        pass
    assert flushes == [1]                   # C's span flushed (no leftover defer)
    assert [p["tp_tag"] for p in logged] == ["C"]
    assert sess._deferred_spans == []


def test_sync_no_cascade_A_first(monkeypatch):                        # assert 15
    _sync_cascade_body(monkeypatch, a_first=True)


def test_sync_no_cascade_B_first(monkeypatch):                        # assert 15
    _sync_cascade_body(monkeypatch, a_first=False)


def _async_cascade_body(monkeypatch, a_first):
    async def go():
        checks = _spy_async_check(monkeypatch)
        logged = _fake_client(monkeypatch)
        flushes = _spy_flush(monkeypatch)
        sess = _current_session.get()
        model = _instrument_async(_AsyncModel)()

        agA = model.astream("A"); await agA.__anext__()
        sess._deferred_spans.append(_tagged("A", 0))
        agB = model.astream("B"); await agB.__anext__()
        sess._deferred_spans.append(_tagged("B", 1))

        first, second = (agA, agB) if a_first else (agB, agA)
        async for _ in first:
            pass
        async for _ in second:
            pass

        assert sess._defer_telemetry is False
        assert getattr(sess, "_lc_stream_depth", 0) == 0
        assert sess._deferred_spans == []
        assert flushes == [1]
        assert sorted(p["tp_tag"] for p in logged) == ["A", "B"]

        checks.clear(); logged.clear(); flushes.clear()
        agC = model.astream("C")
        await agC.__anext__()
        assert checks == [1]
        sess._deferred_spans.append(_tagged("C", 2))
        async for _ in agC:
            pass
        assert flushes == [1]
        assert [p["tp_tag"] for p in logged] == ["C"]
        assert sess._deferred_spans == []

    _run(go())


def test_async_no_cascade_A_first(monkeypatch):                       # assert 15
    _async_cascade_body(monkeypatch, a_first=True)


def test_async_no_cascade_B_first(monkeypatch):                       # assert 15
    _async_cascade_body(monkeypatch, a_first=False)


# ─────────────────────────────────────────────────────────────────────────
# Assertion 8 — HOT-AREA: mid-stream provider error re-raised VERBATIM,
# guard restored, subsequent stream enforced
# ─────────────────────────────────────────────────────────────────────────
def test_sync_midstream_error_verbatim_then_enforces(monkeypatch):    # assert 8
    checks = _spy_sync_check(monkeypatch)
    model = _instrument_sync(_SyncModelErr)()
    gA = model.stream("A")
    assert next(gA) == "A-0"
    caught = None
    try:
        next(gA)
    except RuntimeError as e:
        caught = e
    assert type(caught) is RuntimeError
    assert str(caught) == "boom-A"          # verbatim (not wrapped/masked)
    assert in_langchain() is False

    checks.clear()
    ok = _instrument_sync(_SyncModel)()
    gB = ok.stream("B")
    assert checks == [1]                    # subsequent stream enforced
    for _ in gB:
        pass


def test_async_midstream_error_verbatim_then_enforces(monkeypatch):   # assert 8
    async def go():
        checks = _spy_async_check(monkeypatch)
        model = _instrument_async(_AsyncModelErr)()
        agA = model.astream("A")
        assert await agA.__anext__() == "A-0"
        caught = None
        try:
            await agA.__anext__()
        except RuntimeError as e:
            caught = e
        assert type(caught) is RuntimeError
        assert str(caught) == "boom-A"
        assert in_langchain() is False

        checks.clear()
        ok = _instrument_async(_AsyncModel)()
        agB = ok.astream("B")
        await agB.__anext__()
        assert checks == [1]

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 9 — HOT-AREA: early abandonment (break, no GC reliance) + explicit
# close/aclose; later stream enforced
# ─────────────────────────────────────────────────────────────────────────
def test_sync_early_abandonment_break_and_close(monkeypatch):         # assert 9
    checks = _spy_sync_check(monkeypatch)
    model = _instrument_sync(_SyncModel)()

    gA = model.stream("A")
    for c in gA:                            # break after ONE chunk, drop the gen
        assert c == "A-0"
        break
    assert in_langchain() is False          # already restored BEFORE the yield

    checks.clear()
    gB = model.stream("B")
    assert checks == [1]                    # C/B enforced with NO GC reliance
    next(gB)
    gB.close()                              # GeneratorExit → outer finally runs
    assert in_langchain() is False


def test_async_early_abandonment_break_and_aclose(monkeypatch):       # assert 9
    async def go():
        checks = _spy_async_check(monkeypatch)
        model = _instrument_async(_AsyncModel)()

        agA = model.astream("A")
        async for c in agA:
            assert c == "A-0"
            break
        assert in_langchain() is False      # already False BEFORE any close/GC
        await agA.aclose()                  # explicit (no GC reliance) — silences warning
        assert in_langchain() is False

        checks.clear()
        agB = model.astream("B")
        await agB.__anext__()
        assert checks == [1]
        await agB.aclose()                  # GeneratorExit → finally
        assert in_langchain() is False

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 10: save/restore is fail-open. _in_langchain.set raising
# (covers entry set(True) AND restore set(prev)) must NOT break the customer.
# ─────────────────────────────────────────────────────────────────────────
class _RaisingSetVar:
    """Wraps the real ContextVar but makes `.set()` ALWAYS raise (covers both the
    entry set(True) and the restore set(prev)). `.get()` still works."""

    def __init__(self, real):
        self._real = real

    def get(self):
        return self._real.get()

    def set(self, value):
        raise RuntimeError("set boom (contextvar set failed)")


def test_sync_failopen_when_set_raises(monkeypatch):                  # assert 10
    _spy_sync_check(monkeypatch)
    monkeypatch.setattr(enforcer, "_in_langchain", _RaisingSetVar(_in_langchain))
    model = _instrument_sync(_SyncModel)()
    gA = model.stream("A")
    got = [c for c in gA]                   # NO SDK error escapes into the loop
    assert got == ["A-0", "A-1", "A-2"]     # every chunk delivered verbatim


def test_async_failopen_when_set_raises(monkeypatch):                 # assert 10
    async def go():
        _spy_async_check(monkeypatch)
        monkeypatch.setattr(enforcer, "_in_langchain", _RaisingSetVar(_in_langchain))
        model = _instrument_async(_AsyncModel)()
        agA = model.astream("A")
        got = [c async for c in agA]
        assert got == ["A-0", "A-1", "A-2"]

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 12 — no-interleave baseline: single stream unchanged (happy path)
# ─────────────────────────────────────────────────────────────────────────
def test_sync_no_interleave_baseline(monkeypatch):                    # assert 12
    checks = _spy_sync_check(monkeypatch)
    logged = _fake_client(monkeypatch)
    flushes = _spy_flush(monkeypatch)
    sess = _current_session.get()
    model = _instrument_sync(_SyncModel)()

    gA = model.stream("A")
    assert checks == [1]                    # exactly one pre-flight check
    assert _is_sync_guard(gA)
    out = []
    for c in gA:
        out.append(c)
        sess._deferred_spans.append(_tagged("A", len(out) - 1)) if len(out) == 1 else None
    assert out == ["A-0", "A-1", "A-2"]     # all chunks, in order
    assert flushes == [1]                   # finalize flushed once
    assert [p["tp_tag"] for p in logged] == ["A"]
    assert sess._deferred_spans == []
    assert sess._defer_telemetry is False   # defer restored
    assert getattr(sess, "_lc_stream_depth", 0) == 0
    assert in_langchain() is False


def test_async_no_interleave_baseline(monkeypatch):                   # assert 12
    async def go():
        checks = _spy_async_check(monkeypatch)
        logged = _fake_client(monkeypatch)
        flushes = _spy_flush(monkeypatch)
        sess = _current_session.get()
        model = _instrument_async(_AsyncModel)()

        agA = model.astream("A")
        assert _is_async_guard(agA)
        out = []
        async for c in agA:
            out.append(c)
            if len(out) == 1:
                sess._deferred_spans.append(_tagged("A", 0))
        assert checks == [1]
        assert out == ["A-0", "A-1", "A-2"]
        assert flushes == [1]
        assert [p["tp_tag"] for p in logged] == ["A"]
        assert sess._deferred_spans == []
        assert sess._defer_telemetry is False
        assert getattr(sess, "_lc_stream_depth", 0) == 0
        assert in_langchain() is False

    _run(go())
