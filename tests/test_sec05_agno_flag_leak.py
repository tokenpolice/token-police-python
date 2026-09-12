"""Regression suite — the Agno `_in_agno` re-entrancy guard must be
reset on a mid-stream provider error / iterator teardown, not only on clean
exhaustion.

Finding (MIRROR of the Golden Rule the risk is UNDER-enforcement, not a
throw): the Agno wrappers set `_in_agno=True` across stream iteration so nested
provider enforcer wrappers short-circuit (a single pre-flight `/check` at the
Agno boundary). The reset (`_finalize()` -> `_in_agno.set(False)`) fired ONLY on
`StopIteration` / `StopAsyncIteration`. A provider error raised MID-stream (or a
`GeneratorExit`/`CancelledError` teardown) escaped `__next__`/`__anext__` without
resetting the flag, so a subsequent `Agent.run`/`Agent.arun` in the same
context/task saw `in_agno()==True` at the top guard (enforcer.py:5882-5884 /
:5928-5930) and returned `original(...)` directly — skipping the pre-flight
`/check` (under-enforcement) and framework accounting.

The fix adds a defensively-total `except BaseException: try: self._finalize()
except BaseException: pass; raise` clause ALONGSIDE the existing `except Stop*`
clause in both iterators, and widens `_AgnoLazyAsyncStreamIter._start()`'s
construction guard from `except Exception:` to `except BaseException:` so a
startup cancellation after the SET also resets. The reset fires ONLY on the
exception/stop path — NEVER on a normal per-event return (a `finally` would
destroy cross-event enforcement).

These tests are written to FAIL on pre-fix code (flag leaks True) and PASS
post-fix. The clean-exhaustion / no-mid-event-reset tests are the anti-over-fix
guards: they prove the fix did not turn into a `finally` that breaks
cross-event enforcement.
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
from token_police.context import _in_agno, in_agno
from token_police.enforcer import (
    _AgnoSyncStreamIter,
    _AgnoLazyAsyncStreamIter,
    _make_agno_run_wrapper,
    _make_agno_arun_wrapper,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(autouse=True)
def _reset_flag():
    """Reset the guard around every test so nothing leaks between cases."""
    _in_agno.set(False)
    yield
    _in_agno.set(False)


class _SyncInner:
    """Plain sync iterator (NOT a generator — mirrors Agno's RunOutput stream
    proxy). Yields ``event-{n}`` until ``raise_at``, then raises ``exc`` (or
    ``StopIteration`` if ``exc is None``)."""

    def __init__(self, exc=None, raise_at=1):
        self._exc = exc
        self._raise_at = raise_at
        self._n = 0

    def __iter__(self):
        return self

    def __next__(self):
        self._n += 1
        if self._n < self._raise_at:
            return f"event-{self._n}"
        if self._exc is None:
            raise StopIteration
        raise self._exc


class _AsyncInner:
    """Plain async iterator standing in for Agno's `_arun_stream` async-gen."""

    def __init__(self, exc=None, raise_at=1):
        self._exc = exc
        self._raise_at = raise_at
        self._n = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        self._n += 1
        if self._n < self._raise_at:
            return f"event-{self._n}"
        if self._exc is None:
            raise StopAsyncIteration
        raise self._exc


def _spy_sync_check(monkeypatch):
    """Spy `_run_sync_check`; returns a list appended to on each invocation."""
    calls = []

    def spy(*a, **k):
        calls.append(1)

    monkeypatch.setattr(enforcer, "_run_sync_check", spy)
    return calls


def _spy_async_check(monkeypatch):
    calls = []

    async def spy(*a, **k):
        calls.append(1)

    monkeypatch.setattr(enforcer, "_run_async_check", spy)
    return calls


class _ResetRaisingVar:
    """Wraps the real ContextVar but makes `.set(False)` (the reset) raise —
    exercising real production behavior: the failure lands INSIDE `_finalize`'s
    existing try/except. `.set(True)` and `.get()` still work so `_start()` is
    unaffected."""

    def __init__(self, real):
        self._real = real

    def get(self):
        return self._real.get()

    def set(self, value):
        if value is False:
            raise RuntimeError("reset boom (contextvar set failed)")
        return self._real.set(value)


# ─────────────────────────────────────────────────────────────────────────
# Assertions 1 & 3 — mid-stream error resets AND a subsequent call enforces
# ─────────────────────────────────────────────────────────────────────────
def test_sync_midstream_resets_and_enforces(monkeypatch):          # assert 1, 3
    _in_agno.set(True)  # simulate: we are inside an Agno stream
    it = _AgnoSyncStreamIter(_SyncInner(RuntimeError("boom"), raise_at=2), agent=None)
    with pytest.raises(RuntimeError):
        for _ in it:
            pass

    # (assertion 1) flag reset on the mid-stream error path
    assert in_agno() is False

    # (assertion 3) the ENFORCE DECISION is restored — the actual wrapper for a
    # SUBSEQUENT run runs the pre-flight /check (spy invoked). Pre-fix the flag
    # leaked True, the top guard short-circuited, and the spy was NOT called.
    calls = _spy_sync_check(monkeypatch)
    wrapper = _make_agno_run_wrapper(lambda self, *a, **k: "result")
    res = wrapper(object())
    assert res == "result"
    assert calls == [1]
    assert in_agno() is False  # wrapper reset its own guard on the non-stream path


def test_async_midstream_resets_and_enforces(monkeypatch):         # assert 2, 3
    async def go():
        _in_agno.set(False)
        calls = _spy_async_check(monkeypatch)

        it = _AgnoLazyAsyncStreamIter(
            original=lambda agent, *a, **k: _AsyncInner(RuntimeError("boom"), raise_at=2),
            agent=None, args=(), kwargs={},
        )
        with pytest.raises(RuntimeError):
            async for _ in it:
                pass

        # (assertion 2) flag reset in the same task after the mid-stream error
        assert in_agno() is False

        # (assertion 3 async) subsequent arun enforces via the real wrapper
        calls.clear()

        async def orig_ok(self, *a, **k):
            return "result"

        wrapper = _make_agno_arun_wrapper(orig_ok)
        res = await wrapper(object())  # non-streaming -> _coro -> _run_async_check
        assert res == "result"
        assert calls == [1]
        assert in_agno() is False

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 4 — provider exception re-raised VERBATIM (object identity)
# ─────────────────────────────────────────────────────────────────────────
def test_sync_provider_error_verbatim():                           # assertion 4
    err = RuntimeError("exact-sentinel")
    _in_agno.set(True)
    it = _AgnoSyncStreamIter(_SyncInner(err, raise_at=2), agent=None)
    caught = None
    try:
        for _ in it:
            pass
    except RuntimeError as e:
        caught = e
    assert caught is err          # same object, not swallowed/wrapped/chained
    assert type(caught) is RuntimeError
    assert in_agno() is False


def test_async_provider_error_verbatim(monkeypatch):               # assertion 4
    err = RuntimeError("exact-sentinel-async")

    async def go():
        _in_agno.set(False)
        _spy_async_check(monkeypatch)
        it = _AgnoLazyAsyncStreamIter(
            original=lambda agent, *a, **k: _AsyncInner(err, raise_at=2),
            agent=None, args=(), kwargs={},
        )
        caught = None
        try:
            async for _ in it:
                pass
        except RuntimeError as e:
            caught = e
        assert caught is err
        assert type(caught) is RuntimeError
        assert in_agno() is False

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 5 — reset stays TOTAL: original error surfaces even if reset fails
# ─────────────────────────────────────────────────────────────────────────
def test_sync_reset_total_when_set_raises(monkeypatch):            # assertion 5
    err = RuntimeError("original-provider-error")
    monkeypatch.setattr(enforcer, "_in_agno", _ResetRaisingVar(_in_agno))
    it = _AgnoSyncStreamIter(_SyncInner(err, raise_at=2), agent=None)
    caught = None
    try:
        for _ in it:
            pass
    except RuntimeError as e:
        caught = e
    # The reset failing must NOT replace the ORIGINAL provider error.
    assert caught is err
    assert "reset boom" not in str(caught)


def test_async_reset_total_when_set_raises(monkeypatch):           # assertion 5
    err = RuntimeError("original-provider-error-async")

    async def go():
        _spy_async_check(monkeypatch)
        monkeypatch.setattr(enforcer, "_in_agno", _ResetRaisingVar(_in_agno))
        it = _AgnoLazyAsyncStreamIter(
            original=lambda agent, *a, **k: _AsyncInner(err, raise_at=2),
            agent=None, args=(), kwargs={},
        )
        caught = None
        try:
            async for _ in it:
                pass
        except RuntimeError as e:
            caught = e
        assert caught is err
        assert "reset boom" not in str(caught)

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertions 6 & 7 — deterministic thrown-teardown early abandonment
# ─────────────────────────────────────────────────────────────────────────
def test_sync_early_abandonment_thrown(monkeypatch):               # assertion 6
    # Deterministically THROW GeneratorExit into __next__ (by raising it from
    # the consumed inner stream) — travels the fix's `except BaseException`.
    _in_agno.set(True)
    it = _AgnoSyncStreamIter(_SyncInner(GeneratorExit(), raise_at=2), agent=None)
    assert it.__next__() == "event-1"
    with pytest.raises(GeneratorExit):
        it.__next__()
    assert it._done is True          # _finalize ran
    assert in_agno() is False

    # a subsequent wrapper call enforces
    calls = _spy_sync_check(monkeypatch)
    wrapper = _make_agno_run_wrapper(lambda self, *a, **k: "ok")
    wrapper(object())
    assert calls == [1]


def test_async_early_abandonment_thrown(monkeypatch):              # assertion 7
    async def go():
        _in_agno.set(False)
        calls = _spy_async_check(monkeypatch)
        it = _AgnoLazyAsyncStreamIter(
            original=lambda agent, *a, **k: _AsyncInner(asyncio.CancelledError(), raise_at=2),
            agent=None, args=(), kwargs={},
        )
        assert await it.__anext__() == "event-1"
        with pytest.raises(asyncio.CancelledError):
            await it.__anext__()
        assert it._done is True
        assert in_agno() is False

        calls.clear()

        async def orig_ok(self, *a, **k):
            return "ok"

        wrapper = _make_agno_arun_wrapper(orig_ok)
        await wrapper(object())
        assert calls == [1]

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 8 — SITE C: startup-cancellation during _start() resets the flag
# ─────────────────────────────────────────────────────────────────────────
def test_async_start_cancellation_resets(monkeypatch):             # assertion 8
    async def go():
        _in_agno.set(False)
        calls = _spy_async_check(monkeypatch)

        def orig_raises(agent, *a, **k):
            # Cancellation during inner-iterator construction, AFTER
            # _in_agno.set(True) at enforcer.py:5838.
            raise asyncio.CancelledError()

        it = _AgnoLazyAsyncStreamIter(
            original=orig_raises, agent=None, args=(), kwargs={},
        )
        with pytest.raises(asyncio.CancelledError):
            await it.__anext__()

        # Pre-fix::5847 `except Exception:` does NOT catch CancelledError ->
        # flag leaks True. Post-fix (`except BaseException:`) it is reset.
        assert in_agno() is False

        # subsequent arun enforces
        calls.clear()

        async def orig_ok(self, *a, **k):
            return "ok"

        wrapper = _make_agno_arun_wrapper(orig_ok)
        await wrapper(object())
        assert calls == [1]

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 11 — clean exhaustion resets; per-event return does NOT (anti-over-fix)
# ─────────────────────────────────────────────────────────────────────────
def test_sync_clean_exhaustion_and_no_midevent_reset():            # assertion 11
    _in_agno.set(True)
    it = _AgnoSyncStreamIter(_SyncInner(exc=None, raise_at=3), agent=None)
    assert it.__next__() == "event-1"
    assert in_agno() is True         # NOT reset mid-stream (cross-event guard held)
    assert it.__next__() == "event-2"
    assert in_agno() is True
    with pytest.raises(StopIteration):
        it.__next__()
    assert in_agno() is False        # reset on clean exhaustion


def test_async_clean_exhaustion_and_no_midevent_reset(monkeypatch):  # assertion 11
    async def go():
        _in_agno.set(False)
        _spy_async_check(monkeypatch)
        it = _AgnoLazyAsyncStreamIter(
            original=lambda agent, *a, **k: _AsyncInner(exc=None, raise_at=3),
            agent=None, args=(), kwargs={},
        )
        assert await it.__anext__() == "event-1"
        assert in_agno() is True
        assert await it.__anext__() == "event-2"
        assert in_agno() is True
        with pytest.raises(StopAsyncIteration):
            await it.__anext__()
        assert in_agno() is False

    _run(go())


# ─────────────────────────────────────────────────────────────────────────
# Assertion 12 — idempotent _finalize under error-then-del
# ─────────────────────────────────────────────────────────────────────────
def test_idempotent_finalize_error_then_del():                     # assertion 12
    err = RuntimeError("boom")
    _in_agno.set(True)
    it = _AgnoSyncStreamIter(_SyncInner(err, raise_at=2), agent=None)
    assert it.__next__() == "event-1"
    with pytest.raises(RuntimeError):
        it.__next__()
    assert it._done is True
    assert in_agno() is False
    # __del__ backstop after finalize already ran: no double-work, no error
    it.__del__()
    assert in_agno() is False
