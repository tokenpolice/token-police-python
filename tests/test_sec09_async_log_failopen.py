"""Regression suite — TokenPolice.log (async) must fail open.

Finding: the async ``log()`` schedules its background send with
``asyncio.create_task(_send())`` outside any try/except (client.py scheduling
tail, formerly:555-557). ``create_task`` calls ``asyncio.get_running_loop()``,
which raises ``RuntimeError: no running event loop`` (coroutine driven without a
running loop — manually-stepped coro / stopped-or-closed thread loop) or
``Event loop is closed`` (async teardown/atexit) straight into the awaiter of the
documented fire-and-forget ``log()`` API — a Golden-Rule violation.

The fix mirrors the SHIPPED ``log_sync`` guard (client.py:345-352) and the
in-class ``check()`` idiom (client.py:438-453): wrap the scheduling tail in
``except Exception`` (so ``CancelledError``/``KeyboardInterrupt``/``SystemExit``
still propagate), swallow the failure, ``logger.warning`` ONLY when
``self.log_errors``, and preserve ``log()``'s fire-and-forget ``None`` return.

STRUCTURE NOTE (assertion 2, vacuity trap): ``log()`` is a
coroutine, so the no-loop case is driven by manually stepping it
(``coro.send(None)``). Post-fix the guard swallows and the coroutine returns
``None`` → ``coro.send(None)`` raises ``StopIteration`` (= SUCCESS). Pre-fix it
raises ``RuntimeError`` (= the defect). Since ``StopIteration`` ⊂ ``Exception``, a
naive ``except Exception: pass`` would swallow the pre-fix ``RuntimeError`` too and
vacuously pass on broken code. The ONLY pass condition is ``except StopIteration``;
anything else trips ``pytest.fail`` — so these tests FAIL pre-fix and PASS post-fix.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the local SDK importable without a prior editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police.client import TokenPolice
import token_police.client as client_mod


# Distinctive model name + token counts we scan for in captured logs to prove
# INV-2 (no prompt/completion/token/model content leaks into the warning).
SENTINEL_MODEL = "gpt-4o-secret-model-XYZZY"
SENTINEL_INPUT = 424242
SENTINEL_OUTPUT = 313131


def _make_client(log_errors: bool = False) -> TokenPolice:
    """Construct a bare client directly (no init()/OTel/global-state side
    effects). base_url points at an unroutable host so no real network I/O can
    happen even if a send runs."""
    return TokenPolice(
        api_key="tp_sk_test",
        base_url="http://127.0.0.1:1",
        timeout=0.1,
        deployment="serverless",  # no SSE thread
        log_errors=log_errors,
    )


def _step_no_running_loop(c: TokenPolice):
    """Drive the log() coroutine to its scheduling tail with NO running loop.
    Returns the StopIteration value (the coroutine's return, expected None) on the
    post-fix success path; calls pytest.fail on anything other than StopIteration
    (which includes the pre-fix RuntimeError). Enforces assertion 2's structure."""
    coro = c.log(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
    try:
        coro.send(None)
    except StopIteration as si:
        return si.value  # coroutine completed cleanly (fire-and-forget None)
    except BaseException as e:  # noqa: BLE001 — deliberately catch-all to expose defect
        coro.close()
        pytest.fail(
            f"async log() scheduling tail raised {type(e).__name__}: {e}"
        )
    else:
        coro.close()
        pytest.fail("async log() unexpectedly suspended instead of completing")


# ── Assertion 2 + 5: no running loop → no raise, returns None ────────────────
def test_async_log_no_running_loop_does_not_raise():
    """Manual-step log() with no running loop. Pre-fix: coro.send raises
    RuntimeError('no running event loop') → pytest.fail. Post-fix: StopIteration
    with value None → PASS."""
    c = _make_client()
    value = _step_no_running_loop(c)
    assert value is None  # fire-and-forget: fail-open return is None (no marker)


# ── Assertion 4a / stopped/closed thread loop → no raise ──────────────
def test_async_log_closed_loop_does_not_raise():
    """Fire log() from a worker thread whose loop already ran and closed
    (realistic teardown). get_running_loop() still raises; the guard must swallow.
    Pre-fix: RuntimeError escapes → recorded as error → FAIL. Post-fix: PASS."""
    c = _make_client()
    result: dict = {}

    def worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()  # thread has a current-but-closed loop, none running
        coro = c.log(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
        try:
            coro.send(None)
        except StopIteration as si:
            result["ok"] = True
            result["value"] = si.value
        except BaseException as e:  # noqa: BLE001
            coro.close()
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert result.get("ok") is True, (
        f"closed-loop async log() raised instead of failing open: {result.get('err')}"
    )
    assert result.get("value") is None


# ── Assertion 4b / ANY exception from scheduling is swallowed ─────────
def test_async_log_generic_schedule_error_does_not_raise(monkeypatch):
    """Monkeypatch asyncio.create_task to raise a generic (non-RuntimeError)
    Exception; under a real running loop log() must still return None, proving the
    guard swallows any internal failure of the scheduling tail — not only
    RuntimeError. Pre-fix: ValueError propagates out of log() → asyncio.run raises
    → FAIL. Post-fix: PASS."""
    c = _make_client()
    monkeypatch.setattr(
        client_mod.asyncio,
        "create_task",
        MagicMock(side_effect=ValueError("boom-not-runtimeerror")),
    )

    async def scenario():
        return await c.log(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)

    result = asyncio.run(scenario())
    assert result is None


# ── Assertion 3: healthy-loop path STILL schedules AND runs _send end-to-end ──
def test_async_log_alive_loop_actually_invokes_post():
    """Anti-over-fix guard. Under a real running loop with a fake async client
    injected, await log() then await flush() must (a) track a task in
    _pending_tasks during the call and (b) actually invoke the async
    post('/v1/guard/log', ...) end-to-end. A vacuous fix that catches-and-drops
    create_task passes (a) never but fails (b) — this proves the healthy path is
    NOT turned into a no-op."""
    c = _make_client()
    calls: list = []

    class FakeAsyncClient:
        async def post(self, path, json=None):
            calls.append(path)
            return MagicMock(status_code=200)

    # Seed the cached instance so _get_async_client() returns the fake (no I/O).
    c._async_client_instance = FakeAsyncClient()

    async def scenario():
        # (5) fire-and-forget returns None on the healthy path.
        assert (await c.log(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)) is None
        # (a) a real task was tracked during the call (not yet run — log() does
        # not await _send).
        assert len(c._pending_tasks) >= 1, "healthy path scheduled no task"
        # Drain the scheduled task so _send runs to completion.
        await c.flush()

    asyncio.run(scenario())

    # (b) the fake async transport was actually invoked end-to-end.
    assert "/v1/guard/log" in calls, "async post('/v1/guard/log') was never invoked"


# ── flush() must drain the thread pool, not only the async tasks ─────────────
def test_async_flush_drains_thread_pool_futures():
    """THE gap. Auto-instrumented telemetry is submitted via log_sync() to the
    thread pool (tracked in _pending_futures), NOT as async tasks. An async
    handler that awaits flush() before the container sleeps must therefore await
    those threads too. Submit one telemetry row through the thread-pool path,
    then await flush(); the underlying sync POST must have been invoked.

    FAILS pre-fix (flush() drained only _pending_tasks, so the still-in-flight
    thread-pool future was never awaited and the POST could be lost)."""
    import threading

    c = _make_client()
    calls: list = []
    release = threading.Event()

    def _blocking_post(*args, **kwargs):
        # Hold the background _send open so the future is genuinely in flight
        # when flush() is entered — proving flush() waits for it, not that it
        # merely happened to finish first.
        release.wait(timeout=5)
        calls.append(args[0] if args else None)
        return MagicMock(status_code=200)

    c._sync_client.post = MagicMock(side_effect=_blocking_post)

    # Thread-pool submission (the log_sync path used by auto-instrumentation).
    c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
    with c._lock:
        assert len(c._pending_futures) >= 1, "log_sync enqueued no future"

    async def scenario():
        # Release the blocked send from a worker thread just after flush() begins
        # waiting, so the future is still pending as flush() is entered.
        threading.Timer(0.1, release.set).start()
        await c.flush()

    asyncio.run(scenario())

    assert "/v1/guard/log" in calls, (
        "async flush() did not drain the thread-pool telemetry future "
        "(the auto-instrumented spend row was left in flight)"
    )
    c.close_sync()


def test_async_flush_drains_both_async_task_and_thread_future():
    """Mixed in-flight work: one async log() task AND one log_sync() thread-pool
    future. A single await flush() must drain BOTH end-to-end."""
    import threading

    c = _make_client()
    async_calls: list = []
    sync_calls: list = []
    release = threading.Event()

    class FakeAsyncClient:
        async def post(self, path, json=None):
            async_calls.append(path)
            return MagicMock(status_code=200)

    c._async_client_instance = FakeAsyncClient()

    def _blocking_post(*args, **kwargs):
        release.wait(timeout=5)
        sync_calls.append(args[0] if args else None)
        return MagicMock(status_code=200)

    c._sync_client.post = MagicMock(side_effect=_blocking_post)

    # Thread-pool future in flight.
    c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)

    async def scenario():
        # Async task in flight.
        await c.log(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
        assert len(c._pending_tasks) >= 1
        threading.Timer(0.1, release.set).start()
        await c.flush()

    asyncio.run(scenario())

    assert "/v1/guard/log" in async_calls, "async log() task not drained by flush()"
    assert "/v1/guard/log" in sync_calls, "thread-pool future not drained by flush()"
    c.close_sync()


def test_async_flush_never_raises_when_flush_sync_fails(monkeypatch):
    """. flush() is called from the serverless decorator's finally around the
    customer handler, so it must never raise. If flush_sync() itself blows up, the
    added thread-pool drain must swallow it and flush() must complete cleanly."""
    c = _make_client()

    def _boom():
        raise RuntimeError("flush_sync exploded")

    monkeypatch.setattr(c, "flush_sync", _boom)

    async def scenario():
        # Must return without raising.
        return await c.flush()

    result = asyncio.run(scenario())
    assert result is None
    c.close_sync()


# ── Assertion 6: warning gating matches guard / check_sync ────────────
def test_async_log_schedule_failure_warns_only_when_log_errors_true(caplog):
    c = _make_client(log_errors=True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="token_police"):
        _step_no_running_loop(c)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"


def test_async_log_schedule_failure_silent_when_log_errors_false(caplog):
    c = _make_client(log_errors=False)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="token_police"):
        _step_no_running_loop(c)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 0, f"expected no warning, got {len(warnings)}"


# ── Assertion 7 / INV-2: warning carries no prompt/completion/token/model content ──
def test_async_log_schedule_failure_warning_has_no_content(caplog):
    c = _make_client(log_errors=True)
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="token_police"):
        coro = c.log(
            model=SENTINEL_MODEL,
            input_tokens=SENTINEL_INPUT,
            output_tokens=SENTINEL_OUTPUT,
        )
        try:
            coro.send(None)
        except StopIteration:
            pass
        except BaseException as e:  # noqa: BLE001
            coro.close()
            pytest.fail(f"async log() raised {type(e).__name__}: {e}")
    text = " ".join(r.getMessage() for r in caplog.records)
    assert SENTINEL_MODEL not in text, "model name leaked into warning (INV-2)"
    assert str(SENTINEL_INPUT) not in text, "input token count leaked into warning (INV-2)"
    assert str(SENTINEL_OUTPUT) not in text, "output token count leaked into warning (INV-2)"
