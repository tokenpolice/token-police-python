"""Regression suite — TokenPolice.log_sync must fail open.

Finding: ``TokenPoliceClient.log_sync`` calls ``self._executor.submit(_send)``
outside any try/except (``client.py`` submit region). After ``close_sync()`` (or
the registered ``atexit`` hook) the ThreadPoolExecutor is shut down, so
``submit()`` raises ``RuntimeError: cannot schedule new futures after shutdown``
straight into customer code — a Golden-Rule violation.

The fix mirrors the in-class fail-open idiom of ``check_sync`` (client.py
:201-217): guard the submit, swallow any failure, ``logger.warning`` ONLY when
``self.log_errors``, and preserve ``log_sync``'s normal-path return of ``None``.

These tests are written to FAIL on pre-fix code (the unguarded ``submit`` raises)
and PASS post-fix. The executor-alive test is the anti-over-fix guard: it proves
the guard did NOT silently turn healthy calls into no-ops.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the local SDK importable without a prior editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police.client import TokenPolice


# A distinctive model name + token counts we can scan for in captured logs to
# prove INV-2 (no prompt/completion/token/model content leaks into the warning).
SENTINEL_MODEL = "gpt-4o-secret-model-XYZZY"
SENTINEL_INPUT = 424242
SENTINEL_OUTPUT = 313131


def _make_client(log_errors: bool = False) -> TokenPolice:
    """Construct a bare client directly (no init()/OTel/global-state side
    effects). base_url points at an unroutable host so no real network I/O can
    happen even if a submit runs."""
    return TokenPolice(
        api_key="tp_sk_test",
        base_url="http://127.0.0.1:1",
        timeout=0.1,
        deployment="serverless",  # no SSE thread
        log_errors=log_errors,
    )


# ── Assertion 2 + 5: post-close fail-open, returns None, never raises ────────
def test_log_sync_after_close_does_not_raise():
    """After close_sync() the executor is shut down; log_sync must NOT raise and
    must return None (fail-open). FAILS pre-fix (RuntimeError escapes)."""
    c = _make_client()
    c.close_sync()
    # Must not raise anything.
    result = c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
    assert result is None


# ── Assertion 5: return-value parity — None on BOTH normal and fail-open ─────
def test_log_sync_returns_none_before_and_after_close():
    c = _make_client()
    # Normal path (executor alive) — returns None.
    assert c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1) is None
    c.close_sync()
    # Fail-open path (executor shut down) — also returns None, no marker.
    assert c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1) is None


# ── Assertion 3: executor-alive path STILL submits AND executes real work ────
def test_log_sync_alive_executor_actually_invokes_post():
    """Anti-over-fix guard. On a live client with a fake injected for the exact
    attribute _send calls (client._sync_client.post at client.py:329), log_sync
    then flush_sync must (a) add a future to _pending_futures during the call and
    (b) actually invoke _sync_client.post("/v1/guard/log", ...) end-to-end.

    A vacuous fix that enqueues a no-op future passes (a) but fails (b)."""
    import threading

    c = _make_client()
    release = threading.Event()

    def _blocking_post(*args, **kwargs):
        # Block inside the background _send so the tracked future is still
        # pending when we inspect _pending_futures (otherwise it may complete
        # and be discarded by the _cleanup callback before we look).
        release.wait(timeout=5)
        return MagicMock(status_code=200)

    fake_post = MagicMock(side_effect=_blocking_post)
    c._sync_client.post = fake_post

    c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)

    # (a) a real future was tracked during the call (still pending, blocked).
    with c._lock:
        pending_count = len(c._pending_futures)
    assert pending_count >= 1, "executor-alive path enqueued no future"

    # Release the blocked _send and drain the executor so it runs to completion.
    release.set()
    c.flush_sync()

    # (b) the fake HTTP transport was actually invoked end-to-end.
    assert fake_post.call_count >= 1, "_sync_client.post was never invoked"
    # It was called against the log endpoint.
    called_paths = [call.args[0] for call in fake_post.call_args_list if call.args]
    assert "/v1/guard/log" in called_paths

    c.close_sync()


# ── Assertion 4 / ANY exception from submit is swallowed (not just RuntimeError) ──
def test_log_sync_swallows_non_runtimeerror_from_submit():
    """Monkeypatch _executor.submit to raise a generic (non-RuntimeError)
    Exception; log_sync must still return None, proving the guard swallows any
    internal failure of the submit path. FAILS pre-fix."""
    c = _make_client()
    c._executor.submit = MagicMock(side_effect=ValueError("boom-not-runtimeerror"))
    result = c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
    assert result is None
    c.close_sync()


# ── Assertion 6: warning gating matches check_sync (log_errors gate) ─────────
def test_log_sync_failure_warns_only_when_log_errors_true(caplog):
    c = _make_client(log_errors=True)
    c.close_sync()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="token_police"):
        c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"


def test_log_sync_failure_silent_when_log_errors_false(caplog):
    c = _make_client(log_errors=False)
    c.close_sync()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="token_police"):
        c.log_sync(model=SENTINEL_MODEL, input_tokens=1, output_tokens=1)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 0, f"expected no warning, got {len(warnings)}"


# ── Assertion 7 / INV-2: warning carries no prompt/completion/token/model content ──
def test_log_sync_after_close_warning_has_no_content(caplog):
    c = _make_client(log_errors=True)
    c.close_sync()
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="token_police"):
        c.log_sync(
            model=SENTINEL_MODEL,
            input_tokens=SENTINEL_INPUT,
            output_tokens=SENTINEL_OUTPUT,
        )
    text = " ".join(r.getMessage() for r in caplog.records)
    assert SENTINEL_MODEL not in text, "model name leaked into warning (INV-2)"
    assert str(SENTINEL_INPUT) not in text, "input token count leaked into warning (INV-2)"
    assert str(SENTINEL_OUTPUT) not in text, "output token count leaked into warning (INV-2)"
