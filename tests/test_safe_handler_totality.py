"""GOLDEN RULE: fail_safe's own except-handler must be total.

The `except Exception` handler in token_police._safe is the SDK's outermost
fail-open boundary — it runs bare inside the customer's LLM call frame (e.g.
`_run_sync_check` invoked from a patched provider method). Its best-effort
reporting (client lookup, `log_errors` read, the f-string's `str(e)`, the
`logger.error` call) must therefore be guarded too: a hostile exception
`__str__`, a raising customer logging config, or a raising client accessor
must never escape into the caller's call.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from token_police import state
from token_police._safe import fail_safe
from token_police.exceptions import TokenPoliceBlockedError


class _HostileStr(Exception):
    """Exception whose __str__ raises — models a broken third-party error."""

    def __str__(self):  # pragma: no cover - the raise IS the behavior
        raise RuntimeError("hostile __str__")


class _StubClient:
    def __init__(self, log_errors=True):
        self.log_errors = log_errors


class _RaisingLogErrorsClient:
    @property
    def log_errors(self):
        raise RuntimeError("log_errors property exploded")


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@fail_safe
def _sync_boom():
    raise ValueError("sdk internal failure")


@fail_safe
async def _async_boom():
    raise ValueError("sdk internal async failure")


@fail_safe
def _sync_hostile():
    raise _HostileStr()


@fail_safe
async def _async_hostile():
    raise _HostileStr()


@fail_safe
def _sync_blocked():
    raise TokenPoliceBlockedError("deliberate block")


@fail_safe
async def _async_blocked():
    raise TokenPoliceBlockedError("deliberate block")


@fail_safe
def _sync_ok(x):
    return x * 2


def test_sync_handler_survives_raising_get_client(monkeypatch):
    def _boom_client():
        raise RuntimeError("get_client exploded")

    monkeypatch.setattr(state, "get_client", _boom_client)
    assert _sync_boom() is None  # swallowed, nothing propagates


def test_async_handler_survives_raising_get_client(monkeypatch):
    def _boom_client():
        raise RuntimeError("get_client exploded")

    monkeypatch.setattr(state, "get_client", _boom_client)
    assert _run(_async_boom()) is None


def test_handler_survives_hostile_exception_str(monkeypatch):
    """Highest-value case: log_errors=True forces the f-string to evaluate
    str(e); a hostile __str__ must not escape the fail-open boundary."""
    monkeypatch.setattr(state, "get_client", lambda: _StubClient(log_errors=True))
    assert _sync_hostile() is None
    assert _run(_async_hostile()) is None


def test_handler_survives_raising_logger(monkeypatch):
    import token_police._safe as safe_mod

    def _boom_error(*a, **k):
        raise RuntimeError("customer logging config exploded")

    monkeypatch.setattr(state, "get_client", lambda: _StubClient(log_errors=True))
    monkeypatch.setattr(safe_mod.logger, "error", _boom_error)
    assert _sync_boom() is None
    assert _run(_async_boom()) is None


def test_handler_survives_raising_log_errors_property(monkeypatch):
    monkeypatch.setattr(state, "get_client", lambda: _RaisingLogErrorsClient())
    assert _sync_boom() is None
    assert _run(_async_boom()) is None


def test_blocked_error_still_propagates(monkeypatch):
    """Regression: the deliberate-block re-raise must survive the hardening,
    even when the reporting path would itself raise."""
    def _boom_client():
        raise RuntimeError("get_client exploded")

    monkeypatch.setattr(state, "get_client", _boom_client)
    with pytest.raises(TokenPoliceBlockedError):
        _sync_blocked()
    with pytest.raises(TokenPoliceBlockedError):
        _run(_async_blocked())


def test_log_errors_true_still_logs_once(monkeypatch, caplog):
    monkeypatch.setattr(state, "get_client", lambda: _StubClient(log_errors=True))
    with caplog.at_level(logging.ERROR, logger="token_police"):
        assert _sync_boom() is None
    errors = [
        r for r in caplog.records
        if r.levelno == logging.ERROR
        and "instrumentation failed (fail-open)" in r.getMessage()
    ]
    assert len(errors) == 1
    assert "sdk internal failure" in errors[0].getMessage()


def test_log_errors_false_logs_nothing(monkeypatch, caplog):
    monkeypatch.setattr(state, "get_client", lambda: _StubClient(log_errors=False))
    with caplog.at_level(logging.ERROR, logger="token_police"):
        assert _sync_boom() is None
    assert not [
        r for r in caplog.records
        if "instrumentation failed (fail-open)" in r.getMessage()
    ]


def test_normal_return_passes_through(monkeypatch):
    monkeypatch.setattr(state, "get_client", lambda: None)
    assert _sync_ok(21) == 42
