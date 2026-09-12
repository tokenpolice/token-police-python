"""
Init() must NEVER throw into customer code at setup time.

The Python SDK's ``_wrap_method`` guards only ``importlib.import_module`` with
``except ImportError``; everything AFTER the import (getattr of the class /
method, descriptor access, setattr of the wrapper) is unguarded, so a module
whose import-time body raises a non-``ImportError`` (RuntimeError / OSError /
AttributeError / KeyError from a malformed target) escapes ``auto_instrument``
and crashes the customer app at boot.

These tests pin the two Python guard sites:
  1. a per-target ``try/except Exception: logger.debug`` around each
     ``_wrap_method(target)`` call in ``auto_instrument`` (isolation — one bad
     SDK doesn't skip instrumenting the others), and
  2. a call-site guard around ``setup_opentelemetry`` in ``init``.

The inner ``except ImportError: return`` fast path is deliberately left intact
(silent "SDK not installed → skip"). Mirrors
token-police-node/tests/initNeverThrows.test.ts (same intent; Node has two
extra prep-loop / call-site guards because its ``autoInstrument`` has a
pre-loop prep body that Python's ``auto_instrument`` does not).
"""
from __future__ import annotations

import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import enforcer
from token_police import client as tp_client


def _make_good_target(module_name: str):
    """Register a fake importable module with a wrappable class.method and
    return (target_dict, FakeClient, original_method)."""
    mod = types.ModuleType(module_name)

    class FakeClient:
        def create(self, **kwargs):
            return None

    mod.FakeClient = FakeClient
    sys.modules[module_name] = mod
    target = {"module": module_name, "object": "FakeClient", "method": "create", "async": False}
    return target, FakeClient, FakeClient.create


# A target dict missing the required "async" key → KeyError raised at
# target["async"], BEFORE the inner import try/except → exercises the OUTER
# per-target guard with a non-ImportError.
_BAD_TARGET_KEYERROR = {"module": "os", "object": "Nope", "method": "x"}


def _make_raising_body_target(module_name: str):
    """A fake module whose attribute access raises a non-ImportError from its
    'import-time body' (simulated via PEP-562 module __getattr__) — the exact
    Escape: import succeeds, post-import getattr raises RuntimeError."""
    mod = types.ModuleType(module_name)

    def _ga(name):
        raise RuntimeError("boom: import-time body error")

    mod.__getattr__ = _ga  # type: ignore[attr-defined]
    sys.modules[module_name] = mod
    return {"module": module_name, "object": "Anything", "method": "create", "async": False}


# ── No throw escapes auto_instrument() on the setup path ──────────────

def test_throwing_target_does_not_propagate(monkeypatch):
    # Assertion 1 — a malformed target raising a non-ImportError (KeyError)
    # must be swallowed; auto_instrument returns normally.
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [_BAD_TARGET_KEYERROR])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        enforcer.auto_instrument()  # must NOT raise
    finally:
        tp.uninstrument()


def test_non_import_error_from_target_body_does_not_propagate(monkeypatch):
    # Assertion 1 (post-import escape) — a RuntimeError from a module's
    # import-time body (the real hole) must not escape.
    target = _make_raising_body_target("tp_fake_raise_body")
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [target])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        enforcer.auto_instrument()  # must NOT raise
    finally:
        tp.uninstrument()
        sys.modules.pop("tp_fake_raise_body", None)


def test_init_never_throws_on_setup_failure(monkeypatch):
    # Assertion 3 — a full tp.init(...) whose instrumentation step has a
    # throwing target completes and returns a client (does not raise).
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [_BAD_TARGET_KEYERROR])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        client = tp.init(api_key="tp_sk_test", deployment="serverless", firewall="dry_run")
        assert client is not None
    finally:
        tp.uninstrument()


# ── setup_opentelemetry guarded ───────────────────────────────────────

def test_setup_opentelemetry_failure_does_not_propagate(monkeypatch):
    # Assertion 5 — setup_opentelemetry raising (provider construction failure)
    # must not escape init(). (auto_instrument now runs in 'off' too, but
    # setup_opentelemetry is the sole failure under test here.)
    def _boom(**kwargs):
        raise RuntimeError("boom: otel provider construction")

    monkeypatch.setattr(tp_client, "setup_opentelemetry", _boom)
    try:
        client = tp.init(api_key="tp_sk_test", deployment="serverless", firewall="off")
        assert client is not None
    finally:
        tp.uninstrument()


# ── Per-target isolation (the reason for per-target, not whole-loop, guard) ──

def test_per_target_isolation(monkeypatch):
    # Assertion 7 — a good target AFTER a throwing one is still instrumented:
    # one bad SDK must not skip instrumenting the rest.
    good_target, FakeClient, orig = _make_good_target("tp_fake_iso_sdk")
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [_BAD_TARGET_KEYERROR, good_target])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        enforcer.auto_instrument()  # must NOT raise
        assert FakeClient.create is not orig  # the valid target WAS wrapped
    finally:
        tp.uninstrument()
        sys.modules.pop("tp_fake_iso_sdk", None)


# ── Happy path / off-mode no-regression ───────────────────────────────

def test_healthy_target_is_wrapped(monkeypatch):
    # Assertion 9 — positive control: a valid target IS wrapped (identity changes).
    good_target, FakeClient, orig = _make_good_target("tp_fake_happy_sdk")
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [good_target])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        enforcer.auto_instrument()
        assert FakeClient.create is not orig
    finally:
        tp.uninstrument()
        sys.modules.pop("tp_fake_happy_sdk", None)


def test_off_mode_installs_tap_log_only(monkeypatch):
    # Assertion 10 — off mode DOES install the tap ("off = telemetry only"
    # must deliver full telemetry). The tap is log-only in off: the enforcer
    # choke points (_run_sync_check/_run_async_check) short-circuit on
    # firewall=="off" before any /check, block, or reroute — so installing the
    # wrapper is safe and required for manual-tap providers to emit /log.
    good_target, FakeClient, orig = _make_good_target("tp_fake_off_sdk")
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [good_target])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        tp.init(api_key="tp_sk_test", deployment="serverless", firewall="off")
        assert FakeClient.create is not orig  # auto_instrument runs in 'off'
    finally:
        tp.uninstrument()
        sys.modules.pop("tp_fake_off_sdk", None)


# ── Logging gated / silent by default ─────────────────────────────────

def test_failure_is_silent_by_default(monkeypatch, caplog):
    # Assertion 13 — a throwing target produces NO WARNING/ERROR record; the
    # failure is logged only at DEBUG (silent at default config).
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [_BAD_TARGET_KEYERROR])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        with caplog.at_level(logging.DEBUG, logger="token_police"):
            enforcer.auto_instrument()
        failure_records = [
            r for r in caplog.records if "instrumentation failed for a target" in r.getMessage()
        ]
        assert failure_records, "expected a debug record for the swallowed failure"
        assert all(r.levelno == logging.DEBUG for r in failure_records)
        # Nothing at WARNING or above from the swallowed setup failure.
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    finally:
        tp.uninstrument()


# ── Scope guard: inner except ImportError preserved ───────────────────

def test_import_error_target_silently_skipped(monkeypatch, caplog):
    # Assertion 16 — an uninstalled SDK (ImportError) is still silently skipped
    # by the inner fast path: no 'instrumentation failed' debug record fires
    # (the outer guard never sees it), proving the inner except ImportError
    # branch is intact and not widened.
    target = {"module": "tp_definitely_not_installed_xyz", "object": "C", "method": "m", "async": False}
    monkeypatch.setattr(enforcer, "_TARGET_METHODS", [target])
    monkeypatch.setattr(enforcer, "_is_instrumented", False)
    try:
        with caplog.at_level(logging.DEBUG, logger="token_police"):
            enforcer.auto_instrument()  # must NOT raise
        assert not [
            r for r in caplog.records if "instrumentation failed for a target" in r.getMessage()
        ]
    finally:
        tp.uninstrument()
