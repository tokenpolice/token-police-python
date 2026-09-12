"""
The module-level ``init()`` convenience wrapper must accept and forward
the ``capture_stream_usage`` flag to the ``TokenPolice`` constructor.

Before the fix, ``init(..., capture_stream_usage=...)`` raised
``TypeError: got an unexpected keyword argument 'capture_stream_usage'`` straight
into customer setup code, and callers who use the documented ``init()`` entrypoint
could not toggle stream-usage capture at all (the constructor supported it).

The fix threads a single passthrough kwarg with a safe ``None`` default so the
constructor still owns env resolution (``client.py:94-95``): an omitted call is
byte-identical to today. Mirrors Node's ``captureStreamUsage`` option
(``client.ts:71,216-217``), which never dropped the flag.

All tests run with ``firewall="off"`` + ``deployment="serverless"`` so no real
instrumentation wrappers are installed, and reset the module-global client in a
``finally`` so no state leaks into sibling tests.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police
from token_police import enforcer
from token_police import state as tp_state


@pytest.fixture(autouse=True)
def _reset_client_and_env(monkeypatch):
    """Assertion 12 — hygiene: ensure a clean env + no leaked global client.

    ``monkeypatch`` auto-reverts any ``TP_CAPTURE_STREAM_USAGE`` change; the
    ``finally`` tears down instrumentation and clears the module-global client so
    state cannot leak into ``test_init_never_throws.py`` / parity tests.
    """
    # Start each test from a known env baseline (individual tests re-set as needed).
    monkeypatch.delenv("TP_CAPTURE_STREAM_USAGE", raising=False)
    try:
        yield
    finally:
        try:
            token_police.uninstrument()
        except Exception:
            pass
        tp_state.set_client(None)  # clear the module-global singleton


def _init(**kwargs):
    return token_police.init(
        api_key="tp_sk_test",
        deployment="serverless",
        firewall="off",
        **kwargs,
    )


# ── Assertion 1 — signature has the param, default None ───────────────────────

def test_init_signature_accepts_capture_stream_usage():
    sig = inspect.signature(token_police.init)
    assert "capture_stream_usage" in sig.parameters
    param = sig.parameters["capture_stream_usage"]
    assert param.default is None


# ── Assertion 11 — no other init() param altered ──────────────────────────────

def test_init_signature_unchanged_for_other_params():
    sig = inspect.signature(token_police.init)
    # The exact, ordered parameter set of init() after the fix. If any other
    # parameter is added/removed/renamed this locks it down.
    assert list(sig.parameters.keys()) == [
        "api_key",
        "base_url",
        "timeout",
        "firewall",
        "enforce",
        "log_errors",
        "max_workers",
        "tracer_provider",
        "deployment",
        "sse_reconnect_max_interval_seconds",
        "capture_stream_usage",
        "error_detail",
        "stream_stale_grace_seconds",
    ]


# ── Assertion 3 — headline: no TypeError ──────────────────────────────────────

def test_init_with_capture_stream_usage_false_does_not_raise():
    client = _init(capture_stream_usage=False)  # pre-fix: TypeError
    assert client is not None


# ── Assertion 4 — explicit False honored ──────────────────────────────────────

def test_init_capture_stream_usage_false_sets_attr_false():
    _init(capture_stream_usage=False)
    assert token_police.get_client().capture_stream_usage is False


# ── Assertion 5 — explicit True honored ───────────────────────────────────────

def test_init_capture_stream_usage_true_sets_attr_true():
    _init(capture_stream_usage=True)
    assert token_police.get_client().capture_stream_usage is True


# ── Assertion 6 — byte-identical-to-today: omitted + env unset → default on ────

def test_init_omitted_defaults_on_when_env_unset(monkeypatch):
    monkeypatch.delenv("TP_CAPTURE_STREAM_USAGE", raising=False)
    _init()  # kwarg omitted
    assert token_police.get_client().capture_stream_usage is True


# ── Assertion 7 — env still respected when kwarg omitted ──────────────────────

def test_init_omitted_respects_env_zero(monkeypatch):
    monkeypatch.setenv("TP_CAPTURE_STREAM_USAGE", "0")
    _init()  # kwarg omitted — constructor resolves env
    assert token_police.get_client().capture_stream_usage is False


# ── Assertion 8 — explicit kwarg wins over env ────────────────────────────────

def test_init_explicit_true_overrides_env_zero(monkeypatch):
    monkeypatch.setenv("TP_CAPTURE_STREAM_USAGE", "0")
    _init(capture_stream_usage=True)
    assert token_police.get_client().capture_stream_usage is True


# ── Assertion 9 (D2) — downstream tap reads the flag off the client ───────────

def test_enforcer_tap_reads_client_flag():
    # Flag False → injection suppressed (early return None on the flag check).
    _init(capture_stream_usage=False)
    kwargs = {"stream": True, "messages": []}
    token = enforcer._inject_stream_usage_option("openai.resources.chat", kwargs)
    assert token is None
    assert "stream_options" not in kwargs  # nothing injected

    # Flip flag True → the tap does NOT early-return on the flag; it injects.
    tp_state.get_client().capture_stream_usage = True
    kwargs2 = {"stream": True, "messages": []}
    token2 = enforcer._inject_stream_usage_option("openai.resources.chat", kwargs2)
    assert token2 is not None
    assert kwargs2.get("stream_options", {}).get("include_usage") is True
