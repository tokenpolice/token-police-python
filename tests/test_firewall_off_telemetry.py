"""``firewall="off"`` = telemetry only must actually deliver full telemetry.

Before this fix, ``tp.init(firewall="off")`` skipped ``auto_instrument()``, so every
manual-tap-only provider (Bedrock, Cohere, @google/genai, streaming-usage taps)
went dark in ``off`` — no ``/log`` at all — contradicting the documented
"telemetry only" promise. The fix opens the install gate so taps install in
``off`` too and run LOG-ONLY: the enforcer choke points
(``_run_sync_check``/``_run_async_check``) short-circuit on ``firewall == "off"``
BEFORE any ``/check``, block, or reroute, so a tapped call in ``off`` emits
exactly one ``/log``, makes zero ``/check``, never reroutes, and never raises —
even with a would-block rule.

Parity with the Node suite ``tests/firewallOffTelemetry.test.ts``. The behavioral
tests exercise the REAL manual-tap handler ``_handle_bedrock_embedding_sync``
directly (per rubric) with a spy ``original``; with no cached pack the enforcer
takes the inline ``/check`` path, so the spied ``check_sync`` drives the verdict.
"""
import io
import json

import pytest
from unittest.mock import MagicMock

import token_police as tp
from token_police import client as tp_client
from token_police import enforcer as _enforcer
from token_police.exceptions import TokenPoliceBlockedError


# ── helpers ──────────────────────────────────────────────────────────
def _init(firewall, check_result=None, check_raises=False):
    """Init a real client in `firewall` mode with check_sync/log_sync spied.

    With no cached pack the enforcer takes the inline /check path, so
    `check_result` drives the verdict directly. Returns (client, check_mock,
    log_mock).
    """
    client = tp.init(api_key="tp_sk_test_off", firewall=firewall)
    if check_raises:
        check_mock = MagicMock(side_effect=TokenPoliceBlockedError("should not be called"))
    else:
        check_mock = MagicMock(return_value=check_result or {"status": "allowed"})
    client.check_sync = check_mock
    log_mock = MagicMock()
    client.log_sync = log_mock
    return client, check_mock, log_mock


def _make_bedrock_response():
    """A minimal Bedrock InvokeModel embedding response (Titan shape)."""
    payload = json.dumps({"inputTextTokenCount": 42}).encode()
    return {"body": io.BytesIO(payload), "contentType": "application/json"}


def _bedrock_args():
    api_params = {
        "modelId": "amazon.titan-embed-text-v2:0",
        "body": json.dumps({"inputText": "hello world"}),
    }
    # botocore.client path: original(self, operation_name, api_params)
    return (object(), "invoke_model", api_params)


# ── Assertion 2: install gate opens in off ───────────────────────────
def test_off_calls_auto_instrument(monkeypatch):
    spy = MagicMock()
    monkeypatch.setattr(tp_client, "auto_instrument", spy)
    try:
        tp.init(api_key="tp_sk_test_off", firewall="off")
        spy.assert_called_once()
    finally:
        tp.uninstrument()


# ── Assertion 3 (runnable): off opens NO SSE stream ──────────────────
def test_off_opens_no_sse_stream():
    try:
        client = tp.init(api_key="tp_sk_test_off", firewall="off", deployment="daemon")
        assert client._stream_client is None
    finally:
        tp.uninstrument()


# ── Assertions 5, 7, 8, 12: off manual tap logs once, zero /check, no raise ──
def test_off_manual_tap_logs_once_zero_check_no_raise():
    _client, check_mock, log_mock = _init("off", check_raises=True)
    original = MagicMock(return_value=_make_bedrock_response())
    args = _bedrock_args()
    try:
        # Golden Rule: must not raise even though check_sync would block/raise.
        resp = _enforcer._handle_bedrock_embedding_sync(original, args, {})
        # Assertion 7: zero /check in off.
        assert check_mock.call_count == 0
        # Assertion 5 + 12: exactly one /log, and the provider WAS reached once.
        assert log_mock.call_count == 1
        assert original.call_count == 1
        # Assertion 8: model unchanged (no reroute — /check never ran).
        assert log_mock.call_args.kwargs["model"] == "amazon.titan-embed-text-v2:0"
        assert log_mock.call_args.kwargs["input_tokens"] == 42
        assert resp is not None  # customer result returned untouched
    finally:
        tp.uninstrument()


# ── Assertion 9 (regression): enforce still raises + provider never reached ──
def test_enforce_still_raises_on_block_provider_not_reached():
    _client, check_mock, _log_mock = _init(
        "enforce", check_result={"status": "blocked", "reason": "budget exceeded"})
    original = MagicMock(return_value=_make_bedrock_response())
    args = _bedrock_args()
    try:
        with pytest.raises(TokenPoliceBlockedError):
            _enforcer._handle_bedrock_embedding_sync(original, args, {})
        assert original.call_count == 0
        assert check_mock.call_count == 1
    finally:
        tp.uninstrument()


# ── Assertion 10 (regression): dry_run runs /check but suppresses the block ──
def test_dry_run_runs_check_but_suppresses_block():
    _client, check_mock, log_mock = _init(
        "dry_run", check_result={"status": "blocked", "reason": "budget exceeded"})
    original = MagicMock(return_value=_make_bedrock_response())
    args = _bedrock_args()
    try:
        # No raise (block suppressed), but /check DID run (full parity path).
        resp = _enforcer._handle_bedrock_embedding_sync(original, args, {})
        assert check_mock.call_count >= 1
        assert original.call_count == 1  # provider reached (block suppressed)
        assert log_mock.call_count == 1
        assert log_mock.call_args.kwargs["model"] == "amazon.titan-embed-text-v2:0"
        assert resp is not None
    finally:
        tp.uninstrument()
