"""
TokenPoliceBlockedError carries structured fields so callers can branch on a
block (budget vs loop-detection, which rule) without parsing the message.

Twin of token-police-node/tests/blockedErrorFields.test.ts — the two files
pin the cross-SDK parity contract on the SDK's canonical field names
(snake_case here, camelCase in Node).
"""
from unittest.mock import MagicMock

import pytest

import token_police as tp
from token_police import enforcer as _enforcer
from token_police.exceptions import TokenPoliceBlockedError


# ── unit: the exception itself ───────────────────────────────────────
def test_message_only_construction_back_compat():
    # Customers construct/catch it with a bare message — must be unchanged.
    err = TokenPoliceBlockedError("boom")
    assert str(err) == "boom"
    assert err.reason is None
    assert err.rule_id is None
    assert err.kind is None
    assert err.trace_id is None


def test_fields_populated_via_kwargs():
    err = TokenPoliceBlockedError(
        "msg", reason="over budget", rule_id="rule_1", kind="budget", trace_id="tr_9"
    )
    assert str(err) == "msg"
    assert err.reason == "over budget"
    assert err.rule_id == "rule_1"
    assert err.kind == "budget"
    assert err.trace_id == "tr_9"


# ── integration: caught error from an enforce-mode block ─────────────
def _init_enforce(check_result):
    """Init an enforce client; no cached pack → State-B inline /check drives it."""
    client = tp.init(api_key="tp_sk_test_bef", firewall="enforce")
    client.check_sync = MagicMock(return_value=check_result)
    client.log_sync = MagicMock()
    return client


def test_enforce_budget_block_carries_fields():
    _init_enforce({
        "status": "blocked",
        "reason": "budget exceeded",
        "ruleId": "rule_budget_1",
        "traceId": "tr_budget",
    })
    try:
        with pytest.raises(TokenPoliceBlockedError) as ei:
            _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
        err = ei.value
        # message byte-identical to before the fix
        assert str(err) == "TokenPolice: Budget exceeded — budget exceeded"
        assert err.reason == "budget exceeded"
        assert err.rule_id == "rule_budget_1"
        assert err.trace_id == "tr_budget"
        # no loop detail present → server budget block
        assert err.kind == "budget"
    finally:
        tp.uninstrument()


def test_enforce_loop_block_carries_detector_kind():
    _init_enforce({
        "status": "blocked",
        "reason": "loop detected",
        "ruleId": "rule_loop_1",
        "detail": "HASH_CYCLE",
        "traceId": "tr_loop",
    })
    try:
        with pytest.raises(TokenPoliceBlockedError) as ei:
            _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
        err = ei.value
        assert err.reason == "loop detected"
        assert err.rule_id == "rule_loop_1"
        # loop-detector discriminator flows through as `kind`
        assert err.kind == "HASH_CYCLE"
        assert err.trace_id == "tr_loop"
    finally:
        tp.uninstrument()
