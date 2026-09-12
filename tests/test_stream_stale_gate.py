"""
Stream-stale allow-path /check fallback gate (C-14) — sync AND async.

A locally ALLOWED call whose decision hinged on a missed entity-list arm
(`armable_miss`) is re-verified with the existing inline /check ONLY once the
SSE stream has been disconnected longer than `stream_stale_grace_seconds`.
Healthy stream, within-grace windows, and unguarded (non-armable-miss) traffic
all keep the zero-round-trip hot path untouched.

The fallback reuses the EXACT State-B `/check` code path (`_run_sync_check` /
`_run_async_check`), so bounded timeout + fail-open semantics apply
automatically — see the GOLDEN RULE tests below, the most important tests in
this file.

Mirrors the driving pattern in tests/test_reroute_stash_session_thread.py.

Sibling: token-police-node/tests/streamStaleGate.test.ts pins the same
scenarios against the Node SDK.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as tp_state
from token_police.exceptions import TokenPoliceBlockedError


class _FakeClock:
    def __init__(self, start=1_700_000_000.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, secs):
        self.now += secs


# ENTITY_BLOCK on user_id EXISTS / group_by user_id, with an empty entities
# list. TPSession's default user_id is "anonymous" — always truthy, so the
# selector always matches and "anonymous" is never armed here, producing an
# armable_miss on every locally-allowed call that reaches this directive.
_ARMABLE_MISS_SNAPSHOT = {
    "schema_version": 1, "type": "snapshot", "version": 1,
    "tenant_id": "t", "project_id": "p", "ttl_seconds": 600, "loop_blocks": [],
    "directives": [
        {
            "id": "eb1", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
            "selector": {"match": {"field": "user_id", "operator": "EXISTS"},
                         "group_by": ["user_id"]},
            "entities": [],
        },
    ],
}

# No directives at all → no armable_miss is ever possible.
_NO_MISS_SNAPSHOT = {
    "schema_version": 1, "type": "snapshot", "version": 1,
    "tenant_id": "t", "project_id": "p", "ttl_seconds": 600, "loop_blocks": [],
    "directives": [],
}


@pytest.fixture(autouse=True)
def _isolated_stream_state(monkeypatch):
    # See test_stream_freshness.py — `_stream_*` are process-lifetime module
    # globals not reset by reset_pack(). Force a clean baseline before every
    # test so this file's explicit mark_stream_connected()/disconnected() calls
    # are never raced by a background StreamClient thread from tp.init() below
    # (or from any other test in the same pytest session): that thread's own
    # connect attempts target a closed local port (never a real 200), so its
    # generation never becomes current, and the zombie-generation guard (pinned
    # in test_stream_freshness.py) means it can never affect ours even if it
    # tried.
    monkeypatch.setattr(tp_state, "_stream_connected", False)
    monkeypatch.setattr(tp_state, "_stream_disconnected_at", None)
    monkeypatch.setattr(tp_state, "_stream_generation", 0)
    yield


def setup_function(_fn):
    tp_state.reset_pack()


def teardown_function(_fn):
    tp_state.reset_pack()
    try:
        tp.uninstrument()
    except Exception:
        pass


def _init(check_result, grace_seconds=60):
    """Init a daemon+enforce client with check_sync/check mocked. base_url
    points at a closed local port so the background StreamClient never reaches
    a real server (see the isolation fixture above)."""
    client = tp.init(
        api_key="tp_sk_test_c14",
        base_url="http://localhost:19999",
        firewall="enforce",
        deployment="daemon",
        stream_stale_grace_seconds=grace_seconds,
    )
    client.check_sync = MagicMock(return_value=check_result)
    client.check = AsyncMock(return_value=check_result)
    return client


def _run_async(kwargs):
    asyncio.run(_enforcer._run_async_check(kwargs=kwargs, provider="openai"))


# ── healthy stream + armable_miss → NO /check (hot path preserved) ──────


def test_sync_healthy_stream_armable_miss_no_check():
    client = _init({"status": "allowed"})
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    tp_state.mark_stream_connected()
    _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
    client.check_sync.assert_not_called()


def test_async_healthy_stream_armable_miss_no_check():
    client = _init({"status": "allowed"})
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    tp_state.mark_stream_connected()
    _run_async({"model": "gpt-4"})
    client.check.assert_not_called()


# ── within grace → no /check ─────────────────────────────────────────────


def test_sync_within_grace_no_check(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init({"status": "allowed"}, grace_seconds=60)
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(30)  # < 60s grace
    _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
    client.check_sync.assert_not_called()


def test_async_within_grace_no_check(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init({"status": "allowed"}, grace_seconds=60)
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(30)
    _run_async({"model": "gpt-4"})
    client.check.assert_not_called()


# ── stale beyond grace + armable_miss → /check called; blocked → raises ──


def test_sync_stale_armable_miss_check_called_blocked_raises(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init(
        {"status": "blocked", "reason": "budget exceeded", "ruleId": "eb1"},
        grace_seconds=60,
    )
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(61)  # past the 60s grace
    with pytest.raises(TokenPoliceBlockedError):
        _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
    client.check_sync.assert_called_once()


def test_async_stale_armable_miss_check_called_blocked_raises(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init(
        {"status": "blocked", "reason": "budget exceeded", "ruleId": "eb1"},
        grace_seconds=60,
    )
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(61)
    with pytest.raises(TokenPoliceBlockedError):
        _run_async({"model": "gpt-4"})
    client.check.assert_awaited_once()


# ── GOLDEN RULE ──────────────────────────────────────────────────────────
# The single most important tests in this file: a /check failure on the new
# stale-stream fallback path must NEVER throw anything other than
# TokenPoliceBlockedError into the customer's call — here /check doesn't even
# resolve to a decision, it outright raises (simulating a raw network failure
# escaping client.check_sync()/check(), which in production already
# fail-opens internally — this proves the enforcer's OWN @fail_safe wrapper is
# a second, independent safety net on this new code path).


def test_sync_golden_rule_check_network_error_fail_open(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init({"status": "allowed"}, grace_seconds=60)
    client.check_sync = MagicMock(side_effect=ConnectionError("ECONNREFUSED"))
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(61)
    # Must NOT raise.
    _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
    client.check_sync.assert_called_once()


def test_async_golden_rule_check_network_error_fail_open(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init({"status": "allowed"}, grace_seconds=60)
    client.check = AsyncMock(side_effect=ConnectionError("ECONNREFUSED"))
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(61)
    _run_async({"model": "gpt-4"})  # must NOT raise
    client.check.assert_awaited_once()


# ── stale beyond grace but NO armable_miss → no /check (unguarded traffic) ──


def test_sync_stale_no_armable_miss_no_check(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init({"status": "allowed"}, grace_seconds=60)
    tp_state.apply_snapshot(_NO_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(61)
    _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
    client.check_sync.assert_not_called()


def test_async_stale_no_armable_miss_no_check(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init({"status": "allowed"}, grace_seconds=60)
    tp_state.apply_snapshot(_NO_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(61)
    _run_async({"model": "gpt-4"})
    client.check.assert_not_called()


# ── grace 0: /check fires as soon as the stream is disconnected at all ─────


def test_sync_grace_zero_check_fires_immediately_on_disconnect(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(tp_state.time, "time", clock.time)
    client = _init({"status": "allowed"}, grace_seconds=0)
    tp_state.apply_snapshot(_ARMABLE_MISS_SNAPSHOT)
    gen = tp_state.mark_stream_connected()
    tp_state.mark_stream_disconnected(gen)
    clock.advance(0.001)  # any elapsed time > 0 is already stale at grace 0
    _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
    client.check_sync.assert_called_once()
