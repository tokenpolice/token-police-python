"""
A terminal SSE connect status (revoked/invalid key → 401/403, or a misconfigured
base_url/route → 404) must NOT permanently stop the rule-stream reader. A single
transient 404 from a load-balancer blip during a rolling deploy, or a 401/403
from server-side key-cache staleness, would otherwise kill the fast path for the
whole process lifetime (every call degrading to inline /check forever).

Contract:
  * A terminal status invalidates the pack (→ inline /check, fail-open), records
    the status, and returns WITHOUT setting the stop signal.
  * The outer loop re-probes at the reconnect cap (the ladder TOP, ~one probe per
    cap period) instead of resetting to the ~1s backoff floor — no hammering.
  * A later non-terminal connect resumes normal streaming; a healthy 200 with a
    fresh snapshot self-heals the pack with no manual restart. Terminal also
    drops Last-Event-ID (SSE-12) so the cap re-probe cannot resume-at-head —
    collector skip-when-equal would otherwise withhold the snapshot.
  * A terminal streak is logged at warning once, then at debug (no spam).
  * stop() still wins instantly at every point; manual stop()→start() unchanged.

Sibling: token-police-node/tests/streamTerminalStatus.test.ts pins the same
scenarios (Node has no logging — the streak-warning assertion is Python-only).
"""
import json
import logging
import threading

import httpx

from token_police import state, stream


def _make_client() -> stream.StreamClient:
    return stream.StreamClient(
        base_url="http://localhost:15099",
        api_key="tp_sk_test",
        sdk_version="test",
        deployment="daemon",
        client_id="cid",
        firewall="enforce",
    )


def _snapshot():
    return {
        "version": 1,
        "tenant_id": "t1",
        "project_id": "p1",
        "directives": [],
        "loop_blocks": [],
    }


def setup_function(_fn):
    state.reset_pack()


class _FakeStreamResp:
    """Context manager mirroring `with httpx.stream(...) as resp:`. `status_code`
    drives the connect gate; `iter_lines()` feeds _pump on a 200 (empty → clean
    EOF return)."""

    def __init__(self, status_code, lines=()):
        self.status_code = status_code
        self._lines = list(lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_lines(self):
        return iter(self._lines)


def _patch_stream(monkeypatch, status_code, counter=None):
    def fake_stream(method, url, headers=None, timeout=None):
        if counter is not None:
            counter["n"] += 1
        return _FakeStreamResp(status_code)

    monkeypatch.setattr(httpx, "stream", fake_stream)


# ── Terminal status does NOT stop the reader; it records the status ─────────
def test_401_does_not_stop_records_terminal(monkeypatch):
    _patch_stream(monkeypatch, 401)
    client = _make_client()
    client._last_event_id = 10
    client._connect_and_pump()
    # New contract: a terminal status must NOT set the stop signal (the reader
    # keeps re-probing). Old contract (RED): 401 set _stop_event → reader dead.
    assert client._stop_event.is_set() is False
    assert client._terminal_status == 401
    assert client._last_event_id is None  # SSE-12: drop bookmark with the pack


def test_403_and_404_record_terminal_without_stopping(monkeypatch):
    for status in (403, 404):
        state.reset_pack()
        _patch_stream(monkeypatch, status)
        client = _make_client()
        client._last_event_id = 10
        client._connect_and_pump()
        assert client._stop_event.is_set() is False
        assert client._terminal_status == status
        assert client._last_event_id is None


# ── Terminal give-up STILL invalidates a healthy pack (fail-open) ───────────
def test_401_invalidates_healthy_pack(monkeypatch):
    assert state.apply_snapshot(_snapshot()) is True
    assert state.get_pack() is not None
    assert state.is_cache_healthy() is True

    _patch_stream(monkeypatch, 401)
    client = _make_client()
    client._last_event_id = 10
    client._connect_and_pump()

    # invalidate_pack("stream_terminal_401") fired → cache unhealthy → enforcer
    # (unchanged) bails to inline /check (fail-open). The reader is NOT stopped.
    # SSE-12: the resume cursor is dropped with the pack.
    assert state.get_pack() is None
    assert state.is_cache_healthy() is False
    assert client._last_event_id is None
    assert client._stop_event.is_set() is False
    assert client._terminal_status == 401


# ── Loop-level: terminal does NOT exit the loop; it re-probes at the CAP ─────
def test_terminal_does_not_stop_loop_reprobes_at_cap(monkeypatch):
    counter = {"n": 0}
    _patch_stream(monkeypatch, 401, counter=counter)
    monkeypatch.setattr(stream.random, "random", lambda: 0.5)  # zero jitter

    client = _make_client()
    waits = []

    def fake_wait(sleep_for=None):
        waits.append(sleep_for)
        client._stop_event.set()  # break after the first re-probe is scheduled
        return True

    monkeypatch.setattr(client._stop_event, "wait", fake_wait)

    client._run_forever()

    # Old contract (RED): a terminal set _stop_event so _run_forever exited BEFORE
    # any backoff wait → waits == [], fetch happened once then dead forever.
    # New contract (GREEN): terminal does NOT stop; after exactly one connect the
    # loop schedules a re-probe at the reconnect cap (ladder top), not the 1s floor.
    assert counter["n"] == 1
    assert waits == [float(client.reconnect_cap_seconds)]


# ── Two consecutive terminals → both re-probe at cap; warning logged ONCE ───
def test_two_consecutive_terminals_reprobe_at_cap_and_warn_once(monkeypatch, caplog):
    _patch_stream(monkeypatch, 401)
    monkeypatch.setattr(stream.random, "random", lambda: 0.5)

    client = _make_client()
    waits = []

    def fake_wait(sleep_for=None):
        waits.append(sleep_for)
        if len(waits) >= 2:
            client._stop_event.set()
        return True

    monkeypatch.setattr(client._stop_event, "wait", fake_wait)

    with caplog.at_level(logging.WARNING, logger="token_police"):
        client._run_forever()

    # Both re-probes scheduled at the reconnect cap (the streak stays at ladder top).
    assert waits[0] == float(client.reconnect_cap_seconds)
    assert waits[1] == float(client.reconnect_cap_seconds)
    # Warning logged exactly once for the streak (the 2nd terminal drops to debug).
    warns = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "rule stream got HTTP" in r.getMessage()
    ]
    assert len(warns) == 1


# ── Self-heal: terminal → re-probe → 200 fresh snapshot → pack heals ────────
def test_terminal_then_200_self_heals(monkeypatch):
    snap_lines = [
        "event: snapshot",
        "id: 1",
        "data: " + json.dumps(_snapshot()),
        "",
    ]
    calls = {"n": 0}
    captured_headers = []

    def fake_stream(method, url, headers=None, timeout=None):
        captured_headers.append(dict(headers or {}))
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeStreamResp(401)          # terminal first
        # Collector skip-when-equal: a leftover Last-Event-ID gets keepalives
        # only — no snapshot. Pre-seeded cursor must have been dropped (SSE-12)
        # or this path never heals.
        if headers and "Last-Event-ID" in headers:
            return _FakeStreamResp(200, [": keepalive", ""])
        return _FakeStreamResp(200, snap_lines)

    monkeypatch.setattr(httpx, "stream", fake_stream)
    monkeypatch.setattr(stream.random, "random", lambda: 0.5)

    assert state.apply_snapshot(_snapshot()) is True
    client = _make_client()
    client._last_event_id = 10
    waits = []

    def fake_wait(sleep_for=None):
        waits.append(sleep_for)
        if len(waits) >= 2:
            client._stop_event.set()
        return True

    monkeypatch.setattr(client._stop_event, "wait", fake_wait)

    client._run_forever()

    # First connect terminal (invalidated + cursor dropped), second connect 200
    # with no Last-Event-ID applied the fresh snapshot → pack healed.
    assert calls["n"] == 2
    assert "Last-Event-ID" not in captured_headers[1]
    assert state.is_cache_healthy() is True
    assert state.get_pack() is not None
    # The re-probe after the terminal was scheduled at the cap (ladder top); the
    # healthy disconnect that followed reset back to the ~1s floor.
    assert waits[0] == float(client.reconnect_cap_seconds)
    assert waits[1] == 1.0


# ── stop() still wins instantly even though terminal keeps the loop alive ───
def test_manual_stop_exits_immediately(monkeypatch):
    _patch_stream(monkeypatch, 401)
    client = _make_client()

    def fake_wait(sleep_for=None):
        client.stop()  # simulate customer shutdown during the re-probe wait
        return True

    monkeypatch.setattr(client._stop_event, "wait", fake_wait)

    t = threading.Thread(target=client._run_forever, daemon=True)
    t.start()
    t.join(timeout=5)
    assert not t.is_alive()  # stop() broke the loop immediately
    assert client._stop_event.is_set() is True


# ── Transient statuses still climb the ladder and never give up (unchanged) ─
def test_500_does_not_give_up_or_invalidate(monkeypatch):
    import pytest

    assert state.apply_snapshot(_snapshot()) is True  # pre-seed healthy pack

    _patch_stream(monkeypatch, 500)
    client = _make_client()
    client._last_event_id = 10
    # A transient status RAISES so _run_forever preserves the climbed `attempt`
    # and the reconnect climbs the backoff ladder instead of resetting to the 1s
    # floor. It is not terminal: no stop, no pack invalidation, no terminal mark.
    with pytest.raises(RuntimeError):
        client._connect_and_pump()

    assert client._stop_event.is_set() is False
    assert client._terminal_status is None
    assert state.get_pack() is not None
    assert state.is_cache_healthy() is True
    assert client._last_event_id == 10  # transient path keeps the bookmark


def test_429_does_not_give_up(monkeypatch):
    import pytest

    _patch_stream(monkeypatch, 429)
    client = _make_client()
    client._last_event_id = 10
    with pytest.raises(RuntimeError):
        client._connect_and_pump()
    assert client._stop_event.is_set() is False
    assert client._terminal_status is None
    assert client._last_event_id == 10
