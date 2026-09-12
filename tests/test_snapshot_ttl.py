"""
An expired Decision Pack (TTL stamped on the snapshot by the server but
never enforced) kept enforcing while heartbeats arrived but no snapshot/delta
refreshed it. Fix (state + stream.py pump watchdog, REUSING the flag): the
pump watchdog, after the heartbeat/stale checks, calls `state.is_pack_expired()`
→ `state.invalidate_pack()`, nulls the reconnect cursor, and sets the EXISTING
`_force_reconnect` flag. The loop-top guard returns → the `with
httpx.stream(...)` block closes → the EXISTING backoff reconnects WITHOUT
Last-Event-ID → fresh snapshot → pack un-expires. `is_pack_expired()` and
`invalidate_pack()` are two SEQUENTIAL (non-nested) `_pack_lock` acquisitions
. The dead `_check_snapshot_ttl` `issued_at` stub is removed. No new
backoff/socket code; no 24h sleep (a controllable clock drives the loop).

Sibling: token-police-node/tests/streamSnapshotTtl.test.ts pins the same
scenarios. Assertions 10, 12, 16 (Python side).
"""
import inspect
import time

from token_police import state, stream


class _FakeClock:
    def __init__(self, start=0.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, secs):
        self.now += secs


def _make_client() -> stream.StreamClient:
    return stream.StreamClient(
        base_url="http://localhost:15096",
        api_key="tp_sk_test",
        sdk_version="test",
        deployment="daemon",
        client_id="cid",
        firewall="enforce",
    )


def _ttl_snapshot(ttl):
    snap = {
        "version": 1,
        "tenant_id": "t1",
        "project_id": "p1",
        "directives": [],
        "loop_blocks": [],
    }
    if ttl is not None:
        snap["ttl_seconds"] = ttl
    return snap


class _HeartbeatDriver:
    """iter_lines() backing iterator: yields `:ka` keepalive lines, advancing the
    fake clock by `step` seconds on EACH pull so the pack ages between loop
    iterations without ever tripping the 45s heartbeat watchdog (step < 45)."""

    def __init__(self, clock, step, count):
        self._clock = clock
        self._step = step
        self._count = count
        self.pulls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.pulls += 1
        if self.pulls > self._count:
            raise AssertionError("over-pulled — loop-top guard did not fire")
        self._clock.advance(self._step)
        return ":ka"


class _FakeResponse:
    def __init__(self, backing):
        self._backing = backing

    def iter_lines(self):
        return self._backing


def test_expired_pack_sets_force_reconnect_and_drops_cursor(monkeypatch):
    # Assertion #10: an aged-past-TTL pack makes the watchdog invalidate + set
    # the flag + null the cursor. Heartbeats keep the heartbeat/stale
    # watchdogs quiet so ONLY the TTL branch can fire.
    clock = _FakeClock(0.0)
    monkeypatch.setattr(time, "time", clock.time)  # patches state + stream

    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(50)) is True  # received_at = 0
    client = _make_client()
    client._last_event_id = 1  # simulate a prior snapshot cursor
    client._last_event_at = 0.0
    client._last_data_event_at = 0.0

    # 40s per pull: pull1 @40s (age 40 < 50, ok), pull2 @80s (age 80 > 50 →
    # expired → flag set), pull3 @120s (loop-top guard returns before process).
    driver = _HeartbeatDriver(clock, step=40, count=3)
    client._pump(_FakeResponse(driver))

    assert client._force_reconnect is True       # Flag set on expiry
    assert client._last_event_id is None         # cursor dropped → fresh snapshot
    assert state.is_cache_healthy() is False      # invalidate_pack ran
    assert driver.pulls == 3                       # 2 processed + 1 that guard-returned


def test_no_ttl_pack_never_forces_reconnect(monkeypatch):
    # Assertion #12: a TTL-less (legacy) snapshot never expires — the watchdog
    # branch never fires no matter how long heartbeats flow. Byte-identical.
    clock = _FakeClock(0.0)
    monkeypatch.setattr(time, "time", clock.time)

    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(None)) is True
    client = _make_client()
    client._last_event_id = 1
    client._last_event_at = 0.0
    client._last_data_event_at = 0.0

    # Many heartbeats over a long span; the pack never expires. The driver would
    # raise if the loop over-pulled, but the guard must never fire, so we cap the
    # heartbeat run with a StopIteration-terminating list instead.
    lines = [":ka"] * 8
    it = iter(lines)

    class _Advancing:
        def __init__(self, base):
            self._base = base

        def __iter__(self):
            return self

        def __next__(self):
            clock.advance(40)  # < 45s heartbeat window
            return next(self._base)

    client._pump(_FakeResponse(_Advancing(it)))

    assert client._force_reconnect is False       # never triggered
    assert client._last_event_id == 1             # cursor untouched
    assert state.is_cache_healthy() is True        # still healthy


def test_check_snapshot_ttl_stub_removed_and_no_issued_at(monkeypatch):
    # Assertion #16: the dead `_check_snapshot_ttl` stub and its call site are
    # gone; `issued_at` is not parsed anywhere in the SDK (receipt-based only).
    assert not hasattr(stream.StreamClient, "_check_snapshot_ttl")
    stream_src = inspect.getsource(stream)
    assert "_check_snapshot_ttl" not in stream_src
    assert "issued_at" not in stream_src
    assert "issued_at" not in inspect.getsource(state)


def test_watchdog_branch_adds_no_backoff_or_socket_code():
    # Assertion #10 (static pin): the branch reuses the flag + cursor
    # null only — no new httpx.stream / socket / backoff / sleep code in _pump.
    pump_src = inspect.getsource(stream.StreamClient._pump)
    assert "state.is_pack_expired()" in pump_src
    assert 'state.invalidate_pack("snapshot_ttl_expired")' in pump_src
    # inspect executable code only — comments legitimately mention httpx.stream.
    code_only = "\n".join(
        line for line in pump_src.splitlines()
        if not line.strip().startswith("#")
    )
    # _pump never OPENS a stream (that's _connect_and_pump) — no new socket/backoff
    assert "httpx.stream" not in code_only
    assert "time.sleep" not in code_only
    # the branch sets the flag + nulls the cursor
    assert "self._force_reconnect = True" in pump_src
    assert "self._last_event_id = None" in pump_src


def test_golden_rule_no_throw_or_stop_in_watchdog_branch():
    # Assertion #18: the watchdog never raises into customer code and never sets
    # _stop_event — expiry only INCREASES fail-open fallbacks.
    pump_src = inspect.getsource(stream.StreamClient._pump)
    # locate the branch and prove it contains no raise / _stop_event.set
    idx = pump_src.index("if state.is_pack_expired():")
    branch = pump_src[idx:idx + 400]
    assert "raise" not in branch
    assert "_stop_event.set(" not in branch
