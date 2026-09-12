"""
A `delta` frame arriving while the local cache is UNHEALTHY (no snapshot
landed yet, or a prior apply/TTL failure poisoned the pack) used to reset the
reconnect cursor (`self._last_event_id = None`, stream.py) but NOT force a
reconnect. Because `Last-Event-ID` is only ever sent in the connect headers
(stream.py:99-100), nulling the cursor is inert until an actual reconnect
happens — and while the server keeps streaming deltas the heartbeat/stale
watchdogs keep getting refreshed, so the cache is silently stale forever
(inline /check fallback for the whole duration).

Fix (_dispatch + _pump only): a per-connection `_force_reconnect` bool set ONLY
on the unhealthy-cache `else` branch, plus a single guard
`if self._force_reconnect: return` at the top of the `for line in
response.iter_lines()` loop body. Returning exits `_pump`, the
`with httpx.stream(...)` block closes the socket, and the EXISTING backoff in
`_run_forever` reconnects WITHOUT `Last-Event-ID` → fresh snapshot. Never sets
`_stop_event`; lifecycle untouched.

Sibling: token-police-node/tests/streamForceReconnect.test.ts pins the same
scenarios. These tests encode assertions 2, 5, 7, 9, 11.
"""
import time

import httpx

from token_police import state, stream


def _make_client() -> stream.StreamClient:
    return stream.StreamClient(
        base_url="http://localhost:15097",
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


# ── Assertion 2: flag initialized, default False ──────────────────────────
def test_force_reconnect_defaults_false():
    client = _make_client()
    assert client._force_reconnect is False


# ── Assertion 9 (RED-pre / GREEN-post): poison-generator pump early-return ──
class _CountingLines:
    """iter_lines() backing iterator. Yields the supplied benign lines, then
    raises AssertionError on the NEXT pull — the "poison" that fires pre-fix
    (no guard) when the loop over-reads, and must NOT fire post-fix (the
    loop-top guard returns before pulling it)."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0
        self.pulls = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.pulls += 1
        if self._i < len(self._lines):
            v = self._lines[self._i]
            self._i += 1
            return v
        raise AssertionError("poison line pulled — loop-top guard did not fire")


class _FakeResponse:
    def __init__(self, backing):
        self._backing = backing

    def iter_lines(self):
        return self._backing


def test_unhealthy_delta_forces_reconnect_and_stops_pulling():
    state.reset_pack()  # cache UNHEALTHY: _pack is None → is_cache_healthy() False
    client = _make_client()
    # Reviewer impl note: seed the watchdog clocks to now so the:129 heartbeat
    # watchdog doesn't return on iteration 1 (they default to 0.0 in __init__).
    now = time.time()
    client._last_event_at = now
    client._last_data_event_at = now

    # One delta frame (3 lines: event / data / terminating blank) followed by a
    # benign 4th line (:ka keepalive). The blank line invokes _dispatch → the
    # unhealthy branch sets _force_reconnect=True; the guard can only fire on the
    # NEXT iteration, so the loop first pulls the benign 4th line, then returns
    # before processing it. Pull #5 would hit the poison AssertionError.
    backing = _CountingLines([
        "event: delta",
        'data: {"version": 5, "ops": []}',
        "",       # blank → dispatch → sets _force_reconnect
        ":ka",    # benign 4th line pulled by the loop header, guard returns before processing
    ])
    resp = _FakeResponse(backing)

    # GREEN-post-fix: clean return, no AssertionError propagated.
    client._pump(resp)

    assert client._force_reconnect is True          # flag set on the unhealthy branch
    assert client._last_event_id is None            # cursor dropped → fresh snapshot on reconnect
    assert backing.pulls == 4                        # exactly frame(3) + benign(1); poison never pulled
    assert not state.is_cache_healthy()              # cache stayed unhealthy (no snapshot applied)


# ── Assertion 5: flag reset per-connection at pump-entry ──────────────────
def test_pump_resets_flag_at_entry():
    """A flag left True from a prior connection must NOT immediately kill the
    next connection: _pump resets it at entry, so a healthy snapshot lands."""
    state.reset_pack()
    client = _make_client()
    client._force_reconnect = True  # simulate a leaked flag from connection N
    now = time.time()
    client._last_event_at = now
    client._last_data_event_at = now

    snap = _snapshot()
    import json

    backing = iter([
        "event: snapshot",
        "id: 1",
        f"data: {json.dumps(snap)}",
        "",  # blank → dispatch snapshot
    ])
    client._pump(_FakeResponse(backing))

    assert client._force_reconnect is False   # reset at pump-entry, snapshot processed
    assert state.get_pack_version() == 1       # healthy snapshot applied (flag did not short-circuit)
    assert state.is_cache_healthy()


# ── Assertion 11: healthy-cache delta never forces a reconnect ────────────
def test_healthy_delta_does_not_force_reconnect():
    import json

    state.reset_pack()
    client = _make_client()
    now = time.time()
    client._last_event_at = now
    client._last_data_event_at = now

    snap = _snapshot()
    delta = {"version": 2, "ops": []}
    backing = iter([
        "event: snapshot",
        "id: 1",
        f"data: {json.dumps(snap)}",
        "",  # snapshot → cache healthy
        "event: delta",
        "id: 2",
        f"data: {json.dumps(delta)}",
        "",  # healthy delta applies in place
    ])
    client._pump(_FakeResponse(backing))

    assert client._force_reconnect is False    # healthy path never sets the flag
    assert state.get_pack_version() == 2        # delta applied
    assert client._last_event_id == 2           # cursor advanced (not dropped)


# ── Assertion 7: forced reconnect drops Last-Event-ID (integration) ───────
def test_reconnect_omits_last_event_id(monkeypatch):
    state.reset_pack()
    client = _make_client()
    # Avoid the real ~1s backoff sleep between connections.
    monkeypatch.setattr(client._stop_event, "wait", lambda *a, **k: None)

    captured_headers = []

    class _Resp:
        def __init__(self, lines):
            self.status_code = 200
            self._lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def iter_lines(self):
            return iter(self._lines)

    conn = {"n": 0}

    def fake_stream(method, url, headers=None, timeout=None):
        captured_headers.append(dict(headers or {}))
        conn["n"] += 1
        if conn["n"] == 1:
            # Unhealthy delta → forces a reconnect.
            return _Resp(["event: delta", 'data: {"version": 5, "ops": []}', ""])
        # 2nd connection: stop the reader so _run_forever exits after this one.
        client.stop()
        return _Resp([])

    monkeypatch.setattr(httpx, "stream", fake_stream)

    client._run_forever()

    assert conn["n"] >= 2                              # reconnected
    assert "Last-Event-ID" not in captured_headers[1]  # fresh snapshot requested
    assert client._last_event_id is None
    # (The test harness itself calls stop() on conn2 only to terminate the loop;
    # the fix never touches _stop_event — verified by code inspection.)
