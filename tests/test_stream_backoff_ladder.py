"""
A TRANSIENT SSE connect failure (429 / 5xx / any non-terminal 4xx) used to
bare-`return` out of _connect_and_pump(). A clean return is treated by
_run_forever as a *successful disconnect* → it resets `attempt = 0`,
collapsing the exponential backoff ladder to its ~1s floor — the SDK
tight-loops reconnecting to /v1/guard/stream ~once a second during a
sustained outage (a reconnect storm).

Fix (connect gate): the terminal 401/403/404 branch keeps its `return`, the
non-terminal branch now
    raise RuntimeError(f"stream connect {resp.status_code}")
(stream.py:243) so the EXISTING _run_forever `except` (stream.py:146)
preserves the climbed `attempt` and the next reconnect climbs the ladder
(1→2→4→8…).

Second fix (traffic-evidence gate, RC-2): a clean 200-body EOF is also NOT
automatically a healthy disconnect. Some 200 connections die before ever
delivering a frame or a keepalive (the reconnect-storm bug seen with
Open-WebUI + local collector: the pump never even reaches _dispatch, yet the
old code still reset attempt=0 every time, producing a permanent ~1 Hz
storm). _run_forever now resets `attempt = 0` on a clean return ONLY when
`self._saw_stream_traffic` is True (stream.py:144) — set at frame-dispatch
time or on any SSE comment other than the `: stream-open` banner
(stream.py:316-327, since the banner precedes subscribe/snapshot on every
connection including ones about to fail, so it proves nothing). A no-traffic
EOF preserves `attempt` and climbs exactly like a transient failure.

Sibling: token-police-node/tests/streamConnectBackoff.test.ts pins the same
scenarios. These tests encode assertions 7, 7b, 8, 12, 13.
"""
import json

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


def _run_with_recorded_backoff(monkeypatch, status_code, n_records, lines=()):
    """Drive _run_forever with httpx.stream stubbed to the given status, jitter
    zeroed (random.random → 0.5), and _stop_event.wait recording each requested
    sleep then stopping the loop after n_records so the thread exits. Every
    connect gets a fresh _FakeStreamResp over the same `lines`, so a 200 case
    replays the same SSE traffic (or lack of it) on each reconnect."""
    monkeypatch.setattr(random_module(), "random", lambda: 0.5)

    def fake_stream(method, url, headers=None, timeout=None):
        return _FakeStreamResp(status_code, lines)

    monkeypatch.setattr(httpx, "stream", fake_stream)

    client = _make_client()
    recorded = []

    def fake_wait(sleep_for=None):
        recorded.append(sleep_for)
        if len(recorded) >= n_records:
            client._stop_event.set()  # break the loop after N backoffs
        return True

    monkeypatch.setattr(client._stop_event, "wait", fake_wait)

    client._run_forever()  # runs inline; fake_wait guarantees termination
    return recorded


def random_module():
    # _run_forever calls `random.random()` — patch the module the SUT imported.
    return stream.random


# ── Assertion 7: a 503 connect storm climbs 1→2→4→8… ───────────────────────
def test_503_connect_storm_climbs_the_ladder(monkeypatch):
    recorded = _run_with_recorded_backoff(monkeypatch, 503, n_records=5)
    # GREEN post-fix: the raise preserves `attempt`, so backoff climbs.
    # RED without the fix: the bare return resets attempt=0 → all 1.0.
    assert recorded[0] == 1.0
    assert recorded[1] == 2.0
    assert recorded[2] == 4.0
    assert recorded[3] == 8.0
    for i in range(1, len(recorded)):
        assert recorded[i] > recorded[i - 1]


# ── Assertion 8: a 429 connect storm also climbs (not 503-specific) ─────────
def test_429_connect_storm_also_climbs(monkeypatch):
    recorded = _run_with_recorded_backoff(monkeypatch, 429, n_records=3)
    assert recorded[0] == 1.0
    assert recorded[1] == 2.0
    assert recorded[1] > recorded[0]


# ── Assertion 7b (RC-2): a data-less 200 EOF is a failed connect in disguise
# and MUST climb, exactly like a transient failure ──────────────────────────
def test_200_empty_eof_no_traffic_climbs_the_ladder(monkeypatch):
    # 200 with an immediately-exhausted iter_lines() → _pump returns cleanly
    # having delivered zero frames and zero comments → self._saw_stream_traffic
    # stays False → _run_forever preserves `attempt` instead of resetting it.
    # This is the reconnect-storm bug: every 200 that dies before any traffic
    # used to collapse the ladder to its ~1s floor forever.
    recorded = _run_with_recorded_backoff(monkeypatch, 200, n_records=5, lines=())
    assert recorded[0] == 1.0
    assert recorded[1] == 2.0
    assert recorded[2] == 4.0
    assert recorded[3] == 8.0
    for i in range(1, len(recorded)):
        assert recorded[i] > recorded[i - 1]


def test_200_stream_open_only_then_eof_climbs_the_ladder(monkeypatch):
    # The `: stream-open` banner is written before subscribe/snapshot on EVERY
    # connection, including ones about to fail — it proves nothing about
    # server health, so it must NOT count as traffic evidence. A connection
    # that only ever emits stream-open before EOF climbs exactly like the
    # empty-body case above.
    recorded = _run_with_recorded_backoff(
        monkeypatch, 200, n_records=4, lines=(": stream-open",),
    )
    assert recorded[0] == 1.0
    assert recorded[1] == 2.0
    assert recorded[2] == 4.0
    for i in range(1, len(recorded)):
        assert recorded[i] > recorded[i - 1]


def test_200_keepalive_then_eof_resets_to_floor(monkeypatch):
    # Any comment OTHER than stream-open (e.g. `: keepalive`) IS traffic
    # evidence — it proves the server pipeline is alive even though no data
    # frame was ever sent (e.g. resume-at-head with nothing new to deliver).
    # Every reconnect after a keepalive-then-EOF disconnect stays at the ~1s
    # floor.
    recorded = _run_with_recorded_backoff(
        monkeypatch, 200, n_records=5, lines=(": keepalive",),
    )
    assert len(recorded) >= 5
    assert all(d == 1.0 for d in recorded)


def test_200_snapshot_frame_then_eof_resets_to_floor(monkeypatch):
    # A dispatched snapshot frame is the strongest traffic evidence — a
    # healthy connection that later drops (e.g. server restart) still resets
    # to the ~1s floor. Parity with
    # test_stream_terminal.py::test_terminal_then_200_self_heals, which must
    # stay green.
    snap_lines = ["event: snapshot", "data: " + json.dumps(_snapshot()), ""]
    recorded = _run_with_recorded_backoff(
        monkeypatch, 200, n_records=5, lines=snap_lines,
    )
    assert len(recorded) >= 5
    assert all(d == 1.0 for d in recorded)


# ── Assertion 13: transient path never sets the stop signal itself ──────────
def test_transient_raise_does_not_set_stop_event(monkeypatch):
    # A single direct transient connect raises but must NOT set _stop_event — the
    # stream keeps reconnecting up the ladder (contrast the terminal-status tests).
    def fake_stream(method, url, headers=None, timeout=None):
        return _FakeStreamResp(500)

    monkeypatch.setattr(httpx, "stream", fake_stream)
    client = _make_client()
    try:
        client._connect_and_pump()
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True
    assert client._stop_event.is_set() is False
