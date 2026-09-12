"""
SSE stream liveness (C-14) — `mark_stream_connected` / `mark_stream_disconnected`
/ `is_stream_fresh`. Tracked SEPARATELY from pack health so a disconnect never
invalidates the Decision Pack (last-known-good blocking during an outage is
unchanged design intent) — these three primitives only let the enforcer decide
a locally ALLOWED call whose decision hinged on a streamed entity list has gone
too stale to trust.

`_stream_connected` / `_stream_disconnected_at` / `_stream_generation` are
process-lifetime module globals — state.py deliberately does NOT reset them in
reset_pack() (see the C-14 fix plan). Unlike vitest, pytest does not give each
test FILE its own module registry, so an autouse fixture forces a clean,
never-connected baseline before every test here via monkeypatch (auto-reverted
after each test) — independent of execution order and of any other test/module
in the same pytest session that touches the same globals.

All timing uses a monkeypatched fake clock (`state.time.time`), matching
`tests/test_state.py`'s `_FakeClock` convention — never a real sleep — and
grace-0 boundary checks always advance the fake clock at least a fraction of a
second past the disconnect instant, avoiding a same-instant flake a real clock
would risk.

Sibling: token-police-node/tests/streamFreshness.test.ts pins the same
scenarios against the Node SDK.
"""
import pytest

from token_police import state


class _FakeClock:
    """Monkeypatch-able clock: state.time.time() returns .now."""

    def __init__(self, start=1_700_000_000.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, secs):
        self.now += secs


@pytest.fixture(autouse=True)
def _clean_stream_state(monkeypatch):
    monkeypatch.setattr(state, "_stream_connected", False)
    monkeypatch.setattr(state, "_stream_disconnected_at", None)
    monkeypatch.setattr(state, "_stream_generation", 0)
    yield


def test_never_connected_is_never_fresh():
    assert state.is_stream_fresh(0) is False
    assert state.is_stream_fresh(3600) is False


def test_connected_is_fresh_at_any_grace_including_zero():
    state.mark_stream_connected()
    assert state.is_stream_fresh(0) is True
    assert state.is_stream_fresh(3600) is True


def test_disconnect_fresh_within_grace_stale_beyond(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    gen = state.mark_stream_connected()
    state.mark_stream_disconnected(gen)
    assert state.is_stream_fresh(60) is True  # 0s elapsed
    clock.advance(59)
    assert state.is_stream_fresh(60) is True  # 59s < 60s grace
    clock.advance(1)
    assert state.is_stream_fresh(60) is True  # exactly at the boundary — inclusive
    clock.advance(0.001)
    assert state.is_stream_fresh(60) is False  # just past


def test_disconnect_grace_zero_stale_once_clock_advances(monkeypatch):
    # Fake clock avoids the same-instant flake a real clock would risk at
    # grace=0 (disconnect and freshness check landing at the same timestamp).
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    gen = state.mark_stream_connected()
    state.mark_stream_disconnected(gen)
    assert state.is_stream_fresh(0) is True  # 0 <= 0
    clock.advance(0.001)
    assert state.is_stream_fresh(0) is False  # > 0


def test_repeated_disconnect_same_generation_does_not_refresh_stamp(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    gen = state.mark_stream_connected()
    state.mark_stream_disconnected(gen)  # t=0 — the real disconnect
    clock.advance(40)
    state.mark_stream_disconnected(gen)  # idempotent no-op (a failed reconnect attempt)
    clock.advance(30)  # total elapsed since the ORIGINAL disconnect = 70s
    # grace=60s measured from the ORIGINAL (t=0) disconnect => stale now.
    # If the second call had refreshed the stamp to t=40s, elapsed since it
    # would be only 30s (< 60s) and this would still read fresh.
    assert state.is_stream_fresh(60) is False


def test_earliest_stamp_control_same_elapsed_time_fresh_under_bigger_grace(monkeypatch):
    # Sanity control proving the assertion above is non-vacuous: a SINGLE
    # disconnect with the same total elapsed time (70s) under a grace that
    # covers it (90s) is fresh — confirming the 60s-grace staleness above
    # comes from the EARLIEST stamp, not from some other effect.
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    gen = state.mark_stream_connected()
    state.mark_stream_disconnected(gen)
    clock.advance(70)
    assert state.is_stream_fresh(90) is True


def test_zombie_generation_cannot_mark_a_newer_connection_down(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    gen1 = state.mark_stream_connected()
    gen2 = state.mark_stream_connected()
    assert gen2 != gen1
    state.mark_stream_disconnected(gen1)  # zombie reader's late exit — ignored
    assert state.is_stream_fresh(0) is True  # still connected under gen2
    # the CURRENT generation can still legitimately disconnect
    state.mark_stream_disconnected(gen2)
    clock.advance(0.001)
    assert state.is_stream_fresh(0) is False


def test_reconnect_clears_the_stamp_fresh_again_immediately(monkeypatch):
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    gen = state.mark_stream_connected()
    state.mark_stream_disconnected(gen)
    clock.advance(10_000)  # well beyond any sane grace
    assert state.is_stream_fresh(0) is False  # stale
    state.mark_stream_connected()  # reconnect
    # Connected bypasses the clock check entirely (no residual window).
    assert state.is_stream_fresh(0) is True
