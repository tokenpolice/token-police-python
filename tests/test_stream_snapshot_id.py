"""
A snapshot that applies successfully but arrives WITHOUT a numeric `id:` line
must NOT be discarded. The dispatch snapshot branch mirrors the delta branch:
advance the reconnect cursor only when the apply succeeds AND the frame carries
a numeric id; when the apply fails, invalidate the cache and drop the cursor so
the next reconnect pulls a fresh snapshot; when the apply succeeds but there is
no numeric id, keep the pack healthy and leave the cursor untouched.

Sibling: token-police-node/tests/streamSnapshotId.test.ts pins the same cases.
"""
import json

from token_police import state, stream


def _make_client() -> stream.StreamClient:
    return stream.StreamClient(
        base_url="http://localhost:15098",
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


# ── ok + numeric id → apply + advance cursor (unchanged happy path) ─────────
def test_snapshot_with_numeric_id_advances_cursor():
    client = _make_client()
    client._dispatch("snapshot", "7", json.dumps(_snapshot()))
    assert state.is_cache_healthy() is True
    assert state.get_pack() is not None
    assert client._last_event_id == 7


# ── ok + NO id → healthy pack, cursor NOT advanced, NO invalidate ───────────
def test_snapshot_without_id_stays_healthy_no_invalidate():
    client = _make_client()
    client._dispatch("snapshot", None, json.dumps(_snapshot()))
    assert state.is_cache_healthy() is True   # the applied pack was kept
    assert state.get_pack() is not None
    assert client._last_event_id is None      # cursor not advanced (no numeric id)


# ── ok + non-numeric id → same: healthy, cursor untouched, NO invalidate ────
def test_snapshot_with_nonnumeric_id_stays_healthy():
    client = _make_client()
    client._dispatch("snapshot", "abc", json.dumps(_snapshot()))
    assert state.is_cache_healthy() is True
    assert state.get_pack() is not None
    assert client._last_event_id is None


# ── ok + NO id must LEAVE an existing cursor as-is (not reset it) ───────────
def test_snapshot_ok_no_id_leaves_existing_cursor_untouched():
    client = _make_client()
    client._last_event_id = 3
    client._dispatch("snapshot", None, json.dumps(_snapshot()))
    assert state.is_cache_healthy() is True
    assert client._last_event_id == 3         # untouched — a missing id is not a failure


# ── apply FAILS → invalidate + drop cursor (fresh snapshot next reconnect) ──
def test_snapshot_apply_failure_invalidates_and_drops_cursor():
    client = _make_client()
    client._last_event_id = 5
    # A snapshot missing tenant/project identity makes apply_snapshot return False
    # and poison the cache — the failure path, even with a numeric id present.
    bad = {"version": 1, "directives": [], "loop_blocks": []}
    client._dispatch("snapshot", "9", json.dumps(bad))
    assert state.is_cache_healthy() is False
    assert state.get_pack() is None
    assert client._last_event_id is None      # cursor dropped on a real apply failure
