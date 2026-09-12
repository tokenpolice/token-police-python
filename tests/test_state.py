"""
`directive_removed` index-miss must self-heal, not silently advance.

The delta reducer indexes directives by `directive["id"]` but the
`directive_removed` handler looks up by `op["rule_id"]`. A truthy rule_id that
misses the index used to be a silent no-op WHILE the version still advanced — a
removed rule would keep enforcing until the next full snapshot. The fix poisons
the cache and returns False on that miss so the SSE caller re-snapshots. Falsy /
empty rid stays a benign no-op (version advances). The Python miss branch sets
`_cache_invalid = True` INLINE (it already holds the non-reentrant `_pack_lock`)
and must NOT call `invalidate_pack()` — that would re-acquire the lock and
deadlock. Sibling: token-police-node/tests/state.test.ts pins the same
scenarios.
"""
import inspect
import threading
import time

from token_police import state


def _snapshot_ab():
    return {
        "version": 1,
        "tenant_id": "t1",
        "project_id": "p1",
        "directives": [
            {"id": "A", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce"},
            {"id": "B", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce"},
        ],
        "loop_blocks": [],
    }


def setup_function(_fn):
    state.reset_pack()


def test_directive_removed_miss_returns_false_and_poisons_cache():
    # #2 (miss -> False) + #4 (cache poisoned -> self-heal)
    assert state.apply_snapshot(_snapshot_ab()) is True
    ok = state.apply_deltas([{"op": "directive_removed", "rule_id": "ZZZ"}], 2)
    assert ok is False
    assert state.get_pack() is None
    assert state.is_cache_healthy() is False


def test_directive_removed_miss_does_not_advance_version():
    # #3
    state.apply_snapshot(_snapshot_ab())
    state.apply_deltas([{"op": "directive_removed", "rule_id": "ZZZ"}], 2)
    assert state.get_pack_version() == 1  # NOT 2 — no silent advance


def test_directive_removed_miss_then_fresh_snapshot_rebuilds_whole_pack():
    # #5
    state.apply_snapshot(_snapshot_ab())
    state.apply_deltas([{"op": "directive_removed", "rule_id": "ZZZ"}], 2)
    assert state.get_pack() is None
    assert state.apply_snapshot(_snapshot_ab()) is True
    pack = state.get_pack()
    assert pack is not None
    assert [d["id"] for d in pack["directives"]] == ["A", "B"]
    assert state.is_cache_healthy() is True


def test_directive_removed_hit_path_byte_identical():
    # #6: present rid removes A, advances to v2, rebuilds index
    state.apply_snapshot(_snapshot_ab())
    ok = state.apply_deltas([{"op": "directive_removed", "rule_id": "A"}], 2)
    assert ok is True
    assert state.get_pack_version() == 2
    assert [d["id"] for d in state.get_pack()["directives"]] == ["B"]
    assert state.is_cache_healthy() is True
    # index rebuilt correctly: removing the shifted entry still works
    ok2 = state.apply_deltas([{"op": "directive_removed", "rule_id": "B"}], 3)
    assert ok2 is True
    assert state.get_pack()["directives"] == []


def test_directive_removed_falsy_rid_is_benign_no_op():
    # #7: falsy/empty rid -> version advances, directives unchanged, cache healthy
    for bad_rid, include in (("", True), (None, True), (None, False)):
        state.reset_pack()
        state.apply_snapshot(_snapshot_ab())
        op = {"op": "directive_removed"}
        if include:
            op["rule_id"] = bad_rid
        ok = state.apply_deltas([op], 2)
        assert ok is True
        assert state.get_pack_version() == 2
        assert [d["id"] for d in state.get_pack()["directives"]] == ["A", "B"]
        assert state.is_cache_healthy() is True


def test_directive_removed_dup_delta_short_circuits_before_loop():
    # #9: new_version <= _pack_version -> idempotent discard, never hits new branch
    state.apply_snapshot(_snapshot_ab())
    ok = state.apply_deltas([{"op": "directive_removed", "rule_id": "ZZZ"}], 1)
    assert ok is True
    assert state.get_pack_version() == 1
    assert state.is_cache_healthy() is True
    assert [d["id"] for d in state.get_pack()["directives"]] == ["A", "B"]


def test_directive_removed_miss_no_deadlock_static_pin():
    # #10 PRIMARY (static pin): the miss branch must NOT call invalidate_pack()
    # (re-acquires the non-reentrant _pack_lock -> deadlock). A regression that
    # re-introduces the call fails here deterministically.
    src = inspect.getsource(state.apply_deltas)
    assert "invalidate_pack(" not in src


def test_directive_removed_miss_no_deadlock_runtime_hang_detector():
    # #10 BACKSTOP (mandatory runtime hang-detector): run the miss-path
    # apply_deltas in a worker thread; a deadlock manifests as the thread still
    # alive after 2s (deterministic FAILURE), never a 120s Bash hang.
    state.apply_snapshot(_snapshot_ab())
    result = {}

    def worker():
        result["ret"] = state.apply_deltas(
            [{"op": "directive_removed", "rule_id": "ZZZ"}], 2
        )

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "apply_deltas deadlocked on the miss path"
    assert result.get("ret") is False


# ─────────────────────────────────────────────────────────────────────────
# Delta entity-apply must clone-before-mutate (COW) so a concurrent
# daemon-mode local eval that captured a directive dict via get_pack() is never
# mutated under it. The reducer shallow-copies the directive LIST but used to
# mutate the dict ELEMENTS in place (`d = directives[...]; d["entities"] = ...`)
# — those dicts are the SAME objects the live _pack references. Fix: clone the
# dict inline (`dict(...)`, no _pack_lock-reacquiring helper deadlock
# hazard) before mutating `entities`, then write the clone back into the list.
# Sibling: token-police-node/tests/state.test.ts pins the same scenarios.
# ─────────────────────────────────────────────────────────────────────────


def _snapshot_r1(entities=None):
    return {
        "version": 1,
        "tenant_id": "t1",
        "project_id": "p1",
        "directives": [
            {"id": "R1", "kind": "ENTITY_BLOCK", "mode": "enforce",
             "entities": list(entities or [])},
        ],
        "loop_blocks": [],
    }


def test_entity_blocked_reader_isolation_red_without_fix():
    # #5 (arm): capture the live directive dict, then arm an entity. The captured
    # dict must NOT be mutated, and the live pack must now hold a DIFFERENT dict
    # object. RED pre-fix (shared object mutated + same identity), GREEN post-fix.
    assert state.apply_snapshot(_snapshot_r1([])) is True
    d_before = state.get_pack()["directives"][0]
    assert state.apply_deltas(
        [{"op": "entity_blocked", "rule_id": "R1", "entity": "user:42"}], 2
    ) is True
    # captured dict untouched by the writer
    assert d_before.get("entities") == []
    # live pack now holds a fresh clone, not the object the reader captured
    assert d_before is not state.get_pack()["directives"][0]


def test_entity_unblocked_reader_isolation_red_without_fix():
    # #5 (disarm): start armed, capture the live dict, disarm. Captured dict keeps
    # its entity; live pack holds a different (now-empty) dict object.
    assert state.apply_snapshot(_snapshot_r1(["user:42"])) is True
    d_before = state.get_pack()["directives"][0]
    assert state.apply_deltas(
        [{"op": "entity_unblocked", "rule_id": "R1", "entity": "user:42"}], 2
    ) is True
    assert d_before.get("entities") == ["user:42"]
    assert d_before is not state.get_pack()["directives"][0]


def test_entity_rerouted_reader_isolation():
    # #5 parity for the reroute arm arm (entity_rerouted shares the branch).
    assert state.apply_snapshot(
        {"version": 1, "tenant_id": "t1", "project_id": "p1",
         "directives": [{"id": "R1", "kind": "REROUTE", "mode": "enforce", "entities": []}],
         "loop_blocks": []}
    ) is True
    d_before = state.get_pack()["directives"][0]
    assert state.apply_deltas(
        [{"op": "entity_rerouted", "rule_id": "R1", "entity": "user:9"}], 2
    ) is True
    assert d_before.get("entities") == []
    assert d_before is not state.get_pack()["directives"][0]


def test_entity_apply_outcome_unchanged():
    # #7: the fix isolates the reader WITHOUT changing the applied result. Arm →
    # entity present + version advanced; disarm → entity gone; de-dup → no dupes.
    state.apply_snapshot(_snapshot_r1([]))
    assert state.apply_deltas(
        [{"op": "entity_blocked", "rule_id": "R1", "entity": "user:42"}], 2
    ) is True
    assert state.get_pack()["directives"][0]["entities"] == ["user:42"]
    assert state.get_pack_version() == 2
    # de-dup: arming an already-armed entity yields no duplicate
    assert state.apply_deltas(
        [{"op": "entity_blocked", "rule_id": "R1", "entity": "user:42"}], 3
    ) is True
    assert state.get_pack()["directives"][0]["entities"] == ["user:42"]
    # disarm removes it
    assert state.apply_deltas(
        [{"op": "entity_unblocked", "rule_id": "R1", "entity": "user:42"}], 4
    ) is True
    assert state.get_pack()["directives"][0]["entities"] == []


def test_entity_apply_missing_rule_is_noop():
    # #8 guard: an entity op for a rule absent from the index applies nothing but
    # still advances the version (branch unchanged besides the clone).
    state.apply_snapshot(_snapshot_r1([]))
    assert state.apply_deltas(
        [{"op": "entity_blocked", "rule_id": "NOPE", "entity": "user:1"}], 2
    ) is True
    assert state.get_pack_version() == 2
    assert state.get_pack()["directives"][0]["entities"] == []


def test_entity_apply_falsy_rid_or_none_entity_continue_guard():
    # #8 guard: `not rid or entity is None` still short-circuits (continue), pack
    # unchanged, version advances, cache healthy.
    state.apply_snapshot(_snapshot_r1([]))
    ok = state.apply_deltas([
        {"op": "entity_blocked", "rule_id": "", "entity": "user:1"},
        {"op": "entity_blocked", "rule_id": "R1", "entity": None},
        {"op": "entity_unblocked", "rule_id": "R1"},
    ], 2)
    assert ok is True
    assert state.get_pack_version() == 2
    assert state.get_pack()["directives"][0]["entities"] == []
    assert state.is_cache_healthy() is True


def test_upsert_carry_forward_not_cloned_and_prev_uncorrupted():
    # #9: the directive_upserted carry-forward is deliberately NOT cloned and must
    # stay behaviorally intact — an upsert omitting `entities` carries the prior
    # armed set forward; the prior object read from is not corrupted.
    state.apply_snapshot(_snapshot_r1(["user:7"]))
    ok = state.apply_deltas([
        {"op": "directive_upserted",
         "directive": {"id": "R1", "kind": "ENTITY_BLOCK", "mode": "enforce"}},
    ], 2)
    assert ok is True
    assert state.get_pack()["directives"][0]["entities"] == ["user:7"]


def test_entity_apply_no_deadlock_static_pin():
    # #10 PRIMARY (static pin, broadened): the reducer must NOT call ANY helper
    # that re-acquires the non-reentrant _pack_lock while holding it — that is the
    # Deadlock hazard. A regression reaching for any of these inside the held
    # loop fails here deterministically.
    src = inspect.getsource(state.apply_deltas)
    for helper in (
        "invalidate_pack(",
        "get_pack(",
        "get_pack_version(",
        "is_cache_healthy(",
        "apply_snapshot(",
        "reset_pack(",
    ):
        assert helper not in src, f"apply_deltas must not call {helper} under _pack_lock"


def test_entity_apply_no_deadlock_runtime_hang_detector():
    # #10 BACKSTOP (runtime hang-detector, non-vacuous): snapshot a v1 pack whose
    # directive id == the ops' rule_id ("R1"), so `rid in dir_by_id` is True and
    # the clone runs UNDER the held _pack_lock. Carry BOTH an entity_blocked and
    # an entity_unblocked op for R1 so both cloned branches execute. A deadlock
    # manifests as the thread still alive after 2s (deterministic FAILURE), never
    # a 120s Bash hang.
    assert state.apply_snapshot(_snapshot_r1(["seed"])) is True
    result = {}

    def worker():
        result["ret"] = state.apply_deltas([
            {"op": "entity_blocked", "rule_id": "R1", "entity": "user:42"},
            {"op": "entity_unblocked", "rule_id": "R1", "entity": "seed"},
        ], 2)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "apply_deltas deadlocked on the entity clone path"
    assert result.get("ret") is True


# ─────────────────────────────────────────────────────────────────────────
# Singleton re-init swaps the client under _instance_lock (no
# concurrency guard before). Sibling: token-police-node/tests/state.test.ts
# documents the Node event-loop-atomicity parity note.
# ─────────────────────────────────────────────────────────────────────────


class _CountingClient:
    """Stub client whose close_sync just counts invocations."""

    def __init__(self):
        self.close_calls = 0

    def close_sync(self):
        self.close_calls += 1


def test_set_client_body_runs_under_instance_lock():
    # Assertion #1(a): whole set_client body is wrapped in `with _instance_lock:`
    src = inspect.getsource(state.set_client)
    assert "with _instance_lock:" in src
    assert isinstance(state._instance_lock, type(threading.Lock()))


def test_get_client_is_not_locked_hot_read_path():
    # Assertion #3(b): get_client stays a bare, lock-free read.
    src = inspect.getsource(state.get_client)
    assert "_instance_lock" not in src
    assert "with " not in src
    assert "return _instance" in src


def test_instance_lock_disjoint_from_pack_and_observations_locks():
    # Assertion #4(c) static pin: distinct lock objects → no deadlock coupling.
    assert state._instance_lock is not state._pack_lock
    assert state._instance_lock is not state._observations_lock


def test_reset_pack_and_close_sync_do_not_acquire_instance_lock():
    # Assertion #4(c): neither callee inside set_client re-acquires _instance_lock.
    from token_police.client import TokenPolice

    assert "_instance_lock" not in inspect.getsource(state.reset_pack)
    assert "_instance_lock" not in inspect.getsource(TokenPolice.close_sync)


def test_set_client_concurrent_swap_is_serialized():
    # Assertion #2 (a-runtime), the key one: while thread A is mid-teardown of
    # the previously-installed client (blocked inside close_sync while HOLDING
    # _instance_lock), thread B's set_client must NOT begin its own swap. We
    # observe the timing: B has not closed the client A installed, and A's
    # close has not returned, until the gate is released. Then serialization
    # yields a deterministic final _instance and no exception escapes.
    events = []
    ev_lock = threading.Lock()

    def rec(tag):
        with ev_lock:
            events.append(tag)

    gate = threading.Event()               # blocks c0.close_sync
    c0_close_started = threading.Event()

    class BlockingClient:
        def close_sync(self):
            rec("c0_close_start")
            c0_close_started.set()
            gate.wait(timeout=5.0)         # bounded so a broken test never hangs
            rec("c0_close_end")

    class RecordingClient:
        def __init__(self, name):
            self.name = name

        def close_sync(self):
            rec("close_" + self.name)

    stub_a = RecordingClient("A")
    stub_b = RecordingClient("B")

    # Install the blocking client cleanly as the starting _instance.
    state._instance = BlockingClient()

    errors = []

    def call_set(client, tag):
        try:
            state.set_client(client)
        except Exception as e:  # pragma: no cover — must never happen
            errors.append((tag, e))

    ta = threading.Thread(target=call_set, args=(stub_a, "A"))
    ta.start()
    # A is now inside c0.close_sync, holding _instance_lock.
    assert c0_close_started.wait(timeout=2.0)

    tb = threading.Thread(target=call_set, args=(stub_b, "B"))
    tb.start()
    # Give B a real chance to (try to) acquire the lock and start swapping.
    time.sleep(0.2)

    # TIMING OBSERVATION: with the gate still closed, A holds the lock, so
    # neither has A's close returned nor has B begun its swap (B swaps stub_a
    # out, so close_A would be its first side effect).
    with ev_lock:
        seen = list(events)
    assert "c0_close_end" not in seen, "A's teardown returned before gate release"
    assert "close_A" not in seen, "B's swap began while A still held the lock"

    # Release: A completes its swap, then B acquires the lock and swaps.
    gate.set()
    ta.join(timeout=2.0)
    tb.join(timeout=2.0)
    assert not ta.is_alive() and not tb.is_alive()
    assert errors == []                    # Golden Rule: no exception escaped

    # Deterministic final state: B was installed last.
    assert state.get_client() is stub_b
    # Ordering proof: B serialized behind A — it closed the client A INSTALLED
    # (stub_a), not the original c0, and only after A's close returned. Without
    # the lock both threads read c0 and tear it down concurrently, so close_A
    # never happens (the swap was not serialized).
    with ev_lock:
        order = list(events)
    assert "close_A" in order, "B did not serialize behind A's swap (no lock?)"
    assert order.index("c0_close_end") < order.index("close_A")

    state._instance = None                 # cleanup for sibling tests


def test_set_client_no_deadlock_bounded_completion():
    # Assertion #5 (c-runtime): a normal swap in a worker thread completes
    # within a bounded join → a re-entrant/deadlocking lock would fail here
    # deterministically instead of hanging Bash for 120s.
    state._instance = None
    result = {}

    def worker():
        state.set_client(_CountingClient())
        result["done"] = True

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=2.0)
    assert not t.is_alive(), "set_client deadlocked"
    assert result.get("done") is True
    state._instance = None


def test_reinit_closes_old_client_once_and_resets_pack():
    # Assertion #6 (d): normal single-threaded re-init byte-identical behavior.
    state._instance = None
    c1 = _CountingClient()
    c2 = _CountingClient()

    state.set_client(c1)
    # populate the pack to prove the swap resets it
    state.apply_snapshot(_snapshot_ab())
    assert state.get_pack() is not None

    state.set_client(c2)
    assert c1.close_calls == 1              # old client closed exactly once
    assert c2.close_calls == 0
    assert state.get_client() is c2
    assert state.get_pack() is None
    assert state.get_pack_version() == 0
    assert state.is_cache_healthy() is False
    state._instance = None


def test_first_init_closes_nothing():
    # Assertion #7 (d-first-init): first install (from None) closes no client.
    state._instance = None
    c1 = _CountingClient()
    state.set_client(c1)
    assert c1.close_calls == 0
    assert state.get_client() is c1
    state._instance = None


# ─────────────────────────────────────────────────────────────────────────
# The server stamps every /stream snapshot with a TTL (`ttl_seconds`).
# Neither SDK enforced it — an aged-out Decision Pack stayed "healthy" and kept
# enforcing while heartbeats arrived but no deltas/snapshots. Fix: store the TTL
# on snapshot, add a pure receipt-based `is_pack_expired()` predicate (ttl<=0 ⇒
# never expires) delegating to a LOCK-FREE `_pack_expired_unlocked()` helper, and
# gate BOTH read accessors on it (inline under the held _pack_lock — never via
# the lock-acquiring public predicate: non-reentrant deadlock). Receipt
# refreshes on a successful delta too. RED-without-fix: pre-fix get_pack() /
# is_cache_healthy() ignore age. A controllable clock avoids any 24h sleep.
# Sibling: token-police-node/tests/state.test.ts pins the same scenarios.
# ─────────────────────────────────────────────────────────────────────────


class _FakeClock:
    """Monkeypatch-able clock: state.time.time() returns .now."""

    def __init__(self, start=1_700_000_000.0):
        self.now = start

    def time(self):
        return self.now

    def advance(self, secs):
        self.now += secs


def _ttl_snapshot(ttl):
    snap = {
        "version": 1,
        "tenant_id": "t1",
        "project_id": "p1",
        "directives": [{"id": "A", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce"}],
        "loop_blocks": [],
    }
    if ttl is not None:
        snap["ttl_seconds"] = ttl
    return snap


def test_ttl_predicate_gate_and_boundary(monkeypatch):
    # Assertion #4: absent/<=0 TTL never expires; seconds boundary (no ×1000).
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)

    # absent ttl_seconds ⇒ never expires
    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(None)) is True
    clock.advance(10 ** 8)
    assert state.is_pack_expired() is False

    # ttl_seconds <= 0 ⇒ never expires
    for bad in (0, -5):
        state.reset_pack()
        assert state.apply_snapshot(_ttl_snapshot(bad)) is True
        clock.advance(10 ** 8)
        assert state.is_pack_expired() is False

    # boundary: age exactly ttl ⇒ NOT expired; ttl+epsilon ⇒ expired
    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(100)) is True
    clock.advance(100)  # age == 100, NOT > 100
    assert state.is_pack_expired() is False
    clock.advance(0.001)  # just past
    assert state.is_pack_expired() is True


def test_ttl_predicate_null_pack(monkeypatch):
    # Assertion #4: null pack is never expired.
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    state.reset_pack()
    assert state.is_pack_expired() is False


def test_expired_pack_read_gates_red_without_fix(monkeypatch):
    # Assertion #6, #7, #15 (RED-pre / GREEN-post): an aged-past-TTL pack makes
    # get_pack() None and is_cache_healthy() False. Pre-fix both ignored age.
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(100)) is True
    # pre-expiry: healthy
    assert state.get_pack() is not None
    assert state.is_cache_healthy() is True
    # age past TTL
    clock.advance(101)
    assert state.is_pack_expired() is True
    assert state.get_pack() is None          # RED pre-fix (returned the pack)
    assert state.is_cache_healthy() is False  # RED pre-fix (returned True)


def test_no_ttl_snapshot_byte_identical(monkeypatch):
    # Assertion #12: a TTL-less snapshot behaves exactly as today after any span.
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(None)) is True
    clock.advance(10 ** 9)
    assert state.get_pack() is not None
    assert state.is_cache_healthy() is True
    assert state.is_pack_expired() is False


def test_delta_apply_refreshes_receipt(monkeypatch):
    # Assertion #5: a successful delta resets the receipt clock so an
    # actively-updated pack never falsely expires.
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(100)) is True
    clock.advance(90)  # 90s of the 100s window elapsed
    assert state.apply_deltas([], 2) is True  # contiguous delta → refresh
    clock.advance(90)  # 180s since snapshot, only 90s since delta
    assert state.is_pack_expired() is False
    assert state.get_pack() is not None
    assert state.is_cache_healthy() is True
    clock.advance(11)  # 101s since the delta
    assert state.is_pack_expired() is True


def test_ttl_stored_and_cleared(monkeypatch):
    # Assertion #1, #2: TTL stored on snapshot; reset_pack clears it so a stale
    # window can't leak into a later TTL-less snapshot.
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    state.reset_pack()
    assert state.apply_snapshot(_ttl_snapshot(50)) is True
    assert state._pack_ttl_seconds == 50
    clock.advance(60)
    assert state.is_pack_expired() is True
    state.reset_pack()
    assert state._pack_ttl_seconds == 0
    # a fresh TTL-less snapshot must NOT inherit the old 50s TTL
    assert state.apply_snapshot(_ttl_snapshot(None)) is True
    clock.advance(10 ** 6)
    assert state.is_pack_expired() is False


def test_ttl_non_numeric_coerces_to_zero(monkeypatch):
    # Assertion #1: a non-numeric ttl_seconds coerces to 0 (never expires) and
    # does NOT poison the snapshot.
    clock = _FakeClock()
    monkeypatch.setattr(state.time, "time", clock.time)
    state.reset_pack()
    snap = _ttl_snapshot(100)
    snap["ttl_seconds"] = "not-a-number"
    assert state.apply_snapshot(snap) is True
    assert state._pack_ttl_seconds == 0
    clock.advance(10 ** 8)
    assert state.is_pack_expired() is False
    assert state.get_pack() is not None


def test_ttl_expiry_no_deadlock_static_pin():
    # Assertion #13 PRIMARY (static pins): the non-reentrant _pack_lock
    # discipline. The lock-free helper must NOT itself take _pack_lock, and the
    # accessors that already hold the lock must NOT call the lock-acquiring
    # public is_pack_expired() (→ re-acquire → deadlock).
    def _code_only(fn):
        # strip comment lines so the pin inspects executable code, not the
        # explanatory comments (which legitimately name is_pack_expired).
        return "\n".join(
            line for line in inspect.getsource(fn).splitlines()
            if not line.strip().startswith("#")
        )

    unlocked_src = _code_only(state._pack_expired_unlocked)
    assert "with _pack_lock" not in unlocked_src
    for accessor in (state.get_pack, state.is_cache_healthy):
        src = _code_only(accessor)
        assert "is_pack_expired(" not in src, (
            f"{accessor.__name__} must inline _pack_expired_unlocked(), "
            "never the lock-acquiring is_pack_expired()"
        )
        assert "_pack_expired_unlocked(" in src


def test_ttl_expiry_no_deadlock_runtime_hang_detector():
    # Assertion #13 BACKSTOP (runtime hang-detector): a worker thread hammers the
    # expiry accessors while the main thread applies snapshots/deltas and
    # invalidates. A deadlock (re-entrant _pack_lock) manifests as the thread
    # still alive after 5s — deterministic FAILURE, never a 120s Bash hang.
    state.reset_pack()
    state.apply_snapshot(_ttl_snapshot(100))
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            state.get_pack()
            state.is_cache_healthy()
            state.is_pack_expired()

    t = threading.Thread(target=worker)
    t.start()
    try:
        for i in range(200):
            state.apply_snapshot(_ttl_snapshot(100))
            state.apply_deltas([{"op": "loop_blocked", "trace_id": f"tr{i}"}],
                               state.get_pack_version() + 1)
            state.invalidate_pack("test")
    finally:
        stop.set()
        t.join(timeout=5.0)
    assert not t.is_alive(), "expiry accessors deadlocked under the held _pack_lock"
