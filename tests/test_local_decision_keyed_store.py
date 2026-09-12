"""B3 — per-call keyed store for the applied `local_decision` audit stash.

Bug: `local_decision` (the SOLE provenance the collector turns into a
`REQUEST_REROUTED` audit event) used to live in ONE flat slot on the session
(`session._local_decision`). N concurrent calls overwrote each other (N-1
decisions lost) and the first call to finish drained the survivor: one
applied-reroute event per burst, stamped on an arbitrary row with an
arbitrary sibling's rule/from/to — and in a heterogeneous burst, an
un-rerouted call's row could carry a sibling's decision (a false-positive
REQUEST_REROUTED).

Fix: `session._local_decisions` — a bounded per-call keyed FIFO list of
`{ld, key, ts}` — with `_stash_local_decision_entry` / `_claim_local_decision`
/ `_drop_local_decision` helpers in `token_police/enforcer.py` (module-local;
NOT exported from state.py — see the fix plan's D2 for why). This file covers:

  (1) store unit tests — claim precedence, at-most-one, replace, cap, sweep,
      no-stale-claim, drop-with-predicate, hostile inputs;
  (2) thread safety — N threads stash+claim their OWN keys on ONE session;
  (3) end-to-end asyncio — N concurrent /check+/log calls on ONE session,
      each carrying its OWN decision, 0 residual;
  (4) heterogeneous burst — rerouted + clean calls mixed, no false positive;
  (5) the deferred-flush steal window — a stash landing mid-flight survives.

Mirrors the harness idioms in tests/test_reroute_stash_session_thread.py
(asyncio-check driving) and tests/test_bedrock_failure_single_emitter.py
(store helper access via `_enforcer._stash_local_decision_entry` etc).
"""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as _state
from token_police.context import TPSession


# ══════════════════════════════════════════════════════════════════════════
# (1) Store unit tests
# ══════════════════════════════════════════════════════════════════════════
class TestLocalDecisionStoreUnit(unittest.TestCase):
    def test_claim_own_key_precedence_over_untagged(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "untagged"}, None)
        _enforcer._stash_local_decision_entry(s, {"id": "own"}, "key-a")

        claimed = _enforcer._claim_local_decision(s, "key-a")
        self.assertEqual(claimed, {"id": "own"})
        # The untagged entry is untouched — it stays for its own eventual
        # (degraded-path) claim.
        remaining = [e["ld"] for e in s._local_decisions]
        self.assertEqual(remaining, [{"id": "untagged"}])

    def test_claim_falls_back_to_untagged_when_no_own_key_entry(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "untagged"}, None)

        claimed = _enforcer._claim_local_decision(s, "some-other-key")
        self.assertEqual(claimed, {"id": "untagged"})
        self.assertEqual(s._local_decisions, [])

    def test_claim_is_at_most_one_even_with_multiple_untagged(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "a"}, None)
        _enforcer._stash_local_decision_entry(s, {"id": "b"}, None)

        claimed = _enforcer._claim_local_decision(s, "any-key")
        # Newest untagged entry wins; exactly one is consumed.
        self.assertEqual(claimed, {"id": "b"})
        remaining = [e["ld"] for e in s._local_decisions]
        self.assertEqual(remaining, [{"id": "a"}])

    def test_same_key_stash_replaces_not_appends(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "first"}, "key-a")
        _enforcer._stash_local_decision_entry(s, {"id": "second"}, "key-a")

        self.assertEqual(len(s._local_decisions), 1)
        self.assertEqual(
            _enforcer._claim_local_decision(s, "key-a"), {"id": "second"})

    def test_foreign_key_claim_returns_none_and_leaves_entry_reachable(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "mine"}, "key-a")

        # A stranger's claim (no untagged entry to fall back to) must find
        # nothing — the entry stays put for ITS OWN eventual drain.
        self.assertIsNone(_enforcer._claim_local_decision(s, "key-b"))
        self.assertEqual(len(s._local_decisions), 1)
        self.assertEqual(
            _enforcer._claim_local_decision(s, "key-a"), {"id": "mine"})

    def test_cap_drops_oldest(self):
        s = SimpleNamespace()
        cap = _enforcer._LOCAL_DECISION_CAP
        total = cap + 6
        for i in range(total):
            _enforcer._stash_local_decision_entry(s, {"id": i}, f"key-{i}")

        self.assertEqual(len(s._local_decisions), cap)
        ids = [e["ld"]["id"] for e in s._local_decisions]
        # The oldest 6 (0..5) were dropped; 6..(total-1) survive, oldest-first.
        self.assertEqual(ids, list(range(total - cap, total)))

    def test_sweep_on_claim_retires_expired_even_by_its_own_key(self):
        """D3: NO stale-claim arm — an orphan is swept, never handed to
        anyone, not even the call whose key it was stashed under."""
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "stale"}, "key-a")
        s._local_decisions[0]["ts"] -= (
            _enforcer.LOCAL_DECISION_STALE_SECONDS + 1)

        self.assertIsNone(_enforcer._claim_local_decision(s, "key-a"))
        self.assertEqual(s._local_decisions, [])

    def test_sweep_on_stash_retires_expired(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "stale"}, "key-a")
        s._local_decisions[0]["ts"] -= (
            _enforcer.LOCAL_DECISION_STALE_SECONDS + 1)

        _enforcer._stash_local_decision_entry(s, {"id": "fresh"}, "key-b")

        ids = [e["ld"]["id"] for e in s._local_decisions]
        self.assertEqual(ids, ["fresh"])

    def test_no_stale_claim_by_a_stranger(self):
        """An expired UNTAGGED entry is swept, not handed to a claim that
        would otherwise have fallen back to it."""
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "stale"}, None)
        s._local_decisions[0]["ts"] -= (
            _enforcer.LOCAL_DECISION_STALE_SECONDS + 1)

        self.assertIsNone(_enforcer._claim_local_decision(s, "whoever-asks"))
        self.assertEqual(s._local_decisions, [])

    def test_drop_with_predicate_matches_own_key_only(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(
            s, {"outcome": "rerouted"}, "key-a")
        _enforcer._stash_local_decision_entry(
            s, {"outcome": "rerouted"}, "key-b")

        _enforcer._drop_local_decision(
            s, "key-a", predicate=lambda ld: ld.get("outcome") == "rerouted")

        remaining_keys = [e["key"] for e in s._local_decisions]
        self.assertEqual(remaining_keys, ["key-b"])

    def test_drop_predicate_false_leaves_entry_untouched(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(
            s, {"outcome": "blocked"}, "key-a")

        _enforcer._drop_local_decision(
            s, "key-a", predicate=lambda ld: ld.get("outcome") == "rerouted")

        self.assertEqual(len(s._local_decisions), 1)
        self.assertEqual(
            _enforcer._claim_local_decision(s, "key-a"),
            {"outcome": "blocked"})

    def test_drop_none_key_targets_newest_untagged(self):
        s = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s, {"id": "a"}, None)
        _enforcer._stash_local_decision_entry(s, {"id": "b"}, None)

        _enforcer._drop_local_decision(s, None)

        remaining = [e["ld"] for e in s._local_decisions]
        self.assertEqual(remaining, [{"id": "a"}])

    # ── Golden rule: every helper is fail-open — never raises, on anything ──
    def test_hostile_inputs_never_raise(self):
        # None session.
        _enforcer._stash_local_decision_entry(None, {"id": 1}, "k")
        self.assertIsNone(_enforcer._claim_local_decision(None, "k"))
        _enforcer._drop_local_decision(None, "k")

        # Missing attribute (never stashed to).
        s = SimpleNamespace()
        self.assertIsNone(_enforcer._claim_local_decision(s, "k"))
        _enforcer._drop_local_decision(s, "k")

        # Corrupt attribute (not a list) — claim/drop degrade to no-op;
        # a later stash self-heals by replacing it with a fresh list.
        s2 = SimpleNamespace(_local_decisions="not-a-list")
        self.assertIsNone(_enforcer._claim_local_decision(s2, "k"))
        _enforcer._drop_local_decision(s2, "k")
        _enforcer._stash_local_decision_entry(s2, {"id": 1}, "k")
        self.assertIsInstance(s2._local_decisions, list)

        # Falsy ld (None / {}) — never stashed.
        s3 = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s3, None, "k")
        _enforcer._stash_local_decision_entry(s3, {}, "k")
        # Falsy `ld` never even creates the list.
        self.assertIsNone(getattr(s3, "_local_decisions", None))

        # A predicate that raises must not blow up the drop.
        s4 = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s4, {"id": 1}, "k")

        def _hostile_predicate(_ld):
            raise RuntimeError("boom")

        _enforcer._drop_local_decision(s4, "k", predicate=_hostile_predicate)
        # A raising predicate is treated as non-matching (fail-open) — the
        # entry survives, still reachable by its own key.
        self.assertEqual(
            _enforcer._claim_local_decision(s4, "k"), {"id": 1})

        # Non-string (but truthy) key must not raise, and stash/claim stay
        # consistent as long as the same value is used both times.
        s5 = SimpleNamespace()
        _enforcer._stash_local_decision_entry(s5, {"id": "num"}, 12345)
        self.assertEqual(
            _enforcer._claim_local_decision(s5, 12345), {"id": "num"})


# ══════════════════════════════════════════════════════════════════════════
# (2) Thread safety — N threads stash+claim their OWN keys on ONE session
# ══════════════════════════════════════════════════════════════════════════
class TestLocalDecisionThreadSafety(unittest.TestCase):
    def test_n_threads_own_keys_zero_lost_zero_residual(self):
        session = SimpleNamespace()
        n_threads = 8
        n_rounds = 200
        errors = []

        def worker(tid):
            try:
                for i in range(n_rounds):
                    key = f"t{tid}-{i}"
                    ld = {"tid": tid, "i": i}
                    _enforcer._stash_local_decision_entry(session, ld, key)
                    claimed = _enforcer._claim_local_decision(session, key)
                    if claimed != ld:
                        errors.append((tid, i, claimed))
            except Exception as exc:  # pragma: no cover - diagnostic only
                errors.append((tid, "exception", repr(exc)))

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        # Every thread claimed its own stash — nothing left behind.
        self.assertFalse(getattr(session, "_local_decisions", None))

    def test_n_threads_racing_the_same_key_never_corrupts_the_list(self):
        """A harsher variant: all threads hammer the SAME key (replace
        semantics), so the list itself is what's contended, not just
        independent slots. Must never raise and must never leave duplicate
        entries under one key."""
        session = SimpleNamespace()
        n_threads = 8
        n_rounds = 200
        errors = []

        def worker(tid):
            try:
                for i in range(n_rounds):
                    _enforcer._stash_local_decision_entry(
                        session, {"tid": tid, "i": i}, "shared-key")
            except Exception as exc:  # pragma: no cover
                errors.append((tid, repr(exc)))

        threads = [threading.Thread(target=worker, args=(t,))
                   for t in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        # Same-key replace: never more than one entry under "shared-key".
        matching = [e for e in session._local_decisions
                    if e.get("key") == "shared-key"]
        self.assertLessEqual(len(matching), 1)


# ══════════════════════════════════════════════════════════════════════════
# Shared asyncio driving helpers (mirror test_reroute_stash_session_thread.py)
# ══════════════════════════════════════════════════════════════════════════
def _snapshot(directives):
    return {
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "t", "project_id": "p", "ttl_seconds": 600,
        "loop_blocks": [], "directives": directives,
    }


def _reroute_all(rule_id="rr"):
    return {
        "id": rule_id, "kind": "REROUTE", "mode": "enforce", "priority": 10,
        "selector": {"match": None, "group_by": []},
        "reroute": {"from": {}, "to": {"provider": "openai",
                                       "model": "gpt-4o-mini"}},
    }


def _reroute_for_model(rule_id, from_model, to_model):
    return {
        "id": rule_id, "kind": "REROUTE", "mode": "enforce", "priority": 10,
        "selector": {"match": {"field": "model", "operator": "EQ",
                               "value": from_model},
                     "group_by": []},
        "reroute": {"from": {}, "to": {"provider": "openai",
                                       "model": to_model}},
    }


def _init_client():
    """Init an enforce-mode client with check/check_sync/log_sync mocked —
    mirrors test_reroute_stash_session_thread.py's ``_init``."""
    client = tp.init(api_key="tp_sk_test_keyed_store", firewall="enforce")
    client.check_sync = MagicMock(return_value={"status": "allowed"})
    client.check = AsyncMock(return_value={"status": "allowed"})
    client.log_sync = MagicMock()
    return client


async def _checked_call(session, model):
    """Run one call's /check on the SHARED session (stashes THIS call's
    decision), then claim + return whatever its own key found — exactly the
    pattern every real /log site follows (obs key captured right after the
    check, drained once the provider response is back).

    The ``await asyncio.sleep(0)`` between the check and the claim stands in
    for the real gap in production — the provider's actual LLM HTTP call —
    a genuine suspension point where OTHER concurrent calls' checks+stashes
    land on this SAME shared session before this one claims. Without an
    explicit yield here, asyncio's cooperative scheduling would let each
    mocked check+stash+claim run atomically end-to-end (no real overlap),
    which would pass even against the OLD single-slot bug and prove
    nothing — this reproduces the actual race window."""
    kwargs = {"model": model}
    await _enforcer._run_async_check(
        kwargs=kwargs, provider="openai", session=session)
    obs_key = _state.get_current_obs_key()
    await asyncio.sleep(0)
    return _enforcer._claim_local_decision(session, obs_key)


class _AsyncioTestCase(unittest.TestCase):
    def setUp(self):
        _state.reset_pack()

    def tearDown(self):
        _state.reset_pack()
        tp.uninstrument()

    @staticmethod
    def _run(coro):
        return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════════
# (3) End-to-end asyncio: N concurrent reroute calls on ONE session
# ══════════════════════════════════════════════════════════════════════════
class TestConcurrentSameRuleBurst(_AsyncioTestCase):
    def test_burst_of_same_rule_reroutes_each_gets_its_own_row(self):
        """Today's repro (pre-fix): N concurrent same-rule reroutes on one
        session collapsed to exactly ONE claimed decision total (first
        finisher wins, N-1 lost). Fixed: every call claims its OWN."""
        _init_client()
        _state.apply_snapshot(_snapshot([_reroute_all("rr")]))
        session = TPSession(user_id="u1")
        n = 6

        async def _run():
            return await asyncio.gather(
                *[_checked_call(session, "gpt-4o") for _ in range(n)])

        results = self._run(_run())

        self.assertEqual(len(results), n)
        for ld in results:
            self.assertIsNotNone(ld)
            self.assertEqual(ld["outcome"], "rerouted")
            self.assertEqual(ld["rule_id"], "rr")
        # 0 residual: nothing left in the store for a later call to inherit.
        self.assertFalse(getattr(session, "_local_decisions", None))


# ══════════════════════════════════════════════════════════════════════════
# (4) Heterogeneous burst: distinct rules + one clean call
# ══════════════════════════════════════════════════════════════════════════
class TestHeterogeneousBurst(_AsyncioTestCase):
    def test_three_rerouted_one_clean_no_cross_contamination(self):
        """Second defect the old slot exhibited: a heterogeneous burst could
        stamp an UN-rerouted call's row with a sibling's decision (a false-
        positive REQUEST_REROUTED). Each rerouted call must get its OWN
        rule's decision; the clean call must get none."""
        _init_client()
        _state.apply_snapshot(_snapshot([
            _reroute_for_model("rr-0", "m0", "t0"),
            _reroute_for_model("rr-1", "m1", "t1"),
            _reroute_for_model("rr-2", "m2", "t2"),
        ]))
        session = TPSession(user_id="u1")

        async def _run():
            return await asyncio.gather(
                _checked_call(session, "m0"),
                _checked_call(session, "m1"),
                _checked_call(session, "m2"),
                _checked_call(session, "clean-model"),  # matches no rule
            )

        ld0, ld1, ld2, ld_clean = self._run(_run())

        self.assertEqual(ld0["rule_id"], "rr-0")
        self.assertEqual(ld1["rule_id"], "rr-1")
        self.assertEqual(ld2["rule_id"], "rr-2")
        for ld in (ld0, ld1, ld2):
            self.assertEqual(ld["outcome"], "rerouted")
        # No false positive on the clean row.
        self.assertIsNone(ld_clean)
        self.assertFalse(getattr(session, "_local_decisions", None))


# ══════════════════════════════════════════════════════════════════════════
# (5) Deferred-flush steal window: a stash landing mid-flight survives
# ══════════════════════════════════════════════════════════════════════════
class TestDeferredFlushStealWindow(unittest.TestCase):
    def test_concurrent_stash_during_flush_http_survives(self):
        """`_flush_deferred_spans` used to unconditionally null the flat slot
        AFTER its (blocking) log calls — a concurrent call's fresh stash
        landing during that window was destroyed. Now the clear is
        claim-scoped (only what THIS call's key claimed up front), so a
        sibling's stash under a DIFFERENT key survives untouched."""
        tp_client = MagicMock()
        session = TPSession()
        session._deferred_spans = [{
            "user_id": session.user_id,
            "span": {"trace_id": session.trace_id, "span_order": 0,
                     "span_name": "chat"},
        }]
        session._defer_telemetry = False

        own_key = "flush-key"
        _enforcer._stash_local_decision_entry(
            session, {"rule_id": "own", "outcome": "rerouted"}, own_key)

        stolen = {"rule_id": "concurrent", "outcome": "rerouted"}

        def _mid_flight_log_sync(**_kwargs):
            # A concurrent call's OWN stash landing while this flush's
            # blocking HTTP log call is in flight.
            _enforcer._stash_local_decision_entry(session, stolen, "other-key")

        tp_client.log_sync = MagicMock(side_effect=_mid_flight_log_sync)

        with mock.patch.object(
                _enforcer, "get_client", return_value=tp_client):
            _enforcer._flush_deferred_spans(session, obs_key=own_key)

        payload = tp_client.log_sync.call_args.kwargs
        self.assertEqual(payload.get("local_decision"),
                         {"rule_id": "own", "outcome": "rerouted"})

        remaining = [e["ld"] for e in session._local_decisions]
        self.assertEqual(remaining, [stolen])


if __name__ == "__main__":
    unittest.main()
