"""Fix 1 — precomputed hot-path structures for the local evaluator.

The evaluator now reuses per-pack structures (loop-block set, pre-sorted
directives, per-directive entity frozensets) built ONCE at pack-apply time
instead of rebuilding them on every LLM call. These tests pin two invariants:

  1. Decision-equivalence: a pack evaluated via the PRECOMPUTED path and via the
     on-the-fly FALLBACK path yields byte-identical decisions + observations
     across loop blocks, priority tie-breaks, entity-scoped directives, reroute,
     and empty packs.
  2. Freshness: a pack swap (full snapshot) and a delta update both REBUILD the
     precompute atomically — a stale set can never survive a swap.
"""
import copy
import types
import unittest

from token_police import local_evaluator as le
from token_police import state


def _session(user_id=None, session_id=None, paid_plan=None, metadata=None):
    return types.SimpleNamespace(
        user_id=user_id,
        session_id=session_id,
        paid_plan=paid_plan,
        metadata=metadata or {},
        trace_id="trace-x",
    )


def _eval_both(pack, session, observed):
    """Evaluate the same logical pack via the fallback (no derived key) and the
    precomputed (derived attached) paths. Returns (fallback_result, pre_result).
    """
    raw = copy.deepcopy(pack)
    fallback = le.evaluate(raw, session, observed)

    withd = copy.deepcopy(pack)
    derived = le.build_derived(withd)
    assert derived is not None, "build_derived returned None for a well-formed pack"
    withd[le._DERIVED_KEY] = derived
    pre = le.evaluate(withd, session, observed)
    return fallback, pre


class TestDecisionEquivalence(unittest.TestCase):
    def _assert_equiv(self, pack, session, observed):
        fallback, pre = _eval_both(pack, session, observed)
        self.assertEqual(fallback, pre)

    def test_empty_pack(self):
        self._assert_equiv({"directives": []}, _session(), {"provider": "openai", "model": "gpt-4o"})

    def test_no_directives_key(self):
        self._assert_equiv({}, _session(), {"provider": "openai", "model": "gpt-4o"})

    def test_loop_block_present(self):
        pack = {"directives": [], "loop_blocks": ["trace-x", "trace-y"]}
        self._assert_equiv(pack, _session(), {"provider": "openai", "model": "m", "trace_id": "trace-x"})

    def test_loop_block_absent(self):
        pack = {"directives": [], "loop_blocks": ["trace-y"]}
        self._assert_equiv(pack, _session(), {"provider": "openai", "model": "m", "trace_id": "trace-x"})

    def test_unconditional_block_enforce(self):
        pack = {"directives": [
            {"id": "r1", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
             "selector": {"match": None, "group_by": []}},
        ]}
        self._assert_equiv(pack, _session(), {"provider": "openai", "model": "m"})

    def test_priority_tie_break_by_id(self):
        # Same priority — the id code-point order must break the tie identically
        # on both paths. Multiple matching dry_run blocks reveal the eval order.
        pack = {"directives": [
            {"id": "zzz", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 5,
             "selector": {"match": None, "group_by": []}},
            {"id": "aaa", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 5,
             "selector": {"match": None, "group_by": []}},
            {"id": "mmm", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 5,
             "selector": {"match": None, "group_by": []}},
        ]}
        fallback, pre = _eval_both(pack, _session(), {"provider": "openai", "model": "m"})
        self.assertEqual(fallback, pre)
        self.assertEqual([o["rule_id"] for o in pre["observations"]], ["aaa", "mmm", "zzz"])

    def test_priority_dominates_id(self):
        pack = {"directives": [
            {"id": "aaa", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 50,
             "selector": {"match": None, "group_by": []}},
            {"id": "zzz", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 10,
             "selector": {"match": None, "group_by": []}},
        ]}
        fallback, pre = _eval_both(pack, _session(), {"provider": "openai", "model": "m"})
        self.assertEqual(fallback, pre)
        self.assertEqual([o["rule_id"] for o in pre["observations"]], ["zzz", "aaa"])

    def test_missing_and_nonnumeric_priority_treated_as_100(self):
        pack = {"directives": [
            {"id": "no_prio", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run",
             "selector": {"match": None, "group_by": []}},
            {"id": "bad_prio", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": "high",
             "selector": {"match": None, "group_by": []}},
            {"id": "low", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 5,
             "selector": {"match": None, "group_by": []}},
        ]}
        fallback, pre = _eval_both(pack, _session(), {"provider": "openai", "model": "m"})
        self.assertEqual(fallback, pre)
        self.assertEqual([o["rule_id"] for o in pre["observations"]], ["low", "bad_prio", "no_prio"])

    def test_entity_block_armed(self):
        pack = {"directives": [
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
             "entities": ["alice", "bob"],
             "selector": {"match": None, "group_by": ["user_id"]}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "m"})

    def test_entity_block_not_armed(self):
        pack = {"directives": [
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
             "entities": ["bob"],
             "selector": {"match": None, "group_by": ["user_id"]}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "m"})

    def test_entity_block_empty_entities(self):
        pack = {"directives": [
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
             "entities": [], "selector": {"match": None, "group_by": ["user_id"]}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "m"})

    def test_entity_block_missing_entities_key(self):
        pack = {"directives": [
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
             "selector": {"match": None, "group_by": ["user_id"]}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "m"})

    def test_reroute_ungated_same_provider(self):
        pack = {"directives": [
            {"id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
             "selector": {"match": None, "group_by": []},
             "reroute": {"to": {"provider": "openai", "model": "gpt-4o-mini"}}},
        ]}
        self._assert_equiv(pack, _session(), {"provider": "openai", "model": "gpt-4o"})

    def test_reroute_entity_gated_armed(self):
        pack = {"directives": [
            {"id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
             "entities": ["alice"],
             "selector": {"match": None, "group_by": ["user_id"]},
             "reroute": {"to": {"provider": "openai", "model": "gpt-4o-mini"}}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "gpt-4o"})

    def test_reroute_entity_gated_not_armed(self):
        pack = {"directives": [
            {"id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
             "entities": ["bob"],
             "selector": {"match": None, "group_by": ["user_id"]},
             "reroute": {"to": {"provider": "openai", "model": "gpt-4o-mini"}}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "gpt-4o"})

    def test_reroute_cross_provider_rejected(self):
        pack = {"directives": [
            {"id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
             "selector": {"match": None, "group_by": []},
             "reroute": {"to": {"provider": "anthropic", "model": "claude-3-5-sonnet"}}},
        ]}
        self._assert_equiv(pack, _session(), {"provider": "openai", "model": "gpt-4o"})

    def test_multi_kind_mixed_priorities(self):
        pack = {"directives": [
            {"id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 30,
             "selector": {"match": None, "group_by": []},
             "reroute": {"to": {"provider": "openai", "model": "mini"}}},
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 20,
             "entities": ["alice"], "selector": {"match": None, "group_by": ["user_id"]}},
            {"id": "ub", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 10,
             "selector": {"match": {"field": "model", "operator": "EQ", "value": "gpt-4o"}, "group_by": []}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "gpt-4o"})

    def test_malformed_non_iterable_entities_equiv(self):
        # A non-iterable `entities` cannot be pre-frozen; both paths must behave
        # identically (the _ENTITY_INLINE sentinel reproduces the inline set()).
        pack = {"directives": [
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
             "entities": 5, "selector": {"match": None, "group_by": ["user_id"]}},
        ]}
        self._assert_equiv(pack, _session(user_id="alice"), {"provider": "openai", "model": "m"})


class TestPrecomputeFreshness(unittest.TestCase):
    """Precompute must be rebuilt on every pack swap — the wired-in path via
    state.apply_snapshot / state.apply_deltas, exactly as the SSE reader uses."""

    def setUp(self):
        state.reset_pack()

    def tearDown(self):
        state.reset_pack()

    def _snapshot(self, directives, version=1, loop_blocks=None):
        return {
            "version": version,
            "tenant_id": "t1",
            "project_id": "p1",
            "directives": directives,
            "loop_blocks": loop_blocks or [],
        }

    def test_snapshot_attaches_fresh_derived(self):
        snap = self._snapshot([
            {"id": "ub", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
             "selector": {"match": None, "group_by": []}},
        ])
        self.assertTrue(state.apply_snapshot(snap))
        pack = state.get_pack()
        self.assertIn(le._DERIVED_KEY, pack)
        derived = pack[le._DERIVED_KEY]
        self.assertEqual([d["id"] for d in derived["directives"]], ["ub"])
        res = le.evaluate(pack, _session(), {"provider": "openai", "model": "m"})
        self.assertEqual(res["decision"]["status"], "blocked")

    def test_delta_entity_arm_rebuilds_precompute(self):
        # v1: an ENTITY_BLOCK with NO armed entities -> call is allowed.
        snap = self._snapshot([
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
             "selector": {"match": None, "group_by": ["user_id"]}},
        ])
        self.assertTrue(state.apply_snapshot(snap))
        sess = _session(user_id="alice")
        observed = {"provider": "openai", "model": "m"}
        res1 = le.evaluate(state.get_pack(), sess, observed)
        self.assertEqual(res1["decision"]["status"], "allowed")

        # v2 delta arms "alice". If the precompute were stale, the pre-armed
        # (empty) entity set would still say allowed — proving the rebuild.
        self.assertTrue(state.apply_deltas(
            [{"op": "entity_blocked", "rule_id": "eb", "entity": "alice"}], 2))
        pack2 = state.get_pack()
        self.assertIn("alice", pack2[le._DERIVED_KEY]["entity_sets"][0])
        res2 = le.evaluate(pack2, sess, observed)
        self.assertEqual(res2["decision"]["status"], "blocked")

    def test_delta_entity_unarm_rebuilds_precompute(self):
        snap = self._snapshot([
            {"id": "eb", "kind": "ENTITY_BLOCK", "mode": "enforce", "priority": 10,
             "entities": ["alice"], "selector": {"match": None, "group_by": ["user_id"]}},
        ])
        self.assertTrue(state.apply_snapshot(snap))
        sess = _session(user_id="alice")
        observed = {"provider": "openai", "model": "m"}
        self.assertEqual(le.evaluate(state.get_pack(), sess, observed)["decision"]["status"], "blocked")

        self.assertTrue(state.apply_deltas(
            [{"op": "entity_unblocked", "rule_id": "eb", "entity": "alice"}], 2))
        pack2 = state.get_pack()
        self.assertNotIn("alice", pack2[le._DERIVED_KEY]["entity_sets"][0])
        self.assertEqual(le.evaluate(pack2, sess, observed)["decision"]["status"], "allowed")

    def test_full_snapshot_swap_replaces_directives(self):
        self.assertTrue(state.apply_snapshot(self._snapshot([
            {"id": "ub", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
             "selector": {"match": None, "group_by": []}},
        ], version=1)))
        self.assertEqual(
            le.evaluate(state.get_pack(), _session(), {"provider": "openai", "model": "m"})["decision"]["status"],
            "blocked")

        # Swap to a pack with no blocking directive; the derived directive list
        # must reflect the NEW pack (a stale derived would still block).
        self.assertTrue(state.apply_snapshot(self._snapshot([], version=5)))
        pack2 = state.get_pack()
        self.assertEqual(pack2[le._DERIVED_KEY]["directives"], [])
        self.assertEqual(
            le.evaluate(pack2, _session(), {"provider": "openai", "model": "m"})["decision"]["status"],
            "allowed")

    def test_delta_loop_block_rebuilds_loop_set(self):
        self.assertTrue(state.apply_snapshot(self._snapshot([], version=1)))
        observed = {"provider": "openai", "model": "m", "trace_id": "loopy"}
        self.assertEqual(
            le.evaluate(state.get_pack(), _session(), observed)["decision"]["status"], "allowed")

        self.assertTrue(state.apply_deltas(
            [{"op": "loop_blocked", "trace_id": "loopy"}], 2))
        pack2 = state.get_pack()
        self.assertIn("loopy", pack2[le._DERIVED_KEY]["loop_set"])
        self.assertEqual(le.evaluate(pack2, _session(), observed)["decision"]["status"], "blocked")


if __name__ == "__main__":
    unittest.main()
