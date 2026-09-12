"""B4 — per-call keyed store for the applied-reroute provenance marker
(`_tp_routing`). Python twin of tests/routingMarkerStore.test.ts (Node).

Bug: `_apply_reroute` writes `_tp_routing` onto `session.metadata`, and every
subsequent row-emission site copied session metadata WHOLESALE into its
`/log` payload — so one rerouted call's marker rode EVERY later row of that
session (tool rows, agent/chain structural rows, unrelated sibling calls'
rows, even other modalities).

Fix: `session._routing_markers` — a bounded, session-scoped, per-call keyed
list — via `_stash_routing_marker` / `_peek_routing_marker` /
`_drop_routing_marker` in `token_police/enforcer.py`, PLUS
`_copy_session_metadata` (strip on every copy) / `_stamp_routing_marker`
(re-add ONLY on the owning model-call row) / `_row_metadata_from_session`
(the failure/local-block log sites that used to hand `session.metadata`
itself to `log_sync`).

Deliberately PEEK-MANY (never destructive on peek) — unlike B3's
`local_decision` claim-once store — because one rerouted call can emit
several of its OWN rows and every one of them must carry the marker.
`_drop_routing_marker` exists ONLY for the withdrawal path
(`_AnthropicAsyncStreamMgrWrapper._rebuild_after_reroute`, when a rebuild
fails and the wire request reverts to the original model) — a deliberately
DESTRUCTIVE, narrower operation than peek.

This file covers the MEDIUM "store unit" tier:
  (1) own-key hit / other-key miss / keyed invisible to keyless / untagged
      only reachable by a keyless peek;
  (2) peek is non-destructive;
  (3) same-key stash REPLACES;
  (4) FIFO cap 64 evicts oldest;
  (5) expiry past ROUTING_MARKER_STALE_SECONDS (monotonic clock, as the
      store reads it), swept on both stash and peek;
  (6) fail-open on hostile/garbage inputs;
  (7) session.metadata is never touched by any helper here;
  (8) thread safety — N threads stash+peek their OWN keys on ONE session
      (mirrors test_local_decision_keyed_store.py's thread-safety tier);
  (9) `_drop_routing_marker` — the withdrawal primitive
      `_rebuild_after_reroute` uses (LOW item 15's mechanism, tested at the
      store level; the full stream-manager integration harness was not
      built for this pass — see the B4 test report).
Plus direct coverage of `_copy_session_metadata` / `_stamp_routing_marker` /
`_row_metadata_from_session`, the three row-side helpers
`test_reroute_marker_row_scope.py` exercises through the real SDK paths.
"""
import threading
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer as _enforcer


def entries(session):
    return getattr(session, "_routing_markers", None) or []


class TestAttribution(unittest.TestCase):
    def test_own_key_hits_other_key_misses_no_untagged_fallback(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "mine"}, "key-a")

        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "mine"})
        # Deliberately stricter than _claim_local_decision: a near-miss key
        # gets NOTHING, never a stranger's record.
        self.assertIsNone(_enforcer._peek_routing_marker(s, "key-b"))

    def test_keyed_record_invisible_to_keyless_peek(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "mine"}, "key-a")
        self.assertIsNone(_enforcer._peek_routing_marker(s, None))

    def test_untagged_record_reachable_only_by_keyless_peek(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "untagged"}, None)

        self.assertIsNone(_enforcer._peek_routing_marker(s, "some-key"))
        self.assertEqual(
            _enforcer._peek_routing_marker(s, None), {"id": "untagged"})

    def test_two_keyed_records_each_reachable_only_by_own_key(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "a"}, "key-a")
        _enforcer._stash_routing_marker(s, {"id": "b"}, "key-b")

        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "a"})
        self.assertEqual(_enforcer._peek_routing_marker(s, "key-b"), {"id": "b"})
        self.assertIsNone(_enforcer._peek_routing_marker(s, "key-c"))


class TestPeekMany(unittest.TestCase):
    def test_peeking_same_key_repeatedly_returns_every_time(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "mine"}, "key-a")

        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "mine"})
        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "mine"})
        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "mine"})
        self.assertEqual(len(entries(s)), 1)

    def test_sibling_key_peek_does_not_disturb_other_entry(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "mine"}, "key-a")
        _enforcer._stash_routing_marker(s, {"id": "sibling"}, "key-b")

        self.assertEqual(
            _enforcer._peek_routing_marker(s, "key-b"), {"id": "sibling"})
        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "mine"})
        self.assertEqual(len(entries(s)), 2)


class TestSameKeyReplace(unittest.TestCase):
    def test_restash_same_key_replaces(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "first"}, "key-a")
        _enforcer._stash_routing_marker(s, {"id": "second"}, "key-a")

        self.assertEqual(len(entries(s)), 1)
        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "second"})

    def test_none_key_always_appends(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "a"}, None)
        _enforcer._stash_routing_marker(s, {"id": "b"}, None)

        self.assertEqual(len(entries(s)), 2)
        self.assertEqual(_enforcer._peek_routing_marker(s, None), {"id": "b"})


class TestFifoCap(unittest.TestCase):
    def test_caps_at_64_drops_oldest(self):
        s = NS()
        cap = 64  # _ROUTING_MARKER_CAP in token_police/enforcer.py (not exported)
        total = cap + 6
        for i in range(total):
            _enforcer._stash_routing_marker(s, {"id": i}, f"key-{i}")

        es = entries(s)
        self.assertEqual(len(es), cap)
        ids = [e["routing"]["id"] for e in es]
        self.assertEqual(ids, list(range(total - cap, total)))
        self.assertIsNone(_enforcer._peek_routing_marker(s, "key-0"))
        self.assertEqual(
            _enforcer._peek_routing_marker(s, f"key-{total - 1}"),
            {"id": total - 1})


class TestExpiry(unittest.TestCase):
    def test_entry_older_than_window_unreachable_even_by_own_key(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "stale"}, "key-a")
        entries(s)[0]["ts"] -= _enforcer.ROUTING_MARKER_STALE_SECONDS + 1

        self.assertIsNone(_enforcer._peek_routing_marker(s, "key-a"))
        self.assertEqual(entries(s), [])

    def test_entry_just_under_window_still_reachable(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "fresh"}, "key-a")
        entries(s)[0]["ts"] -= _enforcer.ROUTING_MARKER_STALE_SECONDS - 1.0

        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "fresh"})

    def test_stash_sweeps_expired_sibling(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "stale"}, "key-a")
        entries(s)[0]["ts"] -= _enforcer.ROUTING_MARKER_STALE_SECONDS + 1

        _enforcer._stash_routing_marker(s, {"id": "fresh"}, "key-b")

        ids = [e["routing"]["id"] for e in entries(s)]
        self.assertEqual(ids, ["fresh"])

    def test_expired_untagged_never_falls_back(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "stale"}, None)
        entries(s)[0]["ts"] -= _enforcer.ROUTING_MARKER_STALE_SECONDS + 1

        self.assertIsNone(_enforcer._peek_routing_marker(s, None))
        self.assertEqual(entries(s), [])

    def test_mocked_monotonic_clock_expires_exactly_at_window(self):
        s = NS()
        state = {"now": 1000.0}

        def fake_monotonic():
            return state["now"]

        with mock.patch.object(_enforcer._time, "monotonic", fake_monotonic):
            _enforcer._stash_routing_marker(s, {"id": "a"}, "key-a")
            state["now"] += _enforcer.ROUTING_MARKER_STALE_SECONDS - 1
            self.assertEqual(
                _enforcer._peek_routing_marker(s, "key-a"), {"id": "a"})
            state["now"] += 2
            self.assertIsNone(_enforcer._peek_routing_marker(s, "key-a"))


class TestFailOpen(unittest.TestCase):
    def test_stash_never_raises_on_none_session(self):
        _enforcer._stash_routing_marker(None, {"id": 1}, "k")

    def test_peek_never_raises_returns_none_on_none_session(self):
        self.assertIsNone(_enforcer._peek_routing_marker(None, "k"))
        s = NS()
        self.assertIsNone(_enforcer._peek_routing_marker(s, "k"))

    def test_corrupt_non_list_marker_attr_degrades_and_self_heals(self):
        s = NS(_routing_markers="not-a-list")
        self.assertIsNone(_enforcer._peek_routing_marker(s, "k"))
        _enforcer._stash_routing_marker(s, {"id": 1}, "k")
        self.assertIsInstance(s._routing_markers, list)
        self.assertEqual(_enforcer._peek_routing_marker(s, "k"), {"id": 1})

    def test_falsy_routing_never_stashed(self):
        s = NS()
        _enforcer._stash_routing_marker(s, None, "k")
        _enforcer._stash_routing_marker(s, {}, "k")
        self.assertIsNone(getattr(s, "_routing_markers", None))

    def test_drop_never_raises_on_hostile_inputs(self):
        _enforcer._drop_routing_marker(None, "k")
        s = NS()
        _enforcer._drop_routing_marker(s, "k")  # nothing stashed — no-op


class TestSessionMetadataUntouched(unittest.TestCase):
    def test_stash_and_peek_only_write_routing_markers_attr(self):
        s = NS(metadata={"_tp_routing": {"rule_id": "r1"}, "customer_key": "keep"})
        before = s.metadata
        _enforcer._stash_routing_marker(s, {"rule_id": "r1", "actual_model": "m2"}, "key-a")
        _enforcer._peek_routing_marker(s, "key-a")
        _enforcer._peek_routing_marker(s, "nope")

        self.assertIs(s.metadata, before)
        self.assertEqual(
            s.metadata, {"_tp_routing": {"rule_id": "r1"}, "customer_key": "keep"})

    def test_copy_session_metadata_never_mutates_source(self):
        s = NS(metadata={"_tp_routing": {"rule_id": "r1"}, "customer_key": "keep"})
        dest = {}
        _enforcer._copy_session_metadata(dest, s)

        self.assertEqual(
            s.metadata, {"_tp_routing": {"rule_id": "r1"}, "customer_key": "keep"})
        self.assertEqual(dest, {"customer_key": "keep"})
        self.assertNotIn("_tp_routing", dest)

    def test_row_metadata_from_session_builds_fresh_dict(self):
        s = NS(metadata={"_tp_routing": {"rule_id": "r1"}, "customer_key": "keep"})
        before = s.metadata
        out = _enforcer._row_metadata_from_session(s, None)

        self.assertIs(s.metadata, before)
        self.assertIsNot(out, s.metadata)
        self.assertEqual(out, {"customer_key": "keep"})


class TestCopySessionMetadata(unittest.TestCase):
    def test_copies_every_key_except_tp_routing(self):
        s = NS(metadata={"a": 1, "b": "x", "_tp_routing": {"rule_id": "r"}})
        dest = {}
        _enforcer._copy_session_metadata(dest, s)
        self.assertEqual(dest, {"a": 1, "b": "x"})

    def test_none_dest_never_raises(self):
        s = NS(metadata={"a": 1})
        _enforcer._copy_session_metadata(None, s)

    def test_missing_metadata_leaves_dest_untouched(self):
        dest = {"existing": True}
        _enforcer._copy_session_metadata(dest, NS(metadata=None))
        _enforcer._copy_session_metadata(dest, NS())
        self.assertEqual(dest, {"existing": True})

    def test_empty_but_present_dest_is_written_to(self):
        # The store's own docstring: `dest is None`, never `not dest` — the
        # destination is normally an EMPTY dict (falsy) here, and a
        # truthiness guard would silently drop every customer metadata key.
        s = NS(metadata={"a": 1})
        dest = {}
        _enforcer._copy_session_metadata(dest, s)
        self.assertEqual(dest, {"a": 1})


class TestStampRoutingMarker(unittest.TestCase):
    def test_noop_when_nothing_for_this_key(self):
        s = NS()
        dest = {"keep": True}
        _enforcer._stamp_routing_marker(dest, s, "key-a")
        self.assertEqual(dest, {"keep": True})

    def test_serialize_false_manual_emitters_readd_raw_dict(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"rule_id": "r1", "actual_model": "m2"}, "key-a")
        dest = {}
        _enforcer._stamp_routing_marker(dest, s, "key-a", serialize=False)
        self.assertEqual(dest["_tp_routing"], {"rule_id": "r1", "actual_model": "m2"})
        self.assertIsInstance(dest["_tp_routing"], dict)

    def test_serialize_true_otel_paths_readd_json_string(self):
        import json
        s = NS()
        _enforcer._stash_routing_marker(s, {"rule_id": "r1", "actual_model": "m2"}, "key-a")
        dest = {}
        _enforcer._stamp_routing_marker(dest, s, "key-a", serialize=True)
        self.assertIsInstance(dest["_tp_routing"], str)
        self.assertEqual(
            json.loads(dest["_tp_routing"]), {"rule_id": "r1", "actual_model": "m2"})

    def test_wrong_key_never_stamps(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"rule_id": "r1"}, "key-a")
        dest = {}
        _enforcer._stamp_routing_marker(dest, s, "key-b")
        self.assertNotIn("_tp_routing", dest)

    def test_none_dest_never_raises(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"rule_id": "r1"}, "key-a")
        _enforcer._stamp_routing_marker(None, s, "key-a")


class TestRowMetadataFromSession(unittest.TestCase):
    def test_keyed_row_includes_marker_on_match(self):
        s = NS(metadata={"customer_key": "keep"})
        _enforcer._stash_routing_marker(s, {"rule_id": "r1", "actual_model": "m2"}, "key-a")
        out = _enforcer._row_metadata_from_session(s, "key-a")
        self.assertEqual(
            out, {"customer_key": "keep",
                  "_tp_routing": {"rule_id": "r1", "actual_model": "m2"}})

    def test_keyed_row_excludes_marker_on_miss(self):
        s = NS(metadata={"customer_key": "keep"})
        _enforcer._stash_routing_marker(s, {"rule_id": "r1"}, "key-a")
        out = _enforcer._row_metadata_from_session(s, "key-b")
        self.assertEqual(out, {"customer_key": "keep"})

    def test_stamp_false_never_readds_even_on_key_match(self):
        """`stamp=False` (used by the flush-time unconsumed-Bedrock-failure
        path, where no key can be trusted) suppresses the re-add outright."""
        s = NS(metadata={"customer_key": "keep"})
        _enforcer._stash_routing_marker(s, {"rule_id": "r1"}, "key-a")
        out = _enforcer._row_metadata_from_session(s, "key-a", stamp=False)
        self.assertEqual(out, {"customer_key": "keep"})


class TestDropRoutingMarker(unittest.TestCase):
    """`_drop_routing_marker` is the withdrawal primitive
    `_AnthropicAsyncStreamMgrWrapper._rebuild_after_reroute` calls when a
    reroute rebuild fails and the wire request reverts to the ORIGINAL model
    — no row of that call may claim a reroute afterward. Store-level
    coverage only (see the module docstring)."""

    def test_drop_keyed_removes_only_that_key(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "mine"}, "key-a")
        _enforcer._stash_routing_marker(s, {"id": "sibling"}, "key-b")

        _enforcer._drop_routing_marker(s, "key-a")

        self.assertIsNone(_enforcer._peek_routing_marker(s, "key-a"))
        self.assertEqual(_enforcer._peek_routing_marker(s, "key-b"), {"id": "sibling"})

    def test_drop_none_key_targets_newest_untagged_only(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "a"}, None)
        _enforcer._stash_routing_marker(s, {"id": "b"}, None)

        _enforcer._drop_routing_marker(s, None)

        self.assertEqual(_enforcer._peek_routing_marker(s, None), {"id": "a"})

    def test_drop_missing_key_is_a_noop(self):
        s = NS()
        _enforcer._stash_routing_marker(s, {"id": "mine"}, "key-a")
        _enforcer._drop_routing_marker(s, "key-zzz")
        self.assertEqual(_enforcer._peek_routing_marker(s, "key-a"), {"id": "mine"})


class TestThreadSafety(unittest.TestCase):
    def test_n_threads_own_keys_stash_and_peek_consistently(self):
        session = NS()
        n_threads = 8
        n_rounds = 200
        errors = []

        def worker(tid):
            try:
                for i in range(n_rounds):
                    key = f"t{tid}-{i}"
                    marker = {"tid": tid, "i": i}
                    _enforcer._stash_routing_marker(session, marker, key)
                    seen = _enforcer._peek_routing_marker(session, key)
                    if seen != marker:
                        errors.append((tid, i, seen))
            except Exception as exc:  # pragma: no cover - diagnostic only
                errors.append((tid, "exception", repr(exc)))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])

    def test_n_threads_racing_same_key_never_corrupts_list(self):
        session = NS()
        n_threads = 8
        n_rounds = 200
        errors = []

        def worker(tid):
            try:
                for i in range(n_rounds):
                    _enforcer._stash_routing_marker(
                        session, {"tid": tid, "i": i}, "shared-key")
            except Exception as exc:  # pragma: no cover
                errors.append((tid, repr(exc)))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(errors, [])
        matching = [e for e in entries(session) if e.get("key") == "shared-key"]
        self.assertLessEqual(len(matching), 1)


class TestConstants(unittest.TestCase):
    def test_routing_attr_mirrors_key_under_tp_meta_prefix(self):
        self.assertEqual(_enforcer._TP_ROUTING_ATTR, "tp.meta." + _enforcer._TP_ROUTING_KEY)
        self.assertEqual(_enforcer._TP_ROUTING_KEY, "_tp_routing")

    def test_stale_window_is_300s(self):
        self.assertEqual(_enforcer.ROUTING_MARKER_STALE_SECONDS, 300.0)


if __name__ == "__main__":
    unittest.main()
