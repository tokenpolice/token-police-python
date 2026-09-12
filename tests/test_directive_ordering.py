import unittest

from token_police.local_evaluator import evaluate


# Canonical total order for directives — priority ascending
# (missing/non-numeric -> 100), then id ascending by code-POINT order. Python str
# `<` is the reference implementation; the Node sibling directiveOrdering.test.ts
# pins the SAME literal fixture so Node == Python over the ASCII/BMP CUID id space.

def _block_pack_from_ids(ids, priority=10):
    # Every directive is a matching dry_run UNCONDITIONAL_BLOCK, so each emits one
    # would_block observation in evaluation order — the observation rule_id
    # sequence reveals the canonical sort.
    return {
        "directives": [
            {
                "id": _id,
                "kind": "UNCONDITIONAL_BLOCK",
                "mode": "dry_run",
                "priority": priority,
                "selector": {"match": None, "group_by": []},
            }
            for _id in ids
        ]
    }


def _ordered_ids(ids, priority=10):
    res = evaluate(_block_pack_from_ids(ids, priority), None, {})
    return [o["rule_id"] for o in res["observations"]]


class TestDirectiveOrdering(unittest.TestCase):
    def test_literal_parity_fixture_code_point_not_locale(self):
        # locale collation would put 'a' first and be case-insensitive; code-point
        # puts digits < uppercase < lowercase: '1' '9' 'B' 'Z' 'a'.
        self.assertEqual(_ordered_ids(["Z", "a", "10", "9", "Bb"]), ["10", "9", "Bb", "Z", "a"])

    def test_input_order_does_not_affect_result(self):
        self.assertEqual(_ordered_ids(["a", "9", "Z", "Bb", "10"]), ["10", "9", "Bb", "Z", "a"])

    def test_priority_dominates_id_tiebreak(self):
        pack = {
            "directives": [
                {"id": "aaa", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 50,
                 "selector": {"match": None, "group_by": []}},
                {"id": "zzz", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 10,
                 "selector": {"match": None, "group_by": []}},
            ]
        }
        res = evaluate(pack, None, {})
        self.assertEqual([o["rule_id"] for o in res["observations"]], ["zzz", "aaa"])

    def test_missing_priority_treated_as_100(self):
        pack = {
            "directives": [
                {"id": "aaa", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run",
                 "selector": {"match": None, "group_by": []}},  # -> 100
                {"id": "bbb", "kind": "UNCONDITIONAL_BLOCK", "mode": "dry_run", "priority": 5,
                 "selector": {"match": None, "group_by": []}},
            ]
        }
        res = evaluate(pack, None, {})
        self.assertEqual([o["rule_id"] for o in res["observations"]], ["bbb", "aaa"])

    def test_equal_priority_block_tie_enforce_picks_aaa(self):
        def mk(ids):
            return {
                "directives": [
                    {"id": _id, "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
                     "selector": {"match": None, "group_by": []}}
                    for _id in ids
                ]
            }
        for order in (["aaa", "bbb"], ["bbb", "aaa"]):
            res = evaluate(mk(order), None, {})
            self.assertEqual(res["decision"]["status"], "blocked")
            self.assertEqual(res["decision"]["rule_id"], "aaa")

    def test_equal_priority_reroute_tie_enforce_picks_aaa_model(self):
        def mk(ids):
            return {
                "directives": [
                    {"id": _id, "kind": "REROUTE", "mode": "enforce", "priority": 10,
                     "selector": {"match": None, "group_by": []},
                     "reroute": {"from": None,
                                 "to": {"provider": "openai", "model": "gpt-aaa" if _id == "aaa" else "gpt-bbb"}}}
                    for _id in ids
                ]
            }
        for order in (["aaa", "bbb"], ["bbb", "aaa"]):
            res = evaluate(mk(order), None, {"provider": "openai", "model": "gpt-4"})
            self.assertEqual(res["decision"]["status"], "rerouted")
            self.assertEqual(res["decision"]["rule_id"], "aaa")
            self.assertEqual(res["decision"]["reroute"]["to"]["model"], "gpt-aaa")


if __name__ == "__main__":
    unittest.main()
