"""Python `generate_group_by_tag` must produce the SAME tag the server
(and the Node SDK) produce, so local-fast-path `tag in entities` lookups against a
server-armed blocked/rerouted set hit.

The server's canonical coercion (the server-side rule evaluator) is:

    groupByArr.map(field => resolveField(payload, field) || 'unknown').join('_')

i.e. JS-falsy -> 'unknown' sentinel; ELSE the JS String() of the kept value.

These tests pin the IDENTICAL literal strings the Node oracle yields
(tests/groupByTag.test.ts), so a Python regression that re-encodes the old
keep-falsy / `str()` behavior fails. Every value is delivered THROUGH a real
payload (top-level field OR metadata.* field) so the real resolver
(`_payload_field`) resolves it and feeds the coercion — we never call a coercion
helper directly with a bare value (assertion 5).
"""
import math
import unittest

from token_police.local_evaluator import generate_group_by_tag


# (name, payload, group_by, expected_literal) — IDENTICAL literals to the Node
# oracle. Mix of top-level and metadata.* routing exercises both resolver paths.
SCALAR_FIXTURES = [
    # ── Half 1: JS-falsy -> "unknown" ─────────────────────────────────
    ("int 0 (top-level)", {"score": 0}, ["score"], "unknown"),
    ("float 0.0 (metadata)", {"metadata": {"ratio": 0.0}}, ["ratio"], "unknown"),
    ("empty string (top-level)", {"name": ""}, ["name"], "unknown"),
    ("False (top-level)", {"flag": False}, ["flag"], "unknown"),
    ("False (metadata)", {"metadata": {"is_active": False}}, ["is_active"], "unknown"),
    ("NaN (top-level)", {"val": float("nan")}, ["val"], "unknown"),
    ("negative zero (top-level)", {"z": -0.0}, ["z"], "unknown"),
    ("None (top-level)", {"x": None}, ["x"], "unknown"),
    ("absent key", {}, ["missing"], "unknown"),
    # ── Half 1: TRUTHY string edge values KEPT verbatim ───────────────
    ('string "0" (top-level)', {"code": "0"}, ["code"], "0"),
    ('string "false" (metadata)', {"metadata": {"label": "false"}}, ["label"], "false"),
    ('space " " (top-level)', {"s": " "}, ["s"], " "),
    # ── Half 2: JS-faithful stringification of the survivor ───────────
    ("bool True (top-level)", {"is_premium": True}, ["is_premium"], "true"),
    ("bool True (metadata)", {"metadata": {"is_premium": True}}, ["is_premium"], "true"),
    ("float 1.0 (top-level)", {"m": 1.0}, ["m"], "1"),
    ("float 2.0 (metadata)", {"metadata": {"n": 2.0}}, ["n"], "2"),
    ("float 10.0 (top-level)", {"m": 10.0}, ["m"], "10"),
    ("float 2.5 (top-level)", {"m": 2.5}, ["m"], "2.5"),
    ("plain string alice (top-level)", {"user": "alice"}, ["user"], "alice"),
    ("plain int 5 (top-level)", {"n": 5}, ["n"], "5"),
    # ── Multi-field joins (pinned literals) ───────────────────────────
    ("join (0, '') -> all unknown", {"a": 0, "b": ""}, ["a", "b"], "unknown_unknown"),
    ("join ('alice', False) -> mixed", {"user": "alice", "flag": False}, ["user", "flag"], "alice_unknown"),
]


class GenerateGroupByTagParityTest(unittest.TestCase):
    def test_scalar_fixtures_match_pinned_literals(self):
        for name, payload, group_by, expected in SCALAR_FIXTURES:
            with self.subTest(name=name):
                self.assertEqual(generate_group_by_tag(payload, group_by), expected)

    def test_empty_group_by_is_global(self):
        self.assertEqual(generate_group_by_tag({"a": 1}, []), "global")

    # Assertions 8 & 9: JS []/{} are TRUTHY so the server KEEPS them (the
    # Python predicate must NOT use bare `if not v`). The EXACT byte
    # stringification of a kept collection is a NAMED residual (scope ii) —
    # Python `str([])`="[]" diverges from JS ""/"[object Object]" — so here we
    # only assert the parity-critical guard: NOT coerced to the "unknown"
    # sentinel.
    def test_empty_list_is_kept_not_unknown(self):
        tag = generate_group_by_tag({"metadata": {"tags": []}}, ["tags"])
        self.assertNotEqual(tag, "unknown")

    def test_empty_dict_is_kept_not_unknown(self):
        tag = generate_group_by_tag({"metadata": {"obj": {}}}, ["obj"])
        self.assertNotEqual(tag, "unknown")

    # Explicit NaN branch coverage (v != v predicate) — assertion 5.
    def test_nan_drops_to_unknown(self):
        self.assertEqual(generate_group_by_tag({"v": float("nan")}, ["v"]), "unknown")
        self.assertTrue(math.isnan(float("nan")))

    # Non-regression (rubric assertion f): after the flat-key resolver
    # returns the PRESENT `None` (no longer skips to the metadata shadow), so a
    # present-None groupBy field now collapses to the server-aligned "unknown"
    # sentinel (resolver->None-> the `None`->"unknown" branch), NOT the old "pro".
    # This is a CONVERGENCE with the server (resolveField->null, then
    # `val || 'unknown'`), not a regression. See the flat-key resolver regression notes.
    def test_f27_present_none_shadow_collapses_to_unknown(self):
        payload = {"paid_plan": None, "metadata": {"paid_plan": "pro"}}
        tag = generate_group_by_tag(payload, ["paid_plan"])
        self.assertEqual(tag, "unknown")
        self.assertNotEqual(tag, "pro")


if __name__ == "__main__":
    unittest.main()
