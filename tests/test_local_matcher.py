"""matches_condition AND/OR + full-operator tests. Must stay byte-compatible
with the server-side rule evaluator matcher and the Node localEvaluator — see
match_condition_boolean_composition_design.md."""
import unittest
from types import SimpleNamespace

from token_police.local_evaluator import matches_condition as m
from token_police.local_evaluator import _payload_field
from token_police.local_evaluator import _build_payload
from token_police.local_evaluator import _js_string, _js_strict_eq

PAYLOAD = {"paid_plan": "free", "model": "gpt-4", "provider": "openai", "metadata": {"team": "x"}}

# ── inline server flat-key oracle (the server-side rule evaluator) ──────────
# Present (incl. None) wins; only truly-absent falls to the metadata bag. A
# unique sentinel stands in for the server's `undefined` (which the Python
# SDK represents as its terminal `None` for the absent case).
_MISSING = object()


def _collector_resolve_flat(payload, field):
    if field in payload:
        return payload[field]
    md = payload.get("metadata")
    if isinstance(md, dict) and field in md:
        return md[field]
    return _MISSING


class TestMatchesCondition(unittest.TestCase):
    def test_empty_matches_all(self):
        self.assertTrue(m(PAYLOAD, None))
        self.assertTrue(m(PAYLOAD, {}))

    def test_bare_leaf_eq(self):
        self.assertTrue(m(PAYLOAD, {"field": "paid_plan", "operator": "EQ", "value": "free"}))
        self.assertFalse(m(PAYLOAD, {"field": "paid_plan", "operator": "EQ", "value": "pro"}))

    def test_neq_absent_field_true(self):
        self.assertTrue(m(PAYLOAD, {"field": "paid_plan", "operator": "NEQ", "value": "pro"}))
        self.assertFalse(m(PAYLOAD, {"field": "paid_plan", "operator": "NEQ", "value": "free"}))
        self.assertTrue(m(PAYLOAD, {"field": "absent", "operator": "NEQ", "value": "pro"}))

    def test_contains(self):
        self.assertTrue(m(PAYLOAD, {"field": "provider", "operator": "CONTAINS", "value": "open"}))
        self.assertFalse(m(PAYLOAD, {"field": "provider", "operator": "CONTAINS", "value": "zzz"}))

    def test_in(self):
        self.assertTrue(m(PAYLOAD, {"field": "model", "operator": "IN", "value": ["gpt-4", "gpt-4o"]}))
        self.assertFalse(m(PAYLOAD, {"field": "model", "operator": "IN", "value": ["claude"]}))
        self.assertFalse(m(PAYLOAD, {"field": "model", "operator": "IN", "value": "gpt-4"}))

    def test_exists(self):
        self.assertTrue(m(PAYLOAD, {"field": "paid_plan", "operator": "EXISTS"}))
        self.assertFalse(m(PAYLOAD, {"field": "session_id", "operator": "EXISTS"}))
        # empty string counts as absent, matching SQL col != ''
        self.assertFalse(m({**PAYLOAD, "session_id": ""}, {"field": "session_id", "operator": "EXISTS"}))

    def test_unknown_operator_fails_closed(self):
        self.assertFalse(m(PAYLOAD, {"field": "model", "operator": "LIKE", "value": "gpt"}))

    def test_composite_and(self):
        self.assertTrue(m(PAYLOAD, {"combinator": "AND", "conditions": [
            {"field": "paid_plan", "operator": "EQ", "value": "free"},
            {"field": "model", "operator": "IN", "value": ["gpt-4"]},
        ]}))
        self.assertFalse(m(PAYLOAD, {"combinator": "AND", "conditions": [
            {"field": "paid_plan", "operator": "EQ", "value": "free"},
            {"field": "model", "operator": "EQ", "value": "claude"},
        ]}))

    def test_composite_or(self):
        self.assertTrue(m(PAYLOAD, {"combinator": "OR", "conditions": [
            {"field": "paid_plan", "operator": "EQ", "value": "pro"},
            {"field": "metadata.team", "operator": "EQ", "value": "x"},
        ]}))
        self.assertFalse(m(PAYLOAD, {"combinator": "OR", "conditions": [
            {"field": "paid_plan", "operator": "EQ", "value": "pro"},
            {"field": "metadata.team", "operator": "EQ", "value": "y"},
        ]}))

    def test_composite_defaults_and_and_empties(self):
        self.assertTrue(m(PAYLOAD, {"conditions": [{"field": "paid_plan", "operator": "EQ", "value": "free"}]}))
        self.assertTrue(m(PAYLOAD, {"combinator": "AND", "conditions": []}))
        self.assertFalse(m(PAYLOAD, {"combinator": "OR", "conditions": []}))


class TestF27FlatKeyResolver(unittest.TestCase):
    """The flat-key resolver must NOT treat a present-`None` as absent.

    The authoritative server flat-key resolver (the server-side rule evaluator) returns
    a PRESENT key (even value `None`) and only falls to the metadata bag when the
    key is truly ABSENT. Before the SDK additionally treated a present-`None`
    as absent and skipped to metadata.

    Node<->Python ASYMMETRY (stated verbatim in tests/localMatcher.test.ts too):
    Python's `_payload_field` IS module-importable, so this file asserts the
    RESOLVED VALUE directly; the Node twin infers it through public
    `matchesCondition` EQ/EXISTS/NEQ probes. The parity assertion (rubric 8)
    compares the resolved VALUE, not the invocation mechanism.
    """

    # Shared case table — resolved value pinned; `_MISSING` == server
    # `undefined`, which the SDK represents as terminal `None`.
    CASES = [
        # (name, payload, field, oracle_resolved_value)
        ("(a) present-None + metadata shadow",
         {"paid_plan": None, "metadata": {"paid_plan": "pro"}}, "paid_plan", None),
        ("(b) present-non-null + shadow",
         {"paid_plan": "free", "metadata": {"paid_plan": "pro"}}, "paid_plan", "free"),
        ("(c) absent + metadata present",
         {"metadata": {"source": "cli"}}, "source", "cli"),
        ("(d) absent + no metadata",
         {}, "paid_plan", _MISSING),
        ("(e) present-None + no shadow",
         {"paid_plan": None}, "paid_plan", None),
    ]

    def test_resolved_value_equals_inline_collector_oracle(self):
        # rubric 8/parity: SDK `_payload_field` == inline server oracle for
        # every case (with _MISSING == the SDK's terminal None for absent).
        for name, payload, field, oracle_val in self.CASES:
            with self.subTest(name=name):
                got = _payload_field(payload, field)
                oracle = _collector_resolve_flat(payload, field)
                self.assertIs(oracle, oracle_val)  # oracle matches the pinned expectation
                if oracle is _MISSING:
                    # server `undefined` == the SDK's terminal None
                    self.assertIsNone(got)
                else:
                    self.assertEqual(got, oracle)

    def test_a_present_none_shadow_resolves_to_none_not_metadata(self):
        # rubric 2(a): the headline fix — present-None wins over the metadata shadow.
        payload = {"paid_plan": None, "metadata": {"paid_plan": "pro"}}
        self.assertIsNone(_payload_field(payload, "paid_plan"))
        self.assertNotEqual(_payload_field(payload, "paid_plan"), "pro")
        # ...and the public matcher agrees:
        self.assertFalse(m(payload, {"field": "paid_plan", "operator": "EQ", "value": "pro"}))
        self.assertFalse(m(payload, {"field": "paid_plan", "operator": "EXISTS"}))
        self.assertTrue(m(payload, {"field": "paid_plan", "operator": "EQ", "value": None}))

    def test_a_neq_present_none_shadow_true(self):
        # rubric 13 (a-NEQ): resolved None flows into NEQ and does NOT re-match the
        # shadowed metadata "pro"; NEQ "pro" -> True post-fix.
        payload = {"paid_plan": None, "metadata": {"paid_plan": "pro"}}
        self.assertTrue(m(payload, {"field": "paid_plan", "operator": "NEQ", "value": "pro"}))

    def test_b_present_non_null_unchanged(self):
        payload = {"paid_plan": "free", "metadata": {"paid_plan": "pro"}}
        self.assertEqual(_payload_field(payload, "paid_plan"), "free")
        self.assertTrue(m(payload, {"field": "paid_plan", "operator": "EQ", "value": "free"}))

    def test_c_absent_falls_back_to_metadata(self):
        payload = {"metadata": {"source": "cli"}}
        self.assertEqual(_payload_field(payload, "source"), "cli")
        self.assertTrue(m(payload, {"field": "source", "operator": "EQ", "value": "cli"}))

    def test_d_absent_no_metadata_is_none(self):
        self.assertIsNone(_payload_field({}, "paid_plan"))
        self.assertFalse(m({}, {"field": "paid_plan", "operator": "EXISTS"}))

    def test_e_present_none_no_shadow_is_none(self):
        # regression guard: the no-shadow present-None case is unchanged.
        self.assertIsNone(_payload_field({"paid_plan": None}, "paid_plan"))
        self.assertFalse(m({"paid_plan": None}, {"field": "paid_plan", "operator": "EXISTS"}))

    def test_dotted_leaf_untouched(self):
        # rubric 11: dotted-path branch out of scope for unchanged.
        payload = {"intent": {"kind": "code"}}
        self.assertEqual(_payload_field(payload, "intent.kind"), "code")
        self.assertTrue(m(payload, {"field": "intent.kind", "operator": "EQ", "value": "code"}))


class TestF57Precedence(unittest.TestCase):
    """The payload BUILDER must mirror the server's evaluationPayload
    spread/field order (the remote /check evaluation payload).

    `end_user_id` and `metadata` sit BEFORE the `**md` spread (metadata CAN
    override them); every other canonical field sits AFTER the spread (canonical
    WINS over a metadata bag key of the same name). Pre-fix the SDK spread `**md`
    LAST, so metadata overrode EVERY canonical field — the opposite direction from
    the server (State-A != State-B).

    Node<->Python ASYMMETRY (stated verbatim in tests/localMatcher.test.ts too):
    `_build_payload` IS module-importable, so this file reads the resolved top-level
    key DIRECTLY; the Node twin drives twin block/allow directives through the
    public `evaluate` entry. The parity assertion compares the RESOLVED VALUE, not
    the invocation mechanism.
    """

    @staticmethod
    def _session(**kw):
        # TPSession-like: _build_payload reads user_id/paid_plan/session_id/metadata
        # via getattr, so a SimpleNamespace stands in for the real session.
        return SimpleNamespace(**kw)

    def test_5_provider_both_present_canonical_wins(self):
        p = _build_payload(self._session(user_id="u", metadata={"provider": "meta_prov"}),
                           {"provider": "canon_prov"})
        self.assertEqual(p["provider"], "canon_prov")

    def test_6_model_both_present_canonical_wins(self):
        p = _build_payload(self._session(user_id="u", metadata={"model": "meta_model"}),
                           {"model": "canon_model"})
        self.assertEqual(p["model"], "canon_model")

    def test_7_user_id_both_present_canonical_wins(self):
        p = _build_payload(self._session(user_id="u_canon", metadata={"user_id": "meta_user"}), {})
        self.assertEqual(p["user_id"], "u_canon")

    def test_8_trace_id_both_present_canonical_wins(self):
        p = _build_payload(self._session(user_id="u", metadata={"trace_id": "meta_trace"}),
                           {"trace_id": "canon_trace"})
        self.assertEqual(p["trace_id"], "canon_trace")

    def test_9_session_id_both_present_canonical_wins(self):
        p = _build_payload(self._session(user_id="u", session_id="sess_canon",
                                         metadata={"session_id": "meta_sess"}), {})
        self.assertEqual(p["session_id"], "sess_canon")

    def test_10_end_user_id_asymmetry_metadata_still_wins(self):
        # One payload: metadata carries BOTH end_user_id (before spread -> metadata
        # wins) and user_id (after spread -> canonical wins) — the before/after-spread
        # asymmetry in a single scenario (server:113 vs:117).
        p = _build_payload(
            self._session(user_id="u_canon",
                          metadata={"end_user_id": "meta_eu", "user_id": "meta_user"}),
            {},
        )
        self.assertEqual(p["end_user_id"], "meta_eu")  # BEFORE spread: metadata wins
        self.assertEqual(p["user_id"], "u_canon")      # AFTER spread: canonical wins

    def test_11_paid_plan_and_modality_net_canonical(self):
        p = _build_payload(
            self._session(user_id="u", paid_plan="free",
                          metadata={"paid_plan": "meta_plan", "modality": "meta_mode"}),
            {"modality": "embedding"},
        )
        self.assertEqual(p["paid_plan"], "free")
        self.assertEqual(p["modality"], "embedding")

    def test_12_meta_only_non_reserved_key_surfaces(self):
        p = _build_payload(self._session(user_id="u", metadata={"source": "cli"}), {})
        self.assertEqual(p["source"], "cli")

    def test_13_canonical_only_key_unchanged(self):
        p = _build_payload(self._session(user_id="u", metadata={}), {"provider": "canon_prov"})
        self.assertEqual(p["provider"], "canon_prov")


class TestF60Operation(unittest.TestCase):
    """The payload BUILDER now emits a canonical "operation" field (= the
    resolved "modality") after the **md spread, between "modality" and "intent",
    mirroring the server's evaluationPayload (remote /check evaluation).

    Pre-fix the builder emitted NO "operation" key, so a rule `operation EQ "chat"`
    resolved to None on the warm-local path (State A) while the inline /check path
    (State B) resolved it to a canonical string — the one divergence scoped out.

    Node<->Python ASYMMETRY (stated verbatim in tests/localMatcher.test.ts too):
    `_build_payload` IS module-importable, so this file reads the resolved
    "operation" key DIRECTLY and cross-checks via `evaluate`; the Node twin drives
    twin block/allow directives through the public `evaluate` entry. The parity
    assertion compares the RESOLVED VALUE, not the invocation mechanism.
    """

    @staticmethod
    def _session(**kw):
        return SimpleNamespace(**kw)

    @staticmethod
    def _block_pack(field, value):
        # Enforce-mode UNCONDITIONAL_BLOCK on one leaf: blocks iff <field> == value.
        return {
            "directives": [
                {
                    "id": "d1",
                    "kind": "UNCONDITIONAL_BLOCK",
                    "mode": "enforce",
                    "priority": 10,
                    "selector": {"match": {"field": field, "operator": "EQ", "value": value}},
                }
            ]
        }

    def _blocks(self, pack, session, observed):
        from token_police.local_evaluator import evaluate
        return evaluate(pack, session, observed)["decision"]["status"] == "blocked"

    def test_f60_default_operation_is_chat(self):
        # No modality and no intent.kind -> operation defaults to "chat".
        session = self._session(user_id="u", metadata={})
        p = _build_payload(session, {})
        self.assertEqual(p["operation"], "chat")
        # operation === modality (SDK analog of server operation===modality)
        self.assertEqual(p["operation"], p["modality"])
        # via evaluate: operation EQ "chat" BLOCKS (RED pre-fix, GREEN post-fix)
        self.assertTrue(self._blocks(self._block_pack("operation", "chat"), session, {}))
        self.assertFalse(self._blocks(self._block_pack("operation", "embedding"), session, {}))

    def test_f60_modality_set_drives_operation(self):
        session = self._session(user_id="u", metadata={})
        p = _build_payload(session, {"modality": "embedding"})
        self.assertEqual(p["operation"], "embedding")
        self.assertEqual(p["operation"], p["modality"])
        self.assertTrue(self._blocks(self._block_pack("operation", "embedding"), session,
                                     {"modality": "embedding"}))
        self.assertFalse(self._blocks(self._block_pack("operation", "chat"), session,
                                      {"modality": "embedding"}))

    def test_f60_intent_kind_fallback_drives_operation(self):
        session = self._session(user_id="u", metadata={})
        p = _build_payload(session, {"intent": {"kind": "embedding"}})
        self.assertEqual(p["operation"], "embedding")
        self.assertEqual(p["operation"], p["modality"])

    def test_f60_metadata_cannot_override_operation(self):
        # metadata bag carries operation:"embedding" but modality is unset -> canonical
        # operation resolves to "chat" (default) and sits AFTER the **md spread.
        session = self._session(user_id="u", metadata={"operation": "embedding"})
        p = _build_payload(session, {})
        self.assertEqual(p["operation"], "chat")
        self.assertTrue(self._blocks(self._block_pack("operation", "chat"), session, {}))
        self.assertFalse(self._blocks(self._block_pack("operation", "embedding"), session, {}))


class TestJsCoercionParity(unittest.TestCase):
    """EQ/NEQ/IN/CONTAINS must coerce exactly like the server-side evaluator's
    JS ===/String() — NOT Python's loose ==/in/str(). Divergences that used to
    let the SDK locally ALLOW what the server BLOCKS (e.g. `True` matching a `1`
    rule, or CONTAINS over Python's "True"/"100.0" text) are the enforcement hole
    this closes. Every case pins the SERVER's answer.

    Node<->Python ASYMMETRY (stated verbatim in tests/localMatcher.test.ts too):
    the JS String()/=== analogs `_js_string`/`_js_strict_eq` ARE module-importable
    here, so this file spot-checks them directly; the Node twin can only drive the
    identical matrix through public `matchesCondition` (its String()/=== aren't
    exported). The parity assertion compares the RESOLVED bool, not the mechanism.
    """

    @staticmethod
    def _leaf(op, value, payload_val):
        # Bare leaf on field "f"; payload_val positioned as the payload-side value.
        payload = {"f": payload_val}
        return m(payload, {"field": "f", "operator": op, "value": value})

    def test_eq_js_strict_semantics(self):
        # bool is its own type: True !== 1.
        self.assertFalse(self._leaf("EQ", 1, True))
        self.assertTrue(self._leaf("EQ", True, True))
        # single JS number type: 1.0 === 1.
        self.assertTrue(self._leaf("EQ", 1, 1.0))
        # string vs number never equal.
        self.assertFalse(self._leaf("EQ", 1, "1"))
        # bool vs its string form never equal.
        self.assertFalse(self._leaf("EQ", "true", True))
        # absent field resolves to None; EQ None => null === null.
        self.assertTrue(m({}, {"field": "f", "operator": "EQ", "value": None}))

    def test_neq_mirrors_eq(self):
        self.assertTrue(self._leaf("NEQ", 1, True))
        self.assertFalse(self._leaf("NEQ", True, True))
        self.assertFalse(self._leaf("NEQ", 1, 1.0))
        self.assertTrue(self._leaf("NEQ", 1, "1"))
        self.assertTrue(self._leaf("NEQ", "true", True))

    def test_in_js_strict_membership(self):
        self.assertFalse(self._leaf("IN", [1], True))
        self.assertTrue(self._leaf("IN", [True], True))
        self.assertTrue(self._leaf("IN", [1.0], 1))
        self.assertTrue(self._leaf("IN", ["a"], "a"))
        # non-list value never matches.
        self.assertFalse(self._leaf("IN", "a", "a"))

    def test_contains_js_string_coercion(self):
        # enforcement-hole case: payload True, value "true" -> True (was False
        # under Python str(True)="True").
        self.assertTrue(self._leaf("CONTAINS", "true", True))
        # payload True, value "True" -> False (JS String(true) is "true").
        self.assertFalse(self._leaf("CONTAINS", "True", True))
        # payload 100.0 stringifies to "100" (no ".0"): ".0" absent, "100" present.
        self.assertFalse(self._leaf("CONTAINS", ".0", 100.0))
        self.assertTrue(self._leaf("CONTAINS", "100", 100.0))
        # 1e-05 -> JS "0.00001" (plain decimal, never exponential).
        self.assertTrue(self._leaf("CONTAINS", "0.00001", 1e-05))
        self.assertFalse(self._leaf("CONTAINS", "1e-05", 1e-05))
        # non-finite floats render as JS Infinity / NaN.
        self.assertTrue(self._leaf("CONTAINS", "Infinity", float("inf")))
        self.assertTrue(self._leaf("CONTAINS", "NaN", float("nan")))

    def test_js_string_direct_spot_checks(self):
        # Direct assertions on the String() analog (Python-only privilege).
        self.assertEqual(_js_string(True), "true")
        self.assertEqual(_js_string(False), "false")
        self.assertEqual(_js_string(100.0), "100")
        self.assertEqual(_js_string(1e-05), "0.00001")
        self.assertEqual(_js_string(float("inf")), "Infinity")
        self.assertEqual(_js_string(float("-inf")), "-Infinity")
        self.assertEqual(_js_string(float("nan")), "NaN")
        self.assertEqual(_js_string(1), "1")
        self.assertEqual(_js_string("x"), "x")

    def test_js_strict_eq_direct_spot_checks(self):
        self.assertFalse(_js_strict_eq(True, 1))
        self.assertTrue(_js_strict_eq(True, True))
        self.assertTrue(_js_strict_eq(1, 1.0))
        self.assertFalse(_js_strict_eq("1", 1))
        self.assertTrue(_js_strict_eq(None, None))
        self.assertFalse(_js_strict_eq(None, "x"))

    def test_exists_behavior_unchanged(self):
        # "" absent; 0 and False present (v !== '' on the server side).
        self.assertFalse(m({"f": ""}, {"field": "f", "operator": "EXISTS"}))
        self.assertTrue(m({"f": 0}, {"field": "f", "operator": "EXISTS"}))
        self.assertTrue(m({"f": False}, {"field": "f", "operator": "EXISTS"}))

    def test_no_throw_for_exotic_payload_values(self):
        # Golden rule: hostile/exotic metadata must degrade to a bool, never raise.
        exotic = [b"bytes", object(), {"k": "v"}, ["a", "b"]]
        ops = [
            ("EQ", "x"), ("EQ", 1), ("EQ", None),
            ("NEQ", "x"), ("EXISTS", None),
            ("CONTAINS", "x"), ("CONTAINS", object()),
            ("IN", ["x"]), ("IN", "x"),
        ]
        for val in exotic:
            for op, rule_value in ops:
                with self.subTest(val=type(val).__name__, op=op):
                    r = m({"f": val}, {"field": "f", "operator": op, "value": rule_value})
                    self.assertIsInstance(r, bool)
        # helpers are total on exotic inputs too.
        for val in exotic:
            self.assertIsInstance(_js_string(val), str)
            self.assertIsInstance(_js_strict_eq(val, "x"), bool)
            self.assertIsInstance(_js_strict_eq(val, val), bool)


if __name__ == "__main__":
    unittest.main()
