"""
`armable_miss` — local-evaluator metadata for the C-14 stale-stream gate.

Set only when an entity-gated directive's selector matched the call but the
computed group tag was absent from its streamed `entities` set AND the
directive would otherwise ENFORCE (not a directive-level dry_run, and not
`force_shadow`) — i.e. a missed `entity_blocked` / `entity_rerouted` delta that
would have flipped the decision. Never affects the decision object itself, and
is present as a key ONLY when true — an allow with no armable miss stays
byte-identical to the pre-fix shape (no `armable_miss` key at all).

Sibling: token-police-node/tests/armableMissFlag.test.ts pins the same
scenarios against the Node SDK.
"""
from types import SimpleNamespace

from token_police.local_evaluator import evaluate

SESSION = SimpleNamespace(user_id="u1", paid_plan="free", metadata={}, session_id=None)
CALL = {"model": "gpt-4", "provider": "openai"}


def _entity_block_directive(mode="enforce", entities=None, match_field="user_id"):
    return {
        "id": "eb1",
        "kind": "ENTITY_BLOCK",
        "mode": mode,
        "priority": 10,
        "selector": {
            "match": None if match_field is None else {"field": match_field, "operator": "EXISTS"},
            "group_by": ["user_id"],
        },
        "entities": [] if entities is None else entities,
    }


_NO_ENTITIES_KEY = object()  # sentinel: omit the "entities" key entirely


def _entity_reroute_directive(mode="enforce", entities=_NO_ENTITIES_KEY):
    d = {
        "id": "er1",
        "kind": "REROUTE",
        "mode": mode,
        "priority": 10,
        "selector": {"match": {"field": "user_id", "operator": "EXISTS"}, "group_by": ["user_id"]},
        "reroute": {"from": None, "to": {"provider": "openai", "model": "gpt-3.5-turbo"}},
    }
    if entities is not _NO_ENTITIES_KEY:
        d["entities"] = entities
    return d


# ── ENTITY_BLOCK ────────────────────────────────────────────────────────


def test_enforce_directive_selector_match_tag_not_armed_sets_flag_decision_stays_allowed():
    pack = {"directives": [_entity_block_directive(entities=[])], "loop_blocks": []}
    res = evaluate(pack, SESSION, CALL)
    assert res["decision"] == {"status": "allowed", "rule_id": None, "mode": "enforce", "reroute": None}
    assert res["observations"] == []
    assert res.get("armable_miss") is True


def test_allow_with_no_armable_miss_has_no_key_at_all():
    pack = {"directives": [], "loop_blocks": []}
    res = evaluate(pack, SESSION, CALL)
    assert res == {
        "decision": {"status": "allowed", "rule_id": None, "mode": "enforce", "reroute": None},
        "observations": [],
    }
    assert "armable_miss" not in res


def test_tag_armed_normal_block_no_armable_miss():
    pack = {"directives": [_entity_block_directive(entities=["u1"])], "loop_blocks": []}
    res = evaluate(pack, SESSION, CALL)
    assert res["decision"]["status"] == "blocked"
    assert res["decision"]["rule_id"] == "eb1"
    assert "armable_miss" not in res


def test_selector_non_match_no_armable_miss():
    pack = {
        "directives": [_entity_block_directive(entities=[], match_field="does_not_exist")],
        "loop_blocks": [],
    }
    res = evaluate(pack, SESSION, CALL)
    assert res["decision"]["status"] == "allowed"
    assert "armable_miss" not in res


def test_directive_dry_run_entity_miss_does_not_set_flag():
    pack = {"directives": [_entity_block_directive(mode="dry_run", entities=[])], "loop_blocks": []}
    res = evaluate(pack, SESSION, CALL)
    assert res["decision"]["status"] == "allowed"
    # dry_run only emits would_block on a HIT, not a miss — the miss branch
    # `continue`s before the dry_run observation is ever built.
    assert res["observations"] == []
    assert "armable_miss" not in res


def test_force_shadow_entity_miss_does_not_set_flag_even_for_enforce_directive():
    pack = {"directives": [_entity_block_directive(mode="enforce", entities=[])], "loop_blocks": []}
    res = evaluate(pack, SESSION, CALL, True)
    assert res["decision"]["status"] == "allowed"
    assert "armable_miss" not in res


# ── entity-gated REROUTE ──────────────────────────────────────────────


def test_entity_gated_reroute_enforce_miss_sets_flag():
    pack = {"directives": [_entity_reroute_directive(entities=[])], "loop_blocks": []}
    res = evaluate(pack, SESSION, CALL)
    assert res["decision"]["status"] == "allowed"
    assert res.get("armable_miss") is True


def test_entity_gated_reroute_dry_run_miss_does_not_set_flag():
    pack = {"directives": [_entity_reroute_directive(mode="dry_run", entities=[])], "loop_blocks": []}
    res = evaluate(pack, SESSION, CALL)
    assert res["decision"]["status"] == "allowed"
    assert "armable_miss" not in res


def test_unconditional_reroute_no_entities_key_never_sets_flag():
    pack = {"directives": [_entity_reroute_directive()], "loop_blocks": []}  # no "entities" key
    res = evaluate(pack, SESSION, CALL)
    # Unconditional: applies whenever it matches — no entities check at all.
    assert res["decision"]["status"] == "rerouted"
    assert "armable_miss" not in res


# ── unconditional directives ──────────────────────────────────────────


def test_unconditional_block_never_sets_flag():
    pack = {
        "directives": [
            {
                "id": "u1",
                "kind": "UNCONDITIONAL_BLOCK",
                "mode": "enforce",
                "priority": 10,
                "selector": {"match": None, "group_by": []},
            }
        ],
        "loop_blocks": [],
    }
    res = evaluate(pack, SESSION, CALL)
    assert res["decision"]["status"] == "blocked"
    assert "armable_miss" not in res
