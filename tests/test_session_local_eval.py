import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import token_police as tp
from token_police.local_evaluator import evaluate, _build_payload
from token_police.enforcer import _stash_local_decision


def _entity_block_pack(entities):
    # Per-session sliding-budget rules ship to the SDK as ordinary ENTITY_BLOCK
    # directives whose group_by is ["session_id"].
    return {
        "directives": [
            {
                "id": "rule_sess",
                "kind": "ENTITY_BLOCK",
                "mode": "enforce",
                "selector": {
                    "match": {"field": "session_id", "operator": "EXISTS"},
                    "group_by": ["session_id"],
                },
                "entities": entities,
            }
        ]
    }


class TestSessionLocalEval(unittest.TestCase):
    def test_build_payload_exposes_session_id(self):
        session = SimpleNamespace(user_id="u1", paid_plan="free", metadata={}, session_id="conv_42")
        payload = _build_payload(session, {})
        self.assertEqual(payload["session_id"], "conv_42")

    def test_blocks_session_in_entities_set(self):
        session = SimpleNamespace(user_id="u1", paid_plan="free", metadata={}, session_id="conv_42")
        res = evaluate(_entity_block_pack(["conv_42"]), session, {})
        self.assertEqual(res["decision"]["status"], "blocked")
        self.assertEqual(res["decision"]["rule_id"], "rule_sess")

    def test_allows_session_not_in_entities_set(self):
        # Not armed → local eval returns allowed, so the enforcer skips /check.
        session = SimpleNamespace(user_id="u1", paid_plan="free", metadata={}, session_id="conv_99")
        res = evaluate(_entity_block_pack(["conv_42"]), session, {})
        self.assertEqual(res["decision"]["status"], "allowed")
        self.assertEqual(res["observations"], [])


class TestStashLocalDecisionGuard(unittest.TestCase):
    # A missing (None) session must be a silent no-op (fail-open), never an
    # AttributeError.
    # NOTE (negative control / assertion 16): if `if session is None: return` is
    # removed from _stash_local_decision, test_none_session_no_op below raises
    # AttributeError (NoneType has no attribute _local_decision) and goes RED —
    # proving the guard is exercised (non-vacuous).
    def test_none_session_no_op(self):
        # Must not raise, returns None.
        self.assertIsNone(_stash_local_decision(None, "blocked", "r1", True))

    def test_real_session_stash_unchanged_no_reroute(self):
        # No obs key is current in this unittest context, so the stash lands
        # untagged (key=None) in the keyed store — same-shape entry as before,
        # just addressed via `_local_decisions` instead of the old flat slot.
        session = SimpleNamespace()
        _stash_local_decision(session, "blocked", "r1", True)
        self.assertEqual(len(session._local_decisions), 1)
        self.assertIsNone(session._local_decisions[-1]["key"])
        self.assertEqual(
            session._local_decisions[-1]["ld"],
            {
                "outcome": "blocked",
                "rule_id": "r1",
                "mode": "enforce",
                "verified_by_check": True,
            },
        )

    def test_real_session_stash_with_reroute(self):
        session = SimpleNamespace()
        _stash_local_decision(session, "rerouted", "r2", True, reroute={"to": "x"})
        self.assertEqual(
            session._local_decisions[-1]["ld"],
            {
                "outcome": "rerouted",
                "rule_id": "r2",
                "mode": "enforce",
                "verified_by_check": True,
                "reroute": {"to": "x"},
            },
        )
        # Without reroute, the key is absent.
        session2 = SimpleNamespace()
        _stash_local_decision(session2, "blocked", "r3", False)
        self.assertNotIn("reroute", session2._local_decisions[-1]["ld"])


class TestCheckSendsSessionId(unittest.TestCase):
    def setUp(self):
        self.client = tp.init(api_key="tp_sk_test123", enforce=True)

    def tearDown(self):
        tp.uninstrument()

    @patch("httpx.Client.post")
    def test_session_id_present_when_provided_absent_otherwise(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "allowed"}
        mock_post.return_value = mock_response

        self.client.check_sync(user_id="u1", session_id="conv_42")
        body = mock_post.call_args.kwargs["json"]
        self.assertEqual(body["session_id"], "conv_42")

        self.client.check_sync(user_id="u1", session_id="")
        body = mock_post.call_args.kwargs["json"]
        self.assertNotIn("session_id", body)


if __name__ == "__main__":
    unittest.main()
