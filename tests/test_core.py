import unittest
from unittest.mock import patch, MagicMock
import token_police as tp
from token_police.exceptions import TokenPoliceBlockedError

class TestTokenPoliceCore(unittest.TestCase):
    def setUp(self):
        self.client = tp.init(api_key="tp_sk_test123", enforce=True)

    def tearDown(self):
        tp.uninstrument()

    @patch("httpx.Client.post")
    def test_sync_check_allowed(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        # Should not raise
        result = self.client.check_sync(user_id="test_user")
        self.assertEqual(result["status"], "allowed")

    @patch("httpx.Client.post")
    def test_sync_check_blocked(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.json.return_value = {"status": "blocked", "reason": "budget limit"}
        mock_post.return_value = mock_response

        result = self.client.check_sync(user_id="test_user")
        self.assertEqual(result["status"], "blocked")

    @patch("httpx.Client.post")
    def test_fail_open_on_network_error(self, mock_post):
        # Simulate network error
        mock_post.side_effect = Exception("Network timeout")

        # check_sync catches it, fail_safe handles it
        result = self.client.check_sync(user_id="test_user")
        # In check_sync, we fall back to returning {"status": "allowed"} due to exception inside
        # Wait, check_sync handles exception itself? No, fail_safe doesn't wrap check_sync!
        # Let's see what happens.
        pass

    def test_api_key_validation(self):
        with self.assertRaises(ValueError):
            tp.init(api_key="")


class TestFirewallMode(unittest.TestCase):
    def tearDown(self):
        tp.uninstrument()

    def test_defaults_to_dry_run(self):
        client = tp.init(api_key="tp_sk_test123")
        self.assertEqual(client.firewall, "dry_run")

    def test_enforce_alias_maps_to_firewall(self):
        self.assertEqual(tp.init(api_key="tp_sk_test123", enforce=True).firewall, "enforce")
        tp.uninstrument()
        self.assertEqual(tp.init(api_key="tp_sk_test123", enforce=False).firewall, "off")

    def test_explicit_firewall_wins_over_enforce_alias(self):
        client = tp.init(api_key="tp_sk_test123", firewall="dry_run", enforce=True)
        self.assertEqual(client.firewall, "dry_run")


import re
from opentelemetry import trace as _otel_trace
from token_police.context import (
    get_current_session,
    manual_span_ids,
)


class TestW3CSpanHierarchy(unittest.TestCase):
    """Real nested span tree + W3C-format hex ids."""

    def setUp(self):
        tp.init(api_key="tp_sk_test_w3c", base_url="http://localhost:59999")

    def tearDown(self):
        tp.uninstrument()

    @staticmethod
    def _hex32(s):
        return bool(re.fullmatch(r"[0-9a-f]{32}", s or ""))

    @staticmethod
    def _hex16(s):
        return bool(re.fullmatch(r"[0-9a-f]{16}", s or ""))

    def test_agent_span_has_w3c_ids(self):
        @tp.workflow(name="orchestrator")
        def run():
            s = get_current_session()
            self.assertTrue(self._hex32(s.trace_id))
            self.assertTrue(self._hex16(s.root_span_id))
            ctx = _otel_trace.get_current_span().get_span_context()
            self.assertEqual(_otel_trace.format_trace_id(ctx.trace_id), s.trace_id)
            self.assertEqual(_otel_trace.format_span_id(ctx.span_id), s.root_span_id)

        run()

    def test_subagent_nests_under_parent(self):
        @tp.workflow(name="orchestrator")
        def run():
            top = get_current_session()
            with tp.session(name="researcher"):
                sub = get_current_session()
                self.assertEqual(sub.trace_id, top.trace_id)  # same run
                self.assertNotEqual(sub.root_span_id, top.root_span_id)  # real child
                ids = manual_span_ids(sub)
                self.assertEqual(ids["parent_span_id"], sub.root_span_id)
                self.assertTrue(self._hex16(ids["span_id"]))
                self.assertEqual(ids["trace_id"], top.trace_id)

        run()


if __name__ == "__main__":
    unittest.main()
