"""Constructing a TokenPolice() directly must not clobber the installed client's
id. Only the client installed as the global one (via set_client / init) owns the
module-global client id used for fleet attribution in payloads/headers.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import state
from token_police.client import TokenPolice


class TestClientIdIsolation(unittest.TestCase):
    def tearDown(self):
        try:
            tp.uninstrument()
        except Exception:
            pass

    def test_direct_construction_does_not_change_installed_id(self):
        installed = tp.init(api_key="tp_sk_test_installed", firewall="off")
        installed_id = state.get_client_id()

        # Installed client's payload identity == the module-global id.
        self.assertTrue(installed_id)
        self.assertEqual(installed_id, installed.client_id)
        self.assertEqual(installed.client_id, installed._common_headers["X-TP-Client-Id"])

        # Construct a second client WITHOUT installing it.
        other = TokenPolice(api_key="tp_sk_test_other", firewall="off")

        # Its own id differs, and the installed client's identity is unchanged.
        self.assertNotEqual(other.client_id, installed_id)
        self.assertEqual(state.get_client_id(), installed_id)
        self.assertIs(state.get_client(), installed)
        self.assertEqual(installed._common_headers["X-TP-Client-Id"], installed_id)

    def test_each_direct_construction_gets_a_distinct_id(self):
        a = TokenPolice(api_key="tp_sk_a", firewall="off")
        b = TokenPolice(api_key="tp_sk_b", firewall="off")
        self.assertTrue(a.client_id)
        self.assertTrue(b.client_id)
        self.assertNotEqual(a.client_id, b.client_id)


if __name__ == "__main__":
    unittest.main()
