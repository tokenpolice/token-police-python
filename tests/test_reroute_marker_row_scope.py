"""B4 — row scope of the applied-reroute provenance marker (`_tp_routing`).
Python twin of tests/rerouteMarkerRowScope.test.ts (Node).

Bug: `_apply_reroute` writes `_tp_routing` onto `session.metadata` (a
SESSION-wide field, unchanged by this fix — it's the /check-payload input
rules may match on). Every row-emission site used to copy session metadata
WHOLESALE into its `/log` payload, so ONE rerouted call's marker rode EVERY
later row of that session: tool rows (via the OTel tool-span path — the
manual `tp.tool_span()` path in context.py never copied session metadata at
all, pre- or post-fix, so it carries no B4 exposure and is not covered here),
agent/chain structural anchors, and unrelated sibling calls' rows.

Fix under test: `token_police/enforcer.py`'s per-call keyed `_routing_markers`
store, wired through `_copy_session_metadata` (strip on every row-metadata
copy) / `_stamp_routing_marker` (re-add ONLY on the owning model-call row, by
exact obs-key match) / `_row_metadata_from_session` (the failure-log site that
used to hand `session.metadata` to `log_sync` by reference) / the
`telemetry.py` OTel on_end strip for agent/chain/tool rows.

These tests drive the REAL functions (`_apply_reroute`, `_log_li_py`,
`_emit_call_failure_log`, `TokenPoliceSpanProcessor.on_end`) through the real
session contextvar (`tp.session(...)`) and real per-call obs key
(`_state.mint_obs_key()`) — no network, fully offline, deterministic.

Coverage (numbering matches the B4 test plan):
  (8)  rerouted call's own llm row CONTAINS _tp_routing (correct
       original/actual model) AND session.metadata._tp_routing remains set
  (9)  tool row (OTel path, telemetry.py `_log_tool_span`) after reroute:
       none [FAILS IF THE FIX IS REVERTED]
  (10) agent/chain structural row (telemetry.py `_log_agent_span`): none
  (11) subsequent NON-rerouted call's row: none
       [FAILS IF THE FIX IS REVERTED]
  (12) two concurrent asyncio calls, one rerouted: rerouted row carries its
       OWN marker, sibling carries none
  (13, LOW) rerouted call's FAILURE row keeps the marker
  (14) customer metadata keys survive on every row (folded into 8/9/10/11)
  (15) `_AnthropicAsyncStreamMgrWrapper._rebuild_after_reroute` withdrawal —
       SKIPPED here (no ready harness for a full stream-manager rebuild in
       this pass); the withdrawal PRIMITIVE it calls
       (`_drop_routing_marker`) is covered at the store level in
       test_routing_marker_store.py's TestDropRoutingMarker.
"""
import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from opentelemetry.trace import StatusCode

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as _state
from token_police.telemetry import TokenPoliceSpanProcessor


REROUTE_DIRECTIVE = {
    "reroute": {
        "mode": "enforce",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "rule_id": "rule_rr",
        "original": {"provider": "openai", "model": "gpt-4o"},
    },
}


class _FakeLI:
    """Provider-class stand-in: `_li_provider_from_instance()` keys off
    `type(instance).__name__`; the exact resolved provider/model don't
    matter for these assertions. `last_response=None` short-circuits
    `_extract_li_usage_py` to a clean zeros fallback (see
    tests/test_llamaindex_verbatim_usage.py's usage of the same shape)."""

    model = "gpt-4o-mini"


class _CaptureClient:
    """Captures every log_sync(**kw) call, in call order."""

    def __init__(self):
        self.calls = []

    def log_sync(self, **kw):
        self.calls.append(kw)


def _fake_readable_span(attrs, name="span"):
    """Minimal ReadableSpan stand-in for TokenPoliceSpanProcessor.on_end()."""
    return NS(
        attributes=attrs,
        name=name,
        start_time=1_000_000_000,
        end_time=2_000_000_000,
        status=NS(status_code=StatusCode.OK, description=""),
    )


class Base(unittest.TestCase):
    def setUp(self):
        _state.reset_pack()

    def tearDown(self):
        _state.reset_pack()


# ═════════════════════════════════════════════════════════════════════════
# (8) rerouted call's own llm row — CONTAINS the marker
# ═════════════════════════════════════════════════════════════════════════
class Test8OwnLlmRow(Base):
    def test_carries_marker_correct_fields_session_metadata_unchanged(self):
        client = _CaptureClient()
        with tp.session(name="wf", user_id="u1",
                         metadata={"customer_key": "keep-me"}) as sess:
            key = _state.mint_obs_key()
            kwargs = {"model": "gpt-4o"}
            with mock.patch.object(_enforcer, "get_client", return_value=client):
                status = _enforcer._apply_reroute(REROUTE_DIRECTIVE, kwargs, "openai")
                self.assertEqual(status, "applied")
                self.assertEqual(kwargs["model"], "gpt-4o-mini")
                _enforcer._log_li_py(_FakeLI(), None, 0, "call1", None, obs_key=key)

            self.assertEqual(len(client.calls), 1)
            metadata = client.calls[0]["metadata"]
            routing = metadata.get("_tp_routing")
            self.assertTrue(routing)
            self.assertEqual(routing["rule_id"], "rule_rr")
            self.assertEqual(routing["original_model"], "gpt-4o")
            self.assertEqual(routing["actual_model"], "gpt-4o-mini")
            self.assertEqual(routing["original_provider"], "openai")
            self.assertEqual(routing["actual_provider"], "openai")
            # (14) customer metadata key survives alongside the marker.
            self.assertEqual(metadata.get("customer_key"), "keep-me")

            # The session write is UNCHANGED — still the back-compat /check
            # input.
            self.assertTrue(sess.metadata.get("_tp_routing"))


# ═════════════════════════════════════════════════════════════════════════
# (9) tool row (OTel path) after reroute — NONE
# ═════════════════════════════════════════════════════════════════════════
class Test9ToolRow(Base):
    def test_otel_tool_span_never_carries_marker(self):
        """[FAILS IF FIX REVERTED] — pre-fix `_log_tool_span` copied every
        `tp.meta.*` attribute (including the span-start snapshot of
        `_tp_routing`) straight into the row's metadata."""
        client = _CaptureClient()
        attrs = {
            "tp.kind": "tool",
            "tp.user_id": "u1",
            "tp.paid_plan": "free",
            "tp.workflow_name": "wf",
            "tp.session_id": "s1",
            # Simulates what on_start actually stamps when session.metadata
            # carries `_tp_routing` at span-open time (it JSON-serializes
            # EVERY metadata key into `tp.meta.*` attrs by design — the strip
            # is on_end's job, not the attribute build's).
            "tp.meta._tp_routing": '{"rule_id": "rule_rr", "actual_model": "gpt-4o-mini"}',
            "tp.meta.customer_key": "keep-me",
        }
        with mock.patch("token_police.state.get_client", return_value=client):
            TokenPoliceSpanProcessor().on_end(
                _fake_readable_span(attrs, name="web_search"))

        self.assertEqual(len(client.calls), 1)
        metadata = client.calls[0]["metadata"]
        self.assertNotIn("_tp_routing", metadata)
        self.assertEqual(metadata.get("customer_key"), "keep-me")  # (14)


# ═════════════════════════════════════════════════════════════════════════
# (10) agent/chain structural row — NONE
# ═════════════════════════════════════════════════════════════════════════
class Test10StructuralRow(Base):
    def test_agent_kind_never_carries_marker(self):
        client = _CaptureClient()
        attrs = {
            "tp.kind": "agent",
            "tp.user_id": "u1",
            "tp.paid_plan": "free",
            "tp.workflow_name": "wf",
            "tp.session_id": "s1",
            "tp.meta._tp_routing": '{"rule_id": "rule_rr"}',
            "tp.meta.customer_key": "keep-me",
        }
        with mock.patch("token_police.state.get_client", return_value=client):
            TokenPoliceSpanProcessor().on_end(_fake_readable_span(attrs))

        self.assertEqual(len(client.calls), 1)
        metadata = client.calls[0]["metadata"]
        self.assertNotIn("_tp_routing", metadata)
        self.assertEqual(metadata.get("customer_key"), "keep-me")

    def test_chain_kind_never_carries_marker(self):
        client = _CaptureClient()
        attrs = {
            "tp.kind": "chain",
            "tp.user_id": "u1",
            "tp.paid_plan": "free",
            "tp.workflow_name": "wf",
            "tp.session_id": "s1",
            "tp.meta._tp_routing": '{"rule_id": "rule_rr"}',
        }
        with mock.patch("token_police.state.get_client", return_value=client):
            TokenPoliceSpanProcessor().on_end(_fake_readable_span(attrs))

        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("_tp_routing", client.calls[0]["metadata"])


# ═════════════════════════════════════════════════════════════════════════
# (11) subsequent NON-rerouted call's row — NONE
# ═════════════════════════════════════════════════════════════════════════
class Test11SubsequentNonReroutedCall(Base):
    def test_no_marker_even_after_an_earlier_reroute_in_the_same_session(self):
        """[FAILS IF FIX REVERTED] — pre-fix every row copied
        session.metadata wholesale, including `_tp_routing`."""
        client = _CaptureClient()
        with mock.patch.object(_enforcer, "get_client", return_value=client):
            with tp.session(name="wf", user_id="u1"):
                # Call 1: rerouted. Stashes under call 1's own key; no row
                # emitted for it directly.
                _state.mint_obs_key()
                kwargs = {"model": "gpt-4o"}
                status = _enforcer._apply_reroute(REROUTE_DIRECTIVE, kwargs, "openai")
                self.assertEqual(status, "applied")

                # Call 2: its OWN obs key, no reroute happened under it.
                key2 = _state.mint_obs_key()
                _enforcer._log_li_py(_FakeLI(), None, 1, "call2", None, obs_key=key2)

        self.assertEqual(len(client.calls), 1)  # only call 2's row
        self.assertNotIn("_tp_routing", client.calls[0]["metadata"])


# ═════════════════════════════════════════════════════════════════════════
# (12) two concurrent asyncio calls, one rerouted — isolation at ROW level
# ═════════════════════════════════════════════════════════════════════════
class Test12ConcurrentCalls(Base):
    def test_rerouted_row_carries_own_marker_sibling_carries_none(self):
        client = _CaptureClient()

        async def rerouted_call():
            key = _state.mint_obs_key()
            kwargs = {"model": "gpt-4o"}
            _enforcer._apply_reroute(REROUTE_DIRECTIVE, kwargs, "openai")
            # Yield — stands in for the real gap between check and the
            # provider's own LLM HTTP call, during which the sibling below
            # interleaves on the SAME shared session (mirrors
            # test_local_decision_keyed_store.py's `_checked_call`).
            await asyncio.sleep(0)
            _enforcer._log_li_py(_FakeLI(), None, 0, "rerouted_call", None, obs_key=key)

        async def sibling_call():
            key = _state.mint_obs_key()
            await asyncio.sleep(0)
            _enforcer._log_li_py(_FakeLI(), None, 1, "sibling_call", None, obs_key=key)

        async def run():
            with tp.session(name="wf", user_id="u1"):
                await asyncio.gather(rerouted_call(), sibling_call())

        # Patched ONCE around the whole gather — mock.patch's save/restore is
        # not concurrency-safe across two Tasks entering/exiting it
        # independently (a bug in an earlier draft of this test: patching
        # inside each coroutine let one Task's __exit__ un-patch get_client
        # while the other was still inside its own `with` block, silently
        # dropping its row).
        with mock.patch.object(_enforcer, "get_client", return_value=client):
            asyncio.run(run())

        self.assertEqual(len(client.calls), 2)
        by_name = {c["span"]["span_name"]: c for c in client.calls}
        self.assertTrue(by_name["rerouted_call"]["metadata"].get("_tp_routing"))
        self.assertNotIn("_tp_routing", by_name["sibling_call"]["metadata"])


# ═════════════════════════════════════════════════════════════════════════
# (13, LOW) rerouted call's FAILURE row keeps the marker
# ═════════════════════════════════════════════════════════════════════════
class Test13FailureRow(Base):
    def test_rerouted_calls_own_failure_row_keeps_marker(self):
        client = _CaptureClient()
        with tp.session(name="wf", user_id="u1") as sess:
            key = _state.mint_obs_key()
            kwargs = {"model": "gpt-4o"}
            status = _enforcer._apply_reroute(REROUTE_DIRECTIVE, kwargs, "openai")
            self.assertEqual(status, "applied")

            sess._call_outcome = {"status": "failed", "error_type": "APIError"}
            _enforcer._emit_call_failure_log(client, sess, obs_key=key)

        self.assertEqual(len(client.calls), 1)
        metadata = client.calls[0]["metadata"]
        routing = metadata.get("_tp_routing")
        self.assertTrue(routing)
        self.assertEqual(routing["rule_id"], "rule_rr")

    def test_siblings_failure_row_own_key_no_reroute_carries_no_marker(self):
        client = _CaptureClient()
        with tp.session(name="wf", user_id="u1") as sess:
            kwargs = {"model": "gpt-4o"}
            _state.mint_obs_key()
            _enforcer._apply_reroute(REROUTE_DIRECTIVE, kwargs, "openai")

            key2 = _state.mint_obs_key()
            sess._call_outcome = {"status": "failed", "error_type": "APIError"}
            _enforcer._emit_call_failure_log(client, sess, obs_key=key2)

        self.assertEqual(len(client.calls), 1)
        self.assertNotIn("_tp_routing", client.calls[0]["metadata"])


if __name__ == "__main__":
    unittest.main()
