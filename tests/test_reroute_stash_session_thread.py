"""Reroute audit stash must land on the SAME session object the wrapper later
flushes — even with NO open ``tp.session()`` / ``tp.workflow()`` context.

Defect: outside an open context, ``get_current_session()`` mints a fresh
throwaway ``TPSession`` on EVERY call. ``_run_sync_check`` / ``_run_async_check``
used to resolve their own session internally and stash a REROUTE decision's
``_local_decision`` audit payload onto it; the wrapper then resolved a DIFFERENT
throwaway and handed that to ``_flush_deferred_spans``, whose
``getattr(session, "_local_decision", None)`` read always saw ``None``. Net: for
any call outside a context the reroute audit never reached the log row.

Fix: the check functions accept an optional ``session=`` the wrapper resolves
ONCE up front and threads through, so the stash and the flush share one object.

Mirrors the driving pattern in test_reroute_suppress_bedrock_embedding.py.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as tp_state
from token_police.context import TPSession


# Same-provider (openai) reroute so the cross-provider guard does NOT fire.
_OPENAI_REROUTE_SNAPSHOT = {
    "schema_version": 1, "type": "snapshot", "version": 1,
    "tenant_id": "t", "project_id": "p", "ttl_seconds": 600, "loop_blocks": [],
    "directives": [
        {
            "id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
            "selector": {"match": None, "group_by": []},
            "reroute": {"from": {}, "to": {"provider": "openai",
                                           "model": "gpt-4o-mini"}},
        }
    ],
}


def _init(check_result):
    """Init an enforce-mode client with check_sync/check/log_sync mocked."""
    client = tp.init(api_key="tp_sk_test_thread", firewall="enforce")
    client.check_sync = MagicMock(return_value=check_result)
    client.check = AsyncMock(return_value=check_result)
    client.log_sync = MagicMock()
    return client


def _seed_deferred_span(session):
    """Populate one deferred span the way telemetry.on_end would, so the
    subsequent _flush_deferred_spans actually emits a log row."""
    session._deferred_spans = [{
        "user_id": session.user_id,
        "span": {"trace_id": session.trace_id, "span_order": 0,
                 "span_name": "chat"},
    }]
    session._defer_telemetry = False


class TestRerouteStashSessionThread(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()

    def tearDown(self):
        tp_state.reset_pack()
        tp.uninstrument()

    # ── Core fix: no open context, threaded session receives the stash and the
    # flushed log payload carries the reroute audit. ────────────────────────
    def test_reroute_stash_lands_and_flushes_no_context_sync(self):
        client = _init({"status": "allowed"})
        tp_state.apply_snapshot(_OPENAI_REROUTE_SNAPSHOT)
        sess = TPSession(user_id="u1")
        kwargs = {"model": "gpt-4o"}

        # No `with tp.session(...)` — the wrapper's threaded session is `sess`.
        _enforcer._run_sync_check(kwargs=kwargs, provider="openai", session=sess)

        # Reroute applied AND stashed on the SAME object we passed in — via the
        # keyed store (`session._local_decisions`), not the old flat slot.
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertTrue(getattr(sess, "_local_decisions", None))
        ld = sess._local_decisions[-1]["ld"]
        self.assertEqual(ld["outcome"], "rerouted")
        self.assertEqual(ld["rule_id"], "rr")
        self.assertIn("reroute", ld)

        # The flush carries the reroute audit into the log row.
        _seed_deferred_span(sess)
        _enforcer._flush_deferred_spans(sess)
        payload = client.log_sync.call_args.kwargs
        self.assertEqual(payload["local_decision"]["outcome"], "rerouted")
        self.assertEqual(payload["local_decision"]["rule_id"], "rr")

    def test_reroute_stash_lands_and_flushes_no_context_async(self):
        client = _init({"status": "allowed"})
        tp_state.apply_snapshot(_OPENAI_REROUTE_SNAPSHOT)
        sess = TPSession(user_id="u1")
        kwargs = {"model": "gpt-4o"}
        results = {}

        # The keyed store requires the stash and its later claim to share the
        # same obs-key context. Every real call site awaits the check and
        # later flushes within the SAME coroutine/task (see e.g. :2171/:2313
        # in enforcer.py) — `asyncio.Task` copies context at creation, so a
        # key minted inside one task is invisible once that task has
        # returned. Mirror that shape here instead of splitting the check and
        # the flush across two separate `asyncio.run()` calls.
        async def _run():
            await _enforcer._run_async_check(kwargs=kwargs, provider="openai",
                                             session=sess)
            self.assertEqual(kwargs["model"], "gpt-4o-mini")
            self.assertEqual(sess._local_decisions[-1]["ld"]["outcome"], "rerouted")
            _seed_deferred_span(sess)
            _enforcer._flush_deferred_spans(sess)
            results["payload"] = client.log_sync.call_args.kwargs

        asyncio.run(_run())

        self.assertEqual(results["payload"]["local_decision"]["outcome"], "rerouted")

    # ── Negative control (bug reproduction): with NO session threaded, the
    # stash orphans on the check's own throwaway, so a separately-resolved
    # wrapper session (and its flush) never sees it. This asserts the exact
    # pre-fix breakage the `session=` param closes. ─────────────────────────
    def test_reroute_stash_orphaned_when_not_threaded(self):
        client = _init({"status": "allowed"})
        tp_state.apply_snapshot(_OPENAI_REROUTE_SNAPSHOT)
        # The object a wrapper would resolve+flush; distinct from whatever the
        # unthreaded check mints internally.
        wrapper_session = TPSession(user_id="u1")
        kwargs = {"model": "gpt-4o"}

        _enforcer._run_sync_check(kwargs=kwargs, provider="openai")  # no session=

        # The reroute still fired (kwargs mutated) ...
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        # ... but the wrapper's own session never received the audit, so its
        # flush emits a row with no local_decision (the orphaned-stash bug).
        # NOTE: `_local_decision` (the old flat slot) is never written by
        # ANY code path any more, so asserting it's None would be vacuously
        # true regardless of whether the orphaning bug still holds — assert
        # against the keyed store instead, which is the one actually written.
        self.assertFalse(getattr(wrapper_session, "_local_decisions", None))
        _seed_deferred_span(wrapper_session)
        _enforcer._flush_deferred_spans(wrapper_session)
        payload = client.log_sync.call_args.kwargs
        self.assertNotIn("local_decision", payload)

    # ── Backward compatibility: inside an open workflow the audit still lands
    # and flushes (the context session is authoritative). ──────────────────
    def test_reroute_stash_inside_workflow_still_flushes(self):
        client = _init({"status": "allowed"})
        tp_state.apply_snapshot(_OPENAI_REROUTE_SNAPSHOT)

        with tp.session(name="wf", user_id="u1") as sess:
            kwargs = {"model": "gpt-4o"}
            _enforcer._run_sync_check(kwargs=kwargs, provider="openai",
                                      session=sess)
            self.assertEqual(kwargs["model"], "gpt-4o-mini")
            self.assertEqual(sess._local_decisions[-1]["ld"]["outcome"], "rerouted")
            _seed_deferred_span(sess)
            _enforcer._flush_deferred_spans(sess)
            # Capture the flush's log row now — leaving the `with` fires the
            # structural anchor span's own (local_decision-free) log_sync.
            payload = client.log_sync.call_args.kwargs

        self.assertEqual(payload["local_decision"]["outcome"], "rerouted")

    # ── Golden rule: an allowed→reroute decision NEVER raises into caller. ────
    def test_reroute_never_raises(self):
        _init({"status": "allowed"})
        tp_state.apply_snapshot(_OPENAI_REROUTE_SNAPSHOT)
        sess = TPSession(user_id="u1")
        # Must not raise.
        _enforcer._run_sync_check(kwargs={"model": "gpt-4o"}, provider="openai",
                                  session=sess)


if __name__ == "__main__":
    unittest.main()
