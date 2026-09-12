"""The Anthropic ``.stream()`` pre-flight must carry real call context.

Both stream pre-flights used to run BARE — ``_run_sync_check()`` /
``await _run_async_check()`` with no kwargs and no provider. The local evaluator
therefore built its context with ``model=""`` and ``provider=""``, so on EVERY
streamed Anthropic call:

  * any provider-targeted REROUTE rule looked cross-provider and produced a
    blank-field ``reroute_rejected`` observation, even against a genuinely
    same-provider Anthropic client;
  * a same-provider reroute could never be applied; and
  * a model-conditioned BLOCK rule could never match.

The sync check runs BEFORE ``orig_stream`` builds the manager, so an applied
reroute's in-place ``kwargs["model"]`` swap reaches the wire unaided. The async
check deliberately runs in ``__aenter__`` (never block the caller's loop), but
the anthropic SDK freezes the request body when ``stream()`` constructs the
manager — so an applied reroute additionally rebuilds the manager from the
mutated kwargs before entry. A failed rebuild falls back to the original manager
AND drops the applied-reroute claim, so a request that goes out on the original
model is never audited as rerouted.

All fakes — no anthropic SDK involved. ``_instrument_anthropic_stream()`` is
exercised end-to-end against fake Messages/AsyncMessages patched into
``sys.modules``, so the patched ``.stream()`` is itself under test.
"""
import asyncio
import sys
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police import state as tp_state
from token_police.client import TokenPolice
from token_police.context import _current_session, TPSession
from token_police.exceptions import TokenPoliceBlockedError


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — asyncio.run
    unsets the main-thread event loop, breaking later get_event_loop()-based
    tests in the same pytest process."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _events():
    return [NS(type="content_block_delta", delta=NS(text="hi"))]


class _FakeSyncStream:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return NS(model="claude-3-5-sonnet", content=[], usage=None)

    @property
    def text_stream(self):
        for ev in self._events:
            t = getattr(getattr(ev, "delta", None), "text", None)
            if t:
                yield t


class _FakeAsyncStream:
    def __init__(self, events):
        self._events = events

    def __aiter__(self):
        async def _agen():
            for ev in self._events:
                yield ev
        return _agen()

    async def get_final_message(self):
        return NS(model="claude-3-5-sonnet", content=[], usage=None)


class _SpySyncMgr:
    def __init__(self):
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return _FakeSyncStream(_events())

    def __exit__(self, *a):
        return False


class _SpyAsyncMgr:
    def __init__(self):
        self.enter_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return _FakeAsyncStream(_events())

    async def __aexit__(self, *a):
        return False


class _Factory:
    """Stands in for the ORIGINAL (unpatched) ``.stream()`` body. Records a
    kwargs SNAPSHOT at call time — the whole point of the async rebuild test is
    that the second construction sees the reroute-mutated model. ``raise_after``
    makes construction N+1 blow up so the rebuild-failure branch is drivable."""

    def __init__(self, mgr_cls, raise_after=None):
        self._mgr_cls = mgr_cls
        self.calls = []
        self.mgrs = []
        self.raise_after = raise_after

    def __call__(self, kwargs):
        self.calls.append(dict(kwargs))
        if self.raise_after is not None and len(self.calls) > self.raise_after:
            raise RuntimeError("rebuild boom")
        mgr = self._mgr_cls()
        self.mgrs.append(mgr)
        return mgr


def _install_fake_anthropic(sync_factory, async_factory, base_url=None):
    """Patch fake Messages/AsyncMessages into sys.modules, instrument them, and
    return (sync_client, async_client, cleanup). ``base_url`` (when given) is
    exposed the way the real SDK does — ``resource._client.base_url`` — so the
    serving-provider resolution under test has something to read."""
    client_stub = NS(base_url=base_url) if base_url else None

    class Messages:
        def stream(self, *a, **k):
            return self._factory(k)

    class AsyncMessages:
        def stream(self, *a, **k):
            return self._factory(k)

    messages_mod = types.ModuleType("anthropic.resources.messages")
    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages
    resources_mod = types.ModuleType("anthropic.resources")
    resources_mod.messages = messages_mod
    anthropic_mod = types.ModuleType("anthropic")
    anthropic_mod.resources = resources_mod

    names = ("anthropic", "anthropic.resources", "anthropic.resources.messages")
    saved = {n: sys.modules.get(n) for n in names}
    sys.modules["anthropic"] = anthropic_mod
    sys.modules["anthropic.resources"] = resources_mod
    sys.modules["anthropic.resources.messages"] = messages_mod

    enforcer._instrument_anthropic_stream()

    sync_client = Messages()
    sync_client._factory = sync_factory
    sync_client._client = client_stub
    async_client = AsyncMessages()
    async_client._factory = async_factory
    async_client._client = client_stub

    def cleanup():
        enforcer._originals.pop((Messages, "stream"), None)
        enforcer._originals.pop((AsyncMessages, "stream"), None)
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v

    return sync_client, async_client, cleanup


def _snapshot(directives):
    return {
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "t", "project_id": "p", "ttl_seconds": 600,
        "loop_blocks": [], "directives": directives,
    }


def _reroute_rule(target_provider, target_model):
    return {
        "id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
        "selector": {"match": None, "group_by": []},
        "reroute": {"from": {}, "to": {"provider": target_provider, "model": target_model}},
    }


def _model_block_rule(model):
    return {
        "id": "blk", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
        "selector": {"match": {"field": "model", "operator": "EQ", "value": model},
                     "group_by": []},
    }


MODEL = "claude-3-5-sonnet"
TARGET = "claude-haiku-4-5"
_KW = dict(model=MODEL, messages=[{"role": "user", "content": "hi"}])


class _StreamPreflightBase(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        self._session = TPSession()
        self._token = _current_session.set(self._session)
        self._teardowns = []

    def tearDown(self):
        for fn in reversed(self._teardowns):
            try:
                fn()
            except Exception:
                pass
        _current_session.reset(self._token)
        tp_state.reset_pack()
        tp_state.drain_observations()
        try:
            tp_state.set_client(None)
        except Exception:
            pass

    def _arm(self, directives, check_result=None, firewall="enforce"):
        """Install a daemon client with a warm pack and a stubbed /check."""
        client = TokenPolice(
            api_key="tp_sk_test123", base_url="http://127.0.0.1:59999",
            timeout=0.1, firewall=firewall, deployment="daemon",
        )
        tp_state.set_client(client)
        self.assertTrue(tp_state.apply_snapshot(_snapshot(directives)))
        result = check_result if check_result is not None else {"status": "allowed"}

        async def _acheck(**_kw):
            return result

        patches = [
            mock.patch.object(client, "check_sync", return_value=result),
            mock.patch.object(client, "check", new=_acheck),
            mock.patch.object(client, "log_sync"),
        ]
        for p in patches:
            p.start()
            self._teardowns.append(p.stop)
        return client


class TestSyncStreamPreflightContext(_StreamPreflightBase):

    def test_same_provider_reroute_applies_before_stream_is_built(self):
        """The sync check runs pre-construction, so the swapped model is what
        the original ``stream()`` is invoked with."""
        sync_f = _Factory(_SpySyncMgr)
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        self._arm([_reroute_rule("anthropic", TARGET)])

        with sc.stream(**_KW) as stream:
            list(stream)

        self.assertEqual(len(sync_f.calls), 1)
        self.assertEqual(sync_f.calls[0]["model"], TARGET)
        routing = (self._session.metadata or {}).get("_tp_routing")
        self.assertIsNotNone(routing)
        self.assertEqual(routing["original_model"], MODEL)
        self.assertEqual(routing["actual_model"], TARGET)

    def test_cross_provider_rejection_carries_populated_from_fields(self):
        """Serving minimax + an anthropic-targeted rule is a genuine
        cross-provider rejection — but ``from`` must name the real call."""
        sync_f = _Factory(_SpySyncMgr)
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(
            sync_f, async_f, base_url="https://api.minimax.io/anthropic")
        self._teardowns.append(cleanup)
        client = self._arm([_reroute_rule("anthropic", TARGET)])

        kw = dict(model="MiniMax-M2.5", messages=[{"role": "user", "content": "hi"}])
        with sc.stream(**kw) as stream:
            list(stream)

        # The finalizer now drains the queue INTO its /log call, so the
        # rejection is asserted where the audit row actually reads it from
        # (previously it was still stranded in the global queue — the bug).
        logged = [c.kwargs.get("observations") or []
                  for c in client.log_sync.call_args_list]
        rejected = [o for obs in logged for o in obs
                    if o.get("outcome") == "reroute_rejected"]
        self.assertEqual(tp_state.drain_observations(), [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["reroute"]["from"]["model"], "MiniMax-M2.5")
        self.assertEqual(rejected[0]["reroute"]["from"]["provider"], "minimax")
        # The rejection must not have touched the request body.
        self.assertEqual(sync_f.calls[0]["model"], "MiniMax-M2.5")

    def test_model_conditioned_block_matches_and_never_builds_the_stream(self):
        sync_f = _Factory(_SpySyncMgr)
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        self._arm([_model_block_rule(MODEL)], check_result={"status": "blocked"})

        with self.assertRaises(TokenPoliceBlockedError):
            sc.stream(**_KW)
        # Pre-flight precedes construction — the provider stream never exists.
        self.assertEqual(len(sync_f.calls), 0)


class TestAsyncStreamPreflightContext(_StreamPreflightBase):

    def test_applied_reroute_rebuilds_the_manager_with_the_target_model(self):
        """The first manager is frozen on the pre-swap body and is never
        entered; the rebuilt one carries the target model."""
        sync_f = _Factory(_SpySyncMgr)
        async_f = _Factory(_SpyAsyncMgr)
        _sc, ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        self._arm([_reroute_rule("anthropic", TARGET)])

        async def _run():
            async with ac.stream(**_KW) as stream:
                async for _ev in stream:
                    pass
        _run_coro(_run())

        self.assertEqual(len(async_f.calls), 2)
        self.assertEqual(async_f.calls[0]["model"], MODEL)
        self.assertEqual(async_f.calls[1]["model"], TARGET)
        self.assertEqual(async_f.mgrs[0].enter_count, 0)
        self.assertEqual(async_f.mgrs[1].enter_count, 1)

    def test_no_reroute_builds_the_manager_exactly_once(self):
        sync_f = _Factory(_SpySyncMgr)
        async_f = _Factory(_SpyAsyncMgr)
        _sc, ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        self._arm([])

        async def _run():
            async with ac.stream(**_KW) as stream:
                async for _ev in stream:
                    pass
        _run_coro(_run())

        self.assertEqual(len(async_f.calls), 1)
        self.assertEqual(async_f.mgrs[0].enter_count, 1)

    def test_rebuild_failure_uses_the_original_manager_and_drops_the_claim(self):
        """Honesty guard: the wire carries the ORIGINAL model, so nothing may
        be left claiming the call was rerouted.

        NOTE: `_local_decision` (the old flat slot) is never written by any
        code path any more, so asserting it's None is vacuously true — and
        even checking the keyed store (`session._local_decisions`) after the
        run would ALSO be uninformative here: the finalize's own claim
        consumes whatever is left regardless of whether the drop ran. The
        actual honesty property the drop protects is what reaches /log —
        assert against the captured `local_decision` kwarg instead.
        """
        sync_f = _Factory(_SpySyncMgr)
        async_f = _Factory(_SpyAsyncMgr, raise_after=1)
        _sc, ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm([_reroute_rule("anthropic", TARGET)])

        async def _run():
            async with ac.stream(**_KW) as stream:
                async for _ev in stream:
                    pass
        _run_coro(_run())

        self.assertEqual(len(async_f.calls), 2)   # rebuild attempted
        self.assertEqual(len(async_f.mgrs), 1)    # and failed
        self.assertEqual(async_f.mgrs[0].enter_count, 1)  # original still used
        self.assertNotIn("_tp_routing", self._session.metadata or {})
        # No log row may carry a "rerouted" audit — the wire never swapped.
        logged_lds = [c.kwargs.get("local_decision")
                      for c in client.log_sync.call_args_list]
        self.assertTrue(all(
            not ld or ld.get("outcome") != "rerouted" for ld in logged_lds))
        self.assertFalse(getattr(self._session, "_local_decisions", None))

    def test_model_conditioned_block_matches_and_never_enters_the_stream(self):
        sync_f = _Factory(_SpySyncMgr)
        async_f = _Factory(_SpyAsyncMgr)
        _sc, ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        self._arm([_model_block_rule(MODEL)], check_result={"status": "blocked"})

        async def _run():
            with self.assertRaises(TokenPoliceBlockedError):
                async with ac.stream(**_KW):
                    pass
        _run_coro(_run())

        self.assertEqual(async_f.mgrs[0].enter_count, 0)


if __name__ == "__main__":
    unittest.main()
