"""F-17-B — a rejected Anthropic ``.stream()`` entry must still land ONE row.

The anthropic SDK fires the provider HTTP request inside the vendor
``MessageStreamManager.__enter__`` (async: ``__aenter__``), and our wrapper
called that bare. Python does NOT call ``__exit__`` when ``__enter__`` raises,
so a request-time rejection (a 401 at connect, verified in real-apps run #17)
produced:

  * ZERO rows — ``_finalize`` never ran and the orphaned OpenLLMetry stream span
    never reached ``on_end``; and
  * leaked pre-flight state — ``session._suppress_anthropic_otel_stream`` stayed
    True (it would silently suppress the NEXT legitimate anthropic span), and
    the call's observations + ``_local_decision`` were never drained.

Fix under test: the enter statement (ONLY that statement) is guarded.
``TokenPoliceBlockedError`` re-raises untouched (defense-in-depth: a denial must
never be reclassified as a provider failure), and every other exception routes
through ``_emit_enter_failure`` — restore the suppress flag to its saved
pre-enter value, stash the attempt context (with the serving provider the
wrapper's OWN pre-flight resolved, so minimax-via-anthropic classifies like its
non-stream siblings), stash the prompt at a freshly reserved order, build the
call outcome, then ``_emit_call_failure_log`` — before the ORIGINAL exception is
re-raised BY IDENTITY.

All fakes — no anthropic SDK involved. ``_instrument_anthropic_stream()`` is
exercised end-to-end against fake Messages/AsyncMessages patched into
``sys.modules`` (same harness shape as
tests/test_f6_anthropic_stream_preflight_context.py), so the patched
``.stream()`` is itself under test.
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
from token_police.context import (
    _anthropic_stream_span_window,
    _current_session,
    TPSession,
)
from token_police.exceptions import TokenPoliceBlockedError
from token_police.telemetry import TokenPoliceSpanProcessor


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — asyncio.run
    unsets the main-thread event loop, breaking later get_event_loop()-based
    tests in the same pytest process."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _Auth401(Exception):
    """Anthropic-SDK-shaped 401: ``status_code`` is what ``classify_exception``
    reads, so the row's error_kind/http_status are really derived, not asserted
    into existence."""

    def __init__(self, msg="401 invalid x-api-key"):
        super().__init__(msg)
        self.status_code = 401


class _FailingSyncMgr:
    """Vendor manager whose ``__enter__`` (where the HTTP request fires) raises.
    ``exit_count`` proves the root cause: Python skips ``__exit__`` entirely, so
    the wrapper's ``_finalize`` can never run on this path."""

    def __init__(self, exc):
        self._exc = exc
        self.enter_count = 0
        self.exit_count = 0

    def __enter__(self):
        self.enter_count += 1
        raise self._exc

    def __exit__(self, *a):
        self.exit_count += 1
        return False


class _FailingAsyncMgr:
    def __init__(self, exc):
        self._exc = exc
        self.enter_count = 0
        self.exit_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        raise self._exc

    async def __aexit__(self, *a):
        self.exit_count += 1
        return False


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


class _SpyAsyncMgr:
    """Successful manager — used by the denial arm, which must never enter it."""

    def __init__(self):
        self.enter_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return _FakeAsyncStream([NS(type="content_block_delta", delta=NS(text="hi"))])

    async def __aexit__(self, *a):
        return False


_SUBSEQUENT_CALL_USAGE = NS(
    input_tokens=42, output_tokens=17,
    cache_read_input_tokens=0, cache_creation_input_tokens=0,
)


class _FakeSyncStreamWithUsage:
    """A stream whose `get_final_message()` carries REAL (non-zero) usage —
    the "subsequent call still meters correctly" proof needs a positive
    signal, not just "a row exists"."""

    def __iter__(self):
        return iter([])

    def get_final_message(self):
        return NS(model=MODEL, content=[], usage=_SUBSEQUENT_CALL_USAGE)


class _SuccessSyncMgr:
    """A fully successful vendor manager — models a healthy call made AFTER a
    prior call's enter failure, to prove no suppression state leaked onto it."""

    def __init__(self):
        self.enter_count = 0

    def __enter__(self):
        self.enter_count += 1
        return _FakeSyncStreamWithUsage()

    def __exit__(self, *a):
        return False


class _FakeAsyncStreamWithUsage:
    def __aiter__(self):
        async def _agen():
            return
            yield  # pragma: no cover - makes this an async generator
        return _agen()

    async def get_final_message(self):
        return NS(model=MODEL, content=[], usage=_SUBSEQUENT_CALL_USAGE)


class _SuccessAsyncMgr:
    """Async twin of ``_SuccessSyncMgr``."""

    def __init__(self):
        self.enter_count = 0

    async def __aenter__(self):
        self.enter_count += 1
        return _FakeAsyncStreamWithUsage()

    async def __aexit__(self, *a):
        return False


class _Factory:
    """Stands in for the ORIGINAL (unpatched) ``.stream()`` body: records the
    kwargs it was called with and hands back managers from ``mgrs``."""

    def __init__(self, mgr_factory):
        self._mgr_factory = mgr_factory
        self.calls = []
        self.mgrs = []

    def __call__(self, kwargs):
        self.calls.append(dict(kwargs))
        mgr = self._mgr_factory()
        self.mgrs.append(mgr)
        return mgr


def _install_fake_anthropic(sync_factory, async_factory, base_url=None):
    """Patch fake Messages/AsyncMessages into sys.modules, instrument them, and
    return (sync_client, async_client, cleanup). ``base_url`` is exposed the way
    the real SDK does — ``resource._client.base_url`` — so the wrapper's
    serving-provider resolution has something to read."""
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


def _model_block_rule(model):
    return {
        "id": "blk", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
        "selector": {"match": {"field": "model", "operator": "EQ", "value": model},
                     "group_by": []},
    }


MODEL = "claude-3-5-sonnet"
_KW = dict(model=MODEL, messages=[{"role": "user", "content": "hi"}])


class _EnterFailureBase(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        self.session = TPSession()
        self._token = _current_session.set(self.session)
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

    def _arm(self, directives=(), check_result=None, firewall="enforce"):
        """Install a daemon client with a warm pack and a stubbed /check."""
        client = TokenPolice(
            api_key="tp_sk_test123", base_url="http://127.0.0.1:59999",
            timeout=0.1, firewall=firewall, deployment="daemon",
        )
        tp_state.set_client(client)
        self.assertTrue(tp_state.apply_snapshot(_snapshot(list(directives))))
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

    # ── shared assertions ──────────────────────────────────────────
    def _failure_rows(self, client):
        return [c.kwargs for c in client.log_sync.call_args_list
                if (c.kwargs.get("call_outcome") or {}).get("status") == "failed"]

    def _assert_single_401_row(self, client, provider="anthropic", model=MODEL):
        self.assertEqual(len(client.log_sync.call_args_list), 1)
        rows = self._failure_rows(client)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["model"], model)
        self.assertEqual(row["provider"], provider)
        self.assertEqual(row["call_outcome"]["error_kind"], "auth_error")
        self.assertEqual(row["call_outcome"]["http_status"], 401)
        # Forensics: the failed row must still show WHAT was attempted.
        self.assertTrue(row.get("prompt_composition"))
        # The wire shape stays the module client's, exactly as a successful
        # sibling would have logged it.
        self.assertEqual((row.get("usage") or {}).get("shape"), "anthropic_messages")
        return row

    def _assert_no_leaked_state(self):
        s = self.session
        # NEW MECHANISM (the deleted `_suppress_anthropic_otel_stream` flag's
        # replacement): the per-call suppression window must be fully
        # disarmed after a failed enter — a still-armed window would eat the
        # NEXT legitimate anthropic span's telemetry. Asserting the old
        # flag's absence would now be VACUOUS (the attribute no longer
        # exists at all) — see TestSuppressFlagRestoredAfterEnterFailure
        # below for the falsifiable positive-control proof of this same
        # invariant.
        self.assertIsNone(_anthropic_stream_span_window.get())
        # Driving the real span processor with a fresh fake anthropic span
        # right now must NOT stamp tp.suppress — nothing is armed to claim it.
        span = _FakeSpan()
        TokenPoliceSpanProcessor().on_start(span)
        self.assertNotIn("tp.suppress", span.attributes)
        # Drained by the failure emitter. `_local_decision` (the old flat
        # slot) is never written by any code path any more — assert against
        # the keyed store (`_local_decisions`) instead.
        self.assertFalse(getattr(s, "_local_decisions", None))
        self.assertIsNone(getattr(s, "_call_outcome", None))
        # The composition stashed for the failed row was consumed, not leaked
        # into the next call.
        self.assertEqual(getattr(s, "_pending_compositions", {}) or {}, {})
        # Telemetry deferral belongs to the SUCCESS path (set after enter) —
        # an enter failure must leave it exactly as it was.
        self.assertIs(getattr(s, "_defer_telemetry", False), False)


class TestSyncStreamEnterFailure(_EnterFailureBase):

    def test_enter_failure_emits_one_row_and_reraises_by_identity(self):
        boom = _Auth401()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm()

        with self.assertRaises(_Auth401) as cm:
            with sc.stream(**_KW):
                pass
        # GOLDEN RULE: the provider's own exception object, not a wrapper.
        self.assertIs(cm.exception, boom)
        self.assertNotIsInstance(cm.exception, TokenPoliceBlockedError)

        self._assert_single_401_row(client)
        self._assert_no_leaked_state()

    def test_python_skips_exit_so_the_guard_is_the_only_emitter(self):
        """Root-cause pin: the vendor __exit__ (and therefore _finalize) never
        runs when __enter__ raises — pre-fix that meant ZERO rows."""
        boom = _Auth401()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm()

        with self.assertRaises(_Auth401):
            with sc.stream(**_KW):
                pass

        self.assertEqual(sync_f.mgrs[0].enter_count, 1)
        self.assertEqual(sync_f.mgrs[0].exit_count, 0)
        self.assertEqual(len(self._failure_rows(client)), 1)

    def test_serving_provider_is_threaded_onto_the_failure_row(self):
        """A minimax-via-anthropic client resolves serving=minimax in the
        wrapper's own pre-flight; the enter-failure row must carry THAT
        provider (with the anthropic wire shape), like its non-stream
        siblings — not a hardcoded 'anthropic'."""
        boom = _Auth401()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(
            sync_f, async_f, base_url="https://api.minimax.io/anthropic")
        self._teardowns.append(cleanup)
        client = self._arm()

        kw = dict(model="MiniMax-M2.5", messages=[{"role": "user", "content": "hi"}])
        with self.assertRaises(_Auth401):
            with sc.stream(**kw):
                pass

        self._assert_single_401_row(client, provider="minimax", model="MiniMax-M2.5")

    def test_non_401_failure_classifies_on_its_own_status(self):
        """Classification is real, not hardcoded to auth_error/401."""
        class _ServerError(Exception):
            def __init__(self):
                super().__init__("529 overloaded")
                self.status_code = 529

        boom = _ServerError()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm()

        with self.assertRaises(_ServerError) as cm:
            with sc.stream(**_KW):
                pass
        self.assertIs(cm.exception, boom)
        rows = self._failure_rows(client)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["call_outcome"]["http_status"], 529)
        self.assertEqual(rows[0]["call_outcome"]["error_kind"], "server_error")

    def test_telemetry_failure_never_reaches_the_app(self):
        """GOLDEN RULE: a throwing /log dispatch degrades to telemetry loss —
        the customer still receives the provider's error by identity."""
        boom = _Auth401()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm()
        client.log_sync.side_effect = RuntimeError("log exploded")

        with self.assertRaises(_Auth401) as cm:
            with sc.stream(**_KW):
                pass
        self.assertIs(cm.exception, boom)


class TestAsyncStreamEnterFailure(_EnterFailureBase):

    def test_aenter_failure_emits_one_row_and_reraises_by_identity(self):
        boom = _Auth401()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(lambda: _FailingAsyncMgr(boom))
        _sc, ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm()

        async def _run():
            with self.assertRaises(_Auth401) as cm:
                async with ac.stream(**_KW):
                    pass
            return cm.exception
        caught = _run_coro(_run())

        self.assertIs(caught, boom)
        self.assertNotIsInstance(caught, TokenPoliceBlockedError)
        self._assert_single_401_row(client)
        self._assert_no_leaked_state()
        # __aexit__ is skipped too — the guard is the only emitter here.
        self.assertEqual(async_f.mgrs[0].exit_count, 0)

    def test_serving_provider_is_threaded_on_the_async_path(self):
        boom = _Auth401()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(lambda: _FailingAsyncMgr(boom))
        _sc, ac, cleanup = _install_fake_anthropic(
            sync_f, async_f, base_url="https://api.minimax.io/anthropic")
        self._teardowns.append(cleanup)
        client = self._arm()

        kw = dict(model="MiniMax-M2.5", messages=[{"role": "user", "content": "hi"}])

        async def _run():
            with self.assertRaises(_Auth401):
                async with ac.stream(**kw):
                    pass
        _run_coro(_run())

        self._assert_single_401_row(client, provider="minimax", model="MiniMax-M2.5")

    def test_enforce_denial_still_raises_and_is_not_a_failure_row(self):
        """The pre-flight runs BEFORE the guarded enter. A denial must reach the
        customer as TokenPoliceBlockedError, the vendor manager must never be
        entered, and nothing may be logged as a provider failure."""
        sync_f = _Factory(lambda: _FailingSyncMgr(_Auth401()))
        async_f = _Factory(_SpyAsyncMgr)
        _sc, ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm([_model_block_rule(MODEL)],
                           check_result={"status": "blocked"})

        async def _run():
            with self.assertRaises(TokenPoliceBlockedError):
                async with ac.stream(**_KW):
                    pass
        _run_coro(_run())

        self.assertEqual(async_f.mgrs[0].enter_count, 0)  # no provider request
        self.assertEqual(self._failure_rows(client), [])
        # A denial never arms the suppression window either — the pre-flight
        # check raises BEFORE the W1 arm statement is even reached, so there
        # is nothing to disarm. (The deleted `_suppress_anthropic_otel_stream`
        # flag this replaced no longer exists — asserting its absence would
        # now be vacuous.)
        self.assertIsNone(_anthropic_stream_span_window.get())


class _Scope:
    def __init__(self, name):
        self.name = name


class _FakeSpan:
    """Writable attributes so on_start's set_attribute is observable."""

    def __init__(self, scope="opentelemetry.instrumentation.anthropic"):
        self.attributes = {}
        self.name = "anthropic.chat"
        self.instrumentation_scope = _Scope(scope)
        self.context = NS(trace_id=0x1111, span_id=0x2222)
        self.parent = None
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value


class TestSuppressFlagRestoredAfterEnterFailure(_EnterFailureBase):
    """The leak that mattered most: pre-fix the one-shot suppress flag survived
    a failed enter, so the NEXT legitimate anthropic span was silently dropped
    (a whole successful call lost from the dashboard).

    That flag is deleted; this class is rewritten against its replacement —
    the per-call ``_anthropic_stream_span_window`` ContextVar window (see
    token_police/context.py). Three of the original four assertions across
    this test module used to read
    ``getattr(session, "_suppress_anthropic_otel_stream", False) is False``,
    which passes VACUOUSLY now that the attribute no longer exists at all.
    ``test_control_a_set_flag_really_does_suppress`` was planted as a
    tripwire for exactly this — it is the one test that actually FAILS
    today, proving the other three were measuring nothing. It is replaced
    below by ``test_control_b_armed_window_really_does_suppress``, an
    equivalent tripwire for the new mechanism.
    """

    def _fail_one_stream(self):
        """Fail once, then return (sync_client, client, sync_factory) so
        callers can drive a SECOND call through the SAME instrumented client
        — the real behavioural proof that nothing leaked onto it."""
        boom = _Auth401()
        sync_f = _Factory(lambda: _FailingSyncMgr(boom))
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm()
        with self.assertRaises(_Auth401):
            with sc.stream(**_KW):
                pass
        return sc, client

    def test_next_anthropic_span_is_not_suppressed(self):
        """Negative assertion: the window left behind by a failed enter must
        not claim a fresh, unrelated anthropic span."""
        self._fail_one_stream()
        # NEW MECHANISM: no window survives a failed enter at all.
        self.assertIsNone(_anthropic_stream_span_window.get())
        span = _FakeSpan()
        TokenPoliceSpanProcessor().on_start(span)
        self.assertNotIn("tp.suppress", span.attributes)

    def test_control_b_armed_window_really_does_suppress(self):
        """Proves the assertion above can fail — i.e. it is measuring the
        window mechanism, not an unrelated no-op. Replaces
        ``test_control_a_set_flag_really_does_suppress`` (the deleted flag's
        tripwire) with the equivalent for the new per-call window."""
        handle = enforcer.arm_anthropic_stream_span_window()
        try:
            span = _FakeSpan()
            TokenPoliceSpanProcessor().on_start(span)
            self.assertTrue(span.attributes.get("tp.suppress"))
        finally:
            enforcer.disarm_anthropic_stream_span_window(handle)
        # Sanity: disarming actually clears it back out again.
        self.assertIsNone(_anthropic_stream_span_window.get())

    def test_subsequent_stream_call_still_meters_correctly(self):
        """The REAL behavioural proof, not just internal state: after one
        call's enter fails, the VERY NEXT anthropic stream call on the same
        client must still land a normal, fully-metered success row — not be
        silently eaten (the pre-fix bug: the leaked flag suppressed the next
        call's OTel span, and — outside a workflow — the manual row for a
        LATER call was never at risk, but inside one this exact scenario
        used to swallow it)."""
        boom = _Auth401()
        calls = {"n": 0}

        def _mgr_factory():
            calls["n"] += 1
            if calls["n"] == 1:
                return _FailingSyncMgr(boom)
            return _SuccessSyncMgr()

        sync_f = _Factory(_mgr_factory)
        async_f = _Factory(_SpyAsyncMgr)
        sc, _ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        self._teardowns.append(cleanup)
        client = self._arm()

        with self.assertRaises(_Auth401):
            with sc.stream(**_KW):
                pass

        with sc.stream(**_KW) as stream:
            list(stream)

        self.assertEqual(len(client.log_sync.call_args_list), 2)
        failure_call, success_call = client.log_sync.call_args_list
        self.assertEqual(
            (failure_call.kwargs.get("call_outcome") or {}).get("status"), "failed")
        # The success row is the manual `_finalize` emitter's shape — no
        # call_outcome key at all — carrying the REAL usage from the second
        # call's final message, not zeros and not absent.
        self.assertNotIn("call_outcome", success_call.kwargs)
        self.assertEqual(success_call.kwargs.get("input_tokens"), 42)
        self.assertEqual(success_call.kwargs.get("output_tokens"), 17)


if __name__ == "__main__":
    unittest.main()
