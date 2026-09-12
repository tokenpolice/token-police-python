"""Per-call observation attribution (state.py keyed queue).

The observations queue tags every push with the pushing call's obs key
(``_obs_call_key`` ContextVar, minted by ``_run_sync_check`` /
``_run_async_check``) and a /log drain claims only its own call's entries +
untagged entries + stale orphans. These tests pin the state-layer contract:

- keyed isolation (a drain with key Y never steals key X's fresh entries)
- staleness fallback (an orphan past OBS_STALE_SECONDS ships on any drain)
- untagged fallback (no-key pushes ship on the next keyed drain)
- legacy no-arg drain-all (shutdown paths / older tests)
- concurrency: two asyncio tasks each drain exactly their own observation —
  the cross-trace-theft incident shape (tasks copy context at creation, so
  each task's minted key is invisible to the other).

``TestOverlappingWrappedCallsAttribution`` additionally drives the REAL
wrapped path end-to-end (fake anthropic ``Messages`` patched into
``sys.modules`` + ``_instrument_anthropic_stream()``, warm daemon pack, real
``_run_sync_check`` mint → real ``_AnthropicStreamMgrWrapper`` keyed
``_finalize`` → ``log_sync``) with two OVERLAPPING calls in the theft
ordering, asserting each /log payload carries exactly its own observation.
Harness mirrors tests/test_f6_anthropic_stream_preflight_context.py.

``TestSpanObsKeySurvivesReadableSpanSwap`` drives the REAL OTel span lifecycle
(real ``TracerProvider`` + real ``TokenPoliceSpanProcessor``, ended via
``span.end()``) to pin the span-id side map against the ReadableSpan identity
swap that made the original instance-attribute stamping dead code.
"""
import asyncio
import sys
import time
import types
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from opentelemetry.sdk.trace import TracerProvider

from token_police import enforcer
from token_police import state
from token_police import state as tp_state
from token_police import telemetry
from token_police.client import TokenPolice
from token_police.context import _current_session, TPSession
from token_police.telemetry import TokenPoliceSpanProcessor


OBS_A = {"rule_id": "rA", "outcome": "would_block", "mode": "dry_run"}
OBS_B = {"rule_id": "rB", "outcome": "would_block", "mode": "dry_run"}
UNTAGGED = {"tag": "untagged"}


class TestObservationAttribution(unittest.TestCase):

    def setUp(self):
        # Clears the queue AND the current-context obs key.
        state._observations_set([])
        self.addCleanup(state.drain_observations)

    # ── keyed isolation + staleness ──────────────────────────────────
    def test_keyed_isolation_then_staleness_ships(self):
        # Fresh entry tagged X → a Y-keyed drain must leave it queued.
        state._observations_set_entries(
            [{"obs": OBS_A, "key": "X", "ts": time.monotonic()}])
        self.assertEqual(state.drain_observations("Y"), [])

        # Age X past the stale window (inject ts via the test seam) → the
        # same Y drain now ships it: an orphan whose owning /log never fired
        # must eventually leave the process (loss is worse than late).
        state._observations_set_entries([{
            "obs": OBS_A, "key": "X",
            "ts": time.monotonic() - state.OBS_STALE_SECONDS - 5.0,
        }])
        self.assertEqual(state.drain_observations("Y"), [OBS_A])
        self.assertEqual(state.drain_observations(), [])

    def test_keyed_drain_claims_own_plus_untagged_preserving_rest(self):
        now = time.monotonic()
        state._observations_set_entries([
            {"obs": OBS_A, "key": "X", "ts": now},
            {"obs": UNTAGGED, "key": None, "ts": now},
            {"obs": OBS_B, "key": "Y", "ts": now},
        ])
        self.assertEqual(state.drain_observations("X"), [OBS_A, UNTAGGED])
        # Y's entry survived for its own drain.
        self.assertEqual(state.drain_observations("Y"), [OBS_B])

    def test_explicit_none_claims_only_untagged_and_stale(self):
        now = time.monotonic()
        state._observations_set_entries([
            {"obs": OBS_A, "key": "X", "ts": now},
            {"obs": UNTAGGED, "key": None, "ts": now},
            {"obs": OBS_B, "key": "Y",
             "ts": now - state.OBS_STALE_SECONDS - 5.0},
        ])
        self.assertEqual(state.drain_observations(None), [UNTAGGED, OBS_B])
        # X's fresh tagged entry is still queued.
        self.assertEqual(state.drain_observations("X"), [OBS_A])

    # ── untagged fallback ────────────────────────────────────────────
    def test_untagged_push_ships_on_next_keyed_drain(self):
        state.push_observation(dict(OBS_A))  # no key current → untagged
        self.assertEqual(state.drain_observations("some-other-call"), [OBS_A])

    # ── legacy no-arg drain-all ──────────────────────────────────────
    def test_no_arg_drain_returns_everything(self):
        now = time.monotonic()
        state._observations_set_entries([
            {"obs": OBS_A, "key": "X", "ts": now},
            {"obs": OBS_B, "key": "Y", "ts": now},
            {"obs": UNTAGGED, "key": None, "ts": now},
        ])
        self.assertEqual(state.drain_observations(), [OBS_A, OBS_B, UNTAGGED])
        self.assertEqual(state.drain_observations(), [])

    # ── contextvar key mechanics ─────────────────────────────────────
    def test_push_tags_with_minted_key_and_own_drain_claims(self):
        key = state.mint_obs_key()
        self.assertTrue(key)
        state.push_observation(dict(OBS_A))
        # Another call's drain must not steal it…
        self.assertEqual(state.drain_observations("someone-else"), [])
        # …but this call's own drain does.
        self.assertEqual(state.drain_observations(key), [OBS_A])

    def test_concurrent_tasks_each_drain_only_their_own(self):
        # The incident shape: call A pushes its observation, call B's /log
        # fires first — B must NOT walk away with A's entry. asyncio tasks
        # copy context at creation, so each task's mint is task-local.
        results = {}

        async def call(name, obs, delay):
            key = state.mint_obs_key()          # per-call check mint
            state.push_observation(dict(obs))   # pre-flight push, tagged
            await asyncio.sleep(delay)          # provider call in flight
            results[name] = state.drain_observations(key)

        async def main():
            await asyncio.gather(
                asyncio.create_task(call("A", OBS_A, 0.05)),
                asyncio.create_task(call("B", OBS_B, 0.0)),
            )

        asyncio.run(main())
        self.assertEqual(results["B"], [OBS_B])  # B logged first — own entry only
        self.assertEqual(results["A"], [OBS_A])  # A's entry waited for A's drain
        self.assertEqual(state.drain_observations(), [])  # nothing stranded


# ═══════════════════════════════════════════════════════════════════════
# Integration: two OVERLAPPING calls through the REAL wrapped path.
# Harness mirrors tests/test_f6_anthropic_stream_preflight_context.py —
# fake Messages patched into sys.modules, instrumented by the real
# _instrument_anthropic_stream(), armed daemon client with a warm pack.
# ═══════════════════════════════════════════════════════════════════════

def _events():
    return [NS(type="content_block_delta", delta=NS(text="hi"))]


class _FakeSyncStream:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return NS(model="claude-3-5-sonnet", content=[], usage=None)


class _SpySyncMgr:
    def __enter__(self):
        return _FakeSyncStream(_events())

    def __exit__(self, *a):
        return False


class _Factory:
    """Stands in for the ORIGINAL (unpatched) ``.stream()`` body."""

    def __init__(self, mgr_cls):
        self._mgr_cls = mgr_cls
        self.calls = []

    def __call__(self, kwargs):
        self.calls.append(dict(kwargs))
        return self._mgr_cls()


def _install_fake_anthropic(sync_factory):
    """Patch a fake ``Messages`` into sys.modules, instrument it, and return
    (sync_client, cleanup) — the sync half of the f6 harness."""

    class Messages:
        def stream(self, *a, **k):
            return self._factory(k)

    class AsyncMessages:
        def stream(self, *a, **k):  # pragma: no cover — unused here
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
    sync_client._client = None

    def cleanup():
        enforcer._originals.pop((Messages, "stream"), None)
        enforcer._originals.pop((AsyncMessages, "stream"), None)
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v

    return sync_client, cleanup


def _snapshot(directives):
    return {
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "t", "project_id": "p", "ttl_seconds": 600,
        "loop_blocks": [], "directives": directives,
    }


def _cross_provider_reroute_rule():
    # Targets provider "openai" while the calls run on anthropic → EVERY
    # call's local eval emits one `reroute_rejected` observation whose
    # reroute.from.model names the call — a distinct, per-call marker.
    return {
        "id": "rr", "kind": "REROUTE", "mode": "enforce", "priority": 10,
        "selector": {"match": None, "group_by": []},
        "reroute": {"from": {}, "to": {"provider": "openai", "model": "gpt-4o-mini"}},
    }


class TestOverlappingWrappedCallsAttribution(unittest.TestCase):

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

    def _arm(self, directives):
        """Install a daemon client with a warm pack and a stubbed /check."""
        client = TokenPolice(
            api_key="tp_sk_test123", base_url="http://127.0.0.1:59999",
            timeout=0.1, firewall="enforce", deployment="daemon",
        )
        tp_state.set_client(client)
        self.assertTrue(tp_state.apply_snapshot(_snapshot(directives)))
        result = {"status": "allowed"}

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

    def test_overlapping_stream_calls_each_log_only_their_own_observation(self):
        """The incident shape, on the real wrapped path: call A's pre-flight
        queues its observation first, but call B's /log fires FIRST (theft
        ordering). Before per-call keys, B's drain-all walked away with BOTH
        observations (B's row got 2, A's got 0). Now each payload must carry
        exactly its own call's `reroute_rejected` — matched by
        reroute.from.model — and nothing may be stranded afterwards."""
        sync_f = _Factory(_SpySyncMgr)
        sc, cleanup = _install_fake_anthropic(sync_f)
        self._teardowns.append(cleanup)
        client = self._arm([_cross_provider_reroute_rule()])

        model_a = "claude-sonnet-call-A"
        model_b = "claude-sonnet-call-B"

        # Call A: check runs (mints A's key, pushes A's rejection) and the
        # stream is entered + consumed — but NOT exited, so its /log has not
        # fired yet.
        w1 = sc.stream(model=model_a,
                       messages=[{"role": "user", "content": "hi"}])
        s1 = w1.__enter__()
        list(s1)

        # Call B overlaps: full lifecycle while A is still open — B's check
        # mints a fresh key (overwriting the contextvar) and B's /log fires
        # BEFORE A's.
        with sc.stream(model=model_b,
                       messages=[{"role": "user", "content": "hi"}]) as s2:
            list(s2)

        # Now A finishes; its /log fires second.
        w1.__exit__(None, None, None)

        payloads = [c.kwargs for c in client.log_sync.call_args_list]
        self.assertEqual(len(payloads), 2)

        def rejected_from_models(payload):
            obs = payload.get("observations") or []
            return [o["reroute"]["from"]["model"] for o in obs
                    if o.get("outcome") == "reroute_rejected"]

        # First row is B's (logged first) — exactly B's own observation.
        self.assertEqual(rejected_from_models(payloads[0]), [model_b])
        # Second row is A's — its observation waited for A's own drain.
        self.assertEqual(rejected_from_models(payloads[1]), [model_a])
        # No other observations rode along on either row.
        self.assertEqual(len(payloads[0].get("observations") or []), 1)
        self.assertEqual(len(payloads[1].get("observations") or []), 1)
        # Nothing stranded in the queue after both calls logged.
        self.assertEqual(tp_state.drain_observations(), [])


# ═══════════════════════════════════════════════════════════════════════
# Regression: the span-id SIDE MAP must survive OTel's ReadableSpan swap.
#
# The original implementation stamped `span._tp_obs_key` in on_start and read
# it back in on_end — dead code, because OTel Python's `Span.end()` builds a
# BRAND-NEW ReadableSpan (`self._readable_span()`) and hands THAT to on_end;
# the on_start object never reaches it. The attribute therefore always read
# None, every immediate-path drain ran as an unknown claimant, and each call's
# own tagged observations were stranded until the staleness window.
#
# These tests are only meaningful through the REAL OTel span lifecycle: a real
# TracerProvider + the real TokenPoliceSpanProcessor, ended via `span.end()`
# so on_end receives the genuine fresh ReadableSpan. Hand-passing the on_start
# object to on_end would make the buggy version pass.
# ═══════════════════════════════════════════════════════════════════════

class _SpyProcessor(TokenPoliceSpanProcessor):
    """Real processor + non-destructive observation of the side map. Records
    both span objects so the test can assert the identity swap, and PEEKS at
    (never pops) the map at on_end entry so the real drain still sees its key."""

    def __init__(self):
        super().__init__()
        self.start_objs = []
        self.end_objs = []
        self.keys_at_end = []

    def on_start(self, span, parent_context=None):
        self.start_objs.append(span)
        super().on_start(span, parent_context)

    def on_end(self, span):
        self.end_objs.append(span)
        with telemetry._span_obs_keys_lock:
            self.keys_at_end.append(
                telemetry._span_obs_keys.get(span.context.span_id))
        super().on_end(span)


class TestSpanObsKeySurvivesReadableSpanSwap(unittest.TestCase):

    def setUp(self):
        state._observations_set([])
        self.addCleanup(state.drain_observations)
        with telemetry._span_obs_keys_lock:
            telemetry._span_obs_keys.clear()
        self.addCleanup(telemetry._span_obs_keys.clear)
        self.proc = _SpyProcessor()
        # shutdown_on_exit=False: no atexit hook against a throwaway provider.
        self.provider = TracerProvider(shutdown_on_exit=False)
        self.provider.add_span_processor(self.proc)
        self.tracer = self.provider.get_tracer("tp.test")

    def _end_llm_span(self, model="claude-3-5-sonnet-20241022"):
        """Start + end a real LLM span inside the current obs-key context,
        capturing the payload the immediate path hands to client.log_sync."""
        captured = {}

        class _FakeClient:
            def log_sync(self, **kwargs):
                captured.update(kwargs)

        span = self.tracer.start_span("anthropic.chat", attributes={
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
        })
        with mock.patch("token_police.state.get_client",
                        return_value=_FakeClient()):
            span.end()   # → on_end receives a FRESH ReadableSpan, not `span`
        return span, captured

    def test_real_span_end_drains_this_calls_observations_by_key(self):
        key = state.mint_obs_key()          # the owning call's check mint
        state.push_observation(dict(OBS_A))  # pre-flight push, tagged with key
        started, payload = self._end_llm_span()

        # The immediate path claimed exactly this call's observation. Pre-fix
        # the popped key was always None → unknown claimant → no observations.
        self.assertEqual(payload.get("observations"), [OBS_A])
        # …and the queue is empty: nothing stranded for the staleness fallback.
        self.assertEqual(state.drain_observations(), [])

        # The map resolved the minted key from the FRESH ReadableSpan.
        self.assertEqual(self.proc.keys_at_end, [key])
        # Precondition that makes the side map necessary: on_end's span is a
        # DIFFERENT object with the SAME span id (OTel SDK 1.39.1). If this
        # ever flips, instance stamping would work again — and this assertion
        # is the signal to re-evaluate, not a silent behaviour change.
        ended = self.proc.end_objs[0]
        self.assertIsNot(started, ended)
        self.assertEqual(started.context.span_id, ended.context.span_id)
        # Entry retired at on_end → the bounded map cannot leak per call.
        self.assertNotIn(ended.context.span_id, telemetry._span_obs_keys)

    def test_keyed_not_drain_all_another_calls_entry_is_left_queued(self):
        key = state.mint_obs_key()
        now = time.monotonic()
        # This call's entry PLUS a concurrent call's fresh entry in the same
        # queue: proves the drain is keyed, not a drain-all that looks correct
        # only because a single entry was queued.
        state._observations_set_entries([
            {"obs": OBS_A, "key": key, "ts": now},
            {"obs": OBS_B, "key": "other-call", "ts": now},
        ])
        _started, payload = self._end_llm_span()

        self.assertEqual(payload.get("observations"), [OBS_A])
        self.assertEqual(self.proc.keys_at_end, [key])
        # B survived for its own /log — no theft across calls.
        self.assertEqual(state.drain_observations("other-call"), [OBS_B])

    def test_unrecorded_span_degrades_to_unknown_claimant(self):
        # No key current at on_start → nothing recorded → the drain claims
        # untagged + stale only (fail-open), and a keyed entry is untouched.
        state._observations_set_entries(
            [{"obs": OBS_A, "key": "X", "ts": time.monotonic()},
             {"obs": UNTAGGED, "key": None, "ts": time.monotonic()}])
        _started, payload = self._end_llm_span()
        self.assertEqual(payload.get("observations"), [UNTAGGED])
        self.assertEqual(self.proc.keys_at_end, [None])
        self.assertEqual(state.drain_observations("X"), [OBS_A])


if __name__ == "__main__":
    unittest.main()
