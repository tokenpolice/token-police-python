"""
Fault-injection matrix — verifies the SDK NEVER throws into customer code
from any internal failure. The only exception that may propagate is
``TokenPoliceBlockedError``, and only when ``enforce=True`` and a deny was
explicit.

This is a CI blocker. If any of these tests fail it means a code path can
crash the customer's application, which is the one thing the SDK promises
never to do.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Make the local SDK importable without `pip install -e .` having run.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import _classify
from token_police import local_evaluator
from token_police import state as tp_state


# ─── helpers ─────────────────────────────────────────────────────────
class _DummySession:
    user_id = "u1"
    paid_plan = "free"
    workflow_name = "wf"
    session_id = "s1"
    metadata = {}
    trace_id = "trace_test"


@pytest.fixture(autouse=True)
def _reset_state():
    tp_state.reset_pack()
    yield
    tp_state.reset_pack()


# ─── 1. classifier never throws ──────────────────────────────────────
def test_classify_returns_unknown_on_weird_input():
    class WeirdException(Exception):
        @property
        def status_code(self):
            raise RuntimeError("nope")

    result = _classify.classify_exception(WeirdException("boom"))
    assert result["error_kind"] == "unknown"
    assert result["http_status"] == 0


def test_classify_handles_string_input():
    # We pass a non-Exception on purpose to confirm we don't crash.
    out = _classify.classify_exception("not an exception")
    assert out["error_kind"] == "unknown"


def test_build_call_outcome_success():
    out = _classify.build_call_outcome(None, 123)
    assert out == {"status": "success", "duration_ms": 123}


def test_build_call_outcome_failed_truncates_message():
    # With no client configured the default detail mode is 'redacted',
    # so the raw string no longer ships — a SHA-256 hash of the FULL message
    # does instead (truncation now only applies in opt-in 'raw' mode, covered
    # by tests/test_error_message_scrub.py). The outcome must never carry the
    # raw error text.
    import hashlib
    big = "x" * 5000
    err = type("E", (Exception,), {})(big)
    out = _classify.build_call_outcome(err, 12)
    assert out["status"] == "failed"
    assert "error_message" not in out
    assert out["error_message_hash"] == hashlib.sha256(str(err).encode("utf-8")).hexdigest()
    assert out["duration_ms"] == 12


# ─── 2. local_evaluator never throws on bad input ────────────────────
def test_local_evaluator_no_pack_returns_allowed():
    result = local_evaluator.evaluate(None, _DummySession(), {"model": "gpt-4", "provider": "openai"})
    assert result["decision"]["status"] == "allowed"


def test_local_evaluator_malformed_pack_does_not_raise():
    bad = {"directives": "not a list", "loop_blocks": None}
    result = local_evaluator.evaluate(bad, _DummySession(), {"model": "", "provider": ""})
    # status should still be a sane value
    assert result["decision"]["status"] in {"allowed", "blocked", "rerouted"}


def test_local_evaluator_directive_missing_fields_does_not_raise():
    pack = {
        "directives": [
            {"id": "r1", "kind": "UNCONDITIONAL_BLOCK"},  # no selector
            {"id": "r2"},                                  # no kind at all
            {"id": "r3", "kind": "REROUTE", "selector": {}, "reroute": None},
        ],
        "loop_blocks": [],
    }
    result = local_evaluator.evaluate(pack, _DummySession(), {"model": "m", "provider": "openai"})
    assert "status" in result["decision"]


def test_local_evaluator_cross_provider_reroute_rejected_not_throws():
    pack = {
        "directives": [{
            "id": "r1", "kind": "REROUTE", "mode": "enforce", "priority": 50,
            "selector": {"match": None, "group_by": []},
            "reroute": {"from": None, "to": {"provider": "anthropic", "model": "claude-3-haiku"}},
        }],
        "loop_blocks": [],
    }
    s = _DummySession()
    result = local_evaluator.evaluate(pack, s, {"model": "gpt-4", "provider": "openai"})
    assert result["decision"]["status"] == "allowed"
    assert any(o["outcome"] == "reroute_rejected" for o in result["observations"])


def test_local_evaluator_force_shadow_turns_enforce_block_into_observation():
    pack = {
        "directives": [{
            "id": "r1", "kind": "UNCONDITIONAL_BLOCK", "mode": "enforce", "priority": 10,
            "selector": {"match": None, "group_by": []},
        }],
        "loop_blocks": [],
    }
    # Without force_shadow, an enforce directive blocks.
    live = local_evaluator.evaluate(pack, _DummySession(), {"model": "m", "provider": "openai"})
    assert live["decision"]["status"] == "blocked"
    # With force_shadow (dry_run), it never blocks — only a would_block observation.
    dry = local_evaluator.evaluate(pack, _DummySession(), {"model": "m", "provider": "openai"}, True)
    assert dry["decision"]["status"] == "allowed"
    assert any(o["outcome"] == "would_block" and o["mode"] == "dry_run" for o in dry["observations"])


# ─── 3. state.apply_* never throws ───────────────────────────────────
def test_apply_snapshot_with_garbage_does_not_throw():
    ok = tp_state.apply_snapshot({"this": "is not a snapshot"})
    assert ok is True or ok is False  # either is acceptable; just no exception


def test_apply_deltas_on_empty_pack_returns_false():
    tp_state.reset_pack()
    ok = tp_state.apply_deltas([{"op": "entity_blocked", "rule_id": "r1", "entity": "u1"}], 1)
    assert ok is False  # we have no pack to apply onto


def test_apply_deltas_idempotent_on_old_version():
    tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 5,
        "tenant_id": "t", "project_id": "p",
        "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    ok = tp_state.apply_deltas([], new_version=3)  # older than current
    assert ok is True  # silently discarded


def test_apply_deltas_gap_poisons_cache():
    tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 5,
        "tenant_id": "t", "project_id": "p",
        "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    ok = tp_state.apply_deltas([], new_version=10)  # gap from 5 → 10
    assert ok is False
    assert tp_state.is_cache_healthy() is False


def test_apply_snapshot_tenant_mismatch_poisons_cache():
    tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "tenantA", "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    ok = tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 2,
        "tenant_id": "tenantB", "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    assert ok is False
    assert tp_state.is_cache_healthy() is False


def test_apply_snapshot_missing_ids_refused_and_arms_pin_after_heal():
    # Failure-shape 1: a first snapshot with NO tenant_id is refused (cache
    # poisoned → inline /check), so the cross-tenant pin never arms from a
    # malformed snapshot. A later well-formed snapshot heals the cache AND arms
    # the pin, after which a different-tenant snapshot is correctly rejected.
    ok = tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 1,
        "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    assert ok is False
    assert tp_state.is_cache_healthy() is False

    # A well-formed snapshot heals the cache and arms the pin from a clean state.
    assert tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 2,
        "tenant_id": "tenantA", "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    }) is True
    assert tp_state.is_cache_healthy() is True

    # Pin is now armed: a different-tenant snapshot is rejected.
    assert tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 3,
        "tenant_id": "tenantB", "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    }) is False
    assert tp_state.is_cache_healthy() is False


def test_apply_snapshot_missing_project_id_does_not_arm_half_pin():
    # Failure-shape 2: a first snapshot WITH tenant_id but NO project_id is
    # refused and must NOT arm a half-pin (project=None). Proof: a later snapshot
    # with the same tenant AND a real project_id must still apply cleanly — if a
    # half-pin had armed, it would be rejected as a project mismatch forever.
    ok = tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "tenantA", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    assert ok is False
    assert tp_state.is_cache_healthy() is False

    assert tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 2,
        "tenant_id": "tenantA", "project_id": "realproj", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    }) is True
    assert tp_state.is_cache_healthy() is True


def test_apply_snapshot_empty_string_tenant_id_treated_as_absent():
    # Failure-shape 3: an empty-string tenant_id is as good as absent — refused.
    ok = tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "", "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    assert ok is False
    assert tp_state.is_cache_healthy() is False


def test_apply_deltas_unknown_op_is_skipped():
    tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "t", "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    ok = tp_state.apply_deltas([{"op": "future_op_we_dont_know", "data": 42}], new_version=2)
    assert ok is True
    assert tp_state.get_pack_version() == 2


# ─── 4. runtime detection never throws ───────────────────────────────
def test_runtime_resolve_invalid_string_falls_back():
    from token_police import runtime
    result = runtime.resolve_deployment("not_a_real_mode")
    assert result in ("daemon", "serverless", "edge")


def test_runtime_detect_handles_no_env():
    from token_police import runtime
    # Strip any markers temporarily
    saved = {k: os.environ.pop(k) for k in list(os.environ.keys())
             if k in runtime._SERVERLESS_MARKERS + runtime._EDGE_MARKERS}
    try:
        assert runtime.detect_deployment_mode() == "daemon"
    finally:
        os.environ.update(saved)


# ─── 5. Mock LLM call: failure-path classification ───────────────────
def test_classifier_recognizes_rate_limit_by_name():
    err = type("RateLimitError", (Exception,), {})("slow down")
    out = _classify.classify_exception(err)
    assert out["error_kind"] == "rate_limited"


def test_classifier_recognizes_401_by_status():
    class AuthErr(Exception):
        status_code = 401
    out = _classify.classify_exception(AuthErr("bad key"))
    assert out["error_kind"] == "auth_error"
    assert out["http_status"] == 401


def test_classifier_recognizes_5xx_as_server_error():
    class ServerErr(Exception):
        status_code = 502
    out = _classify.classify_exception(ServerErr("upstream gone"))
    assert out["error_kind"] == "server_error"


# ─── 6. observations queue is safe under bad input ───────────────────
def test_push_observation_ignores_non_dict():
    tp_state.push_observation("not a dict")  # type: ignore[arg-type]
    tp_state.push_observation(None)           # type: ignore[arg-type]
    assert tp_state.drain_observations() == []


def test_drain_observations_returns_then_clears():
    tp_state.push_observation({"rule_id": "r1", "outcome": "would_block", "mode": "dry_run"})
    drained = tp_state.drain_observations()
    assert len(drained) == 1
    assert tp_state.drain_observations() == []


# ─── 7. invalidate / cache health flag ───────────────────────────────
def test_invalidate_pack_marks_unhealthy():
    tp_state.apply_snapshot({
        "schema_version": 1, "type": "snapshot", "version": 1,
        "tenant_id": "t", "project_id": "p", "directives": [], "loop_blocks": [], "ttl_seconds": 10,
    })
    assert tp_state.is_cache_healthy() is True
    tp_state.invalidate_pack("test")
    assert tp_state.is_cache_healthy() is False
    assert tp_state.get_pack() is None


# ─── 8. embedding extraction never throws ────────────────────────────
# Embedding usage extraction runs in the customer's hot path. Any failure
# must be swallowed so the original LLM call still returns to the caller.
def test_extract_embedding_usage_none_response():
    from token_police.enforcer import _extract_embedding_usage
    model, tokens, raw = _extract_embedding_usage(None, "openai")
    assert model == "unknown" and tokens == 0 and raw is None


def test_extract_embedding_usage_broken_response_attributes():
    from token_police.enforcer import _extract_embedding_usage

    class _Broken:
        @property
        def usage(self):
            raise RuntimeError("boom")

        @property
        def model(self):
            raise RuntimeError("boom")

    # Must not raise.
    model, tokens, _ = _extract_embedding_usage(_Broken(), "openai")
    assert model == "unknown"


def test_extract_embedding_usage_unknown_provider_falls_back():
    from token_police.enforcer import _extract_embedding_usage

    class _Resp:
        model = "mystery"
        usage = type("U", (), {"prompt_tokens": 7, "total_tokens": 7})()

    model, tokens, _ = _extract_embedding_usage(_Resp(), "xai")  # not in the embedding table
    # Falls back to the best-effort branch — must still return a sane tuple.
    assert tokens == 7


def test_approximate_hf_embedding_tokens_never_throws_on_bad_input():
    from token_police.enforcer import _approximate_hf_embedding_tokens
    # Pass garbage; must return 0 without raising.
    assert _approximate_hf_embedding_tokens(None, None) == 0
    assert _approximate_hf_embedding_tokens((), {"text": object()}) == 0


# ─── 9. embedding composition is fail-safe on malformed input ────────
def test_embedding_composition_handles_non_string_items():
    from token_police.composition import _parse_embedding_input

    # Mix of bad types — must not raise; produce a structural marker.
    result = _parse_embedding_input("openai", {"input": [object(), 3.14, None]})
    assert isinstance(result, list)
    # Every entry has role="input"
    for entry in result:
        assert entry.get("role") == "input"


# ─── 10. stream protocol preservation + per-chunk tap fail-open ──────
# Regression guards for (Cohere async client returns a sync generator
# but the consumer iterates with `async for`), (litellm's dual-protocol
# CustomStreamWrapper was coerced to an async_generator, breaking sync `for`),
# and the rule that a tap failure (chunk-shape drift) must never abort the
# customer's stream.
import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

from token_police._detect import (
    is_sync_iterator,
    is_async_iterator,
    is_dual_protocol,
)


class _SyncOnlyStream:
    """A plain sync generator-like stream (Cohere-B underlying shape)."""

    def __init__(self, chunks):
        self._it = iter(chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)


class _DualProtocolStream:
    """Exposes BOTH sync and async iteration (litellm CustomStreamWrapper has
    __iter__/__next__ AND __aiter__/__anext__)."""

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self._si = iter(self._chunks)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._si)

    def __aiter__(self):
        self._ai = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._ai)
        except StopIteration:
            raise StopAsyncIteration


def _drain_async(agen):
    async def _run():
        out = []
        async for c in agen:
            out.append(c)
        return out

    # Use an isolated loop and restore the previous global loop afterwards —
    # `asyncio.run` would leave the global loop closed, breaking sibling tests
    # that call `asyncio.get_event_loop()` (e.g. test_pydantic_ai_tools).
    try:
        prev = asyncio.get_event_loop()
    except RuntimeError:
        prev = None
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(_run())
    finally:
        loop.close()
        asyncio.set_event_loop(prev)


def _sess():
    s = MagicMock()
    s._defer_telemetry = False
    return s


def test_detect_classifies_dual_protocol_stream():
    dual = _DualProtocolStream([1, 2, 3])
    assert is_sync_iterator(dual) and is_async_iterator(dual)
    assert is_dual_protocol(dual)

    sync = _SyncOnlyStream([1])
    assert is_sync_iterator(sync) and not is_async_iterator(sync)


def test_sync_tap_yields_all_chunks_for_dual_protocol(monkeypatch):
    # A dual-protocol stream routed to the SYNC tap must iterate with
    # a plain `for` and yield every chunk unchanged.
    from token_police import enforcer

    chunks = ["a", "b", "c"]
    gen = enforcer._wrap_sync_stream(
        _DualProtocolStream(chunks), "cohere", _sess(), {}, 0, None,
        datetime.now(timezone.utc),
    )
    assert list(gen) == chunks


def test_async_tap_bridges_sync_underlying(monkeypatch):
    # A sync underlying stream handed to the async tap must still be
    # consumable with `async for` and yield every chunk.
    from token_police import enforcer

    chunks = ["x", "y", "z"]
    agen = enforcer._wrap_async_stream(
        _SyncOnlyStream(chunks), "cohere", _sess(), {}, 0, None,
        datetime.now(timezone.utc),
    )
    assert _drain_async(agen) == chunks


def test_mode_a_async_tap_bridges_sync_underlying():
    # (Mode-A / Cohere AsyncClientV2.chat_stream): sync underlying,
    # async consumer.
    from token_police import enforcer

    chunks = ["1", "2"]
    agen = enforcer._wrap_mode_a_async_stream(
        _SyncOnlyStream(chunks), "cohere", _sess(), False,
    )
    assert _drain_async(agen) == chunks


def test_sync_tap_failopen_on_accumulator_error(monkeypatch):
    # A tap failure (chunk-shape drift making accumulation throw) must cost
    # telemetry, NOT abort the customer's stream — all chunks still delivered.
    from token_police import enforcer

    def _boom(*a, **k):
        raise RuntimeError("chunk shape drifted")

    monkeypatch.setattr(enforcer, "_accumulate_stream_chunk", _boom)
    chunks = ["p", "q", "r"]
    gen = enforcer._wrap_sync_stream(
        _SyncOnlyStream(chunks), "cohere", _sess(), {}, 0, None,
        datetime.now(timezone.utc),
    )
    assert list(gen) == chunks  # no RuntimeError escapes


def test_async_tap_failopen_on_accumulator_error(monkeypatch):
    from token_police import enforcer

    def _boom(*a, **k):
        raise RuntimeError("chunk shape drifted")

    monkeypatch.setattr(enforcer, "_accumulate_stream_chunk", _boom)
    chunks = ["p", "q"]
    agen = enforcer._wrap_async_stream(
        _SyncOnlyStream(chunks), "cohere", _sess(), {}, 0, None,
        datetime.now(timezone.utc),
    )
    assert _drain_async(agen) == chunks


def test_genuine_stream_error_still_propagates():
    # A real error from the underlying provider stream is the customer's error
    # and MUST propagate (we only swallow OUR tap failures, not theirs).
    from token_police import enforcer

    class _Boom:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("provider stream died")

    gen = enforcer._wrap_sync_stream(
        _Boom(), "cohere", _sess(), {}, 0, None, datetime.now(timezone.utc),
    )
    with pytest.raises(ValueError, match="provider stream died"):
        list(gen)


# ─── 11. dry_run = full enforce parity (golden rule) ─────────────────
# dry_run must run the EXACT same path as enforce — local-eval + inline
# /check — and differ ONLY in never acting (no block, no reroute, no
# TokenPoliceBlockedError). 'off' skips the pre-flight entirely. enforce
# still blocks/reroutes. These guard the SSE/audit standardization change.
import token_police as tp
from token_police import enforcer as _enforcer
from token_police.exceptions import TokenPoliceBlockedError


def _init_with_check(firewall, check_result):
    """Init a client in `firewall` mode with check_sync/log_sync mocked.

    With no cached pack the enforcer takes the State-B path (inline /check)
    in every deployment, so check_result drives the verdict directly.
    Returns (client, check_mock).
    """
    client = tp.init(api_key="tp_sk_test_enf", firewall=firewall)
    check_mock = MagicMock(return_value=check_result)
    client.check_sync = check_mock
    client.log_sync = MagicMock()
    return client, check_mock


def test_dry_run_calls_check_but_never_raises_on_block():
    _client, check_mock = _init_with_check(
        "dry_run", {"status": "blocked", "reason": "budget exceeded"})
    try:
        # Golden rule: dry_run must NOT raise even on an explicit block...
        _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
        # ...but it DID hit /check (full parity with enforce).
        check_mock.assert_called_once()
    finally:
        tp.uninstrument()


def test_dry_run_does_not_reroute():
    _client, check_mock = _init_with_check(
        "dry_run",
        {"status": "allowed",
         "reroute": {"mode": "enforce", "model": "cheap-model", "provider": "openai"}})
    kwargs = {"model": "gpt-4"}
    try:
        _enforcer._run_sync_check(kwargs=kwargs, provider="openai")
        # dry_run suppresses the action — the model is untouched.
        assert kwargs["model"] == "gpt-4"
        check_mock.assert_called_once()
    finally:
        tp.uninstrument()


def test_enforce_raises_on_block():
    _client, check_mock = _init_with_check(
        "enforce", {"status": "blocked", "reason": "budget exceeded"})
    try:
        with pytest.raises(TokenPoliceBlockedError):
            _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
        check_mock.assert_called_once()
    finally:
        tp.uninstrument()


def test_enforce_applies_reroute():
    _client, _check_mock = _init_with_check(
        "enforce",
        {"status": "allowed",
         "reroute": {"mode": "enforce", "model": "cheap-model", "provider": "openai"}})
    kwargs = {"model": "gpt-4"}
    try:
        _enforcer._run_sync_check(kwargs=kwargs, provider="openai")
        assert kwargs["model"] == "cheap-model"
    finally:
        tp.uninstrument()


def test_off_mode_skips_check_entirely():
    _client, check_mock = _init_with_check(
        "off", {"status": "blocked", "reason": "budget exceeded"})
    try:
        _enforcer._run_sync_check(kwargs={"model": "gpt-4"}, provider="openai")
        check_mock.assert_not_called()
    finally:
        tp.uninstrument()


# ─── 12. /stream uses Bearer auth, not a query-string api_key ────────
def test_stream_uses_bearer_auth_not_query_param():
    from token_police.stream import StreamClient

    sc = StreamClient(
        base_url="http://localhost:59999", api_key="tp_sk_secret",
        sdk_version="1.0.0", deployment="daemon", client_id="c1", firewall="dry_run",
    )
    captured = {}

    class _FakeCtx:
        def __enter__(self):
            resp = MagicMock()
            resp.status_code = 500  # short-circuit before _pump()
            return resp

        def __exit__(self, *exc):
            return False

    def _fake_stream(method, url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return _FakeCtx()

    # Status 500 is transient, so _connect_and_pump now RAISES (to climb the
    # backoff ladder). The url/headers are captured in the httpx.stream side_effect
    # BEFORE the status check, so the Bearer/URL invariant is intact at raise time.
    with patch("token_police.stream.httpx.stream", side_effect=_fake_stream):
        with pytest.raises(RuntimeError):
            sc._connect_and_pump()

    assert "api_key=" not in captured["url"]
    assert captured["url"].endswith("/v1/guard/stream")
    assert captured["headers"]["Authorization"] == "Bearer tp_sk_secret"
