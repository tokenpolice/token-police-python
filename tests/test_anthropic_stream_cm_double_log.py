"""Regression suite for the anthropic `messages.stream()` CM telemetry fix
and the anthropic BETA seam enforcement fix (PR502-followups).

Part A/B/C/D-a use the REAL installed `anthropic` package + a real
`MockTransport` SSE fixture + the REAL installed
`opentelemetry-instrumentation-anthropic` package, driven through
`token_police.init()` — no anthropic SDK internals are faked, so these are
the strongest possible regression pins for "does the real instrumented stack
still log exactly the right number of rows with the right numbers". No real
network is ever used (MockTransport intercepts at the transport layer).
Fixtures/expected values are lifted from the harness at
/Users/jaffer/.claude/jobs/12131bd1/tmp/ (harness.py, matrix.py, scenarios2.py)
and re-verified against this checkout.

Part D-b..e and Part E use a lighter, fully-faked harness (same idiom as
tests/test_f6_anthropic_stream_preflight_context.py /
tests/test_anthropic_stream_enter_failure.py: fake Messages/AsyncMessages
patched into sys.modules, `enforcer._instrument_anthropic_stream()` exercised
end-to-end) because those scenarios need direct control over WHEN a
synthetic "instrumentor" span starts (at `.stream()` vs at manager-`__enter__`
vs never) — something the real installed instrumentor's fixed behavior can't
express.

Bug summary (see enforcer.py / context.py / telemetry.py for the fix):
`with client.messages.stream(...)` used to emit 2 LLM rows OUTSIDE any
`@workflow` (~2x OVER-BILL: the manual `_finalize` row plus the OTel
instrumentor's own un-suppressed span). INSIDE a workflow, the old one-shot
`_suppress_anthropic_otel_stream` SESSION flag leaked `True` and silently
killed the NEXT anthropic call's telemetry (UNDER-BILL). The fix replaces
that flag with a per-call ``_anthropic_stream_span_window`` ContextVar record
(token_police/context.py) claimed by ``claim_anthropic_stream_span``.
Separately, `anthropic.resources.beta.messages.messages.Messages` is a
SIBLING class (not a subclass) of the non-beta `Messages`, so
`client.beta.messages.*` got ZERO pre-flight `/check` until 4 new
`_TARGET_METHODS` rows were added for it (and its Bedrock beta twins).
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("anthropic")
pytest.importorskip("opentelemetry.instrumentation.anthropic")

import anthropic  # noqa: E402
# The transport package the INSTALLED anthropic is built on (httpx or httpx2).
# Never `import httpx` for a client handed to anthropic — see _anthropic_httpx.
from _anthropic_httpx import anthropic_httpx as httpx  # noqa: E402
import token_police as tp  # noqa: E402
from token_police import enforcer  # noqa: E402
from token_police.client import TokenPolice  # noqa: E402
from token_police.context import (  # noqa: E402
    _anthropic_stream_span_window,
)
from token_police.exceptions import TokenPoliceBlockedError  # noqa: E402
from token_police.telemetry import TokenPoliceSpanProcessor  # noqa: E402

MODEL = "claude-3-5-sonnet-20241022"
KW = dict(model=MODEL, max_tokens=64, messages=[{"role": "user", "content": "hi"}])


# ═══════════════════════════════════════════════════════════════════
# Part A — real-SDK SSE fixtures (lifted from harness.py / scenarios2.py)
# ═══════════════════════════════════════════════════════════════════

def _sse(events):
    out = []
    for name, obj in events:
        out.append("event: %s\ndata: %s\n\n" % (name, json.dumps(obj)))
    return "".join(out).encode()


# message_start carries output_tokens=1, message_delta carries 7 -> the
# anthropic SDK's own accumulated final_message.usage.output_tokens == 7.
# An emitter that (wrongly) ACCUMULATES both instead of taking the SDK's
# final tally would report out=8 — a DISTINCT, deliberately-chosen
# fingerprint (not a coincidence) that unambiguously identifies a row that
# came from an un-suppressed OTel stream span rather than the manual
# `_finalize` emitter.
ANTHROPIC_SSE = _sse([
    ("message_start", {"type": "message_start", "message": {
        "id": "msg_tp_1", "type": "message", "role": "assistant", "model": MODEL,
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 1}}}),
    ("content_block_start", {"type": "content_block_start", "index": 0,
                              "content_block": {"type": "text", "text": ""}}),
    ("content_block_delta", {"type": "content_block_delta", "index": 0,
                              "delta": {"type": "text_delta", "text": "Hello there"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    ("message_delta", {"type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 7}}),
    ("message_stop", {"type": "message_stop"}),
])

SMALL_JSON = {
    "id": "msg_tp_ns", "type": "message", "role": "assistant", "model": MODEL,
    "content": [{"type": "text", "text": "Hello there"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 11, "output_tokens": 7},
}

# A non-stream `create()` fixture with a DISAMBIGUATING usage shape (100/50,
# distinct from the stream's 11/7) — mixed-arm totals below only add up
# correctly if each row's tokens came from the right emitter.
BIG_JSON = {
    "id": "msg_big", "type": "message", "role": "assistant", "model": MODEL,
    "content": [{"type": "text", "text": "big"}],
    "stop_reason": "end_turn", "stop_sequence": None,
    "usage": {"input_tokens": 100, "output_tokens": 50},
}


def _handler(big=False):
    def handler(request):
        try:
            body = json.loads(request.content or b"{}")
        except Exception:
            body = {}
        if body.get("stream"):
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                   content=ANTHROPIC_SSE)
        return httpx.Response(200, json=BIG_JSON if big else SMALL_JSON)
    return handler


def _clients(handler):
    """Fresh real anthropic Anthropic/AsyncAnthropic clients bound to a
    MockTransport — no real network, real SDK + real installed instrumentor."""
    transport = httpx.MockTransport(handler)
    sc = anthropic.Anthropic(api_key="sk-ant-test", base_url="http://anthropic.test",
                              http_client=httpx.Client(transport=transport))
    async_transport = httpx.MockTransport(handler)
    ac = anthropic.AsyncAnthropic(api_key="sk-ant-test", base_url="http://anthropic.test",
                                   http_client=httpx.AsyncClient(transport=async_transport))
    return sc, ac


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _llm_rows(rows):
    return [r for r in rows if (r.get("span") or {}).get("span_kind") == "llm"]


# ═══════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════

def _reset_stray_real_anthropic_stream_patches():
    """Defensive, run before this module's own `tp.init()`.

    Other test files in this suite (tests/test_f6_anthropic_stream_preflight_
    context.py, tests/test_anthropic_stream_enter_failure.py,
    tests/test_anthropic_async_stream_check.py) call
    ``enforcer._instrument_anthropic_stream()`` directly against FAKE
    ``anthropic.resources.messages`` modules patched into ``sys.modules`` —
    but ``anthropic.resources.beta.messages.messages`` (and the Bedrock beta
    twins) stay cached in ``sys.modules`` under their OWN dotted keys, so
    that call can incidentally ALSO wrap the REAL beta Messages/AsyncMessages
    classes' `.stream`. If that happens before
    ``opentelemetry.instrumentation.anthropic.AnthropicInstrumentor().instrument()``
    has run, the wrap order inverts — TP's wrapper becomes the INNER layer
    and OTel's span-creation ends up OUTSIDE our suppression window — which
    breaks beta-stream suppression for every later real end-to-end test in
    this process (this file's own beta-seam tests included). Restore any
    such stray real-class patches before this module's `tp.init()`, so
    instrumentation always installs fresh, in the correct order (OTel first,
    then TokenPolice's wrapper outermost)."""
    candidates = []
    try:
        candidates += [anthropic.resources.messages.Messages,
                       anthropic.resources.messages.AsyncMessages]
    except Exception:
        pass
    try:
        from anthropic.resources.beta.messages.messages import (
            Messages as _BetaMessages, AsyncMessages as _BetaAsyncMessages,
        )
        candidates += [_BetaMessages, _BetaAsyncMessages]
    except ImportError:
        pass
    try:
        from anthropic.lib.bedrock._beta_messages import (
            Messages as _BedrockBetaMessages, AsyncMessages as _BedrockBetaAsyncMessages,
        )
        candidates += [_BedrockBetaMessages, _BedrockBetaAsyncMessages]
    except ImportError:
        pass
    for cls in candidates:
        original = enforcer._originals.pop((cls, "stream"), None)
        if original is not None:
            try:
                setattr(cls, "stream", original)
            except Exception:
                pass


@pytest.fixture(scope="module")
def _tp_client():
    """One real, module-wide TokenPolice client. `auto_instrument()` is
    process-global-once (`enforcer._is_instrumented`), so this is the single
    point that installs the real pre-flight + stream wrappers on the REAL
    installed anthropic package for this whole file.

    ``enforcer.uninstrument()`` first — a no-op if nothing is instrumented
    yet, or a full clean restore (+ resets ``_is_instrumented`` to False) if
    an earlier test already ran a normal ``tp.init()``. Either way this
    module's own ``tp.init()`` below gets a real, fresh `auto_instrument()`
    pass rather than silently no-op'ing against stale global state."""
    enforcer.uninstrument()
    _reset_stray_real_anthropic_stream_patches()
    client = tp.init(
        api_key="tp_sk_test_cmdbl", base_url="http://127.0.0.1:9",
        firewall="enforce", deployment="serverless", timeout=0.2, log_errors=True,
    )
    return client


@pytest.fixture
def env(_tp_client, monkeypatch):
    """Per-test row/check recorder. `firewall="enforce"` (module-wide) with a
    permissive check stub behaves exactly like `firewall="off"` for every
    success-path assertion in this file, while also letting the beta-seam
    tests count `/check` calls and the golden-rule test deny a call."""
    rows = []
    checks = []

    def log_sync(self, **kw):
        rows.append(kw)

    def check_sync(self, **kw):
        checks.append(kw)
        return {"status": "allowed"}

    async def check_async(self, **kw):
        checks.append(kw)
        return {"status": "allowed"}

    monkeypatch.setattr(TokenPolice, "log_sync", log_sync)
    monkeypatch.setattr(TokenPolice, "check_sync", check_sync)
    monkeypatch.setattr(TokenPolice, "check", check_async)
    return NS(rows=rows, checks=checks, client=_tp_client)


# ═══════════════════════════════════════════════════════════════════
# Part B (item 3) — anthropic BETA seam regression.
# ═══════════════════════════════════════════════════════════════════

class TestBetaSeam:
    def test_beta_stream_cm_checks_and_logs_exactly_once_with_real_tokens(self, env):
        """Was: 0 /check calls and a 0/0-token row (sibling class had no
        pre-flight enforcement at all and no wrapper captured usage)."""
        sc, _ac = _clients(_handler())
        with sc.beta.messages.stream(**KW) as st:
            for _ in st.text_stream:
                pass
        assert len(env.checks) == 1
        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7

    def test_beta_create_non_stream_checks_and_logs_once(self, env):
        sc, _ac = _clients(_handler())
        sc.beta.messages.create(**KW)
        assert len(env.checks) == 1
        assert len(_llm_rows(env.rows)) == 1

    def test_beta_create_stream_true_checks_and_logs_once(self, env):
        sc, _ac = _clients(_handler())
        stream = sc.beta.messages.create(**KW, stream=True)
        for _ in stream:
            pass
        assert len(env.checks) == 1
        assert len(_llm_rows(env.rows)) == 1

    def test_count_tokens_never_checks_or_logs(self, env):
        """Deliberately excluded from _TARGET_METHODS: a /check on a free
        metadata endpoint could block it for zero revenue protection."""
        sc, _ac = _clients(_handler())
        sc.messages.count_tokens(model=MODEL, messages=KW["messages"])
        assert env.checks == []
        assert env.rows == []

    def test_beta_count_tokens_never_checks_or_logs(self, env):
        sc, _ac = _clients(_handler())
        sc.beta.messages.count_tokens(model=MODEL, messages=KW["messages"])
        assert env.checks == []
        assert env.rows == []

    def test_non_beta_create_and_stream_still_check_and_log_once_each(self, env):
        """Control: the non-beta siblings were never broken — pins that the
        beta fix didn't regress them."""
        sc, _ac = _clients(_handler())
        sc.messages.create(**KW)
        assert len(env.checks) == 1
        assert len(_llm_rows(env.rows)) == 1

        env.checks.clear()
        env.rows.clear()
        with sc.messages.stream(**KW) as st:
            for _ in st.text_stream:
                pass
        assert len(env.checks) == 1
        assert len(_llm_rows(env.rows)) == 1


# ═══════════════════════════════════════════════════════════════════
# Part C (item 4) — the two headline bugs, regression-pinned with real
# numbers.
# ═══════════════════════════════════════════════════════════════════

class TestHeadlineBugs:
    def test_outside_workflow_sync_stream_is_exactly_one_row_11_7(self, env):
        sc, _ac = _clients(_handler())
        with sc.messages.stream(**KW) as st:
            for _ in st.text_stream:
                pass
        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7
        assert not any(r["output_tokens"] == 8 for r in llm)

    def test_outside_workflow_async_stream_is_exactly_one_row_11_7(self, env):
        _sc, ac = _clients(_handler())

        async def go():
            async with ac.messages.stream(**KW) as st:
                async for _ in st.text_stream:
                    pass
        _run(go())

        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7
        assert not any(r["output_tokens"] == 8 for r in llm)

    def test_inside_workflow_stream_then_create_totals_111_57(self, env):
        """The under-bill: pre-fix, the create's row was ABSENT (replaced by
        an out=8 dud from the leaked suppress flag eating its span)."""
        sc, _ac = _clients(_handler(big=True))  # create() -> 100/50, stream -> 11/7

        @tp.workflow(name="wf_probe", session_id="sess_probe")
        def _inner():
            with sc.messages.stream(**KW) as st:
                for _ in st.text_stream:
                    pass
            sc.messages.create(**KW)

        _inner()
        llm = _llm_rows(env.rows)
        assert len(llm) == 2
        assert sum(r["input_tokens"] for r in llm) == 111
        assert sum(r["output_tokens"] for r in llm) == 57

    def test_inside_workflow_create_stream_create_three_rows_ordered_012(self, env):
        """Pre-fix span_order was [0, 2, 1] — the leaked flag suppressed the
        WRONG span, corrupting order too."""
        sc, _ac = _clients(_handler(big=True))

        @tp.workflow(name="wf_probe2", session_id="sess_probe2")
        def _inner():
            sc.messages.create(**KW)
            with sc.messages.stream(**KW) as st:
                for _ in st.text_stream:
                    pass
            sc.messages.create(**KW)

        _inner()
        llm = _llm_rows(env.rows)
        assert len(llm) == 3
        assert [r["span"]["span_order"] for r in llm] == [0, 1, 2]
        assert sum(r["input_tokens"] for r in llm) == 211
        assert sum(r["output_tokens"] for r in llm) == 107


# ═══════════════════════════════════════════════════════════════════
# Part D (item 5) — version/shape resilience. NO version reads anywhere;
# every scenario is emulated at runtime.
# ═══════════════════════════════════════════════════════════════════

class TestShapeResilienceRealInstrumentor:
    def test_a_usage_stripped_from_stream_spans_still_one_row(self, env, monkeypatch):
        """0.60-era shape: the instrumentor's own usage extraction is
        neutered. Our suppression is identity-based, not usage-based, so
        this must have ZERO effect on the result."""
        import opentelemetry.instrumentation.anthropic.streaming as _streaming_mod

        def _noop_set_usage(*a, **k):
            return None

        monkeypatch.setattr(_streaming_mod, "_set_token_usage", _noop_set_usage)

        sc, ac = _clients(_handler())
        with sc.messages.stream(**KW) as st:
            for _ in st.text_stream:
                pass
        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7

        env.rows.clear()

        async def go():
            async with ac.messages.stream(**KW) as st:
                async for _ in st.text_stream:
                    pass
        _run(go())
        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7


# ── Fully-faked harness for shapes (b)/(c)/(d)/(e) — direct control over
# WHEN/WHETHER a synthetic instrumentor span starts. Same idiom as
# tests/test_f6_anthropic_stream_preflight_context.py.
# ── ──────────────────────────────────────────────────────────────────

REAL_SCOPE = "opentelemetry.instrumentation.anthropic"
REAL_NAME = "anthropic.chat"


class _Scope:
    def __init__(self, name):
        self.name = name


class _ShapeSpan:
    """Synthetic span carrying the attributes a real OTel anthropic
    instrumentor span would (enough for telemetry.on_end's generic LLM path
    to treat it as a legitimate row when NOT suppressed)."""

    def __init__(self, scope, name, model=MODEL, input_tokens=11, output_tokens=8,
                 hostile=False):
        self.attributes = {
            "gen_ai.request.model": model,
            "gen_ai.system": "anthropic",
        }
        if input_tokens is not None:
            self.attributes["gen_ai.usage.input_tokens"] = input_tokens
        if output_tokens is not None:
            self.attributes["gen_ai.usage.output_tokens"] = output_tokens
        self.name = name
        self.instrumentation_scope = _Scope(scope)
        self.context = NS(trace_id=0x1111, span_id=0x2222)
        self.parent = None
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = None
        self._hostile = hostile

    def set_attribute(self, key, value):
        if self._hostile:
            raise RuntimeError("hostile span.set_attribute")
        self.attributes[key] = value


class _ShapeStream:
    """Sync stream — `get_final_message()` is a plain (non-async) method,
    matching the real `anthropic.MessageStream`, which the sync
    `_AnthropicStreamMgrWrapper._finalize` calls without `await`."""

    def __init__(self, in_tok=11, out_tok=7):
        self._final = NS(
            model=MODEL, content=[],
            usage=NS(input_tokens=in_tok, output_tokens=out_tok,
                      cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )

    def __iter__(self):
        return iter([])

    def get_final_message(self):
        return self._final


class _ShapeStreamAsync:
    """Async stream — `get_final_message()` is `async def`, matching the real
    `anthropic.AsyncMessageStream`, which `_afinalize` awaits."""

    def __init__(self, in_tok=11, out_tok=7):
        self._final = NS(
            model=MODEL, content=[],
            usage=NS(input_tokens=in_tok, output_tokens=out_tok,
                      cache_read_input_tokens=0, cache_creation_input_tokens=0),
        )

    def __aiter__(self):
        async def _agen():
            return
            yield  # pragma: no cover - makes this an async generator
        return _agen()

    async def get_final_message(self):
        return self._final


class _ShapeMgr:
    """Fake vendor MessageStreamManager under direct control of WHEN (and
    whether) the synthetic "instrumentor" starts its span.

    ``span_at``: "construct" (today's real shape — span starts inside
    `.stream()`), "enter" (a hypothetical future instrumentor that starts its
    span at manager-`__enter__` instead — this is what proves W2 is
    non-vacuous), or None (no span at all).
    """

    def __init__(self, stream, span_at="construct", scope=REAL_SCOPE, name=REAL_NAME,
                 processor=None, dud_tokens=(11, 8), hostile=False):
        self._stream = stream
        self._span_at = span_at
        self._scope = scope
        self._name = name
        self._processor = processor or TokenPoliceSpanProcessor()
        self._dud_tokens = dud_tokens
        self._hostile = hostile
        self.span = None
        if span_at == "construct":
            self._start()

    def _start(self):
        self.span = _ShapeSpan(self._scope, self._name,
                                input_tokens=self._dud_tokens[0],
                                output_tokens=self._dud_tokens[1],
                                hostile=self._hostile)
        self._processor.on_start(self.span)

    def __enter__(self):
        if self._span_at == "enter":
            self._start()
        return self._stream

    def __exit__(self, *a):
        if self.span is not None:
            self._processor.on_end(self.span)
        return False

    async def __aenter__(self):
        if self._span_at == "enter":
            self._start()
        return self._stream

    async def __aexit__(self, *a):
        if self.span is not None:
            self._processor.on_end(self.span)
        return False


class _Factory:
    """Stands in for the ORIGINAL (unpatched) `.stream()` body."""

    def __init__(self, mgr_factory):
        self._mgr_factory = mgr_factory
        self.calls = []
        self.mgrs = []

    def __call__(self, kwargs):
        self.calls.append(dict(kwargs))
        mgr = self._mgr_factory()
        self.mgrs.append(mgr)
        return mgr


def _install_fake_anthropic(sync_factory, async_factory):
    """Patch fake Messages/AsyncMessages into sys.modules, instrument them,
    and return (sync_client, async_client, cleanup)."""

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
    async_client = AsyncMessages()
    async_client._factory = async_factory

    def cleanup():
        enforcer._originals.pop((Messages, "stream"), None)
        enforcer._originals.pop((AsyncMessages, "stream"), None)
        for n, v in saved.items():
            if v is None:
                sys.modules.pop(n, None)
            else:
                sys.modules[n] = v

    return sync_client, async_client, cleanup


@pytest.fixture
def fakes(env):
    """Install the fakes, yield (sync_client, async_client), then always
    clean up sys.modules — even on failure."""
    installed = {}

    def _install(sync_mgr_factory, async_mgr_factory=None):
        sync_f = _Factory(sync_mgr_factory)
        async_f = _Factory(async_mgr_factory or sync_mgr_factory)
        sc, ac, cleanup = _install_fake_anthropic(sync_f, async_f)
        installed["cleanup"] = cleanup
        # Exposed so a test needing the REAL anthropic package back mid-test
        # (e.g. to prove a subsequent real create() call still meters
        # correctly) can restore sys.modules early. Safe to call twice — the
        # fixture teardown below calls the same idempotent function again.
        env.cleanup = cleanup
        return sc, ac, sync_f, async_f

    try:
        yield _install, env
    finally:
        cleanup = installed.get("cleanup")
        if cleanup is not None:
            cleanup()


class TestShapeResilienceFaked:
    def test_b_future_shape_span_starts_at_enter_w2_catches_it(self, fakes):
        install, env = fakes
        sc, _ac, _sync_f, _async_f = install(
            lambda: _ShapeMgr(_ShapeStream(), span_at="enter"))

        wrapper = sc.stream(**KW)
        # W1: nothing fired inside .stream() itself in this shape.
        assert wrapper._span_window == {"seen": 0, "limit": 1}
        with wrapper as stream:
            list(stream)
        # W2 (manager-enter) caught it instead.
        assert wrapper._span_window == {"seen": 1, "limit": 1}

        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7

    def test_b_future_shape_async_twin(self, fakes):
        install, env = fakes
        _sc, ac, _sync_f, _async_f = install(
            lambda: _ShapeMgr(_ShapeStreamAsync(), span_at="enter"))

        async def go():
            wrapper = ac.stream(**KW)
            assert wrapper._span_window == {"seen": 0, "limit": 1}
            async with wrapper as stream:
                async for _ in stream:
                    pass
            assert wrapper._span_window == {"seen": 1, "limit": 1}
        _run(go())

        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7

    def test_c_no_stream_manager_span_at_all(self, fakes):
        install, env = fakes
        sc, _ac, _sync_f, _async_f = install(
            lambda: _ShapeMgr(_ShapeStream(), span_at=None))

        with sc.stream(**KW) as stream:
            list(stream)

        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7

    def test_d_scope_renamed_but_name_still_anthropic_prefixed(self, fakes):
        install, env = fakes
        sc, _ac, _sync_f, _async_f = install(
            lambda: _ShapeMgr(_ShapeStream(), span_at="construct",
                               scope="opentelemetry.instrumentation.anthropic_v2",
                               name="anthropic.chat"))

        with sc.stream(**KW) as stream:
            list(stream)

        llm = _llm_rows(env.rows)
        assert len(llm) == 1
        assert llm[0]["input_tokens"] == 11
        assert llm[0]["output_tokens"] == 7

    def test_e_both_renamed_degrades_to_extra_row_never_loss(self, fakes):
        """CHARACTERISATION TEST — deliberate: when BOTH identifiers are
        renamed away, `claim_anthropic_stream_span` genuinely cannot
        recognize the span (by design: never suppress on doubt), so it is
        NOT suppressed and lands as a second row. This degrades to an EXTRA
        row (never a LOST one) — do not "fix" this into a loss by trying to
        make the matcher broader/fuzzier; an extra row is the documented,
        accepted trade-off (see claim_anthropic_stream_span's docstring)."""
        install, env = fakes
        sc, _ac, _sync_f, _async_f = install(
            lambda: _ShapeMgr(_ShapeStream(), span_at="construct",
                               scope="vendor.anthropic.rewrite",
                               name="vendor_anthropic_call",
                               dud_tokens=(11, 8)))

        with sc.stream(**KW) as stream:
            list(stream)

        llm = _llm_rows(env.rows)
        assert len(llm) == 2
        # Manual `_finalize` row: real usage.
        assert any(r["input_tokens"] == 11 and r["output_tokens"] == 7 for r in llm)
        # Un-suppressed dud row: never lost, landed as an extra row.
        assert any(r["input_tokens"] == 11 and r["output_tokens"] == 8 for r in llm)

        # No leaked state: a SUBSEQUENT create() call still meters correctly.
        # Restore the REAL anthropic package into sys.modules first — the
        # fake stub installed above stays live until this fixture's teardown
        # otherwise, and a real client built while it's live would resolve
        # `.messages` against the FAKE Messages class instead.
        env.cleanup()
        env.rows.clear()
        env.checks.clear()
        real_sc, _real_ac = _clients(_handler())
        real_sc.messages.create(**KW)
        assert len(env.checks) == 1
        create_rows = _llm_rows(env.rows)
        assert len(create_rows) == 1
        assert create_rows[0]["input_tokens"] == 11
        assert create_rows[0]["output_tokens"] == 7


# ═══════════════════════════════════════════════════════════════════
# Part E (item 6) — golden rule.
# ═══════════════════════════════════════════════════════════════════

class TestGoldenRule:
    def test_hostile_span_set_attribute_raises_call_still_completes(self, fakes):
        """A hostile span whose set_attribute raises must never crash the
        customer's `with` block — it degrades to an extra (un-enriched) row,
        never an exception."""
        install, env = fakes
        sc, _ac, _sync_f, _async_f = install(
            lambda: _ShapeMgr(_ShapeStream(), span_at="construct", hostile=True))

        # Must not raise.
        with sc.stream(**KW) as stream:
            list(stream)

        llm = _llm_rows(env.rows)
        # Manual finalize row + the hostile span's un-suppressed row (claim
        # succeeded internally — record["seen"] incremented — but the
        # set_attribute("tp.suppress", True) call that would have marked it
        # for drop raised and was swallowed, so on_end never sees tp.suppress
        # and logs it as an ordinary, if under-enriched, LLM row).
        assert len(llm) == 2
        assert any(r["input_tokens"] == 11 and r["output_tokens"] == 7 for r in llm)

    def test_hostile_span_on_start_direct_call_never_raises(self):
        """Direct unit proof at the seam: on_start itself must swallow a
        set_attribute failure rather than propagate — this is what makes the
        integration behavior above possible."""
        from token_police.context import (
            arm_anthropic_stream_span_window,
            disarm_anthropic_stream_span_window,
        )
        handle = arm_anthropic_stream_span_window()
        try:
            span = _ShapeSpan(REAL_SCOPE, REAL_NAME, hostile=True)
            TokenPoliceSpanProcessor().on_start(span)  # must not raise
            assert "tp.suppress" not in span.attributes
        finally:
            disarm_anthropic_stream_span_window(handle)

    def test_orig_stream_raising_propagates_by_identity_and_disarms(self, fakes):
        """`orig_stream(...)` raising must propagate the EXACT exception
        object (never wrapped/reclassified) and must leave the suppression
        window fully disarmed — the `finally: disarm(...)` around the W1 arm
        statement is unconditional."""
        install, env = fakes
        boom = RuntimeError("vendor stream() blew up")

        def _raising_factory():
            raise boom

        sc, _ac, _sync_f, _async_f = install(_raising_factory)

        assert _anthropic_stream_span_window.get() is None
        with pytest.raises(RuntimeError) as excinfo:
            sc.stream(**KW)
        assert excinfo.value is boom
        assert _anthropic_stream_span_window.get() is None
        # No telemetry row for a call that never even built a manager.
        assert env.rows == []

    def test_orig_stream_raising_async_twin(self, fakes):
        install, env = fakes
        boom = RuntimeError("async vendor stream() blew up")

        def _raising_factory():
            raise boom

        _sc, ac, _sync_f, _async_f = install(_raising_factory)

        assert _anthropic_stream_span_window.get() is None
        try:
            ac.stream(**KW)
            raised = None
        except RuntimeError as exc:
            raised = exc
        assert raised is boom
        assert _anthropic_stream_span_window.get() is None
        assert env.rows == []

    def test_enforce_denial_raises_and_leaves_no_armed_window(self, fakes, monkeypatch):
        """The pre-flight check runs BEFORE the W1 arm statement — a denial
        raises before anything is ever armed."""
        install, env = fakes
        sc, _ac, sync_f, _async_f = install(
            lambda: _ShapeMgr(_ShapeStream(), span_at="construct"))

        def _blocked(**_kw):
            raise TokenPoliceBlockedError("budget exhausted")

        monkeypatch.setattr(enforcer, "_run_sync_check", _blocked)

        assert _anthropic_stream_span_window.get() is None
        with pytest.raises(TokenPoliceBlockedError):
            sc.stream(**KW)
        assert _anthropic_stream_span_window.get() is None
        # The vendor stream body was never even reached.
        assert sync_f.calls == []
        assert env.rows == []

    def test_enforce_denial_async_twin(self, fakes, monkeypatch):
        install, env = fakes
        _sc, ac, _sync_f, async_f = install(
            lambda: _ShapeMgr(_ShapeStream(), span_at="construct"))

        async def _blocked(**_kw):
            raise TokenPoliceBlockedError("budget exhausted")

        monkeypatch.setattr(enforcer, "_run_async_check", _blocked)

        assert _anthropic_stream_span_window.get() is None

        async def go():
            with pytest.raises(TokenPoliceBlockedError):
                async with ac.stream(**KW):
                    pass
        _run(go())

        assert _anthropic_stream_span_window.get() is None
        assert env.rows == []
