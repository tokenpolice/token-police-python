"""L-2 (Python): a provider rejection on an EAGER (non-pull) framework path must
land exactly ONE failed row — never zero.

Sibling of tests/test_stream_failure_outcomes.py (L-1, which covered the *pull*
sites). The bug here is the single-emitter design: while a framework guard is
held (`_in_langchain` / `_in_llamaindex` / `_in_pydantic_ai`) the raw provider
wrapper short-circuits and emits nothing, on the understanding that the
framework wrapper owns the row. But the framework wrappers' eager failure paths
re-raised without emitting, so a 401/429 through LangChain / LangGraph /
LlamaIndex / pydantic_ai produced ZERO llm rows — the call simply vanished from
the customer's FinOps view.

Eight seams are pinned here:

  * ``lc_sync`` / ``lc_async``  (``_set_langchain_wrapper``) — also the ONLY
    seam LangGraph traffic crosses (LangGraph has no instrumentation of its
    own, so every LangGraph LLM call is a LangChain invoke/ainvoke).
  * ``li_sync`` / ``li_async`` non-stream (``_set_llamaindex_wrapper``).
  * ``li_stream`` construction failure (the provider raised before a stream
    object existed) → the manual ``_log_li_py`` row, mirroring the async twin's
    zero-pull path.
  * ``_make_pydantic_ai_request_wrapper`` (non-stream request).
  * ``_PydanticAIAsyncStreamMgr.__aenter__`` (vendor manager rejection — Python
    skips ``__aexit__`` when enter raises) and ``__aexit__``/``_finalize``
    (failure inside the customer's ``async with`` body).
  * ``_guard_async_stream`` construction failure (L-1 residual: the flushed row
    was success-defaulted because no outcome was stamped).

Two shapes recur on the LC/LI seams and both are covered:
  - deferred buffer NON-empty → the instrumentor's ERROR span is flushed as the
    canonical failed row (the outcome rides on the first flushed payload);
  - deferred buffer EMPTY → a synthetic failed row via
    ``_stash_attempt_context`` + ``_emit_call_failure_log``, whose model must be
    the check-context hint (the model lives on the framework INSTANCE, never in
    kwargs — without the synthetic kwargs the row reads ``model="unknown"``).

GOLDEN RULE: every seam re-raises the customer's ORIGINAL exception by identity
— asserted with ``is``, not by type — and restores its guard/defer state even
when the emission itself explodes.

All fakes — no langchain / llama_index / pydantic_ai installed. Fixtures mirror
tests/test_stream_failure_outcomes.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

# Make the local (worktree) SDK importable ahead of any editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import (
    TPSession,
    _current_session,
    _in_langchain,
    _in_llamaindex,
    _in_pydantic_ai,
)


class _FakeClient:
    """Captures every log_sync payload (deferred-span flushes, the
    `_emit_call_failure_log` synthetic row, and both manual framework logs all
    land here)."""

    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Pin one session, neutralise both pre-flight checks, and intercept every
    logged payload by swapping the enforcer's client resolver."""
    sess = TPSession(
        user_id="usr_1",
        paid_plan="pro",
        workflow_name="wf",
        trace_id="a" * 32,
        root_span_id="b" * 16,
    )
    sess._deferred_spans = []
    sess._defer_telemetry = False
    sess._call_outcome = None
    tok = _current_session.set(sess)
    lc_guard = _in_langchain.set(False)
    li_guard = _in_llamaindex.set(False)
    pa_guard = _in_pydantic_ai.set(False)

    client = _FakeClient()
    monkeypatch.setattr(enforcer, "_run_sync_check", lambda *a, **k: None)

    async def _noop_async_check(*a, **k):
        return None

    monkeypatch.setattr(enforcer, "_run_async_check", _noop_async_check)
    monkeypatch.setattr(enforcer, "get_client", lambda: client)
    try:
        yield sess, client
    finally:
        for var, token in ((_in_langchain, lc_guard),
                           (_in_llamaindex, li_guard),
                           (_in_pydantic_ai, pa_guard)):
            try:
                var.reset(token)
            except Exception:
                var.set(False)
        _current_session.reset(tok)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _deferred(tag, order=0):
    """A minimally-valid deferred-span payload (the shape telemetry.py queues):
    `**`-expandable into log_sync with a `span` sub-dict."""
    return {"tp_tag": tag, "span": {"trace_id": "a" * 32, "span_order": order}}


def _only_call(client):
    assert len(client.calls) == 1, f"expected exactly one log, got {len(client.calls)}"
    return client.calls[0]


def _seam(method):
    """The enforcer function object actually installed on the class.
    ``functools.wraps`` copies ``__name__``/``__qualname__`` from the ORIGINAL,
    so only ``__code__.co_name`` still names the wrapper — the same path-pinning
    trick tests/test_stream_failure_outcomes.py uses on async generators. Lets a
    test prove it exercised (say) ``lc_sync`` and not some other kind."""
    return getattr(getattr(method, "__code__", None), "co_name", "")


class _Auth401(Exception):
    """OpenAI/Anthropic-SDK-shaped 401 — `.status_code` is what the classifier
    reads first (see token_police/_classify.py::_direct_status_of)."""

    status_code = 401


def _auth401():
    return _Auth401("401 Incorrect API key provided")


def _assert_failed_401(outcome):
    assert outcome is not None, "pre-fix: ZERO rows for the call (L-2) — nothing to classify"
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "auth_error"
    assert outcome["http_status"] == 401
    assert isinstance(outcome["duration_ms"], int)


# ═══════════════════════════════════════════════════════════════════
# LangChain eager path — lc_sync / lc_async.
# Covers LangGraph too: LangGraph ships no instrumentation of its own, so every
# LangGraph node's LLM call arrives here as a LangChain invoke/ainvoke.
# ═══════════════════════════════════════════════════════════════════


def _install_lc_sync(invoke_fn, model="gpt-4o-mini"):
    """`FakeChatOpenAI` is named so `_lc_provider_from_instance` resolves
    "openai" off the class-name needle table (the fake's module is this test
    file, so the module table can't hit)."""
    cls = type("FakeChatOpenAI", (), {"invoke": invoke_fn, "model": model})
    enforcer._set_langchain_wrapper(cls, "invoke", cls.invoke, "sync")
    return cls


def _install_lc_async(ainvoke_fn, model="gpt-4o-mini"):
    cls = type("FakeChatOpenAI", (), {"ainvoke": ainvoke_fn, "model": model})
    enforcer._set_langchain_wrapper(cls, "ainvoke", cls.ainvoke, "async")
    return cls


# ── 1. deferred ERROR span present → it flushes as the failed row ──
def test_lc_sync_failure_flushes_deferred_span_as_failed_row(_env):
    """The instrumentor queued an ERROR span before the rejection surfaced: that
    span IS the call's row, and it must carry the failed outcome."""
    sess, client = _env
    boom = _auth401()

    def invoke(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("ERR"))
        raise boom

    cls = _install_lc_sync(invoke)
    assert _seam(cls.invoke) == "lc_sync"          # pin the seam under test

    with pytest.raises(_Auth401) as ei:
        cls().invoke("hi")
    assert ei.value is boom                        # identity re-raise (golden rule)

    payload = _only_call(client)
    assert payload["tp_tag"] == "ERR"
    _assert_failed_401(payload.get("call_outcome"))

    # State fully restored + buffer consumed at failure time.
    assert sess._deferred_spans == []
    assert getattr(sess, "_call_outcome", None) is None
    assert sess._defer_telemetry is False
    assert enforcer.in_langchain() is False


# ── 2. EMPTY deferred buffer → synthetic failed row with the model HINT ──
def test_lc_sync_failure_with_empty_buffer_emits_synthetic_row(_env):
    """Rejection before the instrumentor ever ran (the common 401 shape): no
    span was queued, so the wrapper must synthesise the row — and it must carry
    the check-context model/provider, not model="unknown"."""
    sess, client = _env
    boom = _auth401()

    def invoke(self, prompt="p", *a, **k):
        raise boom

    cls = _install_lc_sync(invoke, model="gpt-4o-mini")

    with pytest.raises(_Auth401) as ei:
        cls().invoke("hi")
    assert ei.value is boom

    payload = _only_call(client)
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["model"] == "gpt-4o-mini", "the instance model hint, not 'unknown'"
    assert payload["provider"] == "openai"
    assert payload["operation"] == "chat"
    # A failed call has no usage — the row must stay unmeasured.
    assert (payload.get("usage") or {}).get("raw") == {"prompt_tokens": 0, "total_tokens": 0}

    assert getattr(sess, "_call_outcome", None) is None
    assert sess._defer_telemetry is False
    assert enforcer.in_langchain() is False


def test_lc_sync_failure_classification_is_not_hardcoded(_env):
    """The outcome is classified from the real exception, not stamped 401."""
    _sess, client = _env
    boom = RuntimeError("429 slow down")
    boom.status_code = 429

    def invoke(self, prompt="p", *a, **k):
        raise boom

    with pytest.raises(RuntimeError) as ei:
        _install_lc_sync(invoke)().invoke("hi")
    assert ei.value is boom

    outcome = _only_call(client)["call_outcome"]
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "rate_limited"
    assert outcome["http_status"] == 429


# ── 3. async twin ──
def test_lc_async_failure_flushes_deferred_span_as_failed_row(_env):
    sess, client = _env
    boom = _auth401()

    async def ainvoke(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("ERR"))
        raise boom

    cls = _install_lc_async(ainvoke)
    assert _seam(cls.ainvoke) == "lc_async"

    async def go():
        with pytest.raises(_Auth401) as ei:
            await cls().ainvoke("hi")
        assert ei.value is boom
        # Asserted INSIDE the task: contextvar writes made in a Task do not
        # leak back to the caller's context, so an outside check is vacuous.
        assert enforcer.in_langchain() is False

    _run(go())

    payload = _only_call(client)
    assert payload["tp_tag"] == "ERR"
    _assert_failed_401(payload.get("call_outcome"))
    assert sess._deferred_spans == []
    assert getattr(sess, "_call_outcome", None) is None
    assert sess._defer_telemetry is False


def test_lc_async_failure_with_empty_buffer_emits_synthetic_row(_env):
    sess, client = _env
    boom = _auth401()

    async def ainvoke(self, prompt="p", *a, **k):
        raise boom

    cls = _install_lc_async(ainvoke, model="claude-haiku-4-5")

    async def go():
        with pytest.raises(_Auth401) as ei:
            await cls().ainvoke("hi")
        assert ei.value is boom

    _run(go())

    payload = _only_call(client)
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["model"] == "claude-haiku-4-5"
    assert payload["provider"] == "openai"     # class-name needle, not the model id
    assert getattr(sess, "_call_outcome", None) is None


# ── 4. stale-outcome regression ──
def test_lc_failure_does_not_poison_the_next_call(_env):
    """A leftover `session._call_outcome` would mislabel the session's NEXT row
    as failed. First half (a row exists at all) is red pre-fix; the second half
    (the next row is clean) is a control that the new stamp is always cleared."""
    sess, client = _env
    boom = _auth401()

    def bad_invoke(self, prompt="p", *a, **k):
        raise boom

    with pytest.raises(_Auth401):
        _install_lc_sync(bad_invoke)().invoke("hi")

    assert len(client.calls) == 1
    assert client.calls[0]["call_outcome"]["status"] == "failed"
    assert getattr(sess, "_call_outcome", None) is None, \
        "the failed outcome must not survive the call that produced it"

    def ok_invoke(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("OK"))
        return "response"

    assert _install_lc_sync(ok_invoke)().invoke("hi") == "response"

    assert len(client.calls) == 2
    assert client.calls[1]["tp_tag"] == "OK"
    assert client.calls[1].get("call_outcome") is None, \
        "a healthy call must never inherit the previous call's failure"


# ── 5. ghost-row regression: the stranded ERROR span ──
def test_lc_failure_consumes_the_buffer_and_leaves_no_ghost_row(_env):
    """Pre-fix the ERROR span stayed stranded in `_deferred_spans` (the flush
    after the call never ran) and rode out on the session's NEXT successful
    call — a success-defaulted GHOST row attributed to the wrong call. The
    failure must consume the buffer at failure time."""
    sess, client = _env
    boom = _auth401()

    def bad_invoke(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("ERR"))
        raise boom

    with pytest.raises(_Auth401):
        _install_lc_sync(bad_invoke)().invoke("hi")

    assert len(client.calls) == 1 and client.calls[0]["tp_tag"] == "ERR"
    assert sess._deferred_spans == [], "the failure owns (and drains) its own span"

    def ok_invoke(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("OK"))
        return "response"

    _install_lc_sync(ok_invoke)().invoke("hi")

    assert len(client.calls) == 2, "the next call must emit ONLY its own row"
    assert client.calls[1]["tp_tag"] == "OK"
    assert client.calls[1].get("call_outcome") is None


# ── 6. nested / single-emission: an outer defer window owns the emission ──
def test_lc_sync_failure_under_outer_defer_emits_nothing(_env):
    """Control on the single-emitter contract: at a non-outermost defer level
    the inner wrapper must stay silent (buffer + outcome untouched) so the
    OUTERMOST owner emits exactly once. Only the identity re-raise is
    guaranteed here."""
    sess, client = _env
    sess._defer_telemetry = True                   # simulate an outer defer window
    boom = _auth401()

    def invoke(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("ERR"))
        raise boom

    with pytest.raises(_Auth401) as ei:
        _install_lc_sync(invoke)().invoke("hi")
    assert ei.value is boom

    assert client.calls == [], "inner defer level must not emit — the outer owner does"
    assert len(sess._deferred_spans) == 1, "the queued span is left for the outer flush"
    assert getattr(sess, "_call_outcome", None) is None
    assert sess._defer_telemetry is True, "the outer defer window is restored intact"
    assert enforcer.in_langchain() is False


# ═══════════════════════════════════════════════════════════════════
# LangChain astream construction failure — _guard_async_stream (L-1 residual)
# ═══════════════════════════════════════════════════════════════════


def test_lc_astream_construction_failure_lands_failed_not_success(_env):
    """`astream()` rejected at CONSTRUCTION (before any pull), so L-1's
    pull-site handler never ran. The outer finalize still flushes the queued
    ERROR span — pre-fix with NO outcome, so the collector stamped it SUCCESS.
    The row must be labeled failed."""
    sess, client = _env
    boom = _auth401()

    def astream(self, prompt="p", *a, **k):
        # NOT a generator: raises when `original(*args, **kwargs)` is called,
        # i.e. at stream construction inside _guard_async_stream.
        sess._deferred_spans.append(_deferred("ERR"))
        raise boom

    cls = type("FakeChatOpenAI", (), {"astream": astream, "model": "gpt-4o-mini"})
    enforcer._set_langchain_wrapper(cls, "astream", cls.astream, "astream")

    async def go():
        with pytest.raises(_Auth401) as ei:
            async for _c in cls().astream("hi"):
                pass
        assert ei.value is boom

    _run(go())

    payload = _only_call(client)
    assert payload["tp_tag"] == "ERR"
    outcome = payload.get("call_outcome")
    assert outcome is not None, "pre-fix: flushed with no outcome → collector stamps SUCCESS"
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "auth_error"
    assert getattr(sess, "_call_outcome", None) is None
    assert getattr(sess, "_lc_stream_depth", 0) == 0


# ═══════════════════════════════════════════════════════════════════
# LlamaIndex eager paths — li_sync / li_async (non-stream) / li_stream
# ═══════════════════════════════════════════════════════════════════


def _install_li_sync(chat_fn, model="gpt-4o-mini"):
    cls = type("FakeLIOpenAI", (), {"chat": chat_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "chat", cls.chat, "sync")
    return cls


def _install_li_async(achat_fn, model="gpt-4o-mini"):
    cls = type("FakeLIOpenAI", (), {"achat": achat_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "achat", cls.achat, "async")
    return cls


def _install_li_stream(stream_chat_fn, model="gpt-4o-mini"):
    cls = type("FakeLIOpenAI", (), {"stream_chat": stream_chat_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")
    return cls


# ── 8. li_sync, EMPTY buffer → synthetic failed row with the model hint ──
def test_li_sync_failure_with_empty_buffer_emits_synthetic_row(_env):
    sess, client = _env
    boom = _auth401()

    def chat(self, messages="hi", *a, **k):
        raise boom

    cls = _install_li_sync(chat)
    assert _seam(cls.chat) == "li_sync"

    with pytest.raises(_Auth401) as ei:
        cls().chat("hi")
    assert ei.value is boom                        # identity re-raise

    payload = _only_call(client)
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["model"] == "gpt-4o-mini", "instance model hint, not 'unknown'"
    assert payload["provider"] == "openai"
    assert payload["operation"] == "chat"

    assert getattr(sess, "_call_outcome", None) is None
    assert sess._defer_telemetry is False
    assert enforcer.in_llamaindex() is False


# ── 9. li_async non-stream, deferred ERROR span → flush path ──
def test_li_async_nonstream_failure_flushes_deferred_span_as_failed_row(_env):
    sess, client = _env
    boom = _auth401()

    async def achat(self, messages="hi", *a, **k):
        sess._deferred_spans.append(_deferred("ERR"))
        raise boom

    cls = _install_li_async(achat)
    assert _seam(cls.achat) == "li_async"

    async def go():
        with pytest.raises(_Auth401) as ei:
            await cls().achat("hi")
        assert ei.value is boom
        # Inside the task — see the LangChain async twin.
        assert enforcer.in_llamaindex() is False

    _run(go())

    payload = _only_call(client)
    assert payload["tp_tag"] == "ERR"
    _assert_failed_401(payload.get("call_outcome"))
    assert sess._deferred_spans == []
    assert getattr(sess, "_call_outcome", None) is None
    assert sess._defer_telemetry is False


def test_li_async_nonstream_failure_under_outer_defer_emits_nothing(_env):
    """Control (single-emitter): nested defer level stays silent."""
    sess, client = _env
    sess._defer_telemetry = True
    boom = _auth401()

    async def achat(self, messages="hi", *a, **k):
        raise boom

    cls = _install_li_async(achat)

    async def go():
        with pytest.raises(_Auth401) as ei:
            await cls().achat("hi")
        assert ei.value is boom

    _run(go())

    assert client.calls == []
    assert getattr(sess, "_call_outcome", None) is None
    assert sess._defer_telemetry is True


# ── 10. li_stream: the provider rejected before a stream object existed ──
def test_li_stream_construction_failure_emits_manual_failed_row(_env):
    """`stream_chat` raised synchronously, so `_guard_sync_stream_li` (which
    owns the manual log) was never constructed — this wrapper is the sole
    emitter. Mirrors the async twin's zero-pull path: model from the instance,
    0/0 tokens, this call's deferred spans dropped (they are never logged)."""
    sess, client = _env
    boom = _auth401()
    order = sess._span_counter                     # the order li_stream will claim
    sess._deferred_spans.append(_deferred("INNER-DUD", order=order))

    def stream_chat(self, messages="hi", *a, **k):
        # NOT a generator — raises at call time, i.e. stream construction.
        raise boom

    cls = _install_li_stream(stream_chat)
    assert _seam(cls.stream_chat) == "li_stream"

    with pytest.raises(_Auth401) as ei:
        cls().stream_chat("hi")
    assert ei.value is boom                        # identity re-raise

    payload = _only_call(client)
    assert payload.get("tp_tag") is None, "the manual row, not a flushed inner span"
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["model"] == "gpt-4o-mini"
    assert payload["provider"] == "openai"
    assert payload["input_tokens"] == 0
    assert payload["output_tokens"] == 0
    assert payload["cached_tokens"] == 0
    assert payload["latency"]["is_streaming"] is True
    assert payload["span"]["span_order"] == order

    assert sess._deferred_spans == [], "this call's inner dud span must be dropped"
    assert sess._defer_telemetry is False


# ═══════════════════════════════════════════════════════════════════
# pydantic_ai — request wrapper + async stream manager
# ═══════════════════════════════════════════════════════════════════


def _pa_model():
    return NS(model_name="claude-haiku-4-5-20251001", system="anthropic")


def _pa_response(input_tokens=803, output_tokens=124):
    return NS(
        model_name="claude-haiku-4-5-20251001",
        provider_name="anthropic",
        usage=NS(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class _PAStream:
    """StreamedResponse stand-in: `.get()` returns whatever the consumer has
    accumulated so far (the partial response a mid-stream failure must keep)."""

    def __init__(self, response):
        self._response = response

    def get(self):
        return self._response


def _pa_mgr(stream=None, enter_exc=None):
    """`_PydanticAIAsyncStreamMgr` over a vendor manager that either yields
    `stream` or rejects inside `__aenter__` (where the provider HTTP request
    actually fires)."""

    class _Vendor:
        async def __aenter__(_self):
            if enter_exc is not None:
                raise enter_exc
            return stream

        async def __aexit__(_self, *a):
            return False

    return enforcer._PydanticAIAsyncStreamMgr(_Vendor(), _pa_model(), messages=[])


# ── 11. non-stream request wrapper ──
def test_pa_request_failure_emits_failed_row(_env):
    _sess, client = _env
    boom = _auth401()

    async def original(_self, *a, **k):
        raise boom

    wrapper = enforcer._make_pydantic_ai_request_wrapper(original)

    async def go():
        with pytest.raises(_Auth401) as ei:
            await wrapper(_pa_model())
        assert ei.value is boom                    # identity re-raise
        # The guard must not leak into the customer's error-handling path.
        # Asserted inside the task — contextvar writes don't escape it.
        assert enforcer.in_pydantic_ai() is False

    _run(go())

    payload = _only_call(client)
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["provider"] == "anthropic"
    assert payload["input_tokens"] == 0            # no response → no usage
    assert payload["output_tokens"] == 0
    assert payload["cached_tokens"] == 0
    assert payload["latency"]["is_streaming"] is False


# ── 12. stream manager: vendor __aenter__ rejection ──
def test_pa_stream_enter_failure_emits_failed_row(_env):
    """Python skips `__aexit__` when `__aenter__` raises, so without an
    emission here the rejection produced ZERO rows."""
    _sess, client = _env
    boom = _auth401()
    mgr = _pa_mgr(enter_exc=boom)

    async def go():
        with pytest.raises(_Auth401) as ei:
            async with mgr:
                pytest.fail("body must never run")  # pragma: no cover
        assert ei.value is boom                    # identity re-raise
        assert enforcer.in_pydantic_ai() is False  # inside the task

    _run(go())

    payload = _only_call(client)
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["provider"] == "anthropic"
    assert payload["input_tokens"] == 0
    assert payload["output_tokens"] == 0


# ── 13. stream manager: failure inside the customer's `async with` body ──
def test_pa_stream_body_failure_emits_failed_row_with_partial_usage(_env):
    """The provider billed whatever tokens already streamed, so the failed row
    keeps them (`self._stream.get()`), and the exception is NEVER suppressed."""
    _sess, client = _env
    boom = RuntimeError("503 upstream died")
    boom.status_code = 503
    mgr = _pa_mgr(stream=_PAStream(_pa_response(input_tokens=803, output_tokens=124)))

    async def go():
        with pytest.raises(RuntimeError) as ei:
            async with mgr:
                raise boom
        assert ei.value is boom                    # propagated, not swallowed
        assert enforcer.in_pydantic_ai() is False  # inside the task

    _run(go())

    payload = _only_call(client)
    outcome = payload.get("call_outcome")
    assert outcome is not None, "pre-fix: an errored stream logged NO row at all"
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "server_error"
    assert outcome["http_status"] == 503
    assert payload["input_tokens"] == 803, "partial usage the provider already billed"
    assert payload["output_tokens"] == 124
    # A broken stream carries no latency (matches the success path's placement).
    assert payload.get("latency") is None


def test_pa_stream_body_failure_zero_pull_lands_zero_usage(_env):
    """Nothing streamed before the failure → an honest 0/0 failed row (the
    guarded `.get()` returning None must not become a crash or a phantom)."""
    _sess, client = _env
    boom = _auth401()

    class _EmptyStream:
        def get(self):
            raise RuntimeError("nothing accumulated")

    mgr = _pa_mgr(stream=_EmptyStream())

    async def go():
        with pytest.raises(_Auth401) as ei:
            async with mgr:
                raise boom
        assert ei.value is boom

    _run(go())

    payload = _only_call(client)
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["input_tokens"] == 0
    assert payload["output_tokens"] == 0


# ── 14. BaseException teardown keeps today's no-row behavior (control) ──
def test_pa_stream_body_cancellation_logs_no_row(_env):
    """Control: `CancelledError` is a BaseException — task teardown, not a
    provider failure. It must keep today's behavior (no row) and propagate
    untouched, or every cancelled agent run would invent a failed call."""
    _sess, client = _env
    mgr = _pa_mgr(stream=_PAStream(_pa_response()))

    async def go():
        with pytest.raises(asyncio.CancelledError):
            async with mgr:
                raise asyncio.CancelledError()

    _run(go())

    assert client.calls == [], "cancellation must not manufacture a failed row"


def test_pa_stream_body_generator_exit_logs_no_row(_env):
    """Control: same contract for GeneratorExit."""
    _sess, client = _env
    mgr = _pa_mgr(stream=_PAStream(_pa_response()))

    async def go():
        with pytest.raises(GeneratorExit):
            async with mgr:
                raise GeneratorExit()

    _run(go())

    assert client.calls == []


# ── 15. _log_pydantic_ai back-compat ──
def test_log_pydantic_ai_without_call_outcome_is_unchanged(_env):
    """Control: the new `call_outcome` kwarg defaults to None, and the client
    treats an explicit None exactly like an omitted key — so every pre-existing
    success caller's payload is byte-identical."""
    sess, client = _env
    from datetime import datetime, timezone

    enforcer._log_pydantic_ai(
        _pa_model(), sess, [], _pa_response(), 0, "span",
        datetime.now(timezone.utc),
    )

    payload = _only_call(client)
    # `.get()` — absent (pre-fix) and explicit-None (post-fix) are equivalent on
    # the wire, so this control holds on BOTH sides of the change.
    assert payload.get("call_outcome") is None
    assert payload["model"] == "claude-haiku-4-5-20251001"
    assert payload["provider"] == "anthropic"
    assert payload["input_tokens"] == 803
    assert payload["output_tokens"] == 124


def test_pa_stream_healthy_drain_still_logs_success(_env):
    """Control: the happy path is untouched — one row, no call_outcome."""
    _sess, client = _env
    mgr = _pa_mgr(stream=_PAStream(_pa_response()))

    async def go():
        async with mgr as s:
            assert s.get() is not None

    _run(go())

    payload = _only_call(client)
    assert payload.get("call_outcome") is None
    assert payload["latency"]["is_streaming"] is True
    assert payload["input_tokens"] == 803


# ═══════════════════════════════════════════════════════════════════
# GOLDEN RULE: the new emission can never reach the customer.
# The SDK may lose a row; it may NEVER change what the customer's app sees.
# ═══════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("victim", ["build_call_outcome", "_emit_call_failure_log"])
def test_lc_failure_emission_explosion_is_failopen(_env, monkeypatch, victim):
    """Fault-injection on the LangChain seam: whichever piece of the new
    emission blows up, the customer still receives the ORIGINAL exception by
    identity and the guard/defer state is fully restored."""
    sess, client = _env

    def _explode(*a, **k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(enforcer, victim, _explode)
    boom = _auth401()

    def invoke(self, prompt="p", *a, **k):
        raise boom

    with pytest.raises(_Auth401) as ei:
        _install_lc_sync(invoke)().invoke("hi")
    assert ei.value is boom, "the customer must see their provider error, not ours"

    assert enforcer.in_langchain() is False
    assert sess._defer_telemetry is False
    assert client.calls == []


def test_li_failure_emission_explosion_is_failopen(_env, monkeypatch):
    """Fault-injection on the LlamaIndex seam — same contract."""
    sess, _client = _env

    def _explode(*a, **k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(enforcer, "build_call_outcome", _explode)
    boom = _auth401()

    def chat(self, messages="hi", *a, **k):
        raise boom

    with pytest.raises(_Auth401) as ei:
        _install_li_sync(chat)().chat("hi")
    assert ei.value is boom

    assert enforcer.in_llamaindex() is False
    assert sess._defer_telemetry is False


def test_li_stream_failure_emission_explosion_is_failopen(_env, monkeypatch):
    """Fault-injection on li_stream's manual log — same contract."""
    sess, _client = _env

    def _explode(*a, **k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(enforcer, "_log_li_py", _explode)
    boom = _auth401()

    def stream_chat(self, messages="hi", *a, **k):
        raise boom

    with pytest.raises(_Auth401) as ei:
        _install_li_stream(stream_chat)().stream_chat("hi")
    assert ei.value is boom

    assert sess._defer_telemetry is False


@pytest.mark.parametrize("victim", ["build_call_outcome", "_log_pydantic_ai"])
def test_pa_request_failure_emission_explosion_is_failopen(_env, monkeypatch, victim):
    """Fault-injection on the pydantic_ai request seam — same contract, plus the
    `_in_pydantic_ai` guard must still be released."""
    _sess, _client = _env

    def _explode(*a, **k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(enforcer, victim, _explode)
    boom = _auth401()

    async def original(_self, *a, **k):
        raise boom

    wrapper = enforcer._make_pydantic_ai_request_wrapper(original)

    async def go():
        with pytest.raises(_Auth401) as ei:
            await wrapper(_pa_model())
        assert ei.value is boom
        assert enforcer.in_pydantic_ai() is False  # inside the task

    _run(go())


def test_pa_stream_enter_failure_emission_explosion_is_failopen(_env, monkeypatch):
    """Fault-injection on the stream-manager enter seam — same contract."""
    _sess, _client = _env

    def _explode(*a, **k):
        raise RuntimeError("telemetry exploded")

    monkeypatch.setattr(enforcer, "_log_pydantic_ai", _explode)
    boom = _auth401()
    mgr = _pa_mgr(enter_exc=boom)

    async def go():
        with pytest.raises(_Auth401) as ei:
            async with mgr:
                pass  # pragma: no cover
        assert ei.value is boom
        assert enforcer.in_pydantic_ai() is False  # inside the task

    _run(go())
