"""L-1 (Python): a streamed provider rejection through a framework wrapper must
land a FAILED row — never a success-defaulted one.

Two independent halves of the same bug:

1. **LangChain streamed** (`_guard_sync_stream` / `_guard_async_stream`). The
   request only fires on iteration, so a 401/400/429 surfaces on a *pull*. The
   guards had no handler there, so `session._call_outcome` was never stamped and
   the deferred ERROR span the LC callback queued flushed with NO
   ``call_outcome`` — the collector then stamps success. Fix: an
   ``except Exception`` on the pull (after StopIteration/StopAsyncIteration)
   stamps ``build_call_outcome(exc, elapsed)`` and re-raises bare, plus a
   guarded stale-outcome clear in the outermost finally — needed because
   ``_flush_deferred_spans`` early-returns WITHOUT clearing when the deferred
   buffer is empty, which would poison the session's next row.

2. **LlamaIndex streamed** (`_guard_sync_stream_li` / li_async's inner
   ``_drain()``). These log MANUALLY in their finally (the
   inner instrumentor span is dropped), so a rejection produced a synthetic
   success row — priced, if usage-bearing chunks arrived first. Fix: an
   ``except Exception`` builds a LOCAL ``_fail_outcome`` and the finally threads
   it into ``_log_li_py(call_outcome=...)``.

GOLDEN RULE: both handlers re-raise the customer's ORIGINAL exception by
identity — asserted with ``is``, not just by type.

All fakes — no langchain / llama_index installed. Fixtures mirror
tests/test_sec06_langchain_interleaved.py and
tests/test_llamaindex_stream_latency.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

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
)


class _FakeClient:
    """Captures every log_sync payload (both the deferred-span flush and the
    LlamaIndex manual log land here)."""

    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Pin one session, neutralise both pre-flight checks, and intercept every
    logged payload by swapping the enforcer's client resolver."""
    sess = TPSession(trace_id="a" * 32, root_span_id="b" * 16)
    sess._deferred_spans = []
    sess._defer_telemetry = False
    sess._call_outcome = None
    tok = _current_session.set(sess)
    lc_guard = _in_langchain.set(False)
    li_guard = _in_llamaindex.set(False)

    client = _FakeClient()
    monkeypatch.setattr(enforcer, "_run_sync_check", lambda *a, **k: None)

    async def _noop_async_check(*a, **k):
        return None

    monkeypatch.setattr(enforcer, "_run_async_check", _noop_async_check)
    monkeypatch.setattr(enforcer, "get_client", lambda: client)
    try:
        yield sess, client
    finally:
        for var, token in ((_in_langchain, lc_guard), (_in_llamaindex, li_guard)):
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


def _spy_flush(monkeypatch):
    """Wrap the REAL `_flush_deferred_spans`, recording `session._call_outcome`
    as the flush SEES it. Lets a test distinguish "the outcome was never
    stamped" from "it was stamped and then correctly cleared" — without it, the
    stale-clear assertions would pass vacuously on code that never stamps.
    `_finalize_langchain_stream` reads it as a module global, so the fix
    observes the spy too. Signature-transparent so a new kwarg can't TypeError
    into a fail-open caller and silently swallow the flush."""
    real = enforcer._flush_deferred_spans
    seen = []

    def wrapper(session, **kw):
        seen.append(getattr(session, "_call_outcome", None))
        return real(session, **kw)

    monkeypatch.setattr(enforcer, "_flush_deferred_spans", wrapper)
    return seen


class _Auth401(Exception):
    """OpenAI/Anthropic-SDK-shaped 401 — `.status_code` is what the classifier
    reads first (see token_police/_classify.py::_direct_status_of)."""

    status_code = 401


def _auth401():
    return _Auth401("401 Incorrect API key provided")


# ═══════════════════════════════════════════════════════════════════
# LangChain — _guard_sync_stream
# ═══════════════════════════════════════════════════════════════════


def _install_lc_sync(stream_fn):
    cls = type("FakeLCSync", (), {"stream": stream_fn})
    enforcer._set_langchain_wrapper(cls, "stream", cls.stream, "stream")
    return cls


def _install_lc_astream(astream_fn):
    cls = type("FakeLCAsync", (), {"astream": astream_fn})
    enforcer._set_langchain_wrapper(cls, "astream", cls.astream, "astream")
    return cls


def _lc_failing_sync(err, sess=None, chunks=(), queue_span=True):
    """`stream` that yields `chunks`, then (like the OTel callback) queues an
    ERROR span payload and raises `err` from the pull."""

    def stream(self, prompt="p", *a, **k):
        for c in chunks:
            yield c
        if queue_span and sess is not None:
            sess._deferred_spans.append(_deferred("ERR"))
        raise err

    return _install_lc_sync(stream)


# ── 1. the deferred ERROR span flushes WITH the failed outcome ──
def test_sync_stream_failure_attaches_call_outcome(_env):
    sess, client = _env
    boom = _auth401()
    model = _lc_failing_sync(boom, sess=sess, chunks=("A-0",))()

    got = []
    with pytest.raises(_Auth401) as ei:
        for c in model.stream("A"):
            got.append(c)

    assert ei.value is boom              # identity re-raise (golden rule)
    assert got == ["A-0"]                # chunks before the failure delivered

    assert len(client.calls) == 1, "the queued ERROR span must flush exactly once"
    payload = client.calls[0]
    assert payload["tp_tag"] == "ERR"
    outcome = payload.get("call_outcome")
    assert outcome is not None, "pre-fix: no call_outcome → collector stamps success"
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "auth_error"
    assert outcome["http_status"] == 401
    assert isinstance(outcome["duration_ms"], int)

    # Consumed by the flush — never left to mislabel the next row.
    assert getattr(sess, "_call_outcome", None) is None
    assert sess._deferred_spans == []
    assert sess._defer_telemetry is False
    assert getattr(sess, "_lc_stream_depth", 0) == 0


def test_sync_stream_failure_classification_is_not_hardcoded(_env):
    sess, client = _env
    boom = RuntimeError("503 Service Unavailable")
    boom.status_code = 503
    model = _lc_failing_sync(boom, sess=sess)()

    with pytest.raises(RuntimeError) as ei:
        list(model.stream("A"))
    assert ei.value is boom

    outcome = client.calls[0]["call_outcome"]
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "server_error"
    assert outcome["http_status"] == 503


# ── 2. stale-clear regression: EMPTY deferred buffer ──
def test_sync_stream_failure_with_empty_buffer_clears_outcome(_env, monkeypatch):
    """`_flush_deferred_spans` early-returns without clearing when the buffer is
    empty (the ERROR span was gate-dropped). Without the new guarded clear the
    failed outcome survives and mislabels the session's NEXT row.

    The flush spy is what makes this non-vacuous: it pins that the outcome WAS
    stamped (so the test fails on code that never stamps) and that the flush
    saw it (so the only way it ends up None is the guarded clear)."""
    sess, client = _env
    seen = _spy_flush(monkeypatch)
    boom = _auth401()
    model = _lc_failing_sync(boom, sess=sess, queue_span=False)()

    with pytest.raises(_Auth401) as ei:
        list(model.stream("A"))
    assert ei.value is boom

    assert client.calls == []                       # nothing to flush
    assert len(seen) == 1 and seen[0] is not None   # the outcome WAS stamped …
    assert seen[0]["status"] == "failed"            # … and the flush saw it …
    assert getattr(sess, "_call_outcome", None) is None  # … and cleared after


def test_stale_outcome_would_poison_the_next_row(_env, monkeypatch):
    """The damage the clear prevents, spelled out: a later, unrelated deferred
    span must NOT inherit the previous call's failed outcome."""
    sess, client = _env
    seen = _spy_flush(monkeypatch)
    boom = _auth401()
    failing = _lc_failing_sync(boom, sess=sess, queue_span=False)()
    with pytest.raises(_Auth401):
        list(failing.stream("A"))
    # The first stream really did stamp a failed outcome (empty buffer → the
    # flush had nothing to clear it with).
    assert seen == [seen[0]] and seen[0] is not None and seen[0]["status"] == "failed"

    def ok_stream(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("OK"))
        yield "ok-0"

    ok = _install_lc_sync(ok_stream)()
    assert list(ok.stream("B")) == ["ok-0"]

    assert len(client.calls) == 1
    assert client.calls[0]["tp_tag"] == "OK"
    assert client.calls[0].get("call_outcome") is None


# ── 3. async twin ──
def test_async_stream_failure_attaches_call_outcome(_env):
    sess, client = _env
    boom = _auth401()

    async def astream(self, prompt="p", *a, **k):
        yield "A-0"
        sess._deferred_spans.append(_deferred("ERR"))
        raise boom

    model = _install_lc_astream(astream)()

    async def go():
        got = []
        with pytest.raises(_Auth401) as ei:
            async for c in model.astream("A"):
                got.append(c)
        assert ei.value is boom
        return got

    assert _run(go()) == ["A-0"]

    assert len(client.calls) == 1
    outcome = client.calls[0]["call_outcome"]
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "auth_error"
    assert outcome["http_status"] == 401
    assert getattr(sess, "_call_outcome", None) is None
    assert sess._deferred_spans == []
    assert getattr(sess, "_lc_stream_depth", 0) == 0


def test_async_stream_failure_with_empty_buffer_clears_outcome(_env):
    sess, client = _env
    boom = _auth401()

    async def astream(self, prompt="p", *a, **k):
        raise boom
        yield  # pragma: no cover — makes this an async generator

    model = _install_lc_astream(astream)()

    async def go():
        with pytest.raises(_Auth401) as ei:
            async for _c in model.astream("A"):
                pass
        assert ei.value is boom

    _run(go())
    assert client.calls == []
    assert getattr(sess, "_call_outcome", None) is None


# ── 4. consumer break is NOT a failure (GeneratorExit stays untouched) ──
def test_sync_stream_consumer_break_records_no_outcome(_env):
    sess, client = _env

    def stream(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("PARTIAL"))
        for i in range(3):
            yield f"{prompt}-{i}"

    model = _install_lc_sync(stream)()
    g = model.stream("A")
    assert next(g) == "A-0"
    g.close()                      # GeneratorExit — BaseException, not caught

    assert len(client.calls) == 1
    assert client.calls[0]["tp_tag"] == "PARTIAL"
    assert client.calls[0].get("call_outcome") is None
    assert getattr(sess, "_call_outcome", None) is None


# ── 5. nested depth: the outcome survives the inner unwind ──
def test_nested_inner_failure_survives_to_the_outermost_flush(_env):
    """Two guard levels open. The inner stream fails; its finally runs at
    depth>0, so it must NOT clear the outcome (the flush hasn't happened yet) —
    the outermost drain flushes it onto the payload and clears it there."""
    sess, client = _env
    boom = _auth401()

    def outer_stream(self, prompt="p", *a, **k):
        for i in range(2):
            yield f"OUT-{i}"

    def inner_stream(self, prompt="p", *a, **k):
        sess._deferred_spans.append(_deferred("INNER-ERR"))
        raise boom
        yield  # pragma: no cover — makes this a generator

    outer = _install_lc_sync(outer_stream)()
    inner = _install_lc_sync(inner_stream)()

    g_out = outer.stream("O")
    assert next(g_out) == "OUT-0"                       # depth 0 → 1
    assert getattr(sess, "_lc_stream_depth", 0) == 1

    g_in = inner.stream("I")
    with pytest.raises(_Auth401) as ei:
        next(g_in)                                      # depth 1 → 2, then fails
    assert ei.value is boom

    # Inner unwound to depth 1: no flush yet, so the outcome MUST still be here.
    assert getattr(sess, "_lc_stream_depth", 0) == 1
    assert sess._call_outcome is not None
    assert sess._call_outcome["status"] == "failed"
    assert client.calls == []

    for _ in g_out:                                     # outermost → flush once
        pass

    assert len(client.calls) == 1
    assert client.calls[0]["tp_tag"] == "INNER-ERR"
    assert client.calls[0]["call_outcome"]["error_kind"] == "auth_error"
    assert getattr(sess, "_call_outcome", None) is None
    assert getattr(sess, "_lc_stream_depth", 0) == 0


# ═══════════════════════════════════════════════════════════════════
# LlamaIndex — _guard_sync_stream_li / li_async's inner _drain()
# ═══════════════════════════════════════════════════════════════════


class _LIChunk:
    """LlamaIndex ChatResponse-shaped chunk. Loose token ints on
    `additional_kwargs` is exactly where LlamaIndex's OpenAI extractor puts
    them, so this is what `_extract_li_usage_py` reads."""

    def __init__(self, delta="x", prompt_tokens=0, completion_tokens=0):
        self.delta = delta
        self.additional_kwargs = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        self.message = type("_Msg", (), {"additional_kwargs": {}})()
        self.raw = None


def _install_li_sync(stream_fn, model="gpt-4o-mini"):
    cls = type("FakeLI", (), {"stream": stream_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "stream", cls.stream, "stream")
    return cls


def _install_li_async_chat(astream_chat_fn, model="gpt-4o-mini"):
    """The REGISTERED async shape. Every `astream_chat` target in the
    instrumentation table is declared `"llamaindex": "async"` (openai /
    anthropic / google_genai) — real `astream_chat` is `async def`, so the
    live code path is `li_async`, which awaits the original and drains the
    returned async iterable through its inner `_drain()` generator. (The old
    never-registered kind="astream" guard was deleted — a sync-return wrapper
    would break `await llm.astream_chat(...)`.)"""
    cls = type("FakeLIAsyncChat", (), {"astream_chat": astream_chat_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "astream_chat", cls.astream_chat, "async")
    return cls


def _async_iterable(chunks=(), err=None):
    """What a real `astream_chat` returns: an OBJECT with `__aiter__` (not an
    async-generator function), optionally raising `err` from the iterator."""

    class _AsyncStream:
        def __aiter__(self):
            async def _gen():
                for c in chunks:
                    await asyncio.sleep(0)
                    yield c
                if err is not None:
                    raise err

            return _gen()

    return _AsyncStream()


def _is_li_async_drain(obj):
    """True when `obj` is li_async's inner `_drain()` async generator — pins
    which of the two async LlamaIndex guards a test actually exercised (same
    idiom as tests/test_sec06_langchain_interleaved.py::_is_async_guard)."""
    return getattr(getattr(obj, "ag_code", None), "co_name", "") == "_drain"


def _only_call(client):
    assert len(client.calls) == 1, f"expected exactly one log, got {len(client.calls)}"
    return client.calls[0]


# ── 6. failure at the first pull → one failed row, no success row ──
def test_li_sync_stream_failure_first_pull(_env):
    _sess, client = _env
    boom = _auth401()

    def stream(self, prompt="hi", *a, **k):
        raise boom
        yield  # pragma: no cover — makes this a generator

    cls = _install_li_sync(stream)

    with pytest.raises(_Auth401) as ei:
        list(cls().stream("hi"))
    assert ei.value is boom                      # identity re-raise

    payload = _only_call(client)
    outcome = payload.get("call_outcome")
    assert outcome is not None, "pre-fix: synthetic SUCCESS row for a failed call"
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "auth_error"
    assert outcome["http_status"] == 401
    # Model still comes from the instance; a zero-pull failure has no usage.
    assert payload["model"] == "gpt-4o-mini"
    assert payload["provider"] == "openai"
    assert payload["input_tokens"] == 0
    assert payload["output_tokens"] == 0
    assert payload["cached_tokens"] == 0


# ── 7. mid-stream failure keeps the observed usage AND lands failed ──
def test_li_sync_stream_failure_midstream_keeps_usage(_env):
    """The provider billed the tokens that already streamed, so the row keeps
    them — but pre-fix that made it a fully PRICED success row."""
    _sess, client = _env
    boom = _auth401()

    def stream(self, prompt="hi", *a, **k):
        yield _LIChunk("Hel")
        yield _LIChunk("lo", prompt_tokens=100, completion_tokens=20)
        raise boom

    cls = _install_li_sync(stream)

    got = []
    with pytest.raises(_Auth401) as ei:
        for c in cls().stream("hi"):
            got.append(c)
    assert ei.value is boom
    assert len(got) == 2                          # customer stream untouched

    payload = _only_call(client)
    assert payload["call_outcome"]["status"] == "failed"
    assert payload["input_tokens"] == 100         # observed usage retained
    assert payload["output_tokens"] == 20


# ── 8. consumer break → today's partial success row, no outcome ──
def test_li_sync_stream_consumer_break_partial_success_row(_env):
    _sess, client = _env
    boom = _auth401()

    def stream(self, prompt="hi", *a, **k):
        yield _LIChunk("Hel", prompt_tokens=100, completion_tokens=20)
        yield _LIChunk("lo")
        raise boom                                # never reached

    cls = _install_li_sync(stream)

    g = cls().stream("hi")
    assert next(g).delta == "Hel"
    g.close()                                     # GeneratorExit → finally only

    payload = _only_call(client)
    assert payload.get("call_outcome") is None    # unchanged by the fix
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 20
    assert payload["latency"]["is_streaming"] is True


def test_li_sync_stream_healthy_drain_unchanged(_env):
    """Control: the success payload stays byte-identical (call_outcome absent)."""
    _sess, client = _env

    def stream(self, prompt="hi", *a, **k):
        yield _LIChunk("Hel")
        yield _LIChunk("lo", prompt_tokens=100, completion_tokens=20)

    cls = _install_li_sync(stream)
    assert len(list(cls().stream("hi"))) == 2

    payload = _only_call(client)
    assert payload.get("call_outcome") is None
    assert payload["input_tokens"] == 100


def test_li_sync_stream_failure_is_failopen(_env):
    """GOLDEN RULE: the manual log exploding still surfaces only the provider
    error, and the _in_llamaindex guard is still released."""
    _sess, _client = _env
    boom = _auth401()

    def exploding_log(*a, **k):
        raise RuntimeError("log exploded")

    def stream(self, prompt="hi", *a, **k):
        raise boom
        yield  # pragma: no cover

    cls = _install_li_sync(stream)
    orig_log = enforcer._log_li_py
    enforcer._log_li_py = exploding_log
    try:
        with pytest.raises(_Auth401) as ei:
            list(cls().stream("hi"))
        assert ei.value is boom
    finally:
        enforcer._log_li_py = orig_log

    assert enforcer.in_llamaindex() is False


# ── 9. the REGISTERED async LlamaIndex path: li_async's inner _drain() ──
# `astream_chat` is registered as kind="async", so the generator that actually
# ships is `_drain()` inside li_async.
def test_li_registered_async_stream_failure_first_pull(_env):
    _sess, client = _env
    boom = _auth401()

    async def astream_chat(self, prompt="hi", *a, **k):
        return _async_iterable(err=boom)

    cls = _install_li_async_chat(astream_chat)

    async def go():
        agen = await cls().astream_chat("hi")
        # Pin the path: this MUST be li_async's `_drain` — a green test off any
        # other wrapper shape would be pinning the wrong guard.
        assert _is_li_async_drain(agen)
        got = []
        with pytest.raises(_Auth401) as ei:
            async for c in agen:
                got.append(c)
        assert ei.value is boom                   # identity re-raise
        return got

    assert _run(go()) == []

    payload = _only_call(client)
    outcome = payload.get("call_outcome")
    assert outcome is not None, "pre-fix: synthetic SUCCESS row for a failed call"
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "auth_error"
    assert outcome["http_status"] == 401
    assert payload["model"] == "gpt-4o-mini"
    assert payload["provider"] == "openai"
    assert payload["input_tokens"] == 0
    assert payload["output_tokens"] == 0


def test_li_registered_async_stream_failure_midstream_keeps_usage(_env):
    _sess, client = _env
    boom = RuntimeError("429 slow down")
    boom.status_code = 429

    async def astream_chat(self, prompt="hi", *a, **k):
        return _async_iterable(
            chunks=[
                _LIChunk("Hel"),
                _LIChunk("lo", prompt_tokens=100, completion_tokens=20),
            ],
            err=boom,
        )

    cls = _install_li_async_chat(astream_chat)

    async def go():
        got = []
        with pytest.raises(RuntimeError) as ei:
            async for c in await cls().astream_chat("hi"):
                got.append(c)
        assert ei.value is boom
        return got

    assert len(_run(go())) == 2                   # customer stream untouched

    payload = _only_call(client)
    assert payload["call_outcome"]["status"] == "failed"
    assert payload["call_outcome"]["error_kind"] == "rate_limited"
    assert payload["call_outcome"]["http_status"] == 429
    assert payload["input_tokens"] == 100         # observed usage retained
    assert payload["output_tokens"] == 20


def test_li_registered_async_stream_consumer_break_partial_success_row(_env):
    """Control: aclose() throws GeneratorExit (a BaseException) at the yield, so
    the new `except Exception` never sees it — today's partial row stands."""
    _sess, client = _env
    boom = _auth401()

    async def astream_chat(self, prompt="hi", *a, **k):
        return _async_iterable(
            chunks=[
                _LIChunk("Hel", prompt_tokens=100, completion_tokens=20),
                _LIChunk("lo"),
            ],
            err=boom,                             # never reached
        )

    cls = _install_li_async_chat(astream_chat)

    async def go():
        agen = await cls().astream_chat("hi")
        async for c in agen:
            assert c.delta == "Hel"
            break
        await agen.aclose()                       # explicit, no GC reliance

    _run(go())

    payload = _only_call(client)
    assert payload.get("call_outcome") is None    # unchanged by the fix
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 20
    assert payload["latency"]["is_streaming"] is True


def test_li_registered_async_stream_healthy_drain_unchanged(_env):
    """Control: the success payload stays byte-identical (call_outcome absent)."""
    _sess, client = _env

    async def astream_chat(self, prompt="hi", *a, **k):
        return _async_iterable(
            chunks=[
                _LIChunk("Hel"),
                _LIChunk("lo", prompt_tokens=100, completion_tokens=20),
            ]
        )

    cls = _install_li_async_chat(astream_chat)

    async def go():
        return [c async for c in await cls().astream_chat("hi")]

    assert len(_run(go())) == 2

    payload = _only_call(client)
    assert payload.get("call_outcome") is None
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 20
