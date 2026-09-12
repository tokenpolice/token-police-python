"""LlamaIndex streamed rows must carry `latency` (is_streaming + ttft_ms).

LlamaIndex streamed calls are logged MANUALLY via `_log_li_py` (the inner
OpenLLMetry span is deferred and dropped). `_log_li_py` had no `latency`
parameter, so genuinely-streamed LlamaIndex rows landed with complete
tokens/cost but `is_streaming=0` and no `ttft_ms` — the `latency` dict is the
sole source of both on the /log payload.

Fix: every LlamaIndex streaming path anchors `req_start_mono` before the
provider call, stamps `_ttft_mono` at the first chunk carrying renderable
payload (`_li_chunk_marks_ttft`, mirroring Node's
`typeof chunk?.delta === "string" || chunk?.options` gate), and forwards
`latency=_build_stream_latency(...)` into `_log_li_py`.

Assertions:
  - sync stream (kind="stream") drained → is_streaming True, int ttft_ms,
    generation_ms set, clock "monotonic".
  - async stream via kind="async" (the LIVE path: astream_chat returns an
    object with `__aiter__`) → same.
  - empty sync stream → log still fires, latency present, ttft_ms None
    (honest: never fabricate TTFT when no chunk was observed).
  - hostile chunk whose `.delta` access raises → every chunk still yielded, no
    exception propagates, log still fires.
  - chunk with no `.delta` but `message.additional_kwargs` (tool-call) → TTFT
    stamped via the gate's second arm.

All fakes — no llama_index installed. The `_set_llamaindex_wrapper` install
path and session pinning mirror tests/test_llamaindex_stream_defer_restore.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import TPSession, _current_session, _in_llamaindex


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Pin one session, neutralise both pre-flight checks, and intercept the
    logged payload by swapping the enforcer's client resolver."""
    sess = TPSession(trace_id="a" * 32, root_span_id="b" * 16)
    sess._deferred_spans = []
    sess._defer_telemetry = False
    tok = _current_session.set(sess)
    guard = _in_llamaindex.set(False)

    client = _FakeClient()
    monkeypatch.setattr(enforcer, "_run_sync_check", lambda *a, **k: None)

    async def _noop_async_check(*a, **k):
        return None

    monkeypatch.setattr(enforcer, "_run_async_check", _noop_async_check)
    monkeypatch.setattr(enforcer, "get_client", lambda: client)
    try:
        yield sess, client
    finally:
        try:
            _in_llamaindex.reset(guard)
        except Exception:
            _in_llamaindex.set(False)
        _current_session.reset(tok)


class _Chunk:
    """LlamaIndex ChatResponse-shaped chunk: `.delta` str + `.message`."""

    def __init__(self, delta="hi", additional_kwargs=None):
        self.delta = delta
        self.message = type(
            "_Msg", (), {"additional_kwargs": additional_kwargs or {}}
        )()
        self.raw = None


class _ToolChunk:
    """No `.delta` at all, but a tool-call payload on message.additional_kwargs
    — the gate's second arm."""

    def __init__(self):
        self.message = type(
            "_Msg", (), {"additional_kwargs": {"tool_calls": [{"id": "call_1"}]}}
        )()
        self.raw = None


class _HostileChunk:
    """`.delta` is a property that RAISES on access — the golden-rule probe.
    Everything else is well-formed so the failure is isolated to the TTFT gate."""

    def __init__(self):
        self.message = type("_Msg", (), {"additional_kwargs": {}})()
        self.raw = None

    @property
    def delta(self):
        raise RuntimeError("hostile chunk")


def _install_sync(stream_fn):
    cls = type("FakeLI", (), {"stream": stream_fn})
    enforcer._set_llamaindex_wrapper(cls, "stream", cls.stream, "stream")
    return cls


def _install_async(astream_fn):
    cls = type("FakeLIAsync", (), {"astream": astream_fn})
    enforcer._set_llamaindex_wrapper(cls, "astream", cls.astream, "async")
    return cls


def _latency_of(client):
    assert len(client.calls) == 1, f"expected exactly one log, got {len(client.calls)}"
    return client.calls[0].get("latency")


# ── 1. Sync stream, fully drained → full latency object ──
def test_sync_stream_reports_latency(_env):
    _sess, client = _env

    def stream(self, prompt="hi", *a, **k):
        for i in range(3):
            yield _Chunk(delta=f"tok{i}")

    cls = _install_sync(stream)
    out = list(cls().stream("hi"))
    assert len(out) == 3
    assert [c.delta for c in out] == ["tok0", "tok1", "tok2"]

    lat = _latency_of(client)
    assert lat is not None, "streamed row must carry a latency object"
    assert lat["is_streaming"] is True
    assert lat["clock"] == "monotonic"
    assert isinstance(lat["ttft_ms"], int) and lat["ttft_ms"] >= 0
    assert isinstance(lat["total_ms"], int) and lat["total_ms"] >= 0
    assert lat["generation_ms"] is not None
    assert lat["output_tokens"] is None


# ── 2. Async stream via kind="async" (the LIVE astream_chat path) ──
def test_async_stream_reports_latency(_env):
    _sess, client = _env

    class _AsyncStream:
        def __aiter__(self):
            async def _gen():
                for i in range(3):
                    await asyncio.sleep(0)
                    yield _Chunk(delta=f"tok{i}")

            return _gen()

    async def astream(self, prompt="hi", *a, **k):
        return _AsyncStream()

    cls = _install_async(astream)

    async def _run():
        gen = await cls().astream("hi")
        return [c async for c in gen]

    out = asyncio.run(_run())
    assert [c.delta for c in out] == ["tok0", "tok1", "tok2"]

    lat = _latency_of(client)
    assert lat is not None
    assert lat["is_streaming"] is True
    assert lat["clock"] == "monotonic"
    assert isinstance(lat["ttft_ms"], int) and lat["ttft_ms"] >= 0
    assert isinstance(lat["total_ms"], int) and lat["total_ms"] >= 0
    assert lat["generation_ms"] is not None


# ── 3. Empty stream → still logged, is_streaming True, ttft_ms None ──
def test_empty_sync_stream_still_logs_latency(_env):
    _sess, client = _env

    def stream(self, prompt="hi", *a, **k):
        return
        yield  # pragma: no cover — makes this a generator

    cls = _install_sync(stream)
    assert list(cls().stream("hi")) == []

    lat = _latency_of(client)
    assert lat is not None
    assert lat["is_streaming"] is True
    assert lat["ttft_ms"] is None, "no chunk observed → never fabricate TTFT"
    assert lat["generation_ms"] is None
    assert isinstance(lat["total_ms"], int)


# ── 4. GOLDEN RULE: a chunk whose attribute access raises never breaks the app ──
def test_hostile_chunk_does_not_break_iteration(_env):
    _sess, client = _env

    def stream(self, prompt="hi", *a, **k):
        yield _HostileChunk()
        yield _HostileChunk()

    cls = _install_sync(stream)
    out = list(cls().stream("hi"))  # must not raise
    assert len(out) == 2
    assert all(isinstance(c, _HostileChunk) for c in out)

    # Telemetry still fires; TTFT is simply unobservable on these chunks.
    lat = _latency_of(client)
    assert lat is not None
    assert lat["is_streaming"] is True
    assert lat["ttft_ms"] is None


# ── 5. Gate's second arm: tool-call chunk with no `.delta` stamps TTFT ──
def test_tool_call_chunk_stamps_ttft(_env):
    _sess, client = _env

    def stream(self, prompt="hi", *a, **k):
        yield _ToolChunk()
        yield _ToolChunk()

    cls = _install_sync(stream)
    out = list(cls().stream("hi"))
    assert len(out) == 2

    lat = _latency_of(client)
    assert lat is not None
    assert lat["is_streaming"] is True
    assert isinstance(lat["ttft_ms"], int) and lat["ttft_ms"] >= 0
    assert lat["generation_ms"] is not None


# ── 6. The gate itself, unit-level ──
def test_chunk_marks_ttft_gate():
    assert enforcer._li_chunk_marks_ttft(_Chunk(delta="x")) is True
    assert enforcer._li_chunk_marks_ttft(_Chunk(delta="")) is True  # empty str still a delta
    assert enforcer._li_chunk_marks_ttft(_ToolChunk()) is True
    assert enforcer._li_chunk_marks_ttft(_HostileChunk()) is False
    assert enforcer._li_chunk_marks_ttft(None) is False
    assert enforcer._li_chunk_marks_ttft(_Chunk(delta=None)) is False
