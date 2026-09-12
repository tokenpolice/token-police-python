"""G5-1 — Together direct-SDK streaming silently lost 30–50% of llm rows.

Root cause: whether a Together chat stream carries a usage payload AT ALL is
replica-dependent when the request does not ask for it via
``stream_options.include_usage`` (some serving backends attach usage to the
final chunk unasked, others omit it entirely). The manual stream wrappers
logged the row only ``if last is not None`` — a drained stream with no usage
chunk produced NOTHING: no row, no warning, app exit 0.

Fix under test (two independent layers, mirrors the Node SDK):
 1. ``_inject_stream_usage_option_for_provider`` — the manual wrapper injects
    ``stream_options.include_usage`` for together chat streams, making the
    usage chunk deterministic; the synthetic usage-only terminal chunk is
    stripped from the customer-visible iteration via the new
    ``suppress_usage_chunk`` flag on ``_wrap_sync_stream``/``_wrap_async_stream``.
 2. Fallback: a together stream that drains cleanly with NO usage chunk now
    logs an APPROXIMATED row (chars/4 of request messages + accumulated
    response, ``raw.approximated: True``) instead of nothing.

Each behavioral test fails against the pre-fix code (no fallback branch, no
suppress flag, injection gated to openai.resources only).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from token_police import enforcer


# ─── chunk helpers (`_chunk_usage` reads getattr(chunk, "usage")) ─────
def _content_chunk(text="hi"):
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content=text),
                                 finish_reason=None)])


def _finish_chunk_no_usage():
    # Terminal chunk as sent by the usage-LESS Together replicas: finish_reason
    # set, NO usage attribute anywhere on the stream.
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content=None),
                                 finish_reason="stop")])


def _usage_only_chunk(p=50, c=7):
    # What include_usage appends: usage set, choices EMPTY.
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c,
                              total_tokens=p + c))


def _usage_on_content_chunk(text="tail", p=9, c=3):
    # Usage riding a CONTENT chunk (what some replicas send unasked).
    return SimpleNamespace(
        choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content=text),
                                 finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=p, completion_tokens=c,
                              total_tokens=p + c))


class _Sess:
    def __init__(self):
        self._defer_telemetry = False
        self._call_outcome = None


class FakeAsyncStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        self._it = iter(self._chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _drain_async(proxy):
    out = []
    async for c in proxy:
        out.append(c)
    return out


_TS = datetime.now(timezone.utc)
_KWARGS = {
    "model": "MiniMaxAI/MiniMax-M3",
    "messages": [{"role": "user", "content": "My laptop isn't turning on."}],
    "stream": True,
}


@pytest.fixture
def _log_spy(monkeypatch):
    """Spy `_log_manual` + neutralize composition capture (runs before the log
    call in the same try-block, so it must not raise)."""
    spy = MagicMock()
    monkeypatch.setattr(enforcer, "_log_manual", spy)
    monkeypatch.setattr(enforcer, "_capture_composition_at", MagicMock())
    return spy


def _wrap_sync(stream, provider="together", suppress=False, kwargs=None):
    return enforcer._wrap_sync_stream(
        stream, provider, _Sess(), dict(kwargs or _KWARGS), 0, "span", _TS,
        req_start_mono=None, suppress_usage_chunk=suppress)


def _wrap_async(stream, provider="together", suppress=False, kwargs=None):
    return enforcer._wrap_async_stream(
        stream, provider, _Sess(), dict(kwargs or _KWARGS), 0, "span", _TS,
        req_start_mono=None, suppress_usage_chunk=suppress)


# ─── 1. fallback: usage-less together stream still logs a row ─────────
def test_sync_no_usage_together_logs_approximated_row(_log_spy):
    chunks = [_content_chunk("Try a "), _content_chunk("hard reset."),
              _finish_chunk_no_usage()]
    received = list(_wrap_sync(iter(chunks)))
    assert len(received) == 3  # customer iteration untouched

    assert _log_spy.call_count == 1
    result = _log_spy.call_args.args[3]  # (provider, session, kwargs, result, ...)
    assert result["usage"]["approximated"] is True
    assert result["usage"]["prompt_tokens"] > 0
    assert result["usage"]["completion_tokens"] > 0
    assert result["model"] == _KWARGS["model"]


def test_async_no_usage_together_logs_approximated_row(_log_spy):
    chunks = [_content_chunk("a"), _finish_chunk_no_usage()]
    received = _run(_drain_async(_wrap_async(FakeAsyncStream(chunks))))
    assert len(received) == 2

    assert _log_spy.call_count == 1
    result = _log_spy.call_args.args[3]
    assert result["usage"]["approximated"] is True
    assert result["usage"]["prompt_tokens"] > 0


def test_usage_present_logs_real_row_not_approximated(_log_spy):
    chunks = [_content_chunk("hi"), _usage_on_content_chunk("!", 120, 34)]
    list(_wrap_sync(iter(chunks)))
    assert _log_spy.call_count == 1
    result = _log_spy.call_args.args[3]
    # The REAL final usage chunk is passed through — not the synthetic dict.
    assert result.usage.prompt_tokens == 120
    assert result.usage.completion_tokens == 34


def test_fallback_is_together_gated(_log_spy):
    # A usage-less stream on another manual provider stays row-less (unchanged).
    chunks = [_content_chunk("a"), _finish_chunk_no_usage()]
    list(_wrap_sync(iter(chunks), provider="mistral"))
    assert _log_spy.call_count == 0


def test_failed_stream_never_triggers_fallback(_log_spy, monkeypatch):
    monkeypatch.setattr(enforcer, "_emit_call_failure_log", MagicMock())
    sentinel = RuntimeError("provider fail")

    def _boom():
        yield _content_chunk("a")
        raise sentinel

    with pytest.raises(RuntimeError):
        list(_wrap_sync(_boom()))
    assert _log_spy.call_count == 0  # failure path owns the row


# ─── 2. suppress_usage_chunk strips only the synthetic chunk ─────────
def test_sync_suppress_strips_usage_only_chunk_but_logs_real_usage(_log_spy):
    chunks = [_content_chunk("a"), _content_chunk("b"), _usage_only_chunk(50, 7)]
    received = list(_wrap_sync(iter(chunks), suppress=True))
    assert received == chunks[:2]  # synthetic chunk invisible to the customer

    assert _log_spy.call_count == 1
    result = _log_spy.call_args.args[3]
    assert result.usage.prompt_tokens == 50  # tapped BEFORE stripping
    assert result.usage.completion_tokens == 7


def test_async_suppress_strips_usage_only_chunk(_log_spy):
    chunks = [_content_chunk("a"), _usage_only_chunk(5, 2)]
    received = _run(_drain_async(_wrap_async(FakeAsyncStream(chunks), suppress=True)))
    assert received == chunks[:1]
    assert _log_spy.call_count == 1


def test_suppress_false_yields_usage_only_chunk_through(_log_spy):
    # Customer asked for include_usage themselves → the chunk is theirs.
    chunks = [_content_chunk("a"), _usage_only_chunk(5, 2)]
    received = list(_wrap_sync(iter(chunks), suppress=False))
    assert received == chunks


def test_suppress_never_strips_usage_on_content_chunk(_log_spy):
    chunks = [_content_chunk("a"), _usage_on_content_chunk()]
    received = list(_wrap_sync(iter(chunks), suppress=True))
    assert received == chunks


# ─── 3. injection gate ───────────────────────────────────────────────
def _client(capture=True):
    return SimpleNamespace(capture_stream_usage=capture)


def test_inject_for_provider_together_injects_extra_body_and_restores(monkeypatch):
    # extra_body — NOT a top-level stream_options kwarg: together's typed
    # create() has no such parameter and would TypeError before any HTTP.
    monkeypatch.setattr(enforcer, "get_client", lambda: _client())
    kwargs = dict(_KWARGS)
    token = enforcer._inject_stream_usage_option_for_provider("together", kwargs)
    assert token
    assert "stream_options" not in kwargs
    assert kwargs["extra_body"] == {"stream_options": {"include_usage": True}}
    enforcer._restore_stream_usage_option(kwargs, token)
    assert "extra_body" not in kwargs


def test_inject_for_provider_merges_existing_extra_body(monkeypatch):
    monkeypatch.setattr(enforcer, "get_client", lambda: _client())
    kwargs = dict(_KWARGS, extra_body={"foo": 1})
    token = enforcer._inject_stream_usage_option_for_provider("together", kwargs)
    assert token
    assert kwargs["extra_body"] == {"foo": 1, "stream_options": {"include_usage": True}}
    enforcer._restore_stream_usage_option(kwargs, token)
    assert kwargs["extra_body"] == {"foo": 1}  # customer's extra_body restored


def test_inject_for_provider_respects_customer_opt_in(monkeypatch):
    monkeypatch.setattr(enforcer, "get_client", lambda: _client())
    kwargs = dict(_KWARGS,
                  extra_body={"stream_options": {"include_usage": True}})
    assert enforcer._inject_stream_usage_option_for_provider("together", kwargs) is None


def test_inject_for_provider_gates(monkeypatch):
    monkeypatch.setattr(enforcer, "get_client", lambda: _client())
    # Other manual providers refuse.
    assert enforcer._inject_stream_usage_option_for_provider("groq", dict(_KWARGS)) is None
    assert enforcer._inject_stream_usage_option_for_provider("cerebras", dict(_KWARGS)) is None
    # Non-stream / non-chat bodies refuse.
    assert enforcer._inject_stream_usage_option_for_provider(
        "together", {"model": "m", "messages": []}) is None
    assert enforcer._inject_stream_usage_option_for_provider(
        "together", {"model": "m", "stream": True}) is None


def test_inject_for_provider_capture_disabled(monkeypatch):
    monkeypatch.setattr(enforcer, "get_client", lambda: _client(capture=False))
    assert enforcer._inject_stream_usage_option_for_provider("together", dict(_KWARGS)) is None


def test_openai_module_path_injection_unchanged(monkeypatch):
    # The pre-existing Mode-A gate still works and still excludes non-openai
    # module paths (together rides the provider-keyed twin, not this one).
    monkeypatch.setattr(enforcer, "get_client", lambda: _client())
    kwargs = dict(_KWARGS)
    assert enforcer._inject_stream_usage_option("openai.resources.chat.completions", kwargs)
    assert enforcer._inject_stream_usage_option(
        "together.resources.chat.completions", dict(_KWARGS)) is None
