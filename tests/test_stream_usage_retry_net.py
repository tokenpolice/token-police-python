"""
Stream-usage injection retry-net predicate.

The SDK injects ``stream_options.include_usage`` into openai-wire chat streams so
streamed spend isn't silently lost. If the provider then errors, the SDK strips
the injection and retries ONCE. The retry must fire ONLY when the rejection is
plausibly caused by the injected option — a strict-compat server answering
400/422 or naming the param. A rate-limit (429) or auth failure (401/403) is
never caused by the injection, so retrying there would issue a guaranteed SECOND
provider request exactly while the customer is rate-limited / unauthorized.

Two layers are covered:
  * ``_should_retry_without_injection`` in isolation (the retry decision), incl.
    a hostile error object whose attribute/str access raises.
  * The REAL sync/async wrappers driven end to end via ``_wrap_method`` against a
    fake openai.resources class, asserting the provider ``original`` is called
    exactly once on a non-injection error, twice (stripped) on a 400/422, and
    that the customer's kwargs carry no leaked ``stream_options`` on the retry.

Golden rule: an SDK-internal failure must never propagate into customer code, and
the SDK must never cause duplicate provider spend.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police
from token_police import enforcer
from token_police import state as tp_state


# ── predicate: _should_retry_without_injection ────────────────────────────────


class _StatusErr(Exception):
    def __init__(self, status, msg=""):
        super().__init__(msg)
        self.status_code = status


class _ResponseStatusErr(Exception):
    """Status carried on .response.status_code (the httpx/openai shape)."""

    def __init__(self, status, msg=""):
        super().__init__(msg)

        class _R:
            status_code = status

        self.response = _R()


class _HostileErr(Exception):
    """Every duck-typed access the predicate might make raises."""

    @property
    def status_code(self):  # noqa: D401
        raise RuntimeError("boom-status")

    @property
    def response(self):
        raise RuntimeError("boom-response")

    def __str__(self):
        raise RuntimeError("boom-str")


@pytest.mark.parametrize("status", [429, 401, 403, 404, 500, 502])
def test_predicate_no_retry_on_unrelated_status(status):
    # A rate-limit / auth / not-found / server error is never caused by the
    # injected param → no second provider request.
    assert enforcer._should_retry_without_injection(_StatusErr(status)) is False
    assert enforcer._should_retry_without_injection(_ResponseStatusErr(status)) is False


@pytest.mark.parametrize("status", [400, 422])
def test_predicate_retry_on_strict_compat_reject(status):
    assert enforcer._should_retry_without_injection(_StatusErr(status)) is True
    assert enforcer._should_retry_without_injection(_ResponseStatusErr(status)) is True


def test_predicate_message_escape_hatch():
    # A non-400/422 status whose message names the injected param still retries.
    assert enforcer._should_retry_without_injection(
        _StatusErr(404, "unknown field stream_options")
    ) is True
    assert enforcer._should_retry_without_injection(
        _StatusErr(400, "include_usage not supported")  # already-True status, message too
    ) is True
    # Case-insensitive.
    assert enforcer._should_retry_without_injection(
        _StatusErr(0, "Rejected: STREAM_OPTIONS")
    ) is True


def test_predicate_hostile_object_no_throw_no_retry():
    # A hostile error object must not make the predicate raise, and must not
    # trigger a retry (predicate failure ⇒ no retry ⇒ original error propagates).
    assert enforcer._should_retry_without_injection(_HostileErr()) is False


# ── restore round-trip (no leak into the customer's kwargs) ───────────────────


def test_restore_roundtrip_absent_prior():
    kwargs = {"stream": True, "messages": []}
    token = enforcer._inject_stream_usage_option("openai.resources.chat", kwargs)
    assert token is not None
    assert kwargs["stream_options"]["include_usage"] is True
    enforcer._restore_stream_usage_option(kwargs, token)
    assert "stream_options" not in kwargs  # key never existed → removed


def test_restore_roundtrip_preexisting_prior():
    prior = {"foo": 1}
    kwargs = {"stream": True, "messages": [], "stream_options": prior}
    token = enforcer._inject_stream_usage_option("openai.resources.chat", kwargs)
    assert token is not None
    assert kwargs["stream_options"]["include_usage"] is True
    enforcer._restore_stream_usage_option(kwargs, token)
    assert kwargs["stream_options"] == {"foo": 1}


# ── end-to-end wrapper drive ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _client_and_stubs(monkeypatch):
    """A capture-on client, with the non-injection wrapper machinery stubbed to
    no-ops so the test is hermetic (no network) — the injection/predicate/restore
    code under test stays REAL."""
    token_police.init(
        api_key="tp_sk_test",
        deployment="serverless",
        firewall="off",
        capture_stream_usage=True,
    )

    async def _noop_async(*a, **k):
        return None

    def _noop_sync(*a, **k):
        return None

    monkeypatch.setattr(enforcer, "_run_async_check", _noop_async)
    monkeypatch.setattr(enforcer, "_run_sync_check", _noop_sync)
    monkeypatch.setattr(enforcer, "_emit_call_failure_log", _noop_sync)
    monkeypatch.setattr(enforcer, "_flush_deferred_spans", _noop_sync)
    try:
        yield
    finally:
        try:
            token_police.uninstrument()
        except Exception:
            pass
        tp_state.set_client(None)


def _install(is_async, script):
    """Wrap a fresh fake openai.resources class' `create` with the real wrapper.

    `script` is a list of callables invoked one-per-call; each returns a value or
    raises. Records, per call, whether the (mutable) kwargs still carried the
    injected stream_options — so the retry can be shown to strip it.
    """
    calls = {"count": 0, "so_present": []}

    if is_async:
        class Fake:
            async def create(self, **kwargs):
                calls["so_present"].append("stream_options" in kwargs)
                i = calls["count"]
                calls["count"] += 1
                return script[i]()
    else:
        class Fake:
            def create(self, **kwargs):
                calls["so_present"].append("stream_options" in kwargs)
                i = calls["count"]
                calls["count"] += 1
                return script[i]()

    target = {
        "module": "openai.resources.chat.completions",
        "object": "",  # → cls == override_module (the class itself)
        "method": "create",
        "async": is_async,
    }
    enforcer._wrap_method(target, override_module=Fake)
    return Fake, calls


def _raise(exc):
    def _f():
        raise exc

    return _f


def _return(val):
    def _f():
        return val

    return _f


def _drive(fake, is_async):
    coro_or_val = fake().create(stream=True, messages=[{"role": "user", "content": "hi"}])
    if is_async:
        return asyncio.run(coro_or_val)
    return coro_or_val


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("status", [429, 401, 403])
def test_wrapper_no_retry_on_ratelimit_or_auth(is_async, status):
    # PRE-FIX EVIDENCE: the old ANY-4xx predicate retried here → original twice.
    fake, calls = _install(is_async, [_raise(_StatusErr(status, "denied"))])
    with pytest.raises(_StatusErr):
        _drive(fake, is_async)
    assert calls["count"] == 1  # exactly one provider request; NO duplicate spend


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("status", [400, 422])
def test_wrapper_strips_and_retries_on_strict_compat(is_async, status):
    sentinel = {"ok": True, "id": "resp-2"}
    fake, calls = _install(
        is_async, [_raise(_StatusErr(status, "unknown param")), _return(sentinel)]
    )
    out = _drive(fake, is_async)
    assert out is sentinel
    assert calls["count"] == 2
    assert calls["so_present"] == [True, False]  # 1st injected, 2nd stripped


@pytest.mark.parametrize("is_async", [False, True])
def test_wrapper_message_escape_hatch_retries(is_async):
    sentinel = {"ok": True}
    fake, calls = _install(
        is_async,
        [_raise(_StatusErr(404, "unknown field: stream_options")), _return(sentinel)],
    )
    out = _drive(fake, is_async)
    assert out is sentinel
    assert calls["count"] == 2
    assert calls["so_present"] == [True, False]


@pytest.mark.parametrize("is_async", [False, True])
def test_wrapper_no_retry_on_server_error(is_async):
    fake, calls = _install(is_async, [_raise(_StatusErr(500, "oops"))])
    with pytest.raises(_StatusErr):
        _drive(fake, is_async)
    assert calls["count"] == 1


@pytest.mark.parametrize("is_async", [False, True])
def test_wrapper_hostile_error_no_retry_no_sdk_throw(is_async):
    fake, calls = _install(is_async, [_raise(_HostileErr())])
    with pytest.raises(_HostileErr):
        _drive(fake, is_async)
    assert calls["count"] == 1  # no retry, and the customer's own error propagates
