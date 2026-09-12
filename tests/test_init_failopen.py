"""
Setup fail-open — ``init()`` / ``TokenPolice(...)`` must never throw into
customer code, EXCEPT the one intentional ``ValueError`` for a missing api_key.

An SDK-internal setup failure must degrade the SDK, never stop the customer
app. Pinned here:

  1. an HTTP-client construction failure at setup (e.g. a malformed
     HTTP_PROXY/HTTPS_PROXY env var the SDK never controls) → init() returns
     a working-but-degraded client; ``check_sync`` fails open to allowed and
     ``log_sync`` no-ops without raising;
  2. a truthy non-string api_key (int/bytes/object) → construction proceeds
     (coerced), the existing format warning fires, no raise;
  3. the missing-key ``ValueError`` contract is preserved (None / "");
  4. an async HTTP-client construction failure → ``_get_async_client()``
     returns None and the async ``check()`` fails open to allowed.

Companion to tests/test_init_never_throws.py (which pins the instrumentation
and OTel-setup guard sites on the same invariant).
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import client as tp_client
from token_police.client import TokenPolice

# Unroutable base_url so degraded-path calls can never do real network I/O.
DEAD_URL = "http://127.0.0.1:1"


# ── 1. Malformed proxy env — HTTP client construction fails at setup ──────

def test_malformed_proxy_env_does_not_raise(monkeypatch):
    # httpx reads HTTP(S)_PROXY at Client construction; a garbage URL raises
    # from the constructor. That is environment the SDK does not control —
    # init() must still return a client, degraded.
    monkeypatch.setenv("HTTP_PROXY", "http://[::1")
    monkeypatch.setenv("HTTPS_PROXY", "http://[::1")
    try:
        client = tp.init(
            api_key="tp_sk_test",
            base_url=DEAD_URL,
            deployment="serverless",
            firewall="dry_run",
        )
        assert client is not None
        # Degraded per-call paths fail open.
        res = client.check_sync(user_id="u1")
        assert res.get("status") == "allowed"
        assert res.get("fail_open") is True
        # log_sync must no-op without raising, and flush must drain cleanly.
        client.log_sync(user_id="u1", model="gpt-4o", input_tokens=1, output_tokens=1)
        client.flush_sync()
    finally:
        tp.uninstrument()


def test_malformed_proxy_env_constructor_direct(monkeypatch):
    # Same failure via the constructor directly (no init() plumbing).
    monkeypatch.setenv("HTTP_PROXY", "http://[::1")
    monkeypatch.setenv("HTTPS_PROXY", "http://[::1")
    client = TokenPolice(api_key="tp_sk_test", base_url=DEAD_URL, deployment="serverless")
    res = client.check_sync(user_id="u1")
    assert res.get("status") == "allowed"
    client.log_sync(user_id="u1")
    client.flush_sync()
    client.close_sync()


# ── 2. Truthy non-string api_key — coerced, warned, never raises ──────────

def test_non_string_api_key_does_not_raise(caplog):
    with caplog.at_level(logging.WARNING, logger="token_police"):
        try:
            client = tp.init(
                api_key=12345,
                base_url=DEAD_URL,
                deployment="serverless",
                firewall="dry_run",
            )
        finally:
            pass
    try:
        assert client is not None
        assert client.api_key == "12345"
        # The existing "format invalid" warning path fires (same as a
        # malformed string key: warn and proceed).
        assert any("format invalid" in r.getMessage() for r in caplog.records)
        # Degraded client still fails open on use.
        res = client.check_sync(user_id="u1")
        assert res.get("status") == "allowed"
    finally:
        tp.uninstrument()


def test_non_string_api_key_constructor_direct():
    client = TokenPolice(api_key=b"tp_sk_bytes", base_url=DEAD_URL, deployment="serverless")
    assert isinstance(client.api_key, str)
    client.close_sync()


# ── 3. Missing api_key still raises ValueError (intentional contract) ─────

def test_missing_api_key_still_raises_value_error(monkeypatch):
    monkeypatch.delenv("TOKENPOLICE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        tp.init(api_key=None)
    with pytest.raises(ValueError):
        tp.init(api_key="")
    with pytest.raises(ValueError):
        TokenPolice(api_key="")
    with pytest.raises(ValueError):
        TokenPolice(api_key=None)


# ── 4. Async client construction failure — None + async check fails open ──

class _BoomAsyncClient:
    # Deliberately NOT RuntimeError: the no-running-loop RuntimeError is an
    # expected silent no-op on the eager-init path; construction failures are
    # a different class and must be swallowed too.
    def __init__(self, *args, **kwargs):
        raise OSError("boom: async client construction")


def test_async_client_ctor_failure_returns_none_and_check_fails_open(monkeypatch):
    client = TokenPolice(api_key="tp_sk_test", base_url=DEAD_URL, deployment="serverless")
    monkeypatch.setattr(tp_client.httpx, "AsyncClient", _BoomAsyncClient)
    # The accessor itself must not raise.
    assert client._get_async_client() is None
    # And the async pre-flight check fails open to allowed.
    result = asyncio.run(client.check(user_id="u1"))
    assert result.get("status") == "allowed"
    assert result.get("fail_open") is True
    client.close_sync()


def test_init_eager_async_client_failure_does_not_raise(monkeypatch):
    # init() eagerly builds the async client when a loop is running; a
    # construction failure there must not escape init().
    monkeypatch.setattr(tp_client.httpx, "AsyncClient", _BoomAsyncClient)

    async def _run():
        return tp.init(
            api_key="tp_sk_test",
            base_url=DEAD_URL,
            deployment="serverless",
            firewall="dry_run",
        )

    try:
        client = asyncio.run(_run())
        assert client is not None
    finally:
        tp.uninstrument()
