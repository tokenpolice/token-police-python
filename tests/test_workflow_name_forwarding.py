"""Regression suite — the manual API forwards the positional workflow_name.

Both SDKs document a positional ``workflow_name`` on ``check``/``log`` but the
payload builders historically never read it, so a manual caller
(``tp.log_sync(uid, plan, "checkout_agent", ...)``) silently lost workflow
attribution: workflow-scoped rules/budgets never matched and dashboard grouping
fell back to "default". These tests pin the fix: the positional name is surfaced
into ``payload["metadata"]["workflow_name"]`` when (a) it is non-default and
(b) the caller has not already placed the key in metadata — on a shallow copy so
the caller's dict is never mutated, and guarded so hostile metadata is a no-op.

All offline: the sync client's ``post`` is mocked and the async client's
``post`` is a captured ``FakeAsyncClient`` driven by ``asyncio.run`` (the idioms
from test_sec03_log_sync_failopen.py / test_sec09_async_log_failopen.py).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Make the local SDK importable without a prior editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police.client import TokenPolice


def _make_client() -> TokenPolice:
    """Bare client, unroutable base_url, no SSE thread. No real network I/O."""
    return TokenPolice(
        api_key="tp_sk_test",
        base_url="http://127.0.0.1:1",
        timeout=0.1,
        deployment="serverless",
        log_errors=False,
    )


# ── Per-method payload capturers (return the json= dict + the call result) ────

def _capture_check_sync(c: TokenPolice, **kwargs):
    captured: dict = {}

    def fake_post(path, json=None):
        captured[path] = json
        return MagicMock(status_code=200, json=lambda: {})

    c._sync_client.post = fake_post
    result = c.check_sync(**kwargs)
    return captured["/v1/guard/check"], result


def _capture_log_sync(c: TokenPolice, **kwargs):
    captured: dict = {}

    def fake_post(path, json=None):
        captured[path] = json
        return MagicMock(status_code=200)

    c._sync_client.post = fake_post
    kwargs.setdefault("input_tokens", 1)
    kwargs.setdefault("output_tokens", 1)
    result = c.log_sync(**kwargs)
    c.flush_sync()  # drain the background _send so the mock is invoked
    return captured["/v1/guard/log"], result


def _capture_check_async(c: TokenPolice, **kwargs):
    captured: dict = {}

    class FakeAsyncClient:
        async def post(self, path, json=None):
            captured[path] = json
            return MagicMock(status_code=200, json=lambda: {})

    c._async_client_instance = FakeAsyncClient()  # await path records it
    result = asyncio.run(c.check(**kwargs))
    return captured["/v1/guard/check"], result


def _capture_log_async(c: TokenPolice, **kwargs):
    captured: dict = {}

    class FakeAsyncClient:
        async def post(self, path, json=None):
            captured[path] = json
            return MagicMock(status_code=200)

    c._async_client_instance = FakeAsyncClient()
    kwargs.setdefault("input_tokens", 1)
    kwargs.setdefault("output_tokens", 1)

    async def scenario():
        r = await c.log(**kwargs)
        await c.flush()  # drain the create_task-scheduled _send
        return r

    result = asyncio.run(scenario())
    return captured["/v1/guard/log"], result


ALL_CAPTURERS = {
    "check_sync": _capture_check_sync,
    "log_sync": _capture_log_sync,
    "check_async": _capture_check_async,
    "log_async": _capture_log_async,
}


# ── A11: check_sync surfaces the positional workflow_name ────────────────────
def test_check_sync_forwards_workflow_name():
    c = _make_client()
    payload, _ = _capture_check_sync(c, workflow_name="checkout")
    assert payload["metadata"]["workflow_name"] == "checkout"


# ── A12: log_sync surfaces the positional workflow_name ──────────────────────
def test_log_sync_forwards_workflow_name():
    c = _make_client()
    payload, result = _capture_log_sync(c, workflow_name="checkout")
    assert payload["metadata"]["workflow_name"] == "checkout"
    assert result is None  # fire-and-forget


# ── A13: async check actually exercises the awaited async client ─────────────
def test_async_check_forwards_workflow_name():
    c = _make_client()
    payload, _ = _capture_check_async(c, workflow_name="checkout")
    assert payload["metadata"]["workflow_name"] == "checkout"


# ── A14: async log actually exercises the awaited async client ───────────────
def test_async_log_forwards_workflow_name():
    c = _make_client()
    payload, result = _capture_log_async(c, workflow_name="checkout")
    assert payload["metadata"]["workflow_name"] == "checkout"
    assert result is None


# ── A15: default name + no metadata ⇒ no workflow_name key (byte-identical) ───
@pytest.mark.parametrize("method", list(ALL_CAPTURERS))
def test_default_workflow_name_leaves_no_key(method):
    c = _make_client()
    payload, _ = ALL_CAPTURERS[method](c, workflow_name="default")
    assert payload["metadata"] == {}
    assert "workflow_name" not in payload["metadata"]


# ── A9-parity: internal auto-path (positional == pre-seeded key) unchanged ────
@pytest.mark.parametrize("method", list(ALL_CAPTURERS))
def test_internal_autopath_parity_no_override(method):
    c = _make_client()
    # Session default value "default_workflow" is != "default" so gate (a)
    # passes; gate (b) blocks because the key is already present.
    payload, _ = ALL_CAPTURERS[method](
        c, workflow_name="default_workflow",
        metadata={"workflow_name": "default_workflow"},
    )
    assert payload["metadata"]["workflow_name"] == "default_workflow"
    assert payload["metadata"] == {"workflow_name": "default_workflow"}


# ── A16: explicit metadata.workflow_name wins over the positional name ────────
@pytest.mark.parametrize("method", list(ALL_CAPTURERS))
def test_explicit_metadata_wins_over_positional(method):
    c = _make_client()
    payload, _ = ALL_CAPTURERS[method](
        c, workflow_name="other", metadata={"workflow_name": "explicit"},
    )
    assert payload["metadata"]["workflow_name"] == "explicit"


# ── A17: non-mutation — the caller's dict is never mutated (shallow copy) ─────
@pytest.mark.parametrize("method", list(ALL_CAPTURERS))
def test_caller_metadata_not_mutated(method):
    c = _make_client()
    md = {"foo": "bar"}
    payload, _ = ALL_CAPTURERS[method](c, workflow_name="checkout", metadata=md)
    # The captured payload carries the name...
    assert payload["metadata"]["workflow_name"] == "checkout"
    assert payload["metadata"]["foo"] == "bar"
    # ...but the caller's original object is untouched.
    assert md == {"foo": "bar"}
    assert "workflow_name" not in md


# ── A18 (+ Amendment 3): golden rule — hostile metadata never raises ─────────
class _RaisingContainsDict(dict):
    """A dict subclass whose membership test raises — the only place the gate
    block can throw. Non-empty so ``metadata or {}`` keeps the object."""

    def __contains__(self, key):  # noqa: D401
        raise RuntimeError("hostile __contains__")


def test_check_sync_hostile_metadata_does_not_raise():
    c = _make_client()
    payload, result = _capture_check_sync(
        c, workflow_name="x", metadata=_RaisingContainsDict({"seed": 1}),
    )
    # No exception into the caller; check returns a dict.
    assert isinstance(result, dict)


def test_log_sync_hostile_metadata_does_not_raise():
    c = _make_client()
    payload, result = _capture_log_sync(
        c, workflow_name="x", metadata=_RaisingContainsDict({"seed": 1}),
    )
    assert result is None  # returns normally, no raise


def test_async_check_hostile_metadata_does_not_raise():
    # Amendment 3: at least one async method under asyncio.run.
    c = _make_client()
    payload, result = _capture_check_async(
        c, workflow_name="x", metadata=_RaisingContainsDict({"seed": 1}),
    )
    assert isinstance(result, dict)
