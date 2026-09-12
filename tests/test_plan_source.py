"""C-04 plan provenance (``plan_source``) regression suite.

``TPSession`` gains ``plan_source`` ("app" | "default"): "app" when the
resolved ``paid_plan`` was ultimately supplied by application code (this
scope, or an ancestor scope), "default" when the SDK synthesized "free" with
no app input anywhere in the chain. Resolution uses the ``is not None``
sentinel (unlike the Node SDK's ``||``, deliberately — see the pinned quirk
test below): an explicitly passed ``paid_plan=""`` is a PROVIDED value, so it
overrides the parent and reads as "app", carrying the LITERAL "" (not "free")
as the value. The client emits ``user.plan_source`` on /check and /log
payloads only when the value is exactly "app"/"default".

Part 1 mirrors tests/test_nested_scope_default_literals.py's session()/
agent()/chain()/workflow() conventions (the ``None`` sentinel decides
inheritance, not the value).
Part 2 mirrors tests/test_workflow_name_forwarding.py's offline
capture-the-payload idiom (sync client's ``post`` mocked; async client is a
captured ``FakeAsyncClient`` driven by ``asyncio.run``).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make the local SDK importable without a prior editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import get_current_session
from token_police.client import TokenPolice


# ═══════════════════════════════════════════════════════════════════
# Part 1: TPSession / session() plan_source resolution
# ═══════════════════════════════════════════════════════════════════

def test_bare_session_no_paid_plan_root_is_default_source():
    with tp.session(name="wf") as s:
        assert s.paid_plan == "free"
        assert s.plan_source == "default"


def test_root_explicit_paid_plan_is_app_source():
    with tp.session(name="wf", paid_plan="pro") as s:
        assert s.paid_plan == "pro"
        assert s.plan_source == "app"


def test_root_explicit_paid_plan_free_literal_is_still_app_source():
    # Explicitly passing the SAME literal as the default is still "provided"
    # under the None-sentinel semantics — it must read as app-supplied.
    with tp.session(name="wf", paid_plan="free") as s:
        assert s.paid_plan == "free"
        assert s.plan_source == "app"


def test_nested_omitted_inherits_parent_value_and_source_app():
    with tp.session(name="outer", paid_plan="pro"):
        with tp.session(name="inner") as inner:
            assert inner.paid_plan == "pro"
            assert inner.plan_source == "app"


def test_nested_omitted_inherits_parent_value_and_source_default():
    with tp.session(name="outer"):
        with tp.session(name="inner") as inner:
            assert inner.paid_plan == "free"
            assert inner.plan_source == "default"


def test_python_sentinel_quirk_nested_empty_string_paid_plan_overrides_as_app():
    """PYTHON-SPECIFIC (differs from Node deliberately): an explicit ``""``
    is NOT falsy under the ``is not None`` sentinel — it is a PROVIDED value,
    so it overrides the parent's paid_plan AND is recorded as "app"-sourced,
    carrying the literal empty string (not "free")."""
    with tp.session(name="outer", paid_plan="pro"):
        with tp.session(name="inner", paid_plan="") as inner:
            assert inner.paid_plan == ""
            assert inner.plan_source == "app"


def test_python_sentinel_quirk_holds_under_a_default_sourced_parent_too():
    with tp.session(name="outer"):  # parent is "default"-sourced
        with tp.session(name="inner", paid_plan="") as inner:
            assert inner.paid_plan == ""
            assert inner.plan_source == "app"


def test_nested_explicit_override_is_always_app_regardless_of_parent_source():
    with tp.session(name="outer"):  # parent's own source is "default"
        with tp.session(name="inner", paid_plan="enterprise") as inner:
            assert inner.paid_plan == "enterprise"
            assert inner.plan_source == "app"


def test_three_deep_chain_app_source_survives_two_inheriting_levels():
    with tp.session(name="l1", paid_plan="pro"):
        with tp.session(name="l2"):
            with tp.session(name="l3") as l3:
                assert l3.paid_plan == "pro"
                assert l3.plan_source == "app"


def test_agent_chain_nesting_resolves_the_same_way():
    with tp.agent(name="outer", paid_plan="pro"):
        with tp.chain(name="inner") as inner:
            assert inner.paid_plan == "pro"
            assert inner.plan_source == "app"


def test_get_current_session_outside_scope_is_default_source():
    s = get_current_session()
    assert s.plan_source == "default"
    assert s.paid_plan == "free"


def test_workflow_static_paid_plan_is_app_source():
    @tp.workflow(name="wf", paid_plan="enterprise")
    def inner_fn():
        s = get_current_session()
        return (s.paid_plan, s.plan_source)

    assert inner_fn() == ("enterprise", "app")


def test_workflow_dynamic_binding_paid_plan_is_app_source():
    @tp.workflow(name="wf")
    def inner_fn(paid_plan: str):
        s = get_current_session()
        return (s.paid_plan, s.plan_source)

    assert inner_fn(paid_plan="pro") == ("pro", "app")


def test_workflow_omitting_paid_plan_entirely_is_default_source():
    @tp.workflow(name="wf")
    def inner_fn():
        s = get_current_session()
        return (s.paid_plan, s.plan_source)

    assert inner_fn() == ("free", "default")


def test_workflow_nested_inherits_parent_app_source():
    @tp.workflow(name="inner_wf")
    def inner_fn():
        s = get_current_session()
        return (s.paid_plan, s.plan_source)

    with tp.session(name="outer", paid_plan="pro"):
        assert inner_fn() == ("pro", "app")


# ═══════════════════════════════════════════════════════════════════
# Part 2: client wire payload — user.plan_source emission
# ═══════════════════════════════════════════════════════════════════

def _make_client() -> TokenPolice:
    """Bare client, unroutable base_url, no SSE thread. No real network I/O."""
    return TokenPolice(
        api_key="tp_sk_test",
        base_url="http://127.0.0.1:1",
        timeout=0.1,
        deployment="serverless",
        log_errors=False,
    )


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

    c._async_client_instance = FakeAsyncClient()
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


def test_check_sync_plan_source_app_emits_field_paid_plan_unaffected():
    c = _make_client()
    payload, _ = _capture_check_sync(c, paid_plan="pro", plan_source="app")
    assert payload["user"]["paid_plan"] == "pro"
    assert payload["user"]["plan_source"] == "app"


def test_check_sync_plan_source_default_emits_field():
    c = _make_client()
    payload, _ = _capture_check_sync(c, paid_plan="free", plan_source="default")
    assert payload["user"]["paid_plan"] == "free"
    assert payload["user"]["plan_source"] == "default"


def test_log_sync_plan_source_app_emits_field_paid_plan_unaffected():
    c = _make_client()
    payload, result = _capture_log_sync(c, paid_plan="pro", plan_source="app")
    assert payload["user"]["paid_plan"] == "pro"
    assert payload["user"]["plan_source"] == "app"
    assert result is None  # fire-and-forget


def test_log_sync_plan_source_default_emits_field():
    c = _make_client()
    payload, _ = _capture_log_sync(c, paid_plan="free", plan_source="default")
    assert payload["user"]["paid_plan"] == "free"
    assert payload["user"]["plan_source"] == "default"


def test_check_async_plan_source_app_emits_field():
    c = _make_client()
    payload, _ = _capture_check_async(c, paid_plan="pro", plan_source="app")
    assert payload["user"]["paid_plan"] == "pro"
    assert payload["user"]["plan_source"] == "app"


def test_log_async_plan_source_app_emits_field():
    c = _make_client()
    payload, result = _capture_log_async(c, paid_plan="pro", plan_source="app")
    assert payload["user"]["paid_plan"] == "pro"
    assert payload["user"]["plan_source"] == "app"
    assert result is None


def test_all_methods_omit_plan_source_when_not_threaded():
    for name, capturer in ALL_CAPTURERS.items():
        c = _make_client()
        payload, _ = capturer(c, paid_plan="pro")
        assert payload["user"]["paid_plan"] == "pro", name
        assert "plan_source" not in payload["user"], name


def test_all_methods_omit_plan_source_when_invalid_value():
    for name, capturer in ALL_CAPTURERS.items():
        c = _make_client()
        payload, _ = capturer(c, paid_plan="pro", plan_source="banana")
        assert payload["user"]["paid_plan"] == "pro", name
        assert "plan_source" not in payload["user"], name


def test_all_methods_omit_plan_source_when_none_explicitly():
    for name, capturer in ALL_CAPTURERS.items():
        c = _make_client()
        payload, _ = capturer(c, paid_plan="pro", plan_source=None)
        assert payload["user"]["paid_plan"] == "pro", name
        assert "plan_source" not in payload["user"], name
