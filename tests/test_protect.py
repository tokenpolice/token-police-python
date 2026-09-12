"""Python ``protect(target_object=...)`` can patch an in-app class.

Node ``protect({ module })`` accepts a LIVE class/object reference so a customer
can protect a class that is not importable by dotted path (defined in
``__main__``, a closure, or created dynamically). This suite verifies Python now
has the same capability via the keyword-only ``target_object=`` param, at
behavioral parity with the Node reference.

Harness mirrors tests/test_never_fail.py: ``tp.init(firewall=...)`` then mock
``client.check_sync`` / ``client.log_sync``, drive the installed wrapper, assert
``TokenPoliceBlockedError`` on enforce+block, always clean up in a ``finally``.

PINNED (contract): every protected class here is defined in FUNCTION-LOCAL scope
inside the test body so it is genuinely non-importable by dotted path. A
module-level class would be reachable via ``import_module("tests.test_protect")``
+ ``getattr`` and would make the "non-importable" premise (and every import-skip
/ negative-control assertion) VACUOUS. The one exception is the string-path test,
which deliberately uses the importable helper module ``_protect_string_helper``.
"""
from __future__ import annotations

import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# Make the local SDK importable without `pip install -e .` having run.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import enforcer
from token_police import state as tp_state
from token_police.enforcer import protect
from token_police.exceptions import TokenPoliceBlockedError


@pytest.fixture(autouse=True)
def _clean_enforcer_state():
    """Guarantee a clean instrument-table before/after every test.

    ``protect()`` mutates the module-global ``enforcer._originals``; tests that
    don't go through ``tp.init()`` (so ``_is_instrumented`` stays False and
    ``uninstrument()`` early-returns) still must not leak wrapper state to the
    next test. Function-local protected classes are discarded automatically, so
    only the dict needs clearing.
    """
    tp_state.reset_pack()
    yield
    try:
        tp.uninstrument()
    except Exception:
        pass
    enforcer._originals.clear()
    enforcer._is_instrumented = False
    tp_state.reset_pack()


def _init_with_check(firewall, check_result):
    """Init a client in ``firewall`` mode with check_sync/log_sync mocked.

    With no cached pack the enforcer takes the inline /check path, so
    ``check_result`` drives the verdict directly (see test_never_fail.py).
    """
    client = tp.init(api_key="tp_sk_test_f50", firewall=firewall)
    check_mock = MagicMock(return_value=check_result)
    client.check_sync = check_mock
    client.log_sync = MagicMock()
    return client, check_mock


_BLOCK = {"status": "blocked", "reason": "budget exceeded"}


# ── Assertion 1 ──────────────────────────────────────────────────────
def test_target_object_is_keyword_only_default_none():
    params = inspect.signature(protect).parameters
    assert "target_object" in params
    p = params["target_object"]
    assert p.kind == inspect.Parameter.KEYWORD_ONLY
    assert p.default is None
    # The public param is deliberately NOT named `target` (that identifier is
    # rebound to the local dict inside protect()).
    assert "target" not in params


# ── Assertion 2 ──────────────────────────────────────────────────────
def test_override_skips_import_module(monkeypatch):
    class FakeClient:  # function-local → non-importable
        def create(self, *a, **k):
            return {"ok": True}

    orig_create = FakeClient.create
    spy = MagicMock()
    monkeypatch.setattr(enforcer.importlib, "import_module", spy)

    ns = SimpleNamespace(FakeClient=FakeClient)
    protect("__main__", "FakeClient", "create", False, target_object=ns)

    # Skip is real AND non-vacuous: import never called, yet wrapper installed.
    spy.assert_not_called()
    assert FakeClient.create is not orig_create


# ── Assertion 3 — resolve branch A (class passed directly, falsy class_name) ─
def test_local_class_direct_enforce_blocks():
    class FakeClient:
        def create(self, *a, **k):
            return {"ok": True}

    _client, check_mock = _init_with_check("enforce", _BLOCK)
    try:
        # class_name="" → cls = target_object = FakeClient
        protect("__main__", "", "create", False, target_object=FakeClient)
        with pytest.raises(TokenPoliceBlockedError):
            FakeClient().create(model="gpt-4")
        check_mock.assert_called_once()
    finally:
        tp.uninstrument()


# ── Assertion 4 — resolve branch B (container + truthy class_name → getattr) ─
def test_local_class_via_container_enforce_blocks():
    class FakeClient:
        def create(self, *a, **k):
            return {"ok": True}

    ns = SimpleNamespace(FakeClient=FakeClient)
    _client, check_mock = _init_with_check("enforce", _BLOCK)
    try:
        # class_name="FakeClient" → cls = getattr(ns, "FakeClient", None)
        protect("__main__", "FakeClient", "create", False, target_object=ns)
        with pytest.raises(TokenPoliceBlockedError):
            FakeClient().create(model="gpt-4")
        check_mock.assert_called_once()
    finally:
        tp.uninstrument()


# ── Assertion 5 — existing string form is behaviorally identical ─────
def test_string_path_unchanged():
    # Make the importable helper resolvable as a top-level module.
    helper_dir = os.path.dirname(__file__)
    if helper_dir not in sys.path:
        sys.path.insert(0, helper_dir)
    import _protect_string_helper as helper

    _client, check_mock = _init_with_check("enforce", _BLOCK)
    try:
        # No target_object= → classic import path resolves the dotted name.
        protect("_protect_string_helper", "StringHelperClient", "create", False)
        with pytest.raises(TokenPoliceBlockedError):
            helper.StringHelperClient().create(model="gpt-4")
        check_mock.assert_called_once()
    finally:
        tp.uninstrument()


# ── Assertion 6 / 6b — manual rebind honors the override (no re-import) ──
def test_manual_rebind_honors_override(monkeypatch):
    class FakeClient:
        def create(self, *a, **k):
            return {"ok": True}

    orig_create = FakeClient.create
    spy = MagicMock()
    monkeypatch.setattr(enforcer.importlib, "import_module", spy)

    # branch A (class_name="") + manual + provider → rebind must NOT re-import
    # and must resolve the SAME live class so _originals.get((cls, method)) hits.
    protect("__main__", "", "create", False,
            manual=True, provider="openai", target_object=FakeClient)

    spy.assert_not_called()
    # _originals keyed on the live class object (6b): the manual wrapper only
    # installs when that lookup succeeds — a re-imported __main__ dup would miss.
    assert (FakeClient, "create") in enforcer._originals
    assert enforcer._originals[(FakeClient, "create")] is orig_create
    assert FakeClient.create is not orig_create  # manual wrapper installed


# ── Assertion 7 — golden rule: bad target_object no-ops without raising ──
def test_bad_target_noops_without_raising():
    # (a) wrong-type object, falsy class_name → cls=object(), getattr(...,"create",None)=None
    obj = object()
    try:
        protect("__main__", "", "create", False, target_object=obj)
    except Exception as e:  # pragma: no cover - failure path
        pytest.fail(f"object() target raised: {e!r}")
    assert (obj, "create") not in enforcer._originals

    # (b) wrong-type object, truthy class_name → getattr(object(),"Missing",None)=None → cls falsy
    try:
        protect("__main__", "Missing", "create", False, target_object=object())
    except Exception as e:  # pragma: no cover
        pytest.fail(f"object()+class_name target raised: {e!r}")

    # (c) container whose named attr is absent → getattr(ns,"Missing",None)=None
    ns = SimpleNamespace()
    try:
        protect("__main__", "Missing", "create", False, target_object=ns)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"container-missing-attr target raised: {e!r}")

    # (d) target_object=None → falls back to import path; a function-local name
    # is unreachable and ImportError-free but resolves to None → no-op.
    class FakeClient:
        def create(self, *a, **k):
            return {"ok": True}

    orig = FakeClient.create
    try:
        protect("__main__", "FakeClient", "create", False, target_object=None)
    except Exception as e:  # pragma: no cover
        pytest.fail(f"None target raised: {e!r}")
    assert FakeClient.create is orig  # unpatched


# ── Assertion 8 — idempotent re-protect keyed on the live (cls, method) ──
def test_reprotect_is_idempotent():
    class FakeClient:
        def create(self, *a, **k):
            return {"ok": True}

    orig = FakeClient.create
    _client, _check = _init_with_check("enforce", _BLOCK)
    try:
        protect("__main__", "", "create", False, target_object=FakeClient)
        after_first = FakeClient.create
        assert after_first is not orig  # wrapped once

        protect("__main__", "", "create", False, target_object=FakeClient)
        # Second call hits `if key in _originals: return` → not double-wrapped.
        assert FakeClient.create is after_first

        tp.uninstrument()  # restores the ONE stored original
        assert FakeClient.create is orig
    finally:
        # uninstrument already ran on the happy path; safe to call again.
        tp.uninstrument()


# ── Assertion 10 — negative control: same call WITHOUT override is a no-op ──
def test_no_override_local_class_is_noop():
    class FakeClient:
        def create(self, *a, **k):
            return {"ok": True}

    orig = FakeClient.create
    _client, check_mock = _init_with_check("enforce", _BLOCK)
    try:
        # IDENTICAL call minus target_object=: import_module("__main__") +
        # getattr(__main__, "FakeClient") cannot reach a function-local class.
        protect("__main__", "FakeClient", "create", False)
        assert FakeClient.create is orig  # NOT installed
        # And calling under enforce+block does NOT raise (proves non-vacuity:
        # the fix is what makes the target_object= variants above block).
        result = FakeClient().create(model="gpt-4")
        assert result == {"ok": True}
        check_mock.assert_not_called()
    finally:
        tp.uninstrument()
