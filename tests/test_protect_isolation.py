"""protect() must isolate a poisoned/frozen target: instrumenting it may raise
(a throwing descriptor, a frozen-attr setattr, a module whose import-time body
blows up), and that must degrade to a no-op with a warning — never throw into the
customer's setup code, and never abort the wrapping of OTHER targets the customer
protects alongside it. Mirrors auto_instrument()'s per-target isolation guard.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.enforcer import protect


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    yield
    enforcer._originals.clear()
    enforcer._is_instrumented = False


def test_poisoned_single_target_noops_without_raising(monkeypatch, caplog):
    def fake_wrap(target, override_module=None):
        raise RuntimeError("frozen attr / throwing descriptor")

    monkeypatch.setattr(enforcer, "_wrap_method", fake_wrap)

    with caplog.at_level(logging.WARNING, logger="token_police"):
        # Must NOT raise — degrades to a no-op.
        protect("some.module", "Client", "create", False)

    assert any("protect() failed to instrument" in r.message for r in caplog.records)


def test_one_poisoned_target_does_not_abort_the_others(monkeypatch):
    wrapped: list[str] = []

    def fake_wrap(target, override_module=None):
        if target["method"] == "poison":
            raise RuntimeError("frozen attr")
        wrapped.append(target["method"])

    monkeypatch.setattr(enforcer, "_wrap_method", fake_wrap)

    # Customer protects a list of targets; the middle one is poisoned. Each
    # protect() call is independently isolated, so the good ones still wrap.
    targets = [
        ("m", "C", "good_before", False),
        ("m", "C", "poison", False),
        ("m", "C", "good_after", False),
    ]
    for mod, cls, meth, is_async in targets:
        # No call may raise.
        protect(mod, cls, meth, is_async)

    assert wrapped == ["good_before", "good_after"]


def test_manual_rebind_is_skipped_when_wrap_fails(monkeypatch):
    # A failed wrap returns early, so the manual provider-rebind path below it
    # never runs against a wrapper that was never installed.
    def fake_wrap(target, override_module=None):
        raise RuntimeError("boom")

    called = {"import_module": False}

    def spy_import(name):
        called["import_module"] = True
        raise AssertionError("rebind path should not have been reached")

    monkeypatch.setattr(enforcer, "_wrap_method", fake_wrap)
    monkeypatch.setattr(enforcer.importlib, "import_module", spy_import)

    # manual=True + provider would normally reach the rebind block after wrap.
    protect("m", "C", "create", False, manual=True, provider="openai")
    assert called["import_module"] is False


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
