"""Coverage for `_wrap_method`'s `_tp_preflight_wrapper` identity guard.

Context: `anthropic/lib/bedrock/_beta_messages.py` does
``create = FirstPartyMessagesAPI.create`` — a class-attribute ALIAS bound at
import time. On the anthropic version pinned today that import is eager, so
the alias captures the *pre*-TokenPolice function and there is no double-wrap.
But if a future anthropic release made that import lazy, the alias would
capture TokenPolice's ALREADY-WRAPPED function, and a second `_wrap_method`
pass over the aliasing class (a DIFFERENT `_originals` key — the per-(cls,
method) dedupe can't see it) would wrap it AGAIN, running the pre-flight
check and the log TWICE per call: double-billing.

`_wrap_method` guards against this by checking a `_tp_preflight_wrapper`
marker stamped on every wrapper kind: when the resolved "original" already
carries that marker, wrapping is skipped entirely.

This suite reproduces the alias shape directly (no anthropic dependency) via
two synthetic classes, and pins the guard from the angle that survives a
rename of the marker attribute: calling through the aliasing class must bill
EXACTLY ONCE (one /check, one /log), not twice. A positive control (an
unaliased class pair) proves the assertions aren't vacuously true.

Harness idiom mirrors tests/test_protect.py: `tp.init(firewall=...)` with
`check_sync`/`log_sync` mocked, function-local classes, `protect(target_object=)`
so no real provider SDK is needed, and `enforcer._originals.clear()` +
`tp.uninstrument()` cleanup so no wrapper state leaks to later tests.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import enforcer
from token_police import state as tp_state
from token_police.context import claim_anthropic_stream_span
from token_police.enforcer import protect
from token_police.enforcer import _AnthropicAsyncStreamMgrWrapper


@pytest.fixture(autouse=True)
def _clean_enforcer_state():
    """Mirrors tests/test_protect.py — guarantee no wrapper/originals state
    leaks between tests, even on failure."""
    tp_state.reset_pack()
    yield
    try:
        tp.uninstrument()
    except Exception:
        pass
    enforcer._originals.clear()
    enforcer._is_instrumented = False
    tp_state.reset_pack()


def _init_allowed():
    """Init a client in enforce mode with check/log mocked to 'allowed', so
    the manual wrapper runs its full check -> call -> log path."""
    client = tp.init(api_key="tp_sk_test_alias_guard", firewall="enforce")
    check_mock = MagicMock(return_value={"status": "allowed"})
    client.check_sync = check_mock
    log_mock = MagicMock()
    client.log_sync = log_mock
    return client, check_mock, log_mock


def _fake_response():
    """A minimal OpenAI-compatible-shaped response — enough for
    `_extract_openai_compatible_usage` to extract real token counts."""
    return SimpleNamespace(
        model="fake-model",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )


# ── Test 1: the identity guard itself ───────────────────────────────────
def test_alias_of_already_wrapped_method_noops_and_bills_once():
    class A:
        def create(self, **kwargs):
            return _fake_response()

    _client, check_mock, log_mock = _init_allowed()

    # Wrap A normally — A.create becomes a TokenPolice pre-flight wrapper.
    protect("__main__", "", "create", False, manual=True, provider="testprov",
            target_object=A)
    wrapped_create = A.create
    assert getattr(wrapped_create, "_tp_preflight_wrapper", False) is True

    # B aliases A's ALREADY-WRAPPED method as a plain class attribute — the
    # exact shape of `anthropic.lib.bedrock._beta_messages`'s
    # `create = FirstPartyMessagesAPI.create` under a hypothetical future
    # lazy import.
    class B:
        create = wrapped_create

    protect("__main__", "", "create", False, manual=True, provider="testprov",
            target_object=B)

    # The guard must have fired: B.create is untouched — still the identical
    # wrapper function object copied over by the alias, not a fresh
    # second-layer wrapper around it.
    assert B.create is wrapped_create

    # The behavioral assertion that actually encodes "no double-billing":
    # calling through B must produce exactly one /check and one /log, not
    # two, regardless of what the marker attribute is named or how the guard
    # is implemented internally.
    B().create(model="m")
    check_mock.assert_called_once()
    log_mock.assert_called_once()


# ── Test 1b: positive control — no vacuity ──────────────────────────────
def test_unaliased_pair_wraps_independently_and_bills_once_each():
    """Without the already-wrapped alias, two independent classes each wrap
    normally and each bill exactly once per call — proving the no-op /
    once-only assertions above are not trivially true (e.g. from a check/log
    mock that simply never fires)."""

    class C:
        def create(self, **kwargs):
            return _fake_response()

    class D:
        def create(self, **kwargs):
            return _fake_response()

    _client, check_mock, log_mock = _init_allowed()

    protect("__main__", "", "create", False, manual=True, provider="testprov",
            target_object=C)
    protect("__main__", "", "create", False, manual=True, provider="testprov",
            target_object=D)

    # Both independently wrapped (not the same function object) — neither
    # aliases the other, so this is a fair baseline for the guard test above.
    assert C.create is not D.create
    assert getattr(C.create, "_tp_preflight_wrapper", False) is True
    assert getattr(D.create, "_tp_preflight_wrapper", False) is True

    C().create(model="m")
    D().create(model="m")
    assert check_mock.call_count == 2
    assert log_mock.call_count == 2


# ── Test 2: W1-rebuild fold-back mechanism ──────────────────────────────
#
# `_rebuild_after_reroute` arms a FRESH suppression window around its own
# `rebuild()` call (which re-invokes the wrapped `stream()`), then folds
# whatever THAT window claimed back into the outer W1 record — so
# `__aenter__`'s `seen == 0` shape-check still reflects whether ANY span
# fired before manager-enter, across either invocation of `stream()`.
#
# In real anthropic usage this fold-back can never change the OBSERVABLE
# outcome: W1 and the rebuild window both arm around the exact same wrapped
# `stream()` call site, so they see identical instrumentor behavior — W1
# non-zero implies the rebuild window would be too, and vice versa. That is
# why the reviewer's mutation (deleting the fold-back) survived: no test can
# force "W1 saw 0, rebuild saw >=1" through the real seam.
#
# This test does not try to manufacture that real-world scenario (it can't,
# honestly). It instead pins the fold-back MECHANISM directly against the
# unmodified method: `self._rebuild` is the documented dependency-injection
# seam (`_AnthropicAsyncStreamMgrWrapper.__init__` already accepts a
# zero-arg `rebuild` factory), and the real `claim_anthropic_stream_span`
# (unmocked) is invoked from inside it to simulate "the instrumentor started
# a span during the rebuild call" — exactly what a real SpanProcessor.on_start
# does. That is a legitimate, falsifiable exercise of the actual fold-back
# arithmetic, independent of whether today's SDK can ever hand it a
# non-zero rebuild count. If the fold-back is deleted or the arithmetic
# broken, this test fails; if it is intact, W1's "seen" absorbs the
# rebuild-window's claim exactly as designed.
REAL_SCOPE = "opentelemetry.instrumentation.anthropic"
REAL_NAME = "anthropic.chat"


def _make_async_stream_wrapper(kwargs, rebuild, span_window):
    return _AnthropicAsyncStreamMgrWrapper(
        mgr=object(), kwargs=kwargs, rebuild=rebuild, span_window=span_window,
    )


def test_rebuild_fold_back_absorbs_a_span_claimed_during_rebuild():
    w1_window = {"seen": 0, "limit": 1}

    def rebuild():
        # Simulate the instrumentor starting a stream span DURING the
        # rebuilt stream() call — the same call `claim_anthropic_stream_span`
        # is invoked from in production (SpanProcessor.on_start), against
        # whichever window is currently armed (the rebuild's fresh one, not
        # W1 — `_rebuild_after_reroute` arms it before calling `rebuild()`).
        claim_anthropic_stream_span(
            SimpleNamespace(name=REAL_NAME), REAL_SCOPE
        )
        return SimpleNamespace()  # stand-in "new manager"

    wrapper = _make_async_stream_wrapper(
        kwargs={"model": "claude-rerouted"}, rebuild=rebuild, span_window=w1_window,
    )
    wrapper._rebuild_after_reroute(model_before="claude-original")

    # The rebuild window's claim (seen=1) folded back into the outer W1
    # record — the exact bookkeeping __aenter__'s later `seen == 0`
    # shape-check depends on.
    assert w1_window["seen"] == 1


def test_rebuild_with_no_span_claimed_leaves_w1_seen_at_zero():
    """Negative control: proves the increment above isn't a hardcoded/
    unconditional bump — a rebuild that claims nothing folds back nothing."""
    w1_window = {"seen": 0, "limit": 1}

    def rebuild():
        return SimpleNamespace()  # no claim_anthropic_stream_span call

    wrapper = _make_async_stream_wrapper(
        kwargs={"model": "claude-rerouted"}, rebuild=rebuild, span_window=w1_window,
    )
    wrapper._rebuild_after_reroute(model_before="claude-original")

    assert w1_window["seen"] == 0


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
