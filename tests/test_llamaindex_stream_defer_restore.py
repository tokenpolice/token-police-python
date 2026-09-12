"""LlamaIndex `stream` guard must restore `_defer_telemetry` if the underlying
provider call raises SYNCHRONOUSLY at call time.

Bug (state-observed): `li_stream` set `session._defer_telemetry = True` and then
called the provider method with NO try/except. A synchronous raise (e.g. an
auth/quota error) escaped with the flag leaked True forever. Because
`_flush_deferred_spans` is gated on `not session._defer_telemetry`, every
SUBSEQUENT span on that session would buffer into `_deferred_spans` and never be
logged — permanent, silent telemetry loss for the whole session.

Fix mirrors the sibling `li_async`: on ANY exception from the call, restore the
prior `_defer_telemetry` before re-raising the customer's own provider error.

All fakes — no LlamaIndex installed. The `_set_llamaindex_wrapper` install path
and session pinning mirror tests/test_sec06_langchain_interleaved.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import TPSession, _current_session, _in_llamaindex


@pytest.fixture(autouse=True)
def _fresh_session(monkeypatch):
    """Pin one shared session and neutralise the network pre-flight check."""
    sess = TPSession()
    sess._deferred_spans = []
    sess._defer_telemetry = False
    tok = _current_session.set(sess)
    guard = _in_llamaindex.set(False)
    monkeypatch.setattr(enforcer, "_run_sync_check", lambda *a, **k: None)
    try:
        yield sess
    finally:
        try:
            _in_llamaindex.reset(guard)
        except Exception:
            _in_llamaindex.set(False)
        _current_session.reset(tok)


class _RaisingModel:
    """Provider `stream` that raises synchronously at call time (before it could
    ever return a stream iterator) — the auth-error shape."""

    def stream(self, prompt="hi", *a, **k):
        raise RuntimeError("auth boom")


class _OkModel:
    def stream(self, prompt="hi", *a, **k):
        for i in range(2):
            yield f"{prompt}-{i}"


def _install(model_cls):
    cls = type(f"Fake_{model_cls.__name__}", (model_cls,), {})
    enforcer._set_llamaindex_wrapper(cls, "stream", cls.stream, "stream")
    return cls


# ── 1. Py-5: synchronous raise propagates AND defer flag is restored ──
def test_sync_raise_restores_defer_and_reraises(_fresh_session):
    sess = _fresh_session
    assert sess._defer_telemetry is False
    cls = _install(_RaisingModel)

    caught = None
    try:
        cls().stream("hi")
    except RuntimeError as e:
        caught = e

    # Customer's own provider error propagates verbatim (not masked/wrapped).
    assert type(caught) is RuntimeError
    assert str(caught) == "auth boom"
    # The crux: flag restored to its prior value (pre-fix: stays True).
    assert sess._defer_telemetry is False


# ── 2. Prior-True baseline is restored to True (not blindly cleared) ──
def test_sync_raise_restores_prior_true(_fresh_session):
    sess = _fresh_session
    sess._defer_telemetry = True  # simulate being inside an outer defer scope
    cls = _install(_RaisingModel)

    with pytest.raises(RuntimeError):
        cls().stream("hi")

    # Restored to the OUTER scope's value, not forced False.
    assert sess._defer_telemetry is True


# ── 3. A session that raised once still flushes LATER telemetry ──
# This is the real-world damage the leak caused: the whole session went dark.
def test_session_still_flushes_after_a_raise(_fresh_session):
    sess = _fresh_session
    cls = _install(_RaisingModel)
    with pytest.raises(RuntimeError):
        cls().stream("hi")

    # A later, unrelated deferred span must be dispatchable — i.e. the gate
    # (`not _defer_telemetry`) is open again.
    logged = []

    class _FakeTP:
        def log_sync(self, **payload):
            logged.append(payload)

    import token_police.enforcer as _enf
    orig = _enf.get_client
    _enf.get_client = lambda: _FakeTP()
    try:
        sess._deferred_spans.append(
            {"tp_tag": "LATER", "span": {"trace_id": sess.trace_id, "span_order": 0}}
        )
        assert sess._defer_telemetry is False   # gate open
        _enf._flush_deferred_spans(sess)
    finally:
        _enf.get_client = orig

    assert [p.get("tp_tag") for p in logged] == ["LATER"]
    assert sess._deferred_spans == []


# ── 4. Happy path unchanged: normal stream still deferred + guarded ──
def test_sync_ok_stream_defers_during_iteration(_fresh_session):
    sess = _fresh_session
    cls = _install(_OkModel)
    gen = cls().stream("A")
    out = [c for c in gen]
    assert out == ["A-0", "A-1"]
    # After exhaustion the guard restores defer to its prior (False) value.
    assert sess._defer_telemetry is False
