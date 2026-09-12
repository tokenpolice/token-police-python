"""Python setup_opentelemetry must guard the global OTel TracerProvider
the same way Node does (token-police-node telemetry.ts:1385-1422): detect an
existing/injected provider and PIGGYBACK onto it instead of unconditionally
hijacking the process-global provider. Also folds in (honor the injected
`tracer_provider=` param that was previously ignored).

Test strategy (pinned by the contract): NEVER mutate the real process-global
`_TRACER_PROVIDER_SET_ONCE` do_once. Drive branches via the injected
`tracer_provider=` param + monkeypatching `telemetry.trace.get_tracer_provider`
and spying `telemetry.trace.set_tracer_provider`. Teardown assertions target the
module-level stored created-provider reference `telemetry._tp_created_provider`,
so both own-case (shutdown) and piggyback/injected-case (no shutdown) are
falsifiable without touching the global.
"""

import inspect
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from opentelemetry.sdk.trace import TracerProvider as RealTracerProvider
from opentelemetry.sdk.trace import SpanProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import client as tp_client
from token_police import state as tp_state
from token_police import telemetry


# ── Spies ────────────────────────────────────────────────────────────────────

class SpyProcessor(SpanProcessor):
    """Stands in for TokenPoliceSpanProcessor; records the log_errors threaded
    into it so assertions 2/3/4 can prove the enrichment."""

    def __init__(self, log_errors=False):
        self.log_errors = log_errors

    def on_start(self, span, parent_context=None):
        pass

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True


class SpyOwnProvider(RealTracerProvider):
    """Records processors added to a provider WE build, so assertion 1 can prove
    our processor is carried by the provider passed to set_tracer_provider."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.added = []

    def add_span_processor(self, p):
        self.added.append(p)
        return super().add_span_processor(p)


class FakeCustomerProvider:
    """A real-provider stand-in: has add_span_processor + shutdown spies, and is
    NOT a ProxyTracerProvider, so detection treats it as an existing customer
    provider to piggyback onto."""

    def __init__(self):
        self.add_span_processor = Mock()
        self.shutdown = Mock()


class ProbeOnly:
    """Attach-less object that records every attribute *access* and raises
    AttributeError (so hasattr(...) is False). Lets us prove the unknown branch
    only PROBES via hasattr and never *calls* a method on the customer object."""

    def __init__(self):
        self.accessed = []

    def __getattr__(self, name):
        self.accessed.append(name)
        raise AttributeError(name)


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    # Never let a test leak the created-provider reference into the next.
    monkeypatch.setattr(telemetry, "_tp_created_provider", None, raising=False)
    # Every setup builds our spy provider + spy processor instead of the real
    # ones (both are drop-in: SpyOwnProvider subclasses the real TracerProvider,
    # SpyProcessor subclasses the real SpanProcessor).
    monkeypatch.setattr(telemetry, "TracerProvider", SpyOwnProvider)
    monkeypatch.setattr(telemetry, "TokenPoliceSpanProcessor", SpyProcessor)
    yield


# ── Assertion 1: no-provider (Proxy) → build + set our own with processor ─────

def test_a1_proxy_provider_builds_and_sets_own(monkeypatch):
    proxy = telemetry.ProxyTracerProvider()
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: proxy)
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)

    ret = telemetry.setup_opentelemetry(log_errors=True, auto_instrument=False)

    assert set_spy.call_count == 1
    arg = set_spy.call_args.args[0]
    assert isinstance(arg, SpyOwnProvider)
    assert ret is arg
    # provider carries OUR processor, constructed with log_errors threaded
    assert len(arg.added) == 1
    assert isinstance(arg.added[0], SpyProcessor)
    assert arg.added[0].log_errors is True
    # stored created-provider is this object (for teardown)
    assert telemetry._tp_created_provider is arg


# ── Assertion 2: existing real provider → piggyback, NOT replaced ─────────────

def test_a2_existing_provider_piggyback_not_replaced(monkeypatch):
    cust = FakeCustomerProvider()
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: cust)
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)

    ret = telemetry.setup_opentelemetry(log_errors=True, auto_instrument=False)

    set_spy.assert_not_called()                     # global NOT replaced
    cust.add_span_processor.assert_called_once()
    proc = cust.add_span_processor.call_args.args[0]
    assert isinstance(proc, SpyProcessor)
    assert proc.log_errors is True                  # log_errors threaded (enrichment)
    assert ret is cust
    assert telemetry._tp_created_provider is None    # we didn't build it


# ── Assertion 3: explicit tracer_provider=X honored (kills ) ──────────────

def test_a3_injected_provider_honored(monkeypatch):
    X = FakeCustomerProvider()
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)
    # get_tracer_provider must NOT even be consulted on the injected path
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider",
                        Mock(side_effect=AssertionError("detection must not run")))

    ret = telemetry.setup_opentelemetry(
        log_errors=False, auto_instrument=False, tracer_provider=X)

    set_spy.assert_not_called()
    X.add_span_processor.assert_called_once()
    proc = X.add_span_processor.call_args.args[0]
    assert isinstance(proc, SpyProcessor)
    assert proc.log_errors is False
    assert ret is X
    assert telemetry._tp_created_provider is None


# ── Assertion 4: injected takes precedence over detection (branch order) ──────

def test_a4_injected_precedes_detection(monkeypatch):
    X = FakeCustomerProvider()
    Y = FakeCustomerProvider()
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: Y)
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)

    telemetry.setup_opentelemetry(
        log_errors=True, auto_instrument=False, tracer_provider=X)

    X.add_span_processor.assert_called_once()
    proc = X.add_span_processor.call_args.args[0]
    assert isinstance(proc, SpyProcessor) and proc.log_errors is True
    Y.add_span_processor.assert_not_called()
    set_spy.assert_not_called()
    assert telemetry._tp_created_provider is None


# ── Assertion 5: unknown attach-less DETECTED provider → set-our-own, untouched

def test_a5_unknown_detected_provider_falls_open(monkeypatch):
    cust = ProbeOnly()
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: cust)
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)

    ret = telemetry.setup_opentelemetry(log_errors=False, auto_instrument=False)

    # our own provider was set; no exception propagated
    assert set_spy.call_count == 1
    assert isinstance(set_spy.call_args.args[0], SpyOwnProvider)
    assert ret is set_spy.call_args.args[0]
    # customer object was ONLY probed via hasattr (single access), never called
    assert cust.accessed == ["add_span_processor"]
    assert telemetry._tp_created_provider is ret     # our own, not the customer


# ── Assertion 6: teardown shuts down only the provider WE created ─────────────

def test_a6_teardown_own_case_shuts_down_and_resets(monkeypatch):
    proxy = telemetry.ProxyTracerProvider()
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: proxy)
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", Mock())

    telemetry.setup_opentelemetry(log_errors=False, auto_instrument=False)
    own = telemetry._tp_created_provider
    assert own is not None
    own.shutdown = Mock()

    telemetry.unsetup_opentelemetry()

    own.shutdown.assert_called_once()
    assert telemetry._tp_created_provider is None


def test_a6_teardown_piggyback_case_does_not_shutdown_customer(monkeypatch):
    cust = FakeCustomerProvider()
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: cust)
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", Mock())

    telemetry.setup_opentelemetry(log_errors=False, auto_instrument=False)
    assert telemetry._tp_created_provider is None

    telemetry.unsetup_opentelemetry()

    cust.shutdown.assert_not_called()
    assert telemetry._tp_created_provider is None


def test_a6_teardown_injected_case_does_not_shutdown_customer(monkeypatch):
    X = FakeCustomerProvider()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", Mock())

    telemetry.setup_opentelemetry(
        log_errors=False, auto_instrument=False, tracer_provider=X)
    assert telemetry._tp_created_provider is None

    telemetry.unsetup_opentelemetry()

    X.shutdown.assert_not_called()


def test_a6_teardown_noop_when_nothing_created(monkeypatch):
    # No setup ran (reference is None) → teardown shuts nothing down, no throw.
    assert telemetry._tp_created_provider is None
    telemetry.unsetup_opentelemetry()  # must not raise
    assert telemetry._tp_created_provider is None


# ── Assertion 7: ProxyTracerProvider import guarded (version drift fails open) ─

def test_a7_proxy_symbol_none_falls_open(monkeypatch):
    monkeypatch.setattr(telemetry, "ProxyTracerProvider", None)
    cust = object()  # attach-less; without the isinstance guard we set our own
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", lambda: cust)
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)

    ret = telemetry.setup_opentelemetry(log_errors=False, auto_instrument=False)

    assert set_spy.call_count == 1
    assert isinstance(set_spy.call_args.args[0], SpyOwnProvider)
    assert ret is set_spy.call_args.args[0]


# ── Assertion 8: detection body wrapped in its own fail-open try/except ────────

def test_a8_detection_raise_does_not_propagate(monkeypatch):
    def boom():
        raise RuntimeError("get_tracer_provider blew up")
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", boom)
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)

    # must NOT raise into caller
    ret = telemetry.setup_opentelemetry(log_errors=False, auto_instrument=False)

    # best-effort fallback set our own
    assert set_spy.call_count == 1
    assert isinstance(set_spy.call_args.args[0], SpyOwnProvider)
    assert ret is set_spy.call_args.args[0]


# ── Assertion 10: public signature byte-for-byte unchanged ────────────────────

def test_a10_signature_unchanged():
    sig = inspect.signature(telemetry.setup_opentelemetry)
    params = list(sig.parameters.items())
    names = [n for n, _ in params]
    assert names == ["log_errors", "auto_instrument", "tracer_provider"]
    assert sig.parameters["log_errors"].default is False
    assert sig.parameters["auto_instrument"].default is True
    assert sig.parameters["tracer_provider"].default is None


# ── Assertion 13: injected X lacking add_span_processor → no throw, no mutate ──

def test_a13_injected_attachless_falls_open_no_mutate(monkeypatch):
    X = ProbeOnly()
    set_spy = Mock()
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", set_spy)

    ret = telemetry.setup_opentelemetry(
        log_errors=False, auto_instrument=False, tracer_provider=X)

    # (i) no exception; (ii) our own set; (iii) X only probed via hasattr
    assert set_spy.call_count == 1
    assert isinstance(set_spy.call_args.args[0], SpyOwnProvider)
    assert ret is set_spy.call_args.args[0]
    assert X.accessed == ["add_span_processor"]
    assert telemetry._tp_created_provider is ret

    # teardown must NOT shut down X (we never stored it)
    ret.shutdown = Mock()
    telemetry.unsetup_opentelemetry()
    ret.shutdown.assert_called_once()  # our own IS shut down
    assert "shutdown" not in X.accessed  # X never touched again


def test_a13_injected_bare_object_no_throw(monkeypatch):
    # plain object() with no attributes at all
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", Mock())
    ret = telemetry.setup_opentelemetry(
        log_errors=False, auto_instrument=False, tracer_provider=object())
    assert isinstance(ret, SpyOwnProvider)


# ── Assertion 14: init() threads log_errors into setup_opentelemetry ──────────
# Pre-fix client.py called setup_opentelemetry(tracer_provider=...) only, so the
# telemetry layer always ran with log_errors=False and every diagnostic gated on
# it was dead even when the user opted in.

def _reset_client():
    try:
        tp.uninstrument()
    except Exception:
        pass
    tp_state.set_client(None)


@pytest.mark.parametrize("kwargs, expected", [
    ({"log_errors": True}, True),
    ({}, False),
])
def test_a14_init_forwards_log_errors_to_setup(monkeypatch, kwargs, expected):
    calls = []
    monkeypatch.setattr(
        tp_client, "setup_opentelemetry",
        lambda **kw: calls.append(kw) or Mock())

    _reset_client()
    try:
        tp.init(api_key="tp_sk_test", firewall="off", deployment="serverless",
                **kwargs)
    finally:
        _reset_client()

    assert len(calls) == 1
    assert calls[0]["log_errors"] is expected
