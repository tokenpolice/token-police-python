"""G3-17-4 — classifier maps native gRPC StatusCodes (_classify._grpc_*).

Native `xai_sdk` (and any other gRPC-first client) does not speak HTTP: a
rejection surfaces as a raw grpc exception — `grpc._channel._InactiveRpcError`
(sync unary), `grpc._channel._MultiThreadedRendezvous` (sync streaming) or
`grpc.aio._call.AioRpcError` (async) — which carries NO HTTP status anywhere.
The only signal is `.code`, and it is a bound METHOD (not a value) returning a
`grpc.StatusCode` enum member whose `.name` is e.g. "INVALID_ARGUMENT" and
whose `.value` is a TUPLE `(3, 'invalid argument')`. Every such failure
previously logged `error_kind=unknown, http_status=0`, so an xAI auth failure /
quota exhaustion / bad-request was indistinguishable from a generic crash.

The fix synthesizes the canonical gRPC→HTTP mapping (`_GRPC_STATUS_MAP`) and is
narrowly gated on three properties this file pins down:

  (1) MODULE GATE (security-critical) — `_grpc_code_name_of` probes `.code()`
      ONLY when `type(exc).__module__` is `grpc` or starts with `grpc.`. A
      hostile / arbitrary exception must never have one of its methods
      INVOKED by pure telemetry, no matter how grpc-shaped it looks. Tests
      here assert non-invocation via a side-effect flag, not just the verdict.
  (2) SHAPE GATE — only a return whose `.name` is a str present in the map is
      accepted; "OK", None, ints, non-str `.name`, unknown names and the
      awaitable `grpc.aio` `Call.code()` would produce are all rejected → 0.
  (3) PRECEDENCE — the branch runs only when `status == 0`, i.e. after the
      name hints and after every HTTP extraction. A real status (an
      `.status_code` attr, google.api_core's int `.code` class attr, a
      `response`, a parsable message) always outranks the synthesized one.

grpcio is NOT a test dependency, so every grpc shape below is hand-built:
classes whose `__module__` is forced to `grpc._channel` / `grpc.aio._call`,
with a `code()` method returning an enum stand-in (`.name` str, `.value`
tuple). Messages mirror the real `_InactiveRpcError` repr so the tests also
prove the last-resort message regex does not hijack the classification.

The classifier is pure telemetry and must stay TOTAL — hostile inputs return
unknown/0 and NEVER raise into the enforcer's except path (GOLDEN RULE: the
SDK may lose a telemetry row, it may never change what the customer's app
sees).
"""
import unittest

from token_police._classify import (
    _GRPC_STATUS_MAP,
    _grpc_classification_of,
    _grpc_code_name_of,
    _http_status_of,
    build_call_outcome,
    classify_exception,
)


# Canonical gRPC status numbers — only used to make the enum stand-in's
# `.value` tuple and the fake debug string realistic.
_GRPC_NUMBERS = {
    "OK": 0,
    "CANCELLED": 1,
    "UNKNOWN": 2,
    "INVALID_ARGUMENT": 3,
    "DEADLINE_EXCEEDED": 4,
    "NOT_FOUND": 5,
    "ALREADY_EXISTS": 6,
    "PERMISSION_DENIED": 7,
    "RESOURCE_EXHAUSTED": 8,
    "FAILED_PRECONDITION": 9,
    "ABORTED": 10,
    "OUT_OF_RANGE": 11,
    "UNIMPLEMENTED": 12,
    "INTERNAL": 13,
    "UNAVAILABLE": 14,
    "DATA_LOSS": 15,
    "UNAUTHENTICATED": 16,
}

# The canonical gRPC→HTTP mapping, written out INDEPENDENTLY of the source
# dict so a silent edit/typo in `_GRPC_STATUS_MAP` is caught rather than
# mirrored. Covers every canonical code except "OK" (not an error).
# NOTE: "UNAUTHENTICATED" is listed here but is currently MISSING from
# `_GRPC_STATUS_MAP` — see TestUnauthenticatedGap at the bottom of the file.
_EXPECTED_MAP = {
    "CANCELLED": ("client_error", 499),
    "UNKNOWN": ("server_error", 500),
    "INVALID_ARGUMENT": ("client_error", 400),
    "DEADLINE_EXCEEDED": ("timeout", 504),
    "NOT_FOUND": ("client_error", 404),
    "ALREADY_EXISTS": ("client_error", 409),
    "PERMISSION_DENIED": ("auth_error", 403),
    "RESOURCE_EXHAUSTED": ("rate_limited", 429),
    "FAILED_PRECONDITION": ("client_error", 400),
    "ABORTED": ("client_error", 409),
    "OUT_OF_RANGE": ("client_error", 400),
    "UNIMPLEMENTED": ("server_error", 501),
    "INTERNAL": ("server_error", 500),
    "UNAVAILABLE": ("server_error", 503),
    "DATA_LOSS": ("server_error", 500),
    "UNAUTHENTICATED": ("auth_error", 401),
}


class _FakeStatusCode:
    """Stand-in for a `grpc.StatusCode` enum member.

    The real member's `.value` is a TUPLE `(3, 'invalid argument')` — the
    classifier must read `.name` (a str), never `.value`.
    """

    def __init__(self, name, number=None, detail=None):
        self.name = name
        if number is None:
            number = _GRPC_NUMBERS.get(name, 2)
        self.value = (number, detail or name.lower().replace("_", " "))

    def __repr__(self):  # pragma: no cover - debugging aid only
        return "StatusCode.%s" % (self.name,)


def _grpc_repr(status_name, details="request rejected by peer"):
    """A faithful `_InactiveRpcError.__str__`. Deliberately contains the words
    "status" and "grpc_status:<n>" — neither may satisfy the classifier's
    last-resort `HTTP|status[:\\s]+(\\d{3})` regex, otherwise a bogus HTTP
    status would pre-empt the gRPC branch (which only runs at status == 0)."""
    number = _GRPC_NUMBERS.get(status_name, 2)
    return (
        "<_InactiveRpcError of RPC that terminated with:\n"
        "\tstatus = StatusCode.%s\n"
        '\tdetails = "%s"\n'
        '\tdebug_error_string = "UNKNOWN:Error received from peer '
        'ipv4:1.2.3.4:443 {grpc_message:\\"%s\\", grpc_status:%d, '
        'created_time:\\"2026-08-10T00:00:00.000000000+00:00\\"}"\n'
        ">" % (status_name, details, details, number)
    )


def _grpc_class(cls_name, module, namespace=None):
    """An exception class whose `__module__` is forced to a grpc-package path
    (that string is the ONLY thing the classifier's gate looks at)."""
    cls = type(cls_name, (Exception,), dict(namespace or {}))
    cls.__module__ = module
    return cls


def _code_method(value, calls):
    """A `.code()` bound method returning `value`, recording every invocation
    on `calls` so tests can assert NON-invocation (the module gate's whole
    point) rather than only the final verdict."""

    def code(self):
        calls.append(value)
        return value

    return code


# Distinct "argument not supplied" sentinel. It must NOT be None: `None` is
# itself one of the junk returns under test (`code()` returning None), and
# using None as the default marker would silently turn that case into the
# valid-StatusCode default instead.
_DEFAULT_CODE = object()


def _grpc_error(status_name, cls_name="_InactiveRpcError",
                module="grpc._channel", msg=None, code_value=_DEFAULT_CODE,
                **attrs):
    """A raised-shaped fake grpc error.

    `code_value` overrides what `.code()` returns (used for the junk-return
    cases) and is passed through VERBATIM — including `None`. Omit it to get
    a valid `_FakeStatusCode` for `status_name`.
    `exc.code_calls` is the invocation ledger.
    """
    calls = []
    value = _FakeStatusCode(status_name) if code_value is _DEFAULT_CODE else code_value
    cls = _grpc_class(cls_name, module, {"code": _code_method(value, calls)})
    exc = cls(_grpc_repr(status_name) if msg is None else msg)
    exc.code_calls = calls
    for k, v in attrs.items():
        setattr(exc, k, v)
    return exc


# ── 1. Happy paths ──────────────────────────────────────────────────────


class TestGrpcHappyPaths(unittest.TestCase):
    """The three real raised shapes, each with a representative status."""

    def test_sync_unary_inactive_rpc_error_invalid_argument(self):
        # xai_sdk sampler rejection: `_InactiveRpcError` from grpc._channel.
        exc = _grpc_error("INVALID_ARGUMENT")
        self.assertEqual(_grpc_code_name_of(exc), "INVALID_ARGUMENT")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "client_error", "http_status": 400})

    def test_sync_streaming_multi_threaded_rendezvous_resource_exhausted(self):
        # Streaming rejection: `_MultiThreadedRendezvous`, same module.
        exc = _grpc_error("RESOURCE_EXHAUSTED",
                          cls_name="_MultiThreadedRendezvous")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_async_aio_rpc_error_permission_denied(self):
        # grpc.aio path: `AioRpcError` lives in grpc.aio._call.
        exc = _grpc_error("PERMISSION_DENIED", cls_name="AioRpcError",
                          module="grpc.aio._call")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 403})

    def test_deadline_exceeded_is_timeout_504(self):
        exc = _grpc_error("DEADLINE_EXCEEDED")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "timeout", "http_status": 504})

    def test_unavailable_is_server_error_503(self):
        exc = _grpc_error("UNAVAILABLE", cls_name="AioRpcError",
                          module="grpc.aio._call")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 503})

    def test_bare_grpc_module_is_gated_in(self):
        # `grpc.RpcError` itself has __module__ == "grpc" (no dot) — the gate's
        # `mod == "grpc"` arm.
        exc = _grpc_error("INTERNAL", cls_name="RpcError", module="grpc")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 500})

    def test_full_status_table(self):
        """Exhaustive: every MAPPED StatusCode, on both a sync and an aio
        shape, against the expected table written out independently.

        (Coverage of the map — that it holds every canonical code — is
        asserted separately in TestUnauthenticatedGap.)
        """
        for name in sorted(_GRPC_STATUS_MAP):
            kind, status = _EXPECTED_MAP[name]
            for cls_name, module in (("_InactiveRpcError", "grpc._channel"),
                                     ("AioRpcError", "grpc.aio._call")):
                with self.subTest(status_code=name, module=module):
                    exc = _grpc_error(name, cls_name=cls_name, module=module)
                    self.assertEqual(
                        classify_exception(exc),
                        {"error_kind": kind, "http_status": status})

    def test_source_map_values_match_expected_table(self):
        # Guards against a silent edit/typo in _GRPC_STATUS_MAP's values.
        for name, value in _GRPC_STATUS_MAP.items():
            with self.subTest(status_code=name):
                self.assertEqual(tuple(value), _EXPECTED_MAP[name])

    def test_ok_is_never_mapped(self):
        self.assertNotIn("OK", _GRPC_STATUS_MAP)

    def test_realistic_grpc_repr_yields_no_http_status(self):
        # The message regex must NOT fire on "status = StatusCode.X" or
        # "grpc_status:3"; if it did, the gRPC branch would be gated off.
        for name in ("INVALID_ARGUMENT", "RESOURCE_EXHAUSTED", "UNAVAILABLE"):
            with self.subTest(status_code=name):
                exc = _grpc_error(name)
                self.assertEqual(_http_status_of(exc), 0)

    def test_helper_returns_classification_dict(self):
        exc = _grpc_error("NOT_FOUND")
        self.assertEqual(_grpc_classification_of(exc),
                         {"error_kind": "client_error", "http_status": 404})

    def test_helper_returns_none_for_non_grpc(self):
        self.assertIsNone(_grpc_classification_of(Exception("boom")))


# ── 2. "OK" and junk returns ────────────────────────────────────────────


class TestNonStatusReturnsStayUnknown(unittest.TestCase):
    """Only a `.name` str present in the map is accepted; everything else is
    junk and must degrade to unknown/0."""

    def _assert_unknown(self, exc):
        self.assertEqual(_grpc_code_name_of(exc), "")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_ok_is_not_an_error(self):
        # "OK" is deliberately absent from the map — a successful status is
        # not an error and must never synthesize a 200-ish classification.
        exc = _grpc_error("OK", code_value=_FakeStatusCode("OK"),
                          msg="terminated with StatusCode.OK")
        self.assertNotIn("OK", _GRPC_STATUS_MAP)
        self._assert_unknown(exc)

    def test_code_returns_none(self):
        # `code_value=None` means code() genuinely returns None (see the
        # _DEFAULT_CODE sentinel). Assert the ledger too, so a helper
        # regression that quietly substitutes a valid StatusCode is caught
        # here rather than passing for the wrong reason.
        exc = _grpc_error("UNKNOWN", code_value=None, msg="rpc failed")
        self._assert_unknown(exc)
        # (_assert_unknown probes twice — via the helper and via
        # classify_exception — so only the contents are pinned, not the count.)
        self.assertTrue(exc.code_calls, "code() was never invoked")
        self.assertEqual(set(exc.code_calls), {None},
                         "code() must have returned None, not a StatusCode")

    def test_code_returns_int(self):
        # A bare int has no `.name`; it must not be read as an HTTP status
        # either (403 here would be a very plausible mis-map).
        self._assert_unknown(_grpc_error("PERMISSION_DENIED", code_value=403,
                                         msg="rpc failed"))

    def test_code_returns_object_with_non_str_name(self):
        class _Weird:
            name = 3  # the enum's NUMBER leaked into `.name`

        self._assert_unknown(_grpc_error("UNKNOWN", code_value=_Weird(),
                                         msg="rpc failed"))

    def test_code_returns_unknown_status_name(self):
        self._assert_unknown(
            _grpc_error("UNKNOWN", code_value=_FakeStatusCode("BOGUS", 99),
                        msg="rpc failed"))

    def test_code_returns_tuple_value_only(self):
        # Guard against a future implementation reading `.value` (a TUPLE)
        # instead of `.name`.
        self._assert_unknown(_grpc_error("UNKNOWN", code_value=(3, "invalid argument"),
                                         msg="rpc failed"))

    def test_code_returns_awaitable(self):
        # grpc.aio's `Call.code()` is a coroutine function on a non-terminal
        # call. The awaitable has no `.name` → rejected, and nothing is ever
        # awaited. (A real coroutine object is avoided here only so the test
        # does not emit a "never awaited" RuntimeWarning.)
        class _Awaitable:
            def __await__(self):  # pragma: no cover - never awaited
                raise AssertionError("classifier must never await")

        self._assert_unknown(_grpc_error("UNAVAILABLE", code_value=_Awaitable(),
                                         msg="rpc failed"))

    def test_code_returns_string_status_name(self):
        # A bare str has no `.name` attribute — not accepted.
        self._assert_unknown(_grpc_error("UNKNOWN", code_value="INVALID_ARGUMENT",
                                         msg="rpc failed"))

    def test_lowercase_status_name_rejected(self):
        self._assert_unknown(
            _grpc_error("UNKNOWN", code_value=_FakeStatusCode("invalid_argument"),
                        msg="rpc failed"))


# ── 3. Module gate (security-critical) ──────────────────────────────────


class TestModuleGate(unittest.TestCase):
    """A non-grpc exception must NEVER have `code()` invoked, however
    convincingly grpc-shaped it is. Asserted by side effect, not by verdict."""

    def _non_grpc(self, module):
        calls = []
        cls = _grpc_class("_InactiveRpcError", module,
                          {"code": _code_method(_FakeStatusCode("INVALID_ARGUMENT"),
                                                calls)})
        exc = cls("rpc failed")
        exc.code_calls = calls
        return exc

    def test_arbitrary_module_code_is_never_invoked(self):
        exc = self._non_grpc("my_app.errors")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})
        self.assertEqual(exc.code_calls, [],
                         "code() must not be invoked on a non-grpc exception")

    def test_near_miss_module_names_are_rejected(self):
        # "grpcio", "notgrpc.…", "grpc_tools.…" all fail
        # `mod == "grpc" or mod.startswith("grpc.")`.
        for module in ("grpcio", "grpcio._channel", "notgrpc._channel",
                       "grpc_tools.protoc", "mygrpc", "xgrpc.channel",
                       "", "builtins"):
            with self.subTest(module=module):
                exc = self._non_grpc(module)
                self.assertEqual(_grpc_code_name_of(exc), "")
                self.assertEqual(classify_exception(exc),
                                 {"error_kind": "unknown", "http_status": 0})
                self.assertEqual(exc.code_calls, [])

    def test_module_gate_is_case_sensitive(self):
        exc = self._non_grpc("GRPC._channel")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})
        self.assertEqual(exc.code_calls, [])

    def test_non_str_module_is_rejected(self):
        calls = []
        cls = type("_InactiveRpcError", (Exception,),
                   {"code": _code_method(_FakeStatusCode("INTERNAL"), calls)})
        cls.__module__ = 12345  # `__module__` may hold any object
        exc = cls("rpc failed")
        self.assertEqual(_grpc_code_name_of(exc), "")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})
        self.assertEqual(calls, [])

    def test_grpc_module_but_code_not_callable(self):
        # A grpc-module exception whose `.code` is a plain (out-of-range) value
        # is not callable → no invocation attempt, no classification.
        cls = _grpc_class("_InactiveRpcError", "grpc._channel")
        exc = cls("rpc failed")
        exc.code = "INVALID_ARGUMENT"
        self.assertEqual(_grpc_code_name_of(exc), "")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})


# ── 4. Precedence — a real HTTP status always wins ──────────────────────


class TestPrecedence(unittest.TestCase):
    """The gRPC branch is gated on `status == 0` and sits AFTER the name
    hints: anything HTTP-derived outranks the synthesized mapping."""

    def test_status_code_attr_outranks_grpc_code(self):
        exc = _grpc_error("INVALID_ARGUMENT", status_code=429)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})
        self.assertEqual(exc.code_calls, [],
                         "code() must not be invoked once a real status exists")

    def test_server_status_attr_outranks_grpc_code(self):
        exc = _grpc_error("INVALID_ARGUMENT", status_code=500)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 500})
        self.assertEqual(exc.code_calls, [])

    def test_response_carrier_outranks_grpc_code(self):
        class _R:
            status_code = 403

        exc = _grpc_error("RESOURCE_EXHAUSTED", response=_R())
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 403})
        self.assertEqual(exc.code_calls, [])

    def test_parsable_message_outranks_grpc_code(self):
        exc = _grpc_error("INVALID_ARGUMENT",
                          msg="proxy rejected the RPC with HTTP 502")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 502})
        self.assertEqual(exc.code_calls, [])

    def test_int_code_attr_on_grpc_module_still_read_as_status(self):
        # Not callable → the direct extraction's LAST attr wins; the gRPC
        # branch is never consulted.
        cls = _grpc_class("_InactiveRpcError", "grpc._channel")
        exc = cls("rpc failed")
        exc.code = 429
        self.assertEqual(_http_status_of(exc), 429)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_name_hint_outranks_grpc_code(self):
        # A class NAME hint ("RateLimitError") is checked before the branch.
        calls = []
        cls = _grpc_class("RateLimitError", "grpc._channel",
                          {"code": _code_method(_FakeStatusCode("INVALID_ARGUMENT"),
                                                calls)})
        self.assertEqual(classify_exception(cls("slow down")),
                         {"error_kind": "rate_limited", "http_status": 429})
        self.assertEqual(calls, [])

    def test_google_api_core_int_code_class_attr_unchanged(self):
        # google.api_core.exceptions.TooManyRequests carries `code = 429` as an
        # INT class attribute (and is not in the grpc package) — no regression.
        cls = type("TooManyRequests", (Exception,), {"code": 429})
        cls.__module__ = "google.api_core.exceptions"
        self.assertEqual(_http_status_of(cls("429 Quota exceeded")), 429)
        self.assertEqual(classify_exception(cls("429 Quota exceeded")),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_google_api_core_forbidden_int_code_class_attr(self):
        cls = type("Forbidden", (Exception,), {"code": 403})
        cls.__module__ = "google.api_core.exceptions"
        self.assertEqual(classify_exception(cls("403 denied")),
                         {"error_kind": "auth_error", "http_status": 403})


# ── 5. Chain traversal ──────────────────────────────────────────────────


class TestChainTraversal(unittest.TestCase):
    """Framework/SDK wrappers hide the grpc error one or more links down."""

    def test_cause_chain_recovers_grpc_code(self):
        inner = _grpc_error("INVALID_ARGUMENT")
        outer = Exception("xai chat completion failed")
        outer.__cause__ = inner
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "client_error", "http_status": 400})

    def test_context_chain_recovers_grpc_code(self):
        inner = _grpc_error("RESOURCE_EXHAUSTED", cls_name="AioRpcError",
                            module="grpc.aio._call")
        outer = Exception("while handling the stream")
        outer.__context__ = inner
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_original_exception_wrapper_attr_recovers_grpc_code(self):
        inner = _grpc_error("PERMISSION_DENIED")
        outer = Exception("wrapped")
        outer.original_exception = inner
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "auth_error", "http_status": 403})

    def test_inner_wrapper_attr_recovers_grpc_code(self):
        inner = _grpc_error("UNAVAILABLE", cls_name="_MultiThreadedRendezvous")
        outer = Exception("wrapped")
        outer.inner = inner
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "server_error", "http_status": 503})

    def test_deep_chain_within_depth_limit(self):
        deep = _grpc_error("DEADLINE_EXCEEDED")
        mid = Exception("l2")
        mid.__cause__ = deep
        outer = Exception("l1")
        outer.__cause__ = mid
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "timeout", "http_status": 504})

    def test_outer_http_status_outranks_chained_grpc_code(self):
        inner = _grpc_error("INVALID_ARGUMENT")
        outer = Exception("wrapped")
        outer.status_code = 503
        outer.__cause__ = inner
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "server_error", "http_status": 503})
        self.assertEqual(inner.code_calls, [])

    def test_chained_http_status_outranks_grpc_code(self):
        # The whole chain is scanned for a real HTTP status BEFORE the gRPC
        # branch runs, even when the grpc link is nearer the top.
        grpc_link = _grpc_error("INVALID_ARGUMENT")
        http_link = Exception("upstream said HTTP 401")
        grpc_link.__cause__ = http_link
        outer = Exception("wrapped")
        outer.__cause__ = grpc_link
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "auth_error", "http_status": 401})

    def test_non_grpc_chain_link_code_is_never_invoked(self):
        calls = []
        cls = _grpc_class("_InactiveRpcError", "my_app.errors",
                          {"code": _code_method(_FakeStatusCode("INTERNAL"), calls)})
        outer = Exception("wrapped")
        outer.__cause__ = cls("rpc failed")
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "unknown", "http_status": 0})
        self.assertEqual(calls, [])

    def test_cyclic_chain_with_grpc_link_terminates(self):
        # `code_value=None` → code() genuinely returns None, so nothing in the
        # cycle is classifiable and the traversal itself is what is under test:
        # it must terminate (cycle-safe, bounded) and land on unknown/0.
        a = _grpc_error("UNKNOWN", code_value=None, msg="a")
        b = Exception("b")
        a.__cause__ = b
        b.__cause__ = a
        self.assertEqual(classify_exception(a),
                         {"error_kind": "unknown", "http_status": 0})

    def test_self_referential_grpc_error_terminates(self):
        exc = _grpc_error("INVALID_ARGUMENT")
        exc.inner = exc
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "client_error", "http_status": 400})


# ── 6. Never raises (GOLDEN RULE) ───────────────────────────────────────


class TestNeverRaises(unittest.TestCase):
    """Every hostile grpc-shaped input degrades to a dict — the classifier may
    never raise into the enforcer's except path."""

    def test_code_method_raises(self):
        def _boom(self):
            raise RuntimeError("UsageError: RPC not yet terminated")

        cls = _grpc_class("_InactiveRpcError", "grpc._channel", {"code": _boom})
        exc = cls("rpc failed")
        self.assertEqual(_grpc_code_name_of(exc), "")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_code_method_raises_arbitrary_exception_types(self):
        for err in (ValueError("bad"), AttributeError("gone"),
                    TypeError("nope"), OSError("socket closed")):
            with self.subTest(error=type(err).__name__):
                def _boom(self, _e=err):
                    raise _e

                cls = _grpc_class("_InactiveRpcError", "grpc._channel",
                                  {"code": _boom})
                self.assertEqual(classify_exception(cls("rpc failed")),
                                 {"error_kind": "unknown", "http_status": 0})

    def test_code_property_raises(self):
        def _boom(self):
            raise RuntimeError("nope")

        cls = _grpc_class("_InactiveRpcError", "grpc._channel",
                          {"code": property(_boom)})
        exc = cls("rpc failed")
        self.assertEqual(_grpc_code_name_of(exc), "")
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_status_code_name_property_raises(self):
        class _HostileStatus:
            @property
            def name(self):
                raise RuntimeError("nope")

        exc = _grpc_error("UNKNOWN", code_value=_HostileStatus(), msg="rpc failed")
        self.assertEqual(_grpc_code_name_of(exc), "")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_no_code_attribute_at_all(self):
        cls = _grpc_class("_InactiveRpcError", "grpc._channel")
        self.assertEqual(_grpc_code_name_of(cls("rpc failed")), "")
        self.assertEqual(classify_exception(cls("rpc failed")),
                         {"error_kind": "unknown", "http_status": 0})

    def test_hostile_str_on_grpc_error_still_classifies(self):
        # A raising __str__ kills the message regex but must not stop the
        # StatusCode mapping.
        def _bad_str(self):
            raise RuntimeError("no str")

        calls = []
        cls = _grpc_class(
            "_InactiveRpcError", "grpc._channel",
            {"code": _code_method(_FakeStatusCode("INVALID_ARGUMENT"), calls),
             "__str__": _bad_str})
        self.assertEqual(classify_exception(cls()),
                         {"error_kind": "client_error", "http_status": 400})

    def test_hostile_wrapper_attr_property_raises(self):
        class _Hostile(Exception):
            @property
            def original_exception(self):
                raise RuntimeError("nope")

        out = classify_exception(_Hostile("boom"))
        self.assertEqual(out, {"error_kind": "unknown", "http_status": 0})

    def test_type_lookup_hostile_shapes_return_a_dict(self):
        for exc in (_grpc_error("INVALID_ARGUMENT"),
                    _grpc_error("OK", code_value=_FakeStatusCode("OK")),
                    _grpc_error("UNKNOWN", code_value=object(), msg="x")):
            with self.subTest(exc=type(exc).__name__):
                out = classify_exception(exc)
                self.assertIsInstance(out, dict)
                self.assertIn("error_kind", out)
                self.assertIn("http_status", out)


# ── 7. No regression on the pre-existing hostile `.code` shapes ─────────


class TestExistingCodeAttrShapesUnchanged(unittest.TestCase):
    """The bedrock/`.code` suite's expectations must survive verbatim — the
    new branch only runs at status == 0 and only inside the grpc package, so
    none of these NON-grpc shapes may change."""

    def test_str_code_never_coerced(self):
        # litellm attaches a STRING code.
        exc = Exception("boom")
        exc.code = "429"
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_bool_code_rejected(self):
        exc = Exception("boom")
        exc.code = True
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_websocket_close_code_rejected(self):
        for close_code in (1000, 1006):
            with self.subTest(close_code=close_code):
                exc = Exception("boom")
                exc.code = close_code
                self.assertEqual(_http_status_of(exc), 0)
                self.assertEqual(classify_exception(exc),
                                 {"error_kind": "unknown", "http_status": 0})

    def test_small_expat_style_code_rejected(self):
        exc = Exception("boom")
        exc.code = 7
        self.assertEqual(_http_status_of(exc), 0)

    def test_int_code_still_recovered(self):
        exc = Exception("boom")
        exc.code = 429
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_callable_code_on_non_grpc_exception_stays_unknown(self):
        # The pre-fix behavior for a method-valued `.code` outside grpc.
        exc = Exception("boom")
        exc.code = lambda: 429
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_plain_exception_still_unknown(self):
        self.assertEqual(classify_exception(Exception("something odd")),
                         {"error_kind": "unknown", "http_status": 0})

    def test_message_regex_fallback_still_runs(self):
        self.assertEqual(_http_status_of(Exception("upstream returned HTTP 503")), 503)

    def test_name_based_classification_still_wins_without_status(self):
        exc = type("RateLimitError", (Exception,), {})("slow down")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_botocore_dict_response_still_recovered(self):
        exc = type("AccessDeniedException", (Exception,), {})("denied")
        exc.response = {"Error": {"Code": "AccessDeniedException"},
                        "ResponseMetadata": {"HTTPStatusCode": 403}}
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 403})

    def test_huggingface_http_response_still_recovered(self):
        exc = type("InferenceClientProviderApiError", (Exception,), {})("boom")
        exc.httpResponse = {"requestId": "x", "status": 401, "body": {}}
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 401})


# ── 8. build_call_outcome integration ───────────────────────────────────


class TestBuildCallOutcome(unittest.TestCase):
    """End to end through the payload builder the enforcer actually calls."""

    def test_grpc_invalid_argument_failure_row(self):
        out = build_call_outcome(_grpc_error("INVALID_ARGUMENT"), 12)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["duration_ms"], 12)
        self.assertEqual(out["error_kind"], "client_error")
        self.assertEqual(out["http_status"], 400)

    def test_grpc_resource_exhausted_failure_row(self):
        out = build_call_outcome(
            _grpc_error("RESOURCE_EXHAUSTED", cls_name="AioRpcError",
                        module="grpc.aio._call"), 7)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["error_kind"], "rate_limited")
        self.assertEqual(out["http_status"], 429)

    def test_hostile_grpc_error_still_builds_a_row(self):
        def _boom(self):
            raise RuntimeError("nope")

        cls = _grpc_class("_InactiveRpcError", "grpc._channel", {"code": _boom})
        out = build_call_outcome(cls("rpc failed"), 3)
        self.assertEqual(out["status"], "failed")
        self.assertEqual(out["error_kind"], "unknown")
        self.assertEqual(out["http_status"], 0)


# ── 9. KNOWN GAP — UNAUTHENTICATED is missing from the map ──────────────


class TestUnauthenticatedGap(unittest.TestCase):
    """EXPECTED TO FAIL against the fix as currently written.

    gRPC defines 17 canonical codes; `_GRPC_STATUS_MAP` holds 15 — it omits
    "OK" (correct: not an error) and "UNAUTHENTICATED" (code 16), which is the
    status a gRPC server returns for a MISSING OR INVALID API KEY. For native
    xai_sdk that is the single most important rejection to classify (it is the
    exact arm the auth-failure matrix drives with a poisoned credential), and
    it is precisely the case the fix leaves at unknown/0.

    PERMISSION_DENIED (7, "authenticated but not allowed") is mapped to
    auth_error/403; UNAUTHENTICATED (16, "not authenticated") is the 401 twin
    and should map to auth_error/401. Do NOT weaken these assertions — add
    `"UNAUTHENTICATED": ("auth_error", 401)` to `_GRPC_STATUS_MAP` instead
    (`_EXPECTED_MAP` at the top of this file already lists it, so no test
    edit is needed once the source is fixed).
    """

    def test_map_covers_every_canonical_non_ok_status(self):
        self.assertEqual(set(_GRPC_STATUS_MAP), set(_EXPECTED_MAP))
        self.assertEqual(set(_EXPECTED_MAP), set(_GRPC_NUMBERS) - {"OK"})

    def test_unauthenticated_is_auth_error_401(self):
        exc = _grpc_error("UNAUTHENTICATED")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 401})

    def test_unauthenticated_async_is_auth_error_401(self):
        exc = _grpc_error("UNAUTHENTICATED", cls_name="AioRpcError",
                          module="grpc.aio._call")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 401})


if __name__ == "__main__":
    unittest.main()
