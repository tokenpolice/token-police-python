"""Harmonize the two SDKs' exception-classifier chain traversal to their
union. Python gains the Node-shaped camel/raw direct-status attrs
(status / statusCode / httpStatus) so a wrapper exposing e.g. ``{status: 429}``
is caught symmetrically. These tests exercise the PUBLIC ``classify_exception``
surface only.

``_classify`` is pure telemetry (drives call_outcome.{error_kind,http_status}
on /log). It must stay TOTAL / never-raise — the classifier is @fail_safe
externally, but must not rely on it.
"""
import unittest

from token_police._classify import classify_exception


class _AttrError(Exception):
    """An exception that exposes arbitrary attributes (Node-shaped wrappers)."""

    def __init__(self, msg="boom", **attrs):
        super().__init__(msg)
        for k, v in attrs.items():
            setattr(self, k, v)


class TestNodeShapedStatusAttrs(unittest.TestCase):
    # A5 — {status: 429} (camel/raw attr, not status_code) now detected.
    def test_status_attr_detected(self):
        self.assertEqual(
            classify_exception(_AttrError(status=429)),
            {"error_kind": "rate_limited", "http_status": 429},
        )

    # A6 — statusCode / httpStatus attrs also detected, symmetric to Node.
    def test_status_code_camel_attr_detected(self):
        self.assertEqual(
            classify_exception(_AttrError(statusCode=503)),
            {"error_kind": "server_error", "http_status": 503},
        )

    def test_http_status_camel_attr_detected(self):
        self.assertEqual(
            classify_exception(_AttrError(httpStatus=401)),
            {"error_kind": "auth_error", "http_status": 401},
        )

    # A9 — response.* trio symmetric with Node: response.status /
    # response.statusCode now read (previously only response.status_code).
    # RED before adding those two attrs to _direct_status_of's response block.
    def test_response_status_detected(self):
        class _Resp:
            status = 503

        self.assertEqual(
            classify_exception(_AttrError(response=_Resp())),
            {"error_kind": "server_error", "http_status": 503},
        )

    def test_response_status_code_camel_detected(self):
        class _Resp:
            statusCode = 401

        self.assertEqual(
            classify_exception(_AttrError(response=_Resp())),
            {"error_kind": "auth_error", "http_status": 401},
        )

    def test_response_status_code_snake_still_detected(self):
        class _Resp:
            status_code = 429

        self.assertEqual(
            classify_exception(_AttrError(response=_Resp())),
            {"error_kind": "rate_limited", "http_status": 429},
        )


class TestNoRegression(unittest.TestCase):
    # A7 — existing top-level detection byte-identical.
    def test_top_level_status_code_still_wins(self):
        self.assertEqual(
            classify_exception(_AttrError(status_code=429)),
            {"error_kind": "rate_limited", "http_status": 429},
        )

    def test_top_level_message_still_parsed(self):
        self.assertEqual(
            classify_exception(Exception("Request failed HTTP 400")),
            {"error_kind": "client_error", "http_status": 400},
        )

    # A11 — unknown/unclassifiable still -> unknown/0.
    def test_unknown_stays_unknown(self):
        self.assertEqual(
            classify_exception(Exception("something odd happened")),
            {"error_kind": "unknown", "http_status": 0},
        )


class TestNoThrow(unittest.TestCase):
    # A8 — no-throw on hostile inputs.
    def test_throwing_getter_does_not_raise(self):
        class _Hostile(Exception):
            @property
            def inner(self):
                raise RuntimeError("hostile getter")

        out = classify_exception(_Hostile("boom"))
        self.assertIn("error_kind", out)
        self.assertIn("http_status", out)

    def test_self_referential_terminates(self):
        o = _AttrError("loop")
        o.inner = o  # cycle via wrapper attr
        out = classify_exception(o)
        self.assertEqual(out, {"error_kind": "unknown", "http_status": 0})

    def test_none_like_message_no_throw(self):
        # a raising __str__ must not escape
        class _BadStr(Exception):
            def __str__(self):
                raise RuntimeError("no str")

        out = classify_exception(_BadStr())
        self.assertIn("error_kind", out)
        self.assertIn("http_status", out)


if __name__ == "__main__":
    unittest.main()
