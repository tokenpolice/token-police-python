"""Error-classification hardening (_classify.py).

Two recovery passes were added behind a narrow gate (they only run when the
direct attribute extraction yields no status):
  (a) exception-CHAIN traversal — wrapped streaming failures (e.g.
      huggingface_hub wraps the 400-carrying HTTPError) expose the status only
      on an inner exception reachable via __cause__/__context__/.inner;
  (b) message regex — "HTTP 400" / "status code 429" parsed as a last resort.

The recovered status must re-derive error_kind (client_error / rate_limited /
auth_error / server_error), and nothing here may ever raise (GOLDEN RULE).
"""
import unittest

from token_police._classify import classify_exception, _http_status_of


class _StatusError(Exception):
    def __init__(self, msg="boom", status_code=None):
        super().__init__(msg)
        if status_code is not None:
            self.status_code = status_code


class _RespError(Exception):
    def __init__(self, msg, status):
        super().__init__(msg)
        class _R:
            pass
        r = _R()
        r.status_code = status
        self.response = r


class TestDirectExtractionUnchanged(unittest.TestCase):
    def test_direct_status_code_still_wins(self):
        self.assertEqual(_http_status_of(_StatusError(status_code=429)), 429)

    def test_direct_response_status_still_wins(self):
        self.assertEqual(_http_status_of(_RespError("x", 503)), 503)

    def test_direct_status_beats_chained_status(self):
        outer = _StatusError("outer", status_code=500)
        outer.__cause__ = _StatusError("inner", status_code=400)
        self.assertEqual(_http_status_of(outer), 500)


class TestChainTraversal(unittest.TestCase):
    def test_cause_chain_recovers_status(self):
        # huggingface streaming pattern: generic wrapper, inner carries the 400.
        outer = Exception("stream failed")
        outer.__cause__ = _StatusError("Bad Request", status_code=400)
        self.assertEqual(_http_status_of(outer), 400)
        self.assertEqual(classify_exception(outer)["error_kind"], "client_error")

    def test_context_chain_recovers_status(self):
        outer = Exception("while handling")
        outer.__context__ = _RespError("denied", 403)
        info = classify_exception(outer)
        self.assertEqual(info["http_status"], 403)
        self.assertEqual(info["error_kind"], "auth_error")

    def test_inner_attr_wrapper_recovers_status(self):
        outer = Exception("wrapped")
        outer.inner = _StatusError("Too Many Requests", status_code=429)
        info = classify_exception(outer)
        self.assertEqual(info["http_status"], 429)
        self.assertEqual(info["error_kind"], "rate_limited")

    def test_deep_chain_within_depth_limit(self):
        e1 = Exception("l1")
        e2 = Exception("l2")
        e3 = _StatusError("l3", status_code=502)
        e1.__cause__ = e2
        e2.__cause__ = e3
        info = classify_exception(e1)
        self.assertEqual(info["http_status"], 502)
        self.assertEqual(info["error_kind"], "server_error")

    def test_cyclic_chain_is_safe(self):
        a = Exception("a")
        b = Exception("b")
        a.__cause__ = b
        b.__cause__ = a
        # Must terminate, not hang or raise.
        self.assertEqual(_http_status_of(a), 0)
        self.assertEqual(classify_exception(a)["error_kind"], "unknown")

    def test_non_exception_inner_attr_ignored(self):
        outer = Exception("wrapped")
        outer.inner = {"status_code": 400}  # not a BaseException → skipped
        self.assertEqual(_http_status_of(outer), 0)


class TestMessageRegexFallback(unittest.TestCase):
    def test_http_status_in_message(self):
        info = classify_exception(Exception("Request failed: HTTP 400 Bad Request"))
        self.assertEqual(info["http_status"], 400)
        self.assertEqual(info["error_kind"], "client_error")

    def test_status_code_phrase_in_message(self):
        info = classify_exception(Exception("upstream returned status code 503"))
        self.assertEqual(info["http_status"], 503)
        self.assertEqual(info["error_kind"], "server_error")

    def test_status_colon_in_message(self):
        info = classify_exception(Exception("request error, status: 429"))
        self.assertEqual(info["http_status"], 429)
        self.assertEqual(info["error_kind"], "rate_limited")

    def test_chained_message_also_parsed(self):
        outer = Exception("stream aborted")
        outer.__cause__ = Exception("server replied HTTP 401")
        info = classify_exception(outer)
        self.assertEqual(info["http_status"], 401)
        self.assertEqual(info["error_kind"], "auth_error")

    def test_out_of_range_number_rejected(self):
        self.assertEqual(_http_status_of(Exception("HTTP 999 nonsense")), 0)

    def test_unrelated_number_not_matched(self):
        # A bare number without the HTTP/status anchor must not be parsed.
        self.assertEqual(_http_status_of(Exception("processed 404 records")), 0)

    def test_message_regex_only_runs_when_direct_is_zero(self):
        # Direct status wins over a contradictory message.
        exc = _StatusError("got HTTP 500 from upstream", status_code=429)
        self.assertEqual(_http_status_of(exc), 429)


class TestNeverRaises(unittest.TestCase):
    def test_hostile_exception_attributes(self):
        class Hostile(Exception):
            @property
            def status_code(self):
                raise RuntimeError("nope")

            @property
            def inner(self):
                raise RuntimeError("nope")

            def __str__(self):
                raise RuntimeError("nope")

        info = classify_exception(Hostile())
        self.assertIn("error_kind", info)

    def test_plain_unknown_unchanged(self):
        info = classify_exception(Exception("something odd"))
        self.assertEqual(info, {"error_kind": "unknown", "http_status": 0})


if __name__ == "__main__":
    unittest.main()
