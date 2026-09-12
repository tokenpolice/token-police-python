"""F-16-3b — classifier reads botocore/`.code` shapes (_classify._direct_status_of).

Two additions, both narrowly gated:
  (a) `"code"` appended LAST to the direct-attr tuple, guarded
      `isinstance(v, int) and not isinstance(v, bool) and 100 <= v < 600`.
      Last so a real `.status_code`/`.status` always outranks a body `.code`
      (openai errors carry both); the guard rejects the hostile shapes `.code`
      attracts — litellm's str `"429"` (never coerced), booleans, websocket
      close codes (1006), grpc method objects.
  (b) a dict-shaped `exc.response` is read at
      `response["ResponseMetadata"]["HTTPStatusCode"]` — the botocore
      ClientError shape, which carries NO status attribute anywhere else. This
      is what made a Bedrock AccessDeniedException classify unknown/0 instead
      of auth_error/403 on the failure row.

Both passes run only when the earlier extraction yielded nothing, so every
previously-classified exception classifies identically. The classifier is pure
telemetry and must stay TOTAL — hostile inputs return unknown/0, never raise
(GOLDEN RULE).
"""
import unittest

from token_police._classify import classify_exception, _http_status_of


def _client_error(cls_name, code, http_status, msg="denied"):
    """Synthetic botocore ClientError: the real class stores the parsed error
    document on `.response` as a plain dict and exposes no status attribute."""
    exc_cls = type(cls_name, (Exception,), {})
    exc = exc_cls(msg)
    exc.response = {
        "Error": {"Code": code, "Message": msg},
        "ResponseMetadata": {"HTTPStatusCode": http_status,
                             "RequestId": "abc-123"},
    }
    return exc


class TestBotocoreDictResponse(unittest.TestCase):
    """(b) ClientError.response is a dict → ResponseMetadata.HTTPStatusCode."""

    def test_bedrock_access_denied_is_auth_error_403(self):
        exc = _client_error("AccessDeniedException", "AccessDeniedException", 403,
                            "User is not authorized to perform bedrock:InvokeModel")
        self.assertEqual(_http_status_of(exc), 403)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 403})

    def test_bedrock_throttling_is_rate_limited_429(self):
        exc = _client_error("ThrottlingException", "ThrottlingException", 429,
                            "Too many requests")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_bedrock_validation_is_client_error_400(self):
        exc = _client_error("ValidationException", "ValidationException", 400,
                            "malformed input")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "client_error", "http_status": 400})

    def test_bedrock_internal_is_server_error_500(self):
        exc = _client_error("InternalServerException", "InternalServerException", 500)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 500})

    def test_missing_response_metadata_stays_unknown(self):
        exc = Exception("boom")
        exc.response = {"Error": {"Code": "AccessDeniedException"}}
        self.assertEqual(_http_status_of(exc), 0)

    def test_non_int_http_status_code_rejected(self):
        exc = Exception("boom")
        exc.response = {"ResponseMetadata": {"HTTPStatusCode": "403"}}
        self.assertEqual(_http_status_of(exc), 0)

    def test_out_of_range_http_status_code_rejected(self):
        exc = Exception("boom")
        exc.response = {"ResponseMetadata": {"HTTPStatusCode": 99}}
        self.assertEqual(_http_status_of(exc), 0)

    def test_response_metadata_not_a_dict_is_safe(self):
        exc = Exception("boom")
        exc.response = {"ResponseMetadata": [403]}
        self.assertEqual(_http_status_of(exc), 0)

    def test_object_response_attr_still_wins_over_dict_branch(self):
        # Pre-existing object-shaped .response path is untouched.
        class _R:
            status_code = 503

        exc = Exception("boom")
        exc.response = _R()
        self.assertEqual(_http_status_of(exc), 503)


class TestCodeAttr(unittest.TestCase):
    """(a) `.code` as a LAST-resort direct attr, int-only and range-guarded."""

    def test_int_code_recovered(self):
        exc = Exception("boom")
        exc.code = 429  # google.genai APIError.code shape (response=None path)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_str_code_never_coerced(self):
        # litellm attaches a STRING code; coercing it would be a behavior change.
        exc = Exception("boom")
        exc.code = "429"
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_bool_code_rejected(self):
        # bool is an int subclass — True would otherwise sneak past `isinstance`
        # (though not the range guard); assert the explicit bool rejection.
        exc = Exception("boom")
        exc.code = True
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_websocket_close_code_rejected(self):
        exc = Exception("boom")
        exc.code = 1006  # abnormal websocket closure, not an HTTP status
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_small_expat_style_code_rejected(self):
        exc = Exception("boom")
        exc.code = 7
        self.assertEqual(_http_status_of(exc), 0)

    def test_status_code_outranks_code(self):
        # `code` is LAST in the tuple: a real status always wins.
        exc = Exception("boom")
        exc.status_code = 500
        exc.code = 400
        self.assertEqual(_http_status_of(exc), 500)
        self.assertEqual(classify_exception(exc)["error_kind"], "server_error")

    def test_status_outranks_code(self):
        exc = Exception("boom")
        exc.status = 401
        exc.code = 429
        self.assertEqual(_http_status_of(exc), 401)

    def test_response_dict_only_consulted_when_attrs_empty(self):
        exc = Exception("boom")
        exc.status_code = 400
        exc.response = {"ResponseMetadata": {"HTTPStatusCode": 403}}
        self.assertEqual(_http_status_of(exc), 400)


class TestNeverRaises(unittest.TestCase):
    """Hostile inputs degrade to unknown/0 — the classifier is never allowed to
    raise into the enforcer's except path (GOLDEN RULE)."""

    def test_hostile_response_dict_get_raises(self):
        class _HostileDict(dict):
            def get(self, *a, **k):
                raise RuntimeError("nope")

        exc = Exception("boom")
        exc.response = _HostileDict(ResponseMetadata={"HTTPStatusCode": 403})
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_hostile_code_property_raises(self):
        class Hostile(Exception):
            @property
            def code(self):
                raise RuntimeError("nope")

        self.assertEqual(_http_status_of(Hostile("boom")), 0)
        self.assertEqual(classify_exception(Hostile("boom")),
                         {"error_kind": "unknown", "http_status": 0})

    def test_hostile_response_property_raises(self):
        class Hostile(Exception):
            @property
            def response(self):
                raise RuntimeError("nope")

        self.assertEqual(classify_exception(Hostile("boom")),
                         {"error_kind": "unknown", "http_status": 0})


class TestExistingBehaviorUnchanged(unittest.TestCase):
    """Regression guard: neither addition may perturb an already-classified
    exception."""

    def test_status_code_attr_still_rate_limited(self):
        exc = Exception("boom")
        exc.status_code = 429
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_name_based_classification_still_wins_without_status(self):
        exc = type("RateLimitError", (Exception,), {})("slow down")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_plain_exception_still_unknown(self):
        self.assertEqual(classify_exception(Exception("something odd")),
                         {"error_kind": "unknown", "http_status": 0})

    def test_message_regex_fallback_still_runs(self):
        self.assertEqual(_http_status_of(Exception("upstream returned HTTP 503")), 503)

    def test_chained_client_error_recovered_through_wrapper(self):
        # The dict-response read also works on a CHAINED link (the chain pass
        # re-runs _direct_status_of on each wrapper).
        inner = _client_error("ThrottlingException", "ThrottlingException", 429)
        outer = Exception("stream failed")
        outer.__cause__ = inner
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "rate_limited", "http_status": 429})


if __name__ == "__main__":
    unittest.main()
