"""F-17-C — classifier reads the `httpResponse.status` shape (_classify._direct_status_of).

`@huggingface/inference` v4 errors (`InferenceClientProviderApiError` /
`InferenceClientHubApiError`, both extending `InferenceClientHttpRequestError`)
carry the HTTP status ONLY on an `httpResponse` bag
(`{requestId, status, body}`): no `status_code`/`status` attribute anywhere
else, no `.response`, and a message ("Failed to perform inference: Invalid
credentials in Authorization header") with no parsable three-digit code for the
message regex to recover. Every such failure classified unknown/0.

The read is appended LAST in `_direct_status_of` — after the direct-attr tuple
and after the object/dict `.response` passes — so nothing already classified
changes. Both carrier shapes are handled (a dict `{"status": 401}` and an
object with `.status`, since Python shims re-materialize the JS bag either
way), guarded `isinstance(v, int) and not isinstance(v, bool) and
100 <= v < 600` so the non-status shapes the bag holds (`requestId`, a string
status, the body payload) can never be mistaken for one. Only `.status` is
read — `httpResponse.body` is the raw provider error payload and must never be
touched (token-metadata-only invariant).

The classifier is pure telemetry and must stay TOTAL — hostile inputs return
unknown/0, never raise (GOLDEN RULE).
"""
import unittest

from token_police._classify import classify_exception, _http_status_of


class _HttpResponse:
    """Object-shaped carrier: the JS `{requestId, status, body}` bag
    re-materialized as an object with attributes."""

    def __init__(self, status, request_id="x", body=None):
        self.requestId = request_id
        self.status = status
        self.body = body if body is not None else {}


def _hf_error(cls_name, status, msg, carrier="object"):
    """Synthetic @huggingface/inference v4 error: status lives only on
    `httpResponse`. The real class sets a SHORT `.name` ("ProviderApiError")
    while the class name is the long "InferenceClient…" form — neither is in a
    classifier name set, so the status read is the only signal."""
    exc_cls = type(cls_name, (Exception,), {})
    exc = exc_cls(msg)
    if carrier == "dict":
        exc.httpResponse = {"requestId": "x", "status": status, "body": {}}
    else:
        exc.httpResponse = _HttpResponse(status)
    return exc


class TestHttpResponseObjectCarrier(unittest.TestCase):
    """Object-shaped `httpResponse` → `.status`."""

    def test_provider_api_error_401_is_auth_error(self):
        exc = _hf_error(
            "InferenceClientProviderApiError", 401,
            "Failed to perform inference: Invalid credentials in Authorization header")
        self.assertEqual(_http_status_of(exc), 401)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 401})

    def test_rate_limited_429(self):
        exc = _hf_error("InferenceClientProviderApiError", 429,
                        "Failed to perform inference: rate limited")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_server_error_500(self):
        exc = _hf_error("InferenceClientProviderApiError", 500,
                        "Failed to perform inference: server error")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 500})

    def test_client_error_400(self):
        exc = _hf_error("InferenceClientProviderApiError", 400,
                        "Failed to perform inference: Input validation error")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "client_error", "http_status": 400})


class TestHttpResponseDictCarrier(unittest.TestCase):
    """Dict-shaped `httpResponse` — a plain dict has none of the attrs, so the
    mapping read is what recovers it."""

    def test_hub_api_error_403_is_auth_error(self):
        exc = _hf_error(
            "InferenceClientHubApiError", 403,
            "Failed to perform inference: insufficient permissions for this repo",
            carrier="dict")
        self.assertEqual(_http_status_of(exc), 403)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 403})

    def test_dict_carrier_rate_limited_429(self):
        exc = _hf_error("InferenceClientProviderApiError", 429,
                        "Failed to perform inference: rate limited", carrier="dict")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_dict_carrier_server_error_500(self):
        exc = _hf_error("InferenceClientProviderApiError", 500,
                        "Failed to perform inference: server error", carrier="dict")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 500})

    def test_dict_carrier_client_error_400(self):
        exc = _hf_error("InferenceClientProviderApiError", 400,
                        "Failed to perform inference: Input validation error",
                        carrier="dict")
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "client_error", "http_status": 400})


class TestHostileShapesRejected(unittest.TestCase):
    """The bag holds non-status values too — only an in-range int passes."""

    def test_str_status_never_coerced(self):
        exc = _hf_error("InferenceClientProviderApiError", "401", "boom")
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_bool_status_rejected(self):
        # bool is an int subclass — True would otherwise sneak past `isinstance`.
        exc = _hf_error("InferenceClientProviderApiError", True, "boom")
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_out_of_range_status_rejected(self):
        for v in (99, 600):
            exc = _hf_error("InferenceClientProviderApiError", v, "boom")
            self.assertEqual(_http_status_of(exc), 0)
            self.assertEqual(classify_exception(exc),
                             {"error_kind": "unknown", "http_status": 0})

    def test_none_status_rejected(self):
        exc = _hf_error("InferenceClientProviderApiError", None, "boom")
        self.assertEqual(_http_status_of(exc), 0)

    def test_dict_carrier_without_status_key(self):
        exc = Exception("boom")
        exc.httpResponse = {"requestId": "x", "body": {"error": "nope"}}
        self.assertEqual(_http_status_of(exc), 0)

    def test_http_response_none_or_str(self):
        for bag in (None, "401"):
            exc = Exception("boom")
            exc.httpResponse = bag
            self.assertEqual(_http_status_of(exc), 0)
            self.assertEqual(classify_exception(exc),
                             {"error_kind": "unknown", "http_status": 0})

    def test_missing_http_response_stays_unknown(self):
        self.assertEqual(classify_exception(Exception("some failure")),
                         {"error_kind": "unknown", "http_status": 0})


class TestNeverRaises(unittest.TestCase):
    """Hostile carriers degrade to unknown/0 — the classifier is never allowed
    to raise into the enforcer's except path (GOLDEN RULE)."""

    def test_hostile_http_response_property_raises(self):
        class Hostile(Exception):
            @property
            def httpResponse(self):
                raise RuntimeError("nope")

        self.assertEqual(_http_status_of(Hostile("boom")), 0)
        self.assertEqual(classify_exception(Hostile("boom")),
                         {"error_kind": "unknown", "http_status": 0})

    def test_hostile_status_property_on_carrier_raises(self):
        class HostileBag:
            @property
            def status(self):
                raise RuntimeError("nope")

        exc = Exception("boom")
        exc.httpResponse = HostileBag()
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})

    def test_hostile_dict_get_raises(self):
        class _HostileDict(dict):
            def get(self, *a, **k):
                raise RuntimeError("nope")

        exc = Exception("boom")
        exc.httpResponse = _HostileDict(status=401)
        self.assertEqual(_http_status_of(exc), 0)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "unknown", "http_status": 0})


class TestPrecedenceUnchanged(unittest.TestCase):
    """The read is LAST: every pre-existing source still outranks it."""

    def test_status_code_attr_outranks_http_response(self):
        exc = _hf_error("InferenceClientProviderApiError", 401, "boom")
        exc.status_code = 500
        self.assertEqual(_http_status_of(exc), 500)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "server_error", "http_status": 500})

    def test_response_object_outranks_http_response(self):
        class _R:
            status_code = 429

        exc = _hf_error("InferenceClientProviderApiError", 401, "boom")
        exc.response = _R()
        self.assertEqual(_http_status_of(exc), 429)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "rate_limited", "http_status": 429})

    def test_botocore_dict_response_outranks_http_response(self):
        exc = _hf_error("InferenceClientProviderApiError", 500, "boom")
        exc.response = {"ResponseMetadata": {"HTTPStatusCode": 403}}
        self.assertEqual(_http_status_of(exc), 403)
        self.assertEqual(classify_exception(exc),
                         {"error_kind": "auth_error", "http_status": 403})

    def test_chained_hf_error_recovered_through_wrapper(self):
        # The chain pass re-runs _direct_status_of on each wrapper link.
        inner = _hf_error(
            "InferenceClientProviderApiError", 401,
            "Failed to perform inference: Invalid credentials in Authorization header")
        outer = Exception("stream failed")
        outer.__cause__ = inner
        self.assertEqual(classify_exception(outer),
                         {"error_kind": "auth_error", "http_status": 401})


if __name__ == "__main__":
    unittest.main()
