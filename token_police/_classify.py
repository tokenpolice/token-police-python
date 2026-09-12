"""
Provider-agnostic LLM exception classifier.

Used by the enforcer to capture a `call_outcome.error_kind` and
`http_status` when the original LLM call raises. The classifier is wrapped
in @fail_safe externally — internal errors here MUST NEVER reach customer
code. It returns sensible defaults on any uncertainty.

Returns:
    {
      "error_kind": one of {"rate_limited", "timeout", "auth_error",
                            "server_error", "client_error", "network_error",
                            "unknown"},
      "http_status": int (0 if not derivable; for gRPC errors, which carry
                          no HTTP status, the canonical gRPC→HTTP mapping is
                          synthesized),
    }
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict


_RATE_LIMIT_NAMES = {"RateLimitError", "RateLimit", "TooManyRequests"}
_TIMEOUT_NAMES = {"Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout", "APITimeoutError", "APIConnectionTimeoutError"}
_AUTH_NAMES = {"AuthenticationError", "PermissionDeniedError", "Unauthorized", "InvalidAPIKey", "Forbidden"}
_BADREQ_NAMES = {"BadRequestError", "InvalidRequestError", "UnprocessableEntityError"}
_SERVER_NAMES = {"InternalServerError", "APIError", "ServiceUnavailable", "BadGateway"}
_NETWORK_NAMES = {"APIConnectionError", "ConnectionError", "NetworkError"}


def _name_of(exc: BaseException) -> str:
    try:
        return type(exc).__name__
    except Exception:
        return ""


def _direct_status_of(exc: BaseException) -> int:
    # Most LLM SDKs surface .status_code (OpenAI, Anthropic) or .response.status_code.
    # The camel/raw variants (status/statusCode/httpStatus) mirror the Node
    # classifier's attribute set so Node-shaped `{status: 429}` wrappers are
    # classified symmetrically.
    # `code` is deliberately LAST: the loop returns the first in-range int, and
    # e.g. openai errors carry both `.status_code` and a body `.code` — `.code`
    # must never outrank a real status. It covers google.genai's APIError.code
    # on its response=None paths. The int-not-bool + range guard rejects the
    # hostile shapes `.code` attracts: litellm's str "429" (never coerced),
    # grpc's method-valued `.code`, websocket close codes 1000-4999, expat
    # codes 1-40.
    for attr in ("status_code", "http_status", "status", "statusCode", "httpStatus", "code"):
        try:
            v = getattr(exc, attr, None)
            if isinstance(v, int) and not isinstance(v, bool) and 100 <= v < 600:
                return v
        except Exception:
            pass
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            # Read the same response.* trio as the Node classifier
            # (response.{status,statusCode,status_code}) so a JS-shaped response
            # object is caught symmetrically.
            for attr in ("status", "statusCode", "status_code"):
                try:
                    v = getattr(resp, attr, None)
                    if isinstance(v, int) and not isinstance(v, bool) and 100 <= v < 600:
                        return v
                except Exception:
                    continue
            # botocore ClientError.response is a plain dict — the real status
            # sits at ResponseMetadata.HTTPStatusCode (a dict has none of the
            # attrs above, so this only runs when the attr reads yielded
            # nothing). Covers every botocore modeled error (bedrock
            # AccessDeniedException -> 403, ThrottlingException -> 429, ...).
            if isinstance(resp, dict):
                try:
                    meta = resp.get("ResponseMetadata")
                    v = meta.get("HTTPStatusCode") if isinstance(meta, dict) else None
                    if isinstance(v, int) and not isinstance(v, bool) and 100 <= v < 600:
                        return v
                except Exception:
                    pass
    except Exception:
        pass
    try:
        # Node's @huggingface/inference carries status only at httpResponse.status;
        # cross-seeded here so Node-shaped wrappers classify symmetrically (same
        # convention as the camelCase attr variants above). Only .status is read —
        # httpResponse.body is raw provider payload and must never be touched.
        http_resp = getattr(exc, "httpResponse", None)
        if http_resp is not None:
            v = http_resp.get("status") if isinstance(http_resp, dict) else getattr(http_resp, "status", None)
            if isinstance(v, int) and not isinstance(v, bool) and 100 <= v < 600:
                return v
    except Exception:
        pass
    return 0


# Common attribute names wrapper exceptions use to hold the underlying error
# (e.g. huggingface_hub wraps the 4xx-carrying HTTPError during streaming).
_WRAPPER_ATTRS = ("inner", "original", "original_exception", "original_error", "__cause__", "__context__")

# Last-resort message parse: "HTTP 400", "status: 429", "status code 503", ….
_STATUS_MSG_RE = re.compile(r"(?:HTTP|status(?:\s+code)?)[:\s]+(\d{3})", re.I)


def _chain_links(exc: BaseException, max_depth: int = 4):
    """Yield the wrapped/chained exceptions behind `exc` (breadth-first,
    cycle-safe, bounded). Never raises."""
    seen = {id(exc)}
    frontier = [exc]
    for _ in range(max_depth):
        nxt = []
        for cur in frontier:
            for attr in _WRAPPER_ATTRS:
                try:
                    link = getattr(cur, attr, None)
                except Exception:
                    continue
                if isinstance(link, BaseException) and id(link) not in seen:
                    seen.add(id(link))
                    nxt.append(link)
                    yield link
        if not nxt:
            return
        frontier = nxt


def _status_from_message(exc: BaseException) -> int:
    try:
        m = _STATUS_MSG_RE.search(str(exc))
        if m:
            v = int(m.group(1))
            if 100 <= v < 600:
                return v
    except Exception:
        pass
    return 0


def _http_status_of(exc: BaseException) -> int:
    status = _direct_status_of(exc)
    if status:
        return status
    # Hardening — both passes are gated on the direct extraction returning 0,
    # so every exception that already carries a status is classified exactly
    # as before. Everything below is wrapped: a failure degrades to status 0.
    try:
        # (a) Traverse the exception chain (__cause__ / __context__ / common
        # wrapper attrs, bounded depth, cycle-safe) and re-run the attribute
        # checks on each link — fixes wrapped streaming failures where only
        # the inner exception carries the 4xx/5xx.
        links = list(_chain_links(exc))
        for link in links:
            v = _direct_status_of(link)
            if v:
                return v
        # (b) Last resort: parse the message ("HTTP 400" / "status code 429"),
        # top exception first, then the chained ones.
        v = _status_from_message(exc)
        if v:
            return v
        for link in links:
            v = _status_from_message(link)
            if v:
                return v
    except Exception:
        pass
    return 0


# Canonical gRPC StatusCode name -> (error_kind, synthesized http_status).
# Native gRPC SDKs (e.g. xai_sdk) raise grpc errors with NO HTTP status on
# the wire; this is the standard gRPC→HTTP mapping. "OK" is deliberately
# absent (not an error → falls through to unknown/0).
_GRPC_STATUS_MAP = {
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


def _grpc_code_name_of(exc: BaseException) -> str:
    """Return the grpc StatusCode name ("INVALID_ARGUMENT", …) or "".
    Total: never raises, never imports grpc (grpcio is not a dependency)."""
    try:
        # Module gate FIRST — arbitrary/hostile exceptions must never have
        # methods invoked on them. Only types living in the `grpc` package
        # are probed (covers grpc._channel sync + grpc.aio._call async).
        mod = getattr(type(exc), "__module__", "")
        if not (isinstance(mod, str) and (mod == "grpc" or mod.startswith("grpc."))):
            return ""
        code = getattr(exc, "code", None)
        if not callable(code):
            return ""
        # Calling code() is safe here: it can raise UsageError on a
        # pre-terminal RPC, and a live _MultiThreadedRendezvous.code() could
        # block — but the classifier only ever sees RAISED exceptions, on
        # which the RPC is terminal and code() returns immediately.
        val = code()
        # Accept only a StatusCode-shaped return: a `.name` string that is a
        # known status. Rejects junk, including the coroutine grpc.aio's
        # Call.code() would produce (no `.name`; never await anything).
        name = getattr(val, "name", None)
        if isinstance(name, str) and name in _GRPC_STATUS_MAP:
            return name
    except Exception:
        pass
    return ""


def _grpc_classification_of(exc: BaseException):
    """Classification dict for a (possibly wrapped) grpc error, or None.
    Total: never raises."""
    try:
        name = _grpc_code_name_of(exc)
        if not name:
            for link in _chain_links(exc):
                name = _grpc_code_name_of(link)
                if name:
                    break
        if name:
            kind, status = _GRPC_STATUS_MAP[name]
            return {"error_kind": kind, "http_status": status}
    except Exception:
        pass
    return None


def classify_exception(exc: BaseException) -> Dict[str, Any]:
    try:
        name = _name_of(exc)
        status = _http_status_of(exc)

        # Name-based hints first (most specific).
        if name in _RATE_LIMIT_NAMES or status == 429:
            return {"error_kind": "rate_limited", "http_status": status or 429}
        if name in _TIMEOUT_NAMES or "Timeout" in name:
            return {"error_kind": "timeout", "http_status": status}
        if name in _AUTH_NAMES or status in (401, 403):
            return {"error_kind": "auth_error", "http_status": status or 401}
        if name in _NETWORK_NAMES:
            return {"error_kind": "network_error", "http_status": status}

        # gRPC (e.g. native xai_sdk): no HTTP status exists on the wire; map
        # the StatusCode canonically. Only when nothing HTTP-derived exists —
        # a real extracted status (e.g. google.api_core's int .code) wins.
        if status == 0:
            grpc_info = _grpc_classification_of(exc)
            if grpc_info:
                return grpc_info

        # Status-only fallbacks.
        if 500 <= status < 600:
            return {"error_kind": "server_error", "http_status": status}
        if 400 <= status < 500:
            return {"error_kind": "client_error", "http_status": status}
        if name in _SERVER_NAMES:
            return {"error_kind": "server_error", "http_status": status}
        if name in _BADREQ_NAMES:
            return {"error_kind": "client_error", "http_status": status or 400}
    except Exception:
        pass
    return {"error_kind": "unknown", "http_status": 0}


_ERROR_DETAIL_MODES = ("none", "redacted", "raw")


def _coerce_error_detail(mode: Any) -> str:
    return mode if mode in _ERROR_DETAIL_MODES else "redacted"


def resolve_error_detail() -> str:
    """Single resolution point for the configured error-detail mode. Reads the
    global client's ``error_detail``; a missing/raising client (or any
    non-member value) yields the SAFE default ``"redacted"`` — never ``"raw"``.
    Total: never raises.
    """
    try:
        from .state import get_client

        d = getattr(get_client(), "error_detail", None)
        if d in _ERROR_DETAIL_MODES:
            return d
    except Exception:
        pass
    return "redacted"


def scrub_error_message(raw: Any, mode: str) -> Dict[str, Any]:
    """Scrub helper — pure & TOTAL. Takes the RAW error value (``Any``) so
    the ``str(...)`` + truncate + SHA-256 all happen INSIDE this guard; no call
    site stringifies first (a hostile ``__str__`` is contained here). Returns
    ONLY message-derived fields; on ANY internal failure returns ``{}`` (the
    safest shape). Does no I/O, never raises.

    - "none": ``{}`` (nothing derived from the message).
    - "redacted": ``{"error_message_hash": ...}`` (SHA-256 hex of the FULL
                  pre-truncation string; empty string -> ``{}``).
    - "raw": ``{"error_message": str(raw)[:500]}`` (legacy behavior).
    """
    m = _coerce_error_detail(mode)
    if m == "none":
        return {}
    try:
        s = str(raw)
    except Exception:
        return {}
    if m == "raw":
        return {"error_message": s[:500]}
    # redacted
    if not s:
        return {}
    try:
        return {"error_message_hash": hashlib.sha256(s.encode("utf-8")).hexdigest()}
    except Exception:
        return {}


def to_duration_ms(delta_ms: float) -> int:
    """Convert a wall-clock delta (ms, float) to wire ``duration_ms`` (UInt32).

    Positive sub-millisecond deltas must not collapse to 0 — report 1 so
    "ran but fast" is distinguishable from "no duration recorded". True-zero /
    negative / non-finite deltas stay 0. Multi-ms keeps Python's historical
    ``int`` floor (Node uses ``Math.round``; only the sub-ms zero is harmonized).
    Pure arithmetic; never raises.
    """
    try:
        if not isinstance(delta_ms, (int, float)) or isinstance(delta_ms, bool):
            return 0
        if not (delta_ms > 0):
            return 0
        # bool is a subclass of int; rejected above. NaN fails `> 0`.
        return max(1, int(delta_ms))
    except Exception:
        return 0


def build_call_outcome(exc: BaseException | None, duration_ms: int) -> Dict[str, Any]:
    """Build the `call_outcome` payload the server expects on /log."""
    if exc is None:
        return {"status": "success", "duration_ms": int(duration_ms)}
    info = classify_exception(exc)
    mode = resolve_error_detail()
    outcome = {
        "status": "failed",
        "duration_ms": int(duration_ms),
        "error_kind": info["error_kind"],
        "http_status": info["http_status"],
    }
    # Route the RAW exception through the scrub helper (no pre-stringify);
    # default 'redacted' ships a hash, not the raw string.
    outcome.update(scrub_error_message(exc, mode))
    # error_class = exception type name (a type name, not message content).
    # Added ONLY in redacted mode by this central builder (which holds the
    # exception object). Best-effort: guarded so it can never raise.
    if mode == "redacted":
        try:
            cls = type(exc).__name__
        except Exception:
            cls = ""
        if cls:
            outcome["error_class"] = cls
    return outcome
