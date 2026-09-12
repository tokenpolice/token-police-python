"""OpenAI Agents SDK (`openai-agents`) tool-span capture.

The Agents SDK runs `@function_tool` functions itself — there is no patched
provider call for a tool execution — and records them only in its **own**,
non-OpenTelemetry tracing system (`agents.tracing`) as `FunctionSpanData` spans.
TokenPolice's telemetry only observes OTel spans, so tool executions were
invisible (LLM calls are still captured via the patched Responses API).

This module registers a `TracingProcessor` with the Agents SDK that, for
function spans only, emits a TokenPolice `tool` row into the **current session's**
trace — mirroring `telemetry._log_tool_span`. LLM (`response`/`generation`)
spans are left untouched (already captured via the Responses wrapper), so there
is no double-counting.

Everything here is fail-open: a failure must never break the customer's agent
run. Registration is gated on `"agents"` already being imported and is idempotent.
"""

import logging
import sys
from datetime import datetime

# (sha1-16, length) for a tool arg/result — the ONE implementation lives beside
# the canonical serializer in composition.py. Shared verbatim with telemetry.py
# and the pydantic_ai tool-row emit (``enforcer._emit_pydantic_ai_tool_row``),
# so a single definition feeds every call site. Re-imported here under the same
# name.
from .composition import _hash_len

logger = logging.getLogger("token_police")

_REGISTERED = False
_processor = None  # keep a reference so it isn't GC'd


def _tool_duration_ms(span):
    """Best-effort tool-call wall-clock duration in ms, or 0 if unavailable.

    The Agents SDK stamps ``started_at``/``ended_at`` as ISO-8601 strings on the
    span (both are set by the time ``on_span_end`` fires), so we derive real
    duration from them rather than needing a separate start hook. TOTAL +
    NO-THROW: any missing/unparseable timestamp degrades to 0 — a tool row with
    duration_ms=0 is still emitted, never dropped, and never raises into the
    customer's agent run."""
    try:
        started = getattr(span, "started_at", None)
        ended = getattr(span, "ended_at", None)
        if not started or not ended:
            return 0

        def _parse(ts):
            # fromisoformat on the 3.10 floor doesn't accept a trailing 'Z';
            # normalize it to an explicit UTC offset first.
            if isinstance(ts, str) and ts.endswith("Z"):
                ts = ts[:-1] + "+00:00"
            return datetime.fromisoformat(ts) if isinstance(ts, str) else ts

        delta_ms = (_parse(ended) - _parse(started)).total_seconds() * 1000.0
        # Positive sub-ms → 1; true-zero/negative → 0.
        from ._classify import to_duration_ms
        return to_duration_ms(delta_ms)
    except Exception:
        return 0


def _emit_tool_row(span):
    """Build + send a TokenPolice `tool` row from an Agents SDK function span."""
    from .state import get_client
    from .context import (
        get_current_session,
        random_hex16,
        _session_parent_span_id,
        _pop_pending_tool_call_id,
    )

    client = get_client()
    if client is None:
        return

    sd = getattr(span, "span_data", None)
    name = str(getattr(sd, "name", "") or "tool")
    param_hash, param_len = _hash_len(getattr(sd, "input", None))
    result_hash, result_len = _hash_len(getattr(sd, "output", None))

    # FunctionSpanData has no call_id today (name/input/output only). Prefer any
    # future field if present (snake + camel for parity with Node); else FIFO
    # pop from the pending stash filled by the Responses/chat capture on the
    # preceding LLM turn. Never invent ids.
    call_id = str(
        getattr(sd, "call_id", None)
        or getattr(sd, "tool_call_id", None)
        or getattr(sd, "callId", None)
        or getattr(sd, "toolCallId", None)
        or ""
    )
    if not call_id:
        try:
            call_id = _pop_pending_tool_call_id(name) or ""
        except Exception:
            call_id = ""

    # Correlate into the active TokenPolice trace. The Agents SDK has its own
    # `trace_`/`span_`-prefixed ids; we ignore them and use the session's
    # W3C ids so the tool row joins the same trace as the LLM rows (which also
    # attach to the session root).
    session = get_current_session()
    metadata = {"workflow_name": session.workflow_name}
    if session.session_id:
        metadata["session_id"] = session.session_id

    err = getattr(span, "error", None)
    failed = bool(err)
    call_outcome = {
        "status": "failed" if failed else "success",
        "duration_ms": _tool_duration_ms(span),
    }
    if failed:
        # Route the RAW message value through the central scrub helper
        # (no pre-stringify); default 'redacted' ships a hash, not raw text.
        # `.get("message", "")` is a field extraction (not a str()), kept inside
        # the existing guard so a malformed `err` can never raise here.
        try:
            from ._classify import scrub_error_message, resolve_error_detail
            call_outcome.update(
                scrub_error_message((err or {}).get("message", ""), resolve_error_detail())
            )
        except Exception:
            pass

    # Parent onto the anchored agent/chain root only. Throwaway sessions from
    # get_current_session() are unanchored → parent "" (I1 phantom-parent fix).
    span_obj = {
        "trace_id": session.trace_id,
        "span_id": random_hex16(),
        "parent_span_id": _session_parent_span_id(session),
        "span_kind": "tool",
        "span_name": name,
        "span_order": 0,
        # Agents SDK timestamps are already ISO-8601 strings.
        "start_time": getattr(span, "started_at", None),
        "end_time": getattr(span, "ended_at", None),
    }

    client.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model="",
        provider="",
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        metadata=metadata,
        span=span_obj,
        prompt_composition=[],
        response_composition=[],
        tool={
            "name": name,
            "type": "function",
            "call_id": call_id,
            "param_hash": param_hash,
            "param_length": param_len,
            "result_hash": result_hash,
            "result_length": result_len,
        },
        call_outcome=call_outcome,
    )


def _handle_span_end(span):
    """Emit a tool row for function spans only; ignore everything else.

    Module-level (not a closure) so it can be unit-tested without importing the
    real `agents` package.
    """
    try:
        sd = getattr(span, "span_data", None)
        if sd is None or getattr(sd, "type", None) != "function":
            return  # LLM/agent/etc. spans are handled elsewhere — skip
        _emit_tool_row(span)
    except Exception as e:  # fail-open: never break the agent run
        logger.debug("TokenPolice: Agents tool-span emit failed (swallowed): %s", e)


def register_openai_agents_tracing():
    """Register the tool-span processor with the Agents SDK. Idempotent.

    Returns True if registered (or already was), False if the Agents SDK is not
    in use / registration failed. Never imports `agents` itself — only hooks in
    when the app has already imported it.
    """
    global _REGISTERED, _processor
    if _REGISTERED:
        return True
    if "agents" not in sys.modules:
        return False
    try:
        from agents.tracing import add_trace_processor, TracingProcessor

        class _TPAgentsToolProcessor(TracingProcessor):
            def on_trace_start(self, trace):
                pass

            def on_trace_end(self, trace):
                pass

            def on_span_start(self, span):
                pass

            def on_span_end(self, span):
                _handle_span_end(span)

            def shutdown(self):
                pass

            def force_flush(self):
                pass

        _processor = _TPAgentsToolProcessor()
        add_trace_processor(_processor)  # appends — leaves the user's exporter intact
        _REGISTERED = True
        logger.debug("TokenPolice: registered OpenAI Agents SDK tool-span processor")
        return True
    except Exception as e:  # fail-open
        logger.debug("TokenPolice: failed to register Agents tracing processor: %s", e)
        return False


def maybe_register_openai_agents_tracing():
    """Cheap, idempotent guard for the per-call hot path."""
    if _REGISTERED or "agents" not in sys.modules:
        return
    register_openai_agents_tracing()
