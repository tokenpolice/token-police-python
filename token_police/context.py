"""
Session context propagation via Python contextvars.
Allows users to attach user_id, paid_plan, workflow_name, and metadata to all LLM calls
within a context manager or decorator block, without passing them explicitly.

Also carries span hierarchy context (trace_id, root_span_id) for agent-run tracking.
"""
import contextvars
import uuid
import secrets
import functools
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from opentelemetry import trace as _otel_trace


# ── W3C-format id helpers ─────────────────────────────────────────
# W3C Trace Context: trace_id = 32 lowercase hex chars (128-bit),
# span_id = 16 lowercase hex chars (64-bit). We mint these directly so the
# emitted span model matches the OTel/W3C wire format even on paths that have
# no live OTel span (manual-telemetry providers).
def random_hex16() -> str:
    """16 lowercase hex chars (64-bit) — a W3C-format span id."""
    return secrets.token_hex(8)


def random_hex32() -> str:
    """32 lowercase hex chars (128-bit) — a W3C-format trace id."""
    return secrets.token_hex(16)


def _sanitize_session_id(v) -> str:
    """
    Normalize a customer-supplied session id. Conversation threading is opt-in:
    pass the same id across turns to group them. Defensive + fail-open — string-
    coerces, strips control chars, trims, caps at 200 chars, and returns "" on any
    failure or empty/nullish input (so the caller falls back to the per-run UUID).
    Must never raise into the host application (fail-open).
    """
    try:
        if v is None:
            return ""
        raw = v if isinstance(v, str) else str(v)
        out = "".join(ch for ch in raw if ord(ch) > 0x1F and ord(ch) != 0x7F)
        return out.strip()[:200]
    except Exception:
        return ""


@dataclass
class TPSession:
    """Represents a discrete session of LLM interactions tracked by TokenPolice."""
    user_id: str = "anonymous"
    paid_plan: str = "free"
    workflow_name: str = "default_workflow"
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Span hierarchy ──
    # W3C-format ids. When a session opens a real OTel agent span these are
    # overwritten with the span's native trace/span ids (see session()).
    trace_id: str = field(default_factory=random_hex32)
    root_span_id: str = field(default_factory=random_hex16)
    # True only after _structural_span binds real non-zero OTel ids onto this
    # object. Throwaway sessions from get_current_session() stay False so
    # manual/Mode-A parents do not point at a never-logged root_span_id.
    # Lives on the session object (not "is contextvar set now") so post-scope
    # stream finalize holding this ref still parents correctly.
    _anchored: bool = field(default=False, repr=False)
    _span_counter: int = field(default=0, repr=False)
    _counter_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Ordered (id, name) tool-call ids from the most recently captured LLM
    # response. Shared across asyncio task context copies (same session object)
    # so concurrent tool spans pop distinct ids — Node parity (TPSession
    # ._pendingToolCalls). Mutated in place by _pop_pending_tool_call_id.
    _pending_tool_calls: list = field(default_factory=list, repr=False)
    # Provenance of `paid_plan` — wire metadata only, never matching.
    #   "app"     the resolved value was ultimately supplied by application
    #             code (this scope, or inherited from an ancestor that set it)
    #   "default" the SDK synthesized it (the "free" fallback fired with no app
    #             input anywhere in the chain)
    # Resolves along EXACTLY the same path the VALUE does, so an inherited plan
    # carries the parent's source. Sent as `user.plan_source`; the collector
    # maps missing/other to "unknown". Derived, never a user-facing option.
    # Declared LAST on purpose: appending keeps every existing POSITIONAL
    # TPSession(...) construction binding to the same fields as before.
    plan_source: str = "default"

    def next_span_order(self) -> int:
        """Returns the next child span order index (thread-safe)."""
        with self._counter_lock:
            order = self._span_counter
            self._span_counter += 1
            return order


_current_session: contextvars.ContextVar[Optional[TPSession]] = contextvars.ContextVar(
    "tp_session", default=None
)

# Pending span name — set by user via set_span_name(), consumed by next LLM call
_pending_span_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "tp_pending_span_name", default=None
)

# Unscoped fallback for pending tool-call ids when no session is bound.
# Prefer TPSession._pending_tool_calls (shared across asyncio tasks that copy
# the parent context). This ContextVar covers sequential @tp.tool / tool_span
# outside session()/agent()/workflow(). Holds {id, name} only — never args.
# Pop mutates the list in place (not copy-on-write): forked asyncio tasks share
# the list object reference, so concurrent pops see each other's removals and
# get distinct provider ids. Best-effort + fail-open: miss → call_id "".
_pending_tool_calls: contextvars.ContextVar[Optional[list]] = contextvars.ContextVar(
    "tp_pending_tool_calls", default=None
)

# Reserved LLM-span order for the NEXT instrumented span's on_start.
# An instrumented provider wrapper allocates its span order up front (before it
# calls the provider) and reserves it here so the telemetry on_start that fires
# during the provider call CONSUMES this exact order instead of allocating a
# fresh one. This makes concurrent calls on one session unable to cross-attribute
# their post-call stashes — the order is reserved before the provider call, so
# two calls racing on one session key their prompt/response fingerprints onto
# their own spans (see tests/test_concurrent_attribution.py). One-shot dict
# `{"order": int, "consumed": bool}`; contextvars are per-thread/per-task, so
# concurrent wrappers each hold their own reservation.
_reserved_span_order: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "tp_reserved_span_order", default=None
)


def reserve_span_order(order: int):
    """Reserve ``order`` for the next instrumented span's on_start. Returns the
    contextvar token; the caller MUST reset it (finally) so a leaked reservation
    can't mis-key a later unrelated span. Never raises."""
    return _reserved_span_order.set({"order": order, "consumed": False})


def reset_span_order(token) -> None:
    """Reset the reservation contextvar to its prior value. Never raises."""
    try:
        _reserved_span_order.reset(token)
    except Exception:
        try:
            _reserved_span_order.set(None)
        except Exception:
            pass


def consume_reserved_span_order() -> Optional[int]:
    """If a fresh (unconsumed) span-order reservation is present, mark it consumed
    and return its order; else None. One-shot — a second span in the same call
    (e.g. a nested instrumentor span) allocates fresh. Never raises."""
    try:
        res = _reserved_span_order.get()
        if isinstance(res, dict) and res.get("consumed") is False:
            res["consumed"] = True
            return res.get("order")
    except Exception:
        pass
    return None


# Guard flag — True while executing inside a LangChain-instrumented method.
# LangChain calls the underlying provider SDK (openai/anthropic/...) internally,
# and that SDK is ALSO patched by the enforcer. The LangChain wrapper sets this
# flag so the nested provider wrapper becomes a pure pass-through: the LangChain
# layer runs the single pre-flight check and captures composition once.
_in_langchain: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tp_in_langchain", default=False
)

# Process-global backstop for the LangChain guard.
# `_in_langchain` is a ContextVar and can be invisible under LangGraph async
# (Task snapshotted before the wrapper set True; google_genai instrumentor
# on_start then misses the stamp). A plain dict keyed by session.trace_id is
# visible from any asyncio context/thread. Depth-counted so nested LC calls
# stay registered until the outermost leave. Cap + fail-open: never throw,
# never grow unbounded. Telemetry consults this only for google_genai scopes.
_LC_TRACE_REG_CAP = 4096
_lc_trace_depth: "OrderedDict[str, int]" = OrderedDict()
_lc_trace_lock = threading.Lock()


def enter_langchain_trace(trace_id: str) -> None:
    """Register a live LangChain call for ``trace_id`` (depth++). Never raises."""
    try:
        if not trace_id or not isinstance(trace_id, str):
            return
        with _lc_trace_lock:
            d = _lc_trace_depth.get(trace_id, 0) + 1
            _lc_trace_depth[trace_id] = d
            _lc_trace_depth.move_to_end(trace_id)
            while len(_lc_trace_depth) > _LC_TRACE_REG_CAP:
                # Evict oldest distinct key only. Rare under 4096 concurrent traces;
                # if we would evict ourselves, stop (keep the live registration).
                oldest, _ = next(iter(_lc_trace_depth.items()))
                if oldest == trace_id:
                    break
                _lc_trace_depth.popitem(last=False)
    except Exception:
        pass


def leave_langchain_trace(trace_id: str) -> None:
    """Unregister one LangChain call for ``trace_id`` (depth--). Never raises."""
    try:
        if not trace_id or not isinstance(trace_id, str):
            return
        with _lc_trace_lock:
            d = _lc_trace_depth.get(trace_id, 0) - 1
            if d <= 0:
                _lc_trace_depth.pop(trace_id, None)
            else:
                _lc_trace_depth[trace_id] = d
                _lc_trace_depth.move_to_end(trace_id)
    except Exception:
        pass


def langchain_trace_active(trace_id: str) -> bool:
    """True if a LangChain wrapper holds a live registration for ``trace_id``.

    Process-global (not ContextVar) — used by the span processor as a
    backstop when ``in_langchain()`` is False but a LC call is in flight on
    this session trace. Never raises; returns False on any failure.
    """
    try:
        if not trace_id or not isinstance(trace_id, str):
            return False
        with _lc_trace_lock:
            return _lc_trace_depth.get(trace_id, 0) > 0
    except Exception:
        return False

# Same idea for LiteLLM: litellm.completion() internally calls the underlying
# provider SDK (openai / anthropic / ...), which is ALSO patched. The LiteLLM
# wrapper sets this so the nested provider wrapper passes straight through.
_in_litellm: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tp_in_litellm", default=False
)

# Same idea for LlamaIndex: llama_index.llms.{openai,anthropic,google_genai}
# instances internally call the underlying provider SDK, which is ALSO patched.
# The LlamaIndex wrapper sets this so the nested provider wrapper short-circuits
# to a pure pass-through (single pre-flight check, single composition capture).
_in_llamaindex: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tp_in_llamaindex", default=False
)

# Same idea for pydantic_ai: each concrete pydantic_ai Model subclass
# (OpenAIChatModel / AnthropicModel / GoogleModel / GeminiModel / MistralModel /
# OpenAIResponsesModel) internally calls the underlying provider SDK, which is
# ALSO patched. The pydantic_ai wrapper sets this so the nested provider wrapper
# (and the inner provider's OpenLLMetry span) short-circuits to a pure pass-
# through — TokenPolice logs a single, pydantic_ai-attributed entry per
# Model.request / request_stream call.
_in_pydantic_ai: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tp_in_pydantic_ai", default=False
)

# Same idea for Agno (github.com/agno-agi/agno): `agno.agent.Agent.run` /
# `Agent.arun` internally calls the underlying provider SDK (openai, anthropic,
# google.genai, mistralai, ...) which is ALSO patched. The Agno wrapper sets
# this so the nested provider wrapper (and the inner provider's OpenLLMetry
# span) short-circuits to a pure pass-through — TokenPolice logs a single,
# Agno-attributed entry per Agent.run / Agent.arun call.
_in_agno: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "tp_in_agno", default=False
)

# Per-call suppression window for the OTel anthropic stream-manager span.
# The enforcer's Messages.stream wrapper arms a fresh record around the exact
# statements where the instrumentor can start that span — the wrapped
# `stream()` call and, only when nothing fired there, the vendor manager's
# enter — and TokenPoliceSpanProcessor.on_start claims the record to suppress
# the span. A ContextVar holding a PER-CALL dict (the record IS the call's
# key — the same per-call-keyed idiom as the observations queue's obs key and
# the keyed `_local_decision` store) rather than a session flag: nested
# `with`-streams, threads, and asyncio tasks share one TPSession, and outside
# any open session/workflow `get_current_session()` mints a NEW throwaway
# TPSession per call — so a flag written on "the session" was dead outside
# workflows (the SpanProcessor never resolved that object → double row) and,
# unconsumed inside them, leaked True and silently ate the NEXT anthropic
# call's telemetry.
# Record shape: {"seen": <spans claimed so far>, "limit": <max claims>}.
_anthropic_stream_span_window: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "tp_anthropic_stream_span_window", default=None
)


def arm_anthropic_stream_span_window(record=None):
    """Arm a suppression window: set the ContextVar to ``record`` (or a fresh
    ``{"seen": 0, "limit": 1}``) and return an opaque handle for
    :func:`disarm_anthropic_stream_span_window`. Passing an existing record
    re-arms it without resetting its claim count (the wrapper's manager-enter
    window reuses the call's own record so one call never suppresses more
    than its limit). Fail-open: any failure returns None — the paired disarm
    then no-ops and nothing is suppressed. An extra telemetry row is the
    accepted degradation; a raise into the customer call never is."""
    try:
        if record is None:
            record = {"seen": 0, "limit": 1}
        token = _anthropic_stream_span_window.set(record)
        return (record, token)
    except Exception:
        return None


def disarm_anthropic_stream_span_window(handle) -> None:
    """Restore the pre-arm window. ``Token.reset`` (not ``set(None)``) so a
    nested `with`-stream's disarm re-exposes the OUTER call's record instead
    of clearing it. Total no-op on a None handle or any failure — never
    raises into the customer call."""
    try:
        if handle is None:
            return
        _anthropic_stream_span_window.reset(handle[1])
    except Exception:
        pass


def claim_anthropic_stream_span(span, scope_name) -> bool:
    """True iff the current window claims this span (the caller then stamps
    ``tp.suppress``). Matches on TWO INDEPENDENT instrumentor-owned
    identifiers — the instrumentation scope and the "anthropic."-prefixed
    span name — so a rename of either alone still matches. No armed window,
    an exhausted record, a non-matching span, or ANY failure → False: never
    suppress on doubt (an extra row beats silent telemetry loss)."""
    try:
        record = _anthropic_stream_span_window.get()
        if not isinstance(record, dict):
            return False
        if int(record.get("seen", 0)) >= int(record.get("limit", 0)):
            return False
        try:
            name = getattr(span, "name", "") or ""
        except Exception:
            name = ""
        if not ((scope_name or "").startswith("opentelemetry.instrumentation.anthropic")
                or name.startswith("anthropic.")):
            return False
        record["seen"] = int(record.get("seen", 0)) + 1
        return True
    except Exception:
        return False


def in_langchain() -> bool:
    """True while executing inside a LangChain-instrumented call (internal use)."""
    return _in_langchain.get()


def in_litellm() -> bool:
    """True while executing inside a LiteLLM-instrumented call (internal use)."""
    return _in_litellm.get()


def in_llamaindex() -> bool:
    """True while executing inside a LlamaIndex-instrumented call (internal use)."""
    return _in_llamaindex.get()


def in_pydantic_ai() -> bool:
    """True while executing inside a pydantic_ai-instrumented call (internal use)."""
    return _in_pydantic_ai.get()


def in_agno() -> bool:
    """True while executing inside an Agno-instrumented call (internal use)."""
    return _in_agno.get()


def get_current_session() -> TPSession:
    """Returns the active session, or a default one if none is set.

    EPHEMERAL DEFAULT (accepted limitation): outside an open
    session()/agent()/chain()/workflow() block the contextvar is unset, so
    EVERY call mints a NEW throwaway ``TPSession``. Two calls in the same
    wrapped LLM call therefore get DIFFERENT session objects. Any code that
    both writes a per-call stash (e.g. the enforcer's ``_local_decision``
    reroute/block audit) and later reads it back must resolve the session ONCE
    and thread that same object through, rather than re-calling this helper
    (which would read a fresh, empty session). See ``enforcer._run_sync_check``
    / ``_run_async_check`` (``session=`` param) and the wrappers that thread it.
    The telemetry SpanProcessor still mints its own default here when no
    context session is open — that path is accepted design and NOT unified."""
    return _current_session.get() or TPSession()


def set_span_name(name: str) -> None:
    """
    Set a name for the next LLM call's span.
    Consumed once — after the next LLM call, the name is cleared.
    
    Usage:
        tp.set_span_name("route_query")
        response = client.chat.completions.create(...) # This span gets named "route_query"
        
        # Next call has no name set (defaults to model name)
        response2 = client.chat.completions.create(...)
    """
    _pending_span_name.set(name)


def consume_pending_span_name() -> Optional[str]:
    """Consumes and returns the pending span name (internal use by telemetry)."""
    name = _pending_span_name.get()
    if name is not None:
        _pending_span_name.set(None)
    return name


def set_pending_tool_calls(pairs) -> None:
    """REPLACE the pending (id, name) list captured from the latest LLM response.

    Replace-on-capture so a non-tool response clears the stash and stale ids from
    a prior loop iteration don't leak. When a session is bound, the list lives on
    the shared ``TPSession`` (visible to concurrent asyncio tasks). Otherwise the
    unscoped ContextVar is used. Internal use by the enforcer. Never raises.
    """
    try:
        if pairs:
            try:
                normalized = list(pairs)
            except Exception:
                # Fail-open: non-iterable garbage leaves the stash unchanged
                # (matches prior list(pairs)-in-try behaviour).
                return
        else:
            normalized = []
        session = _current_session.get()
        if session is not None:
            session._pending_tool_calls = normalized
            # Keep dual stores from diverging: session is authoritative when bound.
            try:
                _pending_tool_calls.set(None)
            except Exception:
                pass
        else:
            _pending_tool_calls.set(normalized or None)
    except Exception:
        pass


def _pop_from_pending_list(pending, name: str) -> str:
    """FIFO name-match pop on a shared list (mutates in place). Returns id or ""."""
    if not isinstance(pending, list) or not pending:
        return ""
    for i, entry in enumerate(pending):
        try:
            if not isinstance(entry, dict):
                continue
            if entry.get("name") == name:
                cid = entry.get("id", "") or ""
                pending.pop(i)
                return cid
        except Exception:
            continue
    return ""


def _pop_pending_tool_call_id(name: str) -> str:
    """FIFO pop-on-match: return the id of the first pending entry whose name
    matches ``name`` (removing it), else "".

    Mutates the stash list **in place** so concurrent asyncio tasks that share
    the same session object (or the same ContextVar list reference) observe each
    other's pops and receive distinct provider ids. Best-effort; never raises.
    """
    try:
        session = _current_session.get()
        if session is not None:
            pending = getattr(session, "_pending_tool_calls", None)
            return _pop_from_pending_list(pending, name)
        pending = _pending_tool_calls.get()
        return _pop_from_pending_list(pending, name)
    except Exception:
        return ""


def _session_parent_span_id(session: "TPSession") -> str:
    """Parent id for paths that cannot use a live OTel span.

    Returns ``session.root_span_id`` only when this session was **anchored** by a
    structural agent/chain span (real OTel ids bound). Otherwise ``""`` so we
    never invent a phantom parent pointing at a throwaway root that was never
    logged. Never raises.
    """
    try:
        if getattr(session, "_anchored", False):
            return session.root_span_id or ""
    except Exception:
        pass
    return ""


def manual_span_ids(session: "TPSession") -> Dict[str, str]:
    """
    Resolves ``{trace_id, span_id, parent_span_id}`` for a manually-built span
    (Mode C/D providers with no OTel span of their own). The span is a leaf, so
    it gets a fresh 16-hex span id and parents onto the currently active OTel
    span (the enclosing agent/workflow span). Falls back to the session's
    anchored root when no live OTel context is available, or ``""`` for
    unscoped throwaway sessions. Never raises.
    """
    span_id = random_hex16()
    try:
        cur = _otel_trace.get_current_span()
        ctx = cur.get_span_context() if cur is not None else None
        if ctx is not None and ctx.trace_id != 0 and ctx.span_id != 0:
            return {
                "trace_id": _otel_trace.format_trace_id(ctx.trace_id),
                "span_id": span_id,
                "parent_span_id": _otel_trace.format_span_id(ctx.span_id),
            }
    except Exception:
        pass
    return {
        "trace_id": session.trace_id,
        "span_id": span_id,
        "parent_span_id": _session_parent_span_id(session),
    }


@contextmanager
def _structural_span(s: "TPSession", kind: str = "agent"):
    """
    Opens a real OTel structural span (``agent`` or ``chain``) around the session
    body and makes it the active span, so OpenLLMetry LLM spans created inside
    auto-nest under it and manual spans can parent onto it via
    ``trace.get_current_span()``. The span's native (W3C-format) trace/span ids
    become the session's ``trace_id`` / ``root_span_id``. Tagged
    ``tp.kind=<kind>`` so the SpanProcessor emits it as a structural anchor row
    (no usage / cost) rather than treating it as an LLM call.

    Fail-open: if no real tracer provider is configured the span context is
    invalid (all-zero); we keep the session's constructor-default hex ids.
    """
    # Only agent/chain are valid structural anchors; anything else → agent.
    # Normalize case/whitespace so "Chain"/" chain " still map to "chain"; a
    # non-str kind degrades to "agent" without raising.
    try:
        kind = "chain" if str(kind).strip().lower() == "chain" else "agent"
    except Exception:
        kind = "agent"
    try:
        tracer = _otel_trace.get_tracer("token_police")
    except Exception:
        yield
        return

    attributes = {
        "tp.kind": kind,
        "tp.workflow_name": s.workflow_name,
        "tp.user_id": s.user_id,
        "tp.paid_plan": s.paid_plan,
        # Provenance rides with the plan so on_end (which rebuilds the row from
        # span attributes, not the live session) can forward it.
        "tp.plan_source": getattr(s, "plan_source", "default"),
        "tp.session_id": s.session_id,
    }
    for k, v in (s.metadata or {}).items():
        if isinstance(v, (str, int, float, bool)):
            attributes[f"tp.meta.{k}"] = str(v)
        elif v is not None:
            # Lists/dicts → JSON string (metadata values must be strings on the wire).
            import json as _json
            try:
                attributes[f"tp.meta.{k}"] = _json.dumps(v)
            except Exception:
                attributes[f"tp.meta.{k}"] = str(v)

    try:
        cm = tracer.start_as_current_span(s.workflow_name, attributes=attributes)
    except Exception:
        yield
        return

    with cm as span:
        try:
            ctx = span.get_span_context()
            if ctx is not None and ctx.trace_id != 0 and getattr(ctx, "span_id", 0):
                s.trace_id = _otel_trace.format_trace_id(ctx.trace_id)
                s.root_span_id = _otel_trace.format_span_id(ctx.span_id)
                # Real structural ids bound → children may parent onto root.
                # Fail-open (zero/invalid context) leaves _anchored False so
                # parents become "" rather than a never-logged constructor id.
                s._anchored = True
        except Exception:
            pass  # keep constructor-default ids; _anchored stays False
        yield span


@contextmanager
def _session_impl(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    paid_plan: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    kind: str = "agent",
):
    """
    Shared implementation for session()/agent()/chain(). Builds the TPSession
    (inheriting from an enclosing session when nested) and opens a structural
    anchor span of the given ``kind``.
    """
    existing = _current_session.get()
    # Opt-in conversation id. Empty → fall back to the per-run UUID (top-level)
    # or the parent's id (nested).
    explicit_session_id = _sanitize_session_id(session_id)

    if existing:
        merged_meta = existing.metadata.copy()
        # Merge only when the customer metadata is a dict: a non-dict
        # (str/int/list) must never crash the unguarded dict.update() nor be
        # silently misread as key/value pairs. A falsy non-dict skips the
        # merge exactly as before.
        if isinstance(metadata, dict):
            merged_meta.update(metadata)

        # Nested session → a CHILD anchor of the enclosing one. Inherit
        # session_id (grouping) and trace_id (same run) but NOT root_span_id —
        # the nested span gets its own id so it nests under its parent instead
        # of collapsing into the root (the old flat model). When the OTel span
        # opens below, both ids are refreshed from it.
        # Provided-ness (None sentinel), not value, decides inheritance: a None
        # field inherits the parent; any non-None value — INCLUDING the literal
        # defaults "default_workflow"/"anonymous"/"free" — overrides it. The
        # public entry points default these params to None, so "omitted" is
        # distinguishable from "explicitly set to the default literal".
        # Covered by tests/test_nested_scope_default_literals.py.
        s = TPSession(
            session_id=explicit_session_id or existing.session_id,
            trace_id=existing.trace_id,
            workflow_name=name if name is not None else existing.workflow_name,
            user_id=user_id if user_id is not None else existing.user_id,
            paid_plan=paid_plan if paid_plan is not None else existing.paid_plan,
            # Provenance follows the value on the SAME None sentinel: a provided
            # plan is app-supplied here; an omitted one inherits the parent's
            # VALUE and therefore the parent's SOURCE (so "app" set three scopes
            # up still reads "app" here, and a never-set chain stays "default").
            # getattr-guarded: an `existing` from an older/foreign object that
            # lacks the field degrades to "default" instead of raising.
            plan_source=(
                "app"
                if paid_plan is not None
                else getattr(existing, "plan_source", "default")
            ),
            metadata=merged_meta
        )
    else:
        # Empty session_id → TPSession default_factory mints a per-run UUID.
        # None (the "omitted" sentinel) resolves to the legacy default literal so
        # root-scope public behavior is byte-identical; an explicit value passes
        # through unchanged. (Passing None straight to TPSession would wrongly
        # override the dataclass field defaults with None — hence the resolve.)
        kwargs = dict(
            workflow_name=name if name is not None else "default_workflow",
            user_id=user_id if user_id is not None else "anonymous",
            paid_plan=paid_plan if paid_plan is not None else "free",
            # Mirrors the line above: the "free" fallback firing (paid_plan is
            # None) means nobody set a plan → "default"; any provided value
            # (including "") is app-supplied → "app".
            plan_source="app" if paid_plan is not None else "default",
            # Store a dict only, so metadata iteration downstream
            # (s.metadata.items()) can never receive a non-dict. A non-dict is
            # dropped to {} — the defined fail-open behavior.
            metadata=metadata if isinstance(metadata, dict) else {},
        )
        if explicit_session_id:
            kwargs["session_id"] = explicit_session_id
        s = TPSession(**kwargs)
    token = _current_session.set(s)
    try:
        with _structural_span(s, kind):
            yield s
    finally:
        _current_session.reset(token)


@contextmanager
def session(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    paid_plan: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    kind: str = "agent",
):
    """
    Context manager to set session context for all instrumented LLM calls.
    Omitted ``name``/``user_id``/``paid_plan`` (None) resolve to
    ``"default_workflow"``/``"anonymous"``/``"free"`` at the root scope and
    inherit the parent's value in a nested scope; an explicit value (even one of
    those default literals) always wins.
    Defaults to an **agent** span (a dynamic, LLM-driven loop); pass
    ``kind="chain"`` for a static/linear sequence, or use ``chain()`` / ``agent()``.

    Pass a stable ``session_id`` across turns to thread a conversation.

    Usage:
        with tp.session(name="rag_pipeline", user_id="user_42"):
            response = client.chat.completions.create(...)
    """
    with _session_impl(name, user_id, paid_plan, metadata, session_id, kind) as s:
        yield s


@contextmanager
def agent(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    paid_plan: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = "",
):
    """
    Explicit **agent** span — a dynamic, LLM-driven execution loop where the
    model decides the path on the fly. Same as ``session()`` (which defaults to
    agent); use it when you want the intent to read explicitly.

    Usage:
        with tp.agent(name="support_agent", user_id="u1"):
            ...
    """
    with _session_impl(name, user_id, paid_plan, metadata, session_id, "agent") as s:
        yield s


@contextmanager
def chain(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    paid_plan: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = "",
):
    """
    Explicit **chain** span — a static/linear, developer-defined sequence of
    steps (a pipeline or glue code linking non-agentic steps, e.g. retriever →
    LLM). Use it as a root entry point or to group sequential work; a purely
    agentic trace (agent + tool + llm) may not need one at all.

    Usage:
        with tp.chain(name="rag_pipeline", user_id="u1"):
            docs = retrieve(q)
            client.chat.completions.create(...)
    """
    with _session_impl(name, user_id, paid_plan, metadata, session_id, "chain") as s:
        yield s


import inspect

def _resolve_workflow_args(sig, name, user_id, paid_plan, metadata, session_id, kind, bind_args, args, kwargs):
    res = {
        "name": name,
        "user_id": user_id,
        "paid_plan": paid_plan,
        "metadata": (metadata or {}).copy(),
        "session_id": session_id,
        "kind": kind,
    }
    if bind_args:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        # Type guards: merge metadata only when it is a dict, and accept each
        # dynamically bound scalar only when it is a str — otherwise keep the
        # static decorator value. Customer-supplied call args are untrusted; a
        # bad type must never crash or corrupt the session.
        bound_meta = bound.arguments.get("metadata")
        if isinstance(bound_meta, dict):
            res["metadata"].update(bound_meta)
        bound_name = bound.arguments.get("workflow_name")
        if isinstance(bound_name, str):
            res["name"] = bound_name
        bound_user_id = bound.arguments.get("user_id")
        if isinstance(bound_user_id, str):
            res["user_id"] = bound_user_id
        bound_paid_plan = bound.arguments.get("paid_plan")
        if isinstance(bound_paid_plan, str):
            res["paid_plan"] = bound_paid_plan
        # Accept either snake_case or camelCase for the conversation id.
        bound_session_id = bound.arguments.get("session_id")
        if isinstance(bound_session_id, str):
            res["session_id"] = bound_session_id
        else:
            bound_session_id = bound.arguments.get("sessionId")
            if isinstance(bound_session_id, str):
                res["session_id"] = bound_session_id
    return res


def _static_workflow_args(name, user_id, paid_plan, metadata, session_id, kind):
    """Fallback for @workflow when argument resolution fails: the static
    decorator values only. Total — must never raise into the host application,
    and silent (no logging of customer values)."""
    try:
        meta = metadata.copy() if isinstance(metadata, dict) else {}
    except Exception:
        meta = {}
    return {
        "name": name,
        "user_id": user_id,
        "paid_plan": paid_plan,
        "metadata": meta,
        "session_id": session_id,
        "kind": kind,
    }

def workflow(
    name: Optional[str] = None,
    user_id: Optional[str] = None,
    paid_plan: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    kind: str = "chain",
    bind_args: bool = True
):
    """
    Decorator to group a business function under a single workflow. Emits a
    **chain** span by default — a workflow is a predefined, linear sequence of
    steps. Pass ``kind="agent"`` for an autonomous, LLM-driven loop. All nested
    LLM calls share the same session_id and workflow_name.

    DYNAMIC BINDING: If the wrapped function has arguments named 'user_id',
    'paid_plan', or 'session_id', the decorator extracts them at runtime and
    overrides the static values.

    Usage:
        @tp.workflow(name="rag_pipeline")
        def run_pipeline(user_id: str, session_id: str, query: str):
            return client.chat.completions.create(...)

    Bare ``@tp.workflow`` (no parentheses) is also accepted.
    """
    # Bare @tp.workflow — the decorated function arrived as `name`. A real
    # `name` is a str, so this can never intercept a parenthesized call.
    if callable(name):
        return workflow()(name)

    def decorator(func):
        sig = inspect.signature(func)

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Argument resolution must never precede the customer's function
            # with an SDK error: any binding failure falls back silently to
            # the static decorator values.
            try:
                session_kwargs = _resolve_workflow_args(sig, name, user_id, paid_plan, metadata, session_id, kind, bind_args, args, kwargs)
            except Exception:
                session_kwargs = _static_workflow_args(name, user_id, paid_plan, metadata, session_id, kind)
            with session(**session_kwargs):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                session_kwargs = _resolve_workflow_args(sig, name, user_id, paid_plan, metadata, session_id, kind, bind_args, args, kwargs)
            except Exception:
                session_kwargs = _static_workflow_args(name, user_id, paid_plan, metadata, session_id, kind)
            with session(**session_kwargs):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# ── Manual tool capture (frameworks that emit no tool span) ──────────
# CrewAI, hand-rolled function tools, and MCP calls don't produce an OTel
# tool span, so the SpanProcessor never sees them. tool_span() / @tool let a
# user capture those explicitly. Both fail open: a capture error never raises
# into customer code, and the wrapped tool always runs.

def _emit_tool_row(name, tool_type, call_id, args, result,
                   start_dt, end_dt, status, err_kind, err_msg):
    """Emit a zero-usage ``tool`` span row. Args/results are reduced to a
    (sha1-16 hash, length); raw content never leaves the process."""
    try:
        from .state import get_client
        from .telemetry import _hash_len
        client = get_client()
        if not client:
            return
        session = get_current_session()
        # Auto-correlate to the model's tool-call id when the caller didn't
        # supply one (the @tp.tool / tool_span path). Matched by name (FIFO)
        # against ids stashed from the preceding LLM response; "" on any miss.
        if not call_id:
            call_id = _pop_pending_tool_call_id(name)
        ids = manual_span_ids(session)
        param_hash, param_len = _hash_len(args)
        result_hash, result_len = _hash_len(result)

        from ._classify import to_duration_ms
        duration_ms = to_duration_ms(
            (end_dt - start_dt).total_seconds() * 1000.0
        )
        call_outcome = {"status": status, "duration_ms": duration_ms}
        if status == "failed":
            # Route the raw value through the scrub helper (no pre-stringify);
            # the default 'redacted' mode ships a hash, not raw text.
            # `err_msg or ""` is a None-coalesce so raw-mode output matches
            # the pre-scrub behavior exactly.
            from ._classify import scrub_error_message, resolve_error_detail
            call_outcome.update(scrub_error_message(err_msg or "", resolve_error_detail()))
            if err_kind:
                call_outcome["error_kind"] = err_kind

        metadata = {"workflow_name": session.workflow_name}
        if session.session_id:
            metadata["session_id"] = session.session_id

        span_obj = {
            "trace_id": ids["trace_id"],
            "span_id": ids["span_id"],
            "parent_span_id": ids["parent_span_id"],
            "span_kind": "tool",
            "span_name": name,
            "span_order": 0,
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
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
                "type": tool_type,
                "call_id": call_id,
                "param_hash": param_hash,
                "param_length": param_len,
                "result_hash": result_hash,
                "result_length": result_len,
            },
            call_outcome=call_outcome,
        )
    except Exception:
        pass  # fail-open — tool capture must never affect the customer's app


@contextmanager
def tool_span(name: str, tool_type: str = "function", call_id: str = "",
              args: Any = None):
    """
    Capture a tool / function execution as a ``tool`` span in the current
    trace. For tools that no instrumentor sees (CrewAI, raw functions, MCP).

    Usage:
        with tp.tool_span("web_search", args=query) as h:
            result = run_search(query)
            h["result"] = result # optional — recorded as a hash + length

    On an exception inside the block the span is recorded as failed and the
    exception is re-raised unchanged.
    """
    start_dt = datetime.now(timezone.utc)
    holder: Dict[str, Any] = {}
    status, err_kind, err_msg = "success", "", ""
    try:
        yield holder
    except Exception as e:
        status, err_kind, err_msg = "failed", e.__class__.__name__, str(e)
        raise
    finally:
        _emit_tool_row(
            name, tool_type, call_id, args, holder.get("result"),
            start_dt, datetime.now(timezone.utc), status, err_kind, err_msg,
        )


def tool(name: Optional[str] = None, tool_type: str = "function"):
    """
    Decorator that captures a function as a ``tool`` span on every call.

    Usage:
        @tp.tool()
        def web_search(query: str) -> str:
            ...

    Bare ``@tp.tool`` (no parentheses) is also accepted.
    """
    # Bare @tp.tool — the decorated function arrived as `name`. A real `name`
    # is a str, so this can never intercept a parenthesized call.
    if callable(name):
        return tool()(name)

    import asyncio

    def decorator(func):
        tname = name or getattr(func, "__name__", "tool")

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            with tool_span(tname, tool_type=tool_type, args={"args": args, "kwargs": kwargs}) as h:
                result = func(*args, **kwargs)
                h["result"] = result
                return result

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tool_span(tname, tool_type=tool_type, args={"args": args, "kwargs": kwargs}) as h:
                result = await func(*args, **kwargs)
                h["result"] = result
                return result

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def serverless(func):
    """
    Decorator for serverless handlers (AWS Lambda, Vercel) to ensure telemetry
    is flushed before the function returns / the container is frozen.

    In a ``finally`` around the handler it drains all in-flight telemetry —
    ``await client.flush()`` for an async handler, ``client.flush_sync()`` for a
    sync one — so queued /log POSTs actually leave the worker threads before the
    container sleeps. Both paths genuinely join the background threads; this is a
    real drain, unlike Node's diagnostic-only ``flushSync`` (Node relies on
    keepalive POSTs surviving the freeze instead). Fail-open: a flush error is
    swallowed and never masks the handler's own return value or exception.
    """
    import asyncio
    import functools
    from .state import get_client

    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            finally:
                client = get_client()
                if client:
                    await client.flush()
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                client = get_client()
                if client:
                    client.flush_sync()
        return sync_wrapper
