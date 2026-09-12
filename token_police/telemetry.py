"""
OpenTelemetry-based Token Extraction Layer.

This module uses OpenLLMetry's instrumentors to create OTel spans for every LLM call.
A custom SpanProcessor extracts token usage from finished spans and sends it to the
TokenPolice service via fire-and-forget log calls.

NO data ever leaves via OTel exporters — we only use the local span data.

Span Hierarchy:
  - TPSession carries trace_id + root_span_id
  - Each LLM span gets a child span_id linked to the root
  - Prompt/response composition is extracted from the span or from enforcer-captured data
"""
import copy
import json
import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor

# Guarded import: ProxyTracerProvider is the type OTel's `trace` module
# returns from get_tracer_provider() while no global provider has been set. We
# use it to detect the "nobody set a provider yet" case. Import it defensively
# so an OTel-version drift that moves/renames the symbol fails OPEN — the
# sentinel stays None and the provider-detection code falls through to the
# historical build-and-set path (see setup_opentelemetry).
try:
    from opentelemetry.trace import ProxyTracerProvider
except Exception:  # pragma: no cover - only hit on OTel-version drift
    ProxyTracerProvider = None

logger = logging.getLogger("token_police")

# Holds the TracerProvider object WE built and globally set, and only that
# object. Left None when we attached to a customer-owned (piggybacked or
# injected) provider. unsetup_opentelemetry() shuts down this stored object
# only — a customer's provider must never be shut down under them (it would
# kill their exporter).
_tp_created_provider = None


# ── Orphan-rewrite: surviving-ancestor resolution ────────────────────────────
# LangChain/LangGraph emit a stack of intermediate orchestration spans between
# our structural agent/chain anchor and the real LLM/tool call ("invoke_agent X",
# "execute_task RunnableSequence", "LangGraph.workflow", …). None of those carry
# a model, so on_end drops them — they are never logged. A kept LLM/tool
# child whose parent_span_id points at one of them would be orphaned (rendered
# flat, cost not rolled up).
#
# To fix this in the emitted data we record EVERY non-structural span's native
# (span_id -> parent span_id) in on_start (`_span_parent`), AND separately record
# the span ids that will actually emit a row (`_kept_spans`) — tool spans as soon
# as on_start recognizes them, LLM spans once on_end has confirmed them. At
# emission we hop from the child's immediate parent up the parent chain and STOP
# at the first id that is in `_kept_spans` — i.e. the nearest surviving ancestor
# — returning that id. A dropped framework ancestor is in `_span_parent` but NOT
# in `_kept_spans`, so the walk hops over it; a kept tool/LLM ancestor IS in
# `_kept_spans`, so the walk stops there and per-ancestor cost rollup (e.g. an
# LLM call nested under a tool) is preserved. The structural agent/chain anchor
# returns out of on_start BEFORE any recording, so it is in neither map — the
# walk instead terminates when it reaches a parent that is not a `_span_parent`
# key (typically that anchor root) and returns it via the root fallback. For a
# correctly-linked call the immediate parent is already a kept/anchor span so the
# walk is a no-op and the result is byte-identical to before. Both maps share a
# fixed cap + FIFO eviction => constant memory; an evicted entry only degrades
# that one lookup (kept ancestor no longer recognized → falls through to the root
# fallback), never an error.
_SPAN_PARENT_CAP = 4096
_span_parent: "OrderedDict[str, str]" = OrderedDict()
# Span ids that emit their own telemetry row (tool/LLM). A parent chain walk
# stops at the first member — the nearest KEPT ancestor. Bounded/evicted exactly
# like _span_parent and guarded by the same lock.
_kept_spans: "OrderedDict[str, bool]" = OrderedDict()
_span_parent_lock = threading.Lock()


# ── Per-span observations key (side map) ─────────────────────────────
# Maps native span_id (int) → the owning call's obs key, recorded in on_start
# and popped by on_end's immediate-path drain. A SIDE MAP — not an instance
# attribute — because OTel Python's Span.end() hands on_end a BRAND-NEW
# ReadableSpan built by self._readable_span(), so anything stamped on the
# on_start Span object is invisible in on_end (verified on SDK 1.39.1: the
# span_id is stable across the pair, the object identity is not). This is a
# deliberate mechanism divergence from the Node SDK, whose BasicTracerProvider
# passes the SAME Span instance to onStart and onEnd — Node therefore keeps
# its non-exported Symbol marker on the span itself. Bounded + FIFO-evicted
# exactly like _span_parent (spans that never end, or end via a non-draining
# branch, must not leak); an evicted entry only degrades that one drain to the
# unknown-claimant path (untagged + stale — the owning call's entries then
# ride the staleness fallback), never an error.
_OBS_KEY_MAP_CAP = 4096
_span_obs_keys: "OrderedDict[int, str]" = OrderedDict()
_span_obs_keys_lock = threading.Lock()


def _record_span_obs_key(span) -> None:
    """Record the current per-call obs key against this span's native id so
    on_end (which receives a different object) can key its observations drain.
    Skips when no key is current. Never raises — any failure degrades that
    span's drain to the unknown-claimant path."""
    try:
        from . import state as _obs_state
        key = _obs_state.get_current_obs_key()
        if not key:
            return
        sctx = getattr(span, "context", None)
        if sctx is None or not getattr(sctx, "span_id", 0):
            return
        sid = sctx.span_id
        with _span_obs_keys_lock:
            _span_obs_keys[sid] = key
            _span_obs_keys.move_to_end(sid)
            while len(_span_obs_keys) > _OBS_KEY_MAP_CAP:
                _span_obs_keys.popitem(last=False)
    except Exception:
        pass  # fail-open — telemetry bookkeeping must never throw


def _pop_span_obs_key(span):
    """Pop and return this span's recorded obs key (None when absent/evicted/
    any failure → the drain claims untagged + stale only). Never raises."""
    try:
        sctx = getattr(span, "context", None)
        if sctx is None or not getattr(sctx, "span_id", 0):
            return None
        with _span_obs_keys_lock:
            return _span_obs_keys.pop(sctx.span_id, None)
    except Exception:
        return None


def _record_span_parent(span) -> None:
    """Record a non-structural span's native (span_id -> parent_id) so a kept
    descendant can hop over dropped framework ancestors. Never raises."""
    try:
        sctx = getattr(span, "context", None)
        if sctx is None or not getattr(sctx, "span_id", 0):
            return
        sid = trace.format_span_id(sctx.span_id)
        parent = getattr(span, "parent", None)
        pid = (
            trace.format_span_id(parent.span_id)
            if parent is not None and getattr(parent, "span_id", 0)
            else ""
        )
        with _span_parent_lock:
            _span_parent[sid] = pid
            _span_parent.move_to_end(sid)
            while len(_span_parent) > _SPAN_PARENT_CAP:
                _span_parent.popitem(last=False)
    except Exception:
        pass  # fail-open — telemetry bookkeeping must never throw


def _record_kept_span(span) -> None:
    """Mark a span's native id as KEPT (it emits its own telemetry row) so a
    descendant's parent-chain walk terminates at it instead of hopping past it
    onto the structural root. Recorded at the EARLIEST point kept-ness is known —
    on_start for tool spans, on_end for confirmed LLM spans — because a child
    usually ends (and resolves its parent) before its parent span does, so the
    ancestor must already be registered by the time the child resolves. Bounded
    and evicted exactly like _span_parent. Never raises."""
    try:
        sctx = getattr(span, "context", None)
        if sctx is None or not getattr(sctx, "span_id", 0):
            return
        sid = trace.format_span_id(sctx.span_id)
        with _span_parent_lock:
            _kept_spans[sid] = True
            _kept_spans.move_to_end(sid)
            while len(_kept_spans) > _SPAN_PARENT_CAP:
                _kept_spans.popitem(last=False)
    except Exception:
        pass  # fail-open — telemetry bookkeeping must never throw


def _resolve_kept_parent(parent_id: str, root_span_id: str) -> str:
    """Hop from an immediate parent up the parent chain and return the nearest
    KEPT ancestor (a span that emits its own row — tool/LLM — registered in
    `_kept_spans`), stopping at the first one encountered. Dropped framework
    ancestors are in `_span_parent` but not `_kept_spans`, so the walk hops over
    them. When no kept ancestor is found the walk terminates at a parent that is
    not a `_span_parent` key (typically the structural agent/chain root) and
    returns it via the root fallback. For a correctly-linked span (parent already
    kept/anchor) this returns parent_id unchanged. Bounded (<=64 hops); never
    raises."""
    try:
        p = parent_id
        hops = 0
        with _span_parent_lock:
            # Check the immediate parent first, then each intermediate — return
            # the first id that is a kept ancestor. Only keep hopping while the
            # current node is a recorded (dropped) framework span.
            while p and hops < 64:
                if p in _kept_spans:
                    return p
                if p not in _span_parent:
                    break
                p = _span_parent[p]
                hops += 1
        # root_span_id is empty for unanchored throwaway sessions (on_start
        # stamps "" when session._anchored is False) so this never invents a
        # phantom parent. Kept-ancestor hop logic is unchanged.
        return p or root_span_id
    except Exception:
        return parent_id or root_span_id


# Core provider instrumentors ship in token-police's default dependencies, so a
# missing one is abnormal. Map each to the provider package whose presence means
# the app actually uses that provider (used to gate the warning below).
_CORE_INSTRUMENTORS = {
    "opentelemetry.instrumentation.openai": ("openai", "opentelemetry-instrumentation-openai"),
    "opentelemetry.instrumentation.anthropic": ("anthropic", "opentelemetry-instrumentation-anthropic"),
    "opentelemetry.instrumentation.bedrock": ("boto3", "opentelemetry-instrumentation-bedrock"),
}

# Providers we've already warned about — one warning per provider, per process.
_warned_providers: set = set()


def _maybe_warn_missing_instrumentor(module_path: str) -> None:
    """Warn once if a core instrumentor is missing but its provider SDK is in use.

    Stays silent (fail-open) for opt-in instrumentors (google_genai, langchain)
    and for providers the app doesn't actually use. Never raises.
    """
    info = _CORE_INSTRUMENTORS.get(module_path)
    if not info:
        return  # opt-in instrumentor — expected to be absent, stay silent
    provider_pkg, _dist = info
    if module_path in _warned_providers:
        return
    try:
        import importlib.util
        if importlib.util.find_spec(provider_pkg) is None:
            return  # app doesn't use this provider — no warning needed
    except Exception:
        return
    _warned_providers.add(module_path)
    logger.warning(
        f"TokenPolice: token capture for {provider_pkg} is disabled — its instrumentor "
        f"failed to load; reinstall token-police to enable it. Budgets are still enforced."
    )


# ── Attribute keys we extract from OpenLLMetry spans ──

_ATTR_MODEL = "gen_ai.request.model"
_ATTR_SYSTEM = "gen_ai.system"
_ATTR_PROVIDER_NAME = "gen_ai.provider.name"
_ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
_ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
_ATTR_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
_ATTR_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
# OpenLLMetry's Python Anthropic instrumentor emits underscore keys
# (gen_ai.usage.cache_read_input_tokens); some versions/JS use dotted keys.
_ATTR_CACHE_READ = "gen_ai.usage.cache_read_input_tokens"
_ATTR_CACHE_CREATION = "gen_ai.usage.cache_creation_input_tokens"
_ATTR_CACHE_READ_DOTTED = "gen_ai.usage.cache_read.input_tokens"
_ATTR_CACHE_CREATION_DOTTED = "gen_ai.usage.cache_creation.input_tokens"
_ATTR_REASONING = "gen_ai.usage.reasoning_tokens"

# Fallback keys from older OpenLLMetry versions
_ATTR_MODEL_LEGACY = "llm.request.model"
_ATTR_SYSTEM_LEGACY = "llm.system"

# ── Instrumentation-scope names the google-genai instrumentor emits under ──
# ≤0.7b1 owned its tracer (otel_wrapper.py `_SCOPE_NAME`). 1.0b1 dropped that
# module entirely and delegates span creation to the SHARED
# opentelemetry-util-genai TelemetryHandler, whose tracer is
# get_tracer(__name__) == "opentelemetry.util.genai.handler" — the scope moved
# out of Google's namespace altogether. Keying suppression on the old string
# alone is what silently killed the fix twice; see
# _is_google_genai_instrumentor_span below.
_SCOPE_GOOGLE_GENAI_INSTRUMENTOR = "opentelemetry.instrumentation.google_genai"
_SCOPE_UTIL_GENAI_ROOT = "opentelemetry.util.genai"


def _plan_source_from_attrs(attrs: Dict[str, Any]):
    """
    Resolve the ``plan_source`` to report for a row rebuilt from span attributes.

    ``on_start`` stamps ``tp.plan_source`` next to ``tp.paid_plan``, so a span
    produced by this SDK carries both. Rules:
      * a valid stamped source ("app" / "default") is forwarded verbatim;
      * no ``tp.paid_plan`` attribute at all → the reader's own ``"free"``
        default is about to fire, which IS an SDK-synthesized plan → "default";
      * anything else (plan stamped but source missing/garbled — e.g. a span
        from an older SDK build) → ``None``, so the field is OMITTED on the wire
        and the server records it as "unknown" rather than a guess.

    Fail-open: a hostile/non-mapping ``attrs`` degrades to ``None``, never raises.
    """
    try:
        raw = attrs.get("tp.plan_source")
        if raw == "app" or raw == "default":
            return raw
        if attrs.get("tp.paid_plan") is None:
            return "default"
    except Exception:
        pass
    return None


# ── Tool-span recognition (cross-framework) ──
# No single discriminator works across frameworks/versions, so we OR several:
# gen_ai.operation.name="execute_tool" → Node LangChain, OTel-native, OpenAI Agents
# traceloop.span.kind="tool" → Python LangChain, older LlamaIndex
# openinference.span.kind="TOOL" → Arize-instrumented apps
# name "execute_tool <tool>" / "<tool>.tool" → defense-in-depth: catches
# name-only spans none of the
# above attributes cover
def _is_tool_span(attrs: Dict[str, Any], name: str) -> bool:
    try:
        if attrs.get("gen_ai.operation.name") == "execute_tool":
            return True
        if str(attrs.get("traceloop.span.kind", "")).lower() == "tool":
            return True
        if str(attrs.get("openinference.span.kind", "")).upper() == "TOOL":
            return True
        n = name or ""
        if n.startswith("execute_tool "):
            return True
        if n.endswith(".tool"):
            return True
    except Exception:
        pass
    return False


def _span_is_bedrock_or_voyage(attrs: Dict[str, Any], scope_name: str) -> bool:
    """True when a span is attributable to the Bedrock or Voyage instrumentor.

    Used to scope the model-id embedding heuristic below: only these two
    instrumentors emit a chat/converse-shaped span for an embeddings call that
    the SDK also meters manually. gen_ai.system for the Bedrock instrumentor is
    aws / aws.bedrock / bedrock; the instrumentation scope name pins the emitting
    library when the system attr is absent."""
    system = str(
        attrs.get(_ATTR_SYSTEM) or attrs.get(_ATTR_SYSTEM_LEGACY) or ""
    ).lower()
    sn = (scope_name or "").lower()
    return (
        "bedrock" in system
        or system == "aws"
        or "voyage" in system
        or "bedrock" in sn
        or "voyage" in sn
    )


def _is_instrumentor_embedding_span(attrs: Dict[str, Any], name: str,
                                    scope_name: str = "") -> bool:
    """True for an OpenLLMetry / GenAI-semconv embeddings span.

    TokenPolice captures embeddings authoritatively via its own manual wrapper
    (which creates NO OTel span), so any instrumentor-emitted embedding span
    that reaches the SpanProcessor is a spurious duplicate — it would otherwise
    be mis-logged with the generic *_chat / ``llm`` shape. Used to drop it.

    Matchers:
    * OpenLLMetry openai/cohere: ``llm.request.type="embedding"`` or span names
      ``openai.embeddings`` / ``cohere.embed`` / ``*.embeddings``.
    * GenAI semantic conventions (google-genai instrumentor ≥1.0b1, and any
      future instrumentor): ``gen_ai.operation.name`` ∈ {embeddings, embedding},
      or span name ``"embeddings"`` / ``"embeddings <model>"``. Scope is NOT
      required — 1.0b1's tracer is ``opentelemetry.util.genai.handler``, not
      ``opentelemetry.instrumentation.google_genai``.
    * Bedrock/Voyage (scoped): model-id heuristic when the span is attributable
      to those instrumentors only.
    """
    try:
        if str(attrs.get("llm.request.type", "")).lower() == "embedding":
            return True
        # GenAI semconv — google-genai instrumentor 1.0b1+ (and others) set
        # gen_ai.operation.name="embeddings" on embed_content spans. Must NOT
        # match "generate_content" / "chat" (real Google chat spans).
        _op = str(attrs.get("gen_ai.operation.name", "")).lower()
        if _op in ("embeddings", "embedding"):
            return True
        n = name or ""
        if n == "openai.embeddings" or n == "cohere.embed" or n.endswith(".embeddings"):
            return True
        # util-genai span name is f"{operation} {model}" → "embeddings <model>"
        if n == "embeddings" or n.startswith("embeddings "):
            return True
        # opentelemetry-instrumentation-bedrock emits a chat/converse-shaped
        # span on the embeddings InvokeModel call (no llm.request.type=embedding,
        # no *.embeddings name), tagged with the vendor-stripped embedding model id.
        # TokenPolice captures Bedrock/Voyage embeddings on the manual path, so
        # this span is the spurious duplicate. Match by model id — mirrors
        # enforcer._BEDROCK_EMBEDDING_MODEL_PREFIXES (amazon.titan-embed /
        # cohere.embed / voyage.voyage). This model-id shortcut is SCOPED to spans
        # attributable to the Bedrock/Voyage instrumentors: a span from any other
        # system whose model id merely contains "embed" or starts with "voyage" is
        # a genuine call we must not silently drop.
        if _span_is_bedrock_or_voyage(attrs, scope_name):
            model_id = str(
                attrs.get("gen_ai.request.model") or attrs.get("llm.request.model") or ""
            ).lower()
            if model_id and ("embed" in model_id or model_id.startswith("voyage")):
                return True
    except Exception:
        pass
    return False


def _is_google_genai_instrumentor_span(scope_name: str,
                                       attrs: Dict[str, Any]) -> bool:
    """True for a span emitted by opentelemetry-instrumentation-google-genai.

    Used ONLY by the two LangChain-duplicate gates in on_start/on_end to answer
    "is this Google's instrumentor?". The suppression decision itself stays
    gated on the LangChain contextvar/registry — this predicate never suppresses
    anything on its own.

    Two instrumentor generations are simultaneously in support range (pyproject
    pins `>=0.7b0` with no cap, per the lower-bounds-only dependency policy):

    * **≤0.7b1** owns its tracer, scope ``opentelemetry.instrumentation
      .google_genai``. At on_start it carries only ``gen_ai.request.model`` /
      ``gen_ai.operation.name`` — ``gen_ai.system`` is set later during request
      processing — so there is no provider attribute to key on and the scope
      name is the only signal available. Hence arm 1 must stay scope-only.
    * **1.0b1+** delegates to the shared opentelemetry-util-genai
      ``TelemetryHandler``, scope ``opentelemetry.util.genai.handler``, and
      passes ``gen_ai.operation.name`` / ``gen_ai.request.model`` /
      ``gen_ai.provider.name`` as CREATION attributes, so they are readable in
      on_start. ``gen_ai.system`` is never set on this generation.

    util-genai is a SHARED library: any future instrumentor migrating onto
    ``TelemetryHandler`` (OpenAI, Anthropic, …) emits under the SAME scope.
    Matching the scope prefix alone would then suppress LangChain's inner
    ChatOpenAI/ChatAnthropic spans, which TokenPolice deliberately keeps as its
    single canonical priced row → total telemetry loss. So arm 2 conjoins the
    scope prefix with a Google provider signal, and a recognised non-Google
    provider vetoes the match outright.

    Fail-open (``False``) on any exception — never suppress on a bad read.
    """
    try:
        sn = str(scope_name or "")
        # Arm 1 — legacy (≤0.7b1). Prefix rather than equality: that namespace
        # is Google's own package so nothing else can register under it, and a
        # submodule-scoped tracer would still match. 0.7b1 emits only the exact
        # string, so this is behaviourally identical for it.
        if sn == _SCOPE_GOOGLE_GENAI_INSTRUMENTOR or sn.startswith(
            _SCOPE_GOOGLE_GENAI_INSTRUMENTOR + "."
        ):
            return True
        # Arm 2 — util-genai generations. The scope prefix is necessary but
        # never sufficient.
        if not (sn == _SCOPE_UTIL_GENAI_ROOT
                or sn.startswith(_SCOPE_UTIL_GENAI_ROOT + ".")):
            return False
        provider = str(
            (attrs or {}).get(_ATTR_PROVIDER_NAME)
            or (attrs or {}).get(_ATTR_SYSTEM)
            or ""
        ).strip().lower()
        if provider:
            # A present provider attribute is authoritative — a recognised
            # non-Google value falls through to False. Substring matching
            # mirrors the row builder's own Google normalization below; the
            # `gcp.` prefix covers the canonical GenAiProviderNameValues
            # spellings (gcp.gemini / gcp.vertex_ai / gcp.gen_ai) that
            # upstream's open span_utils TODO may rename these to.
            return (
                "gemini" in provider
                or "google" in provider
                or "vertex" in provider
                or provider == "gcp"
                or provider.startswith("gcp.")
            )
        # No provider attribute at all — defensive, for a util-genai that stops
        # emitting one. "generate_content" is Gemini-API vocabulary, but it is
        # also a standard semconv operation name, so it is accepted ONLY here
        # where nothing contradicts it — never as an override of a provider
        # attribute that says otherwise.
        _op = str((attrs or {}).get("gen_ai.operation.name") or "").strip().lower()
        return _op == "generate_content"
    except Exception:
        pass
    return False


def _tool_name_from_span(attrs: Dict[str, Any], name: str) -> str:
    """Tool name lives in different keys per framework — read all of them."""
    tn = (
        attrs.get("gen_ai.tool.name")
        or attrs.get("traceloop.entity.name")
        or attrs.get("tool.name")
    )
    if tn:
        return str(tn)
    n = name or ""
    if n.startswith("execute_tool "):
        return n[len("execute_tool "):]
    if n.endswith(".tool"):
        return n[: -len(".tool")]
    return n or "tool"


# (sha1-16, length) for a tool arg/result — the ONE implementation lives beside
# the canonical serializer in composition.py (shared with openai_agents.py and
# the pydantic_ai tool-row emit). Re-exported here under the historical name so
# existing callers/tests importing ``telemetry._hash_len`` keep working.
from .composition import _hash_len


# ═══════════════════════════════════════════════════════════════════
# TokenPolice Span Processor
# ═══════════════════════════════════════════════════════════════════

# ── B4 routing-marker bridge (lazy, cached) ─────────────────────────────
# `enforcer` owns the per-call routing-marker store; telemetry imports from it
# INSIDE functions (module-level would cycle: enforcer imports telemetry). The
# import ran on every LLM row build, so cache both symbols after the first
# success. Cached as a 2-tuple in one module global, resolved through the
# helper below, which stays fail-open: a failed import degrades to "no marker
# on the row" (and retries next time — a partially-initialized module can fail
# once and succeed later), never to a raise into the customer's call.
_TP_ROUTING_BRIDGE = None


def _routing_bridge():
    """(_TP_ROUTING_ATTR, _stamp_routing_marker) or None. Never raises."""
    global _TP_ROUTING_BRIDGE
    if _TP_ROUTING_BRIDGE is not None:
        return _TP_ROUTING_BRIDGE
    try:
        from .enforcer import _TP_ROUTING_ATTR, _stamp_routing_marker
        _TP_ROUTING_BRIDGE = (_TP_ROUTING_ATTR, _stamp_routing_marker)
    except Exception:
        return None
    return _TP_ROUTING_BRIDGE


def _routing_attr():
    """The `tp.meta._tp_routing` span-attribute name. Falls back to the literal
    so the STRIP can never be skipped just because the import failed — a missed
    strip would reopen the B4 leak, while a redundant strip costs nothing."""
    bridge = _routing_bridge()
    return bridge[0] if bridge else "tp.meta._tp_routing"


class TokenPoliceSpanProcessor(SpanProcessor):
    """
    Processes finished OTel spans to extract LLM token usage and send it to
    the TokenPolice service. Enriches spans with session context and span hierarchy.
    """

    def __init__(self, log_errors: bool = False):
        self.log_errors = log_errors

    def on_start(self, span, parent_context=None):
        """Stamp span hierarchy from the active TPSession."""
        # Per-call observations key: this span starts inside the provider call,
        # in the same context whose check just minted the key — record it in
        # the span-id side map (NOT on the span object: on_end receives a
        # different ReadableSpan, see _span_obs_keys) so on_end's
        # immediate-path drain can claim exactly this call's observations even
        # though it may run after another call's check overwrote the var.
        # Fail-open: an unrecorded span drains as an unknown claimant (None).
        _record_span_obs_key(span)
        # Our own structural anchor span (agent/chain, opened by
        # session()/agent()/chain()/workflow()). It is emitted as an
        # `agent`/`chain` row in on_end; it must NOT consume a span_order or be
        # treated as an LLM span here. Its tp.* attributes were set at creation.
        try:
            if (getattr(span, "attributes", None) or {}).get("tp.kind") in ("agent", "chain"):
                return
        except Exception:
            pass
        # Record this NON-structural span's parent link for orphan-rewrite. The
        # structural anchor returned above, so it is never recorded — that makes
        # it the natural terminator when a kept LLM/tool child hops over dropped
        # LangChain/LangGraph framework ancestors in on_end. (See _span_parent.)
        _record_span_parent(span)
        # LangChain's OpenLLMetry instrumentor emits framework spans (chains,
        # agents, tools) alongside the LLM spans. Skip enriching/counting those:
        # only LLM spans are logged in on_end, and counting framework spans
        # would inflate span_order and let a chain span steal the pending span
        # name set via tp.set_span_name(). LLM spans are named "<Class>.chat".
        name = getattr(span, "name", "") or ""
        # Tool execution span → promote to a `tool` row (emitted in on_end)
        # instead of dropping it. Stamp minimal session context so on_end can
        # attribute it; do NOT consume a span_order (tools are ordered by
        # start_time in the UI) and do NOT steal the pending set_span_name().
        if _is_tool_span(getattr(span, "attributes", None) or {}, name):
            # Register as a KEPT ancestor NOW (at start). A tool span emits its
            # own `tool` row, and its LLM/tool children usually END before it
            # does — so it must be marked before those children resolve their
            # parent in on_end, else the walk would hop past it onto the root
            # (losing per-tool cost rollup). on_start is the earliest reliable
            # point: tool-ness is decidable here and the span is unconditionally
            # emitted in on_end.
            _record_kept_span(span)
            try:
                from .context import get_current_session
                session = get_current_session()
                span.set_attribute("tp.kind", "tool")
                span.set_attribute("tp.trace_id", session.trace_id)
                span.set_attribute("tp.user_id", session.user_id)
                span.set_attribute("tp.paid_plan", session.paid_plan)
                span.set_attribute(
                    "tp.plan_source", getattr(session, "plan_source", "default")
                )
                span.set_attribute("tp.workflow_name", session.workflow_name)
                span.set_attribute("tp.session_id", session.session_id)
                # Custom session metadata → tp.meta.* attributes so on_end can
                # read it back into the tool row (mirrors the anchor span; see
                # context.py _structural_span). Scalars str-coerced; lists/dicts
                # JSON-serialized (metadata values must be strings on the wire).
                for k, v in (session.metadata or {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        span.set_attribute(f"tp.meta.{k}", str(v))
                    elif v is not None:
                        try:
                            span.set_attribute(f"tp.meta.{k}", json.dumps(v))
                        except Exception:
                            span.set_attribute(f"tp.meta.{k}", str(v))
            except Exception:
                pass
            return
        if name.endswith((".workflow", ".task", ".tool", ".agent")):
            return
        # Skip the mistralai SDK's native OTel spans — see the matching guard
        # in on_end. They'd otherwise consume a span_order that on_end then
        # discards, leaving gaps in the manually-logged span hierarchy. Same
        # treatment for xai-sdk: every Chat.sample()/.stream() call opens its
        # own gen_ai.* span via xai_sdk.telemetry.get_tracer(__name__), which
        # flows through our shared global TracerProvider. The enforcer logs
        # those calls on the manual path; without this filter we'd double-log
        # and inflate span_order.
        try:
            scope = getattr(span, "instrumentation_scope", None)
            scope_name = getattr(scope, "name", "") if scope else ""
            if scope_name == "mistralai_sdk_tracer":
                return
            if scope_name.startswith("xai_sdk"):
                return
            # Skip the inner google_genai instrumentor span when inside a
            # LangChain call (ChatGoogleGenerativeAI -> google.genai
            # generate_content). LangchainInstrumentor already emits the
            # canonical ChatGoogleGenerativeAI span; the google_genai
            # instrumentor fires a nested duplicate (~2× cost when both log).
            # Skipping here means it consumes NO span_order — that keeps the
            # langchain wrapper's prompt (captured before the call) and
            # response (captured after) on the SAME span_order; otherwise this
            # inner span bumps the order between the two captures and the
            # response composition gets stranded on a key nothing reads.
            #
            # Stamp tp.suppress (not just early-return). on_end re-checking
            # in_langchain() alone races under LangGraph async load: the
            # contextvar can be invisible by the time the inner span ends,
            # and a bare early-return here would still let on_end emit a ghost
            # row. Attribute survives until on_end regardless of
            # contextvar timing — same pattern as the pydantic_ai stamp below.
            #
            # Structural backstop: also consult the process-global
            # langchain_trace_active(session.trace_id) registry. On LangGraph
            # async the contextvar can already be False *at on_start* (Task
            # snapshotted before the wrapper set True) while session is still
            # visible — stamp-only (PR #195) then never fires. The registry is
            # entered by the LC wrappers alongside `_in_langchain` and is
            # immune to ContextVar propagation loss.
            # Scoped ONLY to google_genai: langchain ChatOpenAI/ChatAnthropic
            # intentionally defer the inner provider OTel span as their single
            # priced row; suppressing those scopes would erase telemetry. That
            # is also why the util-genai arm of the predicate below requires a
            # Google provider attribute and not merely the shared scope.
            #
            # Attempt #3: keying this gate on one exact scope string broke
            # it twice — google-genai 1.0b1 moved span creation into the shared
            # opentelemetry-util-genai handler, so the equality never matched
            # and neither the stamp nor the registry was ever consulted.
            # _is_google_genai_instrumentor_span matches BOTH generations.
            # on_start has no `attrs` local; read the creation attributes off
            # the span (same idiom as the tp.kind check at the top of on_start).
            # 1.0b1 passes provider/model/operation to start_span(attributes=…)
            # so they are readable here; 0.7b1 has none and is matched by scope.
            from .context import (
                in_langchain,
                get_current_session,
                langchain_trace_active,
            )
            if _is_google_genai_instrumentor_span(
                scope_name, getattr(span, "attributes", None) or {}
            ):
                suppress = False
                try:
                    suppress = bool(in_langchain())
                except Exception:
                    suppress = False
                if not suppress:
                    try:
                        _tid = getattr(get_current_session(), "trace_id", "") or ""
                        suppress = bool(_tid) and langchain_trace_active(_tid)
                    except Exception:
                        suppress = False
                if suppress:
                    try:
                        span.set_attribute("tp.suppress", True)
                    except Exception:
                        pass
                    return
        except Exception:
            pass
        # Skip inner provider spans emitted while a pydantic_ai wrapper is
        # holding the framework guard. pydantic_ai's Model.request /
        # request_stream delegates to the underlying openai/anthropic/google/
        # mistral SDK whose OpenLLMetry instrumentor would otherwise consume a
        # span_order and dispatch a duplicate /log — our pydantic_ai wrapper
        # is the single source of truth for these calls.
        #
        # Stamp tp.suppress on the span (not just early-return). Streaming
        # Anthropic ends the OTel span AFTER request_stream's __aexit__ clears
        # `_in_pydantic_ai`, so on_end's in_pydantic_ai() check alone races and
        # used to emit a ghost llm row (anonymous, empty composition, ~2× cost).
        # Attribute survives until on_end regardless of contextvar timing.
        try:
            from .context import in_pydantic_ai
            if in_pydantic_ai():
                try:
                    span.set_attribute("tp.suppress", True)
                except Exception:
                    pass
                return
        except Exception:
            pass
        # Import here to avoid circular dependency at module load
        try:
            from .context import (
                get_current_session, consume_pending_span_name, in_agno,
                claim_anthropic_stream_span,
            )
            session = get_current_session()

            # The Anthropic stream manual-wrapper takes over telemetry for
            # `messages.stream` calls — the OTel anthropic span that fires
            # alongside it would consume a span_order and emit a duplicate
            # row. The wrapper arms a per-call ContextVar window around the
            # statements where the instrumentor starts that span; claim it
            # here and mark the span to be dropped in on_end, skipping
            # span_order consumption so dashboard step numbering stays
            # sequential. (A session flag can't do this: the span starts
            # inside the wrapped `stream()` call, before the wrapper object
            # exists, and outside any open session get_current_session()
            # mints a throwaway object per call — the old one-shot flag never
            # matched here and leaked onto the NEXT anthropic call inside
            # workflows.)
            scope_name = ""
            try:
                scope = getattr(span, "instrumentation_scope", None)
                scope_name = getattr(scope, "name", "") if scope else ""
            except Exception:
                pass
            if claim_anthropic_stream_span(span, scope_name):
                span.set_attribute("tp.suppress", True)
                return

            # An instrumented wrapper may have reserved this span's order up
            # front (so concurrent calls on one session can't cross-attribute
            # their post-call stashes). Consume that reservation if present;
            # otherwise allocate as usual. Fail-open to the legacy allocation.
            order = None
            try:
                from .context import consume_reserved_span_order
                order = consume_reserved_span_order()
            except Exception:
                order = None
            if order is None:
                order = session.next_span_order()

            # Set hierarchy attributes on the span for later extraction in on_end.
            # Only stamp root when the session was anchored by a real structural
            # agent/chain span; otherwise "" so on_end never parents onto a
            # throwaway root_span_id that was never logged as an agent row.
            span.set_attribute("tp.trace_id", session.trace_id)
            try:
                _root = session.root_span_id if getattr(session, "_anchored", False) else ""
            except Exception:
                _root = ""
            span.set_attribute("tp.root_span_id", _root)
            span.set_attribute("tp.span_order", order)
            span.set_attribute("tp.session_id", session.session_id)
            span.set_attribute("tp.user_id", session.user_id)
            span.set_attribute("tp.paid_plan", session.paid_plan)
            span.set_attribute(
                "tp.plan_source", getattr(session, "plan_source", "default")
            )
            span.set_attribute("tp.workflow_name", session.workflow_name)

            # Custom session metadata → tp.meta.* attributes so on_end can read
            # it back into the LLM row (mirrors the anchor span; see context.py
            # _structural_span). Scalars str-coerced; lists/dicts JSON-serialized
            # (metadata values must be strings on the wire).
            for k, v in (session.metadata or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    span.set_attribute(f"tp.meta.{k}", str(v))
                elif v is not None:
                    try:
                        span.set_attribute(f"tp.meta.{k}", json.dumps(v))
                    except Exception:
                        span.set_attribute(f"tp.meta.{k}", str(v))

            # Consume pending span name (set by user via tp.set_span_name()).
            # Falls back to a descriptive default when inside an Agno run so
            # each inner LLM call surfaces as "agent_step_N" instead of the
            # bare model name (the OTel span's default).
            pending_name = consume_pending_span_name()
            if pending_name:
                span.set_attribute("tp.span_name", pending_name)
            elif in_agno():
                span.set_attribute("tp.span_name", f"agent_step_{order + 1}")
                
        except Exception as e:
            if self.log_errors:
                logger.debug(f"TokenPolice on_start enrichment failed (swallowed): {e}")

    def on_end(self, span):
        """
        Extract token usage + span hierarchy from the finished span and send to the service.
        This is the main integration point — everything converges here.
        """
        try:
            # Pop this span's recorded obs key FIRST — every on_end branch
            # (suppress/structural/tool/drop/defer) retires the entry exactly
            # once, so only never-ended spans are left to cap eviction. Only
            # the immediate-path drain below consumes the value; the deferred
            # branch drains later via _flush_deferred_spans with the
            # wrapper-captured key instead.
            _span_obs_key = _pop_span_obs_key(span)
            attrs = span.attributes or {}

            # Drop spans the manual wrapper marked for suppression in on_start.
            if attrs.get("tp.suppress"):
                return

            # ── Structural anchor span → emit a zero-token `agent`/`chain` row ──
            # These anchor the trace tree (workflows, chains, and nested
            # sub-agents). They carry native W3C trace/span ids and parent onto
            # the enclosing span.
            if attrs.get("tp.kind") in ("agent", "chain"):
                self._log_agent_span(span, attrs, attrs.get("tp.kind"))
                return

            # ── Tool execution span → emit a zero-usage `tool` row ──
            # tp.kind is set in on_start for the common frameworks; the
            # discriminator is re-checked here so spans whose tool attributes
            # only appear at end (e.g. OpenInference) still emit.
            _name = getattr(span, "name", "") or ""
            if attrs.get("tp.kind") == "tool" or _is_tool_span(attrs, _name):
                self._log_tool_span(span, attrs, _name)
                return

            # ── Drop instrumentor-emitted embedding spans ──
            # opentelemetry-instrumentation-openai / -cohere patch embeddings too
            # and emit a span tagged llm.request.type="embedding". TokenPolice
            # captures embeddings authoritatively via its own manual wrapper
            # (operation=embedding, *_embeddings/*_embed shape) which creates NO
            # OTel span — so any embedding span here is the instrumentor's
            # duplicate, which would otherwise be mis-logged as a *_chat row.
            _emb_scope = getattr(span, "instrumentation_scope", None)
            _emb_scope_name = getattr(_emb_scope, "name", "") if _emb_scope else ""
            if _is_instrumentor_embedding_span(attrs, _name, _emb_scope_name):
                return

            # ── Only process LLM spans ──
            model = attrs.get(_ATTR_MODEL) or attrs.get(_ATTR_MODEL_LEGACY)
            if not model:
                return  # Not an LLM span (could be framework/tool span)

            # The mistralai SDK ships its own TracingHook that emits OTel spans
            # natively, but its after_success hook fires before streamed chunks
            # are consumed — so streaming spans carry no usage. The enforcer
            # already logs every Mistral call via the manual-telemetry path
            # (see _TARGET_METHODS in enforcer.py). Drop Mistral's native spans
            # here — they are emitted from the tracer named "mistralai_sdk_tracer".
            # Same treatment for xai-sdk's native gen_ai.* spans — the enforcer
            # logs xAI calls on the manual path. xai-sdk tracers are named
            # "xai_sdk.sync.chat" / "xai_sdk.aio.chat".
            try:
                scope = getattr(span, "instrumentation_scope", None)
                scope_name = getattr(scope, "name", "") if scope else ""
                if scope_name == "mistralai_sdk_tracer":
                    return
                if scope_name.startswith("xai_sdk"):
                    return
                # Drop the inner google_genai instrumentor span when it fired
                # INSIDE a LangChain call. ChatGoogleGenerativeAI internally calls
                # google.genai.generate_content, which the google_genai
                # instrumentor wraps — producing a DUPLICATE of langchain's
                # `ChatGoogleGenerativeAI.chat` span (double-logged row + ~2x cost).
                # Primary drop is tp.suppress stamped at on_start (survives the
                # LangGraph async contextvar race). Defense-in-depth:
                # in_langchain() OR the process-global trace registry (covers
                # total contextvar loss at on_start where the stamp never
                # fired). Standalone google_genai (no LC guard, no registry,
                # no stamp) is unaffected.
                # Matches BOTH instrumentor generations — the ≤0.7b1 scope and
                # the 1.0b1 util-genai scope conjoined with a Google provider
                # attribute.
                from .context import (
                    in_langchain,
                    get_current_session,
                    langchain_trace_active,
                )
                if _is_google_genai_instrumentor_span(scope_name, attrs):
                    _drop = False
                    try:
                        _drop = bool(in_langchain())
                    except Exception:
                        _drop = False
                    if not _drop:
                        try:
                            _tid = getattr(get_current_session(), "trace_id", "") or ""
                            _drop = bool(_tid) and langchain_trace_active(_tid)
                        except Exception:
                            _drop = False
                    if _drop:
                        return
            except Exception:
                pass

            # Drop inner provider spans emitted under the pydantic_ai or litellm
            # guard — both wrappers log the call manually, so the OpenLLMetry
            # instrumentor's span for the inner provider call is a duplicate.
            # litellm.completion() calls openai.chat.completions.create()
            # internally; that inner call emits an `openai_chat` span here while
            # the litellm manual wrapper separately logs the authoritative
            # `openai_compatible_chat` row via _log_manual → ~2x cost. The
            # litellm wrapper does NOT defer, so this span would otherwise be
            # logged immediately. `_in_litellm` is still set while the inner
            # span finishes, so dropping here leaves exactly one priced row.
            # NOTE: langchain / llamaindex are intentionally NOT dropped here —
            # they set `_defer_telemetry=True` and flush the inner provider span
            # as their single canonical row, so dropping it would erase their
            # telemetry. See _set_langchain_wrapper in enforcer.py.
            try:
                from .context import in_pydantic_ai, in_litellm
                if in_pydantic_ai() or in_litellm():
                    return
            except Exception:
                pass

            # This span is now CONFIRMED to emit an `llm` row (all drop paths
            # above have returned). Register it as a KEPT ancestor so any span
            # that nests under it and resolves later terminates its parent walk
            # here instead of hopping onto the structural root. LLM-ness is only
            # certain at end (the model check + drop gates above), so unlike tool
            # spans this can't be marked in on_start without risking a false
            # positive that would strand a genuine orphan on a dropped span.
            _record_kept_span(span)

            # ── Extract provider ──
            provider = (
                attrs.get(_ATTR_SYSTEM)
                or attrs.get(_ATTR_PROVIDER_NAME)
                or attrs.get(_ATTR_SYSTEM_LEGACY)
                or ""
            )
            # Normalize Google's varied system values (gemini, gcp.gemini,
            # google_genai, ...) to the canonical "google" provider key so
            # provider-keyed pricing and analytics stay consistent — the
            # service keys them on canonical lowercase provider slugs.
            _pl = str(provider).lower()
            if "gemini" in _pl or "google" in _pl:
                provider = "google"
            # OpenLLMetry's bedrock instrumentor reports varied system values
            # (aws, aws.bedrock, ...) — normalize to the canonical "bedrock" key.
            elif "bedrock" in _pl or _pl == "aws":
                provider = "bedrock"
            # Spans that report gen_ai.system="MistralAI" (the LangChain
            # instrumentor does, for ChatMistralAI) — normalize to the
            # canonical "mistral" key. NOT emitted by
            # opentelemetry-instrumentation-mistralai: that instrumentor is
            # never registered (see `_auto_instrument`'s list below).
            elif "mistral" in _pl:
                provider = "mistral"
            # OpenLLMetry's together instrumentor may emit "TogetherAI" or
            # "together" — collapse both to the canonical "together" key.
            elif "together" in _pl:
                provider = "together"
            # Lowercase everything else — the LangChain instrumentor reports
            # mixed case (gen_ai.system = "openai" / "Anthropic" / "Google"),
            # but provider-keyed pricing and analytics are keyed on canonical
            # lowercase provider slugs.
            else:
                provider = _pl

            # ── Extract token usage ──
            input_tokens = (
                attrs.get(_ATTR_INPUT_TOKENS) or 
                attrs.get(_ATTR_PROMPT_TOKENS) or 0
            )
            output_tokens = (
                attrs.get(_ATTR_OUTPUT_TOKENS) or 
                attrs.get(_ATTR_COMPLETION_TOKENS) or 0
            )
            cache_read = int(
                attrs.get(_ATTR_CACHE_READ) or
                attrs.get(_ATTR_CACHE_READ_DOTTED) or 0
            )
            cache_creation = int(
                attrs.get(_ATTR_CACHE_CREATION) or
                attrs.get(_ATTR_CACHE_CREATION_DOTTED) or 0
            )
            reasoning_tokens = int(attrs.get(_ATTR_REASONING) or 0)
            # Cache read and cache creation (write) MUST stay disjoint: writes bill
            # at a different (higher) rate, so they are forwarded as separate keys
            # below. `cached_tokens` follows OpenAI semantics — read hits only.
            cached_tokens = cache_read

            # ── Extract TokenPolice session context ──
            user_id = attrs.get("tp.user_id", "anonymous")
            paid_plan = attrs.get("tp.paid_plan", "free")
            plan_source = _plan_source_from_attrs(attrs)
            workflow_name = attrs.get("tp.workflow_name", "default")
            session_id = attrs.get("tp.session_id", "")

            # ── Build span hierarchy ──
            trace_id = attrs.get("tp.trace_id", "")
            root_span_id = attrs.get("tp.root_span_id", "")
            span_order = attrs.get("tp.span_order", 0)
            span_name = attrs.get("tp.span_name", model)  # Default span_name to model

            # Use the OTel span's NATIVE W3C ids so the parent pointer resolves
            # to the real parent row (the enclosing agent span, or a nested OTel
            # span). native trace_id equals session.trace_id (set from the agent
            # span), so the composition key below still matches.
            from .context import random_hex16
            native_trace_id = trace_id
            native_span_id = None
            native_parent_id = root_span_id
            try:
                sctx = getattr(span, "context", None)
                if sctx is not None and getattr(sctx, "trace_id", 0):
                    native_trace_id = trace.format_trace_id(sctx.trace_id)
                    native_span_id = trace.format_span_id(sctx.span_id)
                parent = getattr(span, "parent", None)
                if parent is not None and getattr(parent, "span_id", 0):
                    native_parent_id = trace.format_span_id(parent.span_id)
            except Exception:
                pass

            # Rewrite over any dropped framework ancestors (LangChain/LangGraph
            # intermediate spans) so this LLM call nests under the surviving
            # agent/chain root instead of being orphaned. No-op when the parent
            # is a kept span.
            native_parent_id = _resolve_kept_parent(native_parent_id, root_span_id)

            span_obj = {
                "trace_id": native_trace_id,
                "span_id": native_span_id or random_hex16(),
                "parent_span_id": native_parent_id,
                "span_kind": "llm",
                "span_name": span_name,
                "span_order": span_order,
                "start_time": datetime.fromtimestamp(span.start_time / 1e9, timezone.utc).isoformat() if span.start_time else None,
                "end_time": datetime.fromtimestamp(span.end_time / 1e9, timezone.utc).isoformat() if span.end_time else None,
            }

            # ── Extract composition from pending context ──
            prompt_comp = []
            response_comp = []
            api_base = ""
            service_tier = ""
            gw_original_provider = ""
            # Default so the forward-compat `comp_data.get("latency")`
            # read on the immediate log path can't raise NameError when no
            # composition was stashed for this span (comp_data is otherwise
            # only bound inside the `_pending_compositions` block below).
            comp_data: Dict[str, Any] = {}
            # Default so the later `getattr(session, '_defer_telemetry', ...)`
            # read on the immediate log path can't raise NameError when session
            # resolution fails: `session` is otherwise only bound inside the
            # best-effort `get_current_session()` try below, and an import
            # failure / get_current_session() raising would leave it unbound —
            # the outer handler would then drop the whole telemetry row instead
            # of emitting it (just without the session-derived enrichment).
            session = None
            # Did this span end in ERROR? Traceloop ends the span DURING the
            # failing call (before the enforcer's failure handler runs), so a
            # destructive pop here would consume the pending-composition stash
            # that _emit_call_failure_log needs for the failed row's
            # prompt_composition. On errored spans we PEEK instead of pop —
            # the failure logger pops the entry right after. Tradeoff: if an
            # errored instrumented span is ever NOT followed by a failure log
            # (no enforcer wrapper on the call), the entry lingers on the
            # session's _pending_compositions dict until the session object is
            # released — bounded, and preferable to clearing the whole trace's
            # stash (which could discard a concurrent sibling call's entry).
            _span_errored = False
            try:
                from opentelemetry.trace import StatusCode
                _st = getattr(span, "status", None)
                _span_errored = bool(
                    _st is not None
                    and getattr(_st, "status_code", None) == StatusCode.ERROR
                )
            except Exception:
                _span_errored = False
            try:
                from .context import get_current_session
                session = get_current_session()
                # Composition is stored per-span-order by the enforcer
                comp_key = f"{trace_id}:{span_order}"
                if hasattr(session, '_pending_compositions') and comp_key in session._pending_compositions:
                    if _span_errored:
                        comp_data = dict(session._pending_compositions.get(comp_key) or {})
                    else:
                        comp_data = session._pending_compositions.pop(comp_key, {})
                    prompt_comp = comp_data.get("prompt", [])
                    response_comp = comp_data.get("response", [])
                    # Provider-reported service tier stashed by the enforcer
                    # post-hook → forwarded as usage.tier (tier pricing). On the
                    # deferred path the stash usually lands after this pop;
                    # _flush_deferred_spans merges it into the payload then.
                    service_tier = comp_data.get("service_tier", "")
                    # The enforcer stashes the real provider for SDK calls whose
                    # gen_ai.system is misleading (e.g. the OpenAI SDK pointed at
                    # OpenRouter reports gen_ai.system="openai").
                    if comp_data.get("provider"):
                        provider = comp_data["provider"]
                    # The enforcer stashes the full Bedrock model id — the OTel
                    # instrumentor strips its vendor prefix (amazon.nova-lite ->
                    # nova-lite), which would break the service's price lookup.
                    if comp_data.get("model"):
                        model = comp_data["model"]
                    # Serving endpoint the enforcer captured — forwarded so the
                    # service can identify the serving provider (host -> provider).
                    if comp_data.get("api_base"):
                        api_base = comp_data["api_base"]
                    # Gateway-routed call (e.g. OpenAI SDK -> OpenRouter): the
                    # enforcer stashed the vendor head of the customer's model
                    # slug ("openai/gpt-4.1-nano" -> "openai"). Forwarded in
                    # model_extras below. Only the gateway stash sets this key;
                    # every non-gateway span leaves it "".
                    if comp_data.get("original_provider"):
                        gw_original_provider = str(comp_data["original_provider"])
            except Exception:
                pass  # Composition is best-effort

            # ── Build composition from span attributes (fallback) ──
            # OpenLLMetry sets gen_ai.prompt.N.content/role and gen_ai.completion.N.content/role
            try:
                import hashlib

                if not prompt_comp:
                    prompt_entries = []
                    for key, val in attrs.items():
                        k = str(key)
                        if k.startswith("gen_ai.prompt.") and k.endswith(".content"):
                            idx = k.split(".")[2]
                            role_key = f"gen_ai.prompt.{idx}.role"
                            role = str(attrs.get(role_key, "user"))
                            content = str(val).strip()
                            h = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:16]
                            prompt_entries.append({
                                "role": role, "type": "text",
                                "length": len(content), "hash": h,
                            })
                    if prompt_entries:
                        prompt_comp = prompt_entries

                if not response_comp:
                    resp_entries = []
                    _is_cohere = str(provider).lower() == "cohere"
                    # Collect the completion indices present on the span.
                    completion_idxs = set()
                    for key in attrs:
                        k = str(key)
                        if k.startswith("gen_ai.completion."):
                            parts = k.split(".")
                            if len(parts) >= 3:
                                completion_idxs.add(parts[2])
                    for idx in sorted(completion_idxs):
                        role = str(attrs.get(f"gen_ai.completion.{idx}.role", "assistant"))
                        # The Python LangChain OpenLLMetry instrumentor emits
                        # role="unknown" for completion attributes; normalize
                        # so the UI doesn't surface "Unknown" as the response role.
                        if role.lower() == "unknown":
                            role = "assistant"
                        raw_content = attrs.get(f"gen_ai.completion.{idx}.content")
                        if raw_content is not None:
                            content_str = str(raw_content)
                            blocks = None
                            # OpenLLMetry's Cohere instrumentor stores the
                            # response content as a JSON list of {type,text}
                            # blocks (including the tool_plan); other providers
                            # store plain text.
                            if _is_cohere and content_str.strip().startswith("["):
                                try:
                                    parsed = json.loads(content_str)
                                    if isinstance(parsed, list):
                                        blocks = parsed
                                except Exception:
                                    blocks = None
                            if blocks is not None:
                                for block in blocks:
                                    if isinstance(block, dict) and block.get("type") == "text":
                                        txt = str(block.get("text", "")).strip()
                                        h = hashlib.sha1(txt.encode("utf-8", errors="replace")).hexdigest()[:16]
                                        resp_entries.append({
                                            "role": role, "type": "text",
                                            "length": len(txt), "hash": h,
                                        })
                                    elif isinstance(block, dict):
                                        resp_entries.append({
                                            "role": role,
                                            "type": str(block.get("type", "unknown")),
                                        })
                            else:
                                content_str = content_str.strip()
                                h = hashlib.sha1(content_str.encode("utf-8", errors="replace")).hexdigest()[:16]
                                resp_entries.append({
                                    "role": role, "type": "text",
                                    "length": len(content_str), "hash": h,
                                })
                        # Tool calls: gen_ai.completion.{idx}.tool_calls.{m}.{name,arguments}
                        m = 0
                        while True:
                            tc_name = attrs.get(f"gen_ai.completion.{idx}.tool_calls.{m}.name")
                            tc_args = attrs.get(f"gen_ai.completion.{idx}.tool_calls.{m}.arguments")
                            if tc_name is None and tc_args is None:
                                break
                            args_str = str(tc_args or "")
                            entry = {
                                "role": "tool_call", "type": "tool_call",
                                "length": len(args_str),
                                "hash": hashlib.sha1(args_str.encode("utf-8", errors="replace")).hexdigest()[:16],
                            }
                            if tc_name:
                                entry["name"] = str(tc_name)
                            resp_entries.append(entry)
                            m += 1
                    if resp_entries:
                        response_comp = resp_entries
            except Exception:
                pass  # Best-effort

            # Gemini via LangChain reports the model as "models/gemini-2.5-flash";
            # strip the resource prefix so the server-side price lookup matches.
            if isinstance(model, str) and model.startswith("models/"):
                model = model[len("models/"):]

            # ── Build metadata ──
            metadata = {"workflow_name": workflow_name}
            if session_id:
                metadata["session_id"] = session_id
            # Hoisted out of the loop — one resolve per row, not per attribute.
            _routing_attr_name = _routing_attr()
            for k, v in attrs.items():
                # B4: `_tp_routing` was snapshotted off session metadata at
                # span START, so it rides EVERY span opened after ANY reroute —
                # not just the rerouted call's. Strip it here; the block below
                # re-adds it only for the call that was actually rerouted.
                if k == _routing_attr_name:
                    continue
                if isinstance(k, str) and k.startswith("tp.meta."):
                    metadata[k[len("tp.meta."):]] = v
            # B4: re-add this call's OWN marker, keyed by the span's recorded obs
            # key (the same key the local_decision claim below uses). Only the
            # rerouted call's row shows it — and it shows on EVERY row that call
            # emits, since the peek never removes the record. Serialized to JSON
            # to preserve the wire shape this path has always produced for
            # object-valued metadata (the span-attribute hop json.dumps() it).
            # Covers the deferred path too: the payload built below carries this
            # same dict, and `_span_obs_key` is this span's own recorded key.
            try:
                _bridge = _routing_bridge()
                if _bridge is not None:
                    from .context import get_current_session as _gcs_routing
                    _bridge[1](metadata, _gcs_routing(), _span_obs_key,
                               serialize=True)
            except Exception:
                pass  # fail-open: the row simply carries no reroute provenance

            # Synthesise a usage block from the gen_ai.* semconv attributes so
            # the service's usage parsing has a shape it knows. The Mode-A
            # SpanProcessor never sees the raw provider usage object
            # (instrumentors only emit semconv attrs), so this is a best-effort
            # reconstruction. Mode C / D wrappers in enforcer.py forward the
            # verbatim usage object instead — this fallback path only covers
            # the Mode-A case.
            shape_by_provider = {
                "anthropic": "anthropic_messages",
                "openai": "openai_chat",
                "google": "google_genai",
                "gemini": "google_genai",
                "bedrock": "bedrock_converse",
                "cohere": "cohere_chat",
                "openrouter": "openrouter_routed",
            }
            usage_shape = shape_by_provider.get((provider or "").lower(), "openai_compatible_chat")
            # N1 — provider and shape are TWO AXES, and only one of them is the
            # serving vendor:
            # * payload `provider` = SERVING slug: the host that
            # actually billed the call, e.g. an Anthropic SDK client pointed
            # at api.minimax.io reports "minimax" (stashed by the enforcer
            # and applied over `provider` above). Never weaken this.
            # * `usage.shape` = WIRE surface: which client SDK /
            # instrumentor produced these token fields.
            # The lookup above keys on the post-override (serving) provider, so
            # a host remap mislabels Anthropic-wire usage as
            # openai_compatible_chat ("minimax" is not in the table). The
            # server then runs the OpenAI mapper over Anthropic fields:
            # `cache_creation_input_tokens` is unknown there so cache WRITES
            # land unpriced in extra_units ($0), and Anthropic's `input_tokens`
            # is already cache-EXCLUSIVE so cache reads get subtracted a second
            # time (under-counted text input).
            #
            # The wire-first override is deliberately ANTHROPIC-ONLY. Anthropic
            # is the one family whose wire shape is not OpenAI-compatible while
            # still being reachable through a remapped host. Widening it to
            # google/bedrock/cohere would regress paths that correctly emit
            # openai_compatible_chat today (notably the LangChain-Gemini spans,
            # which carry a Google gen_ai.system but OpenAI-shaped token
            # fields). Fully fail-open: any error keeps the lookup result.
            try:
                # Wire signal 1: `_pl` is the raw lower-cased gen_ai.system /
                # gen_ai.provider.name / llm.system value captured before the
                # canonical-slug normalisation and before the serving override
                # (the anthropic instrumentor emits gen_ai.provider.name=
                # "anthropic"; the LangChain instrumentor emits "Anthropic").
                _wire_is_anthropic = "anthropic" in _pl
                if not _wire_is_anthropic:
                    # Signal 2: the emitting instrumentation scope, for spans
                    # carrying no gen_ai.system attr at all. Python scope is
                    # "opentelemetry.instrumentation.anthropic"; the Node SDK's
                    # is "@traceloop/instrumentation-anthropic".
                    _sn = str(_emb_scope_name or "").lower()
                    _wire_is_anthropic = (
                        "instrumentation.anthropic" in _sn
                        or "instrumentation-anthropic" in _sn
                    )
                if _wire_is_anthropic:
                    usage_shape = "anthropic_messages"
            except Exception:
                pass
            raw_synth: Dict[str, Any] = {
                "prompt_tokens": int(input_tokens),
                "completion_tokens": int(output_tokens),
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
            }
            if cache_read:
                raw_synth["prompt_tokens_details"] = {"cached_tokens": cache_read}
                raw_synth["cache_read_input_tokens"] = cache_read
            if cache_creation:
                raw_synth["cache_creation_input_tokens"] = cache_creation
            if reasoning_tokens:
                raw_synth["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}
            usage_block = {"shape": usage_shape, "raw": raw_synth}
            if service_tier:
                usage_block["tier"] = service_tier

            # ── Send to the TokenPolice service ──
            payload = dict(
                user_id=user_id,
                paid_plan=paid_plan,
                # None here is fine: log_sync omits the wire field unless the
                # value is exactly "app"/"default". Rides on both the immediate
                # and the deferred path (enforcer flushes via log_sync(**payload)).
                plan_source=plan_source,
                workflow_name=workflow_name,
                session_id=session_id,
                model=model,
                provider=provider,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cached_tokens=int(cached_tokens),
                metadata=metadata,
                span=span_obj,
                prompt_composition=prompt_comp,
                response_composition=response_comp,
                usage=usage_block,
            )
            _extras = {}
            if api_base:
                _extras["api_base"] = api_base
            if gw_original_provider:
                _extras["original_provider"] = gw_original_provider
            if _extras:
                payload["model_extras"] = _extras

            if getattr(session, '_defer_telemetry', False):
                if not hasattr(session, '_deferred_spans'):
                    session._deferred_spans = []
                session._deferred_spans.append(payload)
            else:
                from .state import get_client, drain_observations
                client = get_client()
                if client:
                    # Attach call_outcome / observations / latency on the
                    # IMMEDIATE (non-deferred) path — the deferred branch is
                    # handled by enforcer._flush_deferred_spans. Both drains
                    # are keyed to the owning call's obs key, so a call's
                    # entries go to whichever of its OWN paths runs first and
                    # never to another call's /log.
                    # call_outcome gate is asymmetric: failed → ALWAYS (mirror
                    # the tool-span call_outcome mapping idiom, which never
                    # gates on duration); success → only when duration_ms>0
                    # (mirror the Node SDK).
                    duration_ms = 0
                    try:
                        if span.start_time and span.end_time:
                            from ._classify import to_duration_ms
                            duration_ms = to_duration_ms(
                                (span.end_time - span.start_time) / 1e6
                            )
                    except Exception:
                        duration_ms = 0
                    if _span_errored:
                        call_outcome = {"status": "failed", "duration_ms": duration_ms}
                        try:
                            # Route the RAW error through the scrub helper —
                            # default 'redacted' ships a hash, never raw text.
                            from ._classify import scrub_error_message, resolve_error_detail
                            err_msg = ""
                            st = getattr(span, "status", None)
                            if st is not None:
                                err_msg = str(getattr(st, "description", "") or "")
                            call_outcome.update(scrub_error_message(err_msg, resolve_error_detail()))
                            ek = attrs.get("error.type")
                            if ek:
                                call_outcome["error_kind"] = str(ek)
                        except Exception:
                            pass
                        payload["call_outcome"] = call_outcome
                    elif duration_ms > 0:
                        payload["call_outcome"] = {"status": "success", "duration_ms": duration_ms}
                    # observations: drain exactly once, attach only when
                    # non-empty. Keyed by the span's recorded obs key (side
                    # map written in on_start inside the owning call's check
                    # context, popped at on_end entry — the on_start object
                    # itself never reaches on_end in OTel Python): a call's
                    # entries go to whichever of ITS OWN drain paths runs
                    # first, so cross-call theft is impossible before the
                    # staleness window. Unrecorded span → None → untagged +
                    # stale entries only.
                    try:
                        observations = drain_observations(_span_obs_key)
                    except Exception:
                        observations = []
                    if observations:
                        payload["observations"] = observations
                    # Ship applied local_decision on the immediate success
                    # path (deferred path already does this in _flush_deferred_spans).
                    # Claimed with the SAME span-recorded obs key as the
                    # observations drain above, so the decision lands on ITS
                    # OWN call's row: a burst of concurrent calls now emits one
                    # REQUEST_REROUTED each, instead of one for the whole burst
                    # stamped on whichever row finished first. Lazy import —
                    # enforcer owns the keyed store and imports telemetry only
                    # inside functions, so neither module-level cycle exists.
                    #
                    # LOAD-BEARING ASYMMETRY (bedrock): this key is only the
                    # call's own key when the span OPENS INSIDE the enforcer
                    # wrapper, i.e. after that call's /check minted it. The
                    # bedrock instrumentor patches the per-client methods
                    # OUTSIDE our wrapper, so its span records a key that
                    # PREDATES the check and the claim below finds nothing —
                    # the decision is left in the store and swept at expiry.
                    # Harmless today (bedrock's kwargs-less call shape is
                    # unappliable, so it never carries an applied reroute), but
                    # if bedrock ever gains an appliable shape, its SUCCESS
                    # rows would silently lose REQUEST_REROUTED. The fix then
                    # is the pattern the FAILURE path already uses: have the
                    # enforcer wrapper claim with its own obs key and hand the
                    # decision over on the session (see
                    # `_bedrock_chat_failure_handoff`'s stash and the merge
                    # block below), never a keyless claim here.
                    try:
                        from .context import get_current_session as _gcs
                        from .enforcer import _claim_local_decision
                        _sess = _gcs()
                        _ld = _claim_local_decision(_sess, _span_obs_key)
                        if _ld:
                            payload["local_decision"] = _ld
                    except Exception:
                        pass
                    # ── Bedrock failed-call merge (single-emitter handoff) ──
                    # The botocore enforcer wrapper runs INSIDE the bedrock
                    # instrumentor's span, so on a failed chat call its except
                    # handler stashes the classified failure payload on the
                    # session (instead of emitting a duplicate row) and this
                    # span — the surviving emitter — absorbs it here.
                    # error_kind convention: the stashed CLASSIFIED kind
                    # (auth_error, rate_limited, ...) wins when truthy and not
                    # "unknown" — consistent with every other provider's
                    # failure rows, which come from the classifier; the span's
                    # error.type (exception class name, absent entirely on the
                    # aiobotocore non-stream path) stays as the fallback for
                    # classifier misses. Fully guarded: any internal error
                    # emits the row exactly as it was built above.
                    try:
                        _bstash = None
                        if (_span_errored and session is not None
                                and ("bedrock" in str(_emb_scope_name or "").lower()
                                     or str(provider).lower() == "bedrock")):
                            _bstash = getattr(session, "_pending_bedrock_failure", None)
                            if isinstance(_bstash, dict):
                                # Consume first: even if the merge below dies,
                                # the stash must not resurface as a late
                                # fallback row (this row still emits).
                                session._pending_bedrock_failure = None
                            else:
                                _bstash = None
                        if _bstash is not None:
                            # Observations drained by the enforcer with the
                            # call-scoped key (this span's recorded key
                            # predates the call's /check, so the drain above
                            # could not claim them).
                            _bobs = _bstash.get("observations") or []
                            if _bobs:
                                payload["observations"] = list(
                                    payload.get("observations") or []) + list(_bobs)
                            _bout = _bstash.get("call_outcome")
                            _co = payload.get("call_outcome")
                            if isinstance(_bout, dict):
                                if not isinstance(_co, dict):
                                    _co = {"status": "failed",
                                           "duration_ms": duration_ms}
                                    payload["call_outcome"] = _co
                                _bkind = _bout.get("error_kind")
                                if _bkind and _bkind != "unknown":
                                    _co["error_kind"] = _bkind
                                if not _co.get("http_status") and _bout.get("http_status"):
                                    _co["http_status"] = _bout["http_status"]
                                if not _co.get("error_class") and _bout.get("error_class"):
                                    _co["error_class"] = _bout["error_class"]
                            _bmodel = _bstash.get("model")
                            if _bmodel and (not payload.get("model")
                                            or payload.get("model") == "unknown"):
                                payload["model"] = _bmodel
                            # local_decision travels IN the stash: this span's
                            # recorded obs key predates the failed call's
                            # /check, so the keyed claim above cannot reach the
                            # call's own decision — the enforcer's handoff
                            # claimed it for us. Never overwrite a decision the
                            # claim above already attached.
                            _bld = _bstash.get("local_decision")
                            if _bld and not payload.get("local_decision"):
                                payload["local_decision"] = _bld
                            # The errored-span peek-not-pop above existed so
                            # the enforcer's failure logger could consume the
                            # composition entry; the handoff consumed it here
                            # instead — pop so it can't linger until session
                            # release.
                            try:
                                if hasattr(session, "_pending_compositions"):
                                    session._pending_compositions.pop(
                                        f"{trace_id}:{span_order}", None)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    # latency: forward-compat read (NameError-safe via comp_data={} default)
                    _latency = comp_data.get("latency")
                    if _latency:
                        payload["latency"] = _latency
                    client.log_sync(**payload)

        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice telemetry extraction failed (swallowed): {e}")

    def _log_agent_span(self, span, attrs, kind="agent"):
        """
        Emit a structural (``agent``/``chain``) span row for a workflow / chain /
        sub-agent anchor span. Zero tokens, no model, no composition — it exists
        only to give the trace tree a real root/branch node. All fields come from
        span attributes + native context (no session lookup, so it is
        timing-independent).
        """
        # Only agent/chain are valid structural anchors; anything else → agent.
        structural_kind = "chain" if kind == "chain" else "agent"
        try:
            from .state import get_client
            from .context import random_hex16
            client = get_client()
            if not client:
                return

            user_id = attrs.get("tp.user_id", "anonymous")
            paid_plan = attrs.get("tp.paid_plan", "free")
            plan_source = _plan_source_from_attrs(attrs)
            workflow_name = attrs.get("tp.workflow_name", "default")
            session_id = attrs.get("tp.session_id", "")

            metadata = {"workflow_name": workflow_name}
            if session_id:
                metadata["session_id"] = session_id
            # Hoisted out of the loop — one resolve per row, not per attribute.
            _routing_attr_name = _routing_attr()
            for k, v in attrs.items():
                # B4: `_tp_routing` was snapshotted off session metadata at
                # span START, so it rides EVERY span opened after ANY reroute.
                # An agent/chain row executes no model call, so it can never own
                # reroute provenance: stripped here and never re-added.
                if k == _routing_attr_name:
                    continue
                if isinstance(k, str) and k.startswith("tp.meta."):
                    metadata[k[len("tp.meta."):]] = v

            native_trace_id = ""
            native_span_id = random_hex16()
            native_parent_id = ""
            try:
                sctx = getattr(span, "context", None)
                if sctx is not None and getattr(sctx, "trace_id", 0):
                    native_trace_id = trace.format_trace_id(sctx.trace_id)
                    native_span_id = trace.format_span_id(sctx.span_id)
                parent = getattr(span, "parent", None)
                if parent is not None and getattr(parent, "span_id", 0):
                    native_parent_id = trace.format_span_id(parent.span_id)
            except Exception:
                pass

            span_obj = {
                "trace_id": native_trace_id,
                "span_id": native_span_id,
                "parent_span_id": native_parent_id,
                "span_kind": structural_kind,
                "span_name": workflow_name,
                "span_order": 0,
                "start_time": datetime.fromtimestamp(span.start_time / 1e9, timezone.utc).isoformat() if span.start_time else None,
                "end_time": datetime.fromtimestamp(span.end_time / 1e9, timezone.utc).isoformat() if span.end_time else None,
            }

            # Map the anchor span's OTel status → call_outcome so a workflow /
            # agent / chain whose body raised uncaught reports status='failed'
            # instead of the server's default 'success'. OTel already
            # stamped StatusCode.ERROR on propagation (set_status_on_exception);
            # we only READ it here inside the already-swallowed on_end, so there
            # is no control-flow change and the customer's exception is untouched.
            from opentelemetry.trace import StatusCode
            failed = False
            err_msg = ""
            try:
                st = getattr(span, "status", None)
                if st is not None and getattr(st, "status_code", None) == StatusCode.ERROR:
                    failed = True
                    err_msg = str(getattr(st, "description", "") or "")
            except Exception:
                pass
            duration_ms = 0
            try:
                if span.start_time and span.end_time:
                    from ._classify import to_duration_ms
                    duration_ms = to_duration_ms(
                        (span.end_time - span.start_time) / 1e6
                    )
            except Exception:
                pass
            call_outcome = {
                "status": "failed" if failed else "success",
                "duration_ms": duration_ms,
            }
            if failed:
                # Route the RAW status text through the scrub helper (no
                # pre-stringify); default 'redacted' ships a hash, never raw text.
                try:
                    from ._classify import scrub_error_message, resolve_error_detail
                    call_outcome.update(scrub_error_message(err_msg, resolve_error_detail()))
                    ek = attrs.get("error.type")
                    if ek:
                        call_outcome["error_kind"] = str(ek)
                except Exception:
                    pass

            # Zero-usage structural row: only span_kind='llm' rows participate
            # in budget/loop evaluation and analytics rollups, so this is safe
            # to send.
            client.log_sync(
                user_id=user_id,
                paid_plan=paid_plan,
                plan_source=plan_source,
                workflow_name=workflow_name,
                session_id=session_id,
                model="",
                provider="",
                input_tokens=0,
                output_tokens=0,
                cached_tokens=0,
                metadata=metadata,
                span=span_obj,
                prompt_composition=[],
                response_composition=[],
                call_outcome=call_outcome,
            )
        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice agent-span logging failed (swallowed): {e}")

    def _log_tool_span(self, span, attrs, name):
        """
        Emit a ``tool`` span row for a tool / function execution. Zero tokens,
        no model, no cost — it slots into the trace tree between LLM calls.
        Tool args/results are reduced to (sha1-16 hash, length); raw content
        never leaves the process.
        """
        try:
            from opentelemetry.trace import StatusCode
            from .state import get_client
            from .context import random_hex16
            client = get_client()
            if not client:
                return

            tool_name = _tool_name_from_span(attrs, name)
            tool_type = str(attrs.get("gen_ai.tool.type") or "function")
            tool_call_id = str(attrs.get("gen_ai.tool.call.id") or "")
            # OTel frameworks (LangChain/LangGraph/CrewAI/Agno) rarely set
            # gen_ai.tool.call.id. Fall back to the pending stash populated from
            # the preceding LLM response (name-exact FIFO). Prefer the attr when
            # present; never invent ids.
            if not tool_call_id:
                try:
                    from .context import _pop_pending_tool_call_id
                    tool_call_id = _pop_pending_tool_call_id(tool_name) or ""
                except Exception:
                    tool_call_id = ""
            param_hash, param_len = _hash_len(
                attrs.get("traceloop.entity.input")
                or attrs.get("gen_ai.tool.call.arguments")
                or attrs.get("tool.parameters")
            )
            result_hash, result_len = _hash_len(
                attrs.get("traceloop.entity.output")
                or attrs.get("gen_ai.tool.call.result")
            )

            user_id = attrs.get("tp.user_id", "anonymous")
            paid_plan = attrs.get("tp.paid_plan", "free")
            plan_source = _plan_source_from_attrs(attrs)
            workflow_name = attrs.get("tp.workflow_name", "default")
            session_id = attrs.get("tp.session_id", "")
            metadata = {"workflow_name": workflow_name}
            if session_id:
                metadata["session_id"] = session_id
            # Hoisted out of the loop — one resolve per row, not per attribute.
            _routing_attr_name = _routing_attr()
            for k, v in attrs.items():
                # B4: `_tp_routing` was snapshotted off session metadata at
                # span START, so it rides EVERY span opened after ANY reroute.
                # A tool row executes no model call, so it can never own reroute
                # provenance: stripped here and never re-added.
                if k == _routing_attr_name:
                    continue
                if isinstance(k, str) and k.startswith("tp.meta."):
                    metadata[k[len("tp.meta."):]] = v

            native_trace_id = attrs.get("tp.trace_id", "")
            native_span_id = random_hex16()
            native_parent_id = ""
            try:
                sctx = getattr(span, "context", None)
                if sctx is not None and getattr(sctx, "trace_id", 0):
                    native_trace_id = trace.format_trace_id(sctx.trace_id)
                    native_span_id = trace.format_span_id(sctx.span_id)
                parent = getattr(span, "parent", None)
                if parent is not None and getattr(parent, "span_id", 0):
                    native_parent_id = trace.format_span_id(parent.span_id)
            except Exception:
                pass

            # Rewrite over dropped framework ancestors so this tool span nests
            # under the surviving agent/chain root (orphan-rewrite). No-op when
            # the parent is a kept span. Tool spans carry no tp.root_span_id, so
            # the fallback is "" — identical to today's default.
            native_parent_id = _resolve_kept_parent(
                native_parent_id, attrs.get("tp.root_span_id", "")
            )

            # Map OTel span status → call_outcome.
            failed = False
            err_msg = ""
            try:
                st = getattr(span, "status", None)
                if st is not None and getattr(st, "status_code", None) == StatusCode.ERROR:
                    failed = True
                    err_msg = str(getattr(st, "description", "") or "")
            except Exception:
                pass
            duration_ms = 0
            try:
                if span.start_time and span.end_time:
                    from ._classify import to_duration_ms
                    duration_ms = to_duration_ms(
                        (span.end_time - span.start_time) / 1e6
                    )
            except Exception:
                pass
            call_outcome = {
                "status": "failed" if failed else "success",
                "duration_ms": duration_ms,
            }
            if failed:
                # Route the RAW value through the scrub helper (no
                # pre-stringify); default 'redacted' ships a hash, not raw text.
                from ._classify import scrub_error_message, resolve_error_detail
                call_outcome.update(scrub_error_message(err_msg, resolve_error_detail()))
                ek = attrs.get("error.type")
                if ek:
                    call_outcome["error_kind"] = str(ek)

            span_obj = {
                "trace_id": native_trace_id,
                "span_id": native_span_id,
                "parent_span_id": native_parent_id,
                "span_kind": "tool",
                "span_name": tool_name,
                "span_order": 0,
                "start_time": datetime.fromtimestamp(span.start_time / 1e9, timezone.utc).isoformat() if span.start_time else None,
                "end_time": datetime.fromtimestamp(span.end_time / 1e9, timezone.utc).isoformat() if span.end_time else None,
            }

            client.log_sync(
                user_id=user_id,
                paid_plan=paid_plan,
                plan_source=plan_source,
                workflow_name=workflow_name,
                session_id=session_id,
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
                    "name": tool_name,
                    "type": tool_type,
                    "call_id": tool_call_id,
                    "param_hash": param_hash,
                    "param_length": param_len,
                    "result_hash": result_hash,
                    "result_length": result_len,
                },
                call_outcome=call_outcome,
            )
        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice tool-span logging failed (swallowed): {e}")

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        pass


# ═══════════════════════════════════════════════════════════════════
# Setup function — called by tp.init()
# ═══════════════════════════════════════════════════════════════════

def setup_opentelemetry(log_errors: bool = False, auto_instrument: bool = True, tracer_provider=None):
    """
    Initialize the local-only OpenTelemetry pipeline:
    1. Attach our SpanProcessor to the right TracerProvider (mirrors the Node
       SDK's proxy-detect + piggyback attach behavior so we neither silently
       miss spans when a customer set their provider first, nor clobber their
       observability when tp.init ran first).
    2. Optionally auto-instrument installed LLM SDKs via OpenLLMetry

    Returns the TracerProvider our processor now lives on (our own in the
    unset/fallback branches, the customer's in the piggyback/injected branches).
    """
    global _tp_created_provider

    processor = TokenPoliceSpanProcessor(log_errors=log_errors)

    def _build_and_set_own():
        # Historical byte-equivalent path: build a fresh provider, attach our
        # processor, and set it as the global. set_tracer_provider runs under
        # OTel's do_once, so if a real global already exists this is a logged
        # no-op and we never clobber the customer's provider.
        global _tp_created_provider
        own = TracerProvider()
        own.add_span_processor(processor)
        trace.set_tracer_provider(own)
        _tp_created_provider = own
        return own

    # Provider-detection is wrapped in its own fail-open guard: any probe
    # (isinstance / hasattr / attach) that raises degrades to build-and-set our
    # own provider, and if even that raises we fall through to a no-op return.
    try:
        provider = None
        if tracer_provider is not None:
            # Injected provider: honor it, checked BEFORE detection.
            if hasattr(tracer_provider, "add_span_processor"):
                # Attach to the injected provider; do NOT set_tracer_provider.
                # Leave _tp_created_provider = None (we didn't build it).
                tracer_provider.add_span_processor(processor)
                provider = tracer_provider
            else:
                # Attach-less injected object: don't touch it, fall open to
                # building + setting our own provider.
                provider = _build_and_set_own()
        else:
            current = trace.get_tracer_provider()
            if ProxyTracerProvider is not None and isinstance(current, ProxyTracerProvider):
                # No global provider set yet — own it (today's default path).
                provider = _build_and_set_own()
            elif hasattr(current, "add_span_processor"):
                # A real customer provider already exists — piggyback onto it,
                # do NOT replace. Leave _tp_created_provider = None.
                current.add_span_processor(processor)
                provider = current
            else:
                # Unknown / attach-less pre-existing provider. INTENTIONAL
                # divergence from Node (which has no final else): attempt
                # set-our-own, which is a harmless do_once no-op if a real
                # global exists. Call NO method on the pre-existing provider.
                provider = _build_and_set_own()
    except Exception:
        try:
            provider = _build_and_set_own()
        except Exception:
            provider = None

    if auto_instrument:
        _auto_instrument(log_errors)

    return provider


def _auto_instrument(log_errors: bool = False):
    """Try to instrument each installed LLM SDK. Silently skip unavailable ones."""
    instrumentors = [
        ("opentelemetry.instrumentation.openai", "OpenAIInstrumentor"),
        ("opentelemetry.instrumentation.anthropic", "AnthropicInstrumentor"),
        # NOTE: the first element is the IMPORT MODULE PATH handed to
        # importlib.import_module below — NOT a tracer scope name. It merely
        # happens to equal the ≤0.7b1 tracer scope. On 1.0b1 this module path is
        # still correct (the module and GoogleGenAiSdkInstrumentor both exist)
        # while the emitted tracer scope moved to
        # opentelemetry.util.genai.handler. Do not "fix" this string when
        # touching the suppression gates.
        ("opentelemetry.instrumentation.google_genai", "GoogleGenAiSdkInstrumentor"),
        # NOTE: we intentionally do NOT register opentelemetry-instrumentation-
        # cohere. It only supports `cohere <6` (no release covers cohere 6.x/7.x),
        # logs a spurious OTel DependencyConflict when activated, and — if it were
        # active on cohere <6 — would double-log alongside our manual path. Cohere
        # v2 chat AND embed are captured on the manual-telemetry path instead
        # (enforcer.py: `_extract_cohere_chat_usage` / cohere embed Mode C),
        # version-resilient across cohere 5/6/7.
        # NOTE: we intentionally do NOT register opentelemetry-instrumentation-
        # mistralai here — on ANY mistralai version. Two independent reasons:
        #   1. Its module body does `from mistralai.models import ...` at
        #      import time. `mistralai.models` does not exist on mistralai>=2
        #      (2.0 made the package a namespace and moved everything under
        #      `mistralai.client.*`), so merely importing the instrumentor
        #      raises ModuleNotFoundError — verified against
        #      opentelemetry-instrumentation-mistralai 0.62.3 + mistralai 2.9.4.
        #   2. Even where it does import, the mistralai SDK's own native
        #      TracingHook fires its httpx `after_success` hook BEFORE a stream
        #      is consumed, so streaming spans carry no usage at all.
        # The TokenPolice enforcer wraps the mistralai SDK on the manual-
        # telemetry path instead, for both majors (see `_TARGET_METHODS` in
        # enforcer.py). Mistral's own native spans are dropped in
        # TokenPoliceSpanProcessor by instrumentation-scope name
        # ("mistralai_sdk_tracer") so nothing double-logs.
        ("opentelemetry.instrumentation.bedrock", "BedrockInstrumentor"),
        ("opentelemetry.instrumentation.langchain", "LangchainInstrumentor"),
        # NOTE: we intentionally do NOT register opentelemetry-instrumentation-
        # together here. The current release imports
        # `together.types.completions.CompletionResponse`, a symbol the
        # together 2.x package no longer exposes — instrumentation crashes at
        # import time. The TokenPolice enforcer wraps the together SDK on the
        # manual-telemetry path instead (see enforcer.py).
        # NOTE: we intentionally do NOT register opentelemetry-instrumentation-crewai
        # here. As of v0.60.0 it sets up its own TracerProvider during
        # CrewAIInstrumentor().instrument(), and OTel rejects ours with
        # "Overriding of current TracerProvider is not allowed" — meaning the
        # spans it emits ("crewai.workflow", "<agent>.agent", "<task>.task",
        # "<model>.llm") never reach TokenPoliceSpanProcessor. CrewAI's
        # LLM.call invokes litellm.completion(**params) directly, so the
        # existing LiteLLM enforcer wrapper already captures every CrewAI
        # LLM call with full pre-flight check + composition + token usage.
    ]

    import importlib

    for module_path, class_name in instrumentors:
        try:
            module = importlib.import_module(module_path)
            instrumentor_class = getattr(module, class_name)
            instrumentor = instrumentor_class()
            if not instrumentor.is_instrumented_by_opentelemetry:
                instrumentor.instrument()
            # opentelemetry-instrumentation-openai (>=0.50.x) wraps the
            # Responses API as well as Chat Completions. The Responses wrapper
            # crashes when openai-agents uses `.with_raw_response.create()`
            # (it tries to read `.id` on a raw AsyncAPIResponse) and otherwise
            # would double-log with our manual-mode Responses wrapper in
            # enforcer.py. Unwrap it here so our enforcer remains the single
            # source of truth for Responses telemetry.
            if class_name == "OpenAIInstrumentor":
                _unwrap_openai_responses_hooks(log_errors)
                # The instrumentor's ChatStream calls its stream accumulator
                # OUTSIDE the __next__/__anext__ try block with no @dont_throw
                # — malformed chunks from OpenAI-compatible endpoints (Gemini
                # etc.) crash the customer's `for chunk in stream:` loop.
                # Guard it so accumulator errors degrade telemetry, never the
                # customer stream (GOLDEN RULE).
                _patch_openai_otel_stream_accumulator(log_errors)
            # Anthropic instrumentor wraps create(stream=True) returns in
            # AnthropicAsyncStream (wrapt.ObjectProxy) which drops async
            # context-manager dunders — breaks `async with stream:`. Restore
            # them so customer/framework code (incl. pydantic_ai beta path)
            # never crashes under TokenPolice (GOLDEN RULE).
            if class_name == "AnthropicInstrumentor":
                _patch_anthropic_otel_stream_cm(log_errors)
        except ImportError:
            # The instrumentor isn't installed. For the providers we now ship in
            # core deps, warn ONCE — but only if the app actually uses that
            # provider (its SDK is importable). This avoids spurious warnings for
            # providers a given app never touches. Budgets are still enforced;
            # this only affects token capture. google_genai / langchain are
            # opt-in extras, so a missing instrumentor there is expected — skip.
            _maybe_warn_missing_instrumentor(module_path)
        except Exception as e:
            if log_errors:
                logger.warning(f"TokenPolice: failed to instrument {class_name}: {e}")


def _patch_anthropic_otel_stream_cm(log_errors: bool = False) -> None:
    """Restore context-manager protocol on OTel Anthropic stream proxies.

    Anthropic SDK ``Stream`` / ``AsyncStream`` implement the (a)sync CM
    protocol so documented usage works::

        with client.beta.messages.create(..., stream=True) as stream: ...
        async with await client.beta.messages.create(..., stream=True) as stream: ...

    ``opentelemetry-instrumentation-anthropic`` substitutes ``AnthropicStream``
    / ``AnthropicAsyncStream`` (``wrapt.ObjectProxy`` subclasses) for those
    return values. Python looks up dunders on the *type*, so ``__getattr__``
    forwarding cannot restore either protocol.

    Async pair: on wrapt < 2.0 ``ObjectProxy`` has no ``__aenter__`` at all,
    so ``async with`` raises::

        TypeError: 'AnthropicAsyncStream' object does not support the
        asynchronous context manager protocol

    On wrapt >= 2.0 ``ObjectProxy`` DOES define a delegating ``__aenter__``,
    so the unpatched symptom there is proxy-identity loss instead: ``async
    with`` binds the raw ``__wrapped__`` stream, bypassing the instrumented
    ``__anext__`` — the span never ends and the call is silently unmetered.
    This patch is still required (a class-own method overrides the inherited
    delegate).

    Sync pair: ``ObjectProxy`` has ALWAYS supplied delegating ``__enter__`` /
    ``__exit__`` (every wrapt version), and ``anthropic.Stream.__enter__``
    returns ``self`` — so ``with proxy as s`` binds the raw stream and the
    call is silently unmetered on every wrapt version. The sync override
    below is therefore unconditional by design (no wrapt-era split).

    One class-level patch per proxy class covers all three seams that return
    it: ``messages.create``, ``beta.messages.create``, and the Bedrock beta
    twin. The non-beta path is additionally shielded by TP's own
    ``_ModeAStreamProxy`` (the enforcer wraps *outside* OTel and defines its
    own CM methods), so the live victim of the sync gap was the beta seam,
    which only OTel wraps.

    Silent metering loss on customer-correct code is a GOLDEN RULE-adjacent
    defect — TP being installed must not change what a ``with`` binds. Patch
    the OTel classes (type-level methods) after instrumentor install: enter
    returns the proxy (keep metering), exit close-throughs to the underlying
    stream. Fail-open: any error here is logged and ignored so ``init`` never
    throws.
    """
    try:
        # Both classes are defined in this module for every supported
        # instrumentor version (floor 0.52.4).
        from opentelemetry.instrumentation.anthropic.streaming import (
            AnthropicAsyncStream,
            AnthropicStream,
        )
    except Exception as e:  # pragma: no cover - instrumentor not installed
        if log_errors:
            logger.debug(
                f"TokenPolice: skip anthropic stream CM patch (import): {e}"
            )
        return

    try:
        # Idempotent — re-init / double instrument must not stack wrappers.
        # The flag lives on each class separately (the classes are siblings,
        # not related by inheritance), so patching one never skips the other;
        # no early return here — the sync block below must still run.
        already_async = getattr(AnthropicAsyncStream, "_tp_cm_patched", False)
        # Upstream may someday ship the methods; only fill gaps.
        if not already_async and "__aenter__" not in AnthropicAsyncStream.__dict__:
            async def __aenter__(self):
                # Return the proxy so iteration still runs instrumented __anext__.
                return self

            AnthropicAsyncStream.__aenter__ = __aenter__  # type: ignore[attr-defined]

        if not already_async and "__aexit__" not in AnthropicAsyncStream.__dict__:
            async def __aexit__(self, exc_type, exc, tb):
                # OTel only ends the stream span inside __anext__ (on
                # StopAsyncIteration or an iteration error). A customer who
                # exits the `async with` early — break, return, or a body
                # exception — would leave the span un-ended and the call
                # unmetered. End it here first, mirroring the proxy's own two
                # paths; gated on its flag so a fully-consumed stream is a
                # no-op, and every failure is swallowed (fail-open).
                try:
                    if getattr(self, "_instrumentation_completed", True) is False:
                        if exc_type is None:
                            self._complete_instrumentation()
                        else:
                            from opentelemetry.trace.status import Status, StatusCode
                            span = getattr(self, "_span", None)
                            if span is not None and span.is_recording():
                                # A hostile exc __str__ (or a raising
                                # set_status) must not skip span.end() below —
                                # that would un-meter the call.
                                try:
                                    detail = str(exc)
                                except Exception:
                                    detail = getattr(exc_type, "__name__", "error")
                                try:
                                    span.set_status(Status(StatusCode.ERROR, detail))
                                except Exception:
                                    pass
                            if span is not None:
                                span.end()
                            self._instrumentation_completed = True
                except Exception:
                    pass
                # Mirror anthropic.AsyncStream.__aexit__ → await close().
                # Fail-open: close errors must not replace a provider exception.
                try:
                    close = getattr(self, "close", None)
                    if callable(close):
                        res = close()
                        if hasattr(res, "__await__"):
                            await res
                    else:
                        aclose = getattr(self, "aclose", None)
                        if callable(aclose):
                            res = aclose()
                            if hasattr(res, "__await__"):
                                await res
                except Exception:
                    pass
                return False

            AnthropicAsyncStream.__aexit__ = __aexit__  # type: ignore[attr-defined]

        AnthropicAsyncStream._tp_cm_patched = True  # type: ignore[attr-defined]
    except Exception as e:
        if log_errors:
            logger.debug(
                f"TokenPolice: failed to patch AnthropicAsyncStream CM: {e}"
            )

    try:
        already_sync = getattr(AnthropicStream, "_tp_cm_patched", False)
        # Unlike the async pair, this override is UNCONDITIONAL by design and
        # gated on the class's OWN __dict__, never hasattr: wrapt.ObjectProxy
        # has always supplied inherited delegating __enter__/__exit__
        # (self.__wrapped__.__enter__() returns the RAW anthropic Stream), so
        # hasattr is always True and would never install. The own-dict check
        # still lets a future upstream OTel release that ships its own
        # __enter__ win.
        if not already_sync and "__enter__" not in AnthropicStream.__dict__:
            def __enter__(self):
                # Return the proxy so iteration still runs instrumented __next__.
                return self

            AnthropicStream.__enter__ = __enter__  # type: ignore[attr-defined]

        if not already_sync and "__exit__" not in AnthropicStream.__dict__:
            def __exit__(self, exc_type, exc, tb):
                # Mirror of __aexit__ above: OTel only ends the stream span
                # inside __next__, so an early exit — break, return, or a body
                # exception — would leave the span un-ended and the call
                # unmetered. End it here first; gated on the proxy's flag so a
                # fully-consumed stream is a no-op, every failure swallowed
                # (fail-open).
                try:
                    if getattr(self, "_instrumentation_completed", True) is False:
                        if exc_type is None:
                            self._complete_instrumentation()
                        else:
                            from opentelemetry.trace.status import Status, StatusCode
                            span = getattr(self, "_span", None)
                            if span is not None and span.is_recording():
                                # A hostile exc __str__ (or a raising
                                # set_status) must not skip span.end() below —
                                # that would un-meter the call.
                                try:
                                    detail = str(exc)
                                except Exception:
                                    detail = getattr(exc_type, "__name__", "error")
                                try:
                                    span.set_status(Status(StatusCode.ERROR, detail))
                                except Exception:
                                    pass
                            if span is not None:
                                span.end()
                            self._instrumentation_completed = True
                except Exception:
                    pass
                # Mirror anthropic.Stream.__exit__ → close(). Fail-open: close
                # errors must not replace a provider/body exception.
                try:
                    close = getattr(self, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
                # Never suppress the customer's exception.
                return False

            AnthropicStream.__exit__ = __exit__  # type: ignore[attr-defined]

        AnthropicStream._tp_cm_patched = True  # type: ignore[attr-defined]
    except Exception as e:
        if log_errors:
            logger.debug(
                f"TokenPolice: failed to patch AnthropicStream CM: {e}"
            )


def _unwrap_openai_responses_hooks(log_errors: bool = False) -> None:
    """Undo opentelemetry-instrumentation-openai's wrap of the Responses API.

    The Chat Completions wrappers remain in place — only the Responses ones
    (create / retrieve / cancel on both sync and async clients) are removed.

    NOTE: opentelemetry.instrumentation.utils.unwrap does not support
    "Class.method" attr syntax — it does a single getattr(module, attr) and
    treats the result as the wrapped function. The Responses wrappers were
    installed via wrapt.wrap_function_wrapper(module, "Class.method", ...),
    which resolves the dotted path correctly. We mirror that resolution here:
    walk module → class → method, and replace the wrapped method with its
    __wrapped__ attribute. Removing the OpenLLMetry wrapper avoids two
    problems: (1) it crashes on openai-agents' streaming path which uses
    `.with_raw_response.create()`, and (2) without removal it would double-log
    with our manual-mode enforcer wrapper.
    """
    try:
        import importlib
        # wrap_function_wrapper installs wrapt.BoundFunctionWrapper instances
        # (NOT plain wrapt.ObjectProxy). Walk the __wrapped__ chain and strip
        # only the wrapt wrappers — leave the plain functions (which carry
        # functools.wraps-set __wrapped__ pointing to the source function) alone.
        try:
            from wrapt import FunctionWrapper, BoundFunctionWrapper
            _WrappedTypes = (FunctionWrapper, BoundFunctionWrapper)
        except Exception:  # pragma: no cover
            _WrappedTypes = ()  # type: ignore[assignment]
        module = importlib.import_module("openai.resources.responses")
        for class_name, method_name in (
            ("Responses", "create"),
            ("Responses", "retrieve"),
            ("Responses", "cancel"),
            ("AsyncResponses", "create"),
            ("AsyncResponses", "retrieve"),
            ("AsyncResponses", "cancel"),
        ):
            cls = getattr(module, class_name, None)
            if cls is None:
                continue
            current = cls.__dict__.get(method_name)
            if current is None:
                continue
            # Strip leading wrapt wrappers off the class-level attribute.
            while _WrappedTypes and isinstance(current, _WrappedTypes):
                inner = getattr(current, "__wrapped__", None)
                if inner is None or inner is current:
                    break
                current = inner
            cls_attr = cls.__dict__.get(method_name)
            if current is not cls_attr:
                setattr(cls, method_name, current)
    except Exception as e:
        if log_errors:
            logger.debug(
                f"TokenPolice: failed to unwrap OpenLLMetry Responses hooks: {e}"
            )


# Warn-once flag for the guarded OpenAI stream accumulator below: the first
# swallowed accumulator failure logs a warning, later ones stay silent so a
# long-running stream of malformed chunks can't spam the customer's logs.
_openai_stream_accum_warned = False


def _tp_accumulated_tool_call_count(complete_response, choice_index):
    """How many tool calls are already accumulated for `choice_index`.

    Seeds the per-chunk synthetic tool-call index counter (see
    `_tp_normalize_stream_item`). The accumulator stores choices positionally
    (`complete_response["choices"][index]`), so position == choice index for
    the well-formed streams we normalize toward. Any unexpected shape → 0.
    """
    try:
        choices = complete_response.get("choices")
        if isinstance(choices, list) and 0 <= choice_index < len(choices):
            message = choices[choice_index].get("message")
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    return len(tool_calls)
    except Exception:
        pass
    return 0


def _tp_normalize_stream_item(item, complete_response):
    """In-place repair of a malformed OpenAI-compatible stream chunk dict.

    `item` here is always the TELEMETRY-SIDE dict (fresh `model_as_dict`
    output or our own deepcopy — see the shim), never the customer's chunk
    object, so in-place mutation is safe. Repairs exactly the shapes known to
    crash the instrumentor's `_accumulate_stream_items`:

      * `choices` absent / None / non-list → replaced with `[]` (some compat
        gateways emit a bare usage-only terminal frame with no choices key);
        non-dict entries inside a choices list are dropped.
      * choice `index` absent / None / non-int / negative → positional index.
      * tool-call delta `index` absent / None / non-int / negative → synthetic
        index: an id-bearing delta (or the first ever for the choice) opens
        the next slot; an id-less delta is an argument-fragment continuation
        of the last opened slot; a delta with a valid int index re-syncs the
        counter past it. Gemini-compatible endpoints omit tool-call `index`
        entirely, so without this every distinct tool call would collapse
        into slot 0 (>=0.54.0) or crash outright (<0.54.0).

    The synthetic-index counter is recomputed per chunk from the tool calls
    already accumulated in `complete_response` — no state is stored anywhere
    (not on `complete_response`, not module-level), so nothing can leak into
    span attributes or outlive the stream.
    """
    choices = item.get("choices")
    if not isinstance(choices, list):
        item["choices"] = []
        return
    if any(not isinstance(c, dict) for c in choices):
        choices = [c for c in choices if isinstance(c, dict)]
        item["choices"] = choices
    for position, choice in enumerate(choices):
        index = choice.get("index")
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            choice["index"] = position
            index = position
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        tool_calls = delta.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        counter = _tp_accumulated_tool_call_count(complete_response, index)
        assigned_any = counter > 0
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tc_index = tool_call.get("index")
            if (
                isinstance(tc_index, int)
                and not isinstance(tc_index, bool)
                and tc_index >= 0
            ):
                counter = max(counter, tc_index + 1)
                assigned_any = True
                continue
            if tool_call.get("id") or not assigned_any:
                tool_call["index"] = counter
                counter += 1
                assigned_any = True
            else:
                tool_call["index"] = counter - 1


def _patch_openai_otel_stream_accumulator(log_errors: bool = False) -> None:
    """Guard opentelemetry-instrumentation-openai's streaming accumulator.

    The instrumentor's ``ChatStream.__next__`` / ``__anext__`` call
    ``self._process_item(chunk)`` OUTSIDE their try block, and
    ``_process_item`` → module-level ``_accumulate_stream_items`` carries no
    ``@dont_throw`` (unlike ``_process_complete_response`` and
    ``_build_from_streaming_response``). Any exception raised while
    accumulating a chunk therefore escapes straight into the customer's
    ``for chunk in stream:`` loop. That is a GOLDEN RULE violation —
    customer-correct code must not break because TP is installed. Live crash
    shapes from OpenAI-compatible endpoints (Gemini and other gateways):

      * choice ``{"index": None}`` → ``len(choices) <= None`` TypeError
        (all instrumentor versions incl. 0.61.0+);
      * chunk with ``choices`` absent or None → ``for choice in None``
        TypeError (bare usage-only terminal frames);
      * tool-call delta ``{"index": None}`` → ``int(None)`` TypeError
        (<0.54.0 — also closed by the pyproject floor bump).

    Fix: replace the module attribute with a shim that (a) pre-normalizes the
    telemetry-side dict (`_tp_normalize_stream_item`) so well-formed telemetry
    still accumulates, and (b) calls the original inside try/except Exception,
    swallowing with a once-per-process warning. BaseException (KeyboardInterrupt
    etc.) is never swallowed. Chunk `usage` lands in ``complete_response``
    before the crashing choices loop, so even a swallowed failure keeps that
    chunk's usage — usage handling is deliberately untouched.

    ``_process_item`` looks up ``_accumulate_stream_items`` as a module global
    at call time (verified in 0.61.0), so patching the module attribute
    intercepts every caller. Fail-open install; idempotent via a marker on the
    shim; the original is preserved on ``__wrapped__``.
    """
    try:
        from opentelemetry.instrumentation.openai.shared import chat_wrappers
    except Exception as e:  # pragma: no cover - instrumentor not installed
        if log_errors:
            logger.debug(
                f"TokenPolice: skip openai stream accumulator patch (import): {e}"
            )
        return

    try:
        original = getattr(chat_wrappers, "_accumulate_stream_items", None)
        if original is None or not callable(original):
            if log_errors:
                logger.debug(
                    "TokenPolice: skip openai stream accumulator patch "
                    "(no _accumulate_stream_items)"
                )
            return
        # Idempotent — re-init / double instrument must not stack shims.
        if getattr(original, "_tp_stream_guard", False):
            return

        model_as_dict = getattr(chat_wrappers, "model_as_dict", None)

        def _tp_accumulate_stream_items(item, complete_response):
            global _openai_stream_accum_warned
            try:
                # Normalize only the telemetry-side dict, never the customer's
                # chunk. model_as_dict on a pydantic chunk returns a fresh
                # dict the customer never sees; if the chunk already IS a dict
                # (dict-yielding compat SDKs / openai v0) model_as_dict passes
                # the SAME object through — deepcopy in that case so we never
                # mutate a customer-visible object. The original re-runs
                # model_as_dict on our dict, which is a pass-through.
                if isinstance(item, dict):
                    item = copy.deepcopy(item)
                elif callable(model_as_dict):
                    converted = model_as_dict(item)
                    if isinstance(converted, dict):
                        item = converted
                if isinstance(item, dict):
                    _tp_normalize_stream_item(item, complete_response)
            except Exception:
                # Normalization must itself be bulletproof: fall through and
                # let the original see the item as-is; the guard below still
                # protects the customer stream.
                pass
            try:
                return original(item, complete_response)
            except Exception:
                if not _openai_stream_accum_warned:
                    _openai_stream_accum_warned = True
                    logger.warning(
                        "TokenPolice: the OpenAI instrumentor's stream "
                        "accumulator failed on a chunk; telemetry for this "
                        "stream may be degraded but the customer stream is "
                        "unaffected."
                    )
                return None

        _tp_accumulate_stream_items._tp_stream_guard = True  # type: ignore[attr-defined]
        _tp_accumulate_stream_items.__wrapped__ = original  # type: ignore[attr-defined]
        chat_wrappers._accumulate_stream_items = _tp_accumulate_stream_items
    except Exception as e:
        if log_errors:
            logger.debug(
                f"TokenPolice: failed to patch openai stream accumulator: {e}"
            )


def unsetup_opentelemetry():
    """Cleanup OpenTelemetry provider. Called by uninstrument().

    Holds and shuts down ONLY the TracerProvider object WE built and globally
    set (_tp_created_provider) — never a customer-owned (piggybacked or
    injected) provider, since shutting that down under them would kill their
    exporter.
    """
    global _tp_created_provider
    try:
        own = _tp_created_provider
        if own is not None and hasattr(own, 'shutdown'):
            own.shutdown()
    except Exception:
        pass
    finally:
        _tp_created_provider = None
