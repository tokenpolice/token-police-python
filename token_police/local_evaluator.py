"""
Pure-function local enforcement evaluator.

Mirrors the server's decision semantics (condition matching, group-by tag
generation, three-pass evaluation order) so daemon-mode SDKs can decide
ALLOW/BLOCK/REROUTE locally in ~10 µs without a network round-trip. When
the local decision says BLOCK or REROUTE the enforcer still verifies with a
synchronous inline /check before acting — local is fast, the server stays
authoritative.

Inputs:
    pack — current Decision Pack snapshot
    session — TPSession (user_id, paid_plan, workflow_name, metadata)
    observed — {model, provider, trace_id} from the in-flight LLM call

Returns:
    {
      "decision": {"status": "allowed"|"blocked"|"rerouted",
                       "rule_id": str|None, "mode": "enforce"|"dry_run",
                       "reroute": {"from":..., "to":...}|None},
      "observations": [{"rule_id", "outcome", "mode", "rejection_reason"?, "reroute"?,
                        "rule_name"?}, ...],
    }

Allow-path is hot — keep it cheap. Observations only allocate when dry-run
rules or rejected reroutes fire, which is the cold path.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .composition import _canon_expand_decimal
from .reroute_noop import is_noop_reroute


# ── Server-parity condition matcher & group-by tag generator ───
def _payload_field(payload: Dict[str, Any], field: str) -> Any:
    if not field:
        return None
    # Dotted paths (e.g. "intent.kind", "metadata.source") for modality-aware
    # rules. Falls back to flat-key + metadata bag to keep legacy rules valid.
    if "." in field:
        cur: Any = payload
        for part in field.split("."):
            if cur is None or not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(part)
        if cur is not None:
            return cur
        md = payload.get("metadata")
        if isinstance(md, dict) and field in md:
            return md[field]
        return None
    if field in payload:
        return payload[field]
    md = payload.get("metadata")
    if isinstance(md, dict) and field in md:
        return md[field]
    return None


def _js_string(v: Any) -> str:
    # JS String() of a JSON scalar, so CONTAINS coerces exactly like the
    # server-side evaluator. bool -> "true"/"false" (NOT Python "True"/"False");
    # non-finite floats -> "Infinity"/"-Infinity"/"NaN"; finite floats reuse the
    # shortest-round-trip decimal expander (JS number text: 100.0 -> "100",
    # 1e-05 -> "0.00001"). Ints/strings and any other object fall through to
    # str(); dict/list here would give Python's repr rather than JS
    # "[object Object]"/comma-join — an accepted residual, since rule values are
    # authored against scalar payload fields. None never reaches here (callers gate).
    if type(v) is bool:
        return "true" if v else "false"
    if isinstance(v, float):
        if v != v:
            return "NaN"
        if v == float("inf"):
            return "Infinity"
        if v == float("-inf"):
            return "-Infinity"
        return _canon_expand_decimal(repr(v))
    return str(v)


def _js_strict_eq(a: Any, b: Any) -> bool:
    # JS === over JSON-shaped values, so EQ/NEQ/IN compare exactly like the
    # server-side evaluator. A bool is its own type (never === a number, so
    # True !== 1); numbers share JS's single number type (1 === 1.0); strings
    # compare by value; null === null. Objects/arrays compare by reference in JS,
    # so any other shape (dict/list/bytes/Decimal/custom) is never equal to a
    # deserialized rule value -> False. Total: never raises for any input.
    a_bool = type(a) is bool
    b_bool = type(b) is bool
    if a_bool or b_bool:
        return a_bool and b_bool and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, str) and isinstance(b, str):
        return a == b
    if a is None and b is None:
        return True
    return False


def _matches_leaf(payload: Dict[str, Any], cond: Optional[Dict[str, Any]]) -> bool:
    # One leaf condition. No field => matches everything. Full operator set;
    # unknown operator fails closed.
    if not cond:
        return True
    op = cond.get("operator") or cond.get("op")
    field = cond.get("field")
    value = cond.get("value")
    if not field:
        return True
    v = _payload_field(payload, field)
    if op == "EQ":
        return _js_strict_eq(v, value)
    if op == "NEQ":
        return not _js_strict_eq(v, value)
    # EXISTS treats "" as absent to match the SQL side (col != '').
    if op == "EXISTS":
        return v is not None and v != ""
    if op == "CONTAINS":
        return v is not None and (_js_string(value) if value is not None else "") in _js_string(v)
    if op == "IN":
        return isinstance(value, (list, tuple)) and any(_js_strict_eq(v, item) for item in value)
    return False


def matches_condition(payload: Dict[str, Any], match: Optional[Dict[str, Any]]) -> bool:
    # Selector: empty => match all; composite {combinator, conditions} => AND/OR
    # (empty AND True, empty OR False); else a bare leaf. Byte-compatible with the
    # server-side matcher.
    if not match:
        return True
    conds = match.get("conditions")
    if isinstance(conds, list):
        combinator = str(match.get("combinator") or "AND").upper()
        if combinator == "OR":
            return any(_matches_leaf(payload, c) for c in conds)
        return all(_matches_leaf(payload, c) for c in conds)
    return _matches_leaf(payload, match)


def generate_group_by_tag(payload: Dict[str, Any], group_by) -> str:
    if not group_by:
        return "global"
    parts: List[str] = []
    for f in group_by:
        v = _payload_field(payload, f)
        # Match the server's `val || 'unknown'` then String()-join semantics.
        # Half 1 (DROP): emulate JS truthiness so the JS-falsy set
        # (None/False/""/0/-0.0/NaN) collapses to the "unknown" sentinel — NOT
        # bare `if not v`, which would wrongly drop the TRUTHY empty
        # collections []/{}. Half 2 (KEEP): JS-faithful String() of the survivor —
        # kept True → "true" (not "True"), kept float → JS number text via the
        # canonical decimal expander (1.0 → "1"); everything else (str/int) →
        # str(v) unchanged.
        if (
            v is None
            or v is False
            or (isinstance(v, str) and v == "")
            or (isinstance(v, (int, float)) and (v == 0 or v != v))
        ):
            parts.append("unknown")
        elif v is True:
            parts.append("true")
        elif isinstance(v, float):
            parts.append(_canon_expand_decimal(repr(v)))
        else:
            parts.append(str(v))
    return "_".join(parts)


# ── Effective provider — pure trim + lowercase of the call's provider ──
# Alias slugs a price source or an SDK instrumentor may emit, mapped onto the
# canonical serving-provider slug the runtime identity layer produces. Kept
# inline (production code never reads the shared test-only contract JSON); mirror
# the published provider-slug `aliases` table when it changes so both sides of a
# provider comparison canonicalize identically.
_PROVIDER_ALIASES = {
    "together_ai": "together",
    "togetherai": "together",
    "fireworks_ai": "fireworks",
    "vertex_ai": "vertex-ai",
    "vertex_ai-language-models": "vertex-ai",
    "google_vertexai": "vertex-ai",
    "azure": "azure-openai",
    "azure_text": "azure-openai",
    "azure_ai": "azure-ai",
    "gemini": "google",
    "google_genai": "google",
    "x-ai": "xai",
    "grok": "xai",
    "moonshotai": "moonshot",
    "minimaxi": "minimax",
    "amazon": "bedrock",
    "aws": "bedrock",
    "cohere_chat": "cohere",
    "huggingface_free_tier": "huggingface",
    "hf": "huggingface",
    "openrouter_byok": "openrouter",
    "zhipuai": "zhipu",
    "zai": "zhipu",
}

# Runtime-only provider aliases (not published provider slugs).
# Keep outside _PROVIDER_ALIASES so the published provider-slug table parity stays intact.
_RUNTIME_ONLY_PROVIDER_ALIASES = {
    # SDK Responses-API pseudo-provider: parse/shape only; serving identity is openai.
    "openai_responses": "openai",
}


def effective_provider(provider: str) -> str:
    slug = (provider or "").lower().strip()
    # Canonicalize alias slugs so equivalent providers (e.g. together_ai vs
    # together) compare equal regardless of which side named the alias.
    return _PROVIDER_ALIASES.get(
        slug, _RUNTIME_ONLY_PROVIDER_ALIASES.get(slug, slug)
    )


# ── Helpers ──────────────────────────────────────────────────────────
def _build_payload(session, observed: Dict[str, Any]) -> Dict[str, Any]:
    """Project the session + observed call into the dict shape match
    conditions and groupBy expect."""
    md = {}
    try:
        if session is not None and getattr(session, "metadata", None):
            md = dict(session.metadata or {})
    except Exception:
        md = {}
    user_id = getattr(session, "user_id", None) if session else None
    paid_plan = getattr(session, "paid_plan", None) if session else None
    session_id = getattr(session, "session_id", None) if session else None
    if paid_plan:
        md.setdefault("paid_plan", paid_plan)
    intent = observed.get("intent") if isinstance(observed, dict) else None
    modality = observed.get("modality") if isinstance(observed, dict) else None
    if modality and isinstance(md, dict):
        md.setdefault("modality", modality)
    # Resolved modality (the emitted "modality" field value — unchanged expression,
    # just extracted to a named local so "operation" reuses the identical value).
    resolved_modality = modality or (intent.get("kind") if isinstance(intent, dict) else None) or "chat"
    # Field order mirrors the server's evaluation payload field-for-field:
    # "end_user_id" and "metadata" sit BEFORE the **md spread (metadata CAN
    # override them); every other canonical field sits AFTER the spread
    # (canonical wins). This keeps the warm-local path resolving reserved keys
    # to the same values the inline /check path does. "operation" equals the
    # resolved modality and sits after the spread between "modality" and
    # "intent", so a metadata-bag "operation" key can't override it.
    payload = {
        "end_user_id": user_id or "anonymous",
        "metadata": md,
        **md,
        "paid_plan": paid_plan or "free",
        "user_id": user_id or "anonymous",
        "model": observed.get("model") or "",
        # Serving-provider identity (not the internal parse pseudo-provider). Aligns
        # groupBy/match tags with server-side entity tag arming on /log.
        "provider": effective_provider(str(observed.get("provider") or "")),
        "trace_id": observed.get("trace_id") or "",
        # session_id lets the local fast path compute a per-session group tag and
        # consult the directive's entities set, so allowed sessions still skip /check.
        "session_id": session_id or None,
        "modality": resolved_modality,
        "operation": resolved_modality,
        "intent": intent or {},
    }
    # Not a matchable selector field — SDK-only signal for the REROUTE
    # guard when the client base_url host is unrecognized.
    if isinstance(observed, dict) and observed.get("serving_unverified") is True:
        payload["serving_unverified"] = True
    return payload


def _allowed() -> Dict[str, Any]:
    return {"status": "allowed", "rule_id": None, "mode": "enforce", "reroute": None}


# ── Precomputed per-pack structures (built once at pack-apply time) ────
# evaluate() runs on EVERY LLM call, but the loop-block set, the sorted
# directive list, and each directive's entity set are invariant for a given
# pack — rebuilding them per call is wasted O(directives x entities) work on the
# hot allow path. state.py builds these ONCE when a pack is applied (snapshot or
# delta) and stashes them on the pack under _DERIVED_KEY; evaluate() reuses them
# when present and rebuilds them on the fly — byte-identically — when absent
# (e.g. packs injected straight into evaluate() by tests, or any path that
# bypasses the precompute). The two paths MUST produce identical decisions.
_DERIVED_KEY = "__tp_derived__"

# Per-directive sentinel: this directive's `entities` couldn't be pre-frozen (a
# malformed, non-iterable value), so evaluate() reproduces the original inline
# set(...) for that one directive — preserving the exact throw timing of the
# build-on-the-fly path rather than failing the whole precompute.
_ENTITY_INLINE = object()


def _directive_sort_key(d: Dict[str, Any]):
    # CANONICAL total order — MUST stay byte-identical between the precomputed
    # and on-the-fly paths (and with the server-side/Node siblings): priority
    # ascending (missing/non-numeric -> 100), then id ascending by code point.
    # `d.get("priority", 100)` is read once here; evaluating it twice (as the
    # original inline lambda did) yields the same value since d is not mutated.
    p = d.get("priority", 100)
    return (p if isinstance(p, (int, float)) else 100, d.get("id", ""))


def _sorted_directives(raw_directives: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_directives, list):
        return []
    # Filter to dict directives only — defensive against malformed packs.
    return sorted([d for d in raw_directives if isinstance(d, dict)], key=_directive_sort_key)


def _entity_set_for(d: Dict[str, Any]):
    """Pre-freeze one directive's entity set, mirroring EXACTLY the expression
    the matching evaluate() pass uses. Returns a frozenset, None (a REROUTE with
    no `entities` key => ungated, or a non-entity kind), or the _ENTITY_INLINE
    sentinel when the value can't be frozen (evaluate() then reproduces the
    original inline behavior — including its throw timing — for that directive).
    """
    kind = d.get("kind")
    if kind == "ENTITY_BLOCK":
        # Mirrors ENTITY_BLOCK's `set(d.get("entities") or [])`.
        try:
            return frozenset(d.get("entities") or [])
        except Exception:
            return _ENTITY_INLINE
    if kind == "REROUTE":
        # Mirrors REROUTE's `entities = d.get("entities"); ... set(entities)` —
        # None means "no `entities` key" (reroute is not entity-gated).
        raw = d.get("entities")
        if raw is None:
            return None
        try:
            return frozenset(raw)
        except Exception:
            return _ENTITY_INLINE
    return None


def _build_derived_raw(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Build the precomputed structures. NOT guarded — used by evaluate()'s
    on-the-fly path so any failure (e.g. an unhashable loop-block id, an
    uncomparable directive id) propagates to evaluate()'s outer try/except and
    degrades to `allowed`, exactly as the original inline code did."""
    raw_loop = pack.get("loop_blocks")
    loop_set = frozenset(raw_loop) if isinstance(raw_loop, list) else frozenset()
    directives = _sorted_directives(pack.get("directives"))
    entity_sets = [_entity_set_for(d) for d in directives]
    return {"loop_set": loop_set, "directives": directives, "entity_sets": entity_sets}


def build_derived(pack: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """GUARDED wrapper for pack-apply time (state.py). Pure + throw-free:
    returns None on any failure so a precompute error can never poison pack
    apply — the pack is left un-derived and evaluate() rebuilds on the fly
    (identical results). Called on EVERY pack swap so the derived structures are
    always rebuilt atomically with the pack they describe; a stale set can never
    survive a swap."""
    try:
        if not isinstance(pack, dict):
            return None
        return _build_derived_raw(pack)
    except Exception:
        return None


def _derived_is_usable(derived: Any) -> bool:
    """Shape guard: only trust a precomputed structure that is internally
    consistent (parallel directives/entity_sets, a real loop set)."""
    return (
        isinstance(derived, dict)
        and isinstance(derived.get("loop_set"), (set, frozenset))
        and isinstance(derived.get("directives"), list)
        and isinstance(derived.get("entity_sets"), list)
        and len(derived["directives"]) == len(derived["entity_sets"])
    )


def _rule_name_of(d: Dict[str, Any]) -> Dict[str, str]:
    """``{"rule_name": ...}`` when the directive carries a name, else ``{}``.

    Merged into an observation so the audit/routing feeds show a label instead
    of the rule UUID. Omit-when-absent keeps the payload shape unchanged for
    unnamed directives. Mirrors ``ruleNameOf`` in the Node SDK.
    """
    name = d.get("name")
    return {"rule_name": name} if isinstance(name, str) and name else {}


def evaluate(pack: Optional[Dict[str, Any]], session, observed: Dict[str, Any],
             force_shadow: bool = False) -> Dict[str, Any]:
    """Main entry. Pack=None means cache miss → caller falls back to /check.

    `force_shadow` (legacy dry-run shortcut): treat every directive's effective
    mode as "dry_run" so the call is observed but never blocked/rerouted. The
    enforcer now passes False here and suppresses the action itself, so the
    real decision still reaches /check; kept for any direct callers.

    Wrapped in a broad try/except: any internal error in evaluation returns
    `allowed` so the customer's call proceeds. The enforcer will hit the
    inline /check path next time the SDK is consulted, which is the
    authoritative source anyway.
    """
    try:
        return _evaluate_inner(pack, session, observed, force_shadow)
    except Exception:
        return {"decision": _allowed(), "observations": []}


def _evaluate_inner(pack: Optional[Dict[str, Any]], session, observed: Dict[str, Any],
                    force_shadow: bool = False) -> Dict[str, Any]:
    observations: List[Dict[str, Any]] = []
    if not pack or not isinstance(pack, dict):
        return {"decision": _allowed(), "observations": observations}

    # Reuse the structures precomputed at pack-apply time when present; otherwise
    # rebuild them on the fly with byte-identical results. The step order below
    # mirrors the ORIGINAL evaluator exactly — loop-block set, loop check, then
    # the directive sort — so that on the fallback path any (pathological) sort
    # failure degrades to `allowed` at the same point the inline code did and can
    # never mask a loop block. On the fallback path the frozenset/sort builders
    # are UNGUARDED: a failure propagates to evaluate()'s outer try/except and
    # degrades to `allowed`, exactly as before.
    derived = pack.get(_DERIVED_KEY)
    usable = _derived_is_usable(derived)

    # Loop-blocked traces short-circuit — computed BEFORE the directive sort.
    if usable:
        loop_set = derived["loop_set"]
    else:
        raw_loop = pack.get("loop_blocks")
        loop_set = frozenset(raw_loop) if isinstance(raw_loop, list) else frozenset()
    trace_id = (observed or {}).get("trace_id") or ""
    if trace_id and trace_id in loop_set:
        return {
            "decision": {"status": "blocked", "rule_id": None, "mode": "enforce",
                         "reroute": None, "reason": "loop_detected"},
            "observations": observations,
        }

    payload = _build_payload(session, observed or {})

    if usable:
        directives = derived["directives"]
        entity_sets = derived["entity_sets"]
    else:
        directives = _sorted_directives(pack.get("directives"))
        entity_sets = [_entity_set_for(d) for d in directives]

    obs_provider = effective_provider(payload.get("provider", ""))

    # Provider the REROUTE cross-provider gate compares a target against.
    # Normally the call's own provider; on the LiteLLM seam that is the
    # FRAMEWORK slug ("litellm"), which no vendor target could ever equal, so
    # the wrapper hands us the vendor its router resolved and we gate on that.
    # Everything else — observations, groupBy, match — keeps obs_provider, so
    # /check and /log still build identical tags.
    reroute_provider_gate = obs_provider
    try:
        _route_vendor = (observed or {}).get("route_vendor")
        if isinstance(_route_vendor, str) and _route_vendor.strip():
            reroute_provider_gate = effective_provider(_route_vendor)
    except Exception:
        reroute_provider_gate = obs_provider

    # Metadata, never part of the decision: set when an entity-gated directive's
    # selector matched but the call's group tag was absent from its streamed
    # `entities` set — i.e. an `entity_blocked`/`entity_rerouted` delta the SDK
    # never received would have flipped the result. Read only by the enforcer's
    # stale-stream gate; parity with the Node evaluator's `armableMiss`.
    # Only directives that would ENFORCE if armed count: `_would_enforce` is the
    # same `force_shadow or mode == "dry_run"` predicate each pass uses to decide
    # observe-vs-act, so a dry_run entity directive (which /check refuses to
    # block on anyway) can never trigger a re-verify round-trip.
    armable_miss = False

    def _would_enforce(d: Dict[str, Any]) -> bool:
        return not force_shadow and d.get("mode") != "dry_run"

    # PASS 0: UNCONDITIONAL_BLOCK
    for d in directives:
        if d.get("kind") != "UNCONDITIONAL_BLOCK":
            continue
        sel = d.get("selector") or {}
        if not matches_condition(payload, sel.get("match")):
            continue
        if force_shadow or d.get("mode") == "dry_run":
            observations.append({"rule_id": d.get("id"), "outcome": "would_block", "mode": "dry_run", **_rule_name_of(d)})
            continue
        return {
            "decision": {"status": "blocked", "rule_id": d.get("id"), "mode": "enforce", "reroute": None},
            "observations": observations,
        }

    # PASS 1: ENTITY_BLOCK
    for i, d in enumerate(directives):
        if d.get("kind") != "ENTITY_BLOCK":
            continue
        sel = d.get("selector") or {}
        if not matches_condition(payload, sel.get("match")):
            continue
        tag = generate_group_by_tag(payload, sel.get("group_by") or [])
        # Precomputed entity frozenset (or _ENTITY_INLINE when it couldn't be
        # frozen); membership test is identical to `set(d.get("entities") or [])`.
        entities = entity_sets[i]
        if entities is _ENTITY_INLINE:
            entities = set(d.get("entities") or [])
        if tag not in entities:
            if _would_enforce(d):
                armable_miss = True
            continue
        if force_shadow or d.get("mode") == "dry_run":
            observations.append({"rule_id": d.get("id"), "outcome": "would_block", "mode": "dry_run", **_rule_name_of(d)})
            continue
        return {
            "decision": {"status": "blocked", "rule_id": d.get("id"), "mode": "enforce", "reroute": None},
            "observations": observations,
        }

    # PASS 2: REROUTE
    for i, d in enumerate(directives):
        if d.get("kind") != "REROUTE":
            continue
        sel = d.get("selector") or {}
        if not matches_condition(payload, sel.get("match")):
            continue
        reroute = d.get("reroute") or {}
        target = reroute.get("to") or {}
        if not target.get("provider") or not target.get("model"):
            continue
        # Entity-armed reroute: skip if the call's groupTag isn't on the list.
        # Precomputed: None => no `entities` key (ungated), a frozenset => gated,
        # _ENTITY_INLINE => reproduce the original inline `set(...)` for this one
        # directive (preserving its throw timing on a malformed value).
        entities = entity_sets[i]
        if entities is _ENTITY_INLINE:
            raw = d.get("entities")
            if raw is not None:
                tag = generate_group_by_tag(payload, sel.get("group_by") or [])
                if tag not in set(raw):
                    if _would_enforce(d):
                        armable_miss = True
                    continue
        elif entities is not None:
            tag = generate_group_by_tag(payload, sel.get("group_by") or [])
            if tag not in entities:
                if _would_enforce(d):
                    armable_miss = True
                continue
        # Cross-provider reroute is rejected (same-provider only) — leave the original call alone
        # but emit an observation so the server audits it. Normalize the
        # directive's target provider through the same helper as obs_provider so
        # alias slugs (e.g. together_ai vs together) compare equal.
        # Serving_unverified (unrecognized custom base_url) also refuses —
        # we cannot prove the target model is servable; provider field still holds
        # the module slug for matchConditions/groupBy (check/log mirror).
        # serving_unverified outranks cross_provider_unsupported, mirroring
        # _apply_reroute's precedence — State A/B selection is invisible to the
        # operator, so both must report the same reason for the same refusal.
        if (
            payload.get("serving_unverified") is True
            or effective_provider(target["provider"]) != reroute_provider_gate
        ):
            observations.append({
                "rule_id": d.get("id"),
                "outcome": "reroute_rejected",
                "mode": d.get("mode", "enforce"),
                "rejection_reason": (
                    "serving_unverified"
                    if payload.get("serving_unverified") is True
                    else "cross_provider_unsupported"
                ),
                "reroute": {
                    "from": {"provider": obs_provider, "model": payload.get("model", "")},
                    "to": target,
                },
                **_rule_name_of(d),
            })
            continue
        # The directive's target resolves to the model already requested
        # (alias rule vs a pinned dated snapshot of the same model). Not a
        # reroute at all — skip silently so analytics see no phantom event.
        payload_model = payload.get("model")
        if (
            isinstance(payload_model, str)
            and payload_model
            and is_noop_reroute(payload_model, target["model"])
        ):
            continue
        if force_shadow or d.get("mode") == "dry_run":
            observations.append({
                "rule_id": d.get("id"),
                "outcome": "would_reroute",
                "mode": "dry_run",
                "reroute": {"from": {"provider": obs_provider, "model": payload.get("model", "")}, "to": target},
                **_rule_name_of(d),
            })
            continue
        # Thread directive `name` into the decision so the synthetic
        # /check-shaped reroute can stamp `_tp_routing.rule_name`.
        decision: Dict[str, Any] = {
            "status": "rerouted",
            "rule_id": d.get("id"),
            "mode": "enforce",
            "reroute": {"from": {"provider": obs_provider, "model": payload.get("model", "")}, "to": target},
        }
        d_name = d.get("name")
        if isinstance(d_name, str) and d_name:
            decision["rule_name"] = d_name
        return {
            "decision": decision,
            "observations": observations,
        }

    # The key is added only when it happened, so an allow with no armable miss
    # keeps its exact prior shape.
    out: Dict[str, Any] = {"decision": _allowed(), "observations": observations}
    if armable_miss:
        out["armable_miss"] = True
    return out
