"""
Token Police Python SDK.
Core client providing manual check/log methods and global configuration.
"""
import httpx
import logging
import asyncio
import math
import os
import atexit
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Optional, Dict, Any

from .enforcer import auto_instrument
from .context import random_hex16
from .span_kind import span_kind_for
from .telemetry import setup_opentelemetry
from .state import set_client, get_client, get_client_id
from .exceptions import TokenPoliceBlockedError
from .runtime import resolve_deployment

# Bumped each release. Sent on every API request so rollout of a new SDK
# version is visible fleet-wide.
SDK_VERSION = "1.0.1"
SDK_SCHEMA_VERSION = "1"

logger = logging.getLogger("token_police")


class TokenPolice:
    """
    Token Police Python SDK Client.
    Provides manual check() and log() methods, plus global configuration.
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        timeout: float = 2.0,
        firewall: Optional[str] = None,
        enforce: Optional[bool] = None,
        log_errors: bool = False,
        max_workers: Optional[int] = None,
        deployment: str = "auto",
        sse_reconnect_max_interval_seconds: int = 300,
        capture_stream_usage: Optional[bool] = None,
        error_detail: str = "redacted",
        # Appended LAST on purpose: inserting a parameter mid-signature would
        # silently re-bind every positional caller past that point.
        stream_stale_grace_seconds: float = 60,
    ):
        if not api_key:
            raise ValueError("Token Police SDK: api_key is required.")
        import re
        if not isinstance(api_key, str):
            # A truthy non-string key (int/bytes/object) cannot be regex-
            # validated or sent as a header verbatim. Setup must never throw
            # for a malformed-but-present key — coerce it and let the format
            # warning below fire, exactly like a malformed string key (warn
            # and proceed). Covered by tests/test_init_failopen.py.
            try:
                api_key = str(api_key)
            except Exception:
                api_key = "<invalid>"
        if not re.match(r"^tp_sk_[a-zA-Z0-9_]+$", api_key):
            logger.warning("TokenPolice: api_key format invalid. Expected 'tp_sk_' prefix followed by alphanumerics.")
        self.api_key = api_key

        self.base_url = base_url or os.getenv("TOKENPOLICE_BASE_URL", "https://collect.tokenpolice.ai")
        self.base_url = self.base_url.rstrip("/")

        self.timeout = timeout
        # Three-state firewall mode, default 'dry_run'. Legacy `enforce` bool is
        # a deprecated alias (True → 'enforce', False → 'off'); `firewall` wins
        # when both are set.
        if firewall is not None:
            resolved_firewall = firewall
        elif enforce is True:
            resolved_firewall = "enforce"
        elif enforce is False:
            resolved_firewall = "off"
        else:
            resolved_firewall = "dry_run"
        # Validate the resolved mode against the exact canonical set,
        # byte-for-byte (case-sensitive, NO strip/casefold). An unrecognized
        # value falls back to the SAFE default 'dry_run' — NEVER 'enforce' — so
        # a typo (e.g. "dryrun", "Enforce", " off ") can never silently enable
        # live blocking. Pure local membership logic: it cannot raise or do I/O.
        # The warning reads the `log_errors` INPUT (self.log_errors is not yet
        # assigned here) and logger.warning is total, so it cannot raise.
        if resolved_firewall in ("enforce", "dry_run", "off"):
            self.firewall = resolved_firewall
        else:
            if log_errors:
                logger.warning(
                    "TokenPolice: unknown firewall mode %r; falling back to 'dry_run'.",
                    resolved_firewall,
                )
            self.firewall = "dry_run"
        self.log_errors = log_errors
        self.deployment = resolve_deployment(deployment)
        self.sse_reconnect_max_interval_seconds = sse_reconnect_max_interval_seconds
        # Grace period, in seconds, after the rule stream drops before a locally
        # ALLOWED call whose decision depended on a streamed entity list is
        # re-verified with an inline /check. Clamped to [0, 3600]; a
        # non-numeric / NaN / inf value is a misconfiguration and falls back to
        # the 60s default rather than silently disabling (or unbounding) the
        # gate. Pure local arithmetic — it cannot raise. Parity: Node clamps
        # identically.
        _grace = stream_stale_grace_seconds
        if (
            isinstance(_grace, bool)
            or not isinstance(_grace, (int, float))
            or not math.isfinite(_grace)
        ):
            _grace = 60
        self.stream_stale_grace_seconds = max(0.0, min(3600.0, float(_grace)))
        # Inject stream_options.include_usage into OpenAI-wire chat streams
        # opened without it (the synthetic usage chunk is stripped from the
        # customer's iterator), so streamed spend isn't silently lost. Default
        # on; disable with capture_stream_usage=False or TP_CAPTURE_STREAM_USAGE=0.
        if capture_stream_usage is None:
            capture_stream_usage = os.getenv("TP_CAPTURE_STREAM_USAGE", "1") != "0"
        self.capture_stream_usage = capture_stream_usage
        # Error-detail mode. Coerce any unrecognized/typo value to the
        # SAFE default 'redacted' (NEVER 'raw') at this constructor choke point,
        # so a mistake can never silently re-open the raw-error plaintext path.
        # Pure local membership logic: it cannot raise or do I/O.
        if error_detail in ("none", "redacted", "raw"):
            self.error_detail = error_detail
        else:
            self.error_detail = "redacted"
        # Per-instance UUID for fleet attribution server-side.
        # Generated locally so merely constructing a TokenPolice() never mutates
        # the module-global client id — that is set only when this instance is
        # installed as the global client (see state.set_client).
        self.client_id = uuid.uuid4().hex

        self._max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        self._pending_futures = set()
        self._pending_tasks = set()

        # Standard identification headers sent on every /check, /log, /stream
        # request, enabling per-SDK-fleet filtering server-side.
        common_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-TP-Sdk-Version": SDK_VERSION,
            "X-TP-Sdk-Schema-Version": SDK_SCHEMA_VERSION,
            "X-TP-Client-Id": self.client_id,
            "X-TP-Deployment-Mode": self.deployment,
            "X-TP-Firewall-Mode": self.firewall,
        }
        self._common_headers = common_headers

        # HTTP-client construction can fail on environment the SDK does not
        # control (e.g. a malformed HTTP_PROXY/HTTPS_PROXY env var, an invalid
        # base_url). Setup must never throw customer-side — run degraded with
        # no transport; every per-call path already fails open when the client
        # is unusable. Covered by tests/test_init_failopen.py.
        try:
            self._sync_client = httpx.Client(
                base_url=self.base_url,
                headers=common_headers,
                timeout=self.timeout,
            )
        except Exception as e:
            self._sync_client = None
            if log_errors:
                logger.warning(f"TokenPolice: HTTP client setup failed (running degraded): {e}")
        self._async_client_instance = None
        self._lock = threading.Lock()
        self._stream_client = None  # set if daemon + firewall != 'off', see init()

        # Try to register atexit cleanly
        try:
            atexit.register(self.close_sync)
        except Exception:
            pass

    def _get_async_client(self) -> Optional[httpx.AsyncClient]:
        # Never raises: async-client construction can fail for the same
        # environmental reasons as the sync client (proxy env, base_url).
        # Returns None on failure — the async check()/log() call paths treat
        # an unusable client as fail-open. Covered by tests/test_init_failopen.py.
        if self._async_client_instance is None:
            with self._lock:
                if self._async_client_instance is None:
                    try:
                        self._async_client_instance = httpx.AsyncClient(
                            base_url=self.base_url,
                            headers=self._common_headers,
                            timeout=self.timeout,
                        )
                    except Exception as e:
                        if self.log_errors:
                            logger.warning(f"TokenPolice: async HTTP client setup failed (running degraded): {e}")
                        return None
        return self._async_client_instance

    # ── Synchronous API ──────────────────────────────────────────────

    def check_sync(
        self,
        user_id: str = "anonymous",
        paid_plan: str = "free",
        workflow_name: str = "default",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        intent: Optional[Dict[str, Any]] = None,
        plan_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Sync pre-flight budget check. Fails open on any error.

        The optional ``model`` and ``provider`` arguments let REROUTE rules
        match on the call's target model.

        ``intent`` carries modality-aware hints for non-text calls
        (``{kind: "image_generation"|"audio_speech"|"audio_transcription"|
        "video_generation"|"ocr", count?, expected_seconds?, character_count?,
        size?, quality?}``). Rules can then match on ``intent.kind`` etc.

        ``plan_source`` is the provenance of ``paid_plan``: ``"app"`` (supplied
        by application code, here or inherited from an ancestor scope) or
        ``"default"`` (the SDK synthesized the ``"free"`` fallback — nobody set
        a plan). Anything else, including the ``None`` default when a caller
        does not thread it, OMITS the field; the server reads an absent
        ``plan_source`` as "unknown". Visibility metadata only — it never
        affects rule matching or enforcement.
        """
        payload: Dict[str, Any] = {
            "user": {
                "id": user_id,
                "paid_plan": paid_plan,
            },
            "metadata": metadata or {},
        }
        if isinstance(plan_source, str) and plan_source in ("app", "default"):
            payload["user"]["plan_source"] = plan_source
        if trace_id:
            payload["trace_id"] = trace_id
        # Per-session rules need session_id available before the call so they
        # can match and group per session. Guarded — never raises.
        if session_id:
            payload["session_id"] = session_id
        if model or provider:
            payload["model"] = {
                "name": model or "",
                "provider": provider or "",
            }
        if intent:
            payload["intent"] = intent
            kind = intent.get("kind") if isinstance(intent, dict) else None
            if isinstance(kind, str) and kind:
                payload["modality"] = kind
        # Surface the positional workflow_name into metadata.workflow_name so
        # workflow-scoped rules/budgets and dashboard grouping match this manual
        # call. Fires only for a non-default name the caller has not already
        # placed in metadata; writes to a shallow copy so the caller's dict is
        # never mutated, and is guarded so hostile/non-dict metadata is a no-op.
        if workflow_name and workflow_name != "default":
            _md = payload["metadata"]
            if isinstance(_md, dict):
                try:
                    if "workflow_name" not in _md:
                        payload["metadata"] = dict(_md, workflow_name=workflow_name)
                except Exception:
                    pass

        try:
            response = self._sync_client.post("/v1/guard/check", json=payload)
            # A malformed or empty 429 response body deliberately degrades to
            # allowed here (fail-open) — a proxy/load-balancer 429 with an HTML
            # error body must not block customer traffic just because it isn't
            # valid JSON. A JSON-parse failure here falls through to the outer
            # except below, which returns {"status": "allowed", "fail_open": True}.
            if response.status_code == 429:
                return response.json()
            # 200 may carry a `reroute` directive — forward the body so the
            # enforcer can act on it. Fall back to allowed if body is empty.
            try:
                body = response.json()
                if isinstance(body, dict) and body:
                    return body
            except Exception:
                pass
            return {"status": "allowed"}
        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice check failed (fail-open): {e}")
            return {"status": "allowed", "fail_open": True}

    def log_sync(
        self,
        user_id: str = "anonymous",
        paid_plan: str = "free",
        workflow_name: str = "default",
        session_id: str = "",
        model: str = "unknown",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        span: Optional[Dict[str, Any]] = None,
        prompt_composition: Optional[list] = None,
        response_composition: Optional[list] = None,
        local_decision: Optional[Dict[str, Any]] = None,
        observations: Optional[list] = None,
        call_outcome: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        model_extras: Optional[Dict[str, Any]] = None,
        operation: str = "chat",
        tool: Optional[Dict[str, Any]] = None,
        latency: Optional[Dict[str, Any]] = None,
        plan_source: Optional[str] = None,
    ):
        """
        Schedules the LLM-request log onto a background thread and returns
        immediately; the network POST happens off the caller's thread.

        Extended fields (optional, all default None for backward compat):
            local_decision what the local evaluator decided (BLOCK / REROUTE)
            observations shadow-rule triggers + rejected reroutes
            call_outcome success / failure of the original LLM call

        Cost-signal fields:
            usage verbatim provider usage object + shape enum + optional tier
                    / duration / items. Forwarded verbatim; the service maps
                    (shape, raw) to billable units.
            model_extras optional original_provider / endpoint / deployment /
                          framework hints the cost engine consults to pick the
                          right resolver and apply gateway fees.

        Provenance field:
            plan_source  "app" / "default" — where ``paid_plan`` came from (see
                         ``check_sync``). Any other value, including the ``None``
                         default, omits the field (server reads it as
                         "unknown"). Never affects matching or enforcement.
        """
        model_block: Dict[str, Any] = {
            "name": model,
            "provider": provider,
        }
        if model_extras:
            # api_base = raw serving endpoint (host+path), used server-side to
            # attribute the serving provider (e.g. api.minimax.io -> minimax).
            for k in ("original_provider", "endpoint", "deployment", "framework", "api_base"):
                v = model_extras.get(k)
                if v:
                    model_block[k] = v

        payload: Dict[str, Any] = {
            "user": {
                "id": user_id,
                "paid_plan": paid_plan,
            },
            "model": model_block,
            "metadata": metadata or {},
            "operation": operation or "chat",
        }
        # Only the two canonical values reach the wire; None/anything else is
        # omitted so the server can tell "not reported" from a claim.
        if isinstance(plan_source, str) and plan_source in ("app", "default"):
            payload["user"]["plan_source"] = plan_source
        # Explicit conversation/session id channel (metadata.session_id is
        # also honored as a fallback). Pure assignment — never raises.
        if session_id:
            payload["session_id"] = session_id

        # Forward verbatim provider usage when available; otherwise
        # synthesise a minimal openai_compatible_chat shape so the service
        # still has something to map.
        if usage:
            payload["usage"] = usage
        else:
            raw_synth: Dict[str, Any] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }
            if cached_tokens:
                raw_synth["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
            payload["usage"] = {"shape": "openai_compatible_chat", "raw": raw_synth}

        # Always send a span block carrying a non-empty span_id. The collector
        # keys BOTH its per-span idempotency guard and the generations row's
        # span_id off this value: with none, a replayed body double-counts
        # budget counters and audit rows, and the row lands with an empty
        # span_id that collapses the trace tree. A caller-supplied span_id —
        # including the deliberately deterministic Anthropic-batch ids that
        # *want* server-side dedup — is forwarded untouched. Never mutates the
        # caller's dict (shallow copy); every step is guarded so a hostile
        # `span` value can never raise into caller code.
        span_out: Dict[str, Any] = {}
        try:
            if isinstance(span, dict):
                span_out = dict(span)
        except Exception:
            span_out = {}
        try:
            _sid = span_out.get("span_id")
            if not isinstance(_sid, str) or not _sid:
                span_out["span_id"] = random_hex16()
        except Exception:
            pass
        # Stamp the modality-aware span_kind from `operation` (best-effort;
        # re-derived authoritatively server-side). Structural kinds the
        # caller set (agent/tool/chain) are preserved.
        try:
            span_out["span_kind"] = span_kind_for(operation, span_out.get("span_kind"))
        except Exception:
            pass
        payload["span"] = span_out
        if prompt_composition:
            payload["prompt_composition"] = prompt_composition
        if response_composition:
            payload["response_composition"] = response_composition
        if local_decision:
            payload["local_decision"] = local_decision
        if observations:
            payload["observations"] = observations
        if call_outcome:
            payload["call_outcome"] = call_outcome
        if tool:
            payload["tool"] = tool
        # Client-side latency primitives (TTFT + streaming throughput);
        # throughput metrics are derived from these server-side. Streaming
        # calls only; absent on non-streaming (is_streaming defaults to 0 there).
        if latency:
            payload["latency"] = latency

        # Surface the positional workflow_name into metadata.workflow_name so
        # workflow-scoped rules/budgets and dashboard grouping match this manual
        # call. Fires only for a non-default name the caller has not already
        # placed in metadata; writes to a shallow copy so the caller's dict is
        # never mutated, and is guarded so hostile/non-dict metadata is a no-op.
        if workflow_name and workflow_name != "default":
            _md = payload["metadata"]
            if isinstance(_md, dict):
                try:
                    if "workflow_name" not in _md:
                        payload["metadata"] = dict(_md, workflow_name=workflow_name)
                except Exception:
                    pass

        def _send():
            try:
                self._sync_client.post("/v1/guard/log", json=payload)
            except Exception as e:
                if self.log_errors:
                    logger.warning(f"TokenPolice log failed (swallowed): {e}")

        def _cleanup(f):
            with self._lock:
                self._pending_futures.discard(f)

        # Guard the submit so log_sync never raises into caller code. After
        # close_sync()/atexit the executor is shut down and .submit() raises
        # RuntimeError; any other internal failure of the submit path is
        # likewise swallowed. Warn only when self.log_errors, with a generic
        # content-free message. Narrowly gated — the executor-alive path still
        # submits and tracks the future exactly as before; only the failing
        # submit fails open.
        try:
            future = self._executor.submit(_send)
            with self._lock:
                self._pending_futures.add(future)
            future.add_done_callback(_cleanup)
        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice log failed (swallowed): {e}")

    def flush_sync(self):
        """
        Block until every pending background telemetry thread finishes.

        This is a genuine drain — it ``wait()``s on the outstanding thread-pool
        futures. (This is where the Python SDK diverges from Node, whose
        ``flushSync`` is a diagnostic-only no-op that relies on keepalive POSTs;
        Python has real worker threads to join.) In async code call
        ``await flush()`` instead so the event loop isn't blocked.
        """
        with self._lock:
            futures = list(self._pending_futures)
        if futures:
            wait(futures)

    def close_sync(self):
        """Synchronously closes the sync HTTP client and background threads."""
        # Stop the SSE reader thread first. It's a daemon thread (won't block
        # interpreter exit), but on a re-init() set_client() calls close_sync()
        # on the OLD client — without this the old reader keeps running and
        # races the new one writing the shared module-level pack state.
        try:
            if getattr(self, "_stream_client", None):
                self._stream_client.stop()
        except Exception:
            pass
        try:
            self.flush_sync()
            self._executor.shutdown(wait=True)
            self._sync_client.close()
        except Exception:
            pass
        try:
            atexit.unregister(self.close_sync)
        except Exception:
            pass

    async def close_async(self):
        """Asynchronously closes the async HTTP client and awaits pending tasks."""
        await self.flush()
        await asyncio.to_thread(self.close_sync)
        if self._async_client_instance:
            try:
                await self._async_client_instance.aclose()
            except Exception:
                pass

    # ── Async API (for direct use) ───────────────────────────────────

    async def check(
        self,
        user_id: str = "anonymous",
        paid_plan: str = "free",
        workflow_name: str = "default",
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        intent: Optional[Dict[str, Any]] = None,
        plan_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Async pre-flight budget check. Fails open on any error.

        ``intent`` carries modality-aware hints for non-text calls; see
        ``check_sync`` for the shape. ``plan_source`` is the provenance of
        ``paid_plan`` ("app" / "default"); see ``check_sync``. Anything else
        omits the field.
        """
        payload: Dict[str, Any] = {
            "user": {
                "id": user_id,
                "paid_plan": paid_plan,
            },
            "metadata": metadata or {},
        }
        if isinstance(plan_source, str) and plan_source in ("app", "default"):
            payload["user"]["plan_source"] = plan_source
        if trace_id:
            payload["trace_id"] = trace_id
        # Per-session rules need session_id available before the call so they
        # can match and group per session. Guarded — never raises.
        if session_id:
            payload["session_id"] = session_id
        if model or provider:
            payload["model"] = {
                "name": model or "",
                "provider": provider or "",
            }
        if intent:
            payload["intent"] = intent
            kind = intent.get("kind") if isinstance(intent, dict) else None
            if isinstance(kind, str) and kind:
                payload["modality"] = kind
        # Surface the positional workflow_name into metadata.workflow_name so
        # workflow-scoped rules/budgets and dashboard grouping match this manual
        # call. Fires only for a non-default name the caller has not already
        # placed in metadata; writes to a shallow copy so the caller's dict is
        # never mutated, and is guarded so hostile/non-dict metadata is a no-op.
        if workflow_name and workflow_name != "default":
            _md = payload["metadata"]
            if isinstance(_md, dict):
                try:
                    if "workflow_name" not in _md:
                        payload["metadata"] = dict(_md, workflow_name=workflow_name)
                except Exception:
                    pass

        try:
            client = self._get_async_client()
            response = await client.post("/v1/guard/check", json=payload)
            # A malformed or empty 429 response body deliberately degrades to
            # allowed here (fail-open) — a proxy/load-balancer 429 with an HTML
            # error body must not block customer traffic just because it isn't
            # valid JSON. A JSON-parse failure here falls through to the outer
            # except below, which returns {"status": "allowed", "fail_open": True}.
            if response.status_code == 429:
                return response.json()
            try:
                body = response.json()
                if isinstance(body, dict) and body:
                    return body
            except Exception:
                pass
            return {"status": "allowed"}
        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice check failed (fail-open): {e}")
            return {"status": "allowed", "fail_open": True}

    async def log(
        self,
        user_id: str = "anonymous",
        paid_plan: str = "free",
        workflow_name: str = "default",
        session_id: str = "",
        model: str = "unknown",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        span: Optional[Dict[str, Any]] = None,
        prompt_composition: Optional[list] = None,
        response_composition: Optional[list] = None,
        local_decision: Optional[Dict[str, Any]] = None,
        observations: Optional[list] = None,
        call_outcome: Optional[Dict[str, Any]] = None,
        usage: Optional[Dict[str, Any]] = None,
        model_extras: Optional[Dict[str, Any]] = None,
        operation: str = "chat",
        tool: Optional[Dict[str, Any]] = None,
        latency: Optional[Dict[str, Any]] = None,
        plan_source: Optional[str] = None,
    ):
        """
        Async log call. Fire-and-forget.

        Extended fields (optional): local_decision, observations, call_outcome.
        Cost-signal fields: usage (verbatim provider usage + shape), model_extras.
        Provenance field: plan_source ("app" / "default"; see ``check_sync``) —
        anything else omits it.
        """
        model_block: Dict[str, Any] = {
            "name": model,
            "provider": provider,
        }
        if model_extras:
            # api_base = raw serving endpoint (host+path), used server-side to
            # attribute the serving provider (e.g. api.minimax.io -> minimax).
            for k in ("original_provider", "endpoint", "deployment", "framework", "api_base"):
                v = model_extras.get(k)
                if v:
                    model_block[k] = v

        payload: Dict[str, Any] = {
            "user": {
                "id": user_id,
                "paid_plan": paid_plan,
            },
            "model": model_block,
            "metadata": metadata or {},
            "operation": operation or "chat",
        }
        # Only the two canonical values reach the wire; None/anything else is
        # omitted so the server can tell "not reported" from a claim.
        if isinstance(plan_source, str) and plan_source in ("app", "default"):
            payload["user"]["plan_source"] = plan_source
        # Explicit conversation/session id channel (metadata.session_id is
        # also honored as a fallback). Pure assignment — never raises.
        if session_id:
            payload["session_id"] = session_id

        if usage:
            payload["usage"] = usage
        else:
            raw_synth: Dict[str, Any] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
            }
            if cached_tokens:
                raw_synth["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
            payload["usage"] = {"shape": "openai_compatible_chat", "raw": raw_synth}

        # Always send a span block carrying a non-empty span_id. The collector
        # keys BOTH its per-span idempotency guard and the generations row's
        # span_id off this value: with none, a replayed body double-counts
        # budget counters and audit rows, and the row lands with an empty
        # span_id that collapses the trace tree. A caller-supplied span_id —
        # including the deliberately deterministic Anthropic-batch ids that
        # *want* server-side dedup — is forwarded untouched. Never mutates the
        # caller's dict (shallow copy); every step is guarded so a hostile
        # `span` value can never raise into caller code.
        span_out: Dict[str, Any] = {}
        try:
            if isinstance(span, dict):
                span_out = dict(span)
        except Exception:
            span_out = {}
        try:
            _sid = span_out.get("span_id")
            if not isinstance(_sid, str) or not _sid:
                span_out["span_id"] = random_hex16()
        except Exception:
            pass
        # Stamp the modality-aware span_kind from `operation` (best-effort;
        # re-derived authoritatively server-side). Structural kinds the
        # caller set (agent/tool/chain) are preserved.
        try:
            span_out["span_kind"] = span_kind_for(operation, span_out.get("span_kind"))
        except Exception:
            pass
        payload["span"] = span_out
        if prompt_composition:
            payload["prompt_composition"] = prompt_composition
        if response_composition:
            payload["response_composition"] = response_composition
        if local_decision:
            payload["local_decision"] = local_decision
        if observations:
            payload["observations"] = observations
        if call_outcome:
            payload["call_outcome"] = call_outcome
        if tool:
            payload["tool"] = tool
        # Client-side latency primitives (see log_sync).
        if latency:
            payload["latency"] = latency

        # Surface the positional workflow_name into metadata.workflow_name so
        # workflow-scoped rules/budgets and dashboard grouping match this manual
        # call. Fires only for a non-default name the caller has not already
        # placed in metadata; writes to a shallow copy so the caller's dict is
        # never mutated, and is guarded so hostile/non-dict metadata is a no-op.
        if workflow_name and workflow_name != "default":
            _md = payload["metadata"]
            if isinstance(_md, dict):
                try:
                    if "workflow_name" not in _md:
                        payload["metadata"] = dict(_md, workflow_name=workflow_name)
                except Exception:
                    pass

        async def _send():
            try:
                client = self._get_async_client()
                await client.post("/v1/guard/log", json=payload)
            except Exception as e:
                if self.log_errors:
                    logger.warning(f"TokenPolice log failed (swallowed): {e}")
                    
        # Guard the schedule so async log() never raises into caller code.
        # asyncio.create_task() calls get_running_loop(), which raises RuntimeError
        # ("no running event loop" when driven without a running loop, or "Event
        # loop is closed" on async teardown/atexit); any other internal failure of
        # the scheduling tail is likewise swallowed. except Exception (so
        # CancelledError/KeyboardInterrupt/SystemExit still propagate), warn only
        # when self.log_errors, with a generic content-free message. Narrowly gated
        # — with a running loop the task is created and tracked exactly as before;
        # only the failing schedule fails open. Fire-and-forget: returns None.
        try:
            task = asyncio.create_task(_send())
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice log failed (swallowed): {e}")

    async def flush(self):
        """
        Asynchronously waits for all pending background telemetry to complete —
        both the async ``log()`` tasks scheduled on the running event loop AND the
        background threads that carry auto-instrumented telemetry (submitted via
        ``log_sync()`` to the thread pool). Draining only the async tasks would let
        the thread-pool rows stay in flight, so an async serverless handler that
        awaits ``flush()`` before the container sleeps could lose that spend.
        Useful for serverless environments before the container sleeps.
        """
        with self._lock:
            tasks = list(self._pending_tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        # Drain the thread pool off-loop so the background telemetry threads are
        # awaited too, without blocking the event loop (mirrors close_async).
        # Guarded so flush() can never raise into customer code (it is invoked
        # from the serverless decorator's finally around the customer handler).
        # Covered by test_async_flush_drains_thread_pool_futures and
        # test_async_flush_never_raises_when_flush_sync_fails in
        # tests/test_sec09_async_log_failopen.py.
        try:
            await asyncio.to_thread(self.flush_sync)
        except Exception as e:
            if self.log_errors:
                logger.warning(f"TokenPolice flush failed (swallowed): {e}")

    async def close(self):
        await self.close_async()


def init(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 2.0,
    firewall: Optional[str] = None,
    enforce: Optional[bool] = None,
    log_errors: bool = False,
    max_workers: Optional[int] = None,
    tracer_provider: Optional[Any] = None,
    deployment: str = "auto",
    sse_reconnect_max_interval_seconds: int = 300,
    capture_stream_usage: Optional[bool] = None,
    error_detail: str = "redacted",
    # Appended LAST on purpose — see TokenPolice.__init__.
    stream_stale_grace_seconds: float = 60,
) -> TokenPolice:
    """
    Global configuration.
    Initializes OpenTelemetry and applies pre-flight wrappers automatically.

    Args:
        api_key: Your TokenPolice API key (tp_sk_...). Optional here:
                     when omitted (or falsy) it falls back to the
                     TOKENPOLICE_API_KEY env var. A ValueError is raised only
                     when neither the argument nor the env var supplies a key.
        base_url: TokenPolice API URL. Defaults to TOKENPOLICE_BASE_URL env var or https://collect.tokenpolice.ai.
        timeout: Max seconds for any remote call (default: 2.0).
        firewall: 'enforce' acts on rules (per each rule's mode); 'dry_run'
                     (default) evaluates and emits WOULD_* telemetry but never
                     acts/throws; 'off' is telemetry only — usage is logged for
                     every provider, but no pre-flight /check, no enforcement,
                     and no SSE stream.
        enforce: Deprecated alias for `firewall` (True → 'enforce',
                     False → 'off'). Ignored when `firewall` is set.
        log_errors: If True, SDK errors are printed at WARNING level.
        max_workers: Size of the background thread pool that carries
                     auto-instrumented telemetry POSTs off the request path
                     (default None → min(32, os.cpu_count() + 4)).
        tracer_provider: An existing OpenTelemetry TracerProvider to attach the
                     private token-extraction SpanProcessor to. Default None →
                     the SDK builds its own exporter-less provider so nothing is
                     shipped to a tracing backend.
        error_detail: How much of a provider error is recorded on failed calls:
                     'redacted' (default) ships a SHA-256 hash of the message,
                     'none' drops it entirely, 'raw' keeps the plaintext. Any
                     unrecognized value is coerced to the safe 'redacted'.
        deployment: 'auto' (default), 'daemon', 'serverless', or 'edge'.
                     In daemon mode with firewall 'enforce'/'dry_run' the SDK
                     opens an SSE connection to /v1/guard/stream and evaluates
                     locally. Serverless and edge always use inline /check.
        sse_reconnect_max_interval_seconds: Maximum SSE reconnect backoff
                     when the TokenPolice API is unreachable (default 300).
        stream_stale_grace_seconds: Grace period after the rule stream drops
                     before a locally allowed call whose decision depended on a
                     streamed entity list is re-verified with an inline /check
                     (default 60, clamped to [0, 3600]; 0 = verify from the
                     first missed moment). Budget blocks reach the SDK only over
                     the stream, so without this a long disconnect would let
                     already-blocked entities keep passing. Only entity-matching
                     calls pay the round-trip, and it stays fail-open.
        capture_stream_usage: Inject stream_options.include_usage into OpenAI-wire
                     chat streams so streamed spend isn't lost (default None →
                     on unless TP_CAPTURE_STREAM_USAGE=0).
    """
    if not api_key:
        import os
        api_key = os.getenv("TOKENPOLICE_API_KEY")

    if not api_key:
        raise ValueError("TokenPolice SDK: api_key is required.")

    # A truthy non-string key would raise on .startswith here — skip the
    # prefix warning for it; the constructor coerces it and emits the format
    # warning instead. init() must never throw except the missing-key
    # ValueError above. Covered by tests/test_init_failopen.py.
    if isinstance(api_key, str) and not api_key.startswith("tp_sk_"):
        logger.warning("TokenPolice: api_key does not start with 'tp_sk_'. It may be invalid.")

    # Warn (once) when init() replaces an already-installed client. The
    # teardown-and-replace in set_client() is intentional (test isolation /
    # hot-reload), but a silent swap hides a double-init bug. Detect the PRIOR
    # instance BEFORE constructing/swapping, and wrap the read+warn so a broken
    # logger can NEVER throw out of init() — it must never raise into the
    # host application (fail-open).
    try:
        if get_client():
            logger.warning(
                "TokenPolice: init() called again — replacing the previously initialized client."
            )
    except Exception:
        # Silent fail-open — a warning must never crash the customer's init().
        pass

    client = TokenPolice(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        firewall=firewall,
        enforce=enforce,
        log_errors=log_errors,
        max_workers=max_workers,
        deployment=deployment,
        sse_reconnect_max_interval_seconds=sse_reconnect_max_interval_seconds,
        stream_stale_grace_seconds=stream_stale_grace_seconds,
        capture_stream_usage=capture_stream_usage,
        error_detail=error_detail,
    )
    set_client(client)

    # 1. Initialize local OpenTelemetry for token extraction.
    # Guarded: a provider-construction / OTel global-state failure must
    # degrade to reduced telemetry, never throw into init().
    try:
        setup_opentelemetry(log_errors=log_errors, tracer_provider=tracer_provider)
    except Exception as err:
        # Silent fail-open — customer app keeps running with reduced telemetry.
        logger.debug("TokenPolice: setup_opentelemetry failed: %s", err)

    # 2. Install provider taps for ALL firewall modes, including 'off'. The taps
    # are the same in every mode; behavior is decided at CALL time inside the
    # enforcer choke points (`_run_sync_check`/`_run_async_check`), which
    # short-circuit to log-only when `firewall == "off"` (no /check, no block,
    # no reroute). So 'off' installs the taps to deliver full telemetry
    # (log-only) for manual-tap providers, while 'enforce'/'dry_run'
    # additionally evaluate.
    auto_instrument()

    # 3. Daemon + wired → start the SSE reader on a background thread. In
    # serverless/edge or 'off' mode the SSE channel is skipped.
    if client.firewall != "off" and client.deployment == "daemon":
        try:
            from .stream import StreamClient
            client._stream_client = StreamClient(
                base_url=client.base_url,
                api_key=api_key,
                sdk_version=SDK_VERSION,
                deployment=client.deployment,
                client_id=client.client_id,
                firewall=client.firewall,
                reconnect_cap_seconds=sse_reconnect_max_interval_seconds,
            )
            client._stream_client.start()
            logger.debug("TokenPolice: SSE stream started (deployment=daemon, firewall=%s)", client.firewall)
        except Exception as err:
            # Stream failure is silent fail-open — the enforcer falls back to
            # a per-call pre-flight check.
            logger.debug("TokenPolice: failed to start SSE stream: %s", err)

    # 4. Eagerly initialize async client if we're inside an async loop.
    # RuntimeError = no running loop → expected silent no-op. Any other
    # failure of this eager warm-up must not escape init() — the client is
    # built lazily (and fail-open) on first async use instead.
    try:
        import asyncio
        asyncio.get_running_loop()
        client._get_async_client()
    except RuntimeError:
        pass
    except Exception as err:
        logger.debug("TokenPolice: eager async client init failed: %s", err)

    # 5. Best-effort: if the OpenAI Agents SDK is already imported, register the
    # tool-span processor now. Normally `agents` is imported after tp.init(),
    # so the manual-wrapper path registers it lazily on first use instead.
    try:
        from .openai_agents import maybe_register_openai_agents_tracing
        maybe_register_openai_agents_tracing()
    except Exception:
        pass

    return client
