"""LangChain + google_genai must not double-log the inner OTel span.

Regression for real-apps support_agent_langgraph_python / lg_gemini_prebuilt:
ChatGoogleGenerativeAI triggers BOTH LangchainInstrumentor (canonical
openai_compatible_chat row with compositions) and GoogleGenAiSdkInstrumentor
(nested google_genai row, empty compositions). When the race fires, both rows
land and cost is counted twice.

Root cause (re-opened 07-25): stamp-only fix (PR #195) gated on the same
`_in_langchain` ContextVar that LangGraph async loses *at on_start* (Task
snapshotted before the wrapper set True). Structural backstop: process-global
trace-keyed registry entered by LC wrappers; telemetry consults it when the
contextvar is False so the stamp still fires and the ghost never logs.

Root cause (re-opened again 07-26 — attempt #3): BOTH of the above were alive
and correctly wired, and neither was ever consulted. They hung off one exact
scope-string equality, and google-genai 1.0b1 stopped owning a tracer: it
delegates span creation to the shared opentelemetry-util-genai TelemetryHandler,
whose scope is `opentelemetry.util.genai.handler`. The equality could never
match, so nothing was stamped and nothing was dropped.

These tests previously passed while testing only the pre-1.0b1 world — the
_FakeSpan default scope and the old-semconv `gen_ai.system` attrs are exactly
the shape the failing environment no longer produces. The suite is therefore
parametrised over BOTH instrumentor generations by subclassing:

  TestLangchainGoogleGenaiOtelDedup → scope opentelemetry.instrumentation.google_genai (≤0.7b1)
  TestUtilGenaiScopeGoogleGenaiDedup → scope opentelemetry.util.genai.handler (1.0b1+) ← the acceptance gate
  TestUtilGenaiVertexAiScopeGoogleGenaiDedup → same, provider=vertex_ai

Subclassing rather than subTest is deliberate: each leg gets a fresh
setUp/tearDown, so the shared client and the process-global registry cannot
bleed between legs.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace as NS

from token_police import context as ctx
from token_police.context import (
    TPSession,
    _in_langchain,
    enter_langchain_trace,
    leave_langchain_trace,
    langchain_trace_active,
    _lc_trace_depth,
    _lc_trace_lock,
)
from token_police.state import set_client
from token_police.telemetry import (
    TokenPoliceSpanProcessor,
    _is_google_genai_instrumentor_span,
)


class _Scope:
    def __init__(self, name):
        self.name = name


class _FakeSpan:
    """Writable attributes so on_start set_attribute is observable."""

    def __init__(self, attributes=None, name="google.genai.generate_content",
                 scope="opentelemetry.instrumentation.google_genai"):
        self.attributes = dict(attributes or {})
        self.name = name
        self.instrumentation_scope = _Scope(scope)
        self.context = NS(trace_id=0x1111, span_id=0x2222)
        self.parent = None
        self.start_time = 1_000_000_000
        self.end_time = 2_000_000_000
        self.status = None

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


def _google_genai_usage_attrs():
    return {
        "gen_ai.system": "google_genai",
        "gen_ai.request.model": "gemini-2.0-flash",
        "gen_ai.usage.input_tokens": 219,
        "gen_ai.usage.output_tokens": 46,
    }


def _openai_usage_attrs():
    return {
        "gen_ai.system": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 20,
    }


def _util_genai_start_attrs(provider="gemini", model="gemini-2.0-flash"):
    """google-genai 1.0b1 creation-time attribute shape.

    opentelemetry-util-genai passes these to start_span(attributes=...) via
    _inference_invocation._get_base_attributes, so they ARE readable in
    on_start. Faithfully omits `gen_ai.system` (never set on this generation)
    and usage (arrives at _apply_finish, i.e. span end).
    """
    return {
        "gen_ai.operation.name": "generate_content",
        "gen_ai.provider.name": provider,
        "gen_ai.request.model": model,
    }


def _util_genai_finish_attrs():
    """Usage attributes util-genai applies when the span ends."""
    return {
        "gen_ai.usage.input_tokens": 219,
        "gen_ai.usage.output_tokens": 46,
    }


def _util_genai_openai_attrs():
    """A hypothetical future OpenAI instrumentor on the SHARED util-genai
    handler: same scope, non-Google provider. Must never be suppressed."""
    return {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o-mini",
        "gen_ai.usage.input_tokens": 100,
        "gen_ai.usage.output_tokens": 20,
    }


class _DedupTestBase(unittest.TestCase):
    """Shared setUp/tearDown for every leg."""

    def setUp(self):
        self.client = _FakeClient()
        set_client(self.client)
        self.session = TPSession(
            user_id="usr_free_001",
            paid_plan="free",
            workflow_name="support_desk_agent_langgraph_python",
            trace_id="a" * 32,
            root_span_id="b" * 16,
            session_id="sess-1",
        )
        self._token = ctx._current_session.set(self.session)
        self.proc = TokenPoliceSpanProcessor()

    def tearDown(self):
        ctx._current_session.reset(self._token)
        set_client(None)
        # Never leave the framework guard stuck True across tests.
        try:
            _in_langchain.set(False)
        except Exception:
            pass
        # Drain any leftover registry entries from this session's trace.
        try:
            while langchain_trace_active(self.session.trace_id):
                leave_langchain_trace(self.session.trace_id)
        except Exception:
            pass
        try:
            with _lc_trace_lock:
                _lc_trace_depth.clear()
        except Exception:
            pass


class TestLangchainGoogleGenaiOtelDedup(_DedupTestBase):
    """Legacy generation (≤0.7b1): the instrumentor owns its own tracer."""

    SCOPE = "opentelemetry.instrumentation.google_genai"
    SPAN_NAME = "google.genai.generate_content"

    # ── Per-generation span factory; subclasses override these three ─────

    def _google_attrs(self):
        return _google_genai_usage_attrs()

    def _google_span(self, extra=None):
        attrs = self._google_attrs()
        if extra:
            attrs.update(extra)
        return _FakeSpan(attrs, name=self.SPAN_NAME, scope=self.SCOPE)

    def _finish(self, span):
        """Attach attributes that only arrive when the span ends.

        No-op on the legacy factory, which carries usage from creation. The
        util-genai subclass adds gen_ai.usage.* here, matching where 1.0b1
        actually sets them (_inference_invocation._apply_finish).
        """
        return span

    def test_on_start_stamps_suppress_under_langchain_guard(self):
        """(a) on_start under LC + google_genai scope stamps tp.suppress."""
        span = self._google_span()
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_langchain.reset(tok)
        self.assertTrue(
            span.attributes.get("tp.suppress"),
            "on_start under langchain must stamp tp.suppress so late on_end drops",
        )
        # Must not reserve/consume a span_order for a suppressed inner span.
        self.assertNotIn("tp.span_order", span.attributes)
        self.assertNotIn("tp.user_id", span.attributes)

    def test_on_end_drops_suppressed_even_after_guard_cleared(self):
        """(b) Race simulation: guard already False when OTel finishes the span."""
        span = self._google_span(extra={"tp.suppress": True})
        self._finish(span)
        self.assertFalse(_in_langchain.get())
        self.proc.on_end(span)
        self.assertEqual(
            self.client.calls, [],
            "tp.suppress must drop the ghost even when in_langchain() is False",
        )

    def test_direct_google_genai_without_guard_is_logged(self):
        """(c) Standalone google-genai apps must keep their row."""
        span = self._google_span()
        self.assertFalse(_in_langchain.get())
        self.proc.on_start(span)
        self._finish(span)
        self.proc.on_end(span)
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["user_id"], "usr_free_001")
        self.assertNotIn("tp.suppress", span.attributes)

    def test_on_end_contextvar_path_still_drops_without_stamp(self):
        """(d) Defense-in-depth: in_langchain() alone still drops at on_end."""
        span = self._google_span()
        self._finish(span)
        # No stamp — simulate older path / stamp write failure while flag is set.
        tok = _in_langchain.set(True)
        try:
            self.proc.on_end(span)
        finally:
            _in_langchain.reset(tok)
        self.assertEqual(
            self.client.calls, [],
            "on_end contextvar guard must still drop google_genai under langchain",
        )

    def test_openai_scope_under_langchain_is_not_stamped(self):
        """(e) Do NOT suppress non-google_genai scopes under langchain.

        ChatOpenAI / ChatAnthropic intentionally defer the inner provider OTel
        span as their single priced row. Broadening suppress to all scopes would
        erase that telemetry.

        Deliberately NOT parametrised — this pins the legacy openai scope
        exactly. The util-genai equivalent lives in
        TestUtilGenaiNonGoogleNotSuppressed.
        """
        span = _FakeSpan(
            _openai_usage_attrs(),
            name="openai.chat",
            scope="opentelemetry.instrumentation.openai",
        )
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_langchain.reset(tok)
        self.assertNotIn(
            "tp.suppress",
            span.attributes,
            "inner openai span under langchain must NOT be stamped suppressed",
        )
        # Span is enriched (span_order allocated) — langchain defer path needs it.
        self.assertIn("tp.span_order", span.attributes)

    def test_end_to_end_race_on_start_then_late_on_end(self):
        """Full race sequence: on_start under guard stamps; guard clears;
        late on_end must not emit a ghost /log row."""
        span = self._google_span()
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_langchain.reset(tok)
        self.assertTrue(span.attributes.get("tp.suppress"))
        self.assertFalse(_in_langchain.get())
        self._finish(span)
        self.proc.on_end(span)
        self.assertEqual(
            self.client.calls, [],
            "stamped google_genai span must not log after guard is cleared",
        )

    # ── Structural backstop (registry) — the 07-25 re-open case ─────────

    def test_registry_stamps_when_contextvar_already_false(self):
        """(f) Total contextvar loss at on_start: registry alone stamps + drops.

        This is the case PR #195's tests could not cover — contextvar is False
        before on_start, not merely between start and end.
        """
        self.assertFalse(_in_langchain.get())
        enter_langchain_trace(self.session.trace_id)
        try:
            self.assertTrue(langchain_trace_active(self.session.trace_id))
            span = self._google_span()
            self.proc.on_start(span)
            self.assertTrue(
                span.attributes.get("tp.suppress"),
                "registry must stamp tp.suppress even when in_langchain() is False",
            )
            self.assertNotIn("tp.span_order", span.attributes)
            self._finish(span)
            self.proc.on_end(span)
            self.assertEqual(
                self.client.calls, [],
                "registry-stamped google_genai span must not /log",
            )
        finally:
            leave_langchain_trace(self.session.trace_id)

    def test_standalone_without_registry_still_logs(self):
        """(g) No contextvar, no registry → row kept (direct google.genai)."""
        self.assertFalse(_in_langchain.get())
        self.assertFalse(langchain_trace_active(self.session.trace_id))
        span = self._google_span()
        self.proc.on_start(span)
        self._finish(span)
        self.proc.on_end(span)
        self.assertEqual(len(self.client.calls), 1)
        self.assertNotIn("tp.suppress", span.attributes)

    def test_leave_clears_registry_suppress_path(self):
        """(h) After leave, google_genai is no longer suppressed."""
        enter_langchain_trace(self.session.trace_id)
        leave_langchain_trace(self.session.trace_id)
        self.assertFalse(langchain_trace_active(self.session.trace_id))
        span = self._google_span()
        self.proc.on_start(span)
        self.assertNotIn("tp.suppress", span.attributes)
        self._finish(span)
        self.proc.on_end(span)
        self.assertEqual(len(self.client.calls), 1)

    def test_nested_depth_stays_active_until_outermost_leave(self):
        """(i) Nested enter: one leave still leaves registration live."""
        enter_langchain_trace(self.session.trace_id)
        enter_langchain_trace(self.session.trace_id)
        try:
            leave_langchain_trace(self.session.trace_id)
            self.assertTrue(langchain_trace_active(self.session.trace_id))
            span = self._google_span()
            self.proc.on_start(span)
            self.assertTrue(span.attributes.get("tp.suppress"))
        finally:
            leave_langchain_trace(self.session.trace_id)
        self.assertFalse(langchain_trace_active(self.session.trace_id))

    def test_openai_scope_under_registry_is_not_stamped(self):
        """(j) Registry must not broaden suppress beyond google_genai.

        Deliberately NOT parametrised — pins the legacy openai scope exactly.
        """
        enter_langchain_trace(self.session.trace_id)
        try:
            span = _FakeSpan(
                _openai_usage_attrs(),
                name="openai.chat",
                scope="opentelemetry.instrumentation.openai",
            )
            self.proc.on_start(span)
            self.assertNotIn("tp.suppress", span.attributes)
            self.assertIn("tp.span_order", span.attributes)
        finally:
            leave_langchain_trace(self.session.trace_id)

    def test_empty_trace_id_never_suppresses(self):
        """(k) Empty/missing trace_id → fail-open (no suppress)."""
        # enter with empty is a no-op; active("") is False.
        enter_langchain_trace("")
        self.assertFalse(langchain_trace_active(""))
        # Bind a session with empty trace_id so get_current_session sees it.
        bad = TPSession(
            user_id="usr_free_001",
            paid_plan="free",
            workflow_name="w",
            trace_id="",
            root_span_id="b" * 16,
            session_id="sess-empty",
        )
        tok = ctx._current_session.set(bad)
        try:
            span = self._google_span()
            self.proc.on_start(span)
            self.assertNotIn(
                "tp.suppress",
                span.attributes,
                "empty trace_id must not suppress (fail-open)",
            )
        finally:
            ctx._current_session.reset(tok)

    def test_on_end_registry_drops_without_stamp(self):
        """on_end registry path alone drops when stamp write never happened."""
        self.assertFalse(_in_langchain.get())
        enter_langchain_trace(self.session.trace_id)
        try:
            span = self._google_span()
            self._finish(span)
            # Skip on_start entirely — stamp absent; registry still live.
            self.proc.on_end(span)
            self.assertEqual(
                self.client.calls, [],
                "on_end registry backstop must drop even without tp.suppress stamp",
            )
        finally:
            leave_langchain_trace(self.session.trace_id)


class TestUtilGenaiScopeGoogleGenaiDedup(TestLangchainGoogleGenaiOtelDedup):
    """THE ACCEPTANCE GATE for attempt #3.

    Re-runs every load-bearing case with the instrumentation scope moved to
    `opentelemetry.util.genai.handler` and the new-semconv attribute shape —
    i.e. the world google-genai 1.0b1 actually produces, which is what silently
    killed PR #195 and PR #214. If a future change re-keys suppression on an
    exact scope string, this class goes red.
    """

    SCOPE = "opentelemetry.util.genai.handler"
    SPAN_NAME = "generate_content gemini-2.0-flash"
    PROVIDER = "gemini"

    def _google_attrs(self):
        return _util_genai_start_attrs(provider=self.PROVIDER)

    def _finish(self, span):
        for k, v in _util_genai_finish_attrs().items():
            span.set_attribute(k, v)
        return span


class TestUtilGenaiVertexAiScopeGoogleGenaiDedup(TestUtilGenaiScopeGoogleGenaiDedup):
    """Same contract when the instrumentor reports Vertex AI rather than the
    Gemini API. `_determine_genai_system` emits exactly these two values."""

    PROVIDER = "vertex_ai"


class TestUtilGenaiNonGoogleNotSuppressed(_DedupTestBase):
    """Over-suppression guards — the severity class.

    opentelemetry-util-genai is a SHARED library. If another instrumentor
    migrates onto TelemetryHandler it emits under the SAME scope, and
    suppressing it under LangChain would erase the only priced row for that
    call. Matching the scope prefix must therefore never be sufficient on its
    own.
    """

    SCOPE = "opentelemetry.util.genai.handler"

    def _openai_util_span(self, extra=None):
        attrs = _util_genai_openai_attrs()
        if extra:
            attrs.update(extra)
        return _FakeSpan(attrs, name="chat gpt-4o-mini", scope=self.SCOPE)

    def test_util_genai_openai_under_contextvar_is_not_stamped(self):
        span = self._openai_util_span()
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_langchain.reset(tok)
        self.assertNotIn(
            "tp.suppress", span.attributes,
            "util-genai scope alone must not suppress a non-Google provider",
        )
        self.assertIn("tp.span_order", span.attributes)

    def test_util_genai_openai_under_registry_is_not_stamped(self):
        enter_langchain_trace(self.session.trace_id)
        try:
            span = self._openai_util_span()
            self.proc.on_start(span)
            self.assertNotIn("tp.suppress", span.attributes)
            self.assertIn("tp.span_order", span.attributes)
        finally:
            leave_langchain_trace(self.session.trace_id)

    def test_util_genai_openai_under_contextvar_still_logs_at_on_end(self):
        """The row must survive end to end — losing it is worse than the dup."""
        span = self._openai_util_span()
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
            self.proc.on_end(span)
        finally:
            _in_langchain.reset(tok)
        self.assertEqual(
            len(self.client.calls), 1,
            "a non-Google util-genai span under langchain must still be logged",
        )

    def test_util_genai_openai_with_generate_content_op_is_not_suppressed(self):
        """A present provider attribute vetoes the operation-name fallback.

        `generate_content` is a standard semconv operation name, not Google
        vocabulary, so treating it as an independent Google signal (as the
        original fix proposal did) would suppress this span.
        """
        span = self._openai_util_span(
            extra={"gen_ai.operation.name": "generate_content"}
        )
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_langchain.reset(tok)
        self.assertNotIn("tp.suppress", span.attributes)
        self.assertIn("tp.span_order", span.attributes)

    def test_unrelated_scope_with_google_provider_is_not_suppressed(self):
        """The util-genai scope prefix is necessary, not merely corroborating."""
        span = _FakeSpan(
            {
                "gen_ai.provider.name": "gemini",
                "gen_ai.request.model": "gemini-2.0-flash",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 5,
            },
            name="ChatGoogleGenerativeAI.chat",
            scope="opentelemetry.instrumentation.langchain",
        )
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_langchain.reset(tok)
        self.assertNotIn("tp.suppress", span.attributes)


class TestUtilGenaiEmbeddingSpan(_DedupTestBase):
    """Google *embedding* spans on 1.0b1 also carry gen_ai.provider.name, so the
    predicate matches them too. That is intended and emits identical data —
    TokenPolice meters embeddings on its own manual wrapper, so the
    instrumentor's span is dropped either way (by tp.suppress under LangChain,
    or by _is_instrumentor_embedding_span standalone).

    The one real delta versus the dead-gate status quo: under LangChain the span
    is now stamped at on_start and therefore no longer consumes a tp.span_order
    before being dropped, closing a step-numbering gap. That matches what the
    legacy 0.7b1 scope already did. Pinned here because it is a behavioural
    change, even though no row count moves.
    """

    def _embedding_span(self):
        return _FakeSpan(
            {
                "gen_ai.operation.name": "embeddings",
                "gen_ai.provider.name": "gemini",
                "gen_ai.request.model": "text-embedding-004",
            },
            name="embeddings text-embedding-004",
            scope="opentelemetry.util.genai.handler",
        )

    def test_embedding_span_under_langchain_is_stamped_and_consumes_no_order(self):
        span = self._embedding_span()
        tok = _in_langchain.set(True)
        try:
            self.proc.on_start(span)
        finally:
            _in_langchain.reset(tok)
        self.assertTrue(span.attributes.get("tp.suppress"))
        self.assertNotIn("tp.span_order", span.attributes)
        self.proc.on_end(span)
        self.assertEqual(self.client.calls, [])

    def test_standalone_embedding_span_still_dropped_at_on_end(self):
        """No LangChain guard → not stamped, but the embedding gate still drops
        it (the manual wrapper is the authoritative embedding row)."""
        span = self._embedding_span()
        self.assertFalse(_in_langchain.get())
        self.proc.on_start(span)
        self.assertNotIn("tp.suppress", span.attributes)
        self.proc.on_end(span)
        self.assertEqual(
            self.client.calls, [],
            "instrumentor embedding span must never become an llm row",
        )


class TestUtilGenaiStandaloneRowShape(_DedupTestBase):
    """Pin the Mode-A row built for a genuine standalone 1.0b1 user.

    Previously unpinned: the ghost row's shape was only ever observed in
    production. Suppression must not be the only thing keeping this correct.
    """

    def test_standalone_util_genai_span_row_shape(self):
        span = _FakeSpan(
            _util_genai_start_attrs(),
            name="generate_content gemini-2.0-flash",
            scope="opentelemetry.util.genai.handler",
        )
        self.assertFalse(_in_langchain.get())
        self.assertFalse(langchain_trace_active(self.session.trace_id))
        self.proc.on_start(span)
        for k, v in _util_genai_finish_attrs().items():
            span.set_attribute(k, v)
        self.proc.on_end(span)
        self.assertEqual(len(self.client.calls), 1)
        call = self.client.calls[0]
        # gen_ai.system is absent on 1.0b1, so the provider must be resolved
        # from gen_ai.provider.name and normalized to the canonical slug.
        self.assertEqual(call["provider"], "google")
        self.assertEqual(call["usage"]["shape"], "google_genai")
        self.assertEqual(call["model"], "gemini-2.0-flash")
        self.assertEqual(call["input_tokens"], 219)
        self.assertEqual(call["output_tokens"], 46)


class TestIsGoogleGenaiInstrumentorSpan(unittest.TestCase):
    """Direct unit tests for the two-arm predicate."""

    LEGACY = "opentelemetry.instrumentation.google_genai"
    UTIL = "opentelemetry.util.genai.handler"

    def test_arm1_legacy_scope_matches_without_attributes(self):
        """0.7b1 has no provider attribute at on_start — scope is all we get."""
        self.assertTrue(_is_google_genai_instrumentor_span(self.LEGACY, {}))

    def test_arm1_matches_submodule_scope(self):
        self.assertTrue(
            _is_google_genai_instrumentor_span(self.LEGACY + ".generate_content", {})
        )

    def test_arm1_rejects_prefix_without_dot_boundary(self):
        self.assertFalse(
            _is_google_genai_instrumentor_span(self.LEGACY + "_other", {})
        )

    def test_arm2_matches_known_google_providers(self):
        for provider in ("gemini", "vertex_ai", "google_genai", "google"):
            with self.subTest(provider=provider):
                self.assertTrue(
                    _is_google_genai_instrumentor_span(
                        self.UTIL, {"gen_ai.provider.name": provider}
                    )
                )

    def test_arm2_matches_canonical_gcp_spellings(self):
        """Upstream's open span_utils TODO may rename these values."""
        for provider in ("gcp.gemini", "gcp.vertex_ai", "gcp.gen_ai"):
            with self.subTest(provider=provider):
                self.assertTrue(
                    _is_google_genai_instrumentor_span(
                        self.UTIL, {"gen_ai.provider.name": provider}
                    )
                )

    def test_arm2_falls_back_to_legacy_system_attribute(self):
        self.assertTrue(
            _is_google_genai_instrumentor_span(self.UTIL, {"gen_ai.system": "gemini"})
        )

    def test_arm2_matches_bare_root_scope(self):
        self.assertTrue(
            _is_google_genai_instrumentor_span(
                "opentelemetry.util.genai", {"gen_ai.provider.name": "gemini"}
            )
        )

    def test_arm2_tolerates_padding_and_case(self):
        self.assertTrue(
            _is_google_genai_instrumentor_span(
                self.UTIL, {"gen_ai.provider.name": "  GEMINI  "}
            )
        )

    def test_arm2_operation_name_only_when_no_provider(self):
        self.assertTrue(
            _is_google_genai_instrumentor_span(
                self.UTIL, {"gen_ai.operation.name": "generate_content"}
            )
        )

    def test_arm2_provider_veto_beats_operation_fallback(self):
        """The single most important negative — see the matching processor test."""
        self.assertFalse(
            _is_google_genai_instrumentor_span(
                self.UTIL,
                {
                    "gen_ai.provider.name": "openai",
                    "gen_ai.operation.name": "generate_content",
                },
            )
        )

    def test_arm2_rejects_non_google_provider(self):
        for provider in ("openai", "anthropic", "cohere"):
            with self.subTest(provider=provider):
                self.assertFalse(
                    _is_google_genai_instrumentor_span(
                        self.UTIL, {"gen_ai.provider.name": provider}
                    )
                )

    def test_arm2_scope_alone_is_never_sufficient(self):
        self.assertFalse(_is_google_genai_instrumentor_span(self.UTIL, {}))

    def test_arm2_rejects_other_operation_without_provider(self):
        self.assertFalse(
            _is_google_genai_instrumentor_span(
                self.UTIL, {"gen_ai.operation.name": "chat"}
            )
        )

    def test_arm2_rejects_dotted_boundary_violation(self):
        self.assertFalse(
            _is_google_genai_instrumentor_span(
                "opentelemetry.util.genai_other", {"gen_ai.provider.name": "gemini"}
            )
        )

    def test_google_provider_on_unrelated_scope_is_rejected(self):
        self.assertFalse(
            _is_google_genai_instrumentor_span(
                "opentelemetry.instrumentation.openai",
                {"gen_ai.provider.name": "gemini"},
            )
        )

    def test_empty_and_none_scope(self):
        self.assertFalse(_is_google_genai_instrumentor_span("", {}))
        self.assertFalse(_is_google_genai_instrumentor_span(None, {}))  # type: ignore[arg-type]
        self.assertFalse(_is_google_genai_instrumentor_span(self.UTIL, None))  # type: ignore[arg-type]

    def test_never_raises_on_hostile_inputs(self):
        """Golden rule: the SDK must never throw into customer code."""

        class _BoomAttrs:
            def get(self, *_a, **_k):
                raise RuntimeError("boom")

        class _BoomScope:
            def __str__(self):
                raise RuntimeError("boom")

        try:
            self.assertFalse(
                _is_google_genai_instrumentor_span(self.UTIL, _BoomAttrs())  # type: ignore[arg-type]
            )
            self.assertFalse(
                _is_google_genai_instrumentor_span(_BoomScope(), {})  # type: ignore[arg-type]
            )
        except Exception as exc:  # pragma: no cover - the assertion is the point
            self.fail(f"predicate must never raise, got {exc!r}")


class TestLangchainTraceRegistryApi(_DedupTestBase):
    """Registry API contract — scope-independent, so it lives outside the
    parametrised classes (it would otherwise run once per scope leg)."""

    def test_registry_apis_never_raise_on_bad_input(self):
        """Golden rule: enter/leave/active must swallow bad inputs."""
        for bad in (None, 123, b"x", [], {}):
            enter_langchain_trace(bad)  # type: ignore[arg-type]
            leave_langchain_trace(bad)  # type: ignore[arg-type]
            self.assertFalse(langchain_trace_active(bad))  # type: ignore[arg-type]
        # Over-leave clamps cleanly.
        leave_langchain_trace("no-such-trace")
        self.assertFalse(langchain_trace_active("no-such-trace"))


if __name__ == "__main__":
    unittest.main()
