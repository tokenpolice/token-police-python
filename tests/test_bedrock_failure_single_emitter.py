"""Bedrock failure-path fixes F-16-1 / F-16-2 / F-16-3a.

Three independent bugs on the *failed* Bedrock call path:

* **F-16-3a (model stash)** — ``_stash_attempt_context`` only read ``modelId``
  for ``Converse``/``ConverseStream``, so a failed ``InvokeModel`` /
  ``InvokeModelWithResponseStream`` row shipped ``model="unknown"``.

* **F-16-2 (embedding failure parent)** — the failure row for a Bedrock
  embedding call parented onto the *active* OTel span, which on that path is
  the instrumentor's ``invoke_model`` span that the SpanProcessor
  unconditionally drops → a dangling ``parent_span_id``. The new
  ``_bedrock_embedding_failure_span`` mirrors the SUCCESS twin's construction
  (session trace id, fresh span id, ``_session_parent_span_id`` parent,
  span_kind ``llm``, span order) and is threaded through the new
  ``span_override=`` kwarg; every other ``_emit_call_failure_log`` call site
  leaves it ``None`` (byte-identical old behavior).

* **F-16-1 (duplicate chat failure row)** — unlike every other provider, the
  OTel Bedrock instrumentor patches the per-client methods OUTSIDE the
  enforcer wrapper, so its span is still open when the wrapper's ``except``
  runs and ends errored right after the re-raise, emitting its own failure
  row. ``_bedrock_chat_failure_handoff`` makes the enforcer hand its payload
  to that span (``session._pending_bedrock_failure``) and skip its own emit;
  ``telemetry.on_end`` merges the stash into the surviving row;
  ``_flush_unconsumed_bedrock_failure`` (called at the top of the next
  ``_stash_attempt_context``) emits a late standalone row if the span never
  came — a late row beats a lost row.

GOLDEN RULE throughout: every new helper is fail-open (returns None/False on
hostile input, never raises) and the customer's exception propagates unchanged.

Harness idioms mirror tests/test_bedrock_body_restore.py (tp.init + mocked
check/log, handlers driven directly inside a ``tp.session``),
tests/test_failure_log_composition.py (``on_end`` driven over a fabricated
span with a patched session) and tests/test_stream_usage_retry_net.py
(``_wrap_method(..., override_module=Fake)`` for the real wrapper).
"""

import asyncio
import unittest
from types import SimpleNamespace as NS
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

from opentelemetry.trace import StatusCode

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as tp_state
from token_police.context import (
    TPSession,
    get_current_session,
    _session_parent_span_id,
)
from token_police.telemetry import TokenPoliceSpanProcessor


_OUTCOME = {"status": "failed", "duration_ms": 12, "error_kind": "auth_error",
            "http_status": 403, "error_class": "AccessDeniedException"}


def _instrumented_fn():
    """A function whose ``__globals__["__name__"]`` is the bedrock
    instrumentor's module — the exact signal ``_bedrock_chat_failure_handoff``
    uses to detect that the client method has been patched. (``@wraps`` copies
    ``__module__`` off the wrapped botocore method, so ``__module__`` is
    useless here; the defining module is only visible via ``__globals__``.)"""
    g = {"__name__": "opentelemetry.instrumentation.bedrock"}
    exec("def _patched(*a, **k):\n    return None\n", g)
    return g["_patched"]


def _plain_fn(*a, **k):
    """Defined in THIS module → __globals__['__name__'] is the test module."""
    return None


class _Meta:
    def __init__(self, service="bedrock-runtime"):
        self.service_model = NS(service_name=service)


class _FakeBedrockClient:
    """botocore client stand-in. ``instrumented`` decides whether the four LLM
    methods look like the OTel instrumentor's patch closures."""

    _LLM_METHODS = ("converse", "converse_stream", "invoke_model",
                    "invoke_model_with_response_stream")

    def __init__(self, instrumented=True, service="bedrock-runtime"):
        self.meta = _Meta(service)
        for m in self._LLM_METHODS:
            setattr(self, m, _instrumented_fn() if instrumented else _plain_fn)


# ──────────────────────────────────────────────────────────────────────────
# F-16-3a — modelId stash for all four LLM ops
# ──────────────────────────────────────────────────────────────────────────
class TestStashAttemptContextModel(unittest.TestCase):
    def _stash(self, op, api_params={"modelId": "amazon.titan-embed-text-v2:0"},
               module="botocore.client"):
        session = TPSession()
        _enforcer._stash_attempt_context(session, "bedrock", module,
                                         ("self-placeholder", op, api_params), {})
        return session

    def test_invoke_model_model_captured(self):
        s = self._stash("InvokeModel")
        self.assertEqual(s._attempted_model, "amazon.titan-embed-text-v2:0")
        self.assertEqual(s._attempted_provider, "bedrock")

    def test_invoke_model_with_response_stream_model_captured(self):
        s = self._stash("InvokeModelWithResponseStream",
                        {"modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0"})
        self.assertEqual(s._attempted_model,
                         "anthropic.claude-3-5-sonnet-20241022-v2:0")

    def test_converse_still_captured(self):
        s = self._stash("Converse", {"modelId": "amazon.nova-lite-v1:0"})
        self.assertEqual(s._attempted_model, "amazon.nova-lite-v1:0")

    def test_converse_stream_still_captured(self):
        s = self._stash("ConverseStream", {"modelId": "amazon.nova-pro-v1:0"})
        self.assertEqual(s._attempted_model, "amazon.nova-pro-v1:0")

    def test_non_llm_op_stays_none(self):
        # ApplyGuardrail carries no modelId — must not be widened accidentally.
        s = self._stash("ApplyGuardrail", {"guardrailIdentifier": "g-1"})
        self.assertIsNone(s._attempted_model)

    def test_aiobotocore_module_also_covered(self):
        s = self._stash("InvokeModel", {"modelId": "cohere.embed-english-v3"},
                        module="aiobotocore.client")
        self.assertEqual(s._attempted_model, "cohere.embed-english-v3")

    def test_non_dict_api_params_is_safe(self):
        s = self._stash("InvokeModel", "not-a-dict")
        self.assertIsNone(s._attempted_model)

    def test_missing_api_params_is_safe(self):
        session = TPSession()
        _enforcer._stash_attempt_context(session, "bedrock", "botocore.client",
                                         ("self", "InvokeModel"), {})
        self.assertIsNone(session._attempted_model)

    def test_non_botocore_module_still_reads_kwargs_model(self):
        session = TPSession()
        _enforcer._stash_attempt_context(session, "openai", "openai.resources",
                                         (), {"model": "gpt-4.1-mini"})
        self.assertEqual(session._attempted_model, "gpt-4.1-mini")


# ──────────────────────────────────────────────────────────────────────────
# F-16-2 — embedding failure span (unit)
# ──────────────────────────────────────────────────────────────────────────
class TestBedrockEmbeddingFailureSpan(unittest.TestCase):
    def _anchored_session(self):
        s = TPSession(root_span_id="c" * 16)
        s._anchored = True
        return s

    def test_anchored_session_parents_onto_root(self):
        from datetime import datetime, timezone
        s = self._anchored_session()
        span = _enforcer._bedrock_embedding_failure_span(
            s, "titan-embed", datetime.now(timezone.utc))
        self.assertEqual(span["parent_span_id"], "c" * 16)
        self.assertEqual(span["parent_span_id"], _session_parent_span_id(s))
        self.assertEqual(span["trace_id"], s.trace_id)
        self.assertEqual(span["span_kind"], "llm")
        self.assertEqual(span["span_name"], "titan-embed")
        self.assertEqual(len(span["span_id"]), 16)
        self.assertNotEqual(span["span_id"], span["parent_span_id"])
        self.assertIsNotNone(span["start_time"])
        self.assertIsNotNone(span["end_time"])

    def test_unanchored_session_parents_empty(self):
        from datetime import datetime, timezone
        s = TPSession()  # throwaway root that was never logged
        span = _enforcer._bedrock_embedding_failure_span(
            s, "titan-embed", datetime.now(timezone.utc))
        self.assertEqual(span["parent_span_id"], "")

    def test_span_order_consumes_session_counter(self):
        from datetime import datetime, timezone
        s = self._anchored_session()
        a = _enforcer._bedrock_embedding_failure_span(s, "e", datetime.now(timezone.utc))
        b = _enforcer._bedrock_embedding_failure_span(s, "e", datetime.now(timezone.utc))
        self.assertEqual(a["span_order"], 0)
        self.assertEqual(b["span_order"], 1)

    def test_span_name_falls_back_to_workflow_name(self):
        from datetime import datetime, timezone
        s = self._anchored_session()
        s.workflow_name = "embed_wf"
        span = _enforcer._bedrock_embedding_failure_span(
            s, None, datetime.now(timezone.utc))
        self.assertEqual(span["span_name"], "embed_wf")

    def test_null_start_time_is_safe(self):
        s = self._anchored_session()
        span = _enforcer._bedrock_embedding_failure_span(s, "e", None)
        self.assertIsNone(span["start_time"])

    def test_hostile_session_returns_none_never_raises(self):
        class _Hostile:
            @property
            def trace_id(self):
                raise RuntimeError("nope")

        self.assertIsNone(
            _enforcer._bedrock_embedding_failure_span(_Hostile(), "e", None))
        self.assertIsNone(_enforcer._bedrock_embedding_failure_span(None, "e", None))


# ──────────────────────────────────────────────────────────────────────────
# F-16-2 — embedding failure row, integration through the real handlers
# ──────────────────────────────────────────────────────────────────────────
def _embed_args():
    """botocore _make_api_call(self, operation_name, api_params)."""
    return (MagicMock(), "InvokeModel",
            {"modelId": "amazon.titan-embed-text-v1", "body": b'{"inputText":"hi"}'})


def _failed_rows(log_sync_mock):
    """The FAILED-call rows only.

    Re-calling tp.init() per test piggybacks an extra span processor onto the
    (do-once) global tracer provider, so the structural agent/chain row for the
    surrounding tp.session() may be emitted several times (same idiom noted in
    tests/test_structural_span_status.py). Those rows are status='success';
    the failure row under test is the only status='failed' one."""
    return [c.kwargs for c in log_sync_mock.call_args_list
            if (c.kwargs.get("call_outcome") or {}).get("status") == "failed"]


class TestBedrockEmbeddingFailureRowIntegration(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()
        self.client = tp.init(api_key="tp_sk_test_f162", firewall="off")
        self.client.check_sync = MagicMock(return_value={"status": "allowed"})
        self.client.check = AsyncMock(return_value={"status": "allowed"})
        self.client.log_sync = MagicMock()

    def tearDown(self):
        tp_state.reset_pack()
        tp.uninstrument()

    def test_sync_failure_row_parents_onto_session_anchor(self):
        boom = RuntimeError("AccessDeniedException: not authorized")
        original = MagicMock(side_effect=boom)
        with tp.session(name="embed_wf"):
            session = get_current_session()
            self.assertTrue(session._anchored)  # assertion below is meaningful
            with self.assertRaises(RuntimeError) as ctx:
                _enforcer._handle_bedrock_embedding_sync(original, _embed_args(), {})
            # GOLDEN RULE: the customer's exact exception object propagates.
            self.assertIs(ctx.exception, boom)
            anchor = _session_parent_span_id(session)

        rows = _failed_rows(self.client.log_sync)
        self.assertEqual(len(rows), 1)
        span = rows[0]["span"]
        self.assertEqual(span["parent_span_id"], anchor)
        # Discriminator vs the OLD default span (manual_span_ids + span_name
        # only): the override carries the success twin's llm kind + order.
        self.assertEqual(span["span_kind"], "llm")
        self.assertIn("span_order", span)
        self.assertEqual(span["trace_id"], session.trace_id)
        # F-16-3a rides along: the failed embedding row knows its model.
        self.assertEqual(rows[0]["model"], "amazon.titan-embed-text-v1")

    def test_async_failure_row_parents_onto_session_anchor(self):
        boom = RuntimeError("ThrottlingException")
        original = AsyncMock(side_effect=boom)
        holder = {}

        async def _run():
            with tp.session(name="embed_wf"):
                session = get_current_session()
                holder["anchor"] = _session_parent_span_id(session)
                holder["session"] = session
                with self.assertRaises(RuntimeError) as ctx:
                    await _enforcer._handle_bedrock_embedding_async(
                        original, _embed_args(), {})
                self.assertIs(ctx.exception, boom)

        asyncio.run(_run())
        rows = _failed_rows(self.client.log_sync)
        self.assertEqual(len(rows), 1)
        span = rows[0]["span"]
        self.assertEqual(span["parent_span_id"], holder["anchor"])
        self.assertEqual(span["span_kind"], "llm")

    def test_other_call_sites_keep_default_span(self):
        # span_override defaults to None → the untouched manual_span_ids build
        # (no span_kind / span_order keys) for the other 12 call sites.
        session = TPSession()
        session._call_outcome = dict(_OUTCOME)
        captured = {}

        class _FakeTP:
            def log_sync(self, **kw):
                captured.update(kw)

        _enforcer._emit_call_failure_log(_FakeTP(), session)
        self.assertNotIn("span_kind", captured["span"])
        self.assertNotIn("span_order", captured["span"])
        self.assertEqual(captured["span"]["span_name"], session.workflow_name)


# ──────────────────────────────────────────────────────────────────────────
# F-16-1 — handoff gate
# ──────────────────────────────────────────────────────────────────────────
class TestBedrockChatFailureHandoffGate(unittest.TestCase):
    def setUp(self):
        tp_state.drain_observations()  # clear the global observation pool

    def tearDown(self):
        tp_state.drain_observations()

    def _session(self):
        s = TPSession()
        s._call_outcome = dict(_OUTCOME)
        s._attempted_model = "amazon.nova-lite-v1:0"
        s._attempted_provider = "bedrock"
        s._attempted_operation = "chat"
        s._attempted_shape = None
        s._attempted_wire_key = "bedrock"
        return s

    def _call(self, op, *, session=None, instrumented=True, is_async=False,
              module="botocore.client", args=None, obs_key=None):
        client = _FakeBedrockClient(instrumented=instrumented)
        if args is None:
            args = (client, op, {"modelId": "amazon.nova-lite-v1:0"})
        return _enforcer._bedrock_chat_failure_handoff(
            session, module, args, obs_key, is_async)

    # (a) happy path — patched client, sync Converse
    def test_patched_client_hands_off_and_stashes(self):
        s = self._session()
        tp_state.push_observation({"type": "would_block", "rule_id": "r1"})
        self.assertTrue(self._call("Converse", session=s))
        stash = s._pending_bedrock_failure
        self.assertIsInstance(stash, dict)
        self.assertEqual(stash["op"], "Converse")
        self.assertEqual(stash["call_outcome"], _OUTCOME)
        self.assertEqual(stash["model"], "amazon.nova-lite-v1:0")
        self.assertEqual(stash["provider"], "bedrock")
        self.assertEqual([o.get("rule_id") for o in stash["observations"]], ["r1"])
        self.assertIn("comp_key", stash)
        # Enforcer-side state cleared so it can't leak into a later flush/row.
        self.assertIsNone(s._call_outcome)
        self.assertIsNone(s._attempted_model)
        self.assertIsNone(s._attempted_provider)
        self.assertIsNone(s._attempted_operation)
        self.assertIsNone(s._attempted_wire_key)

    def test_all_four_llm_ops_hand_off_sync(self):
        for op in ("Converse", "ConverseStream", "InvokeModel",
                   "InvokeModelWithResponseStream"):
            with self.subTest(op=op):
                s = self._session()
                self.assertTrue(self._call(op, session=s))
                self.assertEqual(s._pending_bedrock_failure["op"], op)

    # (b) unpatched client → no handoff, old emit path
    def test_unpatched_client_returns_false(self):
        s = self._session()
        self.assertFalse(self._call("Converse", session=s, instrumented=False))
        self.assertIsNone(getattr(s, "_pending_bedrock_failure", None))
        # Enforcer state untouched so _emit_call_failure_log still has context.
        self.assertEqual(s._call_outcome, _OUTCOME)
        self.assertEqual(s._attempted_model, "amazon.nova-lite-v1:0")

    def test_client_without_the_method_returns_false(self):
        class _Bare:
            meta = _Meta()

        s = self._session()
        self.assertFalse(_enforcer._bedrock_chat_failure_handoff(
            s, "botocore.client", (_Bare(), "Converse", {}), None, False))

    # (c) non-LLM op → no instrumentor span exists → keep emitting
    def test_non_llm_op_returns_false(self):
        s = self._session()
        self.assertFalse(self._call("ApplyGuardrail", session=s))
        self.assertIsNone(getattr(s, "_pending_bedrock_failure", None))

    # (d) aiobotocore streaming exception
    def test_async_streaming_ops_return_false(self):
        for op in ("ConverseStream", "InvokeModelWithResponseStream"):
            with self.subTest(op=op):
                s = self._session()
                self.assertFalse(self._call(op, session=s, is_async=True,
                                            module="aiobotocore.client"))
                self.assertIsNone(getattr(s, "_pending_bedrock_failure", None))

    def test_async_non_streaming_ops_hand_off(self):
        for op in ("Converse", "InvokeModel"):
            with self.subTest(op=op):
                s = self._session()
                self.assertTrue(self._call(op, session=s, is_async=True,
                                           module="aiobotocore.client"))
                self.assertEqual(s._pending_bedrock_failure["op"], op)

    # (e) wrong module
    def test_non_botocore_module_returns_false(self):
        s = self._session()
        self.assertFalse(self._call("Converse", session=s, module="openai"))

    # (f) degenerate inputs — never raise
    def test_empty_args_returns_false(self):
        s = self._session()
        self.assertFalse(_enforcer._bedrock_chat_failure_handoff(
            s, "botocore.client", (), None, False))

    def test_none_session_returns_false(self):
        self.assertFalse(self._call("Converse", session=None))

    def test_hostile_session_returns_false_never_raises(self):
        class _Hostile:
            @property
            def trace_id(self):
                raise RuntimeError("nope")

            def __setattr__(self, k, v):
                raise RuntimeError("nope")

        self.assertFalse(self._call("Converse", session=_Hostile()))

    def test_none_args_returns_false(self):
        s = self._session()
        self.assertFalse(_enforcer._bedrock_chat_failure_handoff(
            s, "botocore.client", None, None, False))


# ──────────────────────────────────────────────────────────────────────────
# F-16-1 — on_end merge (telemetry)
# ──────────────────────────────────────────────────────────────────────────
NS_START = 1_000_000_000
NS_END = 2_000_000_000  # duration_ms == 1000


def _bedrock_span(session, order, *, errored=True, error_type="ClientError",
                  model="anthropic.claude-3-5-sonnet-20241022-v2:0"):
    attrs = {
        "gen_ai.system": "aws.bedrock",   # normalized to provider "bedrock"
        "gen_ai.request.model": model,
        "tp.trace_id": session.trace_id,
        "tp.span_order": order,
        "tp.workflow_name": session.workflow_name,
        "tp.session_id": session.session_id,
    }
    if error_type:
        attrs["error.type"] = error_type
    status = NS(status_code=StatusCode.ERROR,
                description="An error occurred (AccessDeniedException)") if errored else None
    return NS(
        attributes=attrs,
        name="bedrock.converse",
        instrumentation_scope=NS(name="opentelemetry.instrumentation.bedrock"),
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        status=status,
        start_time=NS_START,
        end_time=NS_END,
    )


def _run_on_end(session, span):
    """Drive on_end with `session` bound, capturing every log_sync payload."""
    rows = []

    class _FakeClient:
        def log_sync(self, **kw):
            rows.append(kw)

    with mock.patch("token_police.state.get_client", return_value=_FakeClient()), \
         mock.patch("token_police.context.get_current_session",
                    return_value=session):
        TokenPoliceSpanProcessor().on_end(span)
    return rows


class TestOnEndBedrockMerge(unittest.TestCase):
    def setUp(self):
        tp_state.drain_observations()

    def tearDown(self):
        tp_state.drain_observations()

    def _session(self):
        s = TPSession(workflow_name="bedrock_wf")
        s._pending_compositions = {}
        return s

    def test_stash_merged_into_the_single_row(self):
        s = self._session()
        order = s.next_span_order()
        s._pending_compositions[f"{s.trace_id}:{order}"] = {"prompt": [{"role": "user"}]}
        s._pending_bedrock_failure = {
            "op": "Converse",
            "call_outcome": dict(_OUTCOME),
            "observations": [{"type": "would_block", "rule_id": "r9"}],
            "model": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "provider": "bedrock",
            "comp_key": f"{s.trace_id}:{order}",
            "local_decision": None,
        }
        rows = _run_on_end(s, _bedrock_span(s, order))
        self.assertEqual(len(rows), 1)  # exactly one emitter
        co = rows[0]["call_outcome"]
        self.assertEqual(co["status"], "failed")
        # Classified kind beats the span's error.type ("ClientError").
        self.assertEqual(co["error_kind"], "auth_error")
        self.assertEqual(co["http_status"], 403)
        self.assertEqual(co["error_class"], "AccessDeniedException")
        self.assertEqual([o["rule_id"] for o in rows[0]["observations"]], ["r9"])
        # Stash consumed — the fallback flush must not re-emit it.
        self.assertIsNone(s._pending_bedrock_failure)
        # The peeked composition entry is popped by the merge, not left behind.
        self.assertEqual(s._pending_compositions, {})

    def test_unknown_stashed_kind_leaves_span_error_type(self):
        s = self._session()
        order = s.next_span_order()
        s._pending_bedrock_failure = {
            "call_outcome": {"status": "failed", "duration_ms": 5,
                             "error_kind": "unknown", "http_status": 0},
            "observations": [],
        }
        rows = _run_on_end(s, _bedrock_span(s, order, error_type="ClientError"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["call_outcome"]["error_kind"], "ClientError")

    def test_model_filled_only_when_missing(self):
        s = self._session()
        order = s.next_span_order()
        s._pending_bedrock_failure = {
            "call_outcome": dict(_OUTCOME),
            "model": "amazon.nova-lite-v1:0",
            "observations": [],
        }
        # Span already carries a real model → the stash must NOT overwrite it.
        rows = _run_on_end(s, _bedrock_span(s, order, model="amazon.nova-pro-v1:0"))
        self.assertEqual(rows[0]["model"], "amazon.nova-pro-v1:0")

    def test_control_no_stash_payload_is_unchanged(self):
        """Byte-equal control: without a stash the row must be exactly what the
        pre-fix code emitted — only call_outcome/observations may differ once a
        stash is present."""
        s = self._session()
        order = s.next_span_order()
        baseline = _run_on_end(s, _bedrock_span(s, order))
        self.assertEqual(len(baseline), 1)
        self.assertEqual(baseline[0]["call_outcome"]["error_kind"], "ClientError")
        self.assertNotIn("observations", baseline[0])

        s2 = TPSession(workflow_name=s.workflow_name, session_id=s.session_id,
                       trace_id=s.trace_id)
        s2._pending_compositions = {}
        order2 = s2.next_span_order()
        s2._pending_bedrock_failure = {
            "call_outcome": dict(_OUTCOME),
            "observations": [{"type": "would_block", "rule_id": "r9"}],
        }
        merged = _run_on_end(s2, _bedrock_span(s2, order2))
        self.assertEqual(len(merged), 1)
        differing = {k for k in set(baseline[0]) | set(merged[0])
                     if baseline[0].get(k) != merged[0].get(k)}
        self.assertEqual(differing, {"call_outcome", "observations"})

    def test_non_bedrock_errored_span_ignores_stash(self):
        s = self._session()
        order = s.next_span_order()
        s._pending_bedrock_failure = {"call_outcome": dict(_OUTCOME),
                                      "observations": []}
        span = _bedrock_span(s, order)
        span.attributes["gen_ai.system"] = "openai"
        span.instrumentation_scope = NS(name="opentelemetry.instrumentation.openai")
        rows = _run_on_end(s, span)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["call_outcome"]["error_kind"], "ClientError")
        # Untouched → the fallback flush still owns it (no silent loss).
        self.assertIsInstance(s._pending_bedrock_failure, dict)

    def test_successful_bedrock_span_ignores_stash(self):
        s = self._session()
        order = s.next_span_order()
        s._pending_bedrock_failure = {"call_outcome": dict(_OUTCOME),
                                      "observations": []}
        rows = _run_on_end(s, _bedrock_span(s, order, errored=False))
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(rows[0].get("call_outcome", {}).get("error_kind"),
                            "auth_error")
        self.assertIsInstance(s._pending_bedrock_failure, dict)

    def test_hostile_stash_still_emits_the_row(self):
        # A non-dict stash (or one that blows up mid-merge) must never cost the
        # row — on_end is fully guarded.
        s = self._session()
        order = s.next_span_order()
        s._pending_bedrock_failure = "not-a-dict"
        rows = _run_on_end(s, _bedrock_span(s, order))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["call_outcome"]["status"], "failed")


# ──────────────────────────────────────────────────────────────────────────
# F-16-1 — unconsumed-stash fallback flush
# ──────────────────────────────────────────────────────────────────────────
class TestUnconsumedStashFlush(unittest.TestCase):
    def _flush_via_stash_attempt(self, session, provider="openai",
                                 module="openai.resources"):
        """The flush runs at the top of the NEXT wrapped call's
        _stash_attempt_context, whatever provider that call is for."""
        rows = []

        class _FakeTP:
            def log_sync(self, **kw):
                rows.append(kw)

        with mock.patch.object(_enforcer, "get_client", return_value=_FakeTP()):
            _enforcer._stash_attempt_context(session, provider, module, (),
                                             {"model": "gpt-4.1-mini"})
        return rows

    def test_unconsumed_stash_emits_one_late_row(self):
        s = TPSession(root_span_id="d" * 16, workflow_name="bedrock_wf")
        s._anchored = True
        s._pending_compositions = {f"{s.trace_id}:3": {"prompt": [{"role": "user"}]}}
        obs = [{"type": "would_block", "rule_id": "r5"}]
        ld = {"decision": "allow", "rule_id": "r5"}
        s._pending_bedrock_failure = {
            "op": "Converse",
            "call_outcome": dict(_OUTCOME),
            "observations": obs,
            "model": "amazon.nova-lite-v1:0",
            "provider": "bedrock",
            "comp_key": f"{s.trace_id}:3",
            "local_decision": ld,
        }
        s._local_decision = ld  # same object the failed call stashed

        rows = self._flush_via_stash_attempt(s)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["call_outcome"], _OUTCOME)
        self.assertEqual(row["observations"], obs)
        self.assertEqual(row["model"], "amazon.nova-lite-v1:0")
        self.assertEqual(row["provider"], "bedrock")
        self.assertEqual(row["operation"], "chat")
        self.assertEqual(row["local_decision"], ld)
        self.assertEqual(row["prompt_composition"], [{"role": "user"}])
        # Parent comes from the session anchor, never a stale active span.
        self.assertEqual(row["span"]["parent_span_id"], _session_parent_span_id(s))
        self.assertEqual(row["span"]["trace_id"], s.trace_id)
        # Stash cleared → a second wrapped call must not re-emit it.
        self.assertIsNone(s._pending_bedrock_failure)
        self.assertEqual(self._flush_via_stash_attempt(s), [])
        # The failed call's composition entry was consumed, not leaked.
        self.assertEqual(s._pending_compositions, {})
        # ...and the current call's own attempt context was still stashed.
        self.assertEqual(s._attempted_model, "gpt-4.1-mini")

    def test_local_decision_identity_guard(self):
        """The old identity guard lived in `_flush_unconsumed_bedrock_failure`
        (clear the session slot only when it was still the SAME object the
        failed call stashed). That guard is gone: `_flush_unconsumed_bedrock_
        failure` no longer touches the session's decision store at all —
        `_bedrock_chat_failure_handoff` now CLAIMS this call's OWN keyed
        decision (by its own obs_key) up front, when it *builds* the stash.
        The equivalent safety property now lives there: a DIFFERENT
        (concurrent/sibling) call's decision, stashed under a DIFFERENT key,
        must never be swept up by a claim keyed to this call.
        """
        s = TPSession()
        s._call_outcome = dict(_OUTCOME)
        this_call_ld = {"decision": "allow", "rule_id": "this-call"}
        sibling_ld = {"decision": "allow", "rule_id": "sibling"}
        _enforcer._stash_local_decision_entry(s, this_call_ld, "key-a")
        _enforcer._stash_local_decision_entry(s, sibling_ld, "key-b")

        client = _FakeBedrockClient(instrumented=True)
        args = (client, "Converse", {"modelId": "amazon.nova-lite-v1:0"})
        self.assertTrue(_enforcer._bedrock_chat_failure_handoff(
            s, "botocore.client", args, "key-a", False))
        stash = s._pending_bedrock_failure
        self.assertEqual(stash["local_decision"], this_call_ld)
        # The sibling's decision (a different key) is left in the store,
        # reachable by its own eventual drain — never stolen here.
        remaining = [e["ld"] for e in s._local_decisions]
        self.assertEqual(remaining, [sibling_ld])

    def test_same_object_local_decision_is_cleared(self):
        """The handoff's claim is DESTRUCTIVE (mirrors `claimLocalDecision` /
        `_claim_local_decision`): once it has claimed this call's own keyed
        entry into the stash, that entry must no longer be sitting in
        `session._local_decisions` — nothing left for `_flush_unconsumed_
        bedrock_failure` (or a later unrelated call) to double-claim."""
        s = TPSession()
        s._call_outcome = dict(_OUTCOME)
        ld = {"decision": "allow", "rule_id": "same"}
        _enforcer._stash_local_decision_entry(s, ld, "key-a")

        client = _FakeBedrockClient(instrumented=True)
        args = (client, "Converse", {"modelId": "amazon.nova-lite-v1:0"})
        self.assertTrue(_enforcer._bedrock_chat_failure_handoff(
            s, "botocore.client", args, "key-a", False))
        self.assertEqual(s._pending_bedrock_failure["local_decision"], ld)
        self.assertEqual(s._local_decisions, [])

    def test_absent_stash_emits_nothing(self):
        s = TPSession()
        self.assertEqual(self._flush_via_stash_attempt(s), [])

    def test_empty_stash_emits_nothing_and_clears(self):
        s = TPSession()
        s._pending_bedrock_failure = {"op": "Converse", "call_outcome": None,
                                      "observations": [], "local_decision": None,
                                      "model": "amazon.nova-lite-v1:0"}
        self.assertEqual(self._flush_via_stash_attempt(s), [])
        self.assertIsNone(s._pending_bedrock_failure)

    def test_hostile_stash_never_raises(self):
        s = TPSession()
        s._pending_bedrock_failure = "not-a-dict"
        self.assertEqual(self._flush_via_stash_attempt(s), [])
        self.assertIsNone(s._pending_bedrock_failure)

    def test_flush_direct_call_with_none_session_is_safe(self):
        _enforcer._flush_unconsumed_bedrock_failure(None)  # must not raise


# ──────────────────────────────────────────────────────────────────────────
# F-16-1 — single-emitter through the REAL wrapper
# ──────────────────────────────────────────────────────────────────────────
class TestWrapperSingleEmitter(unittest.TestCase):
    """Drives enforcer._wrap_method over a fake botocore client class (the
    idiom from tests/test_stream_usage_retry_net.py) so the real sync_wrapper
    except path runs: patched client → NO enforcer /log + a stash; unpatched
    client → exactly one enforcer /log (today's behavior)."""

    def setUp(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        self.client = tp.init(api_key="tp_sk_test_f161", firewall="off")
        self.client.check_sync = MagicMock(return_value={"status": "allowed"})
        self.client.check = AsyncMock(return_value={"status": "allowed"})
        self.client.log_sync = MagicMock()

    def tearDown(self):
        tp_state.reset_pack()
        tp_state.drain_observations()
        tp.uninstrument()

    def _install(self, boom):
        """Fresh class per test so _wrap_method's `_originals` guard can't skip."""
        class FakeBotoClient(_FakeBedrockClient):
            def _make_api_call(self, operation_name, api_params):
                raise boom

        _enforcer._wrap_method(
            {"module": "botocore.client", "object": "", "method": "_make_api_call",
             "async": False},
            override_module=FakeBotoClient,
        )
        return FakeBotoClient

    def test_patched_client_suppresses_the_enforcer_row(self):
        boom = RuntimeError("AccessDeniedException")
        cls = self._install(boom)
        client = cls(instrumented=True)
        with tp.session(name="bedrock_wf"):
            session = get_current_session()
            with self.assertRaises(RuntimeError) as ctx:
                client._make_api_call("Converse", {"modelId": "amazon.nova-lite-v1:0"})
            self.assertIs(ctx.exception, boom)  # GOLDEN RULE
            # No duplicate row from the enforcer...
            self.assertEqual(_failed_rows(self.client.log_sync), [])
            # ...because the payload was handed to the instrumentor span.
            stash = session._pending_bedrock_failure
            self.assertIsInstance(stash, dict)
            self.assertEqual(stash["op"], "Converse")
            self.assertEqual(stash["model"], "amazon.nova-lite-v1:0")
            self.assertEqual(stash["call_outcome"]["status"], "failed")

    def test_unpatched_client_keeps_emitting_exactly_one_row(self):
        boom = RuntimeError("AccessDeniedException")
        cls = self._install(boom)
        client = cls(instrumented=False)
        with tp.session(name="bedrock_wf"):
            session = get_current_session()
            with self.assertRaises(RuntimeError) as ctx:
                client._make_api_call("Converse", {"modelId": "amazon.nova-lite-v1:0"})
            self.assertIs(ctx.exception, boom)  # GOLDEN RULE
            rows = _failed_rows(self.client.log_sync)
            self.assertEqual(len(rows), 1)
            self.assertIsNone(getattr(session, "_pending_bedrock_failure", None))
            self.assertEqual(rows[0]["model"], "amazon.nova-lite-v1:0")

    def test_non_llm_op_keeps_emitting(self):
        boom = RuntimeError("ValidationException")
        cls = self._install(boom)
        client = cls(instrumented=True)
        with tp.session(name="bedrock_wf"):
            with self.assertRaises(RuntimeError):
                client._make_api_call("ApplyGuardrail", {"guardrailIdentifier": "g1"})
            self.assertEqual(len(_failed_rows(self.client.log_sync)), 1)


if __name__ == "__main__":
    unittest.main()
