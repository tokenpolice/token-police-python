"""A failed CHAT / modality call must not ship a synthesized OpenAI shape.

N4 fixed failed EMBEDDING rows only; every other failed row still passed
`usage=None` to `log_sync`, so `client.py` synthesized
`{"shape": "openai_compatible_chat", ...}`. In the 2026-07-27 real-apps run a
503-failed huggingface row carried that synth while its 17 successful siblings
on the same client carried `huggingface_chat` (cohere → `cohere_chat`,
google → `google_genai` likewise).

Asserts on `_emit_call_failure_log`:
  - the provider chat table drives the shape (cohere / huggingface / google /
    openai_responses)
  - an unmapped or framework provider still falls back to
    `openai_compatible_chat`
  - the registry `shape` override wins outright (modality rows)
  - the wire-key gate engages ONLY for a serving slug absent from the
    chat table (Anthropic SDK → MiniMax base_url keeps `anthropic_messages`)
  - counts stay zero (the row stays unmeasured)
  - shape resolution blowing up degrades to no usage block, and the customer's
    exception still propagates unmasked (GOLDEN RULE)
"""
from __future__ import annotations

import unittest
from contextlib import ExitStack
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession


class _StatusError(Exception):
    def __init__(self, msg="Service Unavailable", status_code=503):
        super().__init__(msg)
        self.status_code = status_code


def _session(**attempted):
    session = TPSession()
    session.user_id = "u"
    session.paid_plan = "free"
    session.workflow_name = "f1"
    session.session_id = "s"
    session.metadata = {}
    session._call_outcome = {"error_kind": "provider_error", "http_status": 503}
    session._attempted_model = attempted.get("model", "some-model")
    session._attempted_provider = attempted.get("provider", "")
    session._attempted_operation = attempted.get("operation")
    session._attempted_shape = attempted.get("shape")
    session._attempted_wire_key = attempted.get("wire_key")
    return session


def _emit(**attempted):
    """Drive the failure funnel directly and capture the log_sync kwargs."""
    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    session = _session(**attempted)
    with mock.patch("token_police.enforcer._state.drain_observations", return_value=[]):
        enforcer._emit_call_failure_log(_FakeTP(), session)
    return captured, session


class TestFailedChatUsageShape(unittest.TestCase):
    def test_provider_chat_shapes(self):
        for provider, expected in (
            ("cohere", "cohere_chat"),
            ("huggingface", "huggingface_chat"),
            ("google", "google_genai"),
            ("openai_responses", "openai_responses"),
        ):
            with self.subTest(provider=provider):
                captured, _ = _emit(provider=provider)
                usage = captured.get("usage")
                self.assertIsInstance(usage, dict)
                self.assertEqual(usage.get("shape"), expected)
                # Zero counts: a failed call has no usage — row stays unmeasured.
                self.assertEqual(usage.get("raw"),
                                 {"prompt_tokens": 0, "total_tokens": 0})

    def test_unmapped_and_framework_providers_keep_compat_default(self):
        for provider in ("", "langchain", "some-new-vendor"):
            with self.subTest(provider=provider):
                captured, _ = _emit(provider=provider)
                self.assertEqual((captured.get("usage") or {}).get("shape"),
                                 "openai_compatible_chat")

    def test_registry_shape_override_wins(self):
        captured, _ = _emit(provider="openai", operation="image_gen",
                            shape="openai_images")
        self.assertEqual(captured.get("operation"), "image_gen")
        self.assertEqual((captured.get("usage") or {}).get("shape"), "openai_images")

    def test_wire_key_gate_engages_for_remapped_host(self):
        # Anthropic SDK → api.minimax.io: serving slug "minimax" is absent from
        # the chat table, so the module wire slug supplies the shape.
        captured, _ = _emit(provider="minimax", wire_key="anthropic")
        self.assertEqual((captured.get("usage") or {}).get("shape"),
                         "anthropic_messages")

    def test_remapped_host_without_wire_key_keeps_compat_default(self):
        captured, _ = _emit(provider="minimax")
        self.assertEqual((captured.get("usage") or {}).get("shape"),
                         "openai_compatible_chat")

    def test_wire_key_never_overrides_an_in_table_provider(self):
        # A serving slug that IS in the table must be untouched by the gate.
        captured, _ = _emit(provider="cohere", wire_key="openai")
        self.assertEqual((captured.get("usage") or {}).get("shape"), "cohere_chat")

    def test_attempt_context_cleared_after_emit(self):
        _, session = _emit(provider="cohere", shape="openai_images",
                           wire_key="anthropic")
        self.assertIsNone(session._attempted_shape)
        self.assertIsNone(session._attempted_wire_key)

    def test_shape_resolution_failure_degrades_to_no_usage_block(self):
        captured = {}

        class _FakeTP:
            def log_sync(self, **kw):
                captured.update(kw)

        session = _session(provider="cohere")
        with mock.patch("token_police.enforcer._state.drain_observations", return_value=[]), \
             mock.patch("token_police.enforcer._resolve_usage_shape",
                        side_effect=RuntimeError("boom")):
            enforcer._emit_call_failure_log(_FakeTP(), session)
        # Row is still logged; only the usage block is dropped (client synth).
        self.assertEqual(captured.get("provider"), "cohere")
        self.assertIsNone(captured.get("usage"))


class TestFailedChatShapeNeverBreaksCustomer(unittest.TestCase):
    """GOLDEN RULE: a broken shape resolution must not mask or replace the
    customer's own exception."""

    def _call_wrapper(self, patch_resolver):
        captured = {}

        class _FakeTP:
            def log_sync(self, **kw):
                captured.update(kw)

        class _FakeTarget:
            def create(self, *args, **kwargs):
                raise _StatusError()

        original = _FakeTarget.create
        enforcer._set_manual_wrapper(
            _FakeTarget, "create", original, provider="cohere", is_async=False,
        )
        session = _session(provider="cohere")
        session._attempted_provider = ""
        raised = None
        stack = [
            mock.patch("token_police.enforcer.get_current_session", return_value=session),
            mock.patch("token_police.enforcer.get_client", return_value=_FakeTP()),
            mock.patch("token_police.enforcer._run_sync_check", return_value=None),
            mock.patch("token_police.enforcer.in_langchain", return_value=False),
            mock.patch("token_police.enforcer.in_litellm", return_value=False),
            mock.patch("token_police.enforcer.in_llamaindex", return_value=False),
            mock.patch("token_police.enforcer.in_pydantic_ai", return_value=False),
            mock.patch("token_police.enforcer.in_agno", return_value=False),
            mock.patch("token_police.enforcer.maybe_register_openai_agents_tracing"),
            mock.patch("token_police.enforcer.consume_pending_span_name", return_value=None),
            mock.patch("token_police.enforcer._capture_composition_at"),
        ]
        if patch_resolver:
            stack.append(mock.patch("token_police.enforcer._resolve_usage_shape",
                                    side_effect=RuntimeError("boom")))
        try:
            with ExitStack() as patches:
                for ctx in stack:
                    patches.enter_context(ctx)
                try:
                    _FakeTarget().create(model="command-r")
                except Exception as exc:
                    raised = exc
        finally:
            _FakeTarget.create = original
            enforcer._originals.pop((_FakeTarget, "create"), None)
        return captured, raised

    def test_customer_exception_propagates_with_working_resolver(self):
        captured, raised = self._call_wrapper(patch_resolver=False)
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(str(raised), "Service Unavailable")
        self.assertEqual((captured.get("usage") or {}).get("shape"), "cohere_chat")

    def test_customer_exception_propagates_when_resolver_throws(self):
        captured, raised = self._call_wrapper(patch_resolver=True)
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(str(raised), "Service Unavailable")
        self.assertIsNone(captured.get("usage"))


if __name__ == "__main__":
    unittest.main()
