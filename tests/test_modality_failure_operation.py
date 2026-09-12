"""Failed manual modality calls must keep modality as `operation`.

Success path (`_log_modality`) already sends `operation=modality`. The failure
path stashed only registry `operation` (None for modality-only rows), so
`_emit_call_failure_log` defaulted to `operation="chat"` and the server
derived `span_kind="llm"`. Fix: stash `operation or modality`.

Asserts:
  - image_gen / audio_tts failures log the modality operation
  - explicit embedding operation still wins over a null modality
  - no modality → still defaults to "chat"
  - original exception still propagates (GOLDEN RULE)
"""
from __future__ import annotations

import unittest
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession


class _StatusError(Exception):
    def __init__(self, msg="Imagen 3 is only available on paid plans", status_code=400):
        super().__init__(msg)
        self.status_code = status_code


def _install_and_call(
    *,
    modality=None,
    operation=None,
    model="imagen-4.0-generate-001",
    is_async=False,
):
    """Install a one-shot manual wrapper on a throwaway class, call it once,
    capture log_sync kwargs from the failure path. Returns (captured, exc)."""
    import asyncio

    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    if is_async:
        class _FakeTarget:
            async def generate(self, *args, **kwargs):
                raise _StatusError()
    else:
        class _FakeTarget:
            def generate(self, *args, **kwargs):
                raise _StatusError()

    original = _FakeTarget.generate
    enforcer._set_manual_wrapper(
        _FakeTarget,
        "generate",
        original,
        provider="google",
        is_async=is_async,
        modality=modality,
        shape="google_imagen" if modality == "image_gen" else None,
        operation=operation,
    )

    session = TPSession()
    session.user_id = "u"
    session.paid_plan = "free"
    session.workflow_name = "b12"
    session.session_id = "s"
    session.metadata = {}

    raised = None
    check_patch = (
        mock.patch("token_police.enforcer._run_async_check", new=mock.AsyncMock(return_value=None))
        if is_async
        else mock.patch("token_police.enforcer._run_sync_check", return_value=None)
    )
    try:
        with mock.patch("token_police.enforcer.get_current_session", return_value=session), \
             mock.patch("token_police.enforcer.get_client", return_value=_FakeTP()), \
             check_patch, \
             mock.patch("token_police.enforcer.in_langchain", return_value=False), \
             mock.patch("token_police.enforcer.in_litellm", return_value=False), \
             mock.patch("token_police.enforcer.in_llamaindex", return_value=False), \
             mock.patch("token_police.enforcer.in_pydantic_ai", return_value=False), \
             mock.patch("token_police.enforcer.in_agno", return_value=False), \
             mock.patch("token_police.enforcer.maybe_register_openai_agents_tracing"), \
             mock.patch("token_police.enforcer.consume_pending_span_name", return_value=None), \
             mock.patch("token_police.enforcer._capture_composition_at"), \
             mock.patch("token_police.enforcer._build_intent", return_value=None):
            try:
                if is_async:
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(_FakeTarget().generate(model=model))
                    finally:
                        loop.close()
                else:
                    _FakeTarget().generate(model=model)
            except Exception as exc:
                raised = exc
    finally:
        # Restore class + dedup map so later tests / re-runs stay clean.
        _FakeTarget.generate = original
        enforcer._originals.pop((_FakeTarget, "generate"), None)

    return captured, raised


class TestModalityFailureOperation(unittest.TestCase):
    def test_image_gen_failure_logs_image_gen(self):
        captured, raised = _install_and_call(modality="image_gen")
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "image_gen")
        self.assertEqual(captured.get("model"), "imagen-4.0-generate-001")
        self.assertEqual(captured.get("call_outcome", {}).get("http_status"), 400)

    def test_audio_tts_failure_logs_audio_tts(self):
        captured, raised = _install_and_call(
            modality="audio_tts", model="gpt-4o-mini-tts"
        )
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "audio_tts")

    def test_explicit_embedding_operation_unchanged(self):
        captured, raised = _install_and_call(
            modality=None, operation="embedding", model="text-embedding-004"
        )
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "embedding")

    def test_no_modality_defaults_to_chat(self):
        captured, raised = _install_and_call(
            modality=None, operation=None, model="gpt-4.1-mini"
        )
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "chat")

    def test_operation_wins_over_modality_when_both_set(self):
        # Defensive: if a row ever carried both, registry operation is primary.
        captured, raised = _install_and_call(
            modality="image_gen", operation="embedding", model="weird"
        )
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "embedding")

    def test_async_image_gen_failure_logs_image_gen(self):
        # Async stash site is a twin of the sync expression — cover both.
        captured, raised = _install_and_call(modality="image_gen", is_async=True)
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "image_gen")


if __name__ == "__main__":
    unittest.main()
