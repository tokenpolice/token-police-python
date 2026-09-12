"""N4 — a failed embedding call must not ship a CHAT usage shape.

`_emit_call_failure_log` passed no `usage=` block, so `client.log_sync`
synthesized `{"shape": "openai_compatible_chat", ...}` unconditionally and
every failed embedding row landed with `usage_shape='openai_compatible_chat'`
on an embedding span (5/5 failed embedding rows in the 2026-07-27 real-apps
run), while successful siblings carried `voyage_embed` / `openai_embeddings`.

Asserts:
  - native embedding failure (voyage) → usage.shape == "voyage_embed"
  - unmapped/framework provider → usage.shape == "openai_embeddings"
  - counts stay zero (a failed call has no usage; the row stays unmeasured)
  - failed CHAT / modality calls resolve their own real shape too
  - the original exception still propagates (GOLDEN RULE)
"""
from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession


class _StatusError(Exception):
    def __init__(self, msg="Too Many Requests", status_code=429):
        super().__init__(msg)
        self.status_code = status_code


def _install_and_call(
    *,
    provider="voyage",
    operation="embedding",
    modality=None,
    model="voyage-3",
    is_async=False,
):
    """Install a one-shot manual wrapper on a throwaway class, call it once,
    capture the failure-path log_sync kwargs. Returns (captured, exc)."""
    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    if is_async:
        class _FakeTarget:
            async def embed(self, *args, **kwargs):
                raise _StatusError()
    else:
        class _FakeTarget:
            def embed(self, *args, **kwargs):
                raise _StatusError()

    original = _FakeTarget.embed
    enforcer._set_manual_wrapper(
        _FakeTarget,
        "embed",
        original,
        provider=provider,
        is_async=is_async,
        modality=modality,
        operation=operation,
    )

    session = TPSession()
    session.user_id = "u"
    session.paid_plan = "free"
    session.workflow_name = "n4"
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
                    # Own loop, not asyncio.get_event_loop(): under the full
                    # suite an earlier test can leave the main-thread loop
                    # closed/unset, which is unrelated to what this asserts.
                    loop = asyncio.new_event_loop()
                    try:
                        loop.run_until_complete(_FakeTarget().embed(model=model))
                    finally:
                        loop.close()
                else:
                    _FakeTarget().embed(model=model)
            except Exception as exc:
                raised = exc
    finally:
        _FakeTarget.embed = original
        enforcer._originals.pop((_FakeTarget, "embed"), None)

    return captured, raised


class TestFailedEmbeddingUsageShape(unittest.TestCase):
    def test_native_voyage_failure_ships_voyage_embed_shape(self):
        captured, raised = _install_and_call(provider="voyage")
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "embedding")
        usage = captured.get("usage")
        self.assertIsInstance(usage, dict)
        self.assertEqual(usage.get("shape"), "voyage_embed")
        # Zero counts: a failed call has no usage — the row stays unmeasured.
        self.assertEqual(usage.get("raw"), {"prompt_tokens": 0, "total_tokens": 0})
        self.assertEqual(captured.get("input_tokens", 0) or 0, 0)
        self.assertEqual(captured.get("output_tokens", 0) or 0, 0)

    def test_async_embedding_failure_ships_embed_shape(self):
        captured, raised = _install_and_call(provider="cohere", is_async=True)
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual((captured.get("usage") or {}).get("shape"), "cohere_embed")

    def test_openai_embedding_failure_ships_openai_embeddings(self):
        captured, _ = _install_and_call(provider="openai", model="text-embedding-3-small")
        self.assertEqual((captured.get("usage") or {}).get("shape"), "openai_embeddings")

    def test_unmapped_provider_falls_back_to_openai_embeddings(self):
        # Framework paths (langchain / llamaindex) stash the framework as the
        # provider; their success siblings log openai_embeddings too.
        for prov in ("langchain", "llamaindex", "", "some-new-vendor"):
            with self.subTest(provider=prov):
                captured, _ = _install_and_call(provider=prov)
                self.assertEqual(
                    (captured.get("usage") or {}).get("shape"), "openai_embeddings"
                )

    def test_failed_chat_call_ships_resolved_chat_shape(self):
        """Chat failures resolve the provider's chat shape, not the synth."""
        captured, raised = _install_and_call(
            provider="openai", operation=None, model="gpt-4.1-mini"
        )
        self.assertIsInstance(raised, _StatusError)
        self.assertEqual(captured.get("operation"), "chat")
        self.assertEqual((captured.get("usage") or {}).get("shape"), "openai_chat")

    def test_failed_modality_call_without_registry_shape_uses_provider_table(self):
        # No registry `shape` override on this entry → the provider chat table
        # still beats the client's openai_compatible_chat synth.
        captured, _ = _install_and_call(
            provider="google", operation=None, modality="image_gen", model="imagen-4.0"
        )
        self.assertEqual(captured.get("operation"), "image_gen")
        self.assertEqual((captured.get("usage") or {}).get("shape"), "google_genai")


if __name__ == "__main__":
    unittest.main()
