"""Tests for the native Groq SDK (`groq`) manual target.

The native `groq` package (`from groq import Groq;
client.chat.completions.create(...)`) is a separate Stainless SDK with its own
httpx client — it is NOT the `openai` package, so the `openai` OpenLLMetry
instrumentor never patches it. Before this fix a native-groq call was BOTH
un-metered (no /log) AND un-enforced (no /check). The fix adds a native groq
manual-tap target mirroring the shipped cerebras/together targets.

`groq` is not installed in this repo's env, so these assertions drive the
enforcer helpers directly with fakes (mirrors tests/test_xai_usage.py).

Pins verified against the published `groq` package (PyPI v1.5.0):
  - resource classes: `groq.resources.chat.completions.{Completions,
    AsyncCompletions}` (openai-style, NOT `CompletionsResource`) with `.create`
    (src/groq/resources/chat/completions.py:32,513).
  - streaming usage: `ChatCompletionChunk` (src/groq/types/chat/
    chat_completion_chunk.py:183) carries usage on the FINAL chunk under
    `chunk.x_groq.usage` (XGroq.usage:170) — top-level `chunk.usage` is
    documented as "null except for the last chunk" and is absent on the Node
    twin (groq-sdk v1.3.0), so the x_groq unwrap is the robust path. Non-stream
    `ChatCompletion.usage` is top-level OpenAI-shaped.
"""
import unittest

from token_police.enforcer import (
    _detect_provider,
    _extract_openai_compatible_usage,
    _extract_raw_usage,
    _chunk_usage,
    _resolve_usage_shape,
    _OPENAI_SHAPED_STREAM_PROVIDERS,
    _TARGET_METHODS,
)


class _FakeUsage:
    """OpenAI-shaped usage object (attribute access, like the groq SDK's
    pydantic CompletionUsage)."""
    def __init__(self, prompt_tokens=0, completion_tokens=0, total_tokens=0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class _FakeResponse:
    """Non-streaming groq ChatCompletion: top-level OpenAI-shaped `.usage`."""
    def __init__(self, model, usage):
        self.model = model
        self.usage = usage


class _FakeXGroq:
    def __init__(self, usage):
        self.usage = usage


class _FakeStreamChunk:
    """groq ChatCompletionChunk final chunk: usage nested under `.x_groq.usage`,
    top-level `.usage` is None (matches the verified groq chunk shape)."""
    def __init__(self, model, x_groq_usage=None, top_usage=None):
        self.model = model
        self.usage = top_usage  # None on real groq chunks
        self.x_groq = _FakeXGroq(x_groq_usage) if x_groq_usage is not None else None


class TestGroqNativeTarget(unittest.TestCase):
    # ── #3: provider detection. RED without the fix (returns ""). ──
    def test_detect_provider_groq(self):
        self.assertEqual(
            _detect_provider("groq.resources.chat.completions"), "groq"
        )
        self.assertEqual(_detect_provider("GROQ.resources"), "groq")
        # no shadow / collision with neighbours
        self.assertEqual(_detect_provider("together.resources.chat.completions"), "together")
        self.assertEqual(_detect_provider("openai.resources.chat.completions"), "openai")

    # ── #4: slug (already-present map entry). ──
    def test_slug_is_groq_chat(self):
        self.assertEqual(_resolve_usage_shape("groq"), "groq_chat")

    # ── #1 + enforcement #8: manual target entries (sync + async). ──
    def test_target_methods_has_manual_groq_entries(self):
        groq_entries = [
            t for t in _TARGET_METHODS
            if "groq" in t.get("module", "") and t.get("manual") is True
        ]
        # sync + async, both against groq.resources.chat.completions.create
        self.assertTrue(any(t["async"] is False for t in groq_entries))
        self.assertTrue(any(t["async"] is True for t in groq_entries))
        for t in groq_entries:
            self.assertEqual(t["method"], "create")
            self.assertEqual(t["module"], "groq.resources.chat.completions")
            self.assertIn(t["object"], ("Completions", "AsyncCompletions"))

    # ── #1 parity: same manual-wrapper shape as cerebras/together. ──
    def test_groq_entry_mirrors_cerebras_together_shape(self):
        def keyset(module):
            e = next(t for t in _TARGET_METHODS if module in t.get("module", ""))
            return set(e.keys())
        self.assertEqual(keyset("groq"), keyset("cerebras"))
        self.assertEqual(keyset("groq"), keyset("together"))

    # ── #5: NON-streaming OpenAI-shaped usage extraction. ──
    def test_non_streaming_usage_extraction(self):
        resp = _FakeResponse(
            "llama-3.3-70b-versatile", _FakeUsage(120, 34, 154)
        )
        model, inp, out, cached = _extract_openai_compatible_usage(resp)
        self.assertEqual(model, "llama-3.3-70b-versatile")
        self.assertEqual(inp, 120)
        self.assertEqual(out, 34)
        self.assertEqual(cached, 0)

    # ── #6: streaming set membership (wires all three Python stream stages). ──
    def test_groq_in_openai_shaped_stream_providers(self):
        self.assertIn("groq", _OPENAI_SHAPED_STREAM_PROVIDERS)

    # ── #7 (FIRST-CLASS streaming metering): a realistic groq FINAL chunk nests
    # usage under `x_groq.usage`; both the latch (`_chunk_usage`) and the
    # extractor (`_extract_openai_compatible_usage`) must read it. RED without
    # the x_groq unwrap (both return None/0). ──
    def test_streaming_final_chunk_usage_from_x_groq(self):
        final = _FakeStreamChunk(
            "llama-3.3-70b-versatile",
            x_groq_usage=_FakeUsage(210, 88, 298),
            top_usage=None,
        )
        # latched by the stream wrapper
        latched = _chunk_usage(final)
        self.assertIsNotNone(latched)
        # and extracted to real tokens
        model, inp, out, cached = _extract_openai_compatible_usage(final)
        self.assertEqual(inp, 210)
        self.assertEqual(out, 88)
        # raw usage forwarding also finds it under x_groq
        raw = _extract_raw_usage("groq", final)
        self.assertIsNotNone(raw)
        self.assertEqual(raw.prompt_tokens, 210)

    # a non-final chunk (no usage anywhere) must NOT latch
    def test_non_final_chunk_not_latched(self):
        class _Delta:
            content = "hi"

        class _Choice:
            delta = _Delta()

        class _Chunk:
            usage = None
            x_groq = None
            data = None
            choices = [_Choice()]

        self.assertIsNone(_chunk_usage(_Chunk()))

    # ── #7 continued: non-stream raw usage still reads top-level `.usage`. ──
    def test_raw_usage_non_streaming_top_level(self):
        resp = _FakeResponse("m", _FakeUsage(7, 3, 10))
        raw = _extract_raw_usage("groq", resp)
        self.assertEqual(raw.prompt_tokens, 7)

    # ── regression: cerebras/together/openai unchanged; no x_groq leakage. ──
    def test_no_regression_other_providers(self):
        resp = _FakeResponse("m", _FakeUsage(7, 3, 10))
        for p in ("cerebras", "together", "openai"):
            model, inp, out, cached = _extract_openai_compatible_usage(resp)
            self.assertEqual((inp, out), (7, 3))
        # a chunk that only has x_groq must not be picked up by mistral's path
        only_xgroq = _FakeStreamChunk("m", x_groq_usage=_FakeUsage(1, 1), top_usage=None)
        self.assertIsNone(_extract_raw_usage("mistral", only_xgroq))


if __name__ == "__main__":
    unittest.main()
