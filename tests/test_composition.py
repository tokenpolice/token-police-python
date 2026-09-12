"""Tests for prompt/response composition extraction.

Focused on the bug fix where OpenAI legacy TTS responses
(`HttpxBinaryResponseContent`) were misclassified as
`{type: "text", length: <~110_000>}` because their `.text` property decodes
the audio bytes via httpx's response encoding.
"""
import unittest

import types

from token_police.composition import (
    _looks_like_binary_audio,
    _parse_anthropic_messages,
    _parse_google_contents,
    _parse_openai_messages,
    _text_entry,
    _try_modality_response,
    build_prompt_composition,
    build_response_composition,
)


class _FakeHttpxResponse:
    """Minimal stand-in for httpx.Response — has .text/.content/.headers
    that look like an audio response."""
    def __init__(self, payload_bytes: bytes = b"X" * 1024):
        self.content = payload_bytes
        # `.text` is what bites us — httpx decodes binary audio as string of
        # equal length, producing a fake `length=<byte_count>` entry.
        self.text = payload_bytes.decode("latin-1", errors="replace")
        self.headers = {"content-type": "audio/mpeg"}


class TestBinaryAudioDetection(unittest.TestCase):
    """The detector must positively identify every shape of OpenAI / Mistral /
    HF TTS response wrapper, so the downstream parsers never read `.text` on a
    binary body."""

    def test_openai_legacy_httpx_binary_response_content(self):
        """The exact class that ships with `openai.audio.speech.create(...)`."""
        try:
            from openai._legacy_response import HttpxBinaryResponseContent
        except Exception:
            self.skipTest("openai not importable in this environment")
        binary = HttpxBinaryResponseContent(_FakeHttpxResponse())
        self.assertTrue(_looks_like_binary_audio(binary))

    def test_raw_bytes(self):
        self.assertTrue(_looks_like_binary_audio(b"mp3 payload"))
        self.assertTrue(_looks_like_binary_audio(bytearray(b"opus payload")))
        self.assertTrue(_looks_like_binary_audio(memoryview(b"wav payload")))

    def test_bytes_iterator_duck_type(self):
        """Any vendor wrapper that quacks like a bytes iterator (read +
        iter_bytes / aiter_bytes / stream_to_file) is treated as binary."""
        class VendorWrapper:
            text = "X" * 110_000     # would mislead a `.text` parser
            def read(self): return b""
            def iter_bytes(self): return iter([b""])
        self.assertTrue(_looks_like_binary_audio(VendorWrapper()))

    def test_audio_content_type_header(self):
        """If only the headers expose the truth (raw httpx.Response handed
        through), the content-type sniff catches it."""
        class HeadersOnly:
            headers = {"content-type": "audio/mp3; charset=utf-8"}
            text = "X" * 100
            content = b"X" * 100
        self.assertTrue(_looks_like_binary_audio(HeadersOnly()))

    def test_application_octet_stream(self):
        class HeadersOnly:
            headers = {"content-type": "application/octet-stream"}
            text = "junk"
        self.assertTrue(_looks_like_binary_audio(HeadersOnly()))

    def test_plain_text_not_binary(self):
        """A real transcription / chat response must NOT be flagged."""
        class Transcription:
            text = "hello"
            language = "en"
        self.assertFalse(_looks_like_binary_audio(Transcription()))
        self.assertFalse(_looks_like_binary_audio("plain string"))
        self.assertFalse(_looks_like_binary_audio({"choices": []}))


class TestModalityResponse(unittest.TestCase):
    """End-to-end behaviour of `_try_modality_response` for each response shape."""

    def test_openai_tts_legacy_binary(self):
        try:
            from openai._legacy_response import HttpxBinaryResponseContent
        except Exception:
            self.skipTest("openai not importable in this environment")
        binary = HttpxBinaryResponseContent(_FakeHttpxResponse())
        comp = _try_modality_response(binary)
        self.assertEqual(comp, [{"role": "assistant", "type": "audio"}])

    def test_openai_image_gen_response(self):
        img = type("X", (), {"data": [type("Y", (), {"url": "http://x", "b64_json": None})()]})()
        comp = _try_modality_response(img)
        self.assertEqual(comp, [{"role": "assistant", "type": "image"}])

    def test_whisper_transcription_response(self):
        """Whisper transcription has `.text` AND `.language`/`.words`/etc —
        treat as text (not binary)."""
        class Transcription:
            text = "hello world from whisper"
            language = "en"
            words = []
            segments = []
            duration = 1.5
        comp = _try_modality_response(Transcription())
        self.assertIsNotNone(comp)
        self.assertEqual(comp[0]["role"], "assistant")
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[0]["length"], len("hello world from whisper"))
        self.assertIn("hash", comp[0])

    def test_text_only_plain_object(self):
        """A response that exposes only `.text` with no transcription marker AND
        no bytes-iterator surface is still treated as text (HF Mistral style)."""
        class HfText:
            text = "transcribed sentence"
        comp = _try_modality_response(HfText())
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[0]["length"], len("transcribed sentence"))


class TestBuildResponseComposition(unittest.TestCase):
    """Top-level `build_response_composition` contract."""

    def test_tts_audio_no_length_no_hash(self):
        """REGRESSION: TTS response → `{role, type: audio}` with NO length, NO
        hash. Production previously emitted `{type:text, length:110462, hash:...}`."""
        try:
            from openai._legacy_response import HttpxBinaryResponseContent
        except Exception:
            self.skipTest("openai not importable")
        binary = HttpxBinaryResponseContent(_FakeHttpxResponse(b"X" * 110_462))
        comp = build_response_composition("openai", binary)
        self.assertEqual(comp, [{"role": "assistant", "type": "audio"}])
        # Defensive: explicitly assert no length / hash leak.
        self.assertNotIn("length", comp[0])
        self.assertNotIn("hash", comp[0])

    def test_chat_text_response_preserves_length_and_hash(self):
        """Make sure the chat-completions path still produces a proper text entry."""
        chat = {"choices": [{"message": {"content": "hello", "tool_calls": None}}]}
        comp = build_response_composition("openai", chat)
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["role"], "assistant")
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[0]["length"], 5)
        self.assertIn("hash", comp[0])

    def test_usage_shape_authoritative_override(self):
        """Even if heuristic parsers would mis-classify the response, an
        explicit `usage_shape='openai_audio_tts'` MUST produce an audio entry."""
        # Pass something that LOOKS like text (chat completion dict) but lie
        # about the shape — the override must win.
        misleading = {"choices": [{"message": {"content": "shouldn't be hashed"}}]}
        comp = build_response_composition("openai", misleading,
                                          usage_shape="openai_audio_tts")
        self.assertEqual(comp, [{"role": "assistant", "type": "audio"}])

    def test_usage_shape_image_override(self):
        comp = build_response_composition("openai", None, usage_shape="openai_images")
        self.assertEqual(comp, [{"role": "assistant", "type": "image"}])

    def test_usage_shape_video_override(self):
        comp = build_response_composition("google", None, usage_shape="google_veo")
        self.assertEqual(comp, [{"role": "assistant", "type": "video"}])

    def test_raw_bytes_response(self):
        """If a customer passes `.content` (raw bytes) instead of the wrapper."""
        comp = build_response_composition("openai", b"mp3 bytes here")
        self.assertEqual(comp, [{"role": "assistant", "type": "audio"}])

    def test_unknown_shape_falls_through(self):
        """An unknown shape doesn't short-circuit; heuristic parsing still runs."""
        chat = {"choices": [{"message": {"content": "hello"}}]}
        comp = build_response_composition("openai", chat,
                                          usage_shape="some_brand_new_shape")
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[0]["length"], 5)


class _FakeOcrUsageInfo:
    def __init__(self, pages_processed):
        self.pages_processed = pages_processed


class _FakeOcrResponse:
    """Mirrors a Mistral OCRResponse: `.pages` / `.usage_info`, no `.text`."""
    def __init__(self, usage_info=None, pages=None):
        if usage_info is not None:
            self.usage_info = usage_info
        if pages is not None:
            self.pages = pages


class TestOcrComposition(unittest.TestCase):
    """T1 — `mistral_ocr` usage_shape OCR tier. Page count comes from
    `usage_info.pages_processed` (billed `ocr_pages`), NOT `.pages`.
    Privacy-preserving structural markers only (no length, no hash)."""

    def test_pages_processed_dict_seven(self):
        resp = {"usage_info": {"pages_processed": 7}, "pages": [{}, {}]}
        comp = build_response_composition("mistral", resp, usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_page"}] * 7)
        # No length / hash leak.
        for e in comp:
            self.assertNotIn("length", e)
            self.assertNotIn("hash", e)

    def test_pages_processed_object_seven(self):
        resp = _FakeOcrResponse(usage_info=_FakeOcrUsageInfo(7), pages=[object(), object()])
        comp = build_response_composition("mistral", resp, usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_page"}] * 7)

    def test_pages_processed_zero_single_document(self):
        resp = {"usage_info": {"pages_processed": 0}}
        comp = build_response_composition("mistral", resp, usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_document"}])

    def test_usage_info_missing_single_document(self):
        comp = build_response_composition("mistral", {"model": "mistral-ocr-latest"},
                                          usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_document"}])

    def test_usage_info_missing_falls_back_to_pages_len(self):
        resp = {"pages": [{}, {}, {}]}
        comp = build_response_composition("mistral", resp, usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_page"}] * 3)

    def test_pages_not_a_list_single_document(self):
        resp = {"pages": "not-a-list"}
        comp = build_response_composition("mistral", resp, usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_document"}])

    def test_huge_count_capped_at_100(self):
        resp = {"usage_info": {"pages_processed": 5000}}
        comp = build_response_composition("mistral", resp, usage_shape="mistral_ocr")
        self.assertEqual(len(comp), 100)
        self.assertTrue(all(e == {"role": "assistant", "type": "ocr_page"} for e in comp))

    def test_none_response_single_document(self):
        comp = build_response_composition("mistral", None, usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_document"}])

    def test_non_int_pages_processed(self):
        # Coercible string → N pages.
        comp = build_response_composition("mistral", {"usage_info": {"pages_processed": "3"}},
                                          usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_page"}] * 3)
        # Garbage → document fallback.
        comp = build_response_composition("mistral", {"usage_info": {"pages_processed": "abc"}},
                                          usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_document"}])

    def test_bool_pages_processed_rejected(self):
        # bool is an int subclass — True must NOT be treated as 1 page.
        comp = build_response_composition("mistral", {"usage_info": {"pages_processed": True}},
                                          usage_shape="mistral_ocr")
        self.assertEqual(comp, [{"role": "assistant", "type": "ocr_document"}])


class _FakeLCMessage:
    """Minimal LangChain AIMessage stand-in with both content blocks and the
    normalized .tool_calls list (the shape that caused double tool_call rows)."""
    def __init__(self, content, tool_calls):
        self.type = "ai"
        self.content = content
        self.tool_calls = tool_calls
        self.name = None


class _FakeLCGen:
    def __init__(self, message):
        self.message = message
        self.text = None


class _FakeLLMResult:
    def __init__(self, message):
        self.generations = [[_FakeLCGen(message)]]


class TestLangChainToolCallDedup(unittest.TestCase):
    """A tool-calling LangChain AIMessage carries its calls in BOTH content[]
    (tool_use blocks) AND .tool_calls — the parser must emit each call once."""

    def test_tool_call_emitted_once_when_both_sources_present(self):
        msg = _FakeLCMessage(
            content=[
                {"type": "text", "text": "let me look that up"},
                {"type": "tool_use", "name": "search_knowledge_base", "input": {"q": "laptop"}},
                {"type": "tool_use", "name": "get_customer_info", "input": {"email": "a@b.com"}},
            ],
            tool_calls=[
                {"name": "search_knowledge_base", "args": {"q": "laptop"}, "id": "t1"},
                {"name": "get_customer_info", "args": {"email": "a@b.com"}, "id": "t2"},
            ],
        )
        comp = build_response_composition("langchain", _FakeLLMResult(msg))
        tool_calls = [e for e in comp if e.get("type") == "tool_call"]
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(
            sorted(e.get("name") for e in tool_calls),
            ["get_customer_info", "search_knowledge_base"],
        )
        self.assertEqual(len([e for e in comp if e.get("type") == "text"]), 1)

    def test_tool_call_still_emitted_when_only_normalized_present(self):
        msg = _FakeLCMessage(content="", tool_calls=[
            {"name": "only_via_normalized", "args": {}, "id": "t1"},
        ])
        comp = build_response_composition("langchain", _FakeLLMResult(msg))
        tool_calls = [e for e in comp if e.get("type") == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].get("name"), "only_via_normalized")


class TestWhitespaceNormalizedHashing(unittest.TestCase):
    """The same logical text must fingerprint identically across calls even
    when a framework (e.g. CrewAI) rstrips it between turns."""

    def test_trailing_and_leading_whitespace_ignored(self):
        a = build_response_composition("openai", {"choices": [{"message": {"content": "hello world"}}]})
        b = build_response_composition("openai", {"choices": [{"message": {"content": "  hello world\n"}}]})
        self.assertEqual(a[0]["hash"], b[0]["hash"])
        self.assertEqual(a[0]["length"], b[0]["length"])
        self.assertEqual(a[0]["length"], len("hello world"))


class TestGoogleRestNativeContents(unittest.TestCase):
    """Gemini's native REST format (and apps hand-building `contents`) uses
    camelCase part keys (`functionCall` / `functionResponse` / `inlineData`),
    whereas the google-genai Pydantic SDK uses snake_case. Both must parse, or
    tool-call / tool-result turns get dropped and the composition collapses to
    system+first-user — which false-fires the HASH_CYCLE loop detector on a
    legitimately-progressing tool-using agent.
    """

    SYSTEM = "You are a helpful customer support agent."

    def _turn1(self):
        return [{"role": "user", "parts": [{"text": "My email is jane.pro@x.com. What plan am I on?"}]}]

    def _turn2(self):
        # After step 1 the agent appends the model's functionCall turn and a
        # user functionResponse turn (camelCase, REST-native).
        return self._turn1() + [
            {"role": "model", "parts": [{"functionCall": {"name": "get_customer_info",
                                                          "args": {"email": "jane.pro@x.com"}}}]},
            {"role": "user", "parts": [{"functionResponse": {"name": "get_customer_info",
                                                             "response": {"plan": "Pro"}}}]},
        ]

    @staticmethod
    def _has_tool_call(comp):
        return any(e.get("type") == "tool_call" for e in comp)

    @staticmethod
    def _has_tool_result(comp):
        # tool_result entries carry role="tool_result" (type stays "text"),
        # matching the Node SDK convention.
        return any(e.get("role") == "tool_result" for e in comp)

    def test_camelcase_tool_turns_not_dropped(self):
        comp = _parse_google_contents(self._turn2(), self.SYSTEM)
        self.assertTrue(self._has_tool_call(comp))
        self.assertTrue(self._has_tool_result(comp))
        tool_entries = [e for e in comp if e.get("type") == "tool_call" or e.get("role") == "tool_result"]
        self.assertEqual([e.get("name") for e in tool_entries],
                         ["get_customer_info", "get_customer_info"])

    def test_snake_case_still_parses(self):
        # google-genai Pydantic model_dump() shape — must remain supported.
        contents = self._turn1() + [
            {"role": "model", "parts": [{"function_call": {"name": "lookup", "args": {"q": "x"}}}]},
            {"role": "user", "parts": [{"function_response": {"name": "lookup", "response": {"ok": True}}}]},
        ]
        comp = _parse_google_contents(contents, self.SYSTEM)
        self.assertTrue(self._has_tool_call(comp))
        self.assertTrue(self._has_tool_result(comp))

    def test_composition_grows_per_step(self):
        # Regression guard for the phantom loop: successive agent turns must
        # produce distinct composition lengths so the per-step fingerprint
        # differs (no false HASH_CYCLE).
        c1 = build_prompt_composition("google", {"contents": self._turn1(),
                                                 "system_instruction": self.SYSTEM})
        c2 = build_prompt_composition("google", {"contents": self._turn2(),
                                                 "system_instruction": self.SYSTEM})
        self.assertGreater(len(c2), len(c1))

    def test_camelcase_inline_data_image(self):
        contents = [{"role": "user", "parts": [
            {"inlineData": {"mimeType": "image/png", "data": "..."}}]}]
        comp = _parse_google_contents(contents)
        self.assertTrue(any(e.get("type") == "image" for e in comp))

    def test_rest_response_function_call_not_dropped(self):
        # REST responses are decoded to SimpleNamespace with camelCase attrs.
        part = types.SimpleNamespace(functionCall=types.SimpleNamespace(
            name="get_customer_info", args={"email": "jane.pro@x.com"}))
        content = types.SimpleNamespace(role="model", parts=[part])
        response = types.SimpleNamespace(candidates=[types.SimpleNamespace(content=content)])
        comp = build_response_composition("google", response)
        self.assertEqual([e.get("type") for e in comp], ["tool_call"])
        self.assertEqual(comp[0].get("name"), "get_customer_info")


class TestGeminiTextPlusFunctionCallComposition(unittest.TestCase):
    """Google-genai GenerateContentResponse exposes a convenience
    ``.text`` (concatenation of text parts). When a response carries both a
    text part and a function_call part, the modality short-circuit used to
    return text-only composition and drop ``tool_call``. Node already guards
    with ``isStructuredChat``; Python must match.
    """

    @staticmethod
    def _gemini_response(parts, text):
        """Mock google-genai shape: candidates + top-level .text convenience."""
        content = types.SimpleNamespace(role="model", parts=parts)
        return types.SimpleNamespace(
            text=text,
            candidates=[types.SimpleNamespace(content=content)],
        )

    def test_text_plus_function_call_keeps_both_entries(self):
        # The real repro: non-empty .text AND a sibling function_call part.
        parts = [
            types.SimpleNamespace(text="Let me look that up for you."),
            types.SimpleNamespace(function_call=types.SimpleNamespace(
                name="get_customer_info", args={"email": "jane.pro@x.com"})),
        ]
        response = self._gemini_response(parts, text="Let me look that up for you.")
        # Precondition: modality must NOT short-circuit (would be text-only).
        self.assertIsNone(_try_modality_response(response))
        comp = build_response_composition("google", response)
        types_ = [e.get("type") for e in comp]
        self.assertEqual(types_, ["text", "tool_call"])
        self.assertEqual(comp[0].get("role"), "assistant")
        self.assertEqual(comp[0].get("length"), len("Let me look that up for you."))
        self.assertEqual(comp[1].get("name"), "get_customer_info")

    def test_function_call_only_still_tool_call(self):
        # .text is None (fc-only) — already worked; must not regress.
        parts = [
            types.SimpleNamespace(function_call=types.SimpleNamespace(
                name="search_kb", args={"q": "error 505"})),
        ]
        response = self._gemini_response(parts, text=None)
        comp = build_response_composition("google", response)
        self.assertEqual([e.get("type") for e in comp], ["tool_call"])
        self.assertEqual(comp[0].get("name"), "search_kb")

    def test_text_only_with_candidates_and_text_convenience(self):
        # Text-only Gemini still yields a single assistant text entry even
        # when the structured-chat gate skips the modality shortcut.
        parts = [types.SimpleNamespace(text="Hello there")]
        response = self._gemini_response(parts, text="Hello there")
        self.assertIsNone(_try_modality_response(response))
        comp = build_response_composition("google", response)
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0].get("type"), "text")
        self.assertEqual(comp[0].get("role"), "assistant")
        self.assertEqual(comp[0].get("length"), len("Hello there"))

    def test_dict_with_text_key_and_mixed_candidates(self):
        # Dict-shaped REST body with a top-level "text" key + candidates.
        # Gate must treat candidates list as structured (dict-or-attr).
        response = {
            "text": "Let me check.",
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "Let me check."},
                {"functionCall": {"name": "search_kb", "args": {"q": "x"}}},
            ]}}],
        }
        self.assertIsNone(_try_modality_response(response))
        comp = build_response_composition("google", response)
        roles = [(e.get("role"), e.get("type")) for e in comp]
        self.assertIn(("assistant", "text"), roles)
        self.assertIn(("tool_call", "tool_call"), roles)

    def test_parts_list_skips_modality_text_shortcut(self):
        # Joint gate with pydantic_ai ModelResponse has .text + .parts.
        # Modality must decline so the pydantic_ai parser can run.
        class FakePart:
            part_kind = "tool-call"
            tool_name = "lookup"
            args = {"q": "x"}

        class FakeModelResponse:
            text = "I'll look that up."
            parts = [FakePart()]

        response = FakeModelResponse()
        self.assertIsNone(_try_modality_response(response))
        comp = build_response_composition("pydantic_ai", response)
        self.assertTrue(any(e.get("type") == "tool_call" for e in comp))
        tool = next(e for e in comp if e.get("type") == "tool_call")
        self.assertEqual(tool.get("name"), "lookup")

    def test_string_content_not_structured_chat(self):
        # xAI-like: .content is a str, not a list — must still take the
        # modality .text path when present (list-only containers).
        class XaiLike:
            content = "plain string content"
            text = "plain string content"

        self.assertIsNotNone(_try_modality_response(XaiLike()))
        self.assertEqual(_try_modality_response(XaiLike())[0].get("type"), "text")


class TestPydanticAITextPlusToolCallComposition(unittest.TestCase):
    """Pydantic_ai ModelResponse exposes a computed ``.text`` (join of
    TextParts) plus a ``.parts`` list that may also hold ToolCallParts.

    Pre-fix, ``_try_modality_response`` short-circuited on truthy ``.text`` and
    returned a single assistant text entry, so Claude-style text+tool_call
    responses dropped ``tool_call`` entirely from ``response_composition``.
    The structured-chat gate treats list ``parts`` as structured so the
    pydantic_ai parser runs. This class locks the full verification matrix.
    """

    @staticmethod
    def _text_part(content: str):
        return types.SimpleNamespace(part_kind="text", content=content)

    @staticmethod
    def _tool_part(name: str, args=None):
        return types.SimpleNamespace(
            part_kind="tool-call",
            tool_name=name,
            args=args if args is not None else {},
        )

    @staticmethod
    def _model_response(parts, text):
        """Stand-in for pydantic_ai ModelResponse / StreamedResponse.get()."""
        return types.SimpleNamespace(text=text, parts=list(parts))

    def test_text_plus_tool_call_keeps_both_entries(self):
        # Real Claude-via-pydantic_ai repro: preamble text + tool call, with
        # computed .text truthy (would have been swallowed pre-fix).
        preamble = "Let me search that for you."
        parts = [
            self._text_part(preamble),
            self._tool_part("search_knowledge_base", {"q": "error 505"}),
        ]
        response = self._model_response(parts, text=preamble)
        self.assertIsNone(_try_modality_response(response))
        comp = build_response_composition("pydantic_ai", response)
        self.assertEqual([e.get("type") for e in comp], ["text", "tool_call"])
        self.assertEqual(comp[0].get("role"), "assistant")
        self.assertEqual(comp[0].get("length"), len(preamble))
        self.assertEqual(comp[1].get("name"), "search_knowledge_base")
        self.assertEqual(comp[1].get("type"), "tool_call")

    def test_tool_call_only_still_tool_call(self):
        # Pure tool-call (OpenAI "control" survival path) — .text is None so
        # modality already declined; must not regress.
        parts = [self._tool_part("lookup", {"email": "a@b.com"})]
        response = self._model_response(parts, text=None)
        comp = build_response_composition("pydantic_ai", response)
        self.assertEqual([e.get("type") for e in comp], ["tool_call"])
        self.assertEqual(comp[0].get("name"), "lookup")

    def test_text_only_single_text_entry(self):
        body = "Hello from the agent."
        parts = [self._text_part(body)]
        response = self._model_response(parts, text=body)
        self.assertIsNone(_try_modality_response(response))
        comp = build_response_composition("pydantic_ai", response)
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0].get("type"), "text")
        self.assertEqual(comp[0].get("role"), "assistant")
        self.assertEqual(comp[0].get("length"), len(body))

    def test_streamed_response_get_shape_keeps_both(self):
        # Stream finalize uses StreamedResponse.get() → ModelResponse with the
        # same .parts shape; same parser must keep both entries.
        preamble = "Checking now."
        parts = [
            self._text_part(preamble),
            self._tool_part("get_customer_info", {"id": "42"}),
        ]
        response = self._model_response(parts, text=preamble)
        comp = build_response_composition("pydantic_ai", response)
        self.assertEqual([e.get("type") for e in comp], ["text", "tool_call"])
        self.assertEqual(comp[1].get("name"), "get_customer_info")

    def test_multi_tool_call_order_preserved(self):
        preamble = "I'll run both lookups."
        parts = [
            self._text_part(preamble),
            self._tool_part("search_knowledge_base", {"q": "x"}),
            self._tool_part("get_customer_info", {"email": "y@z.com"}),
        ]
        response = self._model_response(parts, text=preamble)
        comp = build_response_composition("pydantic_ai", response)
        types_ = [e.get("type") for e in comp]
        names = [e.get("name") for e in comp]
        self.assertEqual(types_, ["text", "tool_call", "tool_call"])
        self.assertEqual(names, [None, "search_knowledge_base", "get_customer_info"])

    def test_parts_list_skips_modality_text_shortcut(self):
        # Gate precondition: list .parts → modality declines even with truthy .text.
        response = self._model_response(
            [self._tool_part("lookup", {"q": "x"})],
            text="I'll look that up.",
        )
        self.assertIsNone(_try_modality_response(response))

    def test_binary_audio_still_audio_marker(self):
        # Binary/TTS detection stays first — must not be reordered past the
        # structured-chat gate. Existing audio suite is primary; this pins
        # the residual (f) against composition entry order.
        binary = types.SimpleNamespace(
            text="X" * 256,
            content=b"\x00\x01" * 128,
            headers={"content-type": "audio/mpeg"},
            iter_bytes=lambda: iter([b"\x00"]),
        )
        # Even if a mistaken `parts` list were present, binary check is first
        # inside _try_modality_response / build_response_composition.
        binary.parts = []  # empty list is still a list container
        comp = build_response_composition("openai", binary)
        self.assertEqual(comp, [{"role": "assistant", "type": "audio"}])


class TestOpenAIToolResultName(unittest.TestCase):
    """`tool_result` segments must carry the tool name. OpenAI/Cohere tool
    messages carry only `tool_call_id`; the name is resolved from the matching
    assistant tool_call."""

    def test_name_resolved_from_tool_call_id(self):
        # Cohere/OpenAI shape: tool message has tool_call_id, NO name.
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "search_knowledge_base", "arguments": '{"q":"x"}'}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "Article 42 ..."},
        ]
        comp = _parse_openai_messages(messages)
        tool_results = [e for e in comp if e.get("role") == "tool_result"]
        self.assertEqual(len(tool_results), 1)
        self.assertEqual(tool_results[0].get("name"), "search_knowledge_base")

    def test_explicit_name_takes_priority(self):
        # When the tool message DOES carry a name (OpenAI apps that set it),
        # honor it over the resolved one.
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "from_call", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "name": "explicit", "content": "r"},
        ]
        comp = _parse_openai_messages(messages)
        tr = [e for e in comp if e.get("role") == "tool_result"][0]
        self.assertEqual(tr.get("name"), "explicit")

    def test_unmatched_tool_call_id_is_empty(self):
        # No matching tool_call → no name key (consistent with _text_entry, which
        # omits an empty name); must not crash.
        messages = [{"role": "tool", "tool_call_id": "missing", "content": "r"}]
        comp = _parse_openai_messages(messages)
        tr = [e for e in comp if e.get("role") == "tool_result"][0]
        self.assertIsNone(tr.get("name"))


class TestGoogleRestDictResponse(unittest.TestCase):
    """Manual REST callers (tp.protect / tp.log over raw httpx) pass the decoded
    Gemini response body as a PLAIN DICT: {"candidates":[{"content":{"parts":
    [...]}}]}. The attribute-style `hasattr(response, "candidates")` branch never
    matches a dict, so tool-call turns previously fell through to the Tier-3
    complete_response raw blob. The dict branch must emit structured entries."""

    def test_dict_function_call_yields_tool_call_entry(self):
        response = {"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": "get_customer_info",
                              "args": {"email": "jane.pro@x.com"}}},
        ]}}]}
        comp = build_response_composition("google", response)
        tool_calls = [e for e in comp if e.get("type") == "tool_call"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].get("name"), "get_customer_info")
        # Must NOT be the Tier-3 raw blob.
        self.assertFalse(any(e.get("role") == "complete_response" for e in comp))

    def test_dict_snake_case_function_call_also_parses(self):
        response = {"candidates": [{"content": {"role": "model", "parts": [
            {"function_call": {"name": "lookup", "args": {"q": "x"}}},
        ]}}]}
        comp = build_response_composition("google", response)
        self.assertTrue(any(e.get("type") == "tool_call" for e in comp))

    def test_dict_text_part_is_assistant_text(self):
        response = {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "Hello there"},
        ]}}]}
        comp = build_response_composition("google", response)
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["role"], "assistant")
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[0]["length"], len("Hello there"))

    def test_dict_mixed_text_and_tool_call(self):
        response = {"candidates": [{"content": {"role": "model", "parts": [
            {"text": "Let me check."},
            {"functionCall": {"name": "search_kb", "args": {"q": "error 505"}}},
        ]}}]}
        comp = build_response_composition("google", response)
        roles = [(e.get("role"), e.get("type")) for e in comp]
        self.assertIn(("assistant", "text"), roles)
        self.assertIn(("tool_call", "tool_call"), roles)

    def test_dict_without_candidates_unaffected(self):
        # Non-google dicts must keep their existing behavior (fallback blob).
        comp = build_response_composition("google", {"something": "else"})
        self.assertTrue(any(e.get("role") == "complete_response" for e in comp))

    def test_empty_candidates_falls_through_to_blob(self):
        # Gate yields nothing → existing Tier-3 behavior preserved.
        comp = build_response_composition("google", {"candidates": []})
        self.assertTrue(any(e.get("role") == "complete_response" for e in comp))

    def test_attribute_style_response_unchanged(self):
        # The existing SimpleNamespace (SDK/REST-decoded) path must not regress.
        part = types.SimpleNamespace(functionCall=types.SimpleNamespace(
            name="get_customer_info", args={"email": "jane.pro@x.com"}))
        content = types.SimpleNamespace(role="model", parts=[part])
        response = types.SimpleNamespace(candidates=[types.SimpleNamespace(content=content)])
        comp = build_response_composition("google", response)
        self.assertTrue(any(e.get("type") == "tool_call" for e in comp))


class TestGoogleNamespaceArgsResponse(unittest.TestCase):
    """support_agent_gemini_rest_python decodes the REST JSON body with
    object_hook=SimpleNamespace, so a functionCall's ``args`` is ITSELF a
    (recursively) SimpleNamespace — ``dict(args)`` raised TypeError, the
    per-part parse aborted, and tool-call turns collapsed into the Tier-3
    complete_response blob. The args serializer must handle namespace / dict /
    str, and one hostile part must never abort its sibling parts."""

    def _ns_response(self, args):
        parts = [
            types.SimpleNamespace(text="Let me check that for you."),
            types.SimpleNamespace(
                text=None,
                functionCall=types.SimpleNamespace(name="search_kb", args=args)),
        ]
        content = types.SimpleNamespace(role="model", parts=parts)
        return types.SimpleNamespace(
            candidates=[types.SimpleNamespace(content=content)])

    def test_namespace_args_compose_text_and_tool_call(self):
        comp = build_response_composition(
            "google", self._ns_response(types.SimpleNamespace(query="x")))
        kinds = [(e.get("role"), e.get("type")) for e in comp]
        self.assertEqual(kinds[0], ("assistant", "text"))
        self.assertEqual(kinds[1], ("tool_call", "tool_call"))
        self.assertEqual(comp[1].get("name"), "search_kb")
        # Must NOT be the Tier-3 raw blob.
        self.assertFalse(any(e.get("role") == "complete_response" for e in comp))
        # The serialized args drive length/hash — non-empty.
        self.assertGreater(comp[1].get("length", 0), 0)

    def test_recursively_nested_namespace_args(self):
        args = types.SimpleNamespace(
            query="error 505",
            filters=types.SimpleNamespace(severity="high", tags=["a", "b"]))
        comp = build_response_composition("google", self._ns_response(args))
        tool = [e for e in comp if e.get("type") == "tool_call"][0]
        self.assertGreater(tool.get("length", 0), 0)
        self.assertFalse(any(e.get("role") == "complete_response" for e in comp))

    def test_dict_args_still_work(self):
        comp = build_response_composition(
            "google", self._ns_response({"query": "x"}))
        self.assertTrue(any(e.get("type") == "tool_call" for e in comp))

    def test_string_args_pass_through(self):
        comp = build_response_composition(
            "google", self._ns_response('{"query":"x"}'))
        tool = [e for e in comp if e.get("type") == "tool_call"][0]
        self.assertEqual(tool.get("length"), len('{"query":"x"}'))

    def test_hostile_part_does_not_abort_siblings(self):
        class _HostileFC:
            name = "boom"

            @property
            def args(self):
                raise RuntimeError("nope")

        parts = [
            types.SimpleNamespace(text="Still composed."),
            types.SimpleNamespace(text=None, functionCall=_HostileFC()),
        ]
        content = types.SimpleNamespace(role="model", parts=parts)
        response = types.SimpleNamespace(
            candidates=[types.SimpleNamespace(content=content)])
        comp = build_response_composition("google", response)
        # The text sibling must survive the hostile functionCall part.
        self.assertTrue(any(
            e.get("role") == "assistant" and e.get("type") == "text"
            for e in comp))
        self.assertFalse(any(e.get("role") == "complete_response" for e in comp))


class TestMalformedTextPartDegradesGracefully(unittest.TestCase):
    """A malformed (non-string) text part must degrade to an empty-text entry,
    never abort the whole composition. Before the fix, a ``None``/numeric/object
    ``text`` raised inside ``_text_entry`` (``len(None)`` / ``len(5)``), the
    throw escaped to the outer guard in ``build_prompt_composition``, and the
    ENTIRE call collapsed to a single coarse ``complete_prompt`` entry — silently
    losing per-message granularity. These tests pin the per-message survival plus
    the exact empty/numeric/object fingerprints.
    """

    # sha1-first-16 of the trimmed canonical string, PINNED byte-for-byte and
    # shared with the Node suite (composition.test.ts) as the cross-SDK parity
    # proof — computed independently, not derived from the code under test.
    HASH_EMPTY = "da39a3ee5e6b4b0d"   # ""
    HASH_FIVE = "ac3478d69a3c81fa"    # canonical "5"
    HASH_OBJ = "9f89c740ceb46d74"     # canonical '{"a":1}'

    def test_null_text_part_keeps_per_message_composition(self):
        """OpenAI-shape messages with one ``text: None`` part among healthy
        parts → per-message composition retained (NOT a single complete_prompt),
        and the null part fingerprints as empty text.

        PRE-FIX (verified by stashing the helper change): this returned a single
        ``[{role: "complete_prompt", ...}]`` entry because ``len(None)`` raised.
        """
        kwargs = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Healthy part."},
                        {"type": "text", "text": None},
                    ],
                },
            ]
        }
        comp = build_prompt_composition("openai", kwargs)

        # Whole call did NOT collapse to the coarse fallback.
        self.assertFalse(any(e.get("role") == "complete_prompt" for e in comp))
        # system + healthy user part + null user part = 3 per-message entries.
        self.assertEqual(len(comp), 3)

        healthy = comp[1]
        null_entry = comp[2]
        # Sibling unchanged.
        self.assertEqual(healthy["role"], "user")
        self.assertEqual(healthy["length"], len("Healthy part."))
        self.assertEqual(healthy["hash"], _text_entry("user", "Healthy part.")["hash"])
        # Null part → empty-text fingerprint.
        self.assertEqual(null_entry["length"], 0)
        self.assertEqual(null_entry["hash"], self.HASH_EMPTY)

    def test_null_text_part_anthropic_shape(self):
        """Same defect via the Anthropic messages shape (``text: None`` block)."""
        comp = _parse_anthropic_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Healthy part."},
                        {"type": "text", "text": None},
                    ],
                }
            ]
        )
        self.assertEqual(len(comp), 2)
        self.assertEqual(comp[0]["length"], len("Healthy part."))
        self.assertEqual(comp[1]["length"], 0)
        self.assertEqual(comp[1]["hash"], self.HASH_EMPTY)

    def test_numeric_text_part_google_contents(self):
        """Google ``contents`` with a numeric ``text`` part → per-message
        retained, numeric value fingerprinted (not dropped/aborted).

        PRE-FIX: ``len(5)`` raised, collapsing the call to complete_prompt.
        """
        comp = _parse_google_contents(
            [
                {"role": "user", "parts": [{"text": "Healthy part."}]},
                {"role": "user", "parts": [{"text": 5}]},
            ]
        )
        self.assertEqual(len(comp), 2)
        self.assertEqual(comp[0]["length"], len("Healthy part."))
        # Numeric coerced via the canonical serializer.
        self.assertEqual(comp[1]["length"], 1)
        self.assertEqual(comp[1]["hash"], self.HASH_FIVE)

    def test_non_string_text_parity(self):
        """VERBATIM cross-SDK parity: numeric ``5`` and object ``{"a": 1}`` fed
        through ``_text_entry`` produce the pinned length + hash asserted
        identically in the Node suite.
        """
        five = _text_entry("user", 5)
        self.assertEqual(five["length"], 1)
        self.assertEqual(five["hash"], self.HASH_FIVE)

        obj = _text_entry("user", {"a": 1})
        self.assertEqual(obj["length"], 7)   # '{"a":1}'
        self.assertEqual(obj["hash"], self.HASH_OBJ)

    def test_text_entry_malformed_never_raises(self):
        """Helper-level no-throw sweep: hostile / non-JSON inputs always yield an
        entry, never raise.
        """
        class _Hostile:
            def __str__(self):
                raise RuntimeError("nope")

        for value in (None, 5, True, b"bytes", object(), _Hostile(),
                      [1, {"x": [2, 3]}], float("nan"), 3.14):
            entry = _text_entry("user", value)
            self.assertEqual(entry["role"], "user")
            self.assertEqual(entry["type"], "text")
            self.assertIsInstance(entry["length"], int)
            self.assertIsInstance(entry["hash"], str)
            self.assertEqual(len(entry["hash"]), 16)

    def test_healthy_string_entry_is_byte_identical(self):
        """Hash-identity guard: a plain healthy string is unchanged by the fix
        (pinned length + hash).
        """
        entry = _text_entry("user", "hello world")
        self.assertEqual(entry["length"], 11)
        self.assertEqual(entry["hash"], "2aae6c35c94fcfb4")
        # Whitespace trimming still applies identically.
        self.assertEqual(_text_entry("user", "  hello world\n")["hash"], entry["hash"])


if __name__ == "__main__":
    unittest.main()
