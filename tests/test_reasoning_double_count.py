"""Reasoning-token accounting regressions.

OpenAI Chat Completions `usage.completion_tokens` ALREADY includes
`completion_tokens_details.reasoning_tokens` (they are a breakdown of the
completion total, not an addition to it). Previously the SDK added the
reasoning count on top of the completion count in two extractors, inflating
output tokens — and therefore cost — by up to ~2x for reasoning models, which
could trip budget limits prematurely. These tests pin the corrected behavior:

  - `_extract_openai_compatible_usage` (manual chat targets): output tokens
    come from `completion_tokens` alone.
  - `_extract_li_usage_py` (LlamaIndex raw-usage fallback): same.
  - The Responses-API shape (`output_tokens` already includes reasoning) and
    the Gemini shape (`thoughts_token_count` IS reported separately from
    `candidates_token_count`, so summing there is correct) are pinned
    unchanged.

Everything runs offline against fake usage objects (mirrors
tests/test_groq_usage.py).
"""
import unittest
from types import SimpleNamespace

from token_police.enforcer import (
    _extract_openai_compatible_usage,
    _extract_li_usage_py,
)


# ── Chat Completions fakes (attribute access, like pydantic usage models) ──

class _FakeCompletionTokensDetails:
    def __init__(self, reasoning_tokens=0):
        self.reasoning_tokens = reasoning_tokens


class _FakePromptTokensDetails:
    def __init__(self, cached_tokens=0):
        self.cached_tokens = cached_tokens


class _FakeChatUsage:
    def __init__(
        self,
        prompt_tokens=0,
        completion_tokens=0,
        prompt_tokens_details=None,
        completion_tokens_details=None,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        if prompt_tokens_details is not None:
            self.prompt_tokens_details = prompt_tokens_details
        if completion_tokens_details is not None:
            self.completion_tokens_details = completion_tokens_details


class _FakeChatResponse:
    def __init__(self, model, usage):
        self.model = model
        self.usage = usage


# ── Responses-API fakes ──

class _FakeOutputTokensDetails:
    def __init__(self, reasoning_tokens=0):
        self.reasoning_tokens = reasoning_tokens


class _FakeResponsesUsage:
    def __init__(self, input_tokens=0, output_tokens=0, output_tokens_details=None):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        if output_tokens_details is not None:
            self.output_tokens_details = output_tokens_details


# ── LlamaIndex fakes ──

class _FakeLIOpenAI:
    """Class name intentionally avoids anthropic/google so the LI extractor
    takes its OpenAI-default branch."""
    model = "gpt-5-fake"


class _FakeGoogleGenAI:
    """Class name contains 'google' → LI extractor takes the google branch."""
    model = "gemini-fake"


class _FakeLIChatResponse:
    def __init__(self, raw=None, additional_kwargs=None, message=None):
        self.raw = raw
        self.additional_kwargs = additional_kwargs or {}
        self.message = message


class TestChatCompletionsReasoningNotDoubleCounted(unittest.TestCase):
    def test_reasoning_tokens_not_added_to_completion(self):
        # completion_tokens=1000 already contains the 600 reasoning tokens.
        usage = _FakeChatUsage(
            prompt_tokens=100,
            completion_tokens=1000,
            completion_tokens_details=_FakeCompletionTokensDetails(reasoning_tokens=600),
        )
        resp = _FakeChatResponse("o3-fake", usage)
        model, inp, out, cached = _extract_openai_compatible_usage(resp)
        self.assertEqual(model, "o3-fake")
        self.assertEqual(out, 1000)  # NOT 1600
        self.assertEqual(inp, 100)
        self.assertEqual(cached, 0)

    def test_non_reasoning_response_details_absent(self):
        usage = _FakeChatUsage(prompt_tokens=10, completion_tokens=50)
        _, _, out, _ = _extract_openai_compatible_usage(_FakeChatResponse("m", usage))
        self.assertEqual(out, 50)

    def test_non_reasoning_response_details_present_zero(self):
        usage = _FakeChatUsage(
            prompt_tokens=10,
            completion_tokens=50,
            completion_tokens_details=_FakeCompletionTokensDetails(reasoning_tokens=0),
        )
        _, _, out, _ = _extract_openai_compatible_usage(_FakeChatResponse("m", usage))
        self.assertEqual(out, 50)

    def test_cached_prompt_accounting_unchanged(self):
        # Same usage shape as the reasoning case, plus cached prompt tokens:
        # prompt_tokens includes cached_tokens, so input = prompt - cached.
        usage = _FakeChatUsage(
            prompt_tokens=100,
            completion_tokens=1000,
            prompt_tokens_details=_FakePromptTokensDetails(cached_tokens=40),
            completion_tokens_details=_FakeCompletionTokensDetails(reasoning_tokens=600),
        )
        model, inp, out, cached = _extract_openai_compatible_usage(
            _FakeChatResponse("o3-fake", usage)
        )
        self.assertEqual(inp, 60)
        self.assertEqual(cached, 40)
        self.assertEqual(out, 1000)

    def test_responses_api_output_unchanged(self):
        # Responses API: output_tokens already includes reasoning — pinned.
        usage = _FakeResponsesUsage(
            input_tokens=100,
            output_tokens=1000,
            output_tokens_details=_FakeOutputTokensDetails(reasoning_tokens=600),
        )
        model, inp, out, cached = _extract_openai_compatible_usage(
            _FakeChatResponse("o3-fake", usage)
        )
        self.assertEqual(out, 1000)
        self.assertEqual(inp, 100)

    def test_no_usage_returns_zeros(self):
        class _NoUsage:
            model = "m"
            usage = None

        self.assertEqual(_extract_openai_compatible_usage(_NoUsage()), ("m", 0, 0, 0))

    def test_malformed_usage_returns_zeros(self):
        # Truthy usage object with none of the expected token attributes.
        resp = _FakeChatResponse("m", SimpleNamespace())
        self.assertEqual(_extract_openai_compatible_usage(resp), ("m", 0, 0, 0))


class TestLlamaIndexReasoningNotDoubleCounted(unittest.TestCase):
    def test_raw_usage_reasoning_not_added(self):
        # additional_kwargs carries no counts → raw.usage fallback engages.
        raw = SimpleNamespace(
            model="o3-fake",
            usage=_FakeChatUsage(
                prompt_tokens=100,
                completion_tokens=1000,
                completion_tokens_details=_FakeCompletionTokensDetails(
                    reasoning_tokens=600
                ),
            ),
        )
        model, inp, out, cached = _extract_li_usage_py(
            _FakeLIOpenAI(), _FakeLIChatResponse(raw=raw)
        )
        self.assertEqual(out, 1000)  # NOT 1600
        self.assertEqual(inp, 100)
        self.assertEqual(cached, 0)

    def test_raw_usage_non_reasoning_details_absent(self):
        raw = SimpleNamespace(
            model="m", usage=_FakeChatUsage(prompt_tokens=10, completion_tokens=50)
        )
        _, _, out, _ = _extract_li_usage_py(
            _FakeLIOpenAI(), _FakeLIChatResponse(raw=raw)
        )
        self.assertEqual(out, 50)

    def test_raw_usage_non_reasoning_details_zero(self):
        raw = SimpleNamespace(
            model="m",
            usage=_FakeChatUsage(
                prompt_tokens=10,
                completion_tokens=50,
                completion_tokens_details=_FakeCompletionTokensDetails(
                    reasoning_tokens=0
                ),
            ),
        )
        _, _, out, _ = _extract_li_usage_py(
            _FakeLIOpenAI(), _FakeLIChatResponse(raw=raw)
        )
        self.assertEqual(out, 50)

    def test_raw_usage_cached_prompt_accounting_unchanged(self):
        raw = SimpleNamespace(
            model="o3-fake",
            usage=_FakeChatUsage(
                prompt_tokens=100,
                completion_tokens=1000,
                prompt_tokens_details=_FakePromptTokensDetails(cached_tokens=40),
                completion_tokens_details=_FakeCompletionTokensDetails(
                    reasoning_tokens=600
                ),
            ),
        )
        _, inp, out, cached = _extract_li_usage_py(
            _FakeLIOpenAI(), _FakeLIChatResponse(raw=raw)
        )
        self.assertEqual(inp, 60)
        self.assertEqual(cached, 40)
        self.assertEqual(out, 1000)

    def test_gemini_thoughts_summed_separately(self):
        # Gemini DOES report thoughts separately from candidates — the sum
        # there is correct and pinned.
        raw = SimpleNamespace(
            model_version="gemini-fake",
            usage_metadata={
                "prompt_token_count": 10,
                "candidates_token_count": 100,
                "thoughts_token_count": 60,
            },
        )
        _, inp, out, cached = _extract_li_usage_py(
            _FakeGoogleGenAI(), _FakeLIChatResponse(raw=raw)
        )
        self.assertEqual(out, 160)

    def test_none_response_returns_zeros(self):
        self.assertEqual(
            _extract_li_usage_py(_FakeLIOpenAI(), None), ("gpt-5-fake", 0, 0, 0)
        )

    def test_malformed_raw_usage_returns_zeros(self):
        # Truthy raw.usage with none of the expected token attributes.
        raw = SimpleNamespace(model="m", usage=SimpleNamespace())
        self.assertEqual(
            _extract_li_usage_py(_FakeLIOpenAI(), _FakeLIChatResponse(raw=raw)),
            ("m", 0, 0, 0),
        )


if __name__ == "__main__":
    unittest.main()
