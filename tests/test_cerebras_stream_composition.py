"""I5 — Cerebras streaming response_composition (OpenAI-shaped stream allowlist).

Cerebras is OpenAI wire-compatible for stream deltas
(`choices[0].delta.{content,tool_calls}`). Usage extraction already handled it;
stream composition required membership in `_OPENAI_SHAPED_STREAM_PROVIDERS`.

Without the allowlist, `_new_stream_accumulator("cerebras")` returned None and
finalize fell back to the last usage-only chunk → Tier-3 `complete_response`.
"""
import unittest

from token_police.enforcer import (
    _OPENAI_SHAPED_STREAM_PROVIDERS,
    _new_stream_accumulator,
    _accumulate_stream_chunk,
    _stream_accumulator_to_response,
)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [_Choice(_Delta(content=content, tool_calls=tool_calls))]


class _ToolFn:
    def __init__(self, name=None, arguments=None):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, index=0, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _ToolFn(name=name, arguments=arguments)


class TestCerebrasStreamComposition(unittest.TestCase):
    def test_cerebras_in_openai_shaped_stream_providers(self):
        self.assertIn("cerebras", _OPENAI_SHAPED_STREAM_PROVIDERS)

    def test_openrouter_not_in_set(self):
        # Incomplete openrouter fix is worse than leaving it out (Mode-A gate).
        self.assertNotIn("openrouter", _OPENAI_SHAPED_STREAM_PROVIDERS)

    def test_peers_still_members(self):
        for p in ("openai", "groq", "together", "huggingface", "litellm", "mistral"):
            self.assertIn(p, _OPENAI_SHAPED_STREAM_PROVIDERS)

    def test_accumulator_created(self):
        acc = _new_stream_accumulator("cerebras")
        self.assertIsNotNone(acc)
        self.assertIn("text_parts", acc)
        self.assertIn("tool_calls", acc)

    def test_content_deltas_compose_assistant_text(self):
        acc = _new_stream_accumulator("cerebras")
        _accumulate_stream_chunk("cerebras", acc, _Chunk(content="Hello "))
        _accumulate_stream_chunk("cerebras", acc, _Chunk(content="world"))
        composed = _stream_accumulator_to_response("cerebras", acc)
        self.assertIsNotNone(composed)
        self.assertEqual(composed["choices"][0]["message"]["content"], "Hello world")
        self.assertEqual(composed["choices"][0]["message"]["role"], "assistant")

    def test_tool_call_deltas_compose(self):
        acc = _new_stream_accumulator("cerebras")
        _accumulate_stream_chunk(
            "cerebras",
            acc,
            _Chunk(
                tool_calls=[
                    _ToolCall(index=0, id="call_1", name="lookup", arguments='{"q":'),
                ]
            ),
        )
        _accumulate_stream_chunk(
            "cerebras",
            acc,
            _Chunk(tool_calls=[_ToolCall(index=0, arguments='"x"}')]),
        )
        composed = _stream_accumulator_to_response("cerebras", acc)
        self.assertIsNotNone(composed)
        tcs = composed["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(tcs), 1)
        self.assertEqual(tcs[0]["id"], "call_1")
        self.assertEqual(tcs[0]["function"]["name"], "lookup")
        self.assertEqual(tcs[0]["function"]["arguments"], '{"q":"x"}')

    def test_empty_stream_returns_none(self):
        # Fail-open: finalize falls back to last usage chunk when nothing rendered.
        acc = _new_stream_accumulator("cerebras")
        self.assertIsNone(_stream_accumulator_to_response("cerebras", acc))
        # Role-only / empty content chunks must not invent a message.
        _accumulate_stream_chunk("cerebras", acc, _Chunk(content=None))
        self.assertIsNone(_stream_accumulator_to_response("cerebras", acc))


if __name__ == "__main__":
    unittest.main()
