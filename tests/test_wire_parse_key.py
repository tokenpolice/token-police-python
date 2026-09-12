"""Serving provider vs wire/parse key (OpenAI-compatible gateways).

After host remap feeds serving slugs (minimax, xai, …) into Mode-A
composition and stream accumulators. Those must key on the **module** client
(OpenAI wire shape), not the billing vendor.

Also covers Python cohere stream accumulator.
"""
import unittest
from types import SimpleNamespace

from token_police.enforcer import (
    _wire_parse_key,
    _mode_a_new_accumulator,
    _mode_a_accumulate,
    _mode_a_synthetic_response,
    _new_stream_accumulator,
    _accumulate_stream_chunk,
    _stream_accumulator_to_response,
    _OPENAI_SHAPED_STREAM_PROVIDERS,
)
from token_police.composition import (
    build_prompt_composition,
    build_response_composition,
    extract_pending_tool_calls,
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


class TestWireParseKey(unittest.TestCase):
    def test_wire_parse_key_is_module_identity(self):
        self.assertEqual(_wire_parse_key("openai"), "openai")
        self.assertEqual(_wire_parse_key("anthropic"), "anthropic")
        self.assertEqual(_wire_parse_key("xai"), "xai")

    def test_mode_a_accumulator_openai_not_serving_minimax(self):
        # Module/wire key — what the fix passes into Mode-A wrappers
        acc = _mode_a_new_accumulator("openai")
        self.assertIsNotNone(acc)
        # Serving slug (pre-fix bug): no Mode-A branch → None → empty composition
        self.assertIsNone(_mode_a_new_accumulator("minimax"))
        self.assertIsNone(_mode_a_new_accumulator("xai"))

    def test_mode_a_openai_stream_composes_under_wire_key(self):
        acc = _mode_a_new_accumulator("openai")
        _mode_a_accumulate("openai", acc, _Chunk(content="Hello "))
        _mode_a_accumulate("openai", acc, _Chunk(content="world"))
        synthetic = _mode_a_synthetic_response("openai", acc)
        self.assertIsNotNone(synthetic)
        self.assertEqual(
            synthetic["choices"][0]["message"]["content"], "Hello world"
        )
        comp = build_response_composition("openai", synthetic)
        self.assertTrue(any(e.get("role") == "assistant" for e in comp))

    def test_openai_messages_composition_not_xai_protobuf(self):
        # OpenAI SDK pointed at api.x.ai — messages are OpenAI dicts.
        # Wire key openai → OpenAI parser. Serving xai → protobuf parser (empty).
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        openai_comp = build_prompt_composition("openai", {"messages": messages})
        self.assertGreaterEqual(len(openai_comp), 2)
        self.assertEqual(
            [e.get("role") for e in openai_comp[:2]],
            ["system", "user"],
        )

        # xai protobuf path expects protobuf Message objects, not dicts —
        # with plain OpenAI dicts it yields empty.
        xai_comp = build_prompt_composition("xai", {"messages": messages})
        self.assertEqual(xai_comp, [])
        # Wire key identity: module openai stays openai
        self.assertEqual(_wire_parse_key("openai"), "openai")

    def test_openai_shaped_allowlist_unchanged_no_host_slugs(self):
        # Anti-pattern check: do NOT paper over by adding host-map slugs.
        for slug in ("minimax", "xai", "deepseek", "openrouter"):
            self.assertNotIn(slug, _OPENAI_SHAPED_STREAM_PROVIDERS)


class TestCohereStreamAccumulator(unittest.TestCase):
    """Instance 5 — Python cohere stream composition (Node parity)."""

    def test_accumulator_created(self):
        acc = _new_stream_accumulator("cohere")
        self.assertIsNotNone(acc)
        self.assertIn("text_parts", acc)
        self.assertIn("tool_calls", acc)
        self.assertIn("tool_plan_parts", acc)

    def test_content_delta_composes(self):
        acc = _new_stream_accumulator("cohere")
        _accumulate_stream_chunk(
            "cohere",
            acc,
            SimpleNamespace(
                type="content-delta",
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        content=SimpleNamespace(text="Hello ")
                    )
                ),
            ),
        )
        _accumulate_stream_chunk(
            "cohere",
            acc,
            SimpleNamespace(
                type="content-delta",
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        content=SimpleNamespace(text="world")
                    )
                ),
            ),
        )
        resp = _stream_accumulator_to_response("cohere", acc)
        self.assertIsNotNone(resp)
        # Privacy-preserving entries: length+hash, never raw text.
        # Accumulator joins fragments before parse → one assistant text entry.
        comp = build_response_composition("cohere", resp)
        assistant = [e for e in comp if e.get("role") == "assistant" and e.get("type") == "text"]
        self.assertGreaterEqual(len(assistant), 1)
        # "Hello " + "world" → "Hello world" (len 11, trim does not change interior)
        self.assertEqual(assistant[0]["length"], len("Hello world"))

    def test_tool_call_deltas_compose_ids(self):
        acc = _new_stream_accumulator("cohere")
        _accumulate_stream_chunk(
            "cohere",
            acc,
            SimpleNamespace(
                type="tool-call-start",
                index=0,
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=SimpleNamespace(
                            id="tc_abc",
                            function=SimpleNamespace(
                                name="lookup", arguments=""
                            ),
                        )
                    )
                ),
            ),
        )
        _accumulate_stream_chunk(
            "cohere",
            acc,
            SimpleNamespace(
                type="tool-call-delta",
                index=0,
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        tool_calls=SimpleNamespace(
                            function=SimpleNamespace(arguments='{"q":"x"}')
                        )
                    )
                ),
            ),
        )
        resp = _stream_accumulator_to_response("cohere", acc)
        self.assertIsNotNone(resp)
        msg = resp.message
        tcs = msg.get("tool_calls")
        self.assertIsNotNone(tcs)
        self.assertEqual(tcs[0]["id"], "tc_abc")
        self.assertEqual(tcs[0]["function"]["name"], "lookup")
        self.assertEqual(tcs[0]["function"]["arguments"], '{"q":"x"}')
        pending = extract_pending_tool_calls("cohere", resp)
        self.assertTrue(any(p.get("id") == "tc_abc" for p in pending))


if __name__ == "__main__":
    unittest.main()
