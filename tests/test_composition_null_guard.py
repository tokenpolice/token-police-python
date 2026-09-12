"""Fix 3 — message parsers skip null/primitive elements instead of throwing.

Twin of the Node compositionNullGuard.test.ts. A single None/int/str element in
a messages array used to raise (vars(None) / None.get) and the outer catch
collapsed the WHOLE composition to a coarse single-entry fallback, losing
per-message granularity for the valid messages. The parsers now skip such
elements and preserve entries for the valid ones.
"""
import unittest

from token_police.composition import build_prompt_composition


class TestOpenAINullGuard(unittest.TestCase):
    def test_null_and_primitive_elements_skipped(self):
        comp = build_prompt_composition("openai", {"messages": [
            None,
            {"role": "user", "content": "hello"},
            42,
            "loose-string",
            {"role": "assistant", "content": "world"},
        ]})
        # Exactly the two valid object messages produce text entries.
        self.assertEqual(len(comp), 2)
        self.assertEqual(comp[0]["role"], "user")
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[1]["role"], "assistant")
        self.assertEqual(comp[1]["type"], "text")

    def test_leading_null_does_not_collapse_to_fallback(self):
        comp = build_prompt_composition("openai", {"messages": [
            None,
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]})
        # Per-message granularity preserved (2 entries), not one coarse fallback.
        self.assertEqual(len(comp), 2)

    def test_all_null_yields_empty_composition_never_raises(self):
        comp = build_prompt_composition("openai", {"messages": [None, None]})
        self.assertEqual(comp, [])

    def test_float_and_bool_elements_skipped(self):
        comp = build_prompt_composition("openai", {"messages": [
            3.14, True, {"role": "user", "content": "ok"},
        ]})
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["role"], "user")


class TestAnthropicNullGuard(unittest.TestCase):
    def test_null_and_primitive_elements_skipped(self):
        comp = build_prompt_composition("anthropic", {"messages": [
            None,
            {"role": "user", "content": "hi"},
            7,
            {"role": "assistant", "content": [{"type": "text", "text": "yo"}]},
        ]})
        self.assertEqual(len(comp), 2)
        self.assertEqual(comp[0]["role"], "user")
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[1]["role"], "assistant")
        self.assertEqual(comp[1]["type"], "text")

    def test_all_null_yields_empty_never_raises(self):
        comp = build_prompt_composition("anthropic", {"messages": [None, "str", 1]})
        self.assertEqual(comp, [])


if __name__ == "__main__":
    unittest.main()
