"""T-P1 — wire-shape × serving-slug matrix (sdk_test_hardening_design.md §4.2).

Regression class: after host remap, serving slugs (minimax, xai, …) must never
be used as the parse/accumulate key. Wire key stays the module client
(`_wire_parse_key("openai")` → openai). This matrix locks every OpenAI-
compatible host-map slug so the next serving-provider mapping cannot silently
empty stream composition again.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from token_police.enforcer import (
    _HOST_EXACT,
    _HOST_PATTERNS,
    _OPENAI_SHAPED_STREAM_PROVIDERS,
    _accumulate_stream_chunk,
    _match_host_to_provider,
    _new_stream_accumulator,
    _stream_accumulator_to_response,
    _wire_parse_key,
    resolve_serving_from_base_url,
)
from token_police.composition import (
    build_prompt_composition,
    build_response_composition,
    extract_pending_tool_calls,
)


# Design T-P1 / T-N1 list (OpenAI-compatible client reachable serving slugs).
OPENAI_COMPAT_SERVING = (
    "minimax",
    "xai",
    "deepseek",
    "moonshot",
    "zhipu",
    "perplexity",
    "fireworks",
    "deepinfra",
    "novita",
    "nebius",
    "vercel-gateway",
    "azure-openai",
    "azure-ai",
    "self_hosted",
    "openrouter",
    "together",
    "cerebras",
    "openai",
    "groq",
)

HOST_EXAMPLES = {
    "minimax": "api.minimax.io",
    "xai": "api.x.ai",
    "deepseek": "api.deepseek.com",
    "moonshot": "api.moonshot.ai",
    "zhipu": "open.bigmodel.cn",
    "perplexity": "api.perplexity.ai",
    "fireworks": "api.fireworks.ai",
    "deepinfra": "api.deepinfra.com",
    "novita": "api.novita.ai",
    "nebius": "api.studio.nebius.com",
    "vercel-gateway": "ai-gateway.vercel.sh",
    "openrouter": "openrouter.ai",
    "together": "api.together.xyz",
    "cerebras": "api.cerebras.ai",
    "openai": "api.openai.com",
    "groq": "api.groq.com",
}


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


def _shared_types() -> dict:
    path = Path(__file__).resolve().parents[2] / "shared" / "provider-slugs.json"
    data = json.loads(path.read_text())
    return data["types"]


def _host_map_slugs() -> set:
    slugs = set(_HOST_EXACT.values())
    for _pat, provider in _HOST_PATTERNS:
        slugs.add(provider)
    return slugs


class TestHostMapExhaustiveness(unittest.TestCase):
    """T-N2 twin: every host-map slug is classified; wire key is module identity."""

    def test_every_host_map_slug_in_shared_types(self):
        types = _shared_types()
        missing = sorted(s for s in _host_map_slugs() if s not in types)
        self.assertEqual(missing, [], f"host-map slugs missing from shared types: {missing}")

    def test_wire_parse_key_is_module_identity(self):
        for slug in _host_map_slugs():
            self.assertEqual(_wire_parse_key("openai"), "openai")
            self.assertEqual(_wire_parse_key("anthropic"), "anthropic")
            # Identity on the slug itself — never host-derived remapping.
            self.assertEqual(_wire_parse_key(slug), slug)

    def test_host_examples_resolve(self):
        for slug, host in HOST_EXAMPLES.items():
            self.assertEqual(_match_host_to_provider(host), slug, host)
            resolved = resolve_serving_from_base_url(f"https://{host}/v1")
            self.assertEqual(resolved.get("kind"), "recognized", host)
            self.assertEqual(resolved.get("provider"), slug, host)


class TestWireShapeMatrix(unittest.TestCase):
    """T-P1 core: OpenAI-shaped stream under wire key for every serving slug."""

    def test_openai_shaped_allowlist_excludes_host_map_serving_slugs(self):
        # Anti-pattern: do NOT paper over by expanding the allowlist.
        for slug in ("minimax", "xai", "deepseek", "openrouter", "moonshot", "zhipu"):
            self.assertNotIn(slug, _OPENAI_SHAPED_STREAM_PROVIDERS)

    def test_accumulator_none_under_serving_present_under_wire(self):
        wire = _wire_parse_key("openai")
        self.assertEqual(wire, "openai")
        self.assertIsNotNone(_new_stream_accumulator(wire))

        # xai has a *native* (protobuf) stream accumulator — not OpenAI-shaped.
        # Using it for OpenAI-SDK→api.x.ai traffic is the instance-3 bug;
        # the wire key must stay openai. Do not assert None for xai itself.
        native_non_openai_acc = {"xai", "cohere", "openai_responses"}

        for serving in OPENAI_COMPAT_SERVING:
            if serving in _OPENAI_SHAPED_STREAM_PROVIDERS:
                # Module allowlist members (openai, cerebras, together, groq, …)
                self.assertIsNotNone(
                    _new_stream_accumulator(serving),
                    f"module allowlist member {serving}",
                )
            elif serving in native_non_openai_acc:
                # Native accumulator exists but is the wrong wire shape for
                # OpenAI-SDK clients — composition under OpenAI dicts is empty
                # (covered by TestXaiDisambiguation).
                self.assertIsNotNone(_new_stream_accumulator(serving), serving)
            else:
                # Pure serving slugs from host map have no openai-shaped branch
                self.assertIsNone(
                    _new_stream_accumulator(serving),
                    f"serving={serving} must not get openai accumulator (use wire key)",
                )

    def test_openai_deltas_compose_under_wire_for_every_serving_context(self):
        """Round-trip OpenAI-shaped deltas under wire=openai for each serving label.

        The serving label is only documentation here — the parse key is always
        the module wire key. Before the fix fix, callers passed serving and got None.
        """
        for serving in OPENAI_COMPAT_SERVING:
            with self.subTest(serving=serving):
                wire = _wire_parse_key("openai")
                acc = _new_stream_accumulator(wire)
                self.assertIsNotNone(acc, serving)

                _accumulate_stream_chunk(wire, acc, _Chunk(content="Hello "))
                # Object-shaped tool_call delta (SDK objects, not plain dicts).
                tc = SimpleNamespace(
                    index=0,
                    id=f"call_{serving}",
                    type="function",
                    function=SimpleNamespace(
                        name="lookup", arguments='{"q":"x"}'
                    ),
                )
                _accumulate_stream_chunk(
                    wire,
                    acc,
                    SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                delta=SimpleNamespace(content=None, tool_calls=[tc])
                            )
                        ]
                    ),
                )
                _accumulate_stream_chunk(wire, acc, _Chunk(content="world"))

                synthetic = _stream_accumulator_to_response(wire, acc)
                self.assertIsNotNone(synthetic, serving)
                comp = build_response_composition(wire, synthetic)
                self.assertTrue(
                    any(
                        e.get("role") in ("assistant", "tool_call")
                        for e in comp
                    ),
                    f"serving={serving} empty composition under wire key: {comp!r}",
                )
                pending = extract_pending_tool_calls(wire, synthetic) or []
                ids = {p.get("id") for p in pending if isinstance(p, dict)}
                self.assertIn(
                    f"call_{serving}",
                    ids,
                    f"serving={serving}: tool id lost; pending={pending!r} comp={comp!r}",
                )


class TestXaiDisambiguation(unittest.TestCase):
    """OpenAI-shaped dicts vs xai-sdk protobuf — two wire shapes, one host."""

    def test_openai_dicts_parse_via_openai_wire_key(self):
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

    def test_serving_xai_on_openai_dicts_yields_empty(self):
        # Pre-fix symptom: serving slug routed to protobuf parser → [].
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        xai_comp = build_prompt_composition("xai", {"messages": messages})
        self.assertEqual(xai_comp, [])
        # Wire key for OpenAI SDK at api.x.ai is still openai.
        self.assertEqual(_wire_parse_key("openai"), "openai")


class TestCohereAccumulatorParity(unittest.TestCase):
    """Instance 5 / Node parity — cohere is not OpenAI-shaped."""

    def test_cohere_accumulator_exists_and_composes(self):
        acc = _new_stream_accumulator("cohere")
        self.assertIsNotNone(acc)
        _accumulate_stream_chunk(
            "cohere",
            acc,
            SimpleNamespace(
                type="content-delta",
                delta=SimpleNamespace(
                    message=SimpleNamespace(
                        content=SimpleNamespace(text="Hi")
                    )
                ),
            ),
        )
        resp = _stream_accumulator_to_response("cohere", acc)
        self.assertIsNotNone(resp)
        comp = build_response_composition("cohere", resp)
        self.assertTrue(any(e.get("role") == "assistant" for e in comp))

    def test_cohere_not_in_openai_shaped_set(self):
        self.assertNotIn("cohere", _OPENAI_SHAPED_STREAM_PROVIDERS)


if __name__ == "__main__":
    unittest.main()
