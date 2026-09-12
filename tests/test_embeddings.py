"""Tests for embeddings support in the Python SDK.

Covers (a) the embedding registry entries land in _TARGET_METHODS with the
expected provider×shape×operation tuples, (b) _extract_embedding_usage
returns the correct (model, input_tokens, raw) shape per provider, (c) the
composition layer's operation="embedding" mode produces input-only entries
and the response side returns [] for known embedding shapes.

This is the SDK contract that the server's usage-mapper depends on —
adding a new embedding provider requires both a registry entry and an
extractor branch.
"""
import unittest

from token_police.enforcer import (
    _TARGET_METHODS,
    _extract_embedding_usage,
    _resolve_embedding_shape,
    _approximate_hf_embedding_tokens,
    _approximate_google_embedding_tokens,
    _approximate_bedrock_embedding_tokens,
    _approx_chars_to_tokens,
    _EMBEDDING_SHAPE_BY_PROVIDER,
)
from token_police.composition import (
    build_prompt_composition,
    build_response_composition,
    _parse_embedding_input,
    _EMBEDDING_SHAPES,
)


class TestEmbeddingRegistry(unittest.TestCase):
    def test_registry_has_embedding_entries(self):
        emb = [t for t in _TARGET_METHODS if t.get("operation") == "embedding"]
        self.assertGreater(len(emb), 0, "no embedding registry entries found")

    def test_every_embedding_entry_is_manual_and_has_shape(self):
        for e in _TARGET_METHODS:
            if e.get("operation") != "embedding":
                continue
            self.assertTrue(e.get("manual"), f"embedding entry must be manual: {e}")
            self.assertIn(
                e.get("shape"),
                {"openai_embeddings", "google_genai_embeddings", "cohere_embed",
                 "mistral_embed", "voyage_embed", "huggingface_embed", "together_embed"},
                f"embedding entry has unknown shape: {e.get('shape')}",
            )

    def test_expected_providers_covered(self):
        modules = {t["module"] for t in _TARGET_METHODS if t.get("operation") == "embedding"}
        # The Phase 1 minimum SDK contract.
        for mod in [
            "openai.resources.embeddings",
            "google.genai.models",
            "cohere.client_v2",
            # BOTH mistralai majors: 2.0 moved every resource module under
            # `mistralai.client.*`, and a row for the other major's path
            # resolves to nothing (silently — `_wrap_method` swallows the
            # ImportError), leaving embedding spend un-checked and un-logged.
            "mistralai.embeddings",
            "mistralai.client.embeddings",
            "together.resources.embeddings",
            "huggingface_hub",
            "litellm",
        ]:
            self.assertIn(mod, modules, f"missing embedding registry entry for {mod}")


class TestEmbeddingShapeResolution(unittest.TestCase):
    def test_known_providers_resolve(self):
        self.assertEqual(_resolve_embedding_shape("openai"), "openai_embeddings")
        self.assertEqual(_resolve_embedding_shape("google"), "google_genai_embeddings")
        self.assertEqual(_resolve_embedding_shape("cohere"), "cohere_embed")
        self.assertEqual(_resolve_embedding_shape("mistral"), "mistral_embed")
        self.assertEqual(_resolve_embedding_shape("voyage"), "voyage_embed")
        self.assertEqual(_resolve_embedding_shape("huggingface"), "huggingface_embed")
        self.assertEqual(_resolve_embedding_shape("together"), "together_embed")

    def test_unknown_provider_falls_back(self):
        self.assertEqual(_resolve_embedding_shape("xai"), "openai_embeddings")
        self.assertEqual(_resolve_embedding_shape(""), "openai_embeddings")


class _FakeUsage:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeResp:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class TestExtractEmbeddingUsage(unittest.TestCase):
    def test_openai(self):
        resp = _FakeResp(model="text-embedding-3-small",
                         usage=_FakeUsage(prompt_tokens=42, total_tokens=42))
        model, tokens, raw = _extract_embedding_usage(resp, "openai")
        self.assertEqual(model, "text-embedding-3-small")
        self.assertEqual(tokens, 42)
        self.assertEqual(raw, {"prompt_tokens": 42, "total_tokens": 42})

    def test_together_uses_openai_path(self):
        resp = _FakeResp(model="BAAI/bge-large-en-v1.5",
                         usage=_FakeUsage(prompt_tokens=8, total_tokens=8))
        model, tokens, _ = _extract_embedding_usage(resp, "together")
        self.assertEqual(model, "BAAI/bge-large-en-v1.5")
        self.assertEqual(tokens, 8)

    def test_google_vertex_per_embedding_stats(self):
        # Vertex API — each embedding carries `.statistics.token_count`.
        # Multiple embeddings in one call should sum.
        stats_a = _FakeUsage(token_count=20)
        stats_b = _FakeUsage(token_count=22)
        emb_a = _FakeResp(statistics=stats_a)
        emb_b = _FakeResp(statistics=stats_b)
        resp = _FakeResp(embeddings=[emb_a, emb_b])
        _, tokens, raw = _extract_embedding_usage(resp, "google")
        self.assertEqual(tokens, 42)
        self.assertEqual(raw["prompt_token_count"], 42)
        self.assertEqual(raw["total_token_count"], 42)

    def test_google_vertex_billable_chars(self):
        # Vertex char-billed surface — `metadata.billable_character_count`
        # only. Should approximate to chars/4 and mark approximated=True.
        meta = _FakeUsage(billable_character_count=200)
        resp = _FakeResp(embeddings=[], metadata=meta)
        _, tokens, raw = _extract_embedding_usage(resp, "google")
        self.assertEqual(tokens, 50)
        self.assertTrue(raw.get("approximated"))
        self.assertEqual(raw["billable_character_count"], 200)
        self.assertEqual(raw["approx_input_tokens"], 50)

    def test_google_mldev_no_usage(self):
        # mldev/Gemini API — embed_content returns no usage info at all.
        # Extractor must return (_, 0, None) so the caller approximates
        # from the request kwargs.
        resp = _FakeResp(embeddings=[])
        model, tokens, raw = _extract_embedding_usage(resp, "gemini")
        self.assertEqual(tokens, 0)
        self.assertIsNone(raw)

    def test_google_mldev_no_attributes_at_all(self):
        # Defensive — a barely-shaped response with neither embeddings nor
        # metadata must NOT throw; returns (_, 0, None).
        model, tokens, raw = _extract_embedding_usage(_FakeResp(), "google")
        self.assertEqual(tokens, 0)
        self.assertIsNone(raw)

    def test_cohere_with_images(self):
        # Cohere v2 multimodal — meta.billed_units carries input_tokens + images.
        meta = {"billed_units": {"input_tokens": 12, "images": 3}}
        resp = _FakeResp(meta=meta)
        _, tokens, raw = _extract_embedding_usage(resp, "cohere")
        self.assertEqual(tokens, 12)
        self.assertEqual(raw["meta"]["billed_units"]["input_tokens"], 12)
        self.assertEqual(raw["meta"]["billed_units"]["images"], 3)

    def test_mistral(self):
        resp = _FakeResp(model="mistral-embed",
                         usage=_FakeUsage(prompt_tokens=99, total_tokens=99))
        model, tokens, _ = _extract_embedding_usage(resp, "mistral")
        self.assertEqual(model, "mistral-embed")
        self.assertEqual(tokens, 99)

    def test_voyage_usage_fallback(self):
        # Defensive: some shapes expose usage.total_tokens — still read it.
        resp = _FakeResp(model="voyage-3-large",
                         usage=_FakeUsage(total_tokens=128))
        model, tokens, raw = _extract_embedding_usage(resp, "voyage")
        self.assertEqual(model, "voyage-3-large")
        self.assertEqual(tokens, 128)
        self.assertEqual(raw, {"total_tokens": 128})

    def test_voyage_top_level_total_tokens(self):
        # The real voyageai EmbeddingsObject exposes `total_tokens` at the
        # TOP LEVEL (no `.usage`). The extractor must read it from there.
        resp = _FakeResp(total_tokens=15)  # voyage-3 text/async embed
        _, tokens, raw = _extract_embedding_usage(resp, "voyage")
        self.assertEqual(tokens, 15)
        self.assertEqual(raw, {"total_tokens": 15})

    def test_voyage_multimodal_zero_total_tokens(self):
        # multimodal_embed returns MultimodalEmbeddingsObject; image-only calls
        # carry total_tokens=0 (billed via image_pixels) — 0 is expected.
        resp = _FakeResp(total_tokens=0, text_tokens=0, image_pixels=1000)
        _, tokens, raw = _extract_embedding_usage(resp, "voyage")
        self.assertEqual(tokens, 0)
        self.assertEqual(raw, {"total_tokens": 0})

    def test_huggingface_returns_zero_usage(self):
        # feature_extraction returns a vector with no usage object — the
        # wrapper supplies approx_input_tokens separately.
        model, tokens, raw = _extract_embedding_usage(object(), "huggingface")
        self.assertEqual(tokens, 0)
        self.assertIn("approx_input_tokens", raw)

    def test_fail_safe_on_none(self):
        # The SDK MUST never throw into customer code (feedback_sdk_never_fails_app).
        model, tokens, raw = _extract_embedding_usage(None, "openai")
        self.assertEqual(model, "unknown")
        self.assertEqual(tokens, 0)
        self.assertIsNone(raw)

    def test_fail_safe_on_broken_response(self):
        class _Broken:
            @property
            def usage(self_):
                raise RuntimeError("boom")
        # Must not propagate the RuntimeError.
        try:
            _extract_embedding_usage(_Broken(), "openai")
        except Exception as e:
            self.fail(f"_extract_embedding_usage raised: {e}")


class TestApproximateHfTokens(unittest.TestCase):
    def test_single_string(self):
        # len("hello world") = 11 → 11//4 = 2
        self.assertEqual(_approximate_hf_embedding_tokens((), {"text": "hello world"}), 2)

    def test_list_of_strings(self):
        tokens = _approximate_hf_embedding_tokens((), {"text": ["abcd" * 4, "efgh"]})
        self.assertGreater(tokens, 0)

    def test_missing_input_returns_zero(self):
        self.assertEqual(_approximate_hf_embedding_tokens((), {}), 0)


class TestApproximateBedrockEmbeddingTokens(unittest.TestCase):
    # Used when Bedrock strips Cohere's native meta.billed_units block on the
    # Embed route (AWS-documented behaviour, see
    # docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html).
    # The helper decodes the JSON request body and approximates len/4.

    def test_cohere_style_texts_list(self):
        import json
        body = json.dumps({"texts": ["hello world", "abcd" * 4]}).encode("utf-8")
        api_params = {"modelId": "cohere.embed-english-v3", "body": body}
        # len("hello world")=11 → 2; len("abcdabcdabcdabcd")=16 → 4. Sum=6.
        self.assertEqual(_approximate_bedrock_embedding_tokens(api_params), 6)

    def test_titan_style_input_text(self):
        import json
        body = json.dumps({"inputText": "hello world"}).encode("utf-8")
        api_params = {"modelId": "amazon.titan-embed-text-v2:0", "body": body}
        self.assertEqual(_approximate_bedrock_embedding_tokens(api_params), 2)

    def test_missing_body_returns_zero(self):
        self.assertEqual(_approximate_bedrock_embedding_tokens({"modelId": "cohere.embed-english-v3"}), 0)

    def test_garbage_input_returns_zero_safely(self):
        # SDK must never throw into customer code. Bad inputs → 0.
        self.assertEqual(_approximate_bedrock_embedding_tokens(None), 0)
        self.assertEqual(_approximate_bedrock_embedding_tokens({"body": object()}), 0)


class TestApproximateGoogleEmbeddingTokens(unittest.TestCase):
    # Used on the mldev/Gemini API path where embed_content returns no usage
    # info — the SDK approximates from the request `contents` kwarg.

    def test_single_string(self):
        # len("hello world") = 11 → 11//4 = 2
        self.assertEqual(
            _approximate_google_embedding_tokens((), {"contents": "hello world"}),
            2,
        )

    def test_list_of_strings(self):
        tokens = _approximate_google_embedding_tokens(
            (), {"contents": ["foo bar baz", "qux"]},
        )
        self.assertGreater(tokens, 0)

    def test_content_with_parts(self):
        # Google `Content` shape — list of objects each with `parts[i].text`.
        class _Part:
            text = "abcd" * 8  # 32 chars → 8 tokens

        class _Content:
            parts = [_Part(), _Part()]

        tokens = _approximate_google_embedding_tokens((), {"contents": _Content()})
        self.assertEqual(tokens, 16)

    def test_missing_input_returns_zero(self):
        self.assertEqual(_approximate_google_embedding_tokens((), {}), 0)

    def test_garbage_input_returns_zero_safely(self):
        # SDK must never throw into customer code. Bad inputs → 0.
        self.assertEqual(
            _approximate_google_embedding_tokens((), {"contents": object()}),
            0,
        )
        self.assertEqual(
            _approximate_google_embedding_tokens(None, None),
            0,
        )

    def test_approx_chars_to_tokens_edge_cases(self):
        # Direct unit test for the recursive helper.
        self.assertEqual(_approx_chars_to_tokens(None), 0)
        self.assertEqual(_approx_chars_to_tokens(""), 1)  # max(1, 0//4) = 1
        self.assertEqual(_approx_chars_to_tokens("a" * 100), 25)
        self.assertEqual(_approx_chars_to_tokens(["a" * 8, "b" * 8]), 4)


class TestEmbeddingComposition(unittest.TestCase):
    def test_string_input(self):
        comp = build_prompt_composition("openai", {"input": "hello"}, operation="embedding")
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["role"], "input")
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[0]["length"], 5)
        self.assertIn("hash", comp[0])

    def test_list_of_strings(self):
        comp = build_prompt_composition("openai", {"input": ["foo", "bar"]}, operation="embedding")
        self.assertEqual(len(comp), 2)
        for c in comp:
            self.assertEqual(c["role"], "input")
            self.assertEqual(c["type"], "text")

    def test_pre_tokenized_input(self):
        # OpenAI accepts list[int] for pre-tokenized embeddings.
        comp = build_prompt_composition("openai", {"input": [1, 2, 3, 4, 5]}, operation="embedding")
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["role"], "input")
        self.assertEqual(comp[0]["length"], 5)
        self.assertNotIn("hash", comp[0])

    def test_cohere_multimodal_segments(self):
        segments = [
            {"type": "text", "text": "a cat"},
            {"type": "image_url", "image_url": "http://example.com/img"},
        ]
        comp = build_prompt_composition("cohere", {"inputs": segments}, operation="embedding")
        self.assertEqual(len(comp), 2)
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[1]["type"], "image")
        for c in comp:
            self.assertEqual(c["role"], "input")

    def test_voyage_texts_param(self):
        comp = build_prompt_composition("voyage", {"texts": ["doc 1", "doc 2"]}, operation="embedding")
        self.assertEqual(len(comp), 2)

    def test_response_composition_empty_for_embedding_shapes(self):
        # Authoritative override — any embedding shape returns [] regardless
        # of the response object.
        for shape in _EMBEDDING_SHAPES:
            result = build_response_composition("openai", object(), usage_shape=shape)
            self.assertEqual(result, [], f"shape {shape} should return [] but got {result}")

    def test_chat_path_unchanged(self):
        # Sanity — the chat-message path still works when operation != "embedding".
        comp = build_prompt_composition(
            "openai", {"messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["role"], "user")

    def test_empty_input_returns_empty(self):
        self.assertEqual(_parse_embedding_input("openai", {}), [])
        self.assertEqual(_parse_embedding_input("openai", {"input": None}), [])


class TestEmbeddingShapesSet(unittest.TestCase):
    def test_all_provider_shapes_in_embedding_shapes_set(self):
        # Every value the per-provider table emits must be recognised by the
        # response-side short-circuit, otherwise the response composition
        # parser will fall through to a chat parser and mis-decode the vector.
        for shape in _EMBEDDING_SHAPE_BY_PROVIDER.values():
            self.assertIn(shape, _EMBEDDING_SHAPES)


if __name__ == "__main__":
    unittest.main()
