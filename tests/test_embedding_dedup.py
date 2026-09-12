"""Tests for embedding-span de-duplication + (google-genai).

opentelemetry-instrumentation-openai / -cohere patch embeddings and emit an
OTel span tagged `llm.request.type="embedding"` (span names "openai.embeddings"
/ "cohere.embed"). google-genai instrumentor ≥1.0b1 emits GenAI-semconv spans
with `gen_ai.operation.name="embeddings"`. TokenPolice captures embeddings
authoritatively via its own manual wrapper (which creates NO OTel span), so any
such span reaching the SpanProcessor is the instrumentor's spurious duplicate
and must be dropped — otherwise it is mis-logged as a chat-shaped row.
"""
import unittest

from token_police.telemetry import _is_instrumentor_embedding_span


class TestEmbeddingSpanDetection(unittest.TestCase):
    def test_llm_request_type_embedding(self):
        self.assertTrue(_is_instrumentor_embedding_span({"llm.request.type": "embedding"}, "anything"))

    def test_llm_request_type_embedding_uppercase(self):
        self.assertTrue(_is_instrumentor_embedding_span({"llm.request.type": "EMBEDDING"}, ""))

    def test_openai_embeddings_span_name(self):
        self.assertTrue(_is_instrumentor_embedding_span({}, "openai.embeddings"))

    def test_cohere_embed_span_name(self):
        self.assertTrue(_is_instrumentor_embedding_span({}, "cohere.embed"))

    def test_generic_dot_embeddings_name(self):
        self.assertTrue(_is_instrumentor_embedding_span({}, "something.embeddings"))

    # (sdk production-release) — google-genai instrumentor ≥1.0b1 patches
    # embed_content and emits a GenAI-semconv span (gen_ai.operation.name=
    # "embeddings", span name "embeddings <model>"). Without this gate the span
    # is mis-logged as llm/chat alongside the authoritative manual embedding row.
    def test_gen_ai_operation_name_embeddings(self):
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.operation.name": "embeddings",
             "gen_ai.request.model": "gemini-embedding-001"},
            "embeddings gemini-embedding-001"))

    def test_gen_ai_operation_name_embeddings_case_insensitive(self):
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.operation.name": "EMBEDDINGS"}, "anything"))

    def test_gen_ai_operation_name_embedding_singular(self):
        # Forward-compat: accept singular "embedding" as well as semconv "embeddings".
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.operation.name": "embedding"}, ""))

    def test_embeddings_span_name_with_model(self):
        # util-genai names the span f"{operation} {model}"; operation attr may be absent.
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.request.model": "gemini-embedding-001"},
            "embeddings gemini-embedding-001"))

    def test_embeddings_span_name_bare(self):
        self.assertTrue(_is_instrumentor_embedding_span({}, "embeddings"))

    def test_google_genai_embed_scope_not_required(self):
        # 1.0b1 tracer scope is opentelemetry.util.genai.handler, not
        # opentelemetry.instrumentation.google_genai — operation.name alone drops it.
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.operation.name": "embeddings",
             "gen_ai.request.model": "gemini-embedding-001"},
            "embeddings gemini-embedding-001",
            "opentelemetry.util.genai.handler"))

    def test_generate_content_is_not_embedding(self):
        # Real Google chat spans must survive the gate.
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.operation.name": "generate_content",
             "gen_ai.request.model": "gemini-2.0-flash"},
            "generate_content gemini-2.0-flash"))

    def test_generate_content_span_name_not_dropped(self):
        self.assertFalse(_is_instrumentor_embedding_span(
            {}, "generate_content gemini-2.0-flash"))

    def test_unscoped_gemini_embedding_model_id_survives_without_op(self):
        # Model-id "embed" alone (no gen_ai.operation.name, no bedrock scope) must
        # NOT drop — same keep matrix as other non-bedrock systems.
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.request.model": "gemini-embedding-001"},
            "gemini-embedding-001"))

    # The Bedrock instrumentor emits a chat/converse-shaped span on the
    # embeddings InvokeModel call (no llm.request.type=embedding, no .embeddings
    # name) tagged with the vendor-stripped embedding model id. Catch it by model
    # id — but ONLY when the span is attributable to the Bedrock/Voyage
    # instrumentor (gen_ai.system aws/bedrock/voyage, or the scope name), so real
    # spans from other systems whose model id merely contains "embed" survive.
    def test_bedrock_titan_embed_model_id(self):
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.system": "aws", "gen_ai.request.model": "titan-embed-text-v2:0"}, ""))

    def test_bedrock_cohere_embed_model_id(self):
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.system": "bedrock", "gen_ai.request.model": "embed-english-v3"}, ""))

    def test_bedrock_voyage_embed_model_id(self):
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.system": "aws.bedrock", "gen_ai.request.model": "voyage-3"}, ""))

    def test_bedrock_embed_by_scope_name(self):
        # gen_ai.system absent — the instrumentation scope name attributes it.
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.request.model": "titan-embed-text-v2:0"}, "",
            "opentelemetry.instrumentation.bedrock"))

    def test_voyage_embed_by_scope_name(self):
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.request.model": "voyage-3"}, "",
            "opentelemetry.instrumentation.voyage"))

    def test_legacy_llm_request_model_key(self):
        self.assertTrue(_is_instrumentor_embedding_span(
            {"gen_ai.system": "aws", "llm.request.model": "titan-embed-text-v2:0"}, ""))

    # Keep/drop matrix — an "embed"/"voyage" model id from a NON-bedrock/voyage
    # system is a genuine span and must NOT be dropped.
    def test_other_system_embed_model_id_survives(self):
        # An OpenAI-attributed span whose model id contains "embed" is NOT the
        # bedrock duplicate (its embeddings are already caught by name/type).
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.system": "openai", "gen_ai.request.model": "text-embedding-3-small"}, ""))

    def test_other_system_voyage_prefixed_model_survives(self):
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.system": "custom", "gen_ai.request.model": "voyage-chat-mini"}, ""))

    def test_unattributed_embed_model_id_survives(self):
        # No system, no scope → cannot attribute to bedrock/voyage → keep.
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.request.model": "titan-embed-text-v2:0"}, ""))

    def test_bedrock_chat_models_are_not_embedding(self):
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.system": "aws", "gen_ai.request.model": "nova-lite-v1:0"}, ""))
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.system": "aws", "gen_ai.request.model": "claude-3-5-sonnet"}, ""))
        self.assertFalse(_is_instrumentor_embedding_span(
            {"gen_ai.system": "aws", "gen_ai.request.model": "titan-text-express-v1"}, ""))

    def test_chat_span_is_not_embedding(self):
        self.assertFalse(_is_instrumentor_embedding_span({"llm.request.type": "chat"}, "openai.chat"))

    def test_plain_chat_span(self):
        self.assertFalse(_is_instrumentor_embedding_span({"gen_ai.system": "openai"}, "chat gpt-4o-mini"))

    def test_empty_inputs(self):
        self.assertFalse(_is_instrumentor_embedding_span({}, ""))
        self.assertFalse(_is_instrumentor_embedding_span({}, None))


if __name__ == "__main__":
    unittest.main()
