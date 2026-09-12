"""G5 — Python SpanProcessor must forward cache-*creation* (write) tokens
SEPARATELY from cache-*read* tokens, and must extract reasoning tokens.

The Mode-A `TokenPoliceSpanProcessor.on_end` reconstructs a usage block from the
`gen_ai.*` semconv attributes an instrumentor emits. Previously it summed
cache_read + cache_creation into one `cached_tokens` and forwarded the sum as
`cache_read_input_tokens` only — so the server's Anthropic mapper saw
`cache_write_5m_tokens = 0` and billed every cache-*write* token at the cache-
*read* rate (~0.1x input instead of ~1.25x → ~12x under-count). It also never
forwarded reasoning tokens.

These assert the corrected, DISJOINT contract on the synthesized usage block:
  raw.cache_read_input_tokens == cache-read hits only
  raw.cache_creation_input_tokens == cache-creation (write) tokens
  raw.completion_tokens_details.reasoning_tokens == reasoning tokens
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import telemetry
from token_police.telemetry import TokenPoliceSpanProcessor


def _fake_span(attrs, name="anthropic.chat"):
    """A minimal finished span: only `.attributes`, `.name`, and the optional
    `.instrumentation_scope`/`.context`/`.parent` the processor reads."""
    return NS(
        attributes=attrs,
        name=name,
        instrumentation_scope=NS(name="opentelemetry.instrumentation.anthropic"),
        context=NS(trace_id=0x1234, span_id=0xABCD),
        parent=None,
        start_time=0,
        end_time=1,
    )


def _capture_log(span):
    """Run on_end over `span`, capturing the kwargs passed to client.log_sync."""
    captured = {}

    class _FakeClient:
        def log_sync(self, **kwargs):
            captured.update(kwargs)

    proc = TokenPoliceSpanProcessor()
    with mock.patch("token_police.state.get_client", return_value=_FakeClient()):
        proc.on_end(span)
    return captured


class TestCacheCreationDisjoint(unittest.TestCase):
    def test_cache_write_forwarded_separately_from_read(self):
        attrs = {
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-3-5-sonnet-20241022",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 50,
            "gen_ai.usage.cache_read_input_tokens": 200,
            "gen_ai.usage.cache_creation_input_tokens": 1000,
        }
        payload = _capture_log(_fake_span(attrs))
        raw = payload["usage"]["raw"]
        # Read and write MUST be disjoint — the write must NOT be folded into read.
        self.assertEqual(raw.get("cache_read_input_tokens"), 200)
        self.assertEqual(raw.get("cache_creation_input_tokens"), 1000)
        # OpenAI-semantics cached_tokens counts read hits only.
        self.assertEqual(raw["prompt_tokens_details"]["cached_tokens"], 200)

    def test_dotted_keys_also_split(self):
        attrs = {
            "gen_ai.system": "anthropic",
            "gen_ai.request.model": "claude-3-5-sonnet-20241022",
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 5,
            "gen_ai.usage.cache_read.input_tokens": 30,
            "gen_ai.usage.cache_creation.input_tokens": 70,
        }
        raw = _capture_log(_fake_span(attrs))["usage"]["raw"]
        self.assertEqual(raw.get("cache_read_input_tokens"), 30)
        self.assertEqual(raw.get("cache_creation_input_tokens"), 70)

    def test_reasoning_tokens_extracted(self):
        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "o1-mini",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 800,
            "gen_ai.usage.reasoning_tokens": 600,
        }
        raw = _capture_log(_fake_span(attrs))["usage"]["raw"]
        self.assertEqual(
            raw["completion_tokens_details"]["reasoning_tokens"], 600
        )

    def test_no_cache_no_phantom_fields(self):
        # A plain call must not sprout cache/reasoning keys.
        attrs = {
            "gen_ai.system": "openai",
            "gen_ai.request.model": "gpt-4o-mini",
            "gen_ai.usage.input_tokens": 10,
            "gen_ai.usage.output_tokens": 5,
        }
        raw = _capture_log(_fake_span(attrs))["usage"]["raw"]
        self.assertNotIn("cache_read_input_tokens", raw)
        self.assertNotIn("cache_creation_input_tokens", raw)
        self.assertNotIn("completion_tokens_details", raw)


if __name__ == "__main__":
    unittest.main()
