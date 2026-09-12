"""The pydantic_ai manual /log path must price Anthropic cache-WRITE tokens
correctly.

An Anthropic call routed through pydantic_ai reports its token counts in a
`RequestUsage` whose `.details` carries the native Anthropic keys — including
`cache_creation_input_tokens` (cache WRITES) and `cache_read_input_tokens`
(cache READS). Previously `_log_pydantic_ai` folded the writes into a single
`cached_tokens` bucket and forwarded NO usage shape, so the write tokens were
billed at the (~10x cheaper) cache-READ rate — a systematic under-count.

Fix: when the Anthropic usage records a positive cache-WRITE count, forward
`usage={"shape": "anthropic_messages", "raw": <native detail keys>}` and set
`cached_tokens` to reads-only (writes priced via raw). Every other case
(non-anthropic provider, no cache write, missing/odd details, hostile input)
stays byte-identical to today: no usage shape, counts straight from
`_pa_usage_to_counts`.

Crucially the raw block is built from `usage.details` (native, `input_tokens`
EXCLUSIVE of cache), NOT the top-level `RequestUsage.input_tokens` (which is
cache-INCLUSIVE) — forwarding the inclusive value would double-count cache.

Offline: drives `_log_pydantic_ai` directly with a capturing fake `tp`. The
input-exclusivity pin runs against the REAL installed pydantic_ai.
"""
import json
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer


class _CaptureTP:
    """Captures the kwargs of the single log_sync the manual path emits."""

    def __init__(self):
        self.captured = {}

    def log_sync(self, **kw):
        self.captured.update(kw)


def _session():
    return NS(
        user_id="u1",
        paid_plan=False,
        workflow_name="wf",
        session_id="s1",
        metadata={},
        trace_id="0" * 32,
        root_span_id="0" * 16,
    )


def _model():
    return NS(system="anthropic", model_name="claude-3-5-sonnet")


def _response(usage, provider_name="anthropic", model_name="claude-3-5-sonnet"):
    """A ModelResponse-like / StreamedResponse.get()-like object: both carry
    `.usage`, `.model_name`, `.provider_name`."""
    return NS(usage=usage, provider_name=provider_name, model_name=model_name)


def _drive(response):
    """Run `_log_pydantic_ai` against `response` and return the captured kwargs."""
    tp = _CaptureTP()
    from datetime import datetime, timezone
    with mock.patch.object(enforcer, "get_client", return_value=tp):
        enforcer._log_pydantic_ai(
            _model(), _session(), [{"role": "user", "content": "hi"}],
            response, 1, "call", datetime.now(timezone.utc),
        )
    return tp.captured


# ── Real pydantic_ai RequestUsage builder (A6 input-exclusivity pin) ──
def _real_request_usage(input_tokens, output_tokens, cache_read, cache_write):
    """Build a REAL pydantic_ai RequestUsage via the library's own Anthropic
    mapper. Returns None only on genuine ImportError."""
    try:
        from pydantic_ai.models import anthropic as pa_anth
        from anthropic.types.beta import BetaUsage, BetaMessage
        from anthropic.types.beta.beta_text_block import BetaTextBlock
    except ImportError:
        return None
    usage = BetaUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )
    msg = BetaMessage(
        id="msg_1", type="message", role="assistant",
        model="claude-3-5-sonnet",
        content=[BetaTextBlock(type="text", text="hi")],
        stop_reason="end_turn", stop_sequence=None, usage=usage,
    )
    return pa_anth._map_usage(
        msg, provider="anthropic",
        provider_url="https://api.anthropic.com", model="claude-3-5-sonnet",
    )


class TestPydanticAIAnthropicCacheUsage(unittest.TestCase):

    # ── A4/A5: anthropic cache-write forwards the shape; cached=reads-only ──
    def test_anthropic_cache_write_forwards_shape_and_reads_only(self):
        usage = NS(
            input_tokens=150, output_tokens=40,
            details={
                "input_tokens": 100, "output_tokens": 40,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
            },
        )
        kw = _drive(_response(usage))
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        raw = kw["usage"]["raw"]
        self.assertEqual(raw["cache_creation_input_tokens"], 20)  # W
        self.assertEqual(raw["cache_read_input_tokens"], 30)      # R
        # cached_tokens carries READS only, NOT read+write (the under-count).
        self.assertEqual(kw["cached_tokens"], 30)
        self.assertNotEqual(kw["cached_tokens"], 50)

    # ── A6: LOAD-BEARING input-exclusivity pin, against the REAL library ──
    def test_real_library_input_is_cache_exclusive_not_inclusive(self):
        ru = _real_request_usage(input_tokens=100, output_tokens=40,
                                  cache_read=30, cache_write=20)
        if ru is None:
            self.skipTest("pydantic_ai not importable (genuine ImportError)")
        # Sanity-pin the library's own accounting: top-level is cache-INCLUSIVE.
        self.assertEqual(ru.input_tokens, 150)
        self.assertEqual(ru.details["input_tokens"], 100)

        kw = _drive(_response(ru))
        raw = kw["usage"]["raw"]
        # Raw carries the native cache-EXCLUSIVE input, NOT the inclusive 150.
        self.assertEqual(raw["input_tokens"], 100)
        self.assertNotEqual(raw["input_tokens"], 150)
        # Top-level forwarded input_tokens switches to the exclusive value.
        self.assertEqual(kw["input_tokens"], 100)
        self.assertNotEqual(kw["input_tokens"], 150)

    # ── A7: raw.output_tokens is the native detail value; payload JSON-safe ──
    def test_real_library_output_and_json_serializable(self):
        ru = _real_request_usage(input_tokens=100, output_tokens=40,
                                  cache_read=30, cache_write=20)
        if ru is None:
            self.skipTest("pydantic_ai not importable (genuine ImportError)")
        kw = _drive(_response(ru))
        self.assertEqual(kw["usage"]["raw"]["output_tokens"],
                         ru.details["output_tokens"])
        self.assertEqual(kw["usage"]["raw"]["output_tokens"], 40)
        json.dumps(kw["usage"])  # must not raise

    # ── A8: non-anthropic byte-identical (gate does not fire off-anthropic) ──
    def test_non_anthropic_byte_identical(self):
        for provider, cache_key in (("openai", "cached_tokens"),
                                    ("google", "cached_content_token_count")):
            usage = NS(
                input_tokens=100, output_tokens=40,
                details={cache_key: 30},
            )
            kw = _drive(_response(usage, provider_name=provider))
            self.assertIsNone(kw.get("usage"), provider)
            inp, out, cached = enforcer._pa_usage_to_counts(usage)
            self.assertEqual(kw["input_tokens"], inp, provider)
            self.assertEqual(kw["output_tokens"], out, provider)
            self.assertEqual(kw["cached_tokens"], cached, provider)

    # ── A9: anthropic with NO cache-write → fallback, no degenerate raw ──
    def test_anthropic_no_cache_write_falls_back(self):
        # read-only (write absent) and write==0 both must stay on today's path.
        for details in (
            {"input_tokens": 100, "output_tokens": 40,
             "cache_read_input_tokens": 30},
            {"input_tokens": 100, "output_tokens": 40,
             "cache_read_input_tokens": 30, "cache_creation_input_tokens": 0},
        ):
            usage = NS(input_tokens=130, output_tokens=40, details=details)
            kw = _drive(_response(usage))
            self.assertIsNone(kw.get("usage"), details)
            inp, out, cached = enforcer._pa_usage_to_counts(usage)
            self.assertEqual(kw["input_tokens"], inp, details)
            self.assertEqual(kw["cached_tokens"], cached, details)

    # ── A10a: Golden Rule — hostile usage never throws into the caller ──
    def test_hostile_usage_never_throws(self):
        # `.details` property raises on access, and a details mapping whose
        # `.get` throws. Both raise inside the shared counts extractor (called
        # before the helper), so @fail_safe swallows them and the row may drop
        # — but the customer's call MUST return normally with no exception.
        class _RaisingDetails:
            input_tokens = 100
            output_tokens = 40

            @property
            def details(self):
                raise RuntimeError("boom")

        class _HostileDict(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom")

        for usage in (
            _RaisingDetails(),
            NS(input_tokens=100, output_tokens=40,
               details=_HostileDict(cache_creation_input_tokens=20)),
        ):
            try:
                _drive(_response(usage))  # must not raise
            except Exception as e:  # pragma: no cover - fail path
                self.fail("_log_pydantic_ai leaked an exception: %r" % e)

    # ── A10b: helper try/except is load-bearing — bad cache value → fallback ──
    def test_helper_bad_cache_value_falls_back_and_logs(self):
        # A non-int cache-write value: the shared counts extractor tolerates it
        # (its int() is guarded), so the row IS logged; the helper's own
        # try/except must convert its int() failure into the fallback (usage
        # None) rather than letting it surface to @fail_safe and drop the row.
        usage = NS(
            input_tokens=100, output_tokens=40,
            details={
                "input_tokens": 100, "output_tokens": 40,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": object(),  # int() raises
            },
        )
        kw = _drive(_response(usage))
        self.assertIn("model", kw)          # row still logged
        self.assertIsNone(kw.get("usage"))  # fallback, no shape forwarded

    # ── A11: both pydantic_ai sub-paths (request + stream.get()) covered ──
    def test_both_subpaths_yield_shape(self):
        usage = NS(
            input_tokens=150, output_tokens=40,
            details={
                "input_tokens": 100, "output_tokens": 40,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
            },
        )
        # (a) non-stream Model.request → a ModelResponse-like object.
        non_stream = _response(usage)
        # (b) stream StreamedResponse.get() → a synthetic ModelResponse-like
        # object with the same fields populated from the drained stream.
        stream = NS(usage=usage, provider_name="anthropic",
                    model_name="claude-3-5-sonnet")
        for resp in (non_stream, stream):
            kw = _drive(resp)
            self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
            self.assertEqual(kw["cached_tokens"], 30)


if __name__ == "__main__":
    unittest.main()
