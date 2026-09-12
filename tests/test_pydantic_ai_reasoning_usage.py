"""G3-O1 (Fix B) — pydantic_ai must forward OpenAI-arm reasoning tokens.

pydantic_ai reports a call's tokens in a `RequestUsage` whose `output_tokens`
already FOLDS IN the reasoning tokens, and exposes the reasoning count only at
`usage.details['reasoning_tokens']`. `_log_pydantic_ai` forwarded `usage=None`
for every non-anthropic-cache row, so the client synthesized an
`openai_compatible_chat` block with no `completion_tokens_details` — the
reasoning breakdown was silently dropped and every o-series / gpt-5 call through
pydantic_ai landed with `reasoning_output_tokens = 0`.

Fix under test: when no usage block has been built yet, the provider is the
OpenAI arm, and `details['reasoning_tokens'] > 0`, build a byte-mirror of the
client's own synth (`prompt_tokens` / `completion_tokens`, plus
`prompt_tokens_details.cached_tokens` when cached) PLUS
`completion_tokens_details.reasoning_tokens`.

Load-bearing invariants pinned here:
  * SUBSET SEMANTICS — `completion_tokens` stays the folded total and reasoning
    is clamped to it; the openai_compatible_chat mapper computes
    text_output = completion − reasoning, so an unclamped value would produce a
    negative text bucket.
  * xAI/Grok IS EXCLUDED — pydantic_ai's xai arms report provider "openai", but
    the collector's xai mapper treats nested reasoning as EXCLUSIVE of
    completion_tokens, so forwarding it there would DOUBLE-COUNT output.
  * NO-REASONING ROWS ARE BYTE-IDENTICAL — `usage` stays None, so the client
    (which gates on `if usage:`) emits exactly today's wire payload.
  * The anthropic cache-write branch stays authoritative (our branch only fires
    when `usage_block is None`).
  * GOLDEN RULE — hostile details/usage never raise into the customer's call.

Offline: drives `_log_pydantic_ai` directly with a capturing fake client,
mirroring tests/test_pydantic_ai_anthropic_cache_usage.py.
"""
import json
import unittest
from datetime import datetime, timezone
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


def _model(system="openai", model_name="gpt-5"):
    return NS(system=system, model_name=model_name)


def _response(usage, provider_name="openai", model_name="gpt-5"):
    """A ModelResponse-like / StreamedResponse.get()-like object: both carry
    `.usage`, `.model_name`, `.provider_name`."""
    return NS(usage=usage, provider_name=provider_name, model_name=model_name)


def _usage(input_tokens=100, output_tokens=400, details=None):
    return NS(input_tokens=input_tokens, output_tokens=output_tokens,
              details=details if details is not None else {})


def _drive(response, model=None):
    """Run `_log_pydantic_ai` against `response` and return the captured kwargs."""
    tp = _CaptureTP()
    with mock.patch.object(enforcer, "get_client", return_value=tp):
        enforcer._log_pydantic_ai(
            model or _model(), _session(), [{"role": "user", "content": "hi"}],
            response, 1, "call", datetime.now(timezone.utc),
        )
    return tp.captured


# ─────────────────────────────────────────────────────────────────────────
# 1. _pa_reasoning_from_details — helper unit tests
# ─────────────────────────────────────────────────────────────────────────
class TestPaReasoningFromDetails(unittest.TestCase):
    def test_reads_reasoning_tokens(self):
        self.assertEqual(
            enforcer._pa_reasoning_from_details(
                _usage(details={"reasoning_tokens": 357}), 400),
            357)

    def test_clamped_to_output_tokens(self):
        # Reasoning is folded INTO output; a larger value is provider noise and
        # would make the collector's text bucket negative.
        self.assertEqual(
            enforcer._pa_reasoning_from_details(
                _usage(details={"reasoning_tokens": 900}), 400),
            400)

    def test_equal_to_output_is_kept(self):
        self.assertEqual(
            enforcer._pa_reasoning_from_details(
                _usage(details={"reasoning_tokens": 400}), 400),
            400)

    def test_zero_and_absent_are_zero(self):
        for details in ({}, {"reasoning_tokens": 0}, {"reasoning_tokens": None},
                        {"cached_tokens": 30}):
            self.assertEqual(
                enforcer._pa_reasoning_from_details(_usage(details=details), 400),
                0, details)

    def test_negative_is_zero(self):
        self.assertEqual(
            enforcer._pa_reasoning_from_details(
                _usage(details={"reasoning_tokens": -5}), 400),
            0)

    def test_zero_output_tokens_clamps_to_zero(self):
        self.assertEqual(
            enforcer._pa_reasoning_from_details(
                _usage(output_tokens=0, details={"reasoning_tokens": 357}), 0),
            0)

    def test_non_dict_details_is_zero(self):
        for details in (None, "nope", 42, ["reasoning_tokens"], object()):
            self.assertEqual(
                enforcer._pa_reasoning_from_details(_usage(details=details), 400),
                0, details)

    def test_usage_none_is_zero(self):
        self.assertEqual(enforcer._pa_reasoning_from_details(None, 400), 0)

    def test_non_numeric_value_is_zero_never_raises(self):
        self.assertEqual(
            enforcer._pa_reasoning_from_details(
                _usage(details={"reasoning_tokens": object()}), 400),
            0)

    def test_raising_details_property_is_zero(self):
        class Boom:
            @property
            def details(self):
                raise RuntimeError("boom")

        self.assertEqual(enforcer._pa_reasoning_from_details(Boom(), 400), 0)

    def test_hostile_details_get_is_zero(self):
        class _HostileDict(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom")

        self.assertEqual(
            enforcer._pa_reasoning_from_details(
                _usage(details=_HostileDict(reasoning_tokens=10)), 400),
            0)


# ─────────────────────────────────────────────────────────────────────────
# 2. OpenAI arm — the usage block is built and carries the details
# ─────────────────────────────────────────────────────────────────────────
class TestOpenAIArmForwardsReasoning(unittest.TestCase):
    def test_reasoning_forwarded_under_openai_compatible_chat(self):
        kw = _drive(_response(_usage(100, 400, {"reasoning_tokens": 357})))

        self.assertEqual(kw["usage"]["shape"], "openai_compatible_chat")
        raw = kw["usage"]["raw"]
        self.assertEqual(raw["completion_tokens_details"]["reasoning_tokens"], 357)
        # SUBSET: completion stays the folded total (mapper does completion − r).
        self.assertEqual(raw["completion_tokens"], 400)
        self.assertNotEqual(raw["completion_tokens"], 400 + 357)
        self.assertEqual(raw["prompt_tokens"], 100)
        # No cache on this row → no prompt_tokens_details key at all.
        self.assertNotIn("prompt_tokens_details", raw)

    def test_raw_mirrors_the_top_level_counts_exactly(self):
        """The raw is a byte-mirror of the client's own synth — a mismatch
        between raw and the positional counts is what mis-attributes cost."""
        kw = _drive(_response(_usage(1234, 567, {"reasoning_tokens": 89})))
        raw = kw["usage"]["raw"]
        self.assertEqual(raw["prompt_tokens"], kw["input_tokens"])
        self.assertEqual(raw["completion_tokens"], kw["output_tokens"])

    def test_cached_tokens_passed_through_when_positive(self):
        kw = _drive(_response(_usage(
            100, 400, {"reasoning_tokens": 357, "cached_tokens": 64})))
        raw = kw["usage"]["raw"]
        self.assertEqual(raw["prompt_tokens_details"]["cached_tokens"], 64)
        self.assertEqual(kw["cached_tokens"], 64)
        self.assertEqual(raw["completion_tokens_details"]["reasoning_tokens"], 357)

    def test_zero_cached_omits_prompt_tokens_details(self):
        kw = _drive(_response(_usage(
            100, 400, {"reasoning_tokens": 357, "cached_tokens": 0})))
        self.assertNotIn("prompt_tokens_details", kw["usage"]["raw"])

    def test_reasoning_clamped_to_output(self):
        kw = _drive(_response(_usage(100, 400, {"reasoning_tokens": 4000})))
        raw = kw["usage"]["raw"]
        self.assertEqual(raw["completion_tokens_details"]["reasoning_tokens"], 400)
        # text_output = completion − reasoning must never go negative.
        self.assertGreaterEqual(
            raw["completion_tokens"]
            - raw["completion_tokens_details"]["reasoning_tokens"], 0)

    def test_payload_is_json_serializable(self):
        kw = _drive(_response(_usage(100, 400, {"reasoning_tokens": 357})))
        json.dumps(kw["usage"])  # must not raise

    def test_both_subpaths_stream_and_non_stream(self):
        """Both `Model.request` and `StreamedResponse.get()` funnel through
        `_log_pydantic_ai`, so one change fixes both arms."""
        usage = _usage(100, 400, {"reasoning_tokens": 357})
        non_stream = _response(usage)
        stream = NS(usage=usage, provider_name="openai", model_name="gpt-5")
        for resp in (non_stream, stream):
            kw = _drive(resp)
            self.assertEqual(
                kw["usage"]["raw"]["completion_tokens_details"]["reasoning_tokens"],
                357)

    def test_provider_name_openai_variants_still_forward(self):
        for provider_name in ("openai", "OpenAI", "openai-chat"):
            kw = _drive(_response(
                _usage(100, 400, {"reasoning_tokens": 357}),
                provider_name=provider_name))
            self.assertIsNotNone(kw.get("usage"), provider_name)


# ─────────────────────────────────────────────────────────────────────────
# 3. Rows that must stay BYTE-IDENTICAL to today (usage stays None)
# ─────────────────────────────────────────────────────────────────────────
class TestUnchangedRows(unittest.TestCase):
    def _assert_unchanged(self, kw, usage, label=""):
        self.assertIsNone(kw.get("usage"), label)
        inp, out, cached = enforcer._pa_usage_to_counts(usage)
        self.assertEqual(kw["input_tokens"], inp, label)
        self.assertEqual(kw["output_tokens"], out, label)
        self.assertEqual(kw["cached_tokens"], cached, label)

    def test_no_details_key(self):
        usage = _usage(100, 400, {})
        self._assert_unchanged(_drive(_response(usage)), usage)

    def test_zero_reasoning(self):
        usage = _usage(100, 400, {"reasoning_tokens": 0})
        self._assert_unchanged(_drive(_response(usage)), usage)

    def test_details_is_not_a_dict(self):
        for details in (None, "nope", 42):
            usage = _usage(100, 400, details)
            self._assert_unchanged(_drive(_response(usage)), usage, repr(details))

    def test_usage_is_none(self):
        kw = _drive(_response(None))
        self.assertIsNone(kw.get("usage"))
        self.assertEqual(kw["input_tokens"], 0)
        self.assertEqual(kw["output_tokens"], 0)

    def test_failure_row_response_none(self):
        """Failure rows (`response=None`) keep today's behavior exactly."""
        kw = _drive(None)
        self.assertIsNone(kw.get("usage"))
        self.assertEqual(kw["input_tokens"], 0)
        self.assertEqual(kw["output_tokens"], 0)

    def test_zero_output_tokens_never_builds_a_block(self):
        # Clamp drives reasoning to 0 → no block, no degenerate raw.
        usage = _usage(100, 0, {"reasoning_tokens": 357})
        self._assert_unchanged(_drive(_response(usage)), usage)

    def test_non_openai_providers_untouched(self):
        for system, provider_name in (("anthropic", "anthropic"),
                                      ("google", "google-gla"),
                                      ("mistral", "mistral")):
            usage = _usage(100, 400, {"reasoning_tokens": 357,
                                      "thoughts_tokens": 357})
            kw = _drive(_response(usage, provider_name=provider_name),
                        model=_model(system=system, model_name="m"))
            self._assert_unchanged(kw, usage, provider_name)

    def test_google_thoughts_tokens_explicitly_out_of_scope(self):
        usage = _usage(100, 400, {"thoughts_tokens": 357})
        kw = _drive(_response(usage, provider_name="google-gla"),
                    model=_model(system="google", model_name="gemini-2.5-pro"))
        self.assertIsNone(kw.get("usage"))


# ─────────────────────────────────────────────────────────────────────────
# 4. xAI/Grok screen — forwarding there would DOUBLE-COUNT output
# ─────────────────────────────────────────────────────────────────────────
class TestXaiScreen(unittest.TestCase):
    def test_xai_and_grok_provider_names_are_skipped(self):
        # pydantic_ai's xai arms run on the OpenAI driver and report
        # system="openai", so the provider key alone can't distinguish them —
        # the raw provider_name is the only discriminator.
        for provider_name in ("xai", "XAI", "grok", "Grok-4", "xai-openai"):
            kw = _drive(_response(_usage(100, 400, {"reasoning_tokens": 357}),
                                  provider_name=provider_name))
            self.assertIsNone(kw.get("usage"), provider_name)

    def test_openai_sibling_still_forwards(self):
        """Control: the same usage on a genuine openai row DOES forward, so the
        skip above is the screen and not an unrelated early bail."""
        kw = _drive(_response(_usage(100, 400, {"reasoning_tokens": 357}),
                              provider_name="openai"))
        self.assertEqual(
            kw["usage"]["raw"]["completion_tokens_details"]["reasoning_tokens"],
            357)


# ─────────────────────────────────────────────────────────────────────────
# 5. The anthropic cache-write branch stays authoritative
# ─────────────────────────────────────────────────────────────────────────
class TestAnthropicBranchUntouched(unittest.TestCase):
    def test_anthropic_cache_write_block_not_replaced(self):
        usage = NS(
            input_tokens=150, output_tokens=40,
            details={
                "input_tokens": 100, "output_tokens": 40,
                "cache_read_input_tokens": 30,
                "cache_creation_input_tokens": 20,
                # Even with a reasoning key present, the anthropic branch wins.
                "reasoning_tokens": 11,
            },
        )
        kw = _drive(_response(usage, provider_name="anthropic"),
                    model=_model(system="anthropic", model_name="claude"))

        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        self.assertNotIn("completion_tokens_details", kw["usage"]["raw"])
        self.assertEqual(kw["cached_tokens"], 30)


# ─────────────────────────────────────────────────────────────────────────
# 6. GOLDEN RULE — nothing here may raise into the customer's call
# ─────────────────────────────────────────────────────────────────────────
class TestGoldenRule(unittest.TestCase):
    def test_hostile_details_never_throws(self):
        class _RaisingDetails:
            input_tokens = 100
            output_tokens = 400
            provider_name = "openai"

            @property
            def details(self):
                raise RuntimeError("boom")

        class _HostileDict(dict):
            def get(self, *a, **k):
                raise RuntimeError("boom")

        for usage in (
            _RaisingDetails(),
            _usage(100, 400, _HostileDict(reasoning_tokens=357)),
            _usage(100, 400, {"reasoning_tokens": object()}),
        ):
            try:
                _drive(_response(usage))
            except Exception as e:  # pragma: no cover - fail path
                self.fail("_log_pydantic_ai leaked an exception: %r" % e)

    def test_bad_reasoning_value_still_logs_the_row(self):
        """The helper's own try/except is load-bearing: a non-int reasoning
        value must fall back to `usage=None` rather than surfacing to
        @fail_safe and dropping the whole row."""
        kw = _drive(_response(_usage(100, 400, {"reasoning_tokens": object()})))
        self.assertIn("model", kw)          # row still logged
        self.assertIsNone(kw.get("usage"))  # no shape forwarded

    def test_raising_provider_name_never_throws(self):
        class _RaisingProviderName:
            usage = None
            model_name = "gpt-5"

            @property
            def provider_name(self):
                raise RuntimeError("boom")

        try:
            _drive(_RaisingProviderName())
        except Exception as e:  # pragma: no cover - fail path
            self.fail("_log_pydantic_ai leaked an exception: %r" % e)


if __name__ == "__main__":
    unittest.main()
