"""One rerouted model, one model string.

The auto-instrumented span path reports the (post-rewrite) REQUEST model — a
reroute rule's target alias, ``claude-haiku-4-5`` — while the manual taps read
the provider's ECHO, the dated snapshot the alias resolves to
(``claude-haiku-4-5-20251001``). The same traffic therefore landed under two
``generations.model`` strings and split every cost-by-model panel.

``prefer_requested_model`` collapses the echo back onto the request, gated on
the two being the SAME model family (request + a dated suffix). A provider that
genuinely served something else — gateway auto-routing — fails that gate and
keeps its echo.
"""
from __future__ import annotations

import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police.context import TPSession
from token_police.reroute_noop import prefer_requested_model


class TestPreferRequestedModel(unittest.TestCase):
    def test_dated_snapshot_echo_of_the_requested_alias(self):
        self.assertEqual(
            prefer_requested_model("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
            "claude-haiku-4-5",
        )

    def test_openai_date_form(self):
        self.assertEqual(
            prefer_requested_model("gpt-4o-mini", "gpt-4o-mini-2024-07-18"),
            "gpt-4o-mini",
        )

    def test_vertex_at_date_form(self):
        self.assertEqual(
            prefer_requested_model("claude-3-sonnet", "claude-3-sonnet@20240229"),
            "claude-3-sonnet",
        )

    def test_identical_strings_return_the_extracted_model(self):
        self.assertEqual(
            prefer_requested_model("gpt-4o-mini", "gpt-4o-mini"), "gpt-4o-mini"
        )

    def test_genuinely_different_served_model_keeps_its_echo(self):
        self.assertEqual(
            prefer_requested_model("openrouter/auto", "anthropic/claude-3-haiku"),
            "anthropic/claude-3-haiku",
        )
        self.assertEqual(
            prefer_requested_model("gpt-4o-mini", "gpt-4o-2024-08-06"),
            "gpt-4o-2024-08-06",
        )

    def test_pinned_snapshot_request_is_never_rewritten(self):
        self.assertEqual(
            prefer_requested_model(
                "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"
            ),
            "claude-haiku-4-5-20251001",
        )

    def test_non_date_suffixes_are_not_a_family_match(self):
        self.assertEqual(
            prefer_requested_model("gemini-1.5-pro", "gemini-1.5-pro-002"),
            "gemini-1.5-pro-002",
        )
        self.assertEqual(
            prefer_requested_model("claude-3-5-sonnet", "claude-3-5-sonnet-latest"),
            "claude-3-5-sonnet-latest",
        )

    def test_case_insensitive_match_returns_the_original_requested_string(self):
        self.assertEqual(
            prefer_requested_model(" Claude-Haiku-4-5 ", "CLAUDE-HAIKU-4-5-20251001"),
            " Claude-Haiku-4-5 ",
        )

    def test_bad_requested_input_returns_the_extracted_model(self):
        extracted = "claude-haiku-4-5-20251001"
        for requested in (
            None,
            "",
            "   ",
            123,
            {"model": "claude-haiku-4-5"},
            ["claude-haiku-4-5"],
        ):
            self.assertEqual(prefer_requested_model(requested, extracted), extracted)

    def test_bad_extracted_input_is_returned_as_is(self):
        self.assertEqual(prefer_requested_model("claude-haiku-4-5", ""), "")
        self.assertIsNone(prefer_requested_model("claude-haiku-4-5", None))
        self.assertEqual(prefer_requested_model("claude-haiku-4-5", 123), 123)


# NEW-1: on Bedrock the model id IS the price — a cross-region inference
# profile (``us.anthropic.claude-haiku-4-5-20251001-v1:0``) bills at a
# different rate than its bare echo (``claude-haiku-4-5-20251001``). The gate
# must widen to treat the echo as same-family (so the profile id is reported
# verbatim), but ONLY when the requested id is unmistakably Bedrock-shaped — a
# genuinely different served model, or a look-alike that isn't Bedrock-shaped,
# must not widen. Mirrors the Node ``describe("PreferRequestedModel — Bedrock
# family gate")`` block 1:1.
class TestPreferRequestedModelBedrockFamilyGate(unittest.TestCase):
    def test_cris_profile_id_bare_model_echo_the_prod_bug(self):
        self.assertEqual(
            prefer_requested_model(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "claude-haiku-4-5-20251001",
            ),
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

    def test_non_cris_bedrock_id_bare_model_echo(self):
        self.assertEqual(
            prefer_requested_model(
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                "claude-haiku-4-5-20251001",
            ),
            "anthropic.claude-haiku-4-5-20251001-v1:0",
        )

    def test_echo_is_the_region_less_form(self):
        self.assertEqual(
            prefer_requested_model(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
            ),
            "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        )

    def test_amazon_vendor_versioned_echo_dropped(self):
        self.assertEqual(
            prefer_requested_model("amazon.nova-lite-v1:0", "nova-lite-v1:0"),
            "amazon.nova-lite-v1:0",
        )

    def test_amazon_cris_versioned_echo_dropped(self):
        self.assertEqual(
            prefer_requested_model("us.amazon.nova-lite-v1:0", "nova-lite-v1:0"),
            "us.amazon.nova-lite-v1:0",
        )

    def test_eu_region_meta_vendor(self):
        self.assertEqual(
            prefer_requested_model(
                "eu.meta.llama3-2-90b-instruct-v1:0", "llama3-2-90b-instruct-v1:0"
            ),
            "eu.meta.llama3-2-90b-instruct-v1:0",
        )

    def test_global_region_and_dated_snapshot_echo_both_wrappers_peeled(self):
        self.assertEqual(
            prefer_requested_model(
                "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "claude-sonnet-4-5-20250929",
            ),
            "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )

    def test_case_whitespace_insensitive_match_returns_original_string_verbatim(self):
        self.assertEqual(
            prefer_requested_model(
                " US.Anthropic.Claude-Haiku-4-5-20251001-v1:0 ",
                "claude-haiku-4-5-20251001",
            ),
            " US.Anthropic.Claude-Haiku-4-5-20251001-v1:0 ",
        )

    def test_genuinely_different_served_model_on_bedrock_keeps_its_echo(self):
        self.assertEqual(
            prefer_requested_model(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-sonnet-4-6"
            ),
            "claude-sonnet-4-6",
        )
        self.assertEqual(
            prefer_requested_model(
                "us.anthropic.claude-haiku-4-5-20251001-v1:0", "gpt-4o"
            ),
            "gpt-4o",
        )

    def test_unknown_vendor_namespace_not_bedrock_shaped_no_widening(self):
        self.assertEqual(
            prefer_requested_model("us.notavendor.some-model-v1:0", "some-model"),
            "some-model",
        )
        self.assertEqual(
            prefer_requested_model("notavendor.some-model-v1:0", "some-model"),
            "some-model",
        )

    def test_region_shaped_but_not_a_cris_region_token_no_widening(self):
        self.assertEqual(
            prefer_requested_model(
                "us-east-1.anthropic.claude-haiku-4-5", "claude-haiku-4-5"
            ),
            "claude-haiku-4-5",
        )

    # Every CRIS region token in _BEDROCK_REGION_PREFIX, not just "us." — regex
    # alternation backtracks, so listing order between "us-gov" and "us" is
    # not load-bearing; what matters is that the FULL region token is
    # consumed (never a partial match like "us" alone in front of "-gov....").
    def test_every_cris_region_token_widens_correctly(self):
        self.assertEqual(
            prefer_requested_model(
                "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "claude-sonnet-4-5-20250929",
            ),
            "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )
        self.assertEqual(
            prefer_requested_model("global.amazon.nova-lite-v1:0", "nova-lite-v1:0"),
            "global.amazon.nova-lite-v1:0",
        )
        self.assertEqual(
            prefer_requested_model(
                "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "claude-sonnet-4-5-20250929",
            ),
            "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )

    def test_us_gov_strips_the_whole_region_token(self):
        # Guards against a future implementation that stops short of the
        # full "us-gov." token (e.g. a naive split on the first "-"): if only
        # "us" were peeled, the vendor-namespace gate would see
        # "gov.anthropic...." (not a real vendor prefix) and refuse to widen
        # — the echo would stay unchanged. Widening here proves the full
        # "us-gov." token was consumed.
        self.assertEqual(
            prefer_requested_model(
                "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "claude-sonnet-4-5-20250929",
            ),
            "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        )

    def test_only_one_region_prefix_is_peeled_doubled_prefix_no_widen(self):
        # _bedrock_family_candidates strips a single region layer, then
        # requires a real vendor namespace immediately after it. A doubled
        # prefix ("us.eu...") leaves "eu.anthropic...." after the one strip,
        # which is not a recognized vendor namespace, so NO candidates are
        # built — the echo must be kept verbatim.
        self.assertEqual(
            prefer_requested_model(
                "us.eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
                "claude-sonnet-4-5-20250929",
            ),
            "claude-sonnet-4-5-20250929",
        )

    def test_case_insensitive_region_token_still_widens(self):
        self.assertEqual(
            prefer_requested_model(
                "US.ANTHROPIC.claude-haiku-4-5-20251001-v1:0",
                "claude-haiku-4-5-20251001",
            ),
            "US.ANTHROPIC.claude-haiku-4-5-20251001-v1:0",
        )

    # T3: _BEDROCK_VERSION_SUFFIX widened to accept a BARE numeric version tag
    # (no "v") -- "us.openai.gpt-oss-120b-1:0" is a real catalog id. The "v" is
    # optional only when a ":<n>" part follows; that's what keeps a dated
    # snapshot suffix ("-20251001", no colon) from being swallowed as a version.
    def test_bare_numeric_version_tag_widens_correctly(self):
        self.assertEqual(
            prefer_requested_model("us.openai.gpt-oss-120b-1:0", "gpt-oss-120b"),
            "us.openai.gpt-oss-120b-1:0",
        )

    def test_explicit_v_plus_colon_minor_version_tag_widens_correctly(self):
        self.assertEqual(
            prefer_requested_model(
                "us.mistral.mistral-7b-instruct-v0:2", "mistral-7b-instruct"
            ),
            "us.mistral.mistral-7b-instruct-v0:2",
        )

    def test_dated_snapshot_suffix_never_treated_as_version_tag(self):
        # No colon and no leading "v" -- must fail BOTH branches of
        # _BEDROCK_VERSION_SUFFIX. If it were ever swallowed,
        # "claude-haiku-4-5" would land in the candidate set and this would
        # incorrectly widen.
        self.assertEqual(
            prefer_requested_model(
                "anthropic.claude-haiku-4-5-20251001", "claude-haiku-4-5"
            ),
            "claude-haiku-4-5",
        )


# ── _log_manual harness (mirrors tests/test_litellm_original_provider.py) ───
def _logged_model(requested, echo):
    captured = {}

    class _FakeTP:
        def log_sync(self, **kw):
            captured.update(kw)

    result = NS(
        model=echo,
        usage=NS(prompt_tokens=10, completion_tokens=5),
    )
    session = TPSession()
    with mock.patch.object(enforcer, "get_client", return_value=_FakeTP()):
        enforcer._log_manual(
            "anthropic", session, {"model": requested}, result, 0, None,
            datetime.now(timezone.utc),
        )
    return captured.get("model")


class TestLogManualPrefersRequestedModel(unittest.TestCase):
    def test_alias_requested_dated_snapshot_echoed(self):
        self.assertEqual(
            _logged_model("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
            "claude-haiku-4-5",
        )

    def test_customer_pinned_snapshot_logs_unchanged(self):
        self.assertEqual(
            _logged_model("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
            "claude-haiku-4-5-20251001",
        )

    def test_provider_served_a_different_model_keeps_the_echo(self):
        self.assertEqual(
            _logged_model("claude-haiku-4-5", "claude-sonnet-4-5-20250929"),
            "claude-sonnet-4-5-20250929",
        )


# ── B8 regression harness: .stream() context-manager finalizers ────────────
# Mirrors the fakes/driver in tests/test_anthropic_stream_usage_shape.py
# (duplicated in miniature here rather than cross-imported, since tests/ has
# no __init__.py and isn't a package).
def _stream_events():
    return [
        NS(type="message_start"),
        NS(type="content_block_start"),
        NS(type="content_block_delta", delta=NS(text="Hi")),
        NS(type="message_stop"),
    ]


def _stream_final_message(echo_model):
    return NS(
        model=echo_model,
        content=[NS(type="text", text="Hi")],
        usage=NS(input_tokens=10, output_tokens=5,
                 cache_read_input_tokens=0, cache_creation_input_tokens=0),
    )


class _FakeAnthropicStream:
    def __init__(self, events, final_msg):
        self._events = events
        self._final = final_msg

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._final


class _FakeAnthropicAsyncStream(_FakeAnthropicStream):
    def __aiter__(self):
        async def _agen():
            for ev in self._events:
                yield ev
        return _agen()

    async def get_final_message(self):
        return self._final


class _FakeMgr:
    def __init__(self, stream):
        self._stream = stream

    def __enter__(self):
        return self._stream

    def __exit__(self, *a):
        return False


class _FakeAsyncMgr(_FakeMgr):
    async def __aenter__(self):
        return self._stream

    async def __aexit__(self, *a):
        return False


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run() — asyncio.run
    unsets the main-thread event loop, breaking later get_event_loop()-based
    tests in the same pytest process. Mirrors
    tests/test_anthropic_stream_usage_shape.py."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _StreamCaptureTP:
    """Captures the kwargs of the single log_sync the finalizer emits."""

    def __init__(self):
        self.captured = {}

    def log_sync(self, **kw):
        self.captured.update(kw)


def _sync_finalize(kwargs, echo_model):
    tp = _StreamCaptureTP()
    wrapper = enforcer._AnthropicStreamMgrWrapper(
        _FakeMgr(_FakeAnthropicStream(_stream_events(),
                                      _stream_final_message(echo_model))),
        kwargs,
    )
    with mock.patch.object(enforcer, "get_client", return_value=tp):
        with wrapper as stream:
            list(stream)
    return tp.captured


def _async_finalize(kwargs, echo_model):
    tp = _StreamCaptureTP()
    wrapper = enforcer._AnthropicAsyncStreamMgrWrapper(
        _FakeAsyncMgr(_FakeAnthropicAsyncStream(_stream_events(),
                                                _stream_final_message(echo_model))),
        kwargs,
    )

    async def _run():
        async with wrapper as stream:
            async for _ in stream:
                pass

    with mock.patch.object(enforcer, "get_client", return_value=tp):
        _run_coro(_run())
    return tp.captured


class TestAnthropicStreamFinalizerPrefersRequestedModel(unittest.TestCase):
    """B8: `.stream()` context-manager finalizers logged the provider-echoed
    dated snapshot instead of the requested (post-reroute) alias, splitting
    one logical model into two cost-by-model rows. Both `_finalize` (sync)
    and `_afinalize` (async) now run the echo through `prefer_requested_model`
    keyed on `self._kwargs["model"]` before logging."""

    # ── 1. collapse: dated echo of the requested alias ──
    def test_sync_collapses_dated_echo_to_requested_alias(self):
        kw = _sync_finalize(
            {"model": "claude-haiku-4-5", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5")

    def test_async_collapses_dated_echo_to_requested_alias(self):
        kw = _async_finalize(
            {"model": "claude-haiku-4-5", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5")

    # ── 2. negative / family gate: genuinely different served model ──
    def test_sync_keeps_echo_when_served_model_is_a_different_family(self):
        kw = _sync_finalize(
            {"model": "claude-sonnet-4-6", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5-20251001")

    def test_async_keeps_echo_when_served_model_is_a_different_family(self):
        kw = _async_finalize(
            {"model": "claude-sonnet-4-6", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5-20251001")

    # ── 3. exact echo, no date suffix: unchanged, no crash ──
    def test_sync_exact_echo_with_no_date_suffix_logs_unchanged(self):
        kw = _sync_finalize(
            {"model": "claude-haiku-4-5", "messages": []},
            "claude-haiku-4-5",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5")

    def test_async_exact_echo_with_no_date_suffix_logs_unchanged(self):
        kw = _async_finalize(
            {"model": "claude-haiku-4-5", "messages": []},
            "claude-haiku-4-5",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5")

    # ── 4. span_name fallback uses the COLLAPSED alias, not the dated echo ──
    def test_sync_span_name_fallback_uses_the_collapsed_alias(self):
        kw = _sync_finalize(
            {"model": "claude-haiku-4-5", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5")
        self.assertEqual(kw["span"]["span_name"], "claude-haiku-4-5")

    def test_async_span_name_fallback_uses_the_collapsed_alias(self):
        kw = _async_finalize(
            {"model": "claude-haiku-4-5", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "claude-haiku-4-5")
        self.assertEqual(kw["span"]["span_name"], "claude-haiku-4-5")

    # ── 5. missing kwargs: fail-open guard `(self._kwargs or {})` ──
    def test_sync_missing_kwargs_logs_echo_unchanged_without_raising(self):
        kw = _sync_finalize(None, "claude-haiku-4-5-20251001")
        self.assertEqual(kw["model"], "claude-haiku-4-5-20251001")

    def test_async_missing_kwargs_logs_echo_unchanged_without_raising(self):
        kw = _async_finalize(None, "claude-haiku-4-5-20251001")
        self.assertEqual(kw["model"], "claude-haiku-4-5-20251001")

    # ── 6. NEW-1: Bedrock CRIS profile id — the exact seams (enforcer.py
    # `_finalize`/`_afinalize`) that produced the production bug. The provider
    # echoes only the bare model id on the streaming taps; the profile id is
    # what AWS actually bills, so it must reach the logged payload verbatim,
    # and the span_name fallback must follow it.
    def test_sync_bedrock_cris_profile_bare_echo_profile_id_wins(self):
        kw = _sync_finalize(
            {"model": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertEqual(kw["span"]["span_name"], "us.anthropic.claude-haiku-4-5-20251001-v1:0")

    def test_async_bedrock_cris_profile_bare_echo_profile_id_wins(self):
        kw = _async_finalize(
            {"model": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "messages": []},
            "claude-haiku-4-5-20251001",
        )
        self.assertEqual(kw["model"], "us.anthropic.claude-haiku-4-5-20251001-v1:0")
        self.assertEqual(kw["span"]["span_name"], "us.anthropic.claude-haiku-4-5-20251001-v1:0")

    # ── 7. negative twin: a genuinely different served model keeps its echo ──
    def test_sync_bedrock_negative_twin_keeps_the_echo(self):
        kw = _sync_finalize(
            {"model": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "messages": []},
            "claude-sonnet-4-6",
        )
        self.assertEqual(kw["model"], "claude-sonnet-4-6")

    def test_async_bedrock_negative_twin_keeps_the_echo(self):
        kw = _async_finalize(
            {"model": "us.anthropic.claude-haiku-4-5-20251001-v1:0", "messages": []},
            "claude-sonnet-4-6",
        )
        self.assertEqual(kw["model"], "claude-sonnet-4-6")


if __name__ == "__main__":
    unittest.main()
