"""Python↔Node SDK usage-key parity (G1/G5).

Both SDK SpanProcessors reconstruct a `usage` block from the same `gen_ai.*`
semconv attributes (Mode-A). They MUST emit the same `usage.raw` keys/values so
the server's usage-mapper sees one shape regardless of SDK. The shared fixture
is the single source of truth; the Node counterpart
(tests/sdkUsageParity.test.ts) asserts the identical contract.

Specifically locks G5: cache *read* and cache *creation* (write) stay DISJOINT,
and reasoning tokens are forwarded.
"""
import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police.telemetry import TokenPoliceSpanProcessor

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "shared", "sdk-usage-parity-fixture.json"
)


def _load_fixture():
    with open(_FIXTURE) as f:
        return json.load(f)


def _capture_log(attrs):
    captured = {}

    class _FakeClient:
        def log_sync(self, **kwargs):
            captured.update(kwargs)

    span = NS(
        attributes=attrs,
        name="anthropic.chat",
        instrumentation_scope=NS(name="opentelemetry.instrumentation.anthropic"),
        context=NS(trace_id=0xABCD, span_id=0x1234),
        parent=None,
        start_time=0,
        end_time=1,
    )
    proc = TokenPoliceSpanProcessor()
    with mock.patch("token_police.state.get_client", return_value=_FakeClient()):
        proc.on_end(span)
    return captured


class TestSdkUsageParity(unittest.TestCase):
    def test_matches_shared_fixture(self):
        fx = _load_fixture()
        raw = _capture_log(fx["input_attrs"])["usage"]["raw"]
        exp = fx["expected_usage_raw"]
        self.assertEqual(
            raw.get("cache_read_input_tokens"), exp["cache_read_input_tokens"]
        )
        self.assertEqual(
            raw.get("cache_creation_input_tokens"), exp["cache_creation_input_tokens"]
        )
        self.assertEqual(
            raw["prompt_tokens_details"]["cached_tokens"],
            exp["prompt_tokens_details"]["cached_tokens"],
        )
        self.assertEqual(
            raw["completion_tokens_details"]["reasoning_tokens"],
            exp["completion_tokens_details"]["reasoning_tokens"],
        )

    def test_service_tier_canonicalization_matches_fixture(self):
        """A1: both SDK extractors + the server normalize service_tier the
        same way — '' for default/unknown tiers, never a guessed discount."""
        from token_police.enforcer import _extract_service_tier

        fx = _load_fixture()
        for raw_value, expected in fx["service_tier_cases"].items():
            # OpenAI shape: response-level service_tier.
            self.assertEqual(
                _extract_service_tier(NS(service_tier=raw_value)), expected,
                f"response-level service_tier={raw_value!r}",
            )
            # Anthropic shape: inside the usage object.
            self.assertEqual(
                _extract_service_tier({"usage": {"service_tier": raw_value}}), expected,
                f"usage-level service_tier={raw_value!r}",
            )
        self.assertEqual(_extract_service_tier(None), "")
        self.assertEqual(_extract_service_tier({}), "")


class GoogleRequestServiceTierTest(unittest.TestCase):
    """R4: Gemini takes ``service_tier`` on the request GenerateContentConfig and
    never echoes it in the response, so the tier is harvested from the request
    config. ``_stash_request_service_tier`` writes it under the call's comp_key;
    ``_flush_deferred_spans`` reads it back onto ``usage.tier``."""

    ORDER = 3

    def _session(self):
        return NS(trace_id="trace-r4", _span_counter=self.ORDER, _pending_compositions={})

    def _stashed(self, session):
        return session._pending_compositions.get(
            f"{session.trace_id}:{self.ORDER}", {}
        ).get("service_tier")

    def test_dict_config(self):
        from token_police.enforcer import _stash_request_service_tier
        s = self._session()
        _stash_request_service_tier(s, "google", {"config": {"service_tier": "priority"}}, order=self.ORDER)
        self.assertEqual(self._stashed(s), "priority")

    def test_config_object(self):
        from token_police.enforcer import _stash_request_service_tier
        s = self._session()
        cfg = NS(service_tier="priority")
        _stash_request_service_tier(s, "gemini", {"config": cfg}, order=self.ORDER)
        self.assertEqual(self._stashed(s), "priority")

    def test_enum_like_value(self):
        from token_police.enforcer import _stash_request_service_tier

        class _Enum:
            def __str__(self):
                return "ServiceTier.PRIORITY"

        s = self._session()
        _stash_request_service_tier(s, "google", {"config": NS(service_tier=_Enum())}, order=self.ORDER)
        self.assertEqual(self._stashed(s), "priority")

    def test_absent_config(self):
        from token_police.enforcer import _stash_request_service_tier
        s = self._session()
        _stash_request_service_tier(s, "google", {"model": "gemini-2.5-pro"}, order=self.ORDER)
        self.assertIsNone(self._stashed(s))

    def test_standard_dropped(self):
        from token_police.enforcer import _stash_request_service_tier
        s = self._session()
        _stash_request_service_tier(s, "google", {"config": {"service_tier": "standard"}}, order=self.ORDER)
        self.assertIsNone(self._stashed(s))

    def test_raising_attribute_access_swallowed(self):
        from token_police.enforcer import _stash_request_service_tier

        class _Hostile:
            @property
            def service_tier(self):
                raise RuntimeError("boom")

        s = self._session()
        # Fail-open: must not raise, must stash nothing.
        _stash_request_service_tier(s, "google", {"config": _Hostile()}, order=self.ORDER)
        self.assertIsNone(self._stashed(s))

    def test_non_google_ignored(self):
        from token_police.enforcer import _stash_request_service_tier
        s = self._session()
        _stash_request_service_tier(s, "openai", {"config": {"service_tier": "priority"}}, order=self.ORDER)
        self.assertIsNone(self._stashed(s))


if __name__ == "__main__":
    unittest.main()
