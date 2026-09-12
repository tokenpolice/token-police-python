"""Unit coverage for the per-call anthropic stream-span suppression window
(``token_police/context.py``): ``arm_anthropic_stream_span_window``,
``disarm_anthropic_stream_span_window``, and ``claim_anthropic_stream_span``.

This mechanism replaces the deleted session-wide one-shot
``_suppress_anthropic_otel_stream`` flag. A ContextVar holds a PER-CALL record
``{"seen": int, "limit": int}``; ``claim_anthropic_stream_span`` matches on TWO
independent instrumentor-owned identifiers (the instrumentation scope name and
the ``anthropic.``-prefixed span name) so a rename of either alone still
suppresses, and is fail-open on any doubt (never suppress → an extra row beats
silent telemetry loss).

No anthropic SDK, no OTel spans, no session/client plumbing — pure functions
under direct test. Enforcer-level integration (arming the window around the
real ``.stream()`` seam) is covered by tests/test_anthropic_stream_cm_double_log.py
and the rewritten tests/test_anthropic_stream_enter_failure.py /
tests/test_anthropic_async_stream_check.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police.context import (  # noqa: E402
    arm_anthropic_stream_span_window,
    claim_anthropic_stream_span,
    disarm_anthropic_stream_span_window,
    _anthropic_stream_span_window,
)

REAL_SCOPE = "opentelemetry.instrumentation.anthropic"
REAL_NAME = "anthropic.chat"


def _span(name="anthropic.chat"):
    """A minimal stand-in — claim_anthropic_stream_span only reads `.name`."""
    return NS(name=name)


def _assert_window_clear():
    assert _anthropic_stream_span_window.get() is None


# ═══════════════════════════════════════════════════════════════════
# 1a. Matcher: scope-only, name-only, neither.
# ═══════════════════════════════════════════════════════════════════

class TestMatcher:
    def test_scope_match_alone_claims(self):
        """Scope matches the OTel prefix; span name is unrelated (a hypothetical
        instrumentor rename of the span name alone must not break suppression)."""
        _assert_window_clear()
        h = arm_anthropic_stream_span_window()
        try:
            assert claim_anthropic_stream_span(
                _span(name="totally.unrelated.span.name"), REAL_SCOPE
            ) is True
        finally:
            disarm_anthropic_stream_span_window(h)
        _assert_window_clear()

    def test_name_match_alone_claims_even_with_foreign_scope(self):
        """Span name carries the anthropic prefix but the instrumentation scope
        has been renamed/is foreign — the second independent identifier alone
        is still enough to claim."""
        _assert_window_clear()
        h = arm_anthropic_stream_span_window()
        try:
            assert claim_anthropic_stream_span(
                _span(name=REAL_NAME), "some.other.instrumentor.entirely"
            ) is True
        finally:
            disarm_anthropic_stream_span_window(h)
        _assert_window_clear()

    def test_neither_identifier_matches_returns_false(self):
        """A genuinely unrelated span (e.g. the OpenAI instrumentor's own
        span) must never be claimed, even with a window armed."""
        h = arm_anthropic_stream_span_window()
        try:
            assert claim_anthropic_stream_span(
                _span(name="openai.chat"), "opentelemetry.instrumentation.openai"
            ) is False
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_no_armed_window_returns_false(self):
        """Sanity: nothing armed at all -> never suppress on doubt."""
        _assert_window_clear()
        assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False


# ═══════════════════════════════════════════════════════════════════
# 1b. Record exhaustion: limit: 1 means a second claim on the SAME window
# returns False.
# ═══════════════════════════════════════════════════════════════════

class TestExhaustion:
    def test_second_claim_on_same_window_fails(self):
        h = arm_anthropic_stream_span_window()
        try:
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is True
            # Same record, same window -> exhausted.
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_seen_increments_exactly_once_per_successful_claim(self):
        h = arm_anthropic_stream_span_window()
        record = h[0]
        try:
            assert record == {"seen": 0, "limit": 1}
            claim_anthropic_stream_span(_span(), REAL_SCOPE)
            assert record == {"seen": 1, "limit": 1}
            # A failed (exhausted) claim must NOT increment further.
            claim_anthropic_stream_span(_span(), REAL_SCOPE)
            assert record == {"seen": 1, "limit": 1}
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_re_armed_record_with_higher_limit_allows_more_claims(self):
        """Passing an existing record to arm() re-arms it WITHOUT resetting
        its claim count (used by the manager-enter window, which reuses the
        call's own W1 record)."""
        record = {"seen": 0, "limit": 2}
        h = arm_anthropic_stream_span_window(record=record)
        try:
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is True
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is True
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
        finally:
            disarm_anthropic_stream_span_window(h)


# ═══════════════════════════════════════════════════════════════════
# 1c. Malformed records: never raise, always degrade to False (never
# suppress on doubt).
# ═══════════════════════════════════════════════════════════════════

class TestMalformedRecords:
    def test_non_dict_record_returns_false(self):
        h = arm_anthropic_stream_span_window(record="not-a-dict")
        try:
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_non_dict_record_list_returns_false(self):
        h = arm_anthropic_stream_span_window(record=[1, 2, 3])
        try:
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_seen_non_numeric_string_returns_false_no_raise(self):
        h = arm_anthropic_stream_span_window(record={"seen": "x", "limit": 1})
        try:
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_empty_dict_returns_false_no_raise(self):
        h = arm_anthropic_stream_span_window(record={})
        try:
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_limit_none_returns_false_no_raise(self):
        h = arm_anthropic_stream_span_window(record={"seen": 0, "limit": None})
        try:
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is False
        finally:
            disarm_anthropic_stream_span_window(h)

    def test_hostile_span_name_getattr_does_not_raise(self):
        """A span whose `.name` access raises must still degrade gracefully
        (the name-read is independently guarded), not propagate — the caller
        (on_start) must never crash on a hostile span object. Paired with a
        FOREIGN scope so the scope identifier alone can't also match — this
        isolates the name-access guard specifically."""
        class _HostileSpan:
            @property
            def name(self):
                raise RuntimeError("hostile span.name")

        h = arm_anthropic_stream_span_window()
        try:
            assert claim_anthropic_stream_span(
                _HostileSpan(), "some.foreign.scope"
            ) is False
        finally:
            disarm_anthropic_stream_span_window(h)


# ═══════════════════════════════════════════════════════════════════
# 1d. arm/disarm nesting via Token.reset: arming inside an armed window and
# disarming restores the OUTER record intact (not cleared to None).
# ═══════════════════════════════════════════════════════════════════

class TestNesting:
    def test_nested_arm_disarm_restores_outer_record_intact(self):
        _assert_window_clear()
        outer = arm_anthropic_stream_span_window()
        outer_record = outer[0]
        try:
            assert _anthropic_stream_span_window.get() is outer_record
            inner = arm_anthropic_stream_span_window()
            inner_record = inner[0]
            assert _anthropic_stream_span_window.get() is inner_record
            assert inner_record is not outer_record

            # Claim inside the inner (nested) window — must not touch outer.
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is True
            assert inner_record == {"seen": 1, "limit": 1}
            assert outer_record == {"seen": 0, "limit": 1}

            disarm_anthropic_stream_span_window(inner)
            # OUTER record re-exposed, intact (not cleared to None).
            assert _anthropic_stream_span_window.get() is outer_record
            assert outer_record == {"seen": 0, "limit": 1}

            # Outer window is still independently claimable.
            assert claim_anthropic_stream_span(_span(), REAL_SCOPE) is True
            assert outer_record == {"seen": 1, "limit": 1}
        finally:
            disarm_anthropic_stream_span_window(outer)
        _assert_window_clear()


# ═══════════════════════════════════════════════════════════════════
# 1e. disarm with malformed handles is a total no-op — never raises, never
# mutates the ContextVar.
# ═══════════════════════════════════════════════════════════════════

class TestDisarmMalformedHandles:
    def test_malformed_handles_never_raise_and_never_mutate_state(self):
        _assert_window_clear()
        armed = arm_anthropic_stream_span_window()
        try:
            before = _anthropic_stream_span_window.get()
            for bad in (object(), (1,), "x", 5, None):
                disarm_anthropic_stream_span_window(bad)  # must not raise
                assert _anthropic_stream_span_window.get() is before
        finally:
            disarm_anthropic_stream_span_window(armed)
        _assert_window_clear()

    def test_disarm_none_is_explicit_noop(self):
        """None is the documented no-window sentinel from a failed arm()."""
        _assert_window_clear()
        disarm_anthropic_stream_span_window(None)
        _assert_window_clear()


# ═══════════════════════════════════════════════════════════════════
# 1f. arm() itself is fail-open too.
# ═══════════════════════════════════════════════════════════════════

class TestArmFailOpen:
    def test_arm_with_no_record_builds_the_default_shape(self):
        h = arm_anthropic_stream_span_window()
        try:
            assert h is not None
            record, token = h
            assert record == {"seen": 0, "limit": 1}
            assert token is not None
        finally:
            disarm_anthropic_stream_span_window(h)
