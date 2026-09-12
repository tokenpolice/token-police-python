"""G4 presence helpers (sdk_test_hardening_design.md T-P5).

Fail-open code converts failures into silence. Asserting only properties of
rows that exist cannot catch "row never created". After driving a usage-bearing
mock stream, call ``assert_streamed_log_present`` on the fake client's call list.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence


def _tokens_of(row: Mapping[str, Any]) -> int:
    """Best-effort token sum from a log_sync / log kwargs dict."""
    # Common shapes used across the Python suite.
    if "input_tokens" in row or "output_tokens" in row:
        return int(row.get("input_tokens") or 0) + int(row.get("output_tokens") or 0)
    tokens = row.get("tokens") or {}
    if isinstance(tokens, Mapping):
        return int(tokens.get("input_tokens") or 0) + int(
            tokens.get("output_tokens") or 0
        )
    usage = row.get("usage") or {}
    if isinstance(usage, Mapping):
        return int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0) + int(
            usage.get("output_tokens") or usage.get("completion_tokens") or 0
        )
    return 0


def assert_streamed_log_present(
    calls: Sequence[Mapping[str, Any]],
    *,
    provider: Optional[str] = None,
    min_count: int = 1,
    min_tokens: int = 1,
) -> list:
    """Assert at least ``min_count`` log rows exist with tokens >= ``min_tokens``.

    When ``provider`` is set, only rows whose ``provider`` field matches are
    counted. Returns the matching rows for further assertions.
    """
    rows = list(calls)
    if provider is not None:
        rows = [r for r in rows if r.get("provider") == provider]
    assert len(rows) >= min_count, (
        f"G4 presence: expected >= {min_count} log row(s)"
        + (f" for provider={provider!r}" if provider else "")
        + f", got {len(rows)}"
    )
    usable = [r for r in rows if _tokens_of(r) >= min_tokens]
    assert len(usable) >= min_count, (
        f"G4 presence: expected >= {min_count} row(s) with tokens>={min_tokens}, "
        f"got {len(usable)} usable of {len(rows)} total; sample={rows[:2]!r}"
    )
    return usable
