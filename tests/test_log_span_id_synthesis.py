"""C-12 — /log span_id synthesis (``log_sync`` / async ``log`` wire payload).

The collector keys BOTH its per-span idempotency guard and the persisted
generations row's span_id off ``span["span_id"]``. A manual caller that never
built a ``span`` block — or built one without a ``span_id`` — used to send NO
``span`` key at all (or a span with no id), so the row landed with an empty
span_id: it collapsed the trace tree and starved ClickHouse's
insert_deduplication_token of the entropy that keeps replayed flush batches
from silently deduping against each other.

The fix: both payload builders always emit a ``span`` block carrying a
non-empty span_id — synthesized (16 lowercase hex chars, ``random_hex16()``)
only when the caller didn't supply one (or supplied a non-dict / empty-id
one). A caller-supplied span_id is forwarded byte-identical. The caller's
dict is never mutated (shallow copy), and every step is independently
guarded so a hostile ``span`` value can never raise into caller code.

Offline capture idiom (sync client's ``post`` mocked; async client is a
captured ``FakeAsyncClient`` driven by ``asyncio.run``) mirrors
tests/test_workflow_name_forwarding.py / tests/test_plan_source.py.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Make the local SDK importable without a prior editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police.client import TokenPolice

HEX16 = re.compile(r"^[0-9a-f]{16}$")


def _make_client() -> TokenPolice:
    """Bare client, unroutable base_url, no SSE thread. No real network I/O."""
    return TokenPolice(
        api_key="tp_sk_test",
        base_url="http://127.0.0.1:1",
        timeout=0.1,
        deployment="serverless",
        log_errors=False,
    )


def _capture_log_sync(c: TokenPolice, **kwargs):
    captured: dict = {}

    def fake_post(path, json=None):
        captured[path] = json
        return MagicMock(status_code=200)

    c._sync_client.post = fake_post
    kwargs.setdefault("input_tokens", 1)
    kwargs.setdefault("output_tokens", 1)
    result = c.log_sync(**kwargs)
    c.flush_sync()  # drain the background _send so the mock is invoked
    return captured["/v1/guard/log"], result


def _capture_log_async(c: TokenPolice, **kwargs):
    captured: dict = {}

    class FakeAsyncClient:
        async def post(self, path, json=None):
            captured[path] = json
            return MagicMock(status_code=200)

    c._async_client_instance = FakeAsyncClient()
    kwargs.setdefault("input_tokens", 1)
    kwargs.setdefault("output_tokens", 1)

    async def scenario():
        r = await c.log(**kwargs)
        await c.flush()  # drain the create_task-scheduled _send
        return r

    result = asyncio.run(scenario())
    return captured["/v1/guard/log"], result


ALL_LOG_CAPTURERS = {
    "log_sync": _capture_log_sync,
    "log_async": _capture_log_async,
}


# ── (j)/(k): span=None (the manual-call default) → synthesized span_id ───────

def test_log_sync_no_span_synthesizes_hex16_span_id_and_span_kind():
    c = _make_client()
    payload, result = _capture_log_sync(c)
    span = payload["span"]
    assert HEX16.match(span["span_id"])
    assert isinstance(span.get("span_kind"), str) and span["span_kind"]
    assert result is None  # fire-and-forget


def test_log_async_no_span_synthesizes_hex16_span_id_and_span_kind():
    c = _make_client()
    payload, result = _capture_log_async(c)
    span = payload["span"]
    assert HEX16.match(span["span_id"])
    assert isinstance(span.get("span_kind"), str) and span["span_kind"]
    assert result is None


def test_both_builders_produce_different_span_ids_across_calls():
    for name, capturer in ALL_LOG_CAPTURERS.items():
        c = _make_client()
        p1, _ = capturer(c)
        p2, _ = capturer(c)
        assert p1["span"]["span_id"] != p2["span"]["span_id"], name


# ── (l): span dict without span_id → copied + synthesized, no mutation ───────

def test_log_sync_span_missing_id_keeps_other_keys_gains_synthesized_id():
    c = _make_client()
    caller_span = {"trace_id": "t-abc", "span_name": "n-abc"}
    payload, _ = _capture_log_sync(c, span=caller_span)
    wire_span = payload["span"]
    assert wire_span["trace_id"] == "t-abc"
    assert wire_span["span_name"] == "n-abc"
    assert HEX16.match(wire_span["span_id"])
    # The caller's dict is never mutated (shallow copy) — no span_id/span_kind
    # key leaked backward onto the object the application still holds.
    assert caller_span == {"trace_id": "t-abc", "span_name": "n-abc"}
    assert "span_id" not in caller_span
    assert "span_kind" not in caller_span


def test_log_async_span_missing_id_does_not_mutate_caller_dict():
    c = _make_client()
    caller_span = {"trace_id": "t-xyz"}
    payload, _ = _capture_log_async(c, span=caller_span)
    wire_span = payload["span"]
    assert wire_span["trace_id"] == "t-xyz"
    assert HEX16.match(wire_span["span_id"])
    assert caller_span == {"trace_id": "t-xyz"}


# ── (m): explicit span_id passes through byte-identical ──────────────────────

def test_log_sync_explicit_span_id_passes_through_unchanged():
    c = _make_client()
    payload, _ = _capture_log_sync(c, span={"span_id": "my-custom-id"})
    assert payload["span"]["span_id"] == "my-custom-id"


def test_log_async_explicit_span_id_passes_through_unchanged():
    c = _make_client()
    payload, _ = _capture_log_async(c, span={"span_id": "my-custom-id"})
    assert payload["span"]["span_id"] == "my-custom-id"


def test_log_sync_deterministic_batch_style_span_id_is_forwarded_verbatim():
    # Mirrors the Anthropic-batch re-read case: a caller-supplied deterministic
    # id that WANTS server-side dedup must never be replaced by a fresh one.
    c = _make_client()
    payload, _ = _capture_log_sync(c, span={"span_id": "batch-req-42", "trace_id": "t1"})
    assert payload["span"]["span_id"] == "batch-req-42"
    assert payload["span"]["trace_id"] == "t1"


# ── (n)/(o) GOLDEN RULE: hostile / non-dict span never raises ────────────────

def test_log_sync_non_dict_span_string_no_exception_still_sends_synthesized_span():
    c = _make_client()
    payload, result = _capture_log_sync(c, span="not-a-dict")
    assert result is None  # no exception into the caller
    span = payload["span"]
    assert HEX16.match(span["span_id"])


def test_log_async_non_dict_span_list_no_exception_still_sends_synthesized_span():
    c = _make_client()
    payload, result = _capture_log_async(c, span=[1, 2, 3])
    assert result is None
    span = payload["span"]
    assert HEX16.match(span["span_id"])


def test_log_sync_non_dict_span_int_no_exception():
    c = _make_client()
    payload, result = _capture_log_sync(c, span=12345)
    assert result is None
    assert HEX16.match(payload["span"]["span_id"])


class _HostileMapping:
    """A non-``dict`` mapping-like object whose item access raises. Since
    ``isinstance(span, dict)`` is ``False`` for this type, the synthesis gate
    takes the same safe fallback as the string/list/int cases above — this
    pins that a mapping-*shaped* (but not dict-typed) hostile object degrades
    identically rather than being treated specially by duck-typing."""

    def __getitem__(self, key):
        raise RuntimeError("hostile getitem")

    def get(self, *args, **kwargs):
        raise RuntimeError("hostile get")

    def keys(self):
        raise RuntimeError("hostile keys")


def test_log_sync_hostile_mapping_like_span_does_not_raise():
    c = _make_client()
    payload, result = _capture_log_sync(c, span=_HostileMapping())
    # No exception surfaces into the caller; log_sync returns normally
    # (fire-and-forget) and a synthesized span block still made it onto the
    # wire — `isinstance(span, dict)` is False here, so none of the hostile
    # methods above are ever actually invoked (documented, not just assumed).
    assert result is None
    assert HEX16.match(payload["span"]["span_id"])
