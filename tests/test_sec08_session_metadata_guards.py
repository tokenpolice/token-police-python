"""Regression: tp.session()/agent()/chain() must never throw into / silently
corrupt customer code on a non-dict `metadata`.

Defect (pre-fix, token_police/context.py):
  - top-level branch `metadata=metadata or {}` (:353) stored a truthy non-dict verbatim
    → `_structural_span:282` `(s.metadata or {}).items()` raised AttributeError in __enter__
    (before the customer with-body).
  - nested branch `merged_meta.update(metadata)` (:332) raised ValueError/TypeError on a
    non-dict, or SILENTLY misread a list of 2-char strings as key/value pairs.

Fix: two `isinstance(metadata, dict)` gates in `_session_impl`, so TPSession.metadata is
always a dict and __enter__ never throws from any of session()/agent()/chain().

ANTI-VACUITY: every no-throw case invokes `with tp.<entry>(...)` with NO surrounding
swallowing try/except — a pre-fix escape from __enter__ surfaces as a pytest ERROR, not a
quietly-passed catch. A sentinel set INSIDE the body is asserted AFTER the with-block, and
the exact stored metadata value is checked.
"""
import asyncio  # noqa: F401 (kept out of use — session/agent/chain are sync context managers)
import pytest

import token_police as tp
from token_police import context as tp_context


# ── helpers ────────────────────────────────────────────────────────────────

class MD(dict):
    """A dict subclass — must pass the isinstance(dict) gate (parity)."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Assertion 1 & 2 are covered by the repro script (py8_session_metadata.py);
# the direct-API equivalents are assertions 3-14, 23, 24 below.
# ═══════════════════════════════════════════════════════════════════════════


# ── Assertions 3-7: TOP-LEVEL no-throw matrix (no enclosing session) ────────
# NO surrounding try/except: a pre-fix __enter__ escape → test ERROR.

def test_a3_toplevel_session_str_metadata():
    ran = {}
    with tp.session(name="wf", metadata="tenant-A") as s:
        ran["body"] = True
        assert tp.get_current_session().metadata == {}
        assert s.metadata == {}
    assert ran["body"] is True  # body executed (asserted AFTER the with)


def test_a4_toplevel_agent_int_metadata():
    ran = {}
    with tp.agent(name="wf", metadata=123) as s:
        ran["body"] = True
        assert s.metadata == {}
    assert ran["body"] is True


def test_a5_toplevel_chain_list_metadata():
    ran = {}
    with tp.chain(name="wf", metadata=["t1", "t2"]) as s:
        ran["body"] = True
        assert s.metadata == {}
    assert ran["body"] is True


def test_a6_toplevel_falsy_nondict_metadata():
    for falsy in (0, []):
        ran = {}
        with tp.session(name="wf", metadata=falsy) as s:
            ran["body"] = True
            assert s.metadata == {}
        assert ran["body"] is True


def test_a7_toplevel_none_metadata():
    ran = {}
    with tp.session(name="wf", metadata=None) as s:
        ran["body"] = True
        assert s.metadata == {}
    assert ran["body"] is True


# ── Assertions 8-10: NESTED no-throw matrix (parent metadata preserved) ─────

def test_a8_nested_session_str_metadata():
    outer_ran = {}
    with tp.session(name="outer", metadata={"ok": "1"}):
        inner_ran = {}
        with tp.session(name="inner", metadata="tenant-A") as inner:
            inner_ran["body"] = True
            assert inner.metadata == {"ok": "1"}  # parent preserved, non-dict dropped
        assert inner_ran["body"] is True
        outer_ran["body"] = True
    assert outer_ran["body"] is True


def test_a9_nested_agent_list_metadata_no_corruption():
    with tp.session(name="outer", metadata={"ok": "1"}):
        inner_ran = {}
        with tp.agent(name="inner", metadata=["t1", "t2"]) as inner:
            inner_ran["body"] = True
            # Pre-fix this was silently corrupted to {"ok": "1", "t": "2"}.
            assert inner.metadata == {"ok": "1"}
            assert "t" not in inner.metadata
        assert inner_ran["body"] is True


def test_a10_nested_chain_int_metadata():
    with tp.session(name="outer", metadata={"ok": "1"}):
        inner_ran = {}
        with tp.chain(name="inner", metadata=5) as inner:
            inner_ran["body"] = True
            assert inner.metadata == {"ok": "1"}
        assert inner_ran["body"] is True


# ── Assertion 11: structural-span consumption safety (the crash site) ───────

def test_a11_structural_span_never_receives_nondict():
    # The span opens BEFORE `yield` (_structural_span:307); reaching the body
    # proves _structural_span was entered without raising. s.metadata being a
    # dict is the invariant that makes:282 `.items()` safe.
    for entry, meta in ((tp.session, "tenant-A"), (tp.agent, 123), (tp.chain, ["t1", "t2"])):
        entered = {}
        with entry(name="wf", metadata=meta) as s:
            entered["ok"] = True
            assert isinstance(s.metadata, dict) is True
        assert entered["ok"] is True


# ── Assertions 12-14: behavior preservation (valid dict handling) ───────────

def test_a12_toplevel_valid_dict():
    with tp.session(name="wf", metadata={"k": "v"}) as s:
        assert s.metadata == {"k": "v"}


def test_a13_nested_valid_dict_merges_calltime_wins():
    with tp.session(name="outer", metadata={"ok": "1", "shared": "outer"}):
        with tp.session(name="inner", metadata={"shared": "inner", "new": "n"}) as inner:
            assert inner.metadata == {"ok": "1", "shared": "inner", "new": "n"}


def test_a14_dict_subclass_admitted():
    # top-level: subclass stored, items present
    with tp.session(name="wf", metadata=MD({"k": "v"})) as s:
        assert s.metadata.get("k") == "v"
    # nested: subclass merges over parent
    with tp.session(name="outer", metadata={"ok": "1"}):
        with tp.agent(name="inner", metadata=MD({"new": "n"})) as inner:
            assert inner.metadata == {"ok": "1", "new": "n"}


# ── Assertion 15: entry-point coverage (explicit) ───────────────────────────
# Covered by 3/4/5 (top-level, one per entry) and 8/9/10 (nested, one per entry).
# This test asserts all three public entry points exist and are callable CMs.

def test_a15_all_three_entry_points_guarded():
    for entry in (tp.session, tp.agent, tp.chain):
        ran = {}
        with entry(name="wf", metadata="not-a-dict") as s:  # top-level
            ran["body"] = True
            assert s.metadata == {}
        assert ran["body"] is True
        with tp.session(name="outer", metadata={"ok": "1"}):  # nested
            with entry(name="inner", metadata="not-a-dict") as inner:
                assert inner.metadata == {"ok": "1"}


# ── Assertion 16: mode neutrality (no enforcement reference introduced) ─────
# Structural check performed by the Evaluator on the diff (grep enforce|dry_run|
# Blocked|block). Runtime smoke: opening a session with hostile metadata never
# raises TokenPoliceBlockedError.

def test_a16_no_block_on_hostile_metadata():
    from token_police.exceptions import TokenPoliceBlockedError
    try:
        with tp.session(name="wf", metadata="tenant-A"):
            pass
    except TokenPoliceBlockedError:
        pytest.fail("hostile metadata must not trigger a block (mode neutrality)")


# ── Assertion 23: nested falsy preserves the falsy-skip no-op ────────────────

def test_a23_nested_falsy_nondict_preserves_parent():
    for falsy in (0, []):
        with tp.session(name="outer", metadata={"ok": "1"}):
            inner_ran = {}
            with tp.session(name="inner", metadata=falsy) as inner:
                inner_ran["body"] = True
                assert inner.metadata == {"ok": "1"}
            assert inner_ran["body"] is True


# ── Assertion 24: empty-dict behavior preserved on BOTH branches ─────────────

def test_a24_empty_dict_both_branches():
    # (a) top-level {} -> {}
    with tp.session(name="wf", metadata={}) as s:
        assert s.metadata == {}
    # (b) nested {} under parent {"ok":"1"} -> {"ok":"1"}
    with tp.session(name="outer", metadata={"ok": "1"}):
        with tp.session(name="inner", metadata={}) as inner:
            assert inner.metadata == {"ok": "1"}
