"""Tool-arg hash/length SINK parity (Python side).

The Python tool-arg sinks fingerprint an arg as (sha1-16, code-point length):
  - ``telemetry._hash_len`` — dual-use: OTel ``_log_tool_span`` AND
                                    ``context._emit_tool_row`` (values dicts)
  - ``openai_agents._hash_len`` — Agents-SDK function-span path AND the
                                    pydantic_ai tool-row emit
                                    (``enforcer._emit_pydantic_ai_tool_row``,
                                    which imports it from ``.openai_agents``)

A STRING arg is hashed BYTE-UNCHANGED (preserves the OTel span-attr hash
baseline — NOT canonicalized/quoted); ``len()`` already counts code points. A
NON-STRING arg routes through the ``_canonical_json`` then hashed.

Reads the SHARED oracle ``shared/composition-canonical-fixture.json``
``tool_arg_cases`` block (also read by toolArgCanonical.test.ts) — every pinned
value was COMPUTED by running both runtimes' serializers + sha1 + code-point
count and cross-checked byte-identical.
"""
import asyncio
import hashlib
import json
import os

import pytest

from token_police.composition import _canonical_json
from token_police import telemetry as tp_telemetry
from token_police import openai_agents as tp_openai_agents
from token_police import context as ctx
from token_police.context import TPSession
from token_police.state import set_client
from token_police.enforcer import _make_pydantic_ai_tool_wrapper

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "shared", "composition-canonical-fixture.json"
)


def _sha116(s):
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:16]


def _load_cases():
    with open(_FIXTURE) as f:
        return json.load(f)["tool_arg_cases"]


# The two Python sinks, addressed directly (they ARE the sink functions).
SINKS = [
    ("telemetry._hash_len", tp_telemetry._hash_len),
    ("openai_agents._hash_len", tp_openai_agents._hash_len),
]


def test_fixture_present():
    cases = _load_cases()
    assert cases, "tool_arg_cases must not be empty"


def test_sinks_reproduce_shared_oracle():
    """Assertions 4,5,9,10,12,18 — each sink reproduces the pinned oracle."""
    cases = _load_cases()
    for name, sink in SINKS:
        for c in cases:
            h, ln = sink(c["input"])
            assert h == c["hash"], f"{name} hash mismatch on {c['_name']}"
            assert ln == c["length"], f"{name} length mismatch on {c['_name']}"
            if c["_branch"] == "canonical":
                assert _sha116(c["canonical"]) == c["hash"]
                assert len(c["canonical"]) == c["length"]
                # routed through the canonical serializer, NOT str(dict)
                assert sink(c["input"])[0] == _sha116(_canonical_json(c["input"]))
                assert sink(c["input"])[0] != _sha116(str(c["input"]))
            else:
                # string branch: raw input hashed byte-unchanged
                assert _sha116(c["input"]) == c["hash"]
                assert len(c["input"]) == c["length"]


def test_ascii_string_passthrough():
    """Assertion 6 — 'hello' hashed raw (NOT quoted), length 5."""
    for name, sink in SINKS:
        h, ln = sink("hello")
        assert h == "aaf4c61ddcc5e8a2", name
        assert h != _sha116('"hello"'), name  # would be this if we canonicalized
        assert ln == 5, name


def test_astral_string_length_code_points():
    """Assertion 11 — astral STRING length = code points; hash unchanged."""
    for name, sink in SINKS:
        h, ln = sink("😀")
        assert ln == 1, name
        assert h == _sha116("😀"), name
        h2, ln2 = sink('{"emoji":"😀"}')
        assert ln2 == 13, name
        assert h2 == "f71d0df8d03ac2e0", name
        # BMP no-regression
        h3, ln3 = sink('{"query":"weather"}')
        assert ln3 == 19, name
        assert h3 == "ca7991fe11831ace", name


def test_null_early_return():
    """Assertion 7 — None → ('', 0)."""
    for name, sink in SINKS:
        assert sink(None) == ("", 0), name


def test_no_throw_on_hostile_input():
    """Assertion 8 — hostile non-string args return a tuple, never raise."""

    class _Circular:
        pass

    circ = _Circular()
    circ.self = circ  # cycle via attribute (dict-of-cycle also below)
    d = {}
    d["self"] = d

    class _ThrowingStr:
        def __repr__(self):
            raise RuntimeError("repr explodes")

        def __str__(self):
            raise RuntimeError("str explodes")

    hostiles = [d, _ThrowingStr(), object(), 2 ** 70, {"x": _ThrowingStr()}]
    for name, sink in SINKS:
        for h in hostiles:
            out = sink(h)  # must not raise
            assert isinstance(out, tuple) and len(out) == 2, name
            assert isinstance(out[0], str) and isinstance(out[1], int), name


# ── Assertion 17: pydantic_ai non-string arg via BOTH call sites ──


class _FakeCallNonStringArgs:
    """args_as_json_str() raises → enforcer falls back to getattr(call,'args'),
    which here is a NON-STRING dict → hits the canonical (non-string) branch."""

    def __init__(self, args_dict):
        self.tool_name = "lookup"
        self.tool_call_id = "call_9"
        self._args = args_dict

    def args_as_json_str(self):
        raise ValueError("no json form")

    @property
    def args(self):
        return self._args


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_pydantic_ai_non_string_arg_uses_canonical_serializer():
    """Assertion 17 — the pydantic_ai emit path fingerprints a non-string dict
    via the canonical serializer (== direct openai_agents._hash_len(dict)),
    proving the single _hash_len fix reaches BOTH call sites."""
    # representative unsorted dict from the fixture (assertion 9 case)
    case = next(c for c in _load_cases() if c["_branch"] == "canonical" and c["input"].get("a") is True)
    arg = case["input"]

    client = _FakeClient()
    set_client(client)
    session = TPSession(
        user_id="u1", paid_plan="pro", workflow_name="wf",
        trace_id="c" * 32, root_span_id="d" * 16,
    )
    token = ctx._current_session.set(session)
    try:

        async def original(self, call, allow_partial=False, wrap_validation_errors=True):
            return "ok"

        wrapper = _make_pydantic_ai_tool_wrapper(original)
        _run(wrapper(object(), _FakeCallNonStringArgs(arg), False, True))
    finally:
        ctx._current_session.reset(token)

    assert len(client.calls) == 1
    tool = client.calls[0]["tool"]
    # oracle: canonical-serializer fingerprint, NOT str(dict)
    assert tool["param_hash"] == case["hash"]
    assert tool["param_length"] == case["length"]
    assert (tool["param_hash"], tool["param_length"]) == tp_openai_agents._hash_len(arg)
    assert tool["param_hash"] != _sha116(str(arg))
