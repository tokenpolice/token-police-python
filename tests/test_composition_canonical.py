"""Canonical tool-arg serializer parity (Python side).

Locks that ``_canonical_json`` produces a byte-identical canonical string to the
Node SDK's ``canonicalJson`` for JSON-native values, so the composition ``hash``
(sha1-16 over the canonical string) and ``length`` (code points) of a tool-call
entry match across SDKs. Reads the SHARED fixture
``shared/composition-canonical-fixture.json`` (also read by
compositionCanonical.test.ts) so both suites assert the identical inputs.

Also confirms ``_text_entry`` length is already code points and matches.
"""
import hashlib
import json
import os

from token_police.composition import _canonical_json, build_response_composition

_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "..", "shared", "composition-canonical-fixture.json"
)

# PINNED cross-SDK constant — present byte-for-byte in compositionCanonical.test.ts.
# _canonical_json({"b":2,"a":1}) == '{"a":1,"b":2}'; sha1-first-16 of that string.
CANON_HASH_AB = "4acc71e0547112eb"


def _sha116(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _load_fixture():
    with open(_FIXTURE) as f:
        return json.load(f)


def test_shared_fixture_byte_identical():
    """Assertions 1, 3, 5, 21, 22 — every fixture input → pinned canonical + length."""
    cases = _load_fixture()["cases"]
    assert cases, "fixture must not be empty"
    for c in cases:
        got = _canonical_json(c["input"])
        assert got == c["canonical"], f"{c['_name']}: {got!r} != {c['canonical']!r}"
        # assertion 5: length = code points of the (already-trimmed) canonical string
        assert len(got) == c["length"], f"{c['_name']}: len {len(got)} != {c['length']}"


def test_canon_hash_ab():
    """Assertion 4 — identical hash across SDKs via the pinned constant."""
    canon = _canonical_json({"b": 2, "a": 1})
    assert canon == '{"a":1,"b":2}'
    assert _sha116(canon) == CANON_HASH_AB


def test_integer_like_keys_sort_lexicographically():
    """Assertion 8 — string/code-point sort, NOT numeric."""
    assert _canonical_json({"10": 0, "2": 0, "1": 0}) == '{"1":0,"10":0,"2":0}'


def test_non_finite_numbers_become_null():
    """Assertion 6 — NaN / +-inf → null (cannot be a JSON literal)."""
    assert _canonical_json(
        {"x": float("nan"), "y": float("inf"), "z": float("-inf")}
    ) == '{"x":null,"y":null,"z":null}'


def test_integer_valued_float_and_half():
    """Assertion 7 — 1.0 → 1, 0.5 → 0.5."""
    assert _canonical_json({"n": 1.0}) == '{"n":1}'
    assert _canonical_json({"n": 0.5}) == '{"n":0.5}'


def test_negative_zero():
    """Decision rule 5(iii) — -0.0 → 0."""
    assert _canonical_json({"z": -0.0}) == '{"z":0}'


def test_float_notation_plain_decimal():
    """Assertion 21 — exponential repr normalized to plain decimal."""
    # repr(1e-7) == '1e-07' in Python; the expander forces plain decimal.
    assert _canonical_json({"lr": 1e-5}) == '{"lr":0.00001}'
    assert _canonical_json({"x": 1e-7}) == '{"x":0.0000001}'
    assert _canonical_json({"big": 1e30}) == '{"big":1000000000000000000000000000000}'


def test_string_escaping_u0001_and_slash():
    """Assertion 22 — U+0001 → lowercase \\u0001; '/', U+2028, U+2029 NOT escaped."""
    assert _canonical_json({"c": chr(1)}) == '{"c":"\\u0001"}'
    assert _canonical_json({"a": "/"}) == '{"a":"/"}'
    assert _canonical_json({"a": "  "}) == '{"a":"  "}'


def test_canonical_never_throws():
    """Assertion 11 — pathological inputs cannot throw; datetime is within-SDK
    deterministic (NOT cross-SDK; Decision rule 8)."""
    import datetime

    class Hostile:
        def __str__(self):
            raise RuntimeError("str explodes")

    # Must not raise.
    _canonical_json({"x": Hostile()})
    _canonical_json({"d": datetime.datetime(2020, 1, 1)})

    a = _canonical_json({"d": datetime.datetime(2020, 1, 1)})
    b = _canonical_json({"d": datetime.datetime(2020, 1, 1)})
    assert a == b
    assert isinstance(a, str)


def test_big_int_residual_within_sdk_only():
    """Assertion 23 — >2^53 int handled + deterministic within-SDK; DELIBERATELY
    not asserted cross-SDK (Decision rule 5b: JS lost precision at JSON.parse)."""
    big = 1234567890123456789
    out1 = _canonical_json({"big": big})
    out2 = _canonical_json({"big": big})
    assert out1 == out2
    # Python keeps the exact integer; Node would emit the float64-rounded value.
    assert out1 == '{"big":1234567890123456789}'


def test_text_entry_length_is_code_points():
    """Assertion 13 — astral char counts 1 code point."""
    comp = build_response_composition(
        "openai", {"choices": [{"message": {"content": "a\U0001F600b"}}]}
    )
    assert comp[0]["type"] == "text"
    assert comp[0]["length"] == 3

    comp_bmp = build_response_composition(
        "openai", {"choices": [{"message": {"content": "hello world"}}]}
    )
    assert comp_bmp[0]["length"] == 11


def test_anchor_tool_use_routed_through_canonical():
    """Assertions 10, 14, 16 — anchor Anthropic tool_use input + empty + plain text."""
    comp = build_response_composition(
        "anthropic",
        {"content": [{"type": "tool_use", "name": "calc", "input": {"b": 2, "a": 1}}]},
    )
    tc = next(e for e in comp if e["type"] == "tool_call")
    assert tc["hash"] == CANON_HASH_AB
    assert tc["length"] == 13

    comp_empty = build_response_composition(
        "anthropic",
        {"content": [{"type": "tool_use", "name": "noargs", "input": {}}]},
    )
    tce = next(e for e in comp_empty if e["type"] == "tool_call")
    assert tce["length"] == 2
    assert tce["hash"] == _sha116("{}")

    comp_text = build_response_composition(
        "openai", {"choices": [{"message": {"content": "hello"}}]}
    )
    assert comp_text[0]["hash"] == _sha116("hello")
