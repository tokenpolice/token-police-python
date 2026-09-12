"""§4-py-o — `_hash_len` consolidated to a single implementation.

The near-identical copies in telemetry.py and openai_agents.py were merged into
composition.py (beside the canonical serializer). These tests assert:

  * every module still exposes `_hash_len` and they are the SAME object;
  * outputs are byte-identical to the values the two former copies produced
    (pinned below), so the consolidation changed no observable behavior.
"""
from token_police.composition import _hash_len as canon_hash_len
from token_police.telemetry import _hash_len as telemetry_hash_len
from token_police.openai_agents import _hash_len as agents_hash_len


def test_single_shared_implementation():
    assert canon_hash_len is telemetry_hash_len
    assert canon_hash_len is agents_hash_len


# (input, expected (hash16, length)) pinned from the implementation both former
# copies shared — identical serializer + sha1-16 + code-point length.
_PINNED = [
    (None, ("", 0)),
    ("", ("", 0)),
    ("hello", ("aaf4c61ddcc5e8a2", 5)),
    ('{"customer_id": 42}', ("3c8d74fc071a702d", 19)),
    ({"customer_id": 42}, ("c4327f38f6e63195", 18)),
    ([1, 2, 3], ("9ef50cc82ae47427", 7)),
    (3.14, ("983b34771fb7185d", 4)),
    # Astral emoji counts as ONE code point (matches Node codePointLength).
    ("astral \U0001F600 x", ("107be2b6d61fec5c", 10)),
    # Object key order is canonicalized (sorted) before hashing.
    ({"b": 1, "a": 2}, ("1c072775cb3d4104", 13)),
]


def test_pinned_corpus_unchanged():
    for val, expected in _PINNED:
        assert canon_hash_len(val) == expected, val


def test_string_and_dict_agree_across_call_sites():
    for val, _ in _PINNED:
        assert telemetry_hash_len(val) == agents_hash_len(val) == canon_hash_len(val)
