"""Fix 2 — _canon_encode_string fast path byte-identity.

_canon_encode_string now short-circuits strings that need no escaping (the
common case) instead of running the per-char loop. The canonical bytes feed
cross-SDK hashes, so the fast-path output MUST be byte-for-byte identical to the
original per-char algorithm for EVERY input. This test keeps a copy of the old
loop as an oracle and asserts equality over an adversarial corpus.
"""
import unittest

from token_police.composition import _CANON_ESCAPE, _canon_encode_string


def _old_encode(s):
    """Verbatim copy of the pre-fast-path algorithm — the oracle."""
    out = ['"']
    for ch in s:
        esc = _CANON_ESCAPE.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _corpus():
    cases = [
        "",                              # empty
        "hello world",                   # clean ascii
        "a" * 5000,                      # long clean ascii (fast path)
        "/",                             # forward slash NOT escaped
        "   sep",              # line/para separators NOT escaped (>= 0x20)
        "café résumé",    # latin-1 supplement, literal
        "日本語",            # CJK, literal
        "\U0001F600\U0001F4B8",          # astral (emoji), literal
        "mix: café \n tab\t end",   # mixed escape + literal
        '"quoted"',                      # double quote
        "back\\slash",                   # backslash
        "\b\t\n\f\r",                    # all named control escapes
    ]
    # Every char in the escape table on its own + embedded.
    for ch in _CANON_ESCAPE:
        cases.append(ch)
        cases.append("x" + ch + "y")
    # Every control char 0x00-0x1F on its own + embedded in clean text.
    for code in range(0x00, 0x20):
        c = chr(code)
        cases.append(c)
        cases.append("pre" + c + "post")
    # A single string containing every control char at once.
    cases.append("".join(chr(c) for c in range(0x00, 0x20)))
    return cases


class TestCanonEncodeFastPath(unittest.TestCase):
    def test_byte_identity_against_oracle(self):
        for s in _corpus():
            self.assertEqual(_canon_encode_string(s), _old_encode(s),
                             msg=f"mismatch for {s!r}")

    def test_fast_path_wraps_clean_string_verbatim(self):
        self.assertEqual(_canon_encode_string("plain text 123"), '"plain text 123"')

    def test_slow_path_escapes(self):
        self.assertEqual(_canon_encode_string('a"b'), '"a\\"b"')
        self.assertEqual(_canon_encode_string("a\x00b"), '"a\\u0000b"')


if __name__ == "__main__":
    unittest.main()
