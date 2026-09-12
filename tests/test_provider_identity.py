"""Locks the SDK side of provider-aware pricing: the enforcer must forward the
client's base URL (host+path, no query) so the server can identify the
serving provider. All mapping logic lives in the server; the SDK only
extracts the raw string."""

import pytest

from token_police.enforcer import _extract_base_url


class _FakeClient:
    def __init__(self, base_url):
        self.base_url = base_url


class _FakeBoundMethodSelf:
    """Mimics the resource instance (e.g. Messages) whose ._client carries base_url."""

    def __init__(self, base_url):
        self._client = _FakeClient(base_url)


def test_extracts_base_url_and_strips_query():
    args = (_FakeBoundMethodSelf("https://api.minimax.io/anthropic?key=secret"),)
    assert _extract_base_url(args) == "https://api.minimax.io/anthropic"


def test_strips_embedded_credentials():
    # Only host metadata may leave the process — userinfo + query are dropped.
    args = (_FakeBoundMethodSelf("https://user:secret@api.minimax.io/x?k=v"),)
    assert _extract_base_url(args) == "https://api.minimax.io/x"


def test_returns_empty_when_no_client():
    assert _extract_base_url((object(),)) == ""
    assert _extract_base_url(()) == ""
    assert _extract_base_url(None) == ""


def test_never_raises_on_garbage():
    # Fail-safe: any odd input degrades to "" rather than raising into the call.
    class Boom:
        @property
        def _client(self):
            raise RuntimeError("nope")

    assert _extract_base_url((Boom(),)) == ""


# ── Scheme-less / unparseable fallback must also strip credentials ──────────
# When the base URL has no scheme, urlsplit yields no hostname and we hit the
# best-effort fallback. That branch must STILL drop userinfo so a
# `user:pass@host` never reaches telemetry — but an '@' inside the path is not
# userinfo and must survive.
@pytest.mark.parametrize(
    "raw, expected",
    [
        # userinfo in the authority is stripped; query dropped
        ("user:pass@host.example/path?q=1", "host.example/path"),
        # plain scheme-less host+path — unchanged
        ("host.example/path", "host.example/path"),
        # userinfo with no path
        ("user@host.example", "host.example"),
        # '@' in the PATH (not authority) must survive
        ("host.example/a@b", "host.example/a@b"),
        # fragment dropped too
        ("user:pass@host.example/p#frag", "host.example/p"),
    ],
)
def test_schemeless_fallback_strips_userinfo(raw, expected):
    args = (_FakeBoundMethodSelf(raw),)
    assert _extract_base_url(args) == expected


def test_schemeful_urls_unchanged_by_userinfo_change():
    # Scheme-ful URLs go through the parseable branch, not the fallback —
    # behaviour is exactly as before: host+path, no userinfo, no query.
    args = (_FakeBoundMethodSelf("https://user:pass@api.host.io/v1?k=v"),)
    assert _extract_base_url(args) == "https://api.host.io/v1"


def test_schemeless_garbage_never_raises_and_never_leaks_secret():
    for junk in ["", "::::", "@@@@", "user:pass@", "/@x", "%%%"]:
        out = _extract_base_url((_FakeBoundMethodSelf(junk),))
        assert isinstance(out, str)
        assert "pass" not in out
