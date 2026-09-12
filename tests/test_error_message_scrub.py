"""
Raw provider/tool exception text must not ship by default (Python parity).

On a FAILED call the raw error string used to be captured verbatim
(``str(...)[:500]``) into ``call_outcome["error_message"]`` and forwarded to the
server — a plaintext leak (provider 400s / tool errors echo prompt content).
The fix: a centralized, pure & TOTAL scrub helper + an ``error_detail`` config
("none" | "redacted" | "raw", default "redacted").

Mirrors token-police-node/tests/errorMessageScrub.test.ts.
"""
import hashlib

import pytest

import token_police as tp
from token_police import state
from token_police.client import TokenPolice
from token_police._classify import (
    scrub_error_message,
    resolve_error_detail,
    build_call_outcome,
)

API_KEY = "tp_sk_test123"

# SHA-256 hex of the exact string "boom" — a SHARED constant the Node test
# asserts against too (assertion 17: cross-SDK hash parity).
SHA256_BOOM = "81f52337ebb4cb1669bb802c708807dde0519d15cb102a6313d26ad5cd821713"


def sha256hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


@pytest.fixture
def client_factory():
    created = []

    def _make(**kwargs):
        c = TokenPolice(api_key=API_KEY, **kwargs)
        created.append(c)
        return c

    yield _make
    for c in created:
        c.close_sync()


class _FakeToolClient:
    """Captures log_sync(**kwargs) calls; carries an error_detail mode."""

    def __init__(self, error_detail="redacted"):
        self.logged = []
        self.error_detail = error_detail

    def log_sync(self, **kwargs):
        self.logged.append(kwargs)


# ── Config surface (assertions 2, 4) ────────────────────────────────────
def test_default_error_detail_is_redacted(client_factory):
    assert client_factory().error_detail == "redacted"


@pytest.mark.parametrize("bad", ["verbose", "Raw", "", "plaintext", "RAW", "none "])
def test_unknown_error_detail_falls_back_to_redacted(client_factory, bad):
    c = client_factory(error_detail=bad)
    assert c.error_detail == "redacted"


@pytest.mark.parametrize("good", ["none", "redacted", "raw"])
def test_valid_error_detail_passes_through(client_factory, good):
    assert client_factory(error_detail=good).error_detail == good


# ── Helper directly (pure function) ─────────────────────────────────────
def test_helper_none_returns_no_fields():
    assert scrub_error_message("boom", "none") == {}
    assert scrub_error_message(ValueError("boom"), "none") == {}


def test_helper_redacted_returns_hash_not_raw():
    out = scrub_error_message("boom", "redacted")
    assert out == {"error_message_hash": SHA256_BOOM}
    assert "error_message" not in out


def test_helper_redacted_hash_is_pre_truncation():
    # Assertion 7: hash over the FULL string, not the 500-char slice.
    long = "x" * 600
    out = scrub_error_message(long, "redacted")
    assert out["error_message_hash"] == sha256hex(long)
    assert out["error_message_hash"] != sha256hex(long[:500])


def test_helper_empty_raw_no_hash_no_message():
    # Assertion 8.
    assert scrub_error_message("", "redacted") == {}


def test_helper_raw_reproduces_legacy_500_char_string():
    # Assertion 10/11.
    assert scrub_error_message("boom", "raw") == {"error_message": "boom"}
    long = "y" * 600
    out = scrub_error_message(long, "raw")
    assert out["error_message"] == long[:500]
    assert len(out["error_message"]) == 500
    assert "error_message_hash" not in out
    assert "error_class" not in out


def test_helper_unknown_mode_defensively_coerced_to_redacted():
    assert scrub_error_message("boom", "verbose") == {"error_message_hash": SHA256_BOOM}


# ── Pathological input (assertion 16, Golden Rule) ──────────────────────
class _HostileError(Exception):
    def __str__(self):
        raise RuntimeError("__str__ explodes")

    def __repr__(self):
        raise RuntimeError("__repr__ explodes")


def test_pathological_error_never_raises():
    hostile = _HostileError()
    for mode in ("none", "redacted", "raw"):
        # Fed DIRECTLY into the SAME shared helper the string-only sites use.
        out = scrub_error_message(hostile, mode)
        assert isinstance(out, dict)
        # No leak, no hostile object.
        assert out.get("error_message") is not hostile
    # And via the central builder (which also routes through the helper).
    build_call_outcome(hostile, 10)  # must not raise


# ── Central builder shapes ──────────────────────────────────────────────
def _set_client(monkeypatch, mode):
    monkeypatch.setattr(state, "get_client", lambda: _FakeToolClient(error_detail=mode))


def test_redacted_central_builder(monkeypatch):
    # Assertion 6.
    _set_client(monkeypatch, "redacted")
    out = build_call_outcome(ValueError("boom"), 10)
    assert "error_message" not in out
    assert out["error_message_hash"] == SHA256_BOOM
    assert out["error_class"] == "ValueError"
    assert "error_kind" in out
    assert "http_status" in out
    assert out["status"] == "failed"


def test_none_central_builder(monkeypatch):
    # Assertion 9.
    _set_client(monkeypatch, "none")
    out = build_call_outcome(ValueError("boom"), 10)
    assert "error_message" not in out
    assert "error_message_hash" not in out
    assert "error_class" not in out
    assert "error_kind" in out
    assert "http_status" in out


def test_raw_central_builder(monkeypatch):
    # Assertion 11.
    _set_client(monkeypatch, "raw")
    out = build_call_outcome(ValueError("boom"), 10)
    assert out["error_message"] == str(ValueError("boom"))[:500]
    assert "error_message_hash" not in out
    assert "error_class" not in out
    out2 = build_call_outcome(ValueError("z" * 600), 10)
    assert out2["error_message"] == ("z" * 600)[:500]


# ── Success path untouched (assertion 19) ───────────────────────────────
@pytest.mark.parametrize("mode", ["none", "redacted", "raw"])
def test_success_path_unchanged(monkeypatch, mode):
    _set_client(monkeypatch, mode)
    assert build_call_outcome(None, 10) == {"status": "success", "duration_ms": 10}


# ── error_kind / http_status preserved in every mode (assertion 20) ─────
@pytest.mark.parametrize("mode", ["none", "redacted", "raw"])
def test_error_kind_http_status_preserved(monkeypatch, mode):
    _set_client(monkeypatch, mode)

    class _RateLimit(Exception):
        status_code = 429

    out = build_call_outcome(_RateLimit("rate limited"), 5)
    assert out["error_kind"] == "rate_limited"
    assert out["http_status"] == 429


# ── resolve_error_detail single resolution point (assertion 24) ─────────
def test_resolve_error_detail_fallback(monkeypatch):
    monkeypatch.setattr(state, "get_client", lambda: None)
    assert resolve_error_detail() == "redacted"

    def _boom():
        raise RuntimeError("no client")

    monkeypatch.setattr(state, "get_client", _boom)
    assert resolve_error_detail() == "redacted"

    monkeypatch.setattr(state, "get_client", lambda: _FakeToolClient(error_detail="raw"))
    assert resolve_error_detail() == "raw"


# ── Behavioral: string-only emit site (context.py tool path) routed ─────
def test_context_tool_span_failure_is_scrubbed(monkeypatch):
    # Assertion 13.
    fake = _FakeToolClient(error_detail="redacted")
    monkeypatch.setattr(state, "get_client", lambda: fake)
    with pytest.raises(ValueError):
        with tp.tool_span("boom"):
            raise ValueError("kaboom-secret")
    co = fake.logged[0]["call_outcome"]
    assert co["status"] == "failed"
    assert "error_message" not in co
    assert co["error_message_hash"] == sha256hex("kaboom-secret")
    # Raw error text never appears in the logged payload.
    assert "kaboom-secret" not in repr(fake.logged[0])


def test_context_tool_span_failure_raw_mode_restores_message(monkeypatch):
    fake = _FakeToolClient(error_detail="raw")
    monkeypatch.setattr(state, "get_client", lambda: fake)
    with pytest.raises(ValueError):
        with tp.tool_span("boom"):
            raise ValueError("kaboom-raw")
    co = fake.logged[0]["call_outcome"]
    assert co["error_message"] == "kaboom-raw"
    assert "error_message_hash" not in co
