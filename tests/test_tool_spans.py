"""
Tool-span capture tests — the manual tp.tool / tp.tool_span API, the
cross-framework span discriminators, and the fail-open guarantees.

Mirrors token-police-node/tests/toolSpans.test.ts.
"""
import pytest

import token_police as tp
from token_police import state
from token_police.telemetry import _is_tool_span, _tool_name_from_span, _hash_len


class _FakeToolClient:
    """Captures log_sync(**kwargs) calls for assertions."""
    def __init__(self):
        self.logged = []

    def log_sync(self, **kwargs):
        self.logged.append(kwargs)


# ── Cross-framework discriminator ────────────────────────────────────

def test_is_tool_span_discriminators():
    assert _is_tool_span({"gen_ai.operation.name": "execute_tool"}, "")
    assert _is_tool_span({"traceloop.span.kind": "tool"}, "")
    assert _is_tool_span({"openinference.span.kind": "TOOL"}, "")
    assert _is_tool_span({}, "execute_tool web_search")
    assert _is_tool_span({}, "web_search.tool")
    # An ordinary LLM span must NOT be mistaken for a tool span.
    assert not _is_tool_span({}, "gpt-4o.chat")
    assert not _is_tool_span({"gen_ai.request.model": "gpt-4o"}, "ChatOpenAI.chat")


def test_tool_name_from_span_reads_all_keys():
    assert _tool_name_from_span({"gen_ai.tool.name": "ws"}, "x") == "ws"
    assert _tool_name_from_span({"traceloop.entity.name": "db"}, "x") == "db"
    assert _tool_name_from_span({"tool.name": "calc"}, "x") == "calc"
    assert _tool_name_from_span({}, "execute_tool web_search") == "web_search"
    assert _tool_name_from_span({}, "db_query.tool") == "db_query"


def test_hash_len_is_metadata_only():
    h, n = _hash_len("hello")
    assert n == 5
    assert len(h) == 16          # sha1 truncated to 16 hex chars
    assert h != "hello"          # never the raw content
    assert _hash_len(None) == ("", 0)
    assert _hash_len("") == ("", 0)


# ── Manual API ───────────────────────────────────────────────────────

def test_tool_span_emits_tool_row_metadata_only(monkeypatch):
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)
    with tp.tool_span("web_search", call_id="call_1", args="my secret query") as h:
        h["result"] = "secret result text"
    assert len(fake.logged) == 1
    call = fake.logged[0]
    assert call["span"]["span_kind"] == "tool"
    assert call["span"]["span_name"] == "web_search"
    assert call["tool"]["name"] == "web_search"
    assert call["tool"]["call_id"] == "call_1"
    assert call["tool"]["param_length"] == len("my secret query")
    assert call["tool"]["result_length"] == len("secret result text")
    assert call["call_outcome"]["status"] == "success"
    # Privacy: only hashes + lengths leave — never raw content.
    blob = str(call)
    assert "my secret query" not in blob
    assert "secret result text" not in blob


def test_tool_decorator_records_failure_and_reraises(monkeypatch):
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)

    @tp.tool(name="boom")
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()
    assert len(fake.logged) == 1
    assert fake.logged[0]["call_outcome"]["status"] == "failed"
    assert fake.logged[0]["call_outcome"]["error_kind"] == "ValueError"


def test_tool_decorator_records_success(monkeypatch):
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)

    @tp.tool()
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    assert len(fake.logged) == 1
    assert fake.logged[0]["span"]["span_name"] == "add"
    assert fake.logged[0]["call_outcome"]["status"] == "success"


# ── Bare @tp.tool (no parentheses) ───────────────────────────────────

def test_bare_tool_decorator_records_success(monkeypatch):
    """Bare @tp.tool must wrap the function, not rebind it to `decorator`."""
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)

    @tp.tool
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    # The JSON tool-dispatch pattern fn(**args) — pre-fix this raised a
    # TypeError out of the SDK's own `decorator` closure.
    assert add(a=2, b=3) == 5
    assert len(fake.logged) == 2
    assert fake.logged[0]["span"]["span_name"] == "add"
    assert fake.logged[0]["tool"]["name"] == "add"
    assert fake.logged[0]["call_outcome"]["status"] == "success"


def test_bare_tool_decorator_async(monkeypatch):
    import asyncio

    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)

    @tp.tool
    async def fetch(x):
        return x * 2

    assert asyncio.run(fetch(21)) == 42
    assert len(fake.logged) == 1
    assert fake.logged[0]["span"]["span_name"] == "fetch"
    assert fake.logged[0]["call_outcome"]["status"] == "success"


def test_bare_tool_decorator_preserves_name(monkeypatch):
    monkeypatch.setattr(state, "get_client", lambda: None)

    @tp.tool
    def web_search(query):
        return query

    assert web_search.__name__ == "web_search"


def test_bare_tool_decorator_propagates_user_error_unchanged(monkeypatch):
    monkeypatch.setattr(state, "get_client", lambda: None)

    @tp.tool
    def explode():
        raise KeyError("user-error")

    with pytest.raises(KeyError):
        explode()


# ── Fail-open ────────────────────────────────────────────────────────

def test_tool_span_swallows_emit_errors(monkeypatch):
    """A client whose log_sync throws must not surface into customer code."""
    class Broken:
        def log_sync(self, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(state, "get_client", lambda: Broken())
    with tp.tool_span("t", args="x") as h:   # must not raise
        h["result"] = "y"


def test_tool_span_never_raises_without_client(monkeypatch):
    monkeypatch.setattr(state, "get_client", lambda: None)
    with tp.tool_span("t", args="x") as h:   # must not raise
        h["result"] = "y"


def test_tool_decorator_propagates_user_error_unchanged(monkeypatch):
    """A throwing tool still raises the ORIGINAL error type."""
    monkeypatch.setattr(state, "get_client", lambda: None)

    @tp.tool()
    def explode():
        raise KeyError("user-error")

    with pytest.raises(KeyError):
        explode()


# ── OTel tool pending fallback (I7 PR-C) ─────────────────────────────

class _FakeOTelToolSpan:
    def __init__(self, attrs, name="tool"):
        self.attributes = attrs
        self.name = name
        self.start_time = 1_000_000_000  # ns
        self.end_time = 2_000_000_000
        self.context = None
        self.parent = None
        self.status = None


def test_log_tool_span_pops_pending_when_otel_id_empty(monkeypatch):
    """I7: frameworks rarely set gen_ai.tool.call.id — fall back to pending stash."""
    from token_police import context as ctx
    from token_police.telemetry import TokenPoliceSpanProcessor

    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)
    ctx.set_pending_tool_calls([
        {"id": "call_otel_1", "name": "web_search"},
        {"id": "call_other", "name": "other"},
    ])
    proc = TokenPoliceSpanProcessor()
    span = _FakeOTelToolSpan({
        "gen_ai.tool.name": "web_search",
        "gen_ai.tool.type": "function",
        # no gen_ai.tool.call.id
    }, name="web_search.tool")
    proc._log_tool_span(span, span.attributes, span.name)
    assert len(fake.logged) == 1
    assert fake.logged[0]["tool"]["call_id"] == "call_otel_1"
    assert ctx._pop_pending_tool_call_id("other") == "call_other"


def test_log_tool_span_prefers_otel_attr_over_pending(monkeypatch):
    from token_police import context as ctx
    from token_police.telemetry import TokenPoliceSpanProcessor

    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)
    ctx.set_pending_tool_calls([{"id": "stashed", "name": "web_search"}])
    proc = TokenPoliceSpanProcessor()
    span = _FakeOTelToolSpan({
        "gen_ai.tool.name": "web_search",
        "gen_ai.tool.call.id": "from_otel",
    })
    proc._log_tool_span(span, span.attributes, span.name)
    assert fake.logged[0]["tool"]["call_id"] == "from_otel"
    # Pending not consumed when attr present.
    assert ctx._pop_pending_tool_call_id("web_search") == "stashed"
