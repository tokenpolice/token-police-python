"""
Auto-correlation of LLM tool-call ids to manual tool-execution spans.

When an app uses @tp.tool / tp.tool_span without passing a call_id, the SDK
stashes the (id, name) pairs from the preceding LLM response (in the enforcer's
_capture_response_composition) and the tool-span path pops the first name-match
to attach the id — so tool_call_id is populated with no app code changes.

Mirrors token-police-node/tests/toolSpans.test.ts additions.
"""
from types import SimpleNamespace as NS

import token_police as tp
from token_police import state, context
from token_police.composition import extract_pending_tool_calls


class _FakeToolClient:
    def __init__(self):
        self.logged = []

    def log_sync(self, **kwargs):
        self.logged.append(kwargs)


# ── Extractor: per-provider shapes ───────────────────────────────────

def test_extract_openai_chat_dict():
    resp = {"choices": [{"message": {"content": None, "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "get_customer_info", "arguments": "{}"}},
        {"id": "call_2", "type": "function",
         "function": {"name": "search_kb", "arguments": "{}"}},
    ]}}]}
    assert extract_pending_tool_calls("openai", resp) == [
        {"id": "call_1", "name": "get_customer_info"},
        {"id": "call_2", "name": "search_kb"},
    ]


def test_extract_anthropic_object():
    resp = NS(content=[
        NS(type="text", text="hi"),
        NS(type="tool_use", id="toolu_9", name="escalate", input={}),
    ])
    assert extract_pending_tool_calls("anthropic", resp) == [
        {"id": "toolu_9", "name": "escalate"},
    ]


def test_extract_cohere_v2():
    resp = NS(message=NS(tool_calls=[
        NS(id="cohere_x", type="function", function=NS(name="lookup", arguments="{}")),
    ]))
    assert extract_pending_tool_calls("cohere", resp) == [
        {"id": "cohere_x", "name": "lookup"},
    ]


# ── LlamaIndex ChatResponse shapes ──────────────────────────────

def test_extract_llamaindex_blocks():
    """ToolCallBlock entries carry the id as `tool_call_id` / name as `tool_name`."""
    resp = NS(message=NS(blocks=[
        NS(block_type="text", text="on it"),
        NS(block_type="tool_call", tool_call_id="call_li_1",
           tool_name="get_customer_info", tool_kwargs={}),
    ]))
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "call_li_1", "name": "get_customer_info"},
    ]


def test_extract_llamaindex_additional_kwargs_openai_shape():
    resp = NS(message=NS(additional_kwargs={"tool_calls": [
        {"id": "call_li_2", "type": "function",
         "function": {"name": "search_kb", "arguments": "{}"}},
    ]}))
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "call_li_2", "name": "search_kb"},
    ]


def test_extract_llamaindex_additional_kwargs_anthropic_shape():
    # LlamaIndex's Anthropic LLM mirrors tool calls with a top-level `name`.
    resp = NS(message=NS(additional_kwargs={"tool_calls": [
        {"id": "toolu_li_3", "name": "escalate", "input": {}},
    ]}))
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "toolu_li_3", "name": "escalate"},
    ]


def test_extract_llamaindex_blocks_and_mirrored_kwargs_no_double_count():
    """Newer LlamaIndex mirrors block tool-calls into additional_kwargs — the
    same call must be stashed exactly once."""
    resp = NS(message=NS(
        blocks=[NS(block_type="tool_call", tool_call_id="call_li_4",
                   tool_name="lookup", tool_kwargs={})],
        additional_kwargs={"tool_calls": [
            {"id": "call_li_4", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
    ))
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "call_li_4", "name": "lookup"},
    ]


def test_extract_llamaindex_idless_block_falls_back_to_kwargs():
    """An id-less block contributes nothing (never invent) — but the mirrored
    additional_kwargs entry can still supply the real id."""
    resp = NS(message=NS(
        blocks=[NS(block_type="tool_call", tool_name="lookup", tool_kwargs={})],
        additional_kwargs={"tool_calls": [
            {"id": "call_li_5", "function": {"name": "lookup", "arguments": "{}"}},
        ]},
    ))
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "call_li_5", "name": "lookup"},
    ]


def test_extract_llamaindex_idless_everywhere_stays_empty():
    resp = NS(message=NS(blocks=[
        NS(block_type="tool_call", tool_name="lookup", tool_kwargs={}),
    ]))
    assert extract_pending_tool_calls("llamaindex", resp) == []


def test_extract_llamaindex_malformed_block_does_not_cost_the_turn():
    """One unserializable/hostile block must not lose the other ids.

    Blocks are classified BEFORE serialization, so a non-tool block that
    explodes on attribute access is skipped rather than aborting the walk.
    """
    class _Hostile:
        @property
        def block_type(self):
            raise RuntimeError("boom")

    resp = NS(message=NS(blocks=[
        _Hostile(),
        NS(block_type="tool_call", tool_call_id="call_li_safe",
           tool_name="lookup", tool_kwargs={}),
    ]))
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "call_li_safe", "name": "lookup"},
    ]


def test_extract_llamaindex_non_tool_blocks_are_not_serialized():
    """Image/document blocks must not be model_dump()'d on the hot path."""
    calls = []

    class _ImageBlock:
        block_type = "image"

        def model_dump(self):
            calls.append("dumped")
            return {"block_type": "image"}

    resp = NS(message=NS(blocks=[
        _ImageBlock(),
        NS(block_type="tool_call", tool_call_id="call_li_7",
           tool_name="ping", tool_kwargs={}),
    ]))
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "call_li_7", "name": "ping"},
    ]
    assert calls == []


def test_extract_llamaindex_dict_message():
    """model_dump()'d ChatMessage (dict blocks) resolves the same way."""
    resp = {"message": {"blocks": [
        {"block_type": "tool_call", "tool_call_id": "call_li_6", "tool_name": "ping"},
    ]}}
    assert extract_pending_tool_calls("llamaindex", resp) == [
        {"id": "call_li_6", "name": "ping"},
    ]


def test_extract_llamaindex_text_only_stays_empty():
    resp = NS(message=NS(blocks=[NS(block_type="text", text="hello")],
                         additional_kwargs={}))
    assert extract_pending_tool_calls("llamaindex", resp) == []


def test_extract_openai_responses_api():
    resp = {"output": [
        {"type": "reasoning"},
        {"type": "function_call", "call_id": "fc_7", "name": "do_thing", "arguments": "{}"},
    ]}
    assert extract_pending_tool_calls("openai_responses", resp) == [
        {"id": "fc_7", "name": "do_thing"},
    ]


def test_extract_gemini_omitted_id_stays_empty():
    # Google FunctionCall.id is optional — when omitted, nothing stashed (never invent).
    resp = NS(candidates=[NS(content=NS(parts=[
        NS(function_call=NS(name="foo", args={})),
    ]))])
    assert extract_pending_tool_calls("google", resp) == []


def test_extract_gemini_id_when_present():
    # Live/compositional flows (and some generateContent paths) populate id.
    resp = NS(candidates=[NS(content=NS(parts=[
        NS(function_call=NS(id="fc_gem_1", name="lookup", args={"q": "x"})),
        NS(functionCall=NS(id="fc_gem_2", name="search", args={})),  # camelCase
    ]))])
    assert extract_pending_tool_calls("google", resp) == [
        {"id": "fc_gem_1", "name": "lookup"},
        {"id": "fc_gem_2", "name": "search"},
    ]


def test_extract_gemini_dict_candidates():
    resp = {"candidates": [{"content": {"parts": [
        {"functionCall": {"id": "g1", "name": "foo", "args": {}}},
        {"function_call": {"name": "no_id", "args": {}}},  # omitted id → skip
    ]}}]}
    assert extract_pending_tool_calls("google", resp) == [
        {"id": "g1", "name": "foo"},
    ]


def test_extract_langchain_aimessage_shape():
    # AIMessage.tool_calls: [{id, name, args}] — name top-level, not under .function.
    resp = NS(tool_calls=[
        {"id": "call_lc_1", "name": "get_customer_info", "args": {"id": 1}},
        {"id": "call_lc_2", "name": "search_kb", "args": {}},
    ])
    assert extract_pending_tool_calls("langchain", resp) == [
        {"id": "call_lc_1", "name": "get_customer_info"},
        {"id": "call_lc_2", "name": "search_kb"},
    ]


def test_extract_langchain_llmresult_generations():
    # generate/agenerate returns LLMResult with generations[][].message.
    msg = NS(tool_calls=[{"id": "call_gen", "name": "lookup", "args": {}}])
    gen = NS(message=msg)
    resp = NS(generations=[[gen]])
    assert extract_pending_tool_calls("langchain", resp) == [
        {"id": "call_gen", "name": "lookup"},
    ]


def test_extract_xai_dict_tool_calls():
    # Dict-shaped top-level tool_calls (JSON / already-coerced).
    resp = {"tool_calls": [
        {"id": "xai_d1", "type": "function",
         "function": {"name": "get_customer_info", "arguments": "{}"}},
    ]}
    assert extract_pending_tool_calls("xai", resp) == [
        {"id": "xai_d1", "name": "get_customer_info"},
    ]


def test_extract_xai_protobuf_like_tool_calls():
    """xai-sdk returns protobuf ToolCall messages — no model_dump/dict/to_dict.

    A plain attr-only object (not SimpleNamespace) forces the getattr fallback
    in _tool_call_id_name; NS would go through _as_dict via __dict__.
    """
    class _Fn:
        def __init__(self, name, arguments="{}"):
            self.name = name
            self.arguments = arguments

    class _ProtoToolCall:
        # No model_dump / dict / to_dict — mirrors protobuf messages.
        def __init__(self, cid, name):
            self.id = cid
            self.function = _Fn(name)

    resp = NS(tool_calls=[
        _ProtoToolCall("xai_1", "get_customer_info"),
        _ProtoToolCall("xai_2", "search_kb"),
    ])
    assert extract_pending_tool_calls("xai", resp) == [
        {"id": "xai_1", "name": "get_customer_info"},
        {"id": "xai_2", "name": "search_kb"},
    ]


def test_extract_tool_call_top_level_name():
    # LangChain-ish dict: name at top level, not under .function.
    resp = NS(tool_calls=[{"id": "lc_1", "name": "lookup", "args": {}}])
    assert extract_pending_tool_calls("xai", resp) == [
        {"id": "lc_1", "name": "lookup"},
    ]


def test_extract_empty_id_not_invented():
    # Provider omitted id → nothing stashed (never invent).
    class _Fn:
        name = "lookup"
        arguments = "{}"

    class _ProtoNoId:
        id = ""
        function = _Fn()

    assert extract_pending_tool_calls("xai", NS(tool_calls=[_ProtoNoId()])) == []


def test_extract_malformed_returns_empty():
    assert extract_pending_tool_calls("x", object()) == []
    assert extract_pending_tool_calls("x", None) == []
    assert extract_pending_tool_calls("x", {"choices": "nonsense"}) == []


def test_extract_text_only_response_is_empty():
    assert extract_pending_tool_calls("openai", {"choices": [{"message": {"content": "hi"}}]}) == []


# ── Stash + FIFO match helpers ───────────────────────────────────────

def test_fifo_match_distinct_names():
    context.set_pending_tool_calls([
        {"id": "a", "name": "f1"}, {"id": "b", "name": "f2"},
    ])
    assert context._pop_pending_tool_call_id("f2") == "b"
    assert context._pop_pending_tool_call_id("f1") == "a"
    assert context._pop_pending_tool_call_id("f1") == ""  # drained


def test_fifo_match_same_name_in_order():
    context.set_pending_tool_calls([
        {"id": "x1", "name": "f"}, {"id": "x2", "name": "f"},
    ])
    assert context._pop_pending_tool_call_id("f") == "x1"
    assert context._pop_pending_tool_call_id("f") == "x2"


def test_replace_clears_prior_ids():
    context.set_pending_tool_calls([{"id": "old", "name": "f"}])
    context.set_pending_tool_calls([])  # no-tool response replaces/clears
    assert context._pop_pending_tool_call_id("f") == ""


def test_pop_no_pending_is_empty_and_safe():
    context.set_pending_tool_calls(None)
    assert context._pop_pending_tool_call_id("anything") == ""


# ── Integration: decorator auto-correlates without app passing call_id ─

def test_decorator_auto_correlates_call_id(monkeypatch):
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)

    # Simulate what the enforcer does after capturing the LLM response.
    context.set_pending_tool_calls(extract_pending_tool_calls("openai", {"choices": [{"message": {"tool_calls": [
        {"id": "call_42", "function": {"name": "get_customer_info", "arguments": "{}"}},
    ]}}]}))

    @tp.tool()
    def get_customer_info(email):
        return "ok"

    get_customer_info("a@b.com")
    assert fake.logged[0]["tool"]["call_id"] == "call_42"
    assert fake.logged[0]["tool"]["name"] == "get_customer_info"


def test_decorator_auto_correlates_xai_protobuf_like(monkeypatch):
    """I4: native xai-sdk protos → extract → @tp.tool gets call_id."""
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)

    class _Fn:
        name = "get_customer_info"
        arguments = "{}"

    class _ProtoToolCall:
        id = "xai_call_99"
        function = _Fn()

    context.set_pending_tool_calls(extract_pending_tool_calls(
        "xai", NS(tool_calls=[_ProtoToolCall()]),
    ))

    @tp.tool()
    def get_customer_info(email):
        return "ok"

    get_customer_info("a@b.com")
    assert fake.logged[0]["tool"]["call_id"] == "xai_call_99"
    assert fake.logged[0]["tool"]["name"] == "get_customer_info"


def test_explicit_call_id_wins_over_stash(monkeypatch):
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)
    context.set_pending_tool_calls([{"id": "stashed", "name": "web_search"}])
    with tp.tool_span("web_search", call_id="explicit", args="q") as h:
        h["result"] = "r"
    assert fake.logged[0]["tool"]["call_id"] == "explicit"


def test_no_match_leaves_call_id_empty(monkeypatch):
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)
    context.set_pending_tool_calls([{"id": "call_x", "name": "other_tool"}])
    with tp.tool_span("unmatched_tool", args="q") as h:
        h["result"] = "r"
    assert fake.logged[0]["tool"]["call_id"] == ""


def test_set_pending_never_raises_on_garbage():
    # Fail-open: feeding junk to the stash setter must not raise.
    context.set_pending_tool_calls("not-a-list")
    context.set_pending_tool_calls(object())


# ── concurrent pops under a bound session ──────────────────────

def test_concurrent_same_name_pops_distinct_ids():
    """Two asyncio tasks popping the same tool name must get distinct ids."""
    import asyncio
    session = context.TPSession()
    token = context._current_session.set(session)
    try:
        context.set_pending_tool_calls([
            {"id": "a", "name": "f"},
            {"id": "b", "name": "f"},
        ])

        async def _pop():
            await asyncio.sleep(0)
            return context._pop_pending_tool_call_id("f")

        async def _run():
            return await asyncio.gather(_pop(), _pop())

        # Private loop — do not asyncio.run (unsets main-thread loop).
        loop = asyncio.new_event_loop()
        try:
            ids = loop.run_until_complete(_run())
        finally:
            loop.close()
        assert set(ids) == {"a", "b"}, ids
    finally:
        context._current_session.reset(token)


# ── the LlamaIndex producers themselves, not a hand-seeded stash ─
#
# The consumer side was always correct; what was missing is that nothing ever
# filled the stash on the LlamaIndex paths. These drive the real capture
# functions so a future refactor that drops the producer call fails here.

def _bound_session():
    """Bind a fresh TPSession and return (session, reset_token)."""
    sess = context.TPSession()
    return sess, context._current_session.set(sess)


def _li_response(tool_call_id=None, name="get_customer_info"):
    blocks = [NS(block_type="text", text="working on it")]
    if tool_call_id:
        blocks.append(NS(block_type="tool_call", tool_call_id=tool_call_id,
                         tool_name=name, tool_kwargs={}))
    return NS(message=NS(role="assistant", blocks=blocks, additional_kwargs={}))


def test_capture_response_composition_stashes_llamaindex_ids():
    """Non-stream LlamaIndex path (li_sync / li_async tail)."""
    from token_police import enforcer
    sess, token = _bound_session()
    try:
        enforcer._capture_response_composition("llamaindex", _li_response("call_li_ns"))
        assert context._pop_pending_tool_call_id("get_customer_info") == "call_li_ns"
    finally:
        context._current_session.reset(token)


def test_capture_response_composition_at_stashes_llamaindex_ids():
    """The stream paths capture via _capture_response_composition_at,
    which had no producer at all."""
    from token_police import enforcer
    sess, token = _bound_session()
    try:
        enforcer._capture_response_composition_at(
            "llamaindex", _li_response("call_li_stream"), 0)
        assert context._pop_pending_tool_call_id("get_customer_info") == "call_li_stream"
    finally:
        context._current_session.reset(token)


def test_capture_response_composition_at_no_tool_clears_stale_ids():
    """REPLACE-on-capture: a no-tool turn must clear a prior turn's ids."""
    from token_police import enforcer
    sess, token = _bound_session()
    try:
        context.set_pending_tool_calls([{"id": "stale", "name": "get_customer_info"}])
        enforcer._capture_response_composition_at("llamaindex", _li_response(None), 0)
        assert context._pop_pending_tool_call_id("get_customer_info") == ""
    finally:
        context._current_session.reset(token)


def test_capture_response_composition_at_never_raises_on_garbage():
    """Golden rule: capture runs on the customer's hot path."""
    from token_police import enforcer
    sess, token = _bound_session()
    try:
        class _Exploding:
            @property
            def message(self):
                raise RuntimeError("boom")

        enforcer._capture_response_composition_at("llamaindex", _Exploding(), 0)
        enforcer._capture_response_composition_at("llamaindex", None, 0)
    finally:
        context._current_session.reset(token)


def test_llamaindex_stream_capture_end_to_end(monkeypatch):
    """Producer → stash → @tp.tool row carries the provider's id."""
    from token_police import enforcer
    fake = _FakeToolClient()
    monkeypatch.setattr(state, "get_client", lambda: fake)
    sess, token = _bound_session()
    try:
        enforcer._capture_response_composition_at(
            "llamaindex", _li_response("call_li_e2e"), 0)

        @tp.tool()
        def get_customer_info(email):
            return "ok"

        get_customer_info("a@b.com")
        assert fake.logged[0]["tool"]["call_id"] == "call_li_e2e"
    finally:
        context._current_session.reset(token)
