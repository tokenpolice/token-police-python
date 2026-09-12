"""B1 (Python): two independent LlamaIndex bugs fixed together.

(b) ``li_stream`` called the provider BEFORE setting the ``_in_llamaindex``
guard. Anthropic's LlamaIndex ``stream_chat`` dispatches its HTTP request
EAGERLY (before returning the generator), so the inner patched Anthropic
wrapper ran completely unguarded during that window — it applied its own
reroute (the one the framework guard promises to suppress) AND emitted its
own /log row, so every streamed LlamaIndex call landed TWICE at two
different models. Fix: the guard is now held across the eager dispatch
(``_in_llamaindex.set(True)`` before ``_li_call_original``, reset in a
``finally`` before ``li_stream`` returns control) — see tests A/B below.

(a) ``OpenAIResponses`` — a SIBLING of LlamaIndex's ``OpenAI`` class (not a
subclass: it derives from ``FunctionCallingLLM`` and defines its own
chat/achat/stream_chat/astream_chat against ``client.responses.create``, a
MANUAL-wrapper provider with NO OpenLLMetry span) had no registry entries, so
it ran completely unguarded and got rerouted despite the framework guard
contract. Fix: 4 new ``_TARGET_METHODS`` entries, a
``_li_provider_from_instance`` mapping to the ``openai_responses`` pseudo-
provider, an ``_extract_li_usage_py`` branch, a
``_li_openai_responses_verbatim_usage_py`` verbatim-usage builder, and an
``_log_li_py`` dispatch branch — see tests C/D/E/H/I below.

Zero-row hazard (closed alongside (a)): OpenAIResponses' inner call emits no
span at all, so the ordinary flush-on-success path would be a no-op — EVERY
call would land ZERO rows. The non-stream li_sync/li_async tails now RESERVE
a span order, capture composition keyed at it, drain this call's
observations + pop `session._local_decision`, and manually emit via
``_log_li_py`` instead of flushing — see test F. Non-responses providers must
keep the flush path byte-identical (framework layer must NOT touch the span
counter or the pending span name for them) — see test G.

All fakes — no llama_index installed. Fixtures mirror
tests/test_eager_failure_outcomes.py and tests/test_stream_failure_outcomes.py;
extractor unit tests (H) mirror tests/test_llamaindex_verbatim_usage.py; the
wire byte-parity control (F2) mirrors tests/test_log_span_id_synthesis.py.
"""
from __future__ import annotations

import asyncio
import sys
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

# Make the local (worktree) SDK importable ahead of any editable install.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police import state as _state
from token_police.context import TPSession, _current_session, _in_llamaindex
from token_police.enforcer import _extract_li_usage_py


# ═══════════════════════════════════════════════════════════════════
# Shared fixtures / helpers (mirror tests/test_eager_failure_outcomes.py)
# ═══════════════════════════════════════════════════════════════════


class _FakeClient:
    """Captures every log_sync payload as raw kwargs (the shape _log_li_py
    forwards, BEFORE the real client's wire-payload building / truthy gate)."""

    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    """Pin one session, neutralise both pre-flight checks, intercept every
    logged payload, and keep the global observations pool clean across tests
    (mirrors tests/test_bedrock_failure_single_emitter.py's setUp/tearDown)."""
    _state.drain_observations()
    sess = TPSession(trace_id="a" * 32, root_span_id="b" * 16)
    sess._deferred_spans = []
    sess._defer_telemetry = False
    sess._call_outcome = None
    sess._local_decisions = []
    tok = _current_session.set(sess)
    li_guard = _in_llamaindex.set(False)

    client = _FakeClient()
    monkeypatch.setattr(enforcer, "_run_sync_check", lambda *a, **k: None)

    async def _noop_async_check(*a, **k):
        return None

    monkeypatch.setattr(enforcer, "_run_async_check", _noop_async_check)
    monkeypatch.setattr(enforcer, "get_client", lambda: client)
    try:
        yield sess, client
    finally:
        try:
            _in_llamaindex.reset(li_guard)
        except Exception:
            _in_llamaindex.set(False)
        _current_session.reset(tok)
        _state.drain_observations()


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _only_call(client):
    assert len(client.calls) == 1, f"expected exactly one log, got {len(client.calls)}"
    return client.calls[0]


def _seam(method):
    """The enforcer function object actually installed on the class.
    ``functools.wraps`` copies ``__name__``/``__qualname__`` from the
    ORIGINAL, so only ``__code__.co_name`` still names the wrapper. Lets a
    test prove which of li_sync/li_async/li_stream it exercised — same trick
    tests/test_eager_failure_outcomes.py uses."""
    return getattr(getattr(method, "__code__", None), "co_name", "")


class _Auth401(Exception):
    """OpenAI/Anthropic-SDK-shaped 401 — `.status_code` is what the
    classifier reads first (see token_police/_classify.py::_direct_status_of)."""

    status_code = 401


def _auth401():
    return _Auth401("401 Incorrect API key provided")


def _assert_failed_401(outcome):
    assert outcome is not None, "no row to classify"
    assert outcome["status"] == "failed"
    assert outcome["error_kind"] == "auth_error"
    assert outcome["http_status"] == 401


class _Chunk:
    """LlamaIndex ChatResponse-shaped stream chunk: `.delta` str + `.message`."""

    def __init__(self, delta="hi", additional_kwargs=None):
        self.delta = delta
        self.message = NS(additional_kwargs=additional_kwargs or {})
        self.raw = None


def _install_li_stream(stream_chat_fn, model="gpt-4o-mini"):
    """Non-responses provider (class name carries no anthropic/google/responses
    needle) — the SAME li_stream wrapper the real Anthropic bug went through;
    the eager-guard fix is provider-agnostic."""
    cls = type("FakeLIOpenAI", (), {"stream_chat": stream_chat_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")
    return cls


# ═══════════════════════════════════════════════════════════════════
# A. Guard held during the EAGER dispatch, released before li_stream returns,
#    re-taken for the duration of chunk consumption.
# ═══════════════════════════════════════════════════════════════════


def test_a_guard_held_across_eager_dispatch_and_during_iteration(_env):
    """B1 (root cause b): li_stream must hold _in_llamaindex True across the
    EAGER provider call (mirrors Anthropic's eager stream_chat HTTP dispatch),
    release it once the (synchronous) dispatch returns — BEFORE the outer
    generator is even constructed — then re-take it for every pull during
    consumption (inside _guard_sync_stream_li) and release it again once
    iteration completes."""
    _sess, _client = _env
    eager_guard_state = []
    iter_guard_states = []

    class _ProbeStream:
        """Records the guard's value on every pull — mirrors the REAL
        Anthropic path, where the inner patched provider wrapper runs
        (unguarded, pre-fix) exactly when the wrapper's return value is
        iterated/consumed."""

        def __init__(self):
            self._items = [_Chunk("a"), _Chunk("b")]

        def __iter__(self):
            return self

        def __next__(self):
            iter_guard_states.append(enforcer.in_llamaindex())
            if not self._items:
                raise StopIteration
            return self._items.pop(0)

    def stream_chat(self, messages="hi", *a, **k):
        # A plain (non-generator) function: this body runs to completion
        # SYNCHRONOUSLY at call time — the eager-dispatch window.
        eager_guard_state.append(enforcer.in_llamaindex())
        return _ProbeStream()

    cls = _install_li_stream(stream_chat)
    gen = cls().stream_chat("hi")

    assert eager_guard_state == [True], \
        "pre-fix: the provider dispatched with the guard unset"
    assert enforcer.in_llamaindex() is False, \
        "guard must be released once li_stream regains control (before the " \
        "outer generator is even pulled)"

    out = list(gen)
    assert [c.delta for c in out] == ["a", "b"]
    assert iter_guard_states == [True, True, True], \
        "guard must be held for every pull during consumption"
    assert enforcer.in_llamaindex() is False, \
        "guard released again once iteration completes"


# ═══════════════════════════════════════════════════════════════════
# B. Eager-dispatch failure path is unchanged: a synchronous raise still
#    lands exactly ONE failed row, guard released, identity re-raise.
# ═══════════════════════════════════════════════════════════════════


def test_b_eager_dispatch_failure_still_single_emitter_and_guard_released(_env):
    """B1 regression control: L-2's existing eager-construction-failure path
    must still hold under the new guard-across-dispatch structure — exactly
    one manual failed row, the guard fully released, and the customer's
    ORIGINAL exception re-raised by identity."""
    sess, client = _env
    boom = _auth401()
    order = sess._span_counter

    def stream_chat(self, messages="hi", *a, **k):
        # Raises DURING the eager window, before any stream object exists.
        raise boom

    cls = _install_li_stream(stream_chat)
    assert _seam(cls.stream_chat) == "li_stream"

    with pytest.raises(_Auth401) as ei:
        cls().stream_chat("hi")
    assert ei.value is boom  # identity re-raise (golden rule)

    assert enforcer.in_llamaindex() is False, \
        "guard must be released even on a construction-time raise"
    payload = _only_call(client)
    assert payload.get("tp_tag") is None, "the manual row, not a flushed inner span"
    _assert_failed_401(payload.get("call_outcome"))
    assert payload["model"] == "gpt-4o-mini"
    assert payload["span"]["span_order"] == order
    assert sess._defer_telemetry is False
    assert sess._deferred_spans == []


# ═══════════════════════════════════════════════════════════════════
# C. OpenAIResponses registration — sibling of OpenAI, own 4 entries.
# ═══════════════════════════════════════════════════════════════════


def test_c1_openai_responses_all_four_methods_registered_and_wrapped():
    """B1 (root cause a): OpenAIResponses must get its own registry entries
    for chat/achat/stream_chat/astream_chat, wired through the SAME
    _set_llamaindex_wrapper kinds as OpenAI. Without them the inner
    client.responses.create ran completely unguarded."""
    targets = [t for t in enforcer._TARGET_METHODS
               if t.get("module") == "llama_index.llms.openai"
               and t.get("object") == "OpenAIResponses"]
    assert len(targets) == 4
    assert {t["method"] for t in targets} == {
        "chat", "achat", "stream_chat", "astream_chat"
    }

    expected_seam = {
        "chat": "li_sync", "achat": "li_async",
        "stream_chat": "li_stream", "astream_chat": "li_async",
    }

    class OpenAIResponses:
        model = "gpt-5-responses"

        def chat(self, messages, **k):
            return None

        async def achat(self, messages, **k):
            return None

        def stream_chat(self, messages, **k):
            return iter([])

        async def astream_chat(self, messages, **k):
            async def _gen():
                return
                yield  # pragma: no cover — makes this an async generator

            return _gen()

    originals = {m: getattr(OpenAIResponses, m) for m in expected_seam}

    # override_module skips importlib entirely — a fresh container carrying
    # the class under its registered name, same idiom as
    # tests/test_bedrock_failure_single_emitter.py's FakeBotoClient.
    container = types.ModuleType("fake_li_openai_module_c1")
    container.OpenAIResponses = OpenAIResponses
    for t in targets:
        enforcer._wrap_method(dict(t), override_module=container)

    for method, seam in expected_seam.items():
        wrapped = getattr(OpenAIResponses, method)
        assert wrapped is not originals[method], f"{method} was not wrapped at all"
        assert _seam(wrapped) == seam, \
            f"{method} wrapped with the wrong kind ({_seam(wrapped)!r} != {seam!r})"


def test_c2_missing_openai_responses_class_degrades_silently():
    """A llama-index-llms-openai version without OpenAIResponses (pre-
    Responses-API release) must not raise into the customer's protect() call
    — _wrap_method's 3-arg getattr resolves to None and returns early."""
    container = types.ModuleType("fake_li_openai_module_c2")

    class OpenAI:  # only the base class present, no OpenAIResponses
        model = "gpt-4o"

        def chat(self, messages, **k):
            return None

    container.OpenAI = OpenAI
    target = next(t for t in enforcer._TARGET_METHODS
                  if t.get("module") == "llama_index.llms.openai"
                  and t.get("object") == "OpenAIResponses"
                  and t.get("method") == "chat")

    enforcer._wrap_method(dict(target), override_module=container)  # must not raise
    assert not hasattr(container, "OpenAIResponses")


# ═══════════════════════════════════════════════════════════════════
# D/E. OpenAIResponses non-stream success (sync + async) — the manual
#      zero-row-hazard emission path.
# ═══════════════════════════════════════════════════════════════════


class _RespUsage:
    """ResponseUsage-shaped: input_tokens is cache-INCLUSIVE."""

    def __init__(self, input_tokens=1000, output_tokens=200, cached_tokens=400):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.input_tokens_details = NS(cached_tokens=cached_tokens)


def _responses_chat_response(usage=None, model="gpt-5-responses"):
    """A LlamaIndex ChatResponse from OpenAIResponses.chat/achat: the verbatim
    ResponseUsage object lives on the ChatResponse's OWN additional_kwargs on
    every non-stream path (per the fix's docstring)."""
    usage = usage or _RespUsage()
    return NS(
        additional_kwargs={"usage": usage},
        raw=NS(usage=usage, model=model),
        message=NS(role="assistant", content="ok", additional_kwargs={}),
    )


def _install_li_responses_sync(chat_fn, model="gpt-5-responses"):
    cls = type("FakeLIOpenAIResponses", (), {"chat": chat_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "chat", cls.chat, "sync")
    return cls


def _install_li_responses_async(achat_fn, model="gpt-5-responses"):
    cls = type("FakeLIOpenAIResponses", (), {"achat": achat_fn, "model": model})
    enforcer._set_llamaindex_wrapper(cls, "achat", cls.achat, "async")
    return cls


def test_d_openai_responses_chat_nonstream_success_single_row(_env, monkeypatch):
    """B1 (root cause a + zero-row hazard): OpenAIResponses.chat has no inner
    OpenLLMetry span, so the ordinary flush would no-op — exactly ONE manual
    row must land, with cache-netted usage, the verbatim usage block, and a
    RESERVED (never-reused) span order across sequential calls."""
    sess, client = _env
    monkeypatch.setattr(enforcer, "consume_pending_span_name", lambda: "my_span")

    def chat(self, messages="hi", *a, **k):
        return _responses_chat_response()

    cls = _install_li_responses_sync(chat)
    assert _seam(cls.chat) == "li_sync"
    assert sess._span_counter == 0

    cls().chat("hi")
    payload = _only_call(client)
    assert payload["provider"] == "openai_responses"
    assert payload["model"] == "gpt-5-responses"
    assert payload["input_tokens"] == 600     # max(0, 1000 - 400)
    assert payload["output_tokens"] == 200
    assert payload["cached_tokens"] == 400
    assert payload["usage"]["shape"] == "openai_responses"
    assert payload["span"]["span_name"] == "my_span"
    assert payload["span"]["span_order"] == 0
    assert payload["prompt_composition"], "prompt composition must be captured"
    assert payload["response_composition"], "response composition must be captured"
    assert sess._deferred_spans == [], "this call's own reserved-order span is dropped"

    # A second, sequential call must RESERVE the next order — never replay 0.
    cls().chat("hi")
    assert len(client.calls) == 2
    assert client.calls[1]["span"]["span_order"] == 1


def test_e_openai_responses_achat_nonstream_success_single_row(_env, monkeypatch):
    """Async twin of test D — li_async's OpenAIResponses branch."""
    sess, client = _env
    monkeypatch.setattr(enforcer, "consume_pending_span_name", lambda: "my_span")

    async def achat(self, messages="hi", *a, **k):
        return _responses_chat_response()

    cls = _install_li_responses_async(achat)
    assert _seam(cls.achat) == "li_async"

    _run(cls().achat("hi"))

    payload = _only_call(client)
    assert payload["provider"] == "openai_responses"
    assert payload["model"] == "gpt-5-responses"
    assert payload["input_tokens"] == 600
    assert payload["output_tokens"] == 200
    assert payload["cached_tokens"] == 400
    assert payload["usage"]["shape"] == "openai_responses"
    assert payload["span"]["span_name"] == "my_span"
    assert payload["span"]["span_order"] == 0
    assert payload["prompt_composition"]
    assert payload["response_composition"]
    assert sess._deferred_spans == []


# ═══════════════════════════════════════════════════════════════════
# F. Observations + local_decision drained through the manual emission.
# ═══════════════════════════════════════════════════════════════════


def test_f1_observations_and_local_decision_drained_and_forwarded(_env, monkeypatch):
    """B1: the manual OpenAIResponses emission never reaches
    _flush_deferred_spans (nothing to flush), so it must drain THIS call's
    observations and pop session._local_decision itself — otherwise (e.g.)
    PR #482's unappliable_call_shape observation would sit in the queue until
    OBS_STALE_SECONDS and get swept onto an unrelated later call's row."""
    sess, client = _env
    seeded_obs = {"kind": "unappliable_call_shape", "detail": "streaming kwargs-less call"}

    def fake_check(*a, **k):
        _state.mint_obs_key()
        _state.push_observation(seeded_obs)

    monkeypatch.setattr(enforcer, "_run_sync_check", fake_check)
    enforcer._stash_local_decision(
        sess, "rerouted", "rule_1", verified_by_check=True, rule_name="my_rule",
    )

    def chat(self, messages="hi", *a, **k):
        return _responses_chat_response()

    cls = _install_li_responses_sync(chat)
    cls().chat("hi")

    payload = _only_call(client)
    assert payload["observations"] == [seeded_obs]
    assert payload["local_decision"]["outcome"] == "rerouted"
    assert payload["local_decision"]["rule_id"] == "rule_1"
    # `_local_decision` (old flat slot) is never written any more; the real
    # equivalent claim is the keyed store's entry for THIS call's key being
    # gone (consumed by the manual emission, not left for a later row).
    assert not sess._local_decisions, \
        "must be claimed/cleared after the manual log, exactly like the flush does"
    assert _state.drain_observations() == [], \
        "the observation must be CLAIMED, not left in the queue"


def test_f1b_clean_call_carries_no_observations_or_local_decision(_env):
    """Control: with nothing seeded, the manual row's observations/
    local_decision stay falsy — the new kwargs must not manufacture content."""
    _sess, client = _env

    def chat(self, messages="hi", *a, **k):
        return _responses_chat_response()

    cls = _install_li_responses_sync(chat)
    cls().chat("hi")

    payload = _only_call(client)
    # drain_observations() returns [] (not None) when nothing was claimed —
    # both are falsy, which is what the real client's truthy gate keys off
    # (see test F2 for the stronger "key absent on the wire" proof).
    assert not payload.get("observations")
    assert payload.get("local_decision") is None


def test_f2_log_li_py_wire_payload_omits_absent_extras(monkeypatch):
    """Byte-parity gate (B1): through the REAL client's log_sync,
    observations=None / local_decision=None — the signature every
    pre-existing LlamaIndex streaming call site still uses — must produce a
    wire payload with NEITHER key present AT ALL (not merely falsy). Proves
    the new trailing kwargs are fully backward compatible with every existing
    caller. Mirrors tests/test_log_span_id_synthesis.py's offline-capture
    idiom."""
    from token_police.client import TokenPolice

    c = TokenPolice(
        api_key="tp_sk_test", base_url="http://127.0.0.1:1", timeout=0.1,
        deployment="serverless", log_errors=False,
    )
    captured = {}

    def fake_post(path, json=None):
        captured[path] = json
        return NS(status_code=200)

    c._sync_client.post = fake_post

    sess = TPSession(trace_id="a" * 32, root_span_id="b" * 16)
    monkeypatch.setattr(enforcer, "get_client", lambda: c)
    monkeypatch.setattr(enforcer, "get_current_session", lambda: sess)

    instance = type("FakeLIOpenAIResponses", (), {"model": "gpt-5-responses"})()
    resp = _responses_chat_response()
    enforcer._log_li_py(instance, resp, 0, "span_name", datetime.now(timezone.utc))
    c.flush_sync()

    payload = captured["/v1/guard/log"]
    assert "observations" not in payload
    assert "local_decision" not in payload


# ═══════════════════════════════════════════════════════════════════
# G. Non-responses providers keep the flush path byte-identical: the
#    framework layer must NOT touch the span counter or the pending span
#    name for them (only OpenAIResponses reserves).
# ═══════════════════════════════════════════════════════════════════


def test_g_non_responses_provider_does_not_reserve_order_or_consume_span_name(_env, monkeypatch):
    """B1 regression: the OpenAIResponses reservation must be scoped to
    provider == "openai_responses" only. A non-responses LlamaIndex provider
    (anthropic/google/openai) has a real inner OpenLLMetry span, so the
    framework layer must leave the span counter and the pending span name
    alone for its OWN inner on_start to consume — touching either here would
    desync span_order from the flushed inner span."""
    sess, client = _env
    calls = []
    real_consume = enforcer.consume_pending_span_name

    def spy():
        calls.append(1)
        return real_consume()

    monkeypatch.setattr(enforcer, "consume_pending_span_name", spy)

    def chat(self, messages="hi", *a, **k):
        return NS(
            additional_kwargs={},
            raw=NS(model="gpt-4o", usage=NS(prompt_tokens=10, completion_tokens=5)),
            message=NS(role="assistant", content="ok", additional_kwargs={}),
        )

    cls = type("FakeLIOpenAI", (), {"chat": chat, "model": "gpt-4o"})
    enforcer._set_llamaindex_wrapper(cls, "chat", cls.chat, "sync")
    assert sess._span_counter == 0

    cls().chat("hi")

    assert sess._span_counter == 0, \
        "the framework layer must not consume the span counter for non-responses providers"
    assert calls == [], \
        "the framework layer must not consume the pending span name for " \
        "non-responses providers — the (real) inner OpenLLMetry span owns that"
    assert client.calls == [], \
        "no manual row: this must take the flush path, which no-ops here " \
        "because there is no real inner instrumentor span in this fake"


# ═══════════════════════════════════════════════════════════════════
# H. _extract_li_usage_py — openai_responses branch, unit-level.
# ═══════════════════════════════════════════════════════════════════


class _FakeLIOpenAIResponses:
    """Class name contains 'responses' → LI extractor takes the
    openai_responses branch (never matches anthropic/google first)."""
    model = "gpt-5-responses"


class _FakeLIChatResponse:
    def __init__(self, raw=None, additional_kwargs=None, message=None):
        self.raw = raw
        self.additional_kwargs = additional_kwargs if additional_kwargs is not None else {}
        self.message = message


class TestExtractOpenAIResponsesUsage(unittest.TestCase):
    """B1 — mirrors tests/test_llamaindex_verbatim_usage.py's per-provider
    extractor classes."""

    # ── 1. ChatResponse's OWN additional_kwargs["usage"] (highest priority) ──
    def test_from_additional_kwargs_usage(self):
        usage = NS(input_tokens=1000, output_tokens=200,
                   input_tokens_details=NS(cached_tokens=400))
        resp = _FakeLIChatResponse(
            additional_kwargs={"usage": usage}, raw=NS(model="gpt-5-responses"),
        )
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), resp)
        self.assertEqual(inp, 600)   # max(0, 1000 - 400)
        self.assertEqual(out, 200)
        self.assertEqual(cached, 400)
        self.assertEqual(model, "gpt-5-responses")

    # ── 2. Terminal stream event: raw.response.usage (Response nested) ──
    def test_from_stream_terminal_response_usage(self):
        usage = NS(input_tokens=1000, output_tokens=200,
                   input_tokens_details=NS(cached_tokens=400))
        resp = _FakeLIChatResponse(
            additional_kwargs={},  # not yet mirrored onto ChatResponse itself
            raw=NS(type="response.completed",
                  response=NS(usage=usage, model="gpt-5-stream")),
        )
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), resp)
        self.assertEqual((inp, out, cached), (600, 200, 400))
        self.assertEqual(model, "gpt-5-stream")

    # ── 3. Non-stream: raw.usage directly (no .response nesting) ──
    def test_from_raw_usage_direct(self):
        usage = NS(input_tokens=1000, output_tokens=200,
                   input_tokens_details=NS(cached_tokens=400))
        resp = _FakeLIChatResponse(
            additional_kwargs={}, raw=NS(usage=usage, model="gpt-5-direct"),
        )
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), resp)
        self.assertEqual((inp, out, cached), (600, 200, 400))
        self.assertEqual(model, "gpt-5-direct")

    # ── 4. dict variants (REST / json.loads-decoded bodies) ──
    def test_dict_variants(self):
        resp = _FakeLIChatResponse(
            additional_kwargs={"usage": {
                "input_tokens": 1000, "output_tokens": 200,
                "input_tokens_details": {"cached_tokens": 400},
            }},
            raw={"model": "gpt-5-dict"},
        )
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), resp)
        self.assertEqual((inp, out, cached), (600, 200, 400))
        self.assertEqual(model, "gpt-5-dict")

    # ── 5. no usage anywhere → zeros, model from raw fallback ──
    def test_no_usage_anywhere_degrades_to_zeros(self):
        resp = _FakeLIChatResponse(additional_kwargs={}, raw=NS(model="gpt-5-empty"))
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), resp)
        self.assertEqual((inp, out, cached), (0, 0, 0))
        self.assertEqual(model, "gpt-5-empty")

    # ── 6. garbage raw TYPE (int) → degrades gracefully, no raise ──
    def test_garbage_raw_type_degrades_gracefully(self):
        resp = _FakeLIChatResponse(additional_kwargs={}, raw=12345)
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), resp)
        self.assertEqual((inp, out, cached), (0, 0, 0))
        self.assertEqual(model, "gpt-5-responses")  # falls back to instance.model

    # ── 7. golden rule: a RAISING attribute never escapes ──
    def test_hostile_raw_never_raises(self):
        class _HostileRaw:
            model = "gpt-5-fallback"

            @property
            def response(self):
                raise RuntimeError("boom")

        resp = _FakeLIChatResponse(additional_kwargs={}, raw=_HostileRaw())
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), resp)
        self.assertEqual((inp, out, cached), (0, 0, 0))
        self.assertEqual(model, "gpt-5-responses")  # instance hint, not raw's

    # ── 8. no response at all (generic early-return, provider never resolved) ──
    def test_none_response_returns_fallback(self):
        model, inp, out, cached = _extract_li_usage_py(_FakeLIOpenAIResponses(), None)
        self.assertEqual((inp, out, cached), (0, 0, 0))
        self.assertEqual(model, "gpt-5-responses")


# ═══════════════════════════════════════════════════════════════════
# I. OpenAIResponses stream_chat — one row per call via the manual stream
#    path, usage from the terminal event.
# ═══════════════════════════════════════════════════════════════════


class _LIResponsesStreamChunk:
    def __init__(self, delta=None, usage=None, model="gpt-5-responses"):
        self.delta = delta
        self.message = NS(additional_kwargs={})
        self.additional_kwargs = {}
        if usage is not None:
            self.additional_kwargs["usage"] = usage
            self.raw = NS(type="response.completed", response=NS(usage=usage, model=model))
        else:
            self.raw = NS(type="response.output_text.delta")


def test_i_openai_responses_stream_chat_single_row_terminal_usage(_env):
    """B1: OpenAIResponses.stream_chat goes through the SAME li_stream/
    _guard_sync_stream_li manual path as every other LlamaIndex stream — one
    row per call, usage read off the terminal response.completed event, and
    (unlike the non-stream tails) the order is RESERVED via
    session.next_span_order() rather than the plain counter snapshot, since
    nothing else will ever consume it for this provider."""
    sess, client = _env

    def stream_chat(self, messages="hi", *a, **k):
        return iter([
            _LIResponsesStreamChunk(delta="Hel"),
            _LIResponsesStreamChunk(delta="lo"),
            _LIResponsesStreamChunk(delta=None, usage=_RespUsage(1000, 200, 400)),
        ])

    cls = type("FakeLIOpenAIResponses", (), {"stream_chat": stream_chat, "model": "gpt-5-responses"})
    enforcer._set_llamaindex_wrapper(cls, "stream_chat", cls.stream_chat, "stream")
    assert sess._span_counter == 0

    out = list(cls().stream_chat("hi"))
    assert len(out) == 3

    payload = _only_call(client)
    assert payload["provider"] == "openai_responses"
    assert payload["input_tokens"] == 600
    assert payload["output_tokens"] == 200
    assert payload["cached_tokens"] == 400
    assert payload["usage"]["shape"] == "openai_responses"
    assert payload["latency"]["is_streaming"] is True
    assert payload["span"]["span_order"] == 0
    assert sess._span_counter == 1, "the reserved order must advance the counter"


if __name__ == "__main__":
    unittest.main()
