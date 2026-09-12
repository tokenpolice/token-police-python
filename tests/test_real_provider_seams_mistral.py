"""Real-package seam tests for the `mistralai` SDK (major 1.x AND 2.x).

Why this file exists
────────────────────
`mistralai` 2.0 turned the distribution into a namespace package and moved
every resource module one level down: `mistralai.chat` became
`mistralai.client.chat`, and the client itself moved from
`from mistralai import Mistral` to `from mistralai.client import Mistral`.

`enforcer._wrap_method` swallows the `ImportError` for a module that is not
installed (correct: the customer owns their provider versions and the SDK
pins none). The failure mode is therefore SILENT: on `mistralai>=2` every
`_TARGET_METHODS` row aimed at the 1.x paths resolved to nothing, so a
Mistral customer got **no `/check` and no `/log`** — un-enforced and
un-metered spend, with no error anywhere.

Unit tests built on fakes cannot see this class of bug: they name the class
object directly and never exercise the import. So these tests drive the REAL
installed `mistralai` package through a real `httpx.MockTransport` (no
network) and assert both halves:

  1. the registry's module paths still RESOLVE on the installed major, and
     `tp.init()` actually lands a wrapper on the installed classes; and
  2. a real round trip through those wrappers produces exactly one `/check`
     and exactly one `/log` row carrying the right tokens/model/shape.

The file must pass unchanged on BOTH majors — nothing here may name a
version. Version-specific knowledge is confined to `_MOD_PREFIX` /
`_import_first`, which ASK the installed package rather than assume.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("mistralai")

# Plain httpx: the TokenPolice SDK declares it as a dependency and mistralai
# is built on it on both majors (unlike anthropic, which moved to httpx2 —
# see tests/_anthropic_httpx.py; that shim is anthropic-only).
import httpx  # noqa: E402

import token_police as tp  # noqa: E402
from token_police import enforcer  # noqa: E402
from token_police.client import TokenPolice  # noqa: E402
from token_police.exceptions import TokenPoliceBlockedError  # noqa: E402


# ═══════════════════════════════════════════════════════════════════
# Version-agnostic resolution of the installed major
# ═══════════════════════════════════════════════════════════════════

def _import_first(*module_paths):
    """First of ``module_paths`` that imports, or None."""
    for path in module_paths:
        try:
            return importlib.import_module(path)
        except ImportError:
            continue
    return None


def _resolve_client():
    """The module exporting `Mistral`, plus the resource-module prefix.

    `mistralai` 1.x ALSO has a `mistralai.client` module — a legacy migration
    stub exporting only `MistralClient` — so "does `mistralai.client` import"
    is not a usable major discriminator. Ask for the symbol instead.
    """
    for path in ("mistralai.client", "mistralai"):
        mod = _import_first(path)
        cls = getattr(mod, "Mistral", None) if mod is not None else None
        if cls is not None:
            return cls, path + "."
    raise AssertionError(
        "cannot resolve `Mistral` from either `mistralai.client` (2.x) or "
        "`mistralai` (1.x) — the installed mistralai has moved its client "
        "again; find the new home rather than pinning a version."
    )


# `_MOD_PREFIX` is "mistralai.client." on 2.x, "mistralai." on 1.x.
Mistral, _MOD_PREFIX = _resolve_client()


# ═══════════════════════════════════════════════════════════════════
# Wire fixtures — minimal JSON bodies that parse on BOTH majors
# ═══════════════════════════════════════════════════════════════════

CHAT_JSON = {
    "id": "cmpl-1", "object": "chat.completion", "model": "mistral-large-latest",
    "created": 1,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 23, "completion_tokens": 6, "total_tokens": 29},
}

FIM_JSON = {
    "id": "fim-1", "object": "chat.completion", "model": "codestral-latest",
    "created": 1,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "return 1"},
                 "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 23, "completion_tokens": 6, "total_tokens": 29},
}

EMB_JSON = {
    "id": "emb-1", "object": "list", "model": "mistral-embed",
    "usage": {"prompt_tokens": 11, "completion_tokens": 0, "total_tokens": 11},
    "data": [{"object": "embedding", "embedding": [0.1, 0.2], "index": 0}],
}

OCR_JSON = {
    "pages": [{"index": 0, "markdown": "# t", "images": [],
               "dimensions": {"dpi": 200, "height": 100, "width": 100}}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 3, "doc_size_bytes": 1234},
}

TRN_JSON = {
    "model": "voxtral-mini-latest", "text": "hello", "language": "en",
    "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7,
              "prompt_audio_seconds": 12},
}

SPEECH_JSON = {"audio_data": "UklGRmZha2VhdWRpbw=="}

# `chat.parse(...)` funnels into `chat.complete(...)`; the SDK then validates
# the assistant content against the caller's pydantic model, so the content
# must be JSON when the request carried a `response_format`.
PARSE_CONTENT = json.dumps({"answer": "hi"})


def _chat_sse(model="mistral-large-latest"):
    """Mistral chat SSE: `data:` lines only, usage on the final chunk."""
    def chunk(delta, usage=None, finish=None):
        body = {"id": "c1", "model": model, "object": "chat.completion.chunk",
                "created": 1,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage:
            body["usage"] = usage
        return "data: %s\n\n" % json.dumps(body)

    return (
        chunk({"role": "assistant", "content": "he"})
        + chunk({"content": "llo"}, finish="stop",
                usage={"prompt_tokens": 23, "completion_tokens": 6, "total_tokens": 29})
        + "data: [DONE]\n\n"
    ).encode()


def _handler(record=None):
    """One MockTransport handler routing every Mistral surface used here."""
    def handler(request: httpx.Request) -> httpx.Response:
        if record is not None:
            record.append(request.url.path)
        path = request.url.path
        accept = request.headers.get("accept") or ""
        if path.endswith("/fim/completions"):
            return httpx.Response(200, json=FIM_JSON)
        if path.endswith("/chat/completions"):
            try:
                body = json.loads(request.content or b"{}")
            except Exception:
                body = {}
            if body.get("stream"):
                return httpx.Response(
                    200, headers={"content-type": "text/event-stream"},
                    content=_chat_sse())
            if body.get("response_format"):
                parsed = json.loads(json.dumps(CHAT_JSON))
                parsed["choices"][0]["message"]["content"] = PARSE_CONTENT
                return httpx.Response(200, json=parsed)
            return httpx.Response(200, json=CHAT_JSON)
        if path.endswith("/embeddings"):
            return httpx.Response(200, json=EMB_JSON)
        if path.endswith("/ocr"):
            return httpx.Response(200, json=OCR_JSON)
        if "transcriptions" in path:
            return httpx.Response(200, json=TRN_JSON)
        if path.endswith("/audio/speech"):
            return httpx.Response(200, json=SPEECH_JSON)
        return httpx.Response(404, json={"unrouted": path})
    return handler


def _clients(handler):
    """Real Mistral sync+async clients bound to MockTransport (no network)."""
    return Mistral(
        api_key="test-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def _tp_client():
    """One module-wide real client. `auto_instrument()` is process-global-once
    (`enforcer._is_instrumented`), so uninstrument first to guarantee THIS
    module's init installs a fresh set of wrappers on the real mistralai."""
    enforcer.uninstrument()
    return tp.init(
        api_key="tp_sk_test_mistral", base_url="http://127.0.0.1:9",
        firewall="enforce", deployment="serverless", timeout=0.2, log_errors=True,
    )


@pytest.fixture
def env(_tp_client, monkeypatch):
    rows, checks = [], []

    def log_sync(self, **kw):
        rows.append(kw)

    def check_sync(self, **kw):
        checks.append(kw)
        return {"status": "allowed"}

    async def check_async(self, **kw):
        checks.append(kw)
        return {"status": "allowed"}

    monkeypatch.setattr(TokenPolice, "log_sync", log_sync)
    monkeypatch.setattr(TokenPolice, "check_sync", check_sync)
    monkeypatch.setattr(TokenPolice, "check", check_async)
    return NS(rows=rows, checks=checks)


def _llm_rows(rows):
    return [r for r in rows if (r.get("span") or {}).get("span_kind") == "llm"]


def _one(rows):
    llm = _llm_rows(rows)
    assert len(llm) == 1, f"expected exactly one llm row, got {len(llm)}: {llm}"
    return llm[0]


# ═══════════════════════════════════════════════════════════════════
# 1. Seam resolution — derived from the LIVE registry, never hand-copied
# ═══════════════════════════════════════════════════════════════════

def _mistral_rows():
    return [t for t in enforcer._TARGET_METHODS
            if str(t.get("module", "")).startswith("mistralai")]


def _family(module: str) -> str:
    """Deployment family a module path belongs to. Mistral ships THREE
    parallel client trees (direct API, Azure AI, Google Vertex) whose classes
    are distinct objects, not re-exports — a wrapper on one never covers the
    others, so they are graded separately."""
    if ".azure." in module or module.startswith("mistralai_azure"):
        return "azure"
    if ".gcp." in module or module.startswith("mistralai_gcp"):
        return "gcp"
    return "core"


# Surfaces that exist on only ONE major. A row for these is allowed to
# resolve nowhere on the other major — that is a fact about the vendor's
# release, not a broken registry entry.
_MAJOR_GATED_CLASSES = {"Speech"}  # TTS: mistralai>=2 only


def _surfaces():
    """(family, class, method) → list of module paths declared for it."""
    out = {}
    for row in _mistral_rows():
        key = (_family(row["module"]), row["object"], row["method"])
        out.setdefault(key, []).append(row["module"])
    return out


class TestSeamResolution:
    """The bug this file exists for: registry module paths that resolve to
    NOTHING on the installed major, silently."""

    def test_every_declared_surface_resolves_on_this_major(self):
        unresolved = []
        for (family, cls_name, method), modules in sorted(_surfaces().items()):
            resolved = []
            for module in modules:
                mod = _import_first(module)
                if mod is None:
                    continue
                cls = getattr(mod, cls_name, None)
                if cls is None:
                    continue
                fn = getattr(cls, method, None)
                assert callable(fn), (
                    f"{module}.{cls_name}.{method} exists but is not callable "
                    f"— the registry row can never wrap it"
                )
                resolved.append(module)
            if not resolved and cls_name not in _MAJOR_GATED_CLASSES:
                unresolved.append((family, cls_name, method, modules))
        assert not unresolved, (
            "these mistralai registry surfaces resolve to NOTHING on the "
            "installed major — every call through them is un-checked and "
            "un-logged:\n" + "\n".join(map(repr, unresolved))
        )

    def test_core_surfaces_cover_the_billable_endpoints(self):
        """Guards against a surface being dropped from the registry
        wholesale. Derived from the live table, so it fails loudly if a
        class/method is renamed away rather than silently passing."""
        core = {(cls, method) for (family, cls, method) in _surfaces()
                if family == "core"}
        for cls_name, methods in [
            ("Chat", {"complete", "complete_async", "stream", "stream_async"}),
            ("Embeddings", {"create", "create_async"}),
            ("Ocr", {"process", "process_async"}),
            ("Transcriptions", {"complete", "complete_async"}),
            ("Fim", {"complete", "complete_async", "stream", "stream_async"}),
        ]:
            for method in methods:
                assert (cls_name, method) in core, (
                    f"no mistralai registry row for {cls_name}.{method}")

    def test_parse_is_deliberately_not_registered(self):
        """`Chat.parse*` delegates to `Chat.complete*` / `Chat.stream*` on
        both majors. Registering it would run a SECOND pre-flight check and
        emit a SECOND /log row for one call (double-bill)."""
        assert not [r for r in _mistral_rows() if r["method"].startswith("parse")]


class TestWrapperLanded:
    """Resolution is necessary but not sufficient — assert `tp.init()` really
    installed the pre-flight wrapper on the classes of THIS major."""

    @pytest.mark.parametrize("mod_suffix,cls_name,methods", [
        ("chat", "Chat", ["complete", "complete_async", "stream", "stream_async"]),
        ("embeddings", "Embeddings", ["create", "create_async"]),
        ("ocr", "Ocr", ["process", "process_async"]),
        ("transcriptions", "Transcriptions", ["complete", "complete_async"]),
        ("fim", "Fim", ["complete", "complete_async", "stream", "stream_async"]),
    ])
    def test_installed_class_methods_are_wrapped(self, _tp_client, mod_suffix,
                                                  cls_name, methods):
        mod = importlib.import_module(_MOD_PREFIX + mod_suffix)
        cls = getattr(mod, cls_name)
        for method in methods:
            fn = getattr(cls, method)
            assert getattr(fn, "_tp_preflight_wrapper", False), (
                f"{_MOD_PREFIX}{mod_suffix}.{cls_name}.{method} carries no "
                f"TokenPolice wrapper after tp.init() — calls through it are "
                f"un-checked and un-logged on this mistralai major")


# ═══════════════════════════════════════════════════════════════════
# 2. Round trips through the real SDK
# ═══════════════════════════════════════════════════════════════════

MESSAGES = [{"role": "user", "content": "hi"}]


class TestChat:
    def test_complete(self, env):
        c = _clients(_handler())
        c.chat.complete(model="mistral-large-latest", messages=MESSAGES)
        assert len(env.checks) == 1
        row = _one(env.rows)
        assert row["input_tokens"] == 23
        assert row["output_tokens"] == 6
        assert row["model"] == "mistral-large-latest"
        assert row["provider"] == "mistral"
        assert (row.get("usage") or {}).get("shape") == "mistral_chat"

    def test_complete_async(self, env):
        c = _clients(_handler())
        _run(c.chat.complete_async(model="mistral-large-latest", messages=MESSAGES))
        assert len(env.checks) == 1
        row = _one(env.rows)
        assert (row["input_tokens"], row["output_tokens"]) == (23, 6)
        assert (row.get("usage") or {}).get("shape") == "mistral_chat"

    def test_stream(self, env):
        """Usage lives on the FINAL `CompletionEvent{data: CompletionChunk}`;
        the row must only be emitted once the customer has drained it."""
        c = _clients(_handler())
        stream = c.chat.stream(model="mistral-large-latest", messages=MESSAGES)
        assert len(env.checks) == 1
        chunks = list(stream)
        assert len(chunks) == 2
        row = _one(env.rows)
        assert (row["input_tokens"], row["output_tokens"]) == (23, 6)
        assert row["model"] == "mistral-large-latest"

    def test_stream_async(self, env):
        c = _clients(_handler())

        async def go():
            stream = await c.chat.stream_async(model="mistral-large-latest",
                                               messages=MESSAGES)
            return [chunk async for chunk in stream]

        chunks = _run(go())
        assert len(chunks) == 2
        assert len(env.checks) == 1
        row = _one(env.rows)
        assert (row["input_tokens"], row["output_tokens"]) == (23, 6)

    def test_parse_bills_exactly_once(self, env):
        """`Chat.parse` delegates to `Chat.complete`. If `parse` were ALSO
        registered in `_TARGET_METHODS` this would be 2 checks and 2 rows for
        one billable call."""
        import pydantic

        class Answer(pydantic.BaseModel):
            answer: str

        c = _clients(_handler())
        out = c.chat.parse(model="mistral-large-latest", messages=MESSAGES,
                           response_format=Answer)
        assert out.choices[0].message.parsed.answer == "hi"
        assert len(env.checks) == 1
        row = _one(env.rows)
        assert (row["input_tokens"], row["output_tokens"]) == (23, 6)


class TestFim:
    """Codestral fill-in-the-middle: a DIFFERENT class and endpoint from chat
    (`/v1/fim/completions`), with `prompt`/`suffix` and no `messages` — so
    composition capture must degrade rather than raise."""

    def test_complete(self, env):
        c = _clients(_handler())
        c.fim.complete(model="codestral-latest", prompt="def f(", suffix="return 1")
        assert len(env.checks) == 1
        row = _one(env.rows)
        assert (row["input_tokens"], row["output_tokens"]) == (23, 6)
        assert row["model"] == "codestral-latest"
        assert row["provider"] == "mistral"

    def test_complete_async(self, env):
        c = _clients(_handler())
        _run(c.fim.complete_async(model="codestral-latest", prompt="def f(",
                                  suffix="return 1"))
        assert len(env.checks) == 1
        assert (_one(env.rows)["input_tokens"],
                _one(env.rows)["output_tokens"]) == (23, 6)


class TestEmbeddings:
    def test_create(self, env):
        c = _clients(_handler())
        c.embeddings.create(model="mistral-embed", inputs=["a", "b"])
        assert len(env.checks) == 1
        row = _one(env.rows)
        assert row["input_tokens"] == 11
        assert row["model"] == "mistral-embed"
        assert row["operation"] == "embedding"
        assert (row.get("usage") or {}).get("shape") == "mistral_embed"

    def test_create_async(self, env):
        c = _clients(_handler())
        _run(c.embeddings.create_async(model="mistral-embed", inputs=["a"]))
        assert len(env.checks) == 1
        assert _one(env.rows)["input_tokens"] == 11


class TestOcr:
    def test_process(self, env):
        """OCR is billed per PAGE, not per token — `usage_info.pages_processed`
        lands in `usage.items.ocr_pages`."""
        c = _clients(_handler())
        c.ocr.process(model="mistral-ocr-latest",
                      document={"type": "document_url",
                                "document_url": "https://x/y.pdf"})
        assert len(env.checks) == 1
        row = _one(env.rows)
        usage = row.get("usage") or {}
        assert usage.get("shape") == "mistral_ocr"
        assert usage.get("items", {}).get("ocr_pages") == 3
        assert row["operation"] == "ocr"
        assert row["model"] == "mistral-ocr-latest"

    def test_process_async(self, env):
        c = _clients(_handler())
        _run(c.ocr.process_async(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": "https://x/y.pdf"}))
        assert len(env.checks) == 1
        assert (_one(env.rows).get("usage") or {}).get("items", {}).get("ocr_pages") == 3


class TestTranscriptions:
    def test_complete_reports_audio_seconds(self, env):
        """Voxtral STT is billed per AUDIO SECOND. The only place the duration
        appears is `usage.prompt_audio_seconds` on the response — a `file_url=`
        / `file_id=` call sends no local file, so the local-file fallback can
        only ever produce 0."""
        c = _clients(_handler())
        c.audio.transcriptions.complete(model="voxtral-mini-latest",
                                        file_url="https://x/a.mp3")
        assert len(env.checks) == 1
        row = _one(env.rows)
        usage = row.get("usage") or {}
        assert usage.get("shape") == "mistral_audio_stt"
        assert usage.get("duration", {}).get("audio_seconds") == 12.0
        assert row["operation"] == "audio_stt"
        assert row["model"] == "voxtral-mini-latest"

    def test_complete_async_reports_audio_seconds(self, env):
        c = _clients(_handler())
        _run(c.audio.transcriptions.complete_async(model="voxtral-mini-latest",
                                                   file_url="https://x/a.mp3"))
        assert len(env.checks) == 1
        assert (_one(env.rows).get("usage") or {}
                ).get("duration", {}).get("audio_seconds") == 12.0


@pytest.mark.skipif(_MOD_PREFIX == "mistralai.",
                    reason="mistralai 1.x ships no TTS (audio.speech) surface")
class TestSpeech:
    def test_complete(self, env):
        """TTS is billed on the REQUEST's character count, so the extractor
        reads it from kwargs — exact regardless of the response body."""
        c = _clients(_handler())
        c.audio.speech.complete(model="voxtral-mini-latest", input="hello world")
        assert len(env.checks) == 1
        row = _one(env.rows)
        usage = row.get("usage") or {}
        assert usage.get("shape") == "mistral_audio_tts"
        assert usage.get("items", {}).get("tts_characters") == len("hello world")
        assert row["operation"] == "audio_tts"

    def test_response_body_is_never_composed_as_assistant_text(self, env):
        """`SpeechResponse.audio_data` is base64 audio. It must never be
        recorded as text content."""
        c = _clients(_handler())
        c.audio.speech.complete(model="voxtral-mini-latest", input="hello world")
        row = _one(env.rows)
        blob = json.dumps(row.get("response_composition") or [])
        assert SPEECH_JSON["audio_data"] not in blob


class TestGoldenRules:
    def test_enforce_denial_raises_and_never_reaches_the_provider(self, env,
                                                                  monkeypatch):
        """The only exception TokenPolice may raise into a customer app — and
        it must fire BEFORE the HTTP request, or the spend already happened."""
        seen = []

        def _blocked(**_kw):
            raise TokenPoliceBlockedError("budget exhausted")

        monkeypatch.setattr(enforcer, "_run_sync_check", _blocked)
        c = _clients(_handler(record=seen))
        with pytest.raises(TokenPoliceBlockedError):
            c.chat.complete(model="mistral-large-latest", messages=MESSAGES)
        assert seen == [], "provider HTTP request was made despite a block"
        assert _llm_rows(env.rows) == []

    def test_check_failure_is_fail_open(self, env, monkeypatch):
        """A /check that blows up (network down, service outage) must never
        break the customer's LLM call."""
        seen = []

        def _boom(self, **_kw):
            raise RuntimeError("collector unreachable")

        monkeypatch.setattr(TokenPolice, "check_sync", _boom)
        c = _clients(_handler(record=seen))
        out = c.chat.complete(model="mistral-large-latest", messages=MESSAGES)
        assert out.choices[0].message.content == "hi"
        assert seen and seen[0].endswith("/chat/completions")
