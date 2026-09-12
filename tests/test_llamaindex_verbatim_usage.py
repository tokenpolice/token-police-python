"""LlamaIndex manual-log path must forward Gemini's verbatim usage_metadata.

Audit F1 (under-bill): the LlamaIndex google extractor nets cached tokens out of
input (`input = prompt - cached`) and returns positional counts. Sent as
positional counts with no usage= kwarg, client.log_sync synthesised an
`openai_compatible_chat` shape whose prompt_tokens the server's mapOpenAIChat
treats as cache-INCLUSIVE — so it subtracted the cached amount a SECOND time,
under-counting uncached input by the full cached amount.

Fix (google branch): _log_li_py forwards
`usage={"shape": "google_genai", "raw": <usage_metadata>}` so mapGoogleGenAI
applies Gemini's cache-inclusive semantics exactly once. The positional counts
stay as the fallback when no verbatim block is available.

The anthropic branch (F4) and the openai branch (PY-LI-OpenAI) forward their own
verbatim blocks the same way: anthropic_messages so cache WRITES bill at the
write rate, and openai_compatible_chat so cache READS survive (LlamaIndex's
OpenAI extractor drops prompt_tokens_details, so without the block cache reads
vanish → over-bill, and the synthesised fallback nets an already-netted input a
second time). All three reconcile the positional counts from the forwarded raw
and fall back to today's positional behavior when no block is available.

All fakes — no LlamaIndex involved. Mirrors tests/test_reasoning_double_count.py
and tests/test_anthropic_stream_usage_shape.py.
"""
import json
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police import enforcer
from token_police.enforcer import (
    _li_google_verbatim_usage_py,
    _li_anthropic_verbatim_usage_py,
    _li_openai_verbatim_usage_py,
    _extract_li_usage_py,
)


# ── LlamaIndex fakes (mirror test_reasoning_double_count.py) ──

class _FakeLIGemini:
    """Class name contains 'gemini' → LI extractor takes the google branch."""
    model = "gemini-2.0-flash"


class _FakeLIOpenAI:
    """Class name avoids anthropic/google → OpenAI-default branch."""
    model = "gpt-4o"


class _FakeLIAnthropic:
    """Class name contains 'anthropic' → LI extractor takes the anthropic branch."""
    model = "claude-3-5-sonnet-20241022"


class _FakeLIChatResponse:
    def __init__(self, raw=None, additional_kwargs=None, message=None):
        self.raw = raw
        self.additional_kwargs = additional_kwargs or {}
        self.message = message


class _FakeUsageMetadataObj:
    """Object-style usage_metadata (google.genai returns an object, not a dict).
    Plain attrs, NO model_dump — its ``__dict__`` carries exactly the 4 scalars,
    so generic serialization (ladder tier 2) yields those same 4 keys."""

    def __init__(self, prompt=0, candidates=0, cached=0, thoughts=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.cached_content_token_count = cached
        self.thoughts_token_count = thoughts


class _FakeUsageMetadataModel:
    """Pydantic-like usage_metadata mirroring google.genai's real
    ``GenerateContentResponseUsageMetadata``: snake_case fields + ``model_dump()``
    whose output includes the modality/cache detail lists and
    ``tool_use_prompt_token_count`` that the 4-scalar rebuild would drop."""

    def __init__(self):
        self._data = {
            "prompt_token_count": 1000,
            "candidates_token_count": 200,
            "cached_content_token_count": 800,
            "thoughts_token_count": 60,
            "tool_use_prompt_token_count": 40,
            "prompt_tokens_details": [
                {"modality": "TEXT", "token_count": 700},
                {"modality": "AUDIO", "token_count": 300},
            ],
            "cache_tokens_details": [
                {"modality": "TEXT", "token_count": 800},
            ],
        }

    def model_dump(self):
        return dict(self._data)


class _FakeUsageMetadataSlots:
    """Exotic slotted usage object: NO ``__dict__`` and NO ``model_dump`` →
    generic serialization yields ``str(obj)`` (a non-dict), so the helper falls
    back to the 4-scalar snake_case rebuild (ladder tier 3)."""

    __slots__ = ("prompt_token_count", "candidates_token_count",
                 "cached_content_token_count", "thoughts_token_count")

    def __init__(self, prompt=0, candidates=0, cached=0, thoughts=0):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        self.cached_content_token_count = cached
        self.thoughts_token_count = thoughts


class _CaptureTP:
    """Captures the kwargs of the single log_sync _log_li_py emits."""

    def __init__(self):
        self.captured = {}

    def log_sync(self, **kw):
        self.captured.update(kw)


def _fake_session():
    return NS(
        user_id="u1",
        paid_plan="free",
        workflow_name="wf",
        session_id="s1",
        metadata={},
        trace_id="t" * 32,
        root_span_id="r" * 16,
        _pending_compositions={},
    )


class TestLlamaIndexGeminiVerbatimUsage(unittest.TestCase):

    # ── 1. dict usage_metadata with cache → forwarded verbatim ──
    def test_google_dict_usage_metadata_forwarded_verbatim(self):
        um = {
            "prompt_token_count": 1000,
            "cached_content_token_count": 800,
            "candidates_token_count": 200,
            "thoughts_token_count": 60,
        }
        resp = _FakeLIChatResponse(raw=NS(usage_metadata=um))
        block = _li_google_verbatim_usage_py(_FakeLIGemini(), resp)
        self.assertEqual(block, {"shape": "google_genai", "raw": um})
        # The forwarded raw is a JSON snapshot of the cache-inclusive metadata —
        # deep-equal but deliberately NOT the same dict (freezes it against later
        # mutation and pre-validates background-POST serialization).
        self.assertIsNot(block["raw"], um)
        # Positional counts are unchanged from today (netted, cache-subtracted).
        _, inp, out, cached = _extract_li_usage_py(_FakeLIGemini(), resp)
        self.assertEqual(inp, 200)  # max(0, 1000 - 800)
        self.assertEqual(out, 260)  # 200 candidates + 60 thoughts
        self.assertEqual(cached, 800)

    # ── 2. object-style usage_metadata → serialized snake_case dict block ──
    # Tier 2: a plain-attr object serializes via the generic __dict__ path; here
    # __dict__ happens to be exactly the 4 scalars, so the block is identical to
    # the old rebuild (no regression) while richer objects keep their details.
    def test_google_object_usage_metadata_snake_case_dict(self):
        um_obj = _FakeUsageMetadataObj(prompt=1000, candidates=200, cached=800, thoughts=60)
        resp = _FakeLIChatResponse(raw=NS(usage_metadata=um_obj))
        block = _li_google_verbatim_usage_py(_FakeLIGemini(), resp)
        self.assertEqual(
            block,
            {
                "shape": "google_genai",
                "raw": {
                    "prompt_token_count": 1000,
                    "candidates_token_count": 200,
                    "cached_content_token_count": 800,
                    "thoughts_token_count": 60,
                },
            },
        )
        # Block is JSON-serializable (object metadata was flattened to a dict).
        json.dumps(block)

    # ── 2b. model_dump() object (F7) → modality/cache details forwarded verbatim ──
    # Tier 2: the real google.genai type exposes model_dump(); its detail lists
    # and tool_use_prompt_token_count must survive so the server keeps the
    # AUDIO premium instead of folding everything into text.
    def test_google_model_dump_preserves_modality_details(self):
        um_obj = _FakeUsageMetadataModel()
        resp = _FakeLIChatResponse(raw=NS(usage_metadata=um_obj))
        block = _li_google_verbatim_usage_py(_FakeLIGemini(), resp)
        self.assertEqual(block["shape"], "google_genai")
        raw = block["raw"]
        self.assertEqual(raw["prompt_token_count"], 1000)
        self.assertEqual(raw["candidates_token_count"], 200)
        self.assertEqual(raw["cached_content_token_count"], 800)
        self.assertEqual(raw["thoughts_token_count"], 60)
        # These are exactly what the old 4-scalar rebuild dropped.
        self.assertEqual(raw["tool_use_prompt_token_count"], 40)
        self.assertEqual(
            raw["prompt_tokens_details"],
            [{"modality": "TEXT", "token_count": 700},
             {"modality": "AUDIO", "token_count": 300}],
        )
        self.assertEqual(
            raw["cache_tokens_details"],
            [{"modality": "TEXT", "token_count": 800}],
        )
        json.dumps(block)  # background-POST serializable

    # ── 2c. exotic slotted object → 4-scalar rebuild (ladder tier 3) ──
    def test_google_slotted_object_falls_back_to_scalar_rebuild(self):
        um_obj = _FakeUsageMetadataSlots(prompt=1000, candidates=200, cached=800, thoughts=60)
        resp = _FakeLIChatResponse(raw=NS(usage_metadata=um_obj))
        block = _li_google_verbatim_usage_py(_FakeLIGemini(), resp)
        self.assertEqual(
            block,
            {
                "shape": "google_genai",
                "raw": {
                    "prompt_token_count": 1000,
                    "candidates_token_count": 200,
                    "cached_content_token_count": 800,
                    "thoughts_token_count": 60,
                },
            },
        )
        json.dumps(block)

    # ── 3. no / empty usage_metadata → no block, fallback intact ──
    def test_google_no_usage_metadata_returns_none(self):
        resp = _FakeLIChatResponse(raw=NS())  # no usage_metadata attribute
        self.assertIsNone(_li_google_verbatim_usage_py(_FakeLIGemini(), resp))

    def test_google_empty_usage_metadata_all_zero_returns_none(self):
        resp = _FakeLIChatResponse(
            raw=NS(usage_metadata={"prompt_token_count": 0, "candidates_token_count": 0})
        )
        self.assertIsNone(_li_google_verbatim_usage_py(_FakeLIGemini(), resp))

    def test_none_response_returns_none(self):
        self.assertIsNone(_li_google_verbatim_usage_py(_FakeLIGemini(), None))

    def test_hostile_response_never_raises(self):
        # A raw whose usage_metadata attribute access explodes → None, no throw.
        class _Hostile:
            @property
            def usage_metadata(self):
                raise RuntimeError("boom")

        resp = _FakeLIChatResponse(raw=_Hostile())
        self.assertIsNone(_li_google_verbatim_usage_py(_FakeLIGemini(), resp))

    def test_non_serializable_nested_value_falls_back(self):
        # Positive top-level counts, but a nested value json.dumps rejects
        # (a set / SimpleNamespace). Without the snapshot this would only fail
        # in the background POST thread and silently drop the whole log row.
        um = {
            "prompt_token_count": 1000,
            "cached_content_token_count": 800,
            "candidates_token_count": 200,
            "prompt_tokens_details": {1, 2, 3},  # sets are not JSON-serializable
            "raw_obj": NS(x=1),
        }
        resp = _FakeLIChatResponse(raw=NS(usage_metadata=um))
        self.assertIsNone(_li_google_verbatim_usage_py(_FakeLIGemini(), resp))
        # Positional fallback unchanged — the row is still logged from these.
        _, inp, out, cached = _extract_li_usage_py(_FakeLIGemini(), resp)
        self.assertEqual(inp, 200)  # max(0, 1000 - 800)
        self.assertEqual(out, 200)
        self.assertEqual(cached, 800)


class TestLlamaIndexLogForwardsUsage(unittest.TestCase):

    # ── 4. _log_li_py forwards usage= through to log_sync (google) ──
    def test_log_li_py_forwards_google_usage_block(self):
        um = {
            "prompt_token_count": 1000,
            "cached_content_token_count": 800,
            "candidates_token_count": 200,
        }
        resp = _FakeLIChatResponse(raw=NS(usage_metadata=um))
        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIGemini(), resp, 0, "gemini_call", None)

        kw = tp.captured
        self.assertIn("model", kw)  # row was logged
        self.assertEqual(kw["usage"], {"shape": "google_genai", "raw": um})
        # Positional counts remain today's netted fallback.
        self.assertEqual(kw["input_tokens"], 200)
        self.assertEqual(kw["output_tokens"], 200)
        self.assertEqual(kw["cached_tokens"], 800)
        json.dumps(kw["usage"])

    # ── 5. openai branch, no usable usage object → None path (synth fallback) ──
    def test_log_li_py_openai_branch_no_usage_block(self):
        # raw carries no usage object at all → helper returns None, log_sync
        # synthesises the fallback shape itself (usage kwarg stays None).
        resp = _FakeLIChatResponse(raw=NS(model="gpt-4o"))
        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIOpenAI(), resp, 0, "openai_call", None)

        kw = tp.captured
        self.assertIn("model", kw)          # row never lost
        self.assertIsNone(kw.get("usage"))  # None path intact → synth fallback

    # ── 6. fallback google (no usable metadata) still logs, usage=None ──
    def test_log_li_py_google_fallback_still_logs(self):
        resp = _FakeLIChatResponse(raw=NS())  # no usage_metadata
        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIGemini(), resp, 0, "gemini_call", None)

        kw = tp.captured
        self.assertIn("model", kw)  # row never lost
        self.assertIsNone(kw.get("usage"))
        self.assertEqual(kw["input_tokens"], 0)


class TestLlamaIndexAnthropicVerbatimUsage(unittest.TestCase):
    """Audit F4 (under-bill): the LlamaIndex Anthropic path (stream + non-stream)
    dropped cache_creation_input_tokens and never forwarded a verbatim usage
    block, so cache-WRITE tokens billed $0. Fix: forward
    ``usage={"shape":"anthropic_messages","raw":<usage>}`` (input_tokens stays
    cache-EXCLUSIVE) so writes are priced at the write rate. Mirrors the native
    anthropic stream path and the F1 google precedent. All fakes."""

    # ── 1. non-stream Message.usage with cache write → forwarded verbatim ──
    def test_non_stream_forwards_cache_write(self):
        usage = NS(input_tokens=1000, output_tokens=250,
                   cache_read_input_tokens=300, cache_creation_input_tokens=2000)
        resp = _FakeLIChatResponse(
            raw=NS(model="claude-3-5-sonnet-20241022", usage=usage),
            message=NS(additional_kwargs={}),
        )
        block = _li_anthropic_verbatim_usage_py(resp)
        self.assertEqual(block["shape"], "anthropic_messages")
        self.assertEqual(block["raw"]["input_tokens"], 1000)      # cache-EXCLUSIVE
        self.assertEqual(block["raw"]["output_tokens"], 250)
        self.assertEqual(block["raw"]["cache_read_input_tokens"], 300)
        self.assertEqual(block["raw"]["cache_creation_input_tokens"], 2000)
        json.dumps(block)  # background-POST serializable

    # ── 2. nested cache_creation (5m + 1h) → forwarded verbatim ──
    def test_non_stream_nested_cache_creation_forwarded(self):
        usage = NS(
            input_tokens=1000, output_tokens=250,
            cache_read_input_tokens=0, cache_creation_input_tokens=2000,
            cache_creation=NS(ephemeral_5m_input_tokens=1500,
                              ephemeral_1h_input_tokens=500),
        )
        resp = _FakeLIChatResponse(
            raw=NS(model="claude", usage=usage), message=NS(additional_kwargs={}),
        )
        block = _li_anthropic_verbatim_usage_py(resp)
        cc = block["raw"]["cache_creation"]
        self.assertEqual(cc["ephemeral_5m_input_tokens"], 1500)
        self.assertEqual(cc["ephemeral_1h_input_tokens"], 500)
        json.dumps(block)

    # ── 3. _log_li_py forwards block + reconciles positional (reads-only) ──
    def test_log_li_py_forwards_and_reconciles_positional(self):
        usage = NS(input_tokens=1000, output_tokens=250,
                   cache_read_input_tokens=300, cache_creation_input_tokens=2000)
        resp = _FakeLIChatResponse(
            raw=NS(model="claude-3-5-sonnet", usage=usage),
            message=NS(additional_kwargs={}),
        )
        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIAnthropic(), resp, 0, "claude_call", None)

        kw = tp.captured
        self.assertEqual(kw["provider"], "anthropic")
        self.assertEqual(kw["usage"]["shape"], "anthropic_messages")
        self.assertEqual(kw["usage"]["raw"]["cache_creation_input_tokens"], 2000)
        self.assertEqual(kw["input_tokens"], 1000)
        self.assertEqual(kw["output_tokens"], 250)
        # cached_tokens is reads-only; writes are billed via the verbatim block.
        self.assertEqual(kw["cached_tokens"], 300)

    # ── 4. stream: cache on early chunk, last chunk drops it → merged block ──
    def test_stream_merges_cache_from_early_chunk(self):
        start_usage = NS(
            input_tokens=1000, output_tokens=1,
            cache_read_input_tokens=300, cache_creation_input_tokens=2000,
            cache_creation=NS(ephemeral_5m_input_tokens=2000,
                              ephemeral_1h_input_tokens=0),
        )
        start_chunk = _FakeLIChatResponse(
            raw={"type": "message_start", "message": NS(usage=start_usage)},
            message=NS(additional_kwargs={"usage": {"input_tokens": 1000,
                                                    "output_tokens": 1}}),
        )
        # LlamaIndex's last chunk usage dict carries only input/output — no cache.
        delta_chunk = _FakeLIChatResponse(
            raw={"type": "message_delta", "usage": NS(output_tokens=250)},
            message=NS(additional_kwargs={"usage": {"input_tokens": 1000,
                                                    "output_tokens": 250}}),
        )
        acc = {}
        enforcer._merge_li_anthropic_stream_usage(acc, start_chunk)
        enforcer._merge_li_anthropic_stream_usage(acc, delta_chunk)
        self.assertEqual(acc["input_tokens"], 1000)
        self.assertEqual(acc["output_tokens"], 250)       # delta max over start's 1
        self.assertEqual(acc["cache_creation_input_tokens"], 2000)

        block = _li_anthropic_verbatim_usage_py(delta_chunk, stream_usage=acc)
        self.assertEqual(block["raw"]["cache_creation_input_tokens"], 2000)
        self.assertEqual(block["raw"]["cache_creation"]["ephemeral_5m_input_tokens"], 2000)
        self.assertEqual(block["raw"]["output_tokens"], 250)

        # Positional counts off the last chunk stay correct (input/output).
        _, inp, out, _cached = _extract_li_usage_py(_FakeLIAnthropic(), delta_chunk)
        self.assertEqual(inp, 1000)
        self.assertEqual(out, 250)

        # Full path: log_sync receives the write tokens + reads-only cached.
        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIAnthropic(), delta_chunk, 0,
                                "claude_call", None, acc)
        kw = tp.captured
        self.assertEqual(kw["usage"]["raw"]["cache_creation_input_tokens"], 2000)
        self.assertEqual(kw["input_tokens"], 1000)
        self.assertEqual(kw["output_tokens"], 250)
        self.assertEqual(kw["cached_tokens"], 300)

    # ── 5. stream last-chunk usage complete → no double-merge inflation ──
    def test_stream_no_double_merge_inflation(self):
        start_usage = NS(input_tokens=1000, output_tokens=1,
                         cache_read_input_tokens=300, cache_creation_input_tokens=2000)
        start_chunk = _FakeLIChatResponse(
            raw={"type": "message_start", "message": NS(usage=start_usage)},
        )
        # Newer anthropic echoes the full usage on message_delta too.
        delta_usage = NS(input_tokens=1000, output_tokens=250,
                         cache_read_input_tokens=300, cache_creation_input_tokens=2000)
        delta_chunk = _FakeLIChatResponse(
            raw={"type": "message_delta", "usage": delta_usage},
        )
        acc = {}
        enforcer._merge_li_anthropic_stream_usage(acc, start_chunk)
        enforcer._merge_li_anthropic_stream_usage(acc, delta_chunk)
        # max-merge, not sum: re-sent fields never inflate.
        self.assertEqual(acc["cache_creation_input_tokens"], 2000)
        self.assertEqual(acc["cache_read_input_tokens"], 300)
        self.assertEqual(acc["input_tokens"], 1000)
        self.assertEqual(acc["output_tokens"], 250)

    # ── 6. golden rule: serialization-breaking usage → no throw, row still logs ──
    def test_serialization_failure_falls_back_and_still_logs(self):
        usage = NS(input_tokens=1000, output_tokens=250,
                   cache_read_input_tokens=300, cache_creation_input_tokens=2000)
        usage.self_ref = usage  # circular → verbatim serialize fails, extraction unaffected
        resp = _FakeLIChatResponse(
            raw=NS(model="claude", usage=usage), message=NS(additional_kwargs={}),
        )
        # Helper degrades to None rather than raising or emitting a bad block.
        self.assertIsNone(_li_anthropic_verbatim_usage_py(resp))

        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIAnthropic(), resp, 0, "claude_call", None)
        kw = tp.captured
        self.assertIn("model", kw)          # row never lost
        self.assertIsNone(kw.get("usage"))  # fell back to positional
        self.assertEqual(kw["input_tokens"], 1000)
        self.assertEqual(kw["cached_tokens"], 300)

    # ── 7. golden rule: raising usage object never escapes ──
    def test_hostile_usage_never_raises(self):
        class _HostileRaw:
            model = "claude"

            @property
            def usage(self):
                raise RuntimeError("boom")

        resp = _FakeLIChatResponse(raw=_HostileRaw(), message=NS(additional_kwargs={}))
        self.assertIsNone(_li_anthropic_verbatim_usage_py(resp))

    def test_merge_hostile_chunk_never_raises(self):
        class _HostileChunk:
            @property
            def raw(self):
                raise RuntimeError("boom")

        acc = {}
        enforcer._merge_li_anthropic_stream_usage(acc, _HostileChunk())  # no throw
        self.assertEqual(acc, {})

    # ── 8. empty / absent usage → no block (positional fallback intact) ──
    def test_empty_usage_returns_none(self):
        self.assertIsNone(_li_anthropic_verbatim_usage_py(None))
        resp = _FakeLIChatResponse(
            raw=NS(model="claude", usage=NS(input_tokens=0, output_tokens=0)),
            message=NS(additional_kwargs={}),
        )
        self.assertIsNone(_li_anthropic_verbatim_usage_py(resp))


class TestLlamaIndexOpenAIVerbatimUsage(unittest.TestCase):
    """Audit PY-LI-OpenAI: the LlamaIndex OpenAI path forwarded no verbatim usage
    block. LlamaIndex's extractor surfaces only loose token ints and drops
    prompt_tokens_details, so cache-READ tokens vanished (all input billed at the
    full rate — over-bill) and the synthesised fallback would net an
    already-netted input a second time (latent double-subtract). Fix: forward
    ``usage={"shape":"openai_compatible_chat","raw":<usage>}`` (prompt_tokens
    cache-INCLUSIVE, details preserved) so the mapper nets cached once. Mirrors
    the F1 google / F4 anthropic precedent and Node's F1 openai branch. All fakes."""

    # ── 1. raw.usage with cache details → forwarded verbatim + netted positional ──
    def test_forwards_cache_details_and_nets_positional(self):
        usage = NS(prompt_tokens=100, completion_tokens=50,
                   prompt_tokens_details={"cached_tokens": 40})
        resp = _FakeLIChatResponse(raw=NS(model="gpt-4o", usage=usage))
        block = _li_openai_verbatim_usage_py(resp)
        self.assertEqual(block["shape"], "openai_compatible_chat")
        self.assertEqual(block["raw"]["prompt_tokens"], 100)          # cache-INCLUSIVE
        self.assertEqual(block["raw"]["completion_tokens"], 50)
        # Nested details preserved verbatim so the mapper can net cache reads.
        self.assertEqual(block["raw"]["prompt_tokens_details"]["cached_tokens"], 40)
        json.dumps(block)  # background-POST serializable

        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIOpenAI(), resp, 0, "openai_call", None)
        kw = tp.captured
        self.assertEqual(kw["provider"], "openai")
        self.assertEqual(kw["usage"]["shape"], "openai_compatible_chat")
        self.assertEqual(kw["usage"]["raw"]["prompt_tokens_details"]["cached_tokens"], 40)
        self.assertEqual(kw["input_tokens"], 60)   # netted: max(0, 100 - 40)
        self.assertEqual(kw["output_tokens"], 50)
        self.assertEqual(kw["cached_tokens"], 40)

    # ── 2. raw.usage without details → verbatim block, input full, cached 0 ──
    def test_forwards_without_details(self):
        usage = NS(prompt_tokens=100, completion_tokens=50)
        resp = _FakeLIChatResponse(raw=NS(model="gpt-4o", usage=usage))
        block = _li_openai_verbatim_usage_py(resp)
        self.assertEqual(block, {"shape": "openai_compatible_chat",
                                 "raw": {"prompt_tokens": 100, "completion_tokens": 50}})
        json.dumps(block)

        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIOpenAI(), resp, 0, "openai_call", None)
        kw = tp.captured
        self.assertEqual(kw["usage"]["raw"]["prompt_tokens"], 100)
        self.assertEqual(kw["input_tokens"], 100)   # no cache → no netting
        self.assertEqual(kw["output_tokens"], 50)
        self.assertEqual(kw["cached_tokens"], 0)

    # ── 3. no usage object at all → None (positional fallback intact) ──
    def test_no_usage_object_returns_none(self):
        self.assertIsNone(_li_openai_verbatim_usage_py(None))
        # raw present but carries no usage object.
        self.assertIsNone(_li_openai_verbatim_usage_py(_FakeLIChatResponse(raw=NS(model="gpt-4o"))))
        # LlamaIndex's loose token ints on additional_kwargs are NOT a usage
        # object — never fabricated into one.
        resp = _FakeLIChatResponse(
            raw=NS(model="gpt-4o"),
            message=NS(additional_kwargs={"prompt_tokens": 100, "completion_tokens": 50}),
        )
        self.assertIsNone(_li_openai_verbatim_usage_py(resp))

    # ── 4. all-zero usage → None (all-zero block worse than positional) ──
    def test_all_zero_usage_returns_none(self):
        resp = _FakeLIChatResponse(
            raw=NS(model="gpt-4o", usage=NS(prompt_tokens=0, completion_tokens=0))
        )
        self.assertIsNone(_li_openai_verbatim_usage_py(resp))

    # ── 5. golden rule: serialization-breaking usage → None, row still logs ──
    def test_serialization_failure_falls_back_and_still_logs(self):
        usage = NS(prompt_tokens=100, completion_tokens=50)
        usage.self_ref = usage  # circular → verbatim serialize fails
        resp = _FakeLIChatResponse(raw=NS(model="gpt-4o", usage=usage))
        self.assertIsNone(_li_openai_verbatim_usage_py(resp))

        tp = _CaptureTP()
        with mock.patch.object(enforcer, "get_client", return_value=tp), \
                mock.patch.object(enforcer, "get_current_session",
                                  return_value=_fake_session()):
            enforcer._log_li_py(_FakeLIOpenAI(), resp, 0, "openai_call", None)
        kw = tp.captured
        self.assertIn("model", kw)          # row never lost
        self.assertIsNone(kw.get("usage"))  # fell back to positional
        # Positional counts come from _extract_li_usage_py's raw.usage fallback.
        self.assertEqual(kw["input_tokens"], 100)
        self.assertEqual(kw["output_tokens"], 50)

    # ── 6. golden rule: raising usage object never escapes ──
    def test_hostile_usage_never_raises(self):
        class _HostileRaw:
            model = "gpt-4o"

            @property
            def usage(self):
                raise RuntimeError("boom")

        resp = _FakeLIChatResponse(raw=_HostileRaw())
        self.assertIsNone(_li_openai_verbatim_usage_py(resp))


if __name__ == "__main__":
    unittest.main()
