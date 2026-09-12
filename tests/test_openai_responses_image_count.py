"""Part 2 + I2 residual — Responses-API image handling.

Chat-span usage no longer injects image_output_count (that produced partial
text-only pricing). Child image spans carry openai_images + gpt-image-*
attribution. Composition still emits assistant/image for image_generation_call.
"""
from __future__ import annotations

import base64
import struct
import types
import unittest
import zlib
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from token_police.composition import build_response_composition
from token_police.enforcer import (
    _extract_raw_usage,
    _extract_responses_image_tool_config,
    _list_responses_image_calls,
    _log_responses_image_children,
)


class _NS:
    """Minimal attribute bag (dict-like via attrs)."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _make_png(w: int, h: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + (b"\xff\x00\x00" * w) for _ in range(h))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


PNG_64X48_B64 = base64.b64encode(_make_png(64, 48)).decode("ascii")


class TestExtractRawUsageNoImageCount(unittest.TestCase):
    """Part 2: chat raw usage must not carry image_output_count."""

    def test_nonstream_with_images_returns_plain_usage(self):
        usage = {"input_tokens": 50, "output_tokens": 10}
        resp = _NS(
            usage=usage,
            output=[
                _NS(type="image_generation_call", id="ig_1", result="<b64>"),
                _NS(
                    type="message",
                    content=[_NS(type="output_text", text="Here is your image.")],
                ),
            ],
        )
        raw = _extract_raw_usage("openai_responses", resp)
        self.assertIs(raw, usage)
        self.assertNotIn("image_output_count", raw)

    def test_stream_completed_event_plain_usage(self):
        usage = {"input_tokens": 80, "output_tokens": 20}
        completed = _NS(
            type="response.completed",
            response=_NS(
                usage=usage,
                output=[
                    {"type": "image_generation_call", "id": "ig_2"},
                    {"type": "message", "content": [{"type": "output_text", "text": "ok"}]},
                ],
            ),
        )
        raw = _extract_raw_usage("openai_responses", completed)
        self.assertIs(raw, usage)
        self.assertNotIn("image_output_count", raw)

    def test_text_only_unchanged(self):
        usage = {"input_tokens": 10, "output_tokens": 5}
        resp = _NS(
            usage=usage,
            output=[
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
        )
        raw = _extract_raw_usage("openai_responses", resp)
        self.assertIs(raw, usage)

    def test_does_not_mutate_customer_usage(self):
        usage = {"input_tokens": 9, "output_tokens": 3}
        resp = {
            "usage": usage,
            "output": [{"type": "image_generation_call", "id": "ig_1"}],
        }
        raw = _extract_raw_usage("openai_responses", resp)
        self.assertIs(raw, usage)
        self.assertNotIn("image_output_count", usage)

    def test_none_result(self):
        self.assertIsNone(_extract_raw_usage("openai_responses", None))


class TestListAndToolConfig(unittest.TestCase):
    def test_tool_config_defaults(self):
        cfg = _extract_responses_image_tool_config((), {})
        self.assertEqual(cfg["model"], "gpt-image-1")
        self.assertEqual(cfg["size"], "")
        self.assertEqual(cfg["quality"], "")

    def test_tool_config_from_kwargs(self):
        cfg = _extract_responses_image_tool_config(
            (),
            {
                "tools": [
                    {
                        "type": "image_generation",
                        "model": "gpt-image-1-mini",
                        "size": "1024x1536",
                        "quality": "high",
                    }
                ]
            },
        )
        self.assertEqual(cfg["model"], "gpt-image-1-mini")
        self.assertEqual(cfg["size"], "1024x1536")
        self.assertEqual(cfg["quality"], "high")

    def test_tool_config_auto_size_quality_cleared(self):
        cfg = _extract_responses_image_tool_config(
            ({"tools": [{"type": "image_generation", "size": "auto", "quality": "auto"}]},),
            {},
        )
        self.assertEqual(cfg["size"], "")
        self.assertEqual(cfg["quality"], "")

    def test_list_skips_failed(self):
        resp = {
            "output": [
                {"type": "image_generation_call", "status": "failed", "id": "f"},
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ok",
                    "result": "abc",
                },
            ]
        }
        items = _list_responses_image_calls(resp)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["id"], "ok")

    def test_list_from_completed_event(self):
        completed = _NS(
            type="response.completed",
            response=_NS(
                output=[
                    _NS(
                        type="image_generation_call",
                        status="completed",
                        id="ig",
                        result="x",
                    )
                ]
            ),
        )
        self.assertEqual(len(_list_responses_image_calls(completed)), 1)

    def test_list_boom_fail_open(self):
        class _Boom:
            @property
            def output(self):
                raise RuntimeError("boom")

        self.assertEqual(_list_responses_image_calls(_Boom()), [])


class TestLogResponsesImageChildren(unittest.TestCase):
    def _session(self):
        s = MagicMock()
        s.user_id = "u"
        s.paid_plan = "free"
        s.workflow_name = "wf"
        s.session_id = "sid"
        s.metadata = {}
        s.trace_id = "t" * 32
        s.next_span_order = MagicMock(side_effect=[10, 11, 12])
        return s

    def test_emits_one_child_per_image(self):
        session = self._session()
        tp = MagicMock()
        resp = {
            "output": [
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ig1",
                    "result": PNG_64X48_B64,
                },
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ig2",
                    "result": PNG_64X48_B64,
                },
            ]
        }
        kwargs = {
            "tools": [
                {
                    "type": "image_generation",
                    "size": "1024x1024",
                    "quality": "medium",
                }
            ]
        }
        with patch("token_police.enforcer.get_client", return_value=tp):
            with patch(
                "token_police.enforcer.manual_span_ids",
                return_value={
                    "trace_id": "t" * 32,
                    "span_id": "s" * 16,
                    "parent_span_id": "p" * 16,
                },
            ):
                _log_responses_image_children(
                    session,
                    (),
                    kwargs,
                    resp,
                    datetime.now(timezone.utc),
                )
        self.assertEqual(tp.log_sync.call_count, 2)
        first = tp.log_sync.call_args_list[0].kwargs
        self.assertEqual(first["model"], "gpt-image-1")
        self.assertEqual(first["provider"], "openai")
        self.assertEqual(first["operation"], "image_gen")
        usage = first["usage"]
        self.assertEqual(usage["shape"], "openai_images")
        self.assertEqual(usage["items"]["images_generated"], 1)
        self.assertEqual(usage["items"]["image_quality"], "medium")
        # Request size wins over binary dims.
        self.assertEqual(usage["items"]["image_size"], "1024x1024")

    # ── Layer 3: dims echoed on the output item ────────────────────────
    # The image_generation_call item ships only {id,result,status,type} today, so
    # these paths are dormant — but openai-python sets extra="allow", so the
    # moment OpenAI echoes the server-resolved dims we capture them and the
    # server can price the exact tier instead of estimating it.
    def _items_for(self, item_extra, tools=None):
        session = self._session()
        tp = MagicMock()
        resp = {
            "output": [
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "id": "ig1",
                    "result": PNG_64X48_B64,
                    **item_extra,
                }
            ]
        }
        kwargs = {"tools": tools if tools is not None else [{"type": "image_generation"}]}
        with patch("token_police.enforcer.get_client", return_value=tp):
            with patch(
                "token_police.enforcer.manual_span_ids",
                return_value={
                    "trace_id": "t" * 32,
                    "span_id": "s" * 16,
                    "parent_span_id": "p" * 16,
                },
            ):
                _log_responses_image_children(
                    session, (), kwargs, resp, datetime.now(timezone.utc),
                )
        return tp.log_sync.call_args_list[0].kwargs["usage"]["items"]

    def test_item_quality_captured_when_request_omits_it(self):
        self.assertEqual(self._items_for({"quality": "medium"})["image_quality"], "medium")

    def test_item_dims_win_over_request_tool_config(self):
        # The request asked for 'auto'/nothing; the item reports what was produced.
        items = self._items_for(
            {"quality": "high", "size": "1536x1024"},
            tools=[{"type": "image_generation", "quality": "auto", "size": "1024x1024"}],
        )
        self.assertEqual(items["image_quality"], "high")
        self.assertEqual(items["image_size"], "1536x1024")

    def test_item_auto_is_unobserved_not_a_value(self):
        # "" means "not observed" — do not invent a default quality/size
        # as approximated — passing 'auto' through would contradict every tier.
        items = self._items_for({"quality": "auto", "size": "auto"})
        self.assertEqual(items["image_quality"], "")
        self.assertEqual(items["image_size"], "64x48")  # falls through to b64 header

    def test_falls_back_to_tool_config_when_item_silent(self):
        items = self._items_for(
            {}, tools=[{"type": "image_generation", "quality": "low", "size": "1024x1536"}]
        )
        self.assertEqual(items["image_quality"], "low")
        self.assertEqual(items["image_size"], "1024x1536")

    def test_non_string_item_dim_ignored(self):
        items = self._items_for({"quality": {"bogus": 1}, "size": 1024})
        self.assertEqual(items["image_quality"], "")

    def test_object_shaped_item_dims(self):
        # Real openai-python returns pydantic models, not dicts — _attr must read
        # the extra fields off the object form too.
        session = self._session()
        tp = MagicMock()
        resp = {
            "output": [
                _NS(
                    type="image_generation_call",
                    status="completed",
                    id="ig1",
                    result=PNG_64X48_B64,
                    quality="medium",
                    size="1536x1024",
                )
            ]
        }
        with patch("token_police.enforcer.get_client", return_value=tp):
            with patch(
                "token_police.enforcer.manual_span_ids",
                return_value={
                    "trace_id": "t" * 32,
                    "span_id": "s" * 16,
                    "parent_span_id": "p" * 16,
                },
            ):
                _log_responses_image_children(
                    session, (), {"tools": [{"type": "image_generation"}]},
                    resp, datetime.now(timezone.utc),
                )
        items = tp.log_sync.call_args_list[0].kwargs["usage"]["items"]
        self.assertEqual(items["image_quality"], "medium")
        self.assertEqual(items["image_size"], "1536x1024")

    def test_text_only_no_log(self):
        session = self._session()
        tp = MagicMock()
        resp = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        }
        with patch("token_police.enforcer.get_client", return_value=tp):
            _log_responses_image_children(
                session, (), {}, resp, datetime.now(timezone.utc)
            )
        tp.log_sync.assert_not_called()

    def test_binary_size_when_request_omitted(self):
        session = self._session()
        tp = MagicMock()
        resp = {
            "output": [
                {
                    "type": "image_generation_call",
                    "status": "completed",
                    "result": PNG_64X48_B64,
                }
            ]
        }
        with patch("token_police.enforcer.get_client", return_value=tp):
            with patch(
                "token_police.enforcer.manual_span_ids",
                return_value={
                    "trace_id": "t" * 32,
                    "span_id": "s" * 16,
                    "parent_span_id": "p" * 16,
                },
            ):
                _log_responses_image_children(
                    session, (), {"tools": [{"type": "image_generation"}]},
                    resp, datetime.now(timezone.utc),
                )
        usage = tp.log_sync.call_args.kwargs["usage"]
        self.assertEqual(usage["items"]["image_size"], "64x48")

    def test_no_client_fail_open(self):
        session = self._session()
        with patch("token_police.enforcer.get_client", return_value=None):
            # Must not raise
            _log_responses_image_children(
                session,
                (),
                {},
                {"output": [{"type": "image_generation_call", "result": "x"}]},
                datetime.now(timezone.utc),
            )


class TestResponsesCompositionImage(unittest.TestCase):
    def test_image_generation_call_emits_image_entry(self):
        resp = {
            "output": [
                {"type": "image_generation_call", "id": "ig_1", "result": "<b64>"},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "done"}],
                },
            ]
        }
        entries = build_response_composition("openai_responses", resp)
        types_ = [e.get("type") for e in entries]
        self.assertIn("image", types_)


if __name__ == "__main__":
    unittest.main()
