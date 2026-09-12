"""Tests for the OpenAI TTS SSE tap fallback (sync + async).

Invariant under test: the tap must never hand the customer an empty rebuilt
body when the provider actually sent bytes. If the per-frame audio field is
renamed/moved so extraction yields nothing, the tap falls back to the original
raw SSE bytes verbatim (original content-type preserved), while still attaching
the captured usage so metering keeps working.
"""
import asyncio
import base64
import json
import unittest

from token_police.enforcer import (
    _maybe_tap_openai_tts_sse_sync,
    _maybe_tap_openai_tts_sse_async,
    _parse_openai_tts_sse_buffer,
)


_DONE_USAGE = {"input_tokens": 11, "output_tokens": 0, "total_tokens": 11}


def _b64(s: str) -> str:
    return base64.b64encode(s.encode("utf-8")).decode("ascii")


def _frame(obj) -> bytes:
    return b"data: " + json.dumps(obj).encode("utf-8") + b"\n\n"


class _FakeBinaryResponse:
    """Minimal stand-in for openai's HttpxBinaryResponseContent: `.read()` /
    `.aread()` return the raw SSE bytes once, plus a `.headers` mapping so the
    wrapper can delegate content-type."""

    def __init__(self, raw: bytes, content_type: str = "text/event-stream"):
        self._raw = raw
        self.status_code = 200
        self.headers = {"content-type": content_type, "x-request-id": "req_123"}

    def read(self):
        return self._raw

    async def aread(self):
        return self._raw

    def close(self):
        pass


_KWARGS = {"stream_format": "sse", "model": "gpt-4o-mini-tts", "voice": "alloy"}


def _renamed_field_sse() -> bytes:
    # Provider renamed the per-frame audio field: `delta` instead of `audio`.
    return (
        _frame({"type": "speech.audio.delta", "delta": _b64("AUDIO_CHUNK_1")})
        + _frame({"type": "speech.audio.delta", "delta": _b64("AUDIO_CHUNK_2")})
        + _frame({"type": "speech.audio.done", "usage": _DONE_USAGE})
        + b"data: [DONE]\n\n"
    )


def _wellformed_sse():
    c1, c2 = "AUDIO_CHUNK_1", "AUDIO_CHUNK_2"
    sse = (
        _frame({"type": "speech.audio.delta", "audio": _b64(c1)})
        + _frame({"type": "speech.audio.delta", "audio": _b64(c2)})
        + _frame({"type": "speech.audio.done", "usage": _DONE_USAGE})
        + b"data: [DONE]\n\n"
    )
    return sse, (c1 + c2).encode("utf-8")


class TestTtsSseTapFallback(unittest.TestCase):
    def test_prefix_evidence_renamed_field_extracts_no_audio(self):
        # Documents the defect root: parsing succeeds, usage is recovered, but
        # zero audio is extracted. Pre-fix, that empty buffer became the whole
        # response body (silent data loss). Post-fix the tap falls back instead.
        audio, usage = _parse_openai_tts_sse_buffer(_renamed_field_sse())
        self.assertEqual(len(audio), 0)
        self.assertEqual(usage, _DONE_USAGE)

    def test_regression_sync_falls_back_to_original_bytes(self):
        raw = _renamed_field_sse()
        original = _FakeBinaryResponse(raw)
        out = _maybe_tap_openai_tts_sse_sync("openai", "audio_tts", _KWARGS, original)
        # Body equals the original raw SSE bytes verbatim (NOT empty).
        self.assertEqual(out.read(), raw)
        self.assertTrue(len(out.read()) > 0)
        # Original content-type preserved (delegated to the original response).
        self.assertEqual(out.headers["content-type"], "text/event-stream")
        # Captured usage still attached for metering.
        self.assertEqual(out._tp_captured_usage, _DONE_USAGE)

    def test_regression_async_falls_back_to_original_bytes(self):
        raw = _renamed_field_sse()

        async def _run():
            out = await _maybe_tap_openai_tts_sse_async(
                "openai", "audio_tts", _KWARGS, _FakeBinaryResponse(raw)
            )
            return out, await out.aread()

        out, body = asyncio.run(_run())
        self.assertEqual(body, raw)
        self.assertEqual(out.headers["content-type"], "text/event-stream")
        self.assertEqual(out._tp_captured_usage, _DONE_USAGE)

    def test_happy_path_sync_rebuilds_decoded_audio(self):
        sse, expected_audio = _wellformed_sse()
        out = _maybe_tap_openai_tts_sse_sync(
            "openai", "audio_tts", _KWARGS, _FakeBinaryResponse(sse)
        )
        self.assertEqual(out.read(), expected_audio)
        self.assertEqual(out._tp_captured_usage, _DONE_USAGE)

    def test_happy_path_async_rebuilds_decoded_audio(self):
        sse, expected_audio = _wellformed_sse()

        async def _run():
            out = await _maybe_tap_openai_tts_sse_async(
                "openai", "audio_tts", _KWARGS, _FakeBinaryResponse(sse)
            )
            return out, await out.aread()

        out, body = asyncio.run(_run())
        self.assertEqual(body, expected_audio)
        self.assertEqual(out._tp_captured_usage, _DONE_USAGE)

    def test_empty_everything_unchanged(self):
        # Genuinely empty 200: raw bytes empty AND audio empty -> prior behavior,
        # no throw, empty body.
        out = _maybe_tap_openai_tts_sse_sync(
            "openai", "audio_tts", _KWARGS, _FakeBinaryResponse(b"")
        )
        self.assertEqual(out.read(), b"")

    def test_malformed_json_frames_skipped(self):
        sse = (
            b"data: {not valid json\n\n"
            + _frame({"type": "speech.audio.delta", "audio": _b64("GOOD")})
            + _frame({"type": "speech.audio.done", "usage": _DONE_USAGE})
        )
        out = _maybe_tap_openai_tts_sse_sync(
            "openai", "audio_tts", _KWARGS, _FakeBinaryResponse(sse)
        )
        # Malformed frame skipped, the valid audio frame still extracted.
        self.assertEqual(out.read(), b"GOOD")
        self.assertEqual(out._tp_captured_usage, _DONE_USAGE)


if __name__ == "__main__":
    unittest.main()
