"""I3 — Gemini TTS reclass: operation=audio_tts + composition audio parts.

Gemini TTS uses the same generate_content surface as chat. Request-side
response_modalities=AUDIO and/or response candidates_tokens_details AUDIO
must reclass operation to audio_tts (check + log mirror). Typed-SDK object
response parts with inline_data audio must not fall to Tier-3 complete_response.
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace as NS
from unittest import mock

from token_police.composition import build_response_composition
from token_police.enforcer import (
    _flush_deferred_spans,
    _google_tts_intent_if_wanted,
    _is_google_audio_output,
    _stash_google_tts_hints,
    _wants_google_audio_out,
)
from token_police.context import TPSession
from token_police.span_kind import span_kind_for


class TestWantsGoogleAudioOut(unittest.TestCase):
    def test_config_response_modalities_audio(self):
        self.assertTrue(
            _wants_google_audio_out(
                {"model": "gemini-2.5-flash-preview-tts", "config": {"response_modalities": ["AUDIO"]}}
            )
        )

    def test_config_response_modalities_camel(self):
        self.assertTrue(
            _wants_google_audio_out({"config": {"responseModalities": ["AUDIO"]}})
        )

    def test_speech_config_present(self):
        self.assertTrue(
            _wants_google_audio_out({"config": {"speech_config": {"voice": "Kore"}}})
        )

    def test_pydantic_like_config_object(self):
        cfg = NS(response_modalities=["AUDIO"], speech_config=None)
        self.assertTrue(_wants_google_audio_out({"config": cfg}))

    def test_text_only_chat_false(self):
        self.assertFalse(
            _wants_google_audio_out(
                {"model": "gemini-2.5-flash", "config": {"max_output_tokens": 80}}
            )
        )
        self.assertFalse(_wants_google_audio_out({}))
        self.assertFalse(_wants_google_audio_out(None))  # type: ignore[arg-type]

    def test_hostile_config_fail_open(self):
        class Boom:
            @property
            def response_modalities(self):
                raise RuntimeError("nope")

        self.assertFalse(_wants_google_audio_out({"config": Boom()}))


class TestIsGoogleAudioOutput(unittest.TestCase):
    def test_camel_candidates_tokens_details(self):
        result = NS(
            usageMetadata=NS(
                candidatesTokensDetails=[NS(modality="AUDIO", tokenCount=42)]
            )
        )
        self.assertTrue(_is_google_audio_output(result))

    def test_snake_candidates_tokens_details(self):
        result = {
            "usage_metadata": {
                "candidates_tokens_details": [
                    {"modality": "AUDIO", "token_count": 10}
                ]
            }
        }
        self.assertTrue(_is_google_audio_output(result))

    def test_zero_audio_tokens_false(self):
        result = {
            "usageMetadata": {
                "candidatesTokensDetails": [{"modality": "AUDIO", "tokenCount": 0}]
            }
        }
        self.assertFalse(_is_google_audio_output(result))

    def test_text_only_false(self):
        result = {
            "usage_metadata": {
                "candidates_tokens_details": [
                    {"modality": "TEXT", "token_count": 20}
                ]
            }
        }
        self.assertFalse(_is_google_audio_output(result))

    def test_inline_audio_part(self):
        result = NS(
            candidates=[
                NS(
                    content=NS(
                        parts=[
                            NS(
                                inline_data=NS(mime_type="audio/pcm", data="abc"),
                                text=None,
                            )
                        ]
                    )
                )
            ]
        )
        self.assertTrue(_is_google_audio_output(result))

    def test_hostile_fail_open(self):
        class Boom:
            @property
            def usage_metadata(self):
                raise RuntimeError("x")

            @property
            def candidates(self):
                raise RuntimeError("y")

        self.assertFalse(_is_google_audio_output(Boom()))


class TestIntentAndSpanKind(unittest.TestCase):
    def test_intent_when_wanted(self):
        intent = _google_tts_intent_if_wanted(
            "google", {"config": {"response_modalities": ["AUDIO"]}}
        )
        self.assertEqual(intent, {"kind": "audio_tts"})

    def test_intent_none_for_chat(self):
        self.assertIsNone(
            _google_tts_intent_if_wanted("google", {"config": {"max_output_tokens": 10}})
        )
        self.assertIsNone(
            _google_tts_intent_if_wanted("openai", {"config": {"response_modalities": ["AUDIO"]}})
        )

    def test_span_kind_for_audio_tts(self):
        self.assertEqual(span_kind_for("audio_tts", "llm"), "tts")
        self.assertEqual(span_kind_for("chat", "llm"), "llm")


class TestStashAndFlush(unittest.TestCase):
    def test_stash_operation_from_request(self):
        session = TPSession(trace_id="a" * 32)
        _stash_google_tts_hints(
            session,
            "google",
            0,
            kwargs={"config": {"response_modalities": ["AUDIO"]}},
        )
        key = f"{session.trace_id}:0"
        self.assertEqual(session._pending_compositions[key]["operation"], "audio_tts")

    def test_stash_operation_from_response(self):
        session = TPSession(trace_id="b" * 32)
        result = {
            "usage_metadata": {
                "candidates_tokens_details": [
                    {"modality": "AUDIO", "token_count": 5}
                ],
                "prompt_token_count": 3,
                "candidates_token_count": 5,
            }
        }
        _stash_google_tts_hints(session, "google", 1, result=result)
        key = f"{session.trace_id}:1"
        self.assertEqual(session._pending_compositions[key]["operation"], "audio_tts")
        self.assertIn("usage_raw", session._pending_compositions[key])
        raw = session._pending_compositions[key]["usage_raw"]
        self.assertIn("candidates_tokens_details", raw)

    def test_flush_injects_operation_and_usage_raw(self):
        session = TPSession(trace_id="c" * 32)
        session._deferred_spans = [
            {
                "user_id": "u",
                "paid_plan": "free",
                "workflow_name": "w",
                "session_id": "",
                "model": "gemini-2.5-flash-preview-tts",
                "provider": "google",
                "input_tokens": 3,
                "output_tokens": 5,
                "cached_tokens": 0,
                "metadata": {},
                "span": {
                    "trace_id": session.trace_id,
                    "span_id": "d" * 16,
                    "parent_span_id": "",
                    "span_kind": "llm",
                    "span_name": "m",
                    "span_order": 0,
                },
                "usage": {
                    "shape": "google_genai",
                    "raw": {"prompt_tokens": 3, "completion_tokens": 5},
                },
            }
        ]
        session._pending_compositions = {
            f"{session.trace_id}:0": {
                "operation": "audio_tts",
                "usage_raw": {
                    "candidates_tokens_details": [
                        {"modality": "AUDIO", "token_count": 5}
                    ],
                    "prompt_token_count": 3,
                    "candidates_token_count": 5,
                },
                "response": [{"role": "assistant", "type": "audio"}],
            }
        }
        logged = {}

        class FakeClient:
            def log_sync(self, **kwargs):
                logged.update(kwargs)

        with mock.patch("token_police.enforcer.get_client", return_value=FakeClient()):
            _flush_deferred_spans(session)

        self.assertEqual(logged.get("operation"), "audio_tts")
        self.assertEqual(logged["usage"]["shape"], "google_genai")
        self.assertIn("candidates_tokens_details", logged["usage"]["raw"])
        self.assertEqual(logged.get("response_composition")[0]["type"], "audio")

    def test_flush_text_chat_unchanged(self):
        session = TPSession(trace_id="e" * 32)
        session._deferred_spans = [
            {
                "user_id": "u",
                "paid_plan": "free",
                "workflow_name": "w",
                "session_id": "",
                "model": "gemini-2.5-flash",
                "provider": "google",
                "input_tokens": 1,
                "output_tokens": 2,
                "cached_tokens": 0,
                "metadata": {},
                "span": {
                    "trace_id": session.trace_id,
                    "span_id": "f" * 16,
                    "parent_span_id": "",
                    "span_kind": "llm",
                    "span_name": "m",
                    "span_order": 0,
                },
                "usage": {"shape": "google_genai", "raw": {"prompt_tokens": 1}},
            }
        ]
        session._pending_compositions = {
            f"{session.trace_id}:0": {
                "response": [{"role": "assistant", "type": "text", "length": 1, "hash": "x"}],
            }
        }
        logged = {}

        class FakeClient:
            def log_sync(self, **kwargs):
                logged.update(kwargs)

        with mock.patch("token_police.enforcer.get_client", return_value=FakeClient()):
            _flush_deferred_spans(session)

        # No operation key → log_sync defaults to chat.
        self.assertNotIn("operation", logged)
        self.assertEqual(logged["usage"]["raw"], {"prompt_tokens": 1})


class TestCompositionAudioParts(unittest.TestCase):
    def test_object_inline_data_audio(self):
        resp = NS(
            candidates=[
                NS(
                    content=NS(
                        parts=[
                            NS(
                                text=None,
                                function_call=None,
                                functionCall=None,
                                inline_data=NS(mime_type="audio/L16;codec=pcm", data="AQID"),
                                inlineData=None,
                            )
                        ]
                    )
                )
            ]
        )
        comp = build_response_composition("google", resp)
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp[0]["role"], "assistant")
        self.assertEqual(comp[0]["type"], "audio")
        # Must not be Tier-3 complete_response
        self.assertNotEqual(comp[0].get("type"), "complete_response")

    def test_object_inline_data_camel(self):
        resp = NS(
            candidates=[
                NS(
                    content=NS(
                        parts=[
                            NS(
                                text=None,
                                function_call=None,
                                functionCall=None,
                                inline_data=None,
                                inlineData=NS(mimeType="audio/mpeg", data="xx"),
                            )
                        ]
                    )
                )
            ]
        )
        comp = build_response_composition("google", resp)
        self.assertEqual(comp[0]["type"], "audio")

    def test_dict_path_still_works(self):
        resp = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inline_data": {"mime_type": "audio/pcm", "data": "x"}}
                        ]
                    }
                }
            ]
        }
        comp = build_response_composition("google", resp)
        self.assertEqual(comp[0]["type"], "audio")

    def test_text_chat_still_text(self):
        resp = NS(
            candidates=[
                NS(
                    content=NS(
                        parts=[
                            NS(
                                text="hello",
                                function_call=None,
                                functionCall=None,
                                inline_data=None,
                                inlineData=None,
                            )
                        ]
                    )
                )
            ]
        )
        comp = build_response_composition("google", resp)
        self.assertEqual(comp[0]["type"], "text")
        self.assertEqual(comp[0]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
