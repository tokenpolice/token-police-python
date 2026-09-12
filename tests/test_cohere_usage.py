"""Manual token extraction for Cohere v2 chat.

Cohere chat capture relies on the OpenLLMetry cohere instrumentor, which only
supports `cohere <6` (no release covers cohere 6.x/7.x). So TokenPolice extracts
cohere chat usage manually (Mode C), version-resilient across cohere 5.x/6.x/7.x:
all use Cohere v2 `ClientV2`, with usage at `response.usage` (non-stream) and at
`message_end_event.delta.usage` (stream), tokens under
`usage.tokens.{input,output}_tokens` (billed_units as fallback).

These tests use fakes — no real `cohere` dependency.
"""
import unittest
from types import SimpleNamespace as NS

from token_police.enforcer import _extract_cohere_chat_usage, _chunk_usage


def _usage(inp, out, cached=None, billed=False):
    if billed:
        u = NS(tokens=None, billed_units=NS(input_tokens=inp, output_tokens=out),
               cached_tokens=cached)
    else:
        u = NS(tokens=NS(input_tokens=inp, output_tokens=out),
               billed_units=None, cached_tokens=cached)
    return u


class TestCohereUsage(unittest.TestCase):
    def test_non_stream_response_tokens(self):
        # Cohere returns floats for token counts (e.g. 27.0).
        resp = NS(usage=_usage(27.0, 14.0))
        model, inp, out, cached = _extract_cohere_chat_usage(resp)
        self.assertEqual((inp, out), (27, 14))

    def test_stream_message_end_event(self):
        # message-end event: usage lives under .delta.usage
        event = NS(usage=None, delta=NS(usage=_usage(100, 50)))
        _, inp, out, _ = _extract_cohere_chat_usage(event)
        self.assertEqual((inp, out), (100, 50))

    def test_billed_units_fallback(self):
        resp = NS(usage=_usage(11, 7, billed=True))
        _, inp, out, _ = _extract_cohere_chat_usage(resp)
        self.assertEqual((inp, out), (11, 7))

    def test_cached_tokens(self):
        resp = NS(usage=_usage(10, 5, cached=3))
        _, _, _, cached = _extract_cohere_chat_usage(resp)
        self.assertEqual(cached, 3)

    def test_dict_shape(self):
        resp = {"usage": {"tokens": {"input_tokens": 8, "output_tokens": 4}}}
        _, inp, out, _ = _extract_cohere_chat_usage(resp)
        self.assertEqual((inp, out), (8, 4))

    def test_fail_open_on_garbage(self):
        self.assertEqual(_extract_cohere_chat_usage(object()), ("", 0, 0, 0))
        self.assertEqual(_extract_cohere_chat_usage(None), ("", 0, 0, 0))

    def test_chunk_usage_latches_message_end_event(self):
        # _chunk_usage must return truthy for the message-end event (carries
        # .delta.usage) so the stream wrapper latches it as the usage chunk...
        end_event = NS(delta=NS(usage=_usage(100, 50)))
        self.assertIsNotNone(_chunk_usage(end_event))
        # ...and falsy for a content-delta event (delta without usage).
        content_event = NS(delta=NS(message=NS(content="hello")))
        self.assertIsNone(_chunk_usage(content_event))


if __name__ == "__main__":
    unittest.main()
