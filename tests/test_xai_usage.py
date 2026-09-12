"""Tests for xAI native SDK (`xai_sdk`) usage capture.

`xai_sdk` returns `usage` as a protobuf message (`xai.api.v1.usage_pb2`). The
generic `_as_dict` + json serialization can't read protobuf fields — it dumps
the message *class's* `__dict__` (DESCRIPTOR / __slots__ / ...), so the server
receives no real numbers and the call lands as 0 tokens / unmeasured.

`_extract_raw_usage` must convert the protobuf usage into a plain OpenAI-shaped
dict, for both non-streaming (result = Response) and streaming (result =
(Response, Chunk) tuple) paths.
"""
import unittest

from token_police.enforcer import _extract_raw_usage


class _FakeFieldDescriptor:
    """Mimics a protobuf FieldDescriptor — only `.name` is used."""
    def __init__(self, name):
        self.name = name


class _FakeProtoUsage:
    """Mimics an xai_sdk protobuf usage message: scalar fields exposed both as
    attributes and via ListFields(), like real protobuf messages."""
    def __init__(self, **fields):
        self._fields = fields
        for k, v in fields.items():
            setattr(self, k, v)

    def ListFields(self):
        # Real protobuf ListFields() returns only populated (non-default) fields.
        return [(_FakeFieldDescriptor(k), v) for k, v in self._fields.items()]


class _FakeXaiResponse:
    def __init__(self, usage):
        self.usage = usage


class TestXaiRawUsage(unittest.TestCase):
    def test_non_streaming_proto_usage_to_dict(self):
        resp = _FakeXaiResponse(
            _FakeProtoUsage(prompt_tokens=382, completion_tokens=66, total_tokens=448)
        )
        raw = _extract_raw_usage("xai", resp)
        self.assertEqual(
            raw, {"prompt_tokens": 382, "completion_tokens": 66, "total_tokens": 448}
        )
        # Must NOT be the protobuf class-metadata garbage.
        self.assertNotIn("DESCRIPTOR", raw)
        self.assertNotIn("__slots__", raw)

    def test_streaming_tuple_is_unwrapped(self):
        usage = _FakeProtoUsage(prompt_tokens=744, completion_tokens=95, total_tokens=839)
        # xai-sdk streams yield (Response, Chunk) tuples; latched `last` is the tuple.
        resp = _extract_raw_usage("xai", (_FakeXaiResponse(usage), object()))
        self.assertEqual(
            resp,
            {"prompt_tokens": 744, "completion_tokens": 95, "total_tokens": 839},
        )

    def test_carries_reasoning_and_cost_fields_when_present(self):
        resp = _FakeXaiResponse(
            _FakeProtoUsage(
                prompt_tokens=100,
                completion_tokens=50,
                reasoning_tokens=20,
                cost_in_usd_ticks=1234,
            )
        )
        raw = _extract_raw_usage("xai", resp)
        self.assertEqual(raw["prompt_tokens"], 100)
        self.assertEqual(raw["completion_tokens"], 50)
        self.assertEqual(raw["reasoning_tokens"], 20)
        self.assertEqual(raw["cost_in_usd_ticks"], 1234)

    def test_none_usage_returns_none(self):
        self.assertIsNone(_extract_raw_usage("xai", _FakeXaiResponse(None)))

    def test_none_result_returns_none(self):
        self.assertIsNone(_extract_raw_usage("xai", None))

    def test_non_protobuf_usage_falls_back_unchanged(self):
        # Backup-only-if-protobuf: a non-protobuf usage object (no ListFields)
        # is returned UNCHANGED so the existing _as_dict+json path handles it
        # exactly as before. Must not raise.
        class _Bare:
            pass

        bare = _Bare()
        resp = _FakeXaiResponse(bare)
        self.assertIs(_extract_raw_usage("xai", resp), bare)


if __name__ == "__main__":
    unittest.main()
