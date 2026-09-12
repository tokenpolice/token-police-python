"""The Bedrock InvokeModel embedding handlers drain ``response['body']`` to read
telemetry, then restore a fresh stream so customer code can still ``.read()`` it.

Golden rule: the customer owns the response. Once the stream has been drained the
restore MUST run — even if parsing the telemetry blows up midway — otherwise the
customer's app sees a permanently-consumed body. If the read ITSELF fails there is
nothing to restore and the call must still not crash.

Harness mirrors tests/test_reroute_suppress_bedrock_embedding.py: ``tp.init`` with
check/log mocked, drive the handler directly inside a ``tp.session`` context, and
always clean up.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as tp_state


_VALID = b'{"embedding": [0.1, 0.2], "inputTextTokenCount": 3}'
_BADJSON = b"not-json{{{"


class _SyncBody:
    """Sync StreamingBody stand-in: one-shot .read() that records consumption."""

    def __init__(self, data):
        self._data = data
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        return self._data


class _SyncRaisingBody:
    def __init__(self):
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        raise IOError("stream gone")


class _AsyncBody:
    def __init__(self, data):
        self._data = data
        self.read_calls = 0

    async def read(self):
        self.read_calls += 1
        return self._data


class _AsyncRaisingBody:
    def __init__(self):
        self.read_calls = 0

    async def read(self):
        self.read_calls += 1
        raise IOError("stream gone")


def _init():
    """Init a client with check/log mocked to 'allowed' so the handler runs its
    full read/restore/log path without ever blocking."""
    client = tp.init(api_key="tp_sk_test_body", firewall="off")
    client.check_sync = MagicMock(return_value={"status": "allowed"})
    client.check = AsyncMock(return_value={"status": "allowed"})
    client.log_sync = MagicMock()
    return client


def _args(body):
    # botocore _make_api_call(self, operation_name, api_params)
    self_obj = MagicMock()
    api_params = {"modelId": "amazon.titan-embed-text-v1",
                  "body": b'{"inputText": "hello"}'}
    response = {"body": body, "ResponseMetadata": {}}
    return (self_obj, "InvokeModel", api_params), response


class TestBedrockBodyRestoreSync(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()
        _init()

    def tearDown(self):
        tp_state.reset_pack()
        tp.uninstrument()

    def test_restore_runs_when_parse_fails(self):
        body = _SyncBody(_BADJSON)
        args, response = _args(body)
        original = MagicMock(return_value=response)
        with tp.session(name="t"):
            out = _enforcer._handle_bedrock_embedding_sync(original, args, {})
        # Read consumed the original stream once...
        self.assertEqual(body.read_calls, 1)
        # ...but the body was restored to a fresh, readable stream even though
        # the JSON parse failed — customer can still consume it.
        self.assertIsNot(out["body"], body)
        self.assertEqual(out["body"].read(), _BADJSON)

    def test_restore_runs_on_happy_path(self):
        body = _SyncBody(_VALID)
        args, response = _args(body)
        original = MagicMock(return_value=response)
        with tp.session(name="t"):
            out = _enforcer._handle_bedrock_embedding_sync(original, args, {})
        self.assertIsNot(out["body"], body)
        self.assertEqual(out["body"].read(), _VALID)

    def test_read_failure_does_not_crash_and_skips_restore(self):
        body = _SyncRaisingBody()
        args, response = _args(body)
        original = MagicMock(return_value=response)
        with tp.session(name="t"):
            # Must NOT raise into customer code.
            out = _enforcer._handle_bedrock_embedding_sync(original, args, {})
        self.assertEqual(body.read_calls, 1)
        # Nothing to restore — body left as-is (no fresh stream substituted).
        self.assertIs(out["body"], body)


class TestBedrockBodyRestoreAsync(unittest.TestCase):
    def setUp(self):
        tp_state.reset_pack()
        _init()

    def tearDown(self):
        tp_state.reset_pack()
        tp.uninstrument()

    def test_restore_runs_when_parse_fails(self):
        body = _AsyncBody(_BADJSON)
        args, response = _args(body)
        original = AsyncMock(return_value=response)

        async def _run():
            with tp.session(name="t"):
                return await _enforcer._handle_bedrock_embedding_async(original, args, {})

        out = asyncio.run(_run())
        self.assertEqual(body.read_calls, 1)
        self.assertIsNot(out["body"], body)
        self.assertEqual(out["body"].read(), _BADJSON)

    def test_restore_runs_on_happy_path(self):
        body = _AsyncBody(_VALID)
        args, response = _args(body)
        original = AsyncMock(return_value=response)

        async def _run():
            with tp.session(name="t"):
                return await _enforcer._handle_bedrock_embedding_async(original, args, {})

        out = asyncio.run(_run())
        self.assertIsNot(out["body"], body)
        self.assertEqual(out["body"].read(), _VALID)

    def test_read_failure_does_not_crash_and_skips_restore(self):
        body = _AsyncRaisingBody()
        args, response = _args(body)
        original = AsyncMock(return_value=response)

        async def _run():
            with tp.session(name="t"):
                return await _enforcer._handle_bedrock_embedding_async(original, args, {})

        out = asyncio.run(_run())
        self.assertEqual(body.read_calls, 1)
        self.assertIs(out["body"], body)


if __name__ == "__main__":
    unittest.main()
