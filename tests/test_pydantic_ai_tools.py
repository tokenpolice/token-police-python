"""Tests for PydanticAI tool-span capture.

PydanticAI runs tools via `ToolManager.handle_call`, outside any patched/OTel
path (unless the app enables PydanticAI/logfire instrumentation), so tool
executions were invisible to TokenPolice. The fix wraps `handle_call` to emit a
TokenPolice `tool` row, correlated to the current session. These tests exercise
the wrapper with fakes — no real `pydantic_ai` dependency.
"""
import asyncio
import unittest

from token_police import context as ctx
from token_police.context import TPSession
from token_police.state import set_client
from token_police.enforcer import _make_pydantic_ai_tool_wrapper


class _FakeToolDef:
    def __init__(self, kind):
        self.kind = kind


class _FakeTool:
    def __init__(self, kind="function"):
        self.tool_def = _FakeToolDef(kind)


class _FakeCall:
    def __init__(self, tool_name, args='{"email": "a@b.com"}', tool_call_id="call_1"):
        self.tool_name = tool_name
        self.tool_call_id = tool_call_id
        self._args = args

    def args_as_json_str(self):
        return self._args


class _FakeToolManager:
    """Stands in for pydantic_ai ToolManager `self`."""
    def __init__(self, kind="function"):
        self.tools = {"get_customer_info": _FakeTool(kind)}


class _FakeClient:
    def __init__(self):
        self.calls = []

    def log_sync(self, **kwargs):
        self.calls.append(kwargs)


def _make_original(result=None, exc=None):
    async def original(self, call, allow_partial=False, wrap_validation_errors=True):
        if exc is not None:
            raise exc
        return result
    return original


class TestPydanticAITools(unittest.TestCase):
    def setUp(self):
        self.client = _FakeClient()
        set_client(self.client)
        self.session = TPSession(
            user_id="u1", paid_plan="pro", workflow_name="wf",
            trace_id="c" * 32, root_span_id="d" * 16,
        )
        self._token = ctx._current_session.set(self.session)

    def tearDown(self):
        ctx._current_session.reset(self._token)

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def test_function_tool_emits_row(self):
        wrapper = _make_pydantic_ai_tool_wrapper(_make_original(result="Premium customer"))
        mgr = _FakeToolManager(kind="function")
        out = self._run(wrapper(mgr, _FakeCall("get_customer_info"), False, True))

        self.assertEqual(out, "Premium customer")
        self.assertEqual(len(self.client.calls), 1)
        kw = self.client.calls[0]
        self.assertEqual(kw["span"]["span_kind"], "tool")
        self.assertEqual(kw["span"]["span_name"], "get_customer_info")
        self.assertEqual(kw["span"]["trace_id"], "c" * 32)
        self.assertEqual(kw["tool"]["name"], "get_customer_info")
        self.assertEqual(kw["tool"]["call_id"], "call_1")
        self.assertTrue(kw["tool"]["param_hash"])
        self.assertTrue(kw["tool"]["result_hash"])
        self.assertEqual(kw["call_outcome"]["status"], "success")
        # raw args/result never stored
        flat = repr(kw)
        self.assertNotIn("a@b.com", flat)
        self.assertNotIn("Premium customer", flat)

    def test_v1_signature_kwargs_passthrough(self):
        """pydantic_ai 1.x renamed the module (`_tool_manager` -> `tool_manager`)
        and changed the signature to
        `handle_call(call, *, approved=False, metadata=None, wrap_validation_errors=True)`
        — `allow_partial` is gone and the extras are keyword-only. The wrapper
        must accept/forward arbitrary kwargs and still emit a `tool` row
        (a real, non-partial execution)."""
        seen = {}

        async def original(self, call, *, approved=False, metadata=None,
                           wrap_validation_errors=True):
            seen.update(approved=approved, metadata=metadata,
                        wrap_validation_errors=wrap_validation_errors)
            return "Premium customer"

        wrapper = _make_pydantic_ai_tool_wrapper(original)
        mgr = _FakeToolManager(kind="function")
        out = self._run(wrapper(
            mgr, _FakeCall("get_customer_info"),
            approved=True, metadata={"m": 1}, wrap_validation_errors=False,
        ))

        self.assertEqual(out, "Premium customer")
        # kwargs forwarded untouched to the original
        self.assertEqual(seen, {"approved": True, "metadata": {"m": 1},
                                "wrap_validation_errors": False})
        # tool row still emitted
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["tool"]["name"], "get_customer_info")

    def test_v1_execute_tool_call_extractor(self):
        """1.x agent drives `ToolManager.execute_tool_call(validated, *, ...)`,
        where the ToolCallPart is nested as `validated.call`. The extractor must
        reach it and still emit a `tool` row."""
        class _Validated:
            def __init__(self, call):
                self.call = call

        async def original(self, validated, *, wrap_validation_errors=True):
            return "Premium customer"

        wrapper = _make_pydantic_ai_tool_wrapper(original, lambda v: v.call)
        mgr = _FakeToolManager(kind="function")
        validated = _Validated(_FakeCall("get_customer_info"))
        out = self._run(wrapper(mgr, validated, wrap_validation_errors=True))

        self.assertEqual(out, "Premium customer")
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["tool"]["name"], "get_customer_info")
        self.assertEqual(self.client.calls[0]["span"]["span_kind"], "tool")

    def test_output_tool_is_skipped(self):
        wrapper = _make_pydantic_ai_tool_wrapper(_make_original(result="final"))
        mgr = _FakeToolManager(kind="output")
        out = self._run(wrapper(mgr, _FakeCall("get_customer_info"), False, True))
        self.assertEqual(out, "final")  # still returns original result
        self.assertEqual(self.client.calls, [])

    def test_partial_validation_is_skipped(self):
        wrapper = _make_pydantic_ai_tool_wrapper(_make_original(result="x"))
        mgr = _FakeToolManager(kind="function")
        self._run(wrapper(mgr, _FakeCall("get_customer_info"), True, True))
        self.assertEqual(self.client.calls, [])

    def test_exception_propagates_and_emits_failed_row(self):
        boom = RuntimeError("tool blew up")
        wrapper = _make_pydantic_ai_tool_wrapper(_make_original(exc=boom))
        mgr = _FakeToolManager(kind="function")
        with self.assertRaises(RuntimeError):
            self._run(wrapper(mgr, _FakeCall("get_customer_info"), False, True))
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0]["call_outcome"]["status"], "failed")

    def test_emit_failure_never_breaks_the_call(self):
        # Broken client must not stop the tool from returning its result.
        class _BoomClient:
            def log_sync(self, **kwargs):
                raise RuntimeError("network down")
        set_client(_BoomClient())
        wrapper = _make_pydantic_ai_tool_wrapper(_make_original(result="ok"))
        mgr = _FakeToolManager(kind="function")
        out = self._run(wrapper(mgr, _FakeCall("get_customer_info"), False, True))
        self.assertEqual(out, "ok")


if __name__ == "__main__":
    unittest.main()
