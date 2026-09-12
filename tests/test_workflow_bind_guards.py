"""
Regression tests — @tp.workflow dynamic argument binding type guards.

Contract: security regression — workflow bind type guards (assertions 2-13, 22).
Reference semantics: token-police-node/src/context.ts:651-677 (str-only
scalar binds, dict-only metadata merge; Python deliberately uses
isinstance(m, dict) which also rejects lists — the case-B vector).

GOLDEN RULE under test: hostile call-time argument types must never raise
into the customer's call stack before the wrapped function runs, and must
never silently corrupt the session; binding falls back to the static
decorator values. No network, no server — everything asserts via
tp.get_current_session() inside the wrapped function.
"""
import asyncio
import unittest
from unittest.mock import patch

import token_police as tp
from token_police.context import get_current_session


STATIC_META = {"static": "s"}


class TestHostileCallTimeMetadata(unittest.TestCase):
    """Contract assertions 2-7: hostile `metadata` call args through the
    decorated call — no exception from any SDK frame, exact resolved session
    metadata via get_current_session()."""

    def _run(self, **call_kwargs):
        captured = {}

        @tp.workflow(name="wf", metadata={"static": "s"})
        def fn(query, metadata=None):
            s = get_current_session()
            captured["metadata"] = dict(s.metadata)
            return "ok"

        result = fn("q", **call_kwargs)
        self.assertEqual(result, "ok")
        self.assertIn("metadata", captured, "wrapped function body did not run")
        return captured["metadata"]

    def test_str_metadata_ignored(self):
        # Assertion 3 — pre-fix: ValueError before the function body runs.
        meta = self._run(metadata="prod-tag")
        self.assertEqual(meta, STATIC_META)

    def test_list_metadata_not_corrupting(self):
        # Assertions 2 + 4 — pre-fix: dict.update treats ["t1", "t2"] as
        # key/value pairs and silently pollutes metadata with {"t": "2"}.
        meta = self._run(metadata=["t1", "t2"])
        self.assertEqual(meta, STATIC_META)
        self.assertNotIn("t", meta)

    def test_int_metadata_ignored(self):
        # Assertion 5.
        meta = self._run(metadata=123)
        self.assertEqual(meta, STATIC_META)

    def test_omitted_metadata_default_none(self):
        # Assertion 6 — current None handling preserved.
        meta = self._run()
        self.assertEqual(meta, STATIC_META)

    def test_dict_subclass_is_merged(self):
        # Assertion 7 — isinstance(dict) admits subclasses (parity with
        # Node's `typeof === "object"` admitting them).
        class MD(dict):
            pass

        meta = self._run(metadata=MD({"x": "1"}))
        self.assertEqual(meta, {"static": "s", "x": "1"})


class TestLiveBombDetonator(unittest.TestCase):
    """Contract assertion 22: live (non-mocked) hostile dict subclass that
    passes the isinstance(dict) gate but detonates inside dict.update —
    overriding BOTH __iter__ and keys() defeats the PyDict_Merge fast path on
    every tested CPython. Non-empty is mandatory (an empty Bomb is falsy and
    pre-fix short-circuits at the `or {}`). Pre-fix this CRASHES into the
    caller; post-fix the call-site try/except falls back to statics."""

    def test_bomb_metadata_falls_back_to_statics(self):
        class Bomb(dict):
            def keys(self):
                raise RuntimeError("boom")

            def __iter__(self):
                raise RuntimeError("boom")

        captured = {}

        @tp.workflow(name="wf-bomb", metadata={"static": "s"})
        def fn(query, metadata=None):
            s = get_current_session()
            captured["metadata"] = dict(s.metadata)
            return "ok"

        result = fn("q", metadata=Bomb({"x": 1}))
        self.assertEqual(result, "ok")
        self.assertIn("metadata", captured, "wrapped function body did not run")
        self.assertEqual(captured["metadata"], STATIC_META)


class TestCallSiteFallbackDetonator(unittest.TestCase):
    """Contract assertion 8: monkeypatch detonator proving the SYNC call-site
    try/except (context.py sync_wrapper) is live — _resolve_workflow_args is
    resolved module-global per call, so the patch is exercised even with
    fully VALID arguments."""

    def test_sync_callsite_falls_back_to_statics(self):
        captured = {}

        @tp.workflow(
            name="wf-det",
            user_id="static-u",
            paid_plan="static-p",
            session_id="s-det",
            metadata={"static": "s"},
        )
        def fn(query):
            captured["session"] = get_current_session()
            return "ok"

        with patch(
            "token_police.context._resolve_workflow_args",
            side_effect=RuntimeError("detonate"),
        ):
            result = fn("valid")

        self.assertEqual(result, "ok")
        self.assertIn("session", captured, "wrapped function body did not run")
        s = captured["session"]
        self.assertEqual(s.workflow_name, "wf-det")
        self.assertEqual(s.user_id, "static-u")
        self.assertEqual(s.paid_plan, "static-p")
        self.assertEqual(s.session_id, "s-det")
        self.assertEqual(dict(s.metadata), STATIC_META)


class TestNonStrScalarsRejected(unittest.TestCase):
    """Contract assertion 9: non-str dynamically bound scalars are rejected
    (static decorator values kept); valid strs still override."""

    def _decorated(self):
        captured = {}

        @tp.workflow(
            name="wf",
            user_id="static-u",
            paid_plan="static-p",
            session_id="s-static",
            metadata={"static": "s"},
        )
        def fn(query, user_id=None, paid_plan=None, workflow_name=None, session_id=None):
            captured["session"] = get_current_session()
            return "ok"

        return fn, captured

    def test_non_str_scalars_keep_statics(self):
        fn, captured = self._decorated()
        result = fn(
            "q",
            user_id=123,
            paid_plan=["free"],
            workflow_name=123,
            session_id=object(),
        )
        self.assertEqual(result, "ok")
        s = captured["session"]
        self.assertEqual(s.user_id, "static-u")
        self.assertEqual(s.paid_plan, "static-p")
        self.assertEqual(s.workflow_name, "wf")
        # Pre-fix _sanitize_session_id string-coerces object() to
        # "<object object at 0x...>" — a repr ships in place of the static id.
        self.assertEqual(s.session_id, "s-static")

    def test_non_str_camel_case_session_id_keeps_static(self):
        captured = {}

        @tp.workflow(name="wf", session_id="s-static", metadata={"static": "s"})
        def fn(query, sessionId=None):
            captured["session"] = get_current_session()
            return "ok"

        result = fn("q", sessionId=456)
        self.assertEqual(result, "ok")
        self.assertEqual(captured["session"].session_id, "s-static")

    def test_valid_str_still_overrides(self):
        fn, captured = self._decorated()
        result = fn("q", user_id="u9")
        self.assertEqual(result, "ok")
        self.assertEqual(captured["session"].user_id, "u9")


class TestSigBindFailure(unittest.TestCase):
    """Contract assertion 10: on a kwargs mismatch, sig.bind's failure never
    escapes an SDK frame — the decorated call raises exactly the TypeError the
    undecorated function raises (the customer's own error is correct behavior,
    not an SDK escape)."""

    def test_bind_failure_equals_undecorated_type_error(self):
        def plain(a):
            return a

        decorated = tp.workflow(name="wf")(plain)

        with self.assertRaises(TypeError) as undec:
            plain(b=1)
        with self.assertRaises(TypeError) as dec:
            decorated(b=1)

        self.assertIs(type(dec.exception), type(undec.exception))
        self.assertEqual(str(dec.exception), str(undec.exception))

        tb = dec.exception.__traceback__
        frame_names = []
        while tb is not None:
            frame_names.append(tb.tb_frame.f_code.co_name)
            tb = tb.tb_next
        self.assertNotIn("_resolve_workflow_args", frame_names)


class TestHostileStaticDecoratorMetadata(unittest.TestCase):
    """Contract assertion 11: misuse of the decorator itself
    (metadata="not-a-dict"; pre-fix `(metadata or {}).copy()` raises
    AttributeError) — proves the fallback constructor is itself total."""

    def test_non_dict_static_metadata_resolves_empty(self):
        captured = {}

        @tp.workflow(name="wf-static", metadata="not-a-dict")
        def fn(query):
            captured["metadata"] = dict(get_current_session().metadata)
            return "ok"

        result = fn("q")
        self.assertEqual(result, "ok")
        self.assertEqual(captured["metadata"], {})


class TestAsyncPath(unittest.TestCase):
    """Contract assertion 12: assertions 3, 8 (monkeypatch detonator against
    the ASYNC call site), and 9 repeated against an async def function, run
    via asyncio.run, identical expected outcomes."""

    def test_async_str_metadata_ignored(self):
        captured = {}

        @tp.workflow(name="wf", metadata={"static": "s"})
        async def fn(query, metadata=None):
            captured["metadata"] = dict(get_current_session().metadata)
            return "ok"

        result = asyncio.run(fn("q", metadata="prod-tag"))
        self.assertEqual(result, "ok")
        self.assertEqual(captured["metadata"], STATIC_META)

    def test_async_callsite_falls_back_to_statics(self):
        captured = {}

        @tp.workflow(
            name="wf-det",
            user_id="static-u",
            paid_plan="static-p",
            session_id="s-det",
            metadata={"static": "s"},
        )
        async def fn(query):
            captured["session"] = get_current_session()
            return "ok"

        with patch(
            "token_police.context._resolve_workflow_args",
            side_effect=RuntimeError("detonate"),
        ):
            result = asyncio.run(fn("valid"))

        self.assertEqual(result, "ok")
        self.assertIn("session", captured, "wrapped function body did not run")
        s = captured["session"]
        self.assertEqual(s.workflow_name, "wf-det")
        self.assertEqual(s.user_id, "static-u")
        self.assertEqual(s.paid_plan, "static-p")
        self.assertEqual(s.session_id, "s-det")
        self.assertEqual(dict(s.metadata), STATIC_META)

    def test_async_non_str_scalars_keep_statics(self):
        captured = {}

        @tp.workflow(
            name="wf",
            user_id="static-u",
            paid_plan="static-p",
            session_id="s-static",
            metadata={"static": "s"},
        )
        async def fn(query, user_id=None, paid_plan=None, workflow_name=None, session_id=None):
            captured["session"] = get_current_session()
            return "ok"

        result = asyncio.run(
            fn(
                "q",
                user_id=123,
                paid_plan=["free"],
                workflow_name=123,
                session_id=object(),
            )
        )
        self.assertEqual(result, "ok")
        s = captured["session"]
        self.assertEqual(s.user_id, "static-u")
        self.assertEqual(s.paid_plan, "static-p")
        self.assertEqual(s.workflow_name, "wf")
        self.assertEqual(s.session_id, "s-static")


class TestHappyPathPreserved(unittest.TestCase):
    """Contract assertion 13: intended dynamic binding for valid input is
    unchanged — valid strs + a dict bind exactly as pre-fix (call-time
    metadata keys win over static)."""

    def test_valid_dynamic_binding_unchanged(self):
        captured = {}

        @tp.workflow(
            name="wf",
            user_id="static-u",
            metadata={"static": "s", "k": "static-v"},
        )
        def fn(user_id, session_id, workflow_name, metadata=None):
            captured["session"] = get_current_session()
            return "ok"

        result = fn(
            user_id="u-dyn",
            session_id="sess-42",
            workflow_name="dyn-wf",
            metadata={"k": "call-v", "extra": "e"},
        )
        self.assertEqual(result, "ok")
        s = captured["session"]
        self.assertEqual(s.user_id, "u-dyn")
        self.assertEqual(s.session_id, "sess-42")
        self.assertEqual(s.workflow_name, "dyn-wf")
        self.assertEqual(
            dict(s.metadata), {"static": "s", "k": "call-v", "extra": "e"}
        )

    def test_bind_args_false_binding_disabled(self):
        captured = {}

        @tp.workflow(name="wf", user_id="static-u", bind_args=False)
        def fn(query, user_id=None):
            captured["session"] = get_current_session()
            return "ok"

        result = fn("q", user_id="u-dyn")
        self.assertEqual(result, "ok")
        self.assertEqual(captured["session"].user_id, "static-u")


class TestBareDecorator(unittest.TestCase):
    """Bare ``@tp.workflow`` (no parentheses) wraps the function instead of
    rebinding it to the inner `decorator` closure."""

    def test_bare_workflow_runs_and_returns(self):
        captured = {}

        @tp.workflow
        def fn(query):
            captured["session"] = get_current_session()
            return "ok"

        self.assertEqual(fn("q"), "ok")
        self.assertEqual(fn(query="q"), "ok")
        self.assertEqual(fn.__name__, "fn")
        self.assertIn("session", captured, "wrapped function body did not run")


if __name__ == "__main__":
    unittest.main()
