"""Pending tool-call stash must be shared across concurrent asyncio tasks.

OpenAI Agents (and any concurrent @tp.tool path) runs same-turn tools in separate
asyncio Tasks. Each Task gets a copied context. The stash must live on the shared
session object (or, unscoped, on a list mutated in place) so sibling tasks see
each other's pops and receive distinct provider tool_call_ids.

The old copy-on-write ContextVar pop was wrong: it isolated removals per task so
two concurrent same-name tools both got id A.
"""
import asyncio
import contextvars
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import context as ctx
from token_police.context import TPSession


def _run_coro(coro):
    """Drive a coroutine on a private loop WITHOUT asyncio.run().

    asyncio.run unsets the main-thread event loop, breaking later
    get_event_loop()-based tests (see test_anthropic_async_stream_check).
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _bind_session(session=None):
    session = session or TPSession()
    token = ctx._current_session.set(session)
    return session, token


def test_concurrent_same_name_pops_distinct_ids_bound_session():
    """Two concurrent pops for the same tool name must return {A, B}."""
    session, token = _bind_session()
    try:
        ctx.set_pending_tool_calls([
            {"id": "call_A", "name": "get_customer_info"},
            {"id": "call_B", "name": "get_customer_info"},
        ])

        async def _pop():
            # Yield so both tasks are scheduled before either pops (repro race).
            await asyncio.sleep(0)
            return ctx._pop_pending_tool_call_id("get_customer_info")

        async def _run():
            return await asyncio.gather(_pop(), _pop())

        ids = _run_coro(_run())
        assert set(ids) == {"call_A", "call_B"}, ids
        # Stash drained.
        assert ctx._pop_pending_tool_call_id("get_customer_info") == ""
    finally:
        ctx._current_session.reset(token)


def test_concurrent_different_names_map_correctly():
    session, token = _bind_session()
    try:
        ctx.set_pending_tool_calls([
            {"id": "id_foo", "name": "foo"},
            {"id": "id_bar", "name": "bar"},
        ])

        async def _pop(name):
            await asyncio.sleep(0)
            return ctx._pop_pending_tool_call_id(name)

        async def _run():
            return await asyncio.gather(_pop("foo"), _pop("bar"))

        ids = _run_coro(_run())
        assert set(ids) == {"id_foo", "id_bar"}, ids
    finally:
        ctx._current_session.reset(token)


def test_forked_context_pop_visible_on_shared_session():
    """copy_context sibling must see pops because the session object is shared."""
    session, token = _bind_session()
    try:
        ctx.set_pending_tool_calls([
            {"id": "a", "name": "foo"},
            {"id": "b", "name": "bar"},
        ])
        child = contextvars.copy_context()

        def _in_child():
            assert ctx._pop_pending_tool_call_id("foo") == "a"

        child.run(_in_child)

        # Parent sees the removal (shared session list).
        assert [e["name"] for e in session._pending_tool_calls] == ["bar"]
        assert ctx._pop_pending_tool_call_id("bar") == "b"
    finally:
        ctx._current_session.reset(token)


def test_sequential_fifo_and_no_match_bound_session():
    session, token = _bind_session()
    try:
        ctx.set_pending_tool_calls([
            {"id": "1", "name": "dup"},
            {"id": "2", "name": "dup"},
        ])
        assert ctx._pop_pending_tool_call_id("dup") == "1"
        assert ctx._pop_pending_tool_call_id("dup") == "2"
        assert ctx._pop_pending_tool_call_id("dup") == ""
        assert ctx._pop_pending_tool_call_id("missing") == ""
    finally:
        ctx._current_session.reset(token)


def test_unscoped_sequential_fifo_still_works():
    """No bound session: ContextVar path, in-place pop, sequential FIFO."""
    # Ensure no session is bound (pytest may leak from other tests).
    token = ctx._current_session.set(None)
    try:
        ctx.set_pending_tool_calls([
            {"id": "x1", "name": "f"},
            {"id": "x2", "name": "f"},
        ])
        assert ctx._pop_pending_tool_call_id("f") == "x1"
        assert ctx._pop_pending_tool_call_id("f") == "x2"
        assert ctx._pop_pending_tool_call_id("f") == ""
    finally:
        ctx._current_session.reset(token)


def test_unscoped_concurrent_same_name_distinct_ids():
    """In-place ContextVar list is shared across task context copies."""
    token = ctx._current_session.set(None)
    try:
        ctx.set_pending_tool_calls([
            {"id": "u1", "name": "tool"},
            {"id": "u2", "name": "tool"},
        ])

        async def _pop():
            await asyncio.sleep(0)
            return ctx._pop_pending_tool_call_id("tool")

        async def _run():
            return await asyncio.gather(_pop(), _pop())

        ids = _run_coro(_run())
        assert set(ids) == {"u1", "u2"}, ids
    finally:
        ctx._current_session.reset(token)


def test_replace_clears_and_never_raises():
    session, token = _bind_session()
    try:
        ctx.set_pending_tool_calls([{"id": "old", "name": "f"}])
        ctx.set_pending_tool_calls([])
        assert ctx._pop_pending_tool_call_id("f") == ""
        ctx.set_pending_tool_calls(None)
        assert ctx._pop_pending_tool_call_id("f") == ""
        # Fail-open garbage
        ctx.set_pending_tool_calls("not-a-list")
        ctx.set_pending_tool_calls(object())
    finally:
        ctx._current_session.reset(token)
