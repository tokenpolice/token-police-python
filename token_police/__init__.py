"""
TokenPolice Python SDK
"""
from .client import init
from .context import workflow, session, agent, chain, TPSession, get_current_session, serverless, set_span_name, tool, tool_span
from .enforcer import protect, uninstrument
from .state import get_client
from .exceptions import TokenPoliceBlockedError

async def flush():
    """Await all pending telemetry, then return.

    Drains both the async ``log()`` tasks and the background telemetry threads
    (the thread pool is joined off-loop via ``asyncio.to_thread`` so the event
    loop is never blocked). Call this in an async serverless handler before the
    container is frozen so in-flight spend isn't lost. Note the Python and Node
    SDKs differ in mechanism: Python genuinely joins its worker threads here,
    whereas Node's ``flushSync`` is a diagnostic-only no-op that relies on
    keepalive POSTs surviving the freeze."""
    client = get_client()
    if client:
        await client.flush()

def flush_sync():
    """Block until all pending telemetry threads finish, then return.

    A real drain: it joins the background worker threads carrying queued /log
    POSTs (unlike Node's ``flushSync``, which only logs a pending-count
    diagnostic and leans on keepalive POSTs). Use it from a synchronous
    serverless handler; in async code prefer ``await flush()`` so the event loop
    isn't blocked."""
    client = get_client()
    if client:
        client.flush_sync()

__all__ = [
    "init",
    "workflow",
    "session",
    "agent",
    "chain",
    "TPSession",
    "get_current_session", 
    "serverless",
    "set_span_name",
    "tool",
    "tool_span",
    "protect",
    "uninstrument",
    "get_client",
    "TokenPoliceBlockedError",
    "flush",
    "flush_sync"
]
