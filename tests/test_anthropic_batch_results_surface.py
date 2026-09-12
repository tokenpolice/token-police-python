"""Anthropic Message Batches `.results()` wrapper must hand the customer an
object at least as capable as the provider's original results page — the
context-manager protocol (`with page:` / `async with page:`), `.response`,
arbitrary attribute passthrough, and a `close()`/`aclose()` that closes the
underlying HTTP connection — NOT a bare generator that drops `.close()`/
`.response` and leaks the connection on abandonment. Iteration must still flow
verbatim through the metering generator (one log per entry).

Also covers the dedup guard `_batch_result_already_logged`: the check-set-evict
sequence is now lock-guarded so concurrent `.results()` consumers across threads
never both log the same (batch_id, custom_id), and eviction stays bounded.
"""
from __future__ import annotations

import asyncio
import sys
import threading
import types
from unittest.mock import MagicMock

import pytest

from token_police import enforcer


# ─── fakes: results pages mimicking anthropic's sync/async page objects ──
class FakeBatchPage:
    """Mimics a sync results page: iterable + context manager + .response +
    close()."""

    def __init__(self, entries):
        self._entries = list(entries)
        self._it = iter(self._entries)
        self.response = object()
        self.sentinel = object()
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._it)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def close(self):
        self.closed = True


class FakeAsyncBatchPage:
    """Mimics an async results page: async-iterable + .response + aclose()."""

    def __init__(self, entries):
        self._entries = list(entries)
        self.response = object()
        self.sentinel = object()
        self.closed = False

    def __aiter__(self):
        self._it = iter(self._entries)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


def _make_sync_batches(page):
    class FakeBatches:
        def results(self, message_batch_id, **kwargs):
            return page
    return FakeBatches


def _make_async_batches(page):
    class FakeAsyncBatches:
        async def results(self, message_batch_id, **kwargs):
            return page
    return FakeAsyncBatches


@pytest.fixture
def _spy_log(monkeypatch):
    """Neutralize the real per-entry logging (which needs a live client/session)
    and spy that iteration still meters every entry."""
    spy = MagicMock()
    monkeypatch.setattr(enforcer, "_log_anthropic_batch_entry", spy)
    return spy


def _install_anthropic_batches(monkeypatch, sync_cls=None, async_cls=None):
    """Register a fake `anthropic.resources.messages.batches` module so
    `_instrument_anthropic_batches` patches our classes, then run it. Records the
    patched classes in enforcer._originals; clean up on teardown."""
    mod = types.ModuleType("anthropic.resources.messages.batches")
    # `_instrument_anthropic_batches` imports `Batches` first and bails on
    # ImportError before reaching AsyncBatches — so async-only tests still need a
    # `Batches` attribute present. Use a no-op stand-in (no `results` → skipped).
    class _NoBatches:
        pass
    mod.Batches = sync_cls if sync_cls is not None else _NoBatches
    if async_cls is not None:
        mod.AsyncBatches = async_cls
    monkeypatch.setitem(sys.modules, "anthropic.resources.messages.batches", mod)

    to_restore = []
    if sync_cls is not None:
        to_restore.append((sync_cls, "results"))
    if async_cls is not None:
        to_restore.append((async_cls, "results"))

    enforcer._instrument_anthropic_batches()

    def _cleanup():
        for cls, name in to_restore:
            enforcer._originals.pop((cls, name), None)
    return _cleanup


# ─── sync surface restoration ───────────────────────────────────────
def test_sync_returns_proxy_not_bare_generator(monkeypatch, _spy_log):
    page = FakeBatchPage([object(), object()])
    cls = _make_sync_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, sync_cls=cls)
    try:
        returned = cls().results("bid")
        assert not isinstance(returned, types.GeneratorType)
        assert type(returned).__name__ == "_ModeAStreamProxy"
    finally:
        cleanup()


def test_sync_response_and_attr_passthrough(monkeypatch, _spy_log):
    page = FakeBatchPage([object()])
    cls = _make_sync_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, sync_cls=cls)
    try:
        wrapped = cls().results("bid")
        assert wrapped.response is page.response
        assert wrapped.sentinel is page.sentinel
    finally:
        cleanup()


def test_sync_iteration_yields_and_meters(monkeypatch, _spy_log):
    entries = [object(), object(), object()]
    page = FakeBatchPage(entries)
    cls = _make_sync_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, sync_cls=cls)
    try:
        wrapped = cls().results("bid")
        seen = list(wrapped)
        assert seen == entries                 # verbatim, in order
        assert _spy_log.call_count == len(entries)  # one log per entry
    finally:
        cleanup()


def test_sync_context_manager_closes_underlying(monkeypatch, _spy_log):
    entries = [object(), object()]
    page = FakeBatchPage(entries)
    cls = _make_sync_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, sync_cls=cls)
    try:
        wrapped = cls().results("bid")
        seen = []
        with wrapped as p:
            assert p is wrapped
            for e in p:
                seen.append(e)
        assert seen == entries
        assert page.closed is True             # __exit__ closed underlying page
    finally:
        cleanup()


def test_sync_close_on_early_abandonment_closes_underlying(monkeypatch, _spy_log):
    page = FakeBatchPage([object(), object(), object()])
    cls = _make_sync_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, sync_cls=cls)
    try:
        wrapped = cls().results("bid")
        next(wrapped)             # partially consumed
        wrapped.close()           # explicit close-through (leak fix)
        assert page.closed is True
    finally:
        cleanup()


# ─── async surface restoration ──────────────────────────────────────
def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_async_returns_proxy_not_bare_generator(monkeypatch, _spy_log):
    page = FakeAsyncBatchPage([object(), object()])
    cls = _make_async_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, async_cls=cls)
    try:
        returned = _run(cls().results("bid"))
        assert not isinstance(returned, types.AsyncGeneratorType)
        assert type(returned).__name__ == "_ModeAAsyncStreamProxy"
    finally:
        cleanup()


def test_async_response_and_attr_passthrough(monkeypatch, _spy_log):
    page = FakeAsyncBatchPage([object()])
    cls = _make_async_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, async_cls=cls)
    try:
        wrapped = _run(cls().results("bid"))
        assert wrapped.response is page.response
        assert wrapped.sentinel is page.sentinel
    finally:
        cleanup()


def test_async_iteration_yields_and_meters(monkeypatch, _spy_log):
    entries = [object(), object(), object()]
    page = FakeAsyncBatchPage(entries)
    cls = _make_async_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, async_cls=cls)
    try:
        wrapped = _run(cls().results("bid"))

        async def drain():
            out = []
            async for e in wrapped:
                out.append(e)
            return out

        assert _run(drain()) == entries
        assert _spy_log.call_count == len(entries)
    finally:
        cleanup()


def test_async_context_manager_closes_underlying(monkeypatch, _spy_log):
    entries = [object(), object()]
    page = FakeAsyncBatchPage(entries)
    cls = _make_async_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, async_cls=cls)
    try:
        wrapped = _run(cls().results("bid"))
        seen = []

        async def go():
            async with wrapped as p:
                assert p is wrapped
                async for e in p:
                    seen.append(e)

        _run(go())
        assert seen == entries
        assert page.closed is True
    finally:
        cleanup()


def test_async_aclose_on_early_abandonment_closes_underlying(monkeypatch, _spy_log):
    page = FakeAsyncBatchPage([object(), object(), object()])
    cls = _make_async_batches(page)
    cleanup = _install_anthropic_batches(monkeypatch, async_cls=cls)
    try:
        wrapped = _run(cls().results("bid"))

        async def go():
            async for _e in wrapped:
                break
            await wrapped.aclose()

        _run(go())
        assert page.closed is True
    finally:
        cleanup()


# ─── dedup guard: lock-protected check-set-evict ────────────────────
def test_dedup_single_false_under_thread_contention():
    # Reset shared state so this test is deterministic regardless of order.
    enforcer._BATCH_RESULTS_LOGGED.clear()
    key = "batch_xyz:custom_42"
    results = []
    results_lock = threading.Lock()
    start = threading.Barrier(8)

    def worker():
        start.wait()  # maximize the race window
        r = enforcer._batch_result_already_logged(key)
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one thread wins the first-write (returns False); all others see it
    # already logged (True). No double-log of the same (batch_id, custom_id).
    assert results.count(False) == 1
    assert results.count(True) == 7


def test_dedup_eviction_cap_enforced(monkeypatch):
    enforcer._BATCH_RESULTS_LOGGED.clear()
    monkeypatch.setattr(enforcer, "_BATCH_RESULTS_LOGGED_MAX", 10)
    for i in range(100):
        assert enforcer._batch_result_already_logged(f"k:{i}") is False
    # Cap holds: never exceeds MAX (oldest evicted FIFO).
    assert len(enforcer._BATCH_RESULTS_LOGGED) <= 10
    # The most-recent key is retained; an evicted early key logs again (False).
    assert enforcer._batch_result_already_logged("k:99") is True
    assert enforcer._batch_result_already_logged("k:0") is False
