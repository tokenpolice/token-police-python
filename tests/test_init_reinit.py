"""
Init() silently tears down and replaces an already-installed client.

The teardown-and-replace on a second init() is intentional (test isolation /
hot-reload), but doing it SILENTLY hides a double-init bug. This fix emits ONE
developer-facing warning when init() replaces a prior client. Detection reads
get_client() BEFORE construct/swap, and the read+warn is wrapped in
try/except Exception so a broken logger can never throw out of init()
(Golden Rule / fail-open).

Mirrors token-police-node/tests/initReInit.test.ts.

All tests use a valid ``tp_sk_`` key + ``firewall="off"`` so the ONLY
``logger.warning`` reachable on the init path is the new replacement warning
(the unwrapped apiKey-prefix warning at client.py:610-611 is suppressed by the
valid key; ``firewall="off"`` disables /check + the SSE stream whose
connect-failure path emits its own warning). State is reset in a ``finally``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import token_police as tp
from token_police import client as tp_client
from token_police import state as tp_state

REPLACE_MSG = "replacing the previously initialized client"
OPTS = dict(api_key="tp_sk_test", firewall="off", deployment="serverless")


def _reset():
    try:
        tp.uninstrument()
    except Exception:
        pass
    tp_state.set_client(None)


@pytest.fixture(autouse=True)
def _clean_global_client():
    # Reset the module-global singleton BEFORE each test too: sibling test files
    # can leave a client installed, which would make "first init" see a prior
    # client and warn. Guarantees a clean slate regardless of run order.
    _reset()
    yield
    _reset()


def test_does_not_warn_on_first_init(caplog):
    # Assertion 4 — no replacement warning on the FIRST init (no prior client).
    try:
        with caplog.at_level(logging.WARNING, logger="token_police"):
            tp.init(**OPTS)
        replace_records = [r for r in caplog.records if REPLACE_MSG in r.getMessage()]
        assert replace_records == []
    finally:
        _reset()


def test_warns_on_second_init_identical_options(caplog):
    # Assertion 3 — warns on the SECOND init, even with IDENTICAL options.
    try:
        tp.init(**OPTS)  # first — no warning
        with caplog.at_level(logging.WARNING, logger="token_police"):
            tp.init(**OPTS)  # second, same options — must warn
        replace_records = [r for r in caplog.records if REPLACE_MSG in r.getMessage()]
        assert len(replace_records) >= 1
    finally:
        _reset()


def test_exactly_one_warning_on_replacing_init(caplog):
    # Assertion 13 — EXACTLY one WARNING record on the replacing init (valid key
    # + firewall="off" → replacement warning is the only WARNING on the path).
    try:
        tp.init(**OPTS)  # first
        with caplog.at_level(logging.WARNING, logger="token_police"):
            tp.init(**OPTS)  # replacing
        warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warning_records) == 1
        assert REPLACE_MSG in warning_records[0].getMessage()
    finally:
        _reset()


def test_get_client_returns_second_client(caplog):
    # Assertion 6 — get_client() returns the SECOND client after re-init; the
    # second init() return value is that client and NOT the first.
    try:
        first = tp.init(**OPTS)
        second = tp.init(**OPTS)
        assert isinstance(second, tp_client.TokenPolice)
        assert second is not first
        assert tp_state.get_client() is second
    finally:
        _reset()


def test_throwing_logger_cannot_escape_replacing_init(monkeypatch):
    # Assertion 8 — a raising logger.warning (broken logger) must NOT escape
    # init(); the replacing init() still returns a TokenPolice. Valid key +
    # firewall="off" guarantees the replacement warning is the only
    # logger.warning reachable, so surviving proves it is wrapped in try/except.
    try:
        tp.init(**OPTS)  # first, with a normal logger

        def _boom(*args, **kwargs):
            raise RuntimeError("boom: broken logger")

        monkeypatch.setattr(tp_client.logger, "warning", _boom)
        second = tp.init(**OPTS)  # replacing — warn raises, must be swallowed
        assert isinstance(second, tp_client.TokenPolice)
    finally:
        _reset()
