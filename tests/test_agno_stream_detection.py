"""Agno stream detection must consult the Agent instance's own `stream`
attribute when the `stream=` kwarg is absent, while an explicit kwarg (even
False) still wins over an instance-level True.
"""
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import enforcer
from token_police.context import _in_agno
from token_police.enforcer import _agno_streaming, _make_agno_run_wrapper


@pytest.fixture(autouse=True)
def _reset_agno_flag():
    # The streaming branch intentionally leaves _in_agno set (the real iterator
    # resets it on exhaustion); the stub below does not, so reset around every
    # test to avoid leaking the guard into later enforcement tests.
    _in_agno.set(False)
    yield
    _in_agno.set(False)


class _Agent:
    def __init__(self, stream):
        self.stream = stream


def test_agno_streaming_helper_matrix():
    # kwarg absent → instance attribute decides.
    assert _agno_streaming(_Agent(True), {}) is True
    assert _agno_streaming(_Agent(False), {}) is False
    # kwarg present always wins over instance.
    assert _agno_streaming(_Agent(True), {"stream": False}) is False
    assert _agno_streaming(_Agent(False), {"stream": True}) is True
    # instance without a stream attr defaults to False; never raises.
    assert _agno_streaming(object(), {}) is False


def _run_wrapper(instance, **kwargs):
    """Drive run_wrapper with /check and the stream-iter stubbed so we observe
    which branch (streaming vs plain) was taken."""
    _in_agno.set(False)
    with mock.patch.object(enforcer, "_run_sync_check", lambda *a, **k: None), \
         mock.patch.object(enforcer, "_AgnoSyncStreamIter",
                           lambda result, self: ("STREAM", result)):
        def original(self, *a, **k):
            return "RESULT"
        wrapper = _make_agno_run_wrapper(original)
        return wrapper(instance, **kwargs)


def test_wrapper_detects_instance_stream_when_kwarg_absent():
    out = _run_wrapper(_Agent(True))
    assert out == ("STREAM", "RESULT")


def test_wrapper_kwarg_false_beats_instance_true():
    out = _run_wrapper(_Agent(True), stream=False)
    assert out == "RESULT"


def test_wrapper_kwarg_true_beats_instance_false():
    out = _run_wrapper(_Agent(False), stream=True)
    assert out == ("STREAM", "RESULT")
