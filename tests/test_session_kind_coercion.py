"""_structural_span kind coercion: only "chain" (case/space-insensitive) is a
chain anchor; anything else — including a non-str kind — degrades to "agent"
without raising.
"""
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from token_police import context as ctx


class _FakeSpan:
    def get_span_context(self):
        # trace_id 0 → _structural_span keeps constructor-default ids.
        return SimpleNamespace(trace_id=0, span_id=0)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeCM:
    def __enter__(self):
        return _FakeSpan()

    def __exit__(self, *a):
        return False


class _FakeTracer:
    def __init__(self, sink):
        self._sink = sink

    def start_as_current_span(self, name, attributes=None):
        self._sink["attributes"] = attributes or {}
        return _FakeCM()


def _kind_for(kind):
    sink = {}
    fake_trace = SimpleNamespace(get_tracer=lambda name: _FakeTracer(sink))
    s = SimpleNamespace(
        workflow_name="wf", user_id="u", paid_plan="free",
        session_id="sid", metadata={},
    )
    with mock.patch.object(ctx, "_otel_trace", fake_trace):
        with ctx._structural_span(s, kind=kind):
            pass
    return sink["attributes"]["tp.kind"]


def test_chain_variants_map_to_chain():
    assert _kind_for("chain") == "chain"
    assert _kind_for("Chain") == "chain"
    assert _kind_for("  CHAIN  ") == "chain"


def test_other_kinds_map_to_agent():
    assert _kind_for("agent") == "agent"
    assert _kind_for("workflow") == "agent"
    assert _kind_for("") == "agent"


def test_non_str_kind_degrades_to_agent_without_raising():
    assert _kind_for(None) == "agent"
    assert _kind_for(123) == "agent"
    assert _kind_for(object()) == "agent"
