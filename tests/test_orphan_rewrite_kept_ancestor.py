"""Orphan-rewrite must stop at the nearest KEPT ancestor, not hop past it.

`TokenPoliceSpanProcessor` drops framework-noise spans (LangChain/LangGraph
orchestration) and rewrites a surviving child's parent id so the emitted trace
tree stays connected. The walk in `_resolve_kept_parent` must terminate at the
first ANCESTOR THAT ALSO EMITS A ROW (a tool or LLM span) — otherwise a nested
LLM-inside-tool loses its tool parent and per-tool cost rollup is destroyed.

Tree exercised:
    structural root R (agent anchor — never recorded)
      └─ dropped framework span F (no model → dropped, recorded parent-only)
           └─ KEPT tool span T (emits a `tool` row)
                └─ LLM child L (emits an `llm` row)

L's rewritten parent must be T (the tool), NOT R.

A second case asserts the historical behavior is preserved: a chain of
genuinely-dropped framework spans still walks all the way through to the root.
"""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from opentelemetry import trace

from token_police import telemetry
from token_police.telemetry import TokenPoliceSpanProcessor


class _FakeSpan:
    """Minimal finished/started span the processor reads. `set_attribute`
    writes back into `.attributes` so on_start enrichment is observable."""

    def __init__(self, span_id, parent_id, attrs, name,
                 scope="opentelemetry.instrumentation.langchain"):
        self.context = NS(trace_id=0x1234, span_id=span_id)
        self.parent = NS(span_id=parent_id) if parent_id else None
        self.attributes = dict(attrs)
        self.name = name
        self.instrumentation_scope = NS(name=scope)
        self.start_time = 1
        self.end_time = 2
        self.status = None

    def set_attribute(self, k, v):
        self.attributes[k] = v


def _hex(span_id):
    return trace.format_span_id(span_id)


def _capture_on_end(span):
    """Run on_end and capture the kwargs passed to client.log_sync."""
    captured = {}

    class _FakeClient:
        def log_sync(self, **kwargs):
            captured.update(kwargs)

    proc = TokenPoliceSpanProcessor()
    with mock.patch("token_police.state.get_client", return_value=_FakeClient()):
        proc.on_end(span)
    return captured


class TestOrphanRewriteKeptAncestor(unittest.TestCase):
    def setUp(self):
        # Isolate the module-level registries between tests.
        telemetry._span_parent.clear()
        telemetry._kept_spans.clear()
        self.proc = TokenPoliceSpanProcessor()

    # ── ids for the tree ──────────────────────────────────────────────
    R = 0x1111  # structural agent root (never recorded)
    F = 0x2222  # dropped framework span
    T = 0x3333  # kept tool span
    L = 0x4444  # LLM child under the tool

    def test_llm_child_reparents_onto_kept_tool_not_root(self):
        # Dropped framework span F (parent = structural root R). Named
        # "*.workflow" so on_start records its parent link then drops it — no
        # tp.kind, so it never becomes a kept ancestor.
        f_span = _FakeSpan(self.F, self.R, {}, "LangGraph.workflow")
        self.proc.on_start(f_span)

        # Kept tool span T (parent = F). on_start marks it KEPT.
        t_span = _FakeSpan(
            self.T, self.F,
            {"gen_ai.operation.name": "execute_tool"},
            "execute_tool web_search",
        )
        self.proc.on_start(t_span)

        # Sanity: the registries are wired as expected.
        self.assertIn(_hex(self.T), telemetry._kept_spans)
        self.assertNotIn(_hex(self.F), telemetry._kept_spans)
        self.assertEqual(telemetry._span_parent[_hex(self.T)], _hex(self.F))
        self.assertEqual(telemetry._span_parent[_hex(self.F)], _hex(self.R))

        # LLM child L (parent = the tool T). Emit it and read the rewritten parent.
        l_span = _FakeSpan(
            self.L, self.T,
            {
                "gen_ai.system": "openai",
                "gen_ai.request.model": "gpt-4o",
                "gen_ai.usage.input_tokens": 5,
                "gen_ai.usage.output_tokens": 3,
                "tp.root_span_id": _hex(self.R),
            },
            "ChatOpenAI.chat",
        )
        payload = _capture_on_end(l_span)

        # The LLM must nest under the TOOL, preserving per-tool cost rollup —
        # NOT hop over it onto the structural root.
        self.assertEqual(payload["span"]["parent_span_id"], _hex(self.T))
        self.assertNotEqual(payload["span"]["parent_span_id"], _hex(self.R))

    def test_direct_kept_parent_is_noop(self):
        # Immediate parent is itself a kept span → returned unchanged.
        telemetry._kept_spans[_hex(self.T)] = True
        self.assertEqual(
            telemetry._resolve_kept_parent(_hex(self.T), _hex(self.R)),
            _hex(self.T),
        )

    def test_chain_of_dropped_spans_walks_through_to_root(self):
        # R → F1 → F2 → L, where F1 and F2 are genuinely dropped (recorded in
        # _span_parent, never marked kept). The walk must traverse both and land
        # on the real chain top R — the historical behavior.
        F1, F2 = 0x5551, 0x5552
        f1 = _FakeSpan(F1, self.R, {}, "RunnableSequence.workflow")
        f2 = _FakeSpan(F2, F1, {}, "invoke_agent.workflow")
        self.proc.on_start(f1)
        self.proc.on_start(f2)

        self.assertNotIn(_hex(F1), telemetry._kept_spans)
        self.assertNotIn(_hex(F2), telemetry._kept_spans)

        # A distinct root-fallback sentinel proves the returned R came from the
        # walk (reaching the chain top) and not from the fallback argument.
        result = telemetry._resolve_kept_parent(_hex(F2), "ffffffffffffffff")
        self.assertEqual(result, _hex(self.R))
        self.assertNotEqual(result, "ffffffffffffffff")

    def test_walk_hops_over_dropped_below_kept(self):
        # R → F(dropped) → T(kept). A child whose parent is the DROPPED F must
        # still be pulled up to the nearest kept ancestor... but here the child's
        # parent is F and there is no kept ancestor between F and R, so it lands
        # on R. Conversely a child of T lands on T. Guards both directions.
        f = _FakeSpan(self.F, self.R, {}, "LangGraph.workflow")
        t = _FakeSpan(
            self.T, self.F,
            {"gen_ai.operation.name": "execute_tool"},
            "execute_tool db",
        )
        self.proc.on_start(f)
        self.proc.on_start(t)
        # parent = dropped F → no kept ancestor above → root fallback R.
        self.assertEqual(
            telemetry._resolve_kept_parent(_hex(self.F), _hex(self.R)),
            _hex(self.R),
        )
        # parent = kept T → stops at T.
        self.assertEqual(
            telemetry._resolve_kept_parent(_hex(self.T), _hex(self.R)),
            _hex(self.T),
        )


if __name__ == "__main__":
    unittest.main()
