"""P5 / P6 / P7 (embeddings) — the framework embeddings emitters
(`_instrument_langchain_embeddings` / `_instrument_llama_index_embeddings`'s
`_log_embedding_call`, and the Bedrock InvokeModel embedding handler's
`_log_bedrock_embedding`) must ship the observation `_apply_reroute` /
the local evaluator mints for a live REROUTE directive — not just mint it
into the module-global queue and leave it there.

This also closes the orphan `REROUTE_DIRECTIVE_ISSUED` §8.2 flagged on
`embeddings_llama_index_python`: the rejection IS minted (as
`unappliable_call_shape` — the audit's "cross-modality" framing was wrong;
there is no modality gate) and now ships on the success row, so every
directive resolves to a REJECTED.

LangChain / LlamaIndex embeddings (P5a/P5b): these wrappers call
`_run_sync_check(provider=..., model_hint=...)` with NO `kwargs=` — a
body-less shape exactly like the LlamaIndex chat seams — so they hit the same
unconditional `_apply_reroute` unappliable-shape rejection as the chat file's
N1/N5. Harness mirrors tests/test_framework_embedding_failure_log.py's
`_install_fake_langchain()` / `_install_fake_llamaindex()` (fake modules
injected into `sys.modules` so the suite runs with no real langchain_core /
llama_index installed), combined with tests/test_reroute_unappliable_shape.py's
real-`tp.init()` + stubbed-`check()` setup so `_run_sync_check` /
`_run_async_check` run for REAL.

Bedrock (P5c): DIFFERENT mechanism, and only PARTIALLY covered here — see
the `TestP5cBedrockEmbedding` docstring below for why the naive
"unappliable_call_shape on success" case is structurally impossible on this
seam, and what IS tested instead.
"""
import asyncio
import sys
import types
import unittest
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import AsyncMock, MagicMock

import token_police as tp
from token_police import enforcer as _enforcer
from token_police import state as _state
from token_police.context import TPSession, _current_session


REROUTE_RESULT = {
    "status": "allowed",
    "reroute": {
        "mode": "enforce",
        "model": "text-embedding-3-large",
        "provider": "openai",
        "rule_id": "rule_rr",
    },
}


def _init_client(firewall, check_result):
    client = tp.init(api_key="tp_sk_test_rr_embed", firewall=firewall,
                     deployment="serverless")
    client.check_sync = MagicMock(return_value=check_result)
    client.check = AsyncMock(return_value=check_result)
    client.log_sync = MagicMock()
    return client


def _pinned_session(**kwargs):
    """Bare context manager over the session contextvar — see the LlamaIndex
    chat file's twin for why (avoids `tp.session()`'s extra structural row)."""
    class _Ctx:
        def __enter__(_self):
            sess = TPSession(**kwargs)
            _self._tok = _current_session.set(sess)
            return sess

        def __exit__(_self, *a):
            _current_session.reset(_self._tok)
            return False

    return _Ctx()


def _install_fake_langchain():
    """Inject a minimal langchain_core.embeddings.Embeddings and instrument
    it. Mirrors test_framework_embedding_failure_log.py's helper."""
    for key in list(sys.modules):
        if key == "langchain_core" or key.startswith("langchain_core."):
            del sys.modules[key]

    class Embeddings:
        model = "text-embedding-3-small"

        def embed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

        def embed_query(self, text):
            return [0.1, 0.2]

        async def aembed_documents(self, texts):
            return [[0.1, 0.2] for _ in texts]

        async def aembed_query(self, text):
            return [0.1, 0.2]

    lc_core = types.ModuleType("langchain_core")
    lc_emb = types.ModuleType("langchain_core.embeddings")
    lc_emb.Embeddings = Embeddings
    sys.modules["langchain_core"] = lc_core
    sys.modules["langchain_core.embeddings"] = lc_emb

    for key in list(_enforcer._originals):
        cls = key[0] if isinstance(key, tuple) else None
        if cls is not None and getattr(cls, "__name__", None) == "Embeddings":
            _enforcer._originals.pop(key, None)

    _enforcer._instrument_langchain_embeddings()
    return Embeddings


def _install_fake_llamaindex():
    for key in list(sys.modules):
        if key == "llama_index" or key.startswith("llama_index."):
            del sys.modules[key]

    class BaseEmbedding:
        model_name = "text-embedding-ada-002"

        def get_text_embedding(self, text):
            return [0.1, 0.2]

        def get_query_embedding(self, text):
            return [0.1, 0.2]

        def get_text_embedding_batch(self, texts):
            return [[0.1, 0.2] for _ in texts]

        async def aget_text_embedding(self, text):
            return [0.1, 0.2]

        async def aget_query_embedding(self, text):
            return [0.1, 0.2]

    li_root = types.ModuleType("llama_index")
    li_core = types.ModuleType("llama_index.core")
    li_emb = types.ModuleType("llama_index.core.embeddings")
    li_emb.BaseEmbedding = BaseEmbedding
    sys.modules["llama_index"] = li_root
    sys.modules["llama_index.core"] = li_core
    sys.modules["llama_index.core.embeddings"] = li_emb

    for key in list(_enforcer._originals):
        cls = key[0] if isinstance(key, tuple) else None
        if cls is not None and getattr(cls, "__name__", None) == "BaseEmbedding":
            _enforcer._originals.pop(key, None)

    _enforcer._instrument_llama_index_embeddings()
    return BaseEmbedding


class _Base(unittest.TestCase):
    def setUp(self):
        _state.reset_pack()
        try:
            _state.drain_observations()
        except Exception:
            pass

    def tearDown(self):
        _state.reset_pack()
        tp.uninstrument()

    def _run(self, coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ── P5a. LangChain embeddings success ships the rejection ────────

class TestP5aLangChainEmbeddings(_Base):
    def test_sync_embed_documents_success_ships_reroute_rejected(self):
        client = _init_client("enforce", REROUTE_RESULT)
        Embeddings = _install_fake_langchain()
        inst = Embeddings()

        with _pinned_session(workflow_name="wf"):
            out = inst.embed_documents(["hello"])

        self.assertEqual(out, [[0.1, 0.2]])
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertEqual(payload.get("operation"), "embedding")
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")
        self.assertEqual(obs[0]["rule_id"], "rule_rr")

    def test_async_aembed_query_success_ships_reroute_rejected(self):
        client = _init_client("enforce", REROUTE_RESULT)
        Embeddings = _install_fake_langchain()
        inst = Embeddings()

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                return await inst.aembed_query("hi")

        out = self._run(_drive())

        self.assertEqual(out, [0.1, 0.2])
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")


# ── P5b. LlamaIndex embeddings success ships the rejection ───────

class TestP5bLlamaIndexEmbeddings(_Base):
    def test_sync_get_text_embedding_success_ships_reroute_rejected(self):
        client = _init_client("enforce", REROUTE_RESULT)
        BaseEmbedding = _install_fake_llamaindex()
        inst = BaseEmbedding()

        with _pinned_session(workflow_name="wf"):
            out = inst.get_text_embedding("hello")

        self.assertEqual(out, [0.1, 0.2])
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertEqual(payload.get("operation"), "embedding")
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")

    def test_async_aget_query_embedding_success_ships_reroute_rejected(self):
        client = _init_client("enforce", REROUTE_RESULT)
        BaseEmbedding = _install_fake_llamaindex()
        inst = BaseEmbedding()

        async def _drive():
            with _pinned_session(workflow_name="wf"):
                return await inst.aget_query_embedding("hi")

        out = self._run(_drive())

        self.assertEqual(out, [0.1, 0.2])
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["rejection_reason"], "unappliable_call_shape")


# ── P5c. Bedrock embeddings ───────────────────────────────────────

class _SyncBody:
    def __init__(self, data):
        self._data = data

    def read(self):
        return self._data


_VALID_TITAN_BODY = b'{"embedding": [0.1, 0.2], "inputTextTokenCount": 3}'

_CROSS_PROVIDER_SNAPSHOT = {
    "schema_version": 1, "type": "snapshot", "version": 1,
    "tenant_id": "t", "project_id": "p", "ttl_seconds": 600, "loop_blocks": [],
    "directives": [
        {
            "id": "rr-embed", "kind": "REROUTE", "mode": "enforce", "priority": 10,
            "selector": {"match": None, "group_by": []},
            "reroute": {"from": {}, "to": {"provider": "openai",
                                            "model": "text-embedding-3-small"}},
        }
    ],
}


def _bedrock_args():
    self_obj = MagicMock()
    api_params = {"modelId": "amazon.titan-embed-text-v1",
                  "body": b'{"inputText": "hello"}'}
    response = {"body": _SyncBody(_VALID_TITAN_BODY), "ResponseMetadata": {}}
    return (self_obj, "InvokeModel", api_params), response


class TestP5cBedrockEmbedding(_Base):
    """Bedrock's two InvokeModel embedding handlers pass `can_reroute=False`
    to `_run_sync_check`/`_run_async_check` (FIX_PLAN §2.2(c) + the
    pre-existing test_reroute_suppress_bedrock_embedding.py): the throwaway
    `{"model": modelId}` kwargs dict re-reads `*args` unchanged on the real
    InvokeModel call, so a reroute swap could never reach the provider.

    `can_reroute=False` gates BOTH `_apply_reroute` call sites (State A and
    State B) entirely — verified by reading the gate in enforcer.py directly
    (`if local_decision... and can_reroute:` / `if can_reroute: apply_status
    = _apply_reroute(...)`). So the standard "unappliable_call_shape on a
    live ENFORCE directive" case the chat/other-embeddings arms exercise is
    STRUCTURALLY IMPOSSIBLE here: `_apply_reroute` is simply never called,
    can_reroute=False or not.

    What FIX_PLAN §2.2(c) actually means by "local-evaluator observations
    are pushed unconditionally" is a SEPARATE, earlier mechanism:
    `_local_evaluate`'s own `reroute_rejected` push (cross_provider_unsupported
    / serving_unverified — see local_evaluator.py:566-578) happens BEFORE the
    can_reroute gate is ever consulted (the caller does
    `for obs in observations: _state.push_observation(obs)` unconditionally,
    then separately gates the apply/would-reroute logic on can_reroute). That
    mechanism requires State A (`deployment="daemon"` + a healthy local pack)
    — this test drives exactly that, via the SAME real handler precedent as
    test_bedrock_body_restore.py (`_handle_bedrock_embedding_sync`, no real
    boto3/AWS mocking needed).
    """

    def test_state_a_cross_provider_reroute_ships_rejected_on_success_row(self):
        client = _init_client("enforce", {"status": "allowed"})
        client.deployment = "daemon"
        _state.apply_snapshot(_CROSS_PROVIDER_SNAPSHOT)

        args, response = _bedrock_args()
        original = MagicMock(return_value=response)

        with _pinned_session(workflow_name="wf"):
            out = _enforcer._handle_bedrock_embedding_sync(original, args, {})

        self.assertIs(out, response)
        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        obs = payload.get("observations")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0]["outcome"], "reroute_rejected")
        self.assertEqual(obs[0]["rejection_reason"], "cross_provider_unsupported")
        self.assertEqual(obs[0]["rule_id"], "rr-embed")

    def test_can_reroute_gate_actually_suppresses_apply_not_just_the_mint(self):
        """Control: even with the SAME-provider directive (no cross-provider
        rejection to unconditionally mint), can_reroute=False means nothing
        applies and nothing ships — proving the gate, not a coincidence."""
        client = _init_client("enforce", {
            "status": "allowed",
            "reroute": {"mode": "enforce", "model": "amazon.titan-embed-text-v2:0",
                        "provider": "bedrock", "rule_id": "rr-same"},
        })
        # State B (no snapshot applied) — the same-provider reroute would
        # normally APPLY under can_reroute=True; here it must do nothing.
        args, response = _bedrock_args()
        original = MagicMock(return_value=response)

        with _pinned_session(workflow_name="wf"):
            _enforcer._handle_bedrock_embedding_sync(original, args, {})

        self.assertEqual(client.log_sync.call_count, 1)
        payload = client.log_sync.call_args.kwargs
        self.assertFalse(payload.get("observations"))
        self.assertIsNone(payload.get("local_decision"))


# ── P6. local_decision fence (LangChain embeddings arm) ───────────

class TestP6LocalDecisionFence(_Base):
    def test_untagged_decision_ships_nowhere_and_survives_the_call(self):
        client = _init_client("enforce", REROUTE_RESULT)
        Embeddings = _install_fake_langchain()
        inst = Embeddings()

        untagged = {
            "outcome": "rerouted",
            "rule_id": "someone_elses_rule",
            "reroute": {"from": {"model": "x"}, "to": {"model": "y"}},
        }
        with _pinned_session(workflow_name="wf") as sess:
            _enforcer._stash_local_decision_entry(sess, untagged, None)

            inst.embed_documents(["hello"])

            self.assertEqual(client.log_sync.call_count, 1)
            payload = client.log_sync.call_args.kwargs
            self.assertEqual(len(payload.get("observations") or []), 1)
            self.assertIsNone(payload.get("local_decision"))

            claimed = _enforcer._claim_local_decision(sess, "any-later-key")
            self.assertEqual(claimed, untagged)


# ── P7. No-op reroute — nothing minted, nothing shipped ───────────

class TestP7NoopReroute(_Base):
    def test_dry_run_directive_ships_no_observations(self):
        client = _init_client("enforce", {
            "status": "allowed",
            "reroute": {"mode": "dry_run", "model": "text-embedding-3-large",
                        "provider": "openai", "rule_id": "rule_rr"},
        })
        Embeddings = _install_fake_langchain()
        inst = Embeddings()

        with _pinned_session(workflow_name="wf"):
            inst.embed_documents(["hello"])

        self.assertEqual(client.log_sync.call_count, 1)
        self.assertFalse(client.log_sync.call_args.kwargs.get("observations"))
        self.assertEqual(_state.drain_observations(), [])

    def test_no_reroute_directive_at_all(self):
        client = _init_client("enforce", {"status": "allowed"})
        BaseEmbedding = _install_fake_llamaindex()
        inst = BaseEmbedding()

        with _pinned_session(workflow_name="wf"):
            inst.get_text_embedding("hi")

        self.assertEqual(client.log_sync.call_count, 1)
        self.assertFalse(client.log_sync.call_args.kwargs.get("observations"))


if __name__ == "__main__":
    unittest.main()
