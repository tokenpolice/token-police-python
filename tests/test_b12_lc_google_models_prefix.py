"""B12 — langchain_google_genai rewrites its own ``.model`` to the Google API
resource name, silently breaking model-scoped rules.

``ChatGoogleGenerativeAI`` / ``GoogleGenerativeAIEmbeddings`` rewrite their own
``.model`` field inside the pydantic ``validate_environment`` validator
(``chat_models.py:1579-1580``, ``embeddings.py:109-110``):

    if not self.model.startswith("models/"):
        self.model = f"models/{self.model}"

TokenPolice's LangChain seams read that attribute for the pre-flight
``model_hint``, which feeds both the local rule evaluator and the
``/check`` payload (``client.py:255`` -> ``payload["model"]["name"]``). The
collector matches rule conditions on that string EXACTLY, so a customer rule
``model EQ gemini-2.5-flash`` silently never matched the
``models/gemini-2.5-flash`` hint: an ENFORCE BLOCK never fired and a
model-scoped REROUTE stayed silent. Invisible in ``generations`` because
/log already stripped the prefix in telemetry.py before pricing.

Fix (all in ``token_police/enforcer.py``): a new helper
``_lc_google_bare_model(instance, model)`` gates on the *library* — some
class in ``type(instance).__mro__`` whose ``__module__`` root component is
exactly ``"langchain_google_genai"`` — not on the provider slug or the
string shape, so no other provider/framework/hand-rolled integration is
perturbed. Applied at the chat pre-flight (``_lc_check_ctx``) and at both
the embeddings sync/async wrappers + the embeddings failure-log re-read.
The customer's own instance is NEVER mutated — LangChain must keep sending
the ``models/<id>`` resource name to Google on the wire.

``langchain_google_genai`` is NOT installed in this SDK's own venv. Every
fixture below is a tiny fake class with ``__module__`` set by hand — that is
precisely what the helper's gate inspects, so the real gate is exercised
with zero third-party install.
"""
from __future__ import annotations

import unittest

from token_police.enforcer import _lc_check_ctx, _lc_google_bare_model


# ── Fixture builders ───────────────────────────────────────────────────

def _cls(name, module, **attrs):
    """A fresh class named ``name``, ``__module__`` forced to ``module``,
    with ``attrs`` as class-level attributes (mirrors ``_chat_cls`` in
    tests/test_f7_preflight_context.py)."""
    c = type(name, (), attrs)
    c.__module__ = module
    return c


def _google_chat_cls(model="models/gemini-2.5-flash"):
    return _cls("ChatGoogleGenerativeAI", "langchain_google_genai.chat_models",
                model=model)


def _google_embeddings_cls(model="models/text-embedding-004"):
    return _cls("GoogleGenerativeAIEmbeddings", "langchain_google_genai.embeddings",
                model=model)


class _Boom:
    """Every attribute read explodes (mirrors tests/test_f7_preflight_context.py)."""

    def __getattr__(self, name):
        raise RuntimeError("boom")


# ═══════════════════════════════════════════════════════════════════════
# 1-8. Helper unit tests — string normalization + gate precision
# ═══════════════════════════════════════════════════════════════════════

class TestLcGoogleBareModelStripping(unittest.TestCase):
    def test_1_google_chat_instance_strips_prefix(self):
        inst = _google_chat_cls()()
        self.assertEqual(
            _lc_google_bare_model(inst, "models/gemini-2.5-flash"),
            "gemini-2.5-flash",
        )

    def test_2_google_embeddings_instance_strips_prefix(self):
        inst = _google_embeddings_cls()()
        self.assertEqual(
            _lc_google_bare_model(inst, "models/text-embedding-004"),
            "text-embedding-004",
        )

    def test_3_out_of_tree_subclass_still_stripped_via_full_mro(self):
        # A customer/vendor subclass defined OUTSIDE langchain_google_genai —
        # the gate must walk the whole MRO, not just the concrete class.
        base = _google_chat_cls()
        sub = type("MyCustomGoogleChat", (base,), {})
        sub.__module__ = "acme.llms.google_wrapper"
        inst = sub()
        self.assertEqual(
            _lc_google_bare_model(inst, "models/gemini-2.5-flash"),
            "gemini-2.5-flash",
        )

    def test_4_non_google_instances_are_untouched(self):
        cases = (
            ("langchain_openai.chat_models.base", "ChatOpenAI", "gpt-4o"),
            ("langchain_cohere.chat_models", "ChatCohere", "command-r"),
            ("langchain_mistralai.chat_models", "ChatMistralAI", "mistral-large"),
            ("langchain_huggingface.embeddings", "HuggingFaceEmbeddings",
             "sentence-transformers/all-MiniLM-L6-v2"),
            ("langchain_voyageai.embeddings", "VoyageAIEmbeddings", "voyage-2"),
            ("langchain_together.chat_models", "ChatTogether", "meta-llama/Llama-3"),
            ("langchain_aws.chat_models", "ChatBedrock",
             "anthropic.claude-3-5-sonnet"),
            ("langchain_google_vertexai.chat_models", "ChatVertexAI",
             "gemini-2.5-flash"),
        )
        for module, name, model in cases:
            with self.subTest(module=module, model=model):
                inst = _cls(name, module, model=model)()
                self.assertEqual(_lc_google_bare_model(inst, model), model)

        # Critical: a NON-Google instance whose model literally starts with
        # "models/" must be returned UNCHANGED — the gate is the library,
        # never the string shape.
        inst = _cls("ChatOpenAI", "langchain_openai.chat_models.base",
                    model="models/gpt-4o")()
        self.assertEqual(_lc_google_bare_model(inst, "models/gpt-4o"),
                         "models/gpt-4o")

    def test_5_lookalike_package_root_is_untouched(self):
        # Exact root-component match only — a lookalike package must not slip
        # through a substring/prefix check.
        inst = _cls("ChatEvilGoogle", "langchain_google_genai_evil.chat_models",
                    model="models/gemini-2.5-flash")()
        self.assertEqual(
            _lc_google_bare_model(inst, "models/gemini-2.5-flash"),
            "models/gemini-2.5-flash",
        )

    def test_6_bare_id_is_already_idempotent(self):
        inst = _google_chat_cls(model="gemini-2.5-flash")()
        self.assertEqual(_lc_google_bare_model(inst, "gemini-2.5-flash"),
                         "gemini-2.5-flash")

    def test_7_double_prefixed_strips_exactly_once(self):
        inst = _google_chat_cls(model="models/models/foo")()
        self.assertEqual(_lc_google_bare_model(inst, "models/models/foo"),
                         "models/foo")

    def test_8_non_str_and_missing_instance_inputs_are_untouched(self):
        google_inst = _google_chat_cls()()

        # Non-str model values — the isinstance(model, str) guard rejects
        # them before any instance inspection.
        self.assertIsNone(_lc_google_bare_model(google_inst, None))
        self.assertEqual(_lc_google_bare_model(google_inst, 7), 7)
        self.assertEqual(_lc_google_bare_model(google_inst, b"models/foo"),
                         b"models/foo")

        # instance=None with a "models/"-shaped string: type(None).__mro__
        # never contains a langchain_google_genai class, so it falls through
        # to "unchanged" rather than raising.
        self.assertEqual(_lc_google_bare_model(None, "models/foo"), "models/foo")

        # A class object passed where an instance is expected: type(cls) is
        # the metaclass ``type``, whose MRO never matches either.
        google_cls = _google_chat_cls()
        self.assertEqual(_lc_google_bare_model(google_cls, "models/foo"),
                         "models/foo")


# ═══════════════════════════════════════════════════════════════════════
# 9-11. Golden-rule / fail-open tests — the helper must NEVER raise.
#
# This is the highest-value section: the SDK's one hard invariant is that
# only TokenPoliceBlockedError may ever propagate to the customer's app on
# an enforce-mode denial. _lc_google_bare_model is called from wrapper
# bodies that are not themselves @fail_safe, so a raise here would leak
# straight into the customer's call.
# ═══════════════════════════════════════════════════════════════════════

class TestLcGoogleBareModelFailOpen(unittest.TestCase):
    def test_9_mro_access_itself_raises(self):
        class _MroBoomMeta(type):
            @property
            def __mro__(cls):  # noqa: N805 - metaclass property, cls is the class
                raise RuntimeError("mro boom")

        class _Hostile(metaclass=_MroBoomMeta):
            pass

        inst = _Hostile()
        # Must not raise, and must return the input unchanged.
        self.assertEqual(_lc_google_bare_model(inst, "models/foo"), "models/foo")

    def test_10_module_attribute_is_hostile(self):
        # (a) __module__ raises on access.
        class _ModuleBoomMeta(type):
            @property
            def __module__(cls):  # noqa: N805
                raise RuntimeError("module boom")

        class _Hostile(metaclass=_ModuleBoomMeta):
            pass

        self.assertEqual(_lc_google_bare_model(_Hostile(), "models/foo"),
                         "models/foo")

        # (b) __module__ is an int — .split(".") on it raises AttributeError,
        # which must be swallowed by the outer try/except, not just the
        # getattr() default (getattr's default only covers AttributeError on
        # the attribute lookup itself, not errors from using the value).
        int_module_cls = _cls("ChatWeird", "langchain_google_genai.chat_models")
        int_module_cls.__module__ = 12345
        self.assertEqual(
            _lc_google_bare_model(int_module_cls(), "models/foo"), "models/foo"
        )

        # (c) __module__ is None — falls through every MRO entry with no
        # match; unchanged, no raise.
        none_module_cls = _cls("ChatWeird2", "langchain_google_genai.chat_models")
        none_module_cls.__module__ = None
        self.assertEqual(
            _lc_google_bare_model(none_module_cls(), "models/foo"), "models/foo"
        )

    def test_11_hostile_str_subclass_model(self):
        # (a) THE regression case: startswith() raising must be caught. This
        # specifically pins the guard living INSIDE the try/except — moving
        # it back above the try (checking isinstance+startswith before
        # entering the try) is exactly the bug that was fixed once already.
        class _BoomStartswith(str):
            def startswith(self, *a, **kw):
                raise RuntimeError("startswith boom")

        google_inst = _google_chat_cls()()
        hostile = _BoomStartswith("models/gemini-2.5-flash")
        self.assertEqual(_lc_google_bare_model(google_inst, hostile), hostile)

        # (b) __getitem__ raising during the actual slice/strip must be
        # caught too — the input is returned unchanged.
        class _BoomGetitem(str):
            def __getitem__(self, item):
                raise RuntimeError("getitem boom")

        hostile_getitem = _BoomGetitem("models/gemini-2.5-flash")
        self.assertEqual(
            _lc_google_bare_model(google_inst, hostile_getitem), hostile_getitem
        )

        # (c) __len__ raising must not blow up either. len() is only ever
        # called on the constant prefix "models/", never on `model` itself,
        # so this hostile __len__ is never actually invoked — the strip
        # still succeeds correctly rather than merely failing open.
        class _BoomLen(str):
            def __len__(self):
                raise RuntimeError("len boom")

        hostile_len = _BoomLen("models/gemini-2.5-flash")
        result = _lc_google_bare_model(google_inst, hostile_len)
        self.assertEqual(result, "gemini-2.5-flash")


# ═══════════════════════════════════════════════════════════════════════
# 12-13. Seam tests — _lc_check_ctx and the no-mutation invariant.
# ═══════════════════════════════════════════════════════════════════════

class TestLcCheckCtxGoogleModelsPrefix(unittest.TestCase):
    def test_12_check_ctx_returns_bare_model_and_google_provider(self):
        inst = _google_chat_cls(model="models/gemini-2.5-flash")()
        hint, provider = _lc_check_ctx((inst,))
        self.assertEqual(hint, "gemini-2.5-flash")
        # Provider derivation is unaffected by the strip.
        self.assertEqual(provider, "google")

        # No-args / None-instance fail-open shape is unaffected by this fix.
        self.assertEqual(_lc_check_ctx(()), (None, None))
        self.assertEqual(_lc_check_ctx((None,)), (None, None))

    def test_13_customer_instance_is_never_mutated(self):
        inst = _google_chat_cls(model="models/gemini-2.5-flash")()
        hint, _provider = _lc_check_ctx((inst,))
        self.assertEqual(hint, "gemini-2.5-flash")
        # LangChain must keep sending the resource name to Google on the
        # wire — only OUR copy of the string (used for matching/audit/log)
        # is normalized, never the instance's own attribute.
        self.assertEqual(inst.model, "models/gemini-2.5-flash")

    def test_13b_embeddings_instance_is_never_mutated(self):
        inst = _google_embeddings_cls(model="models/text-embedding-004")()
        stripped = _lc_google_bare_model(inst, inst.model)
        self.assertEqual(stripped, "text-embedding-004")
        self.assertEqual(inst.model, "models/text-embedding-004")


if __name__ == "__main__":
    unittest.main()
