"""M7 regression: a nested session()/agent()/chain()/workflow() scope must be
able to EXPLICITLY set the literal default strings ``"default_workflow"`` /
``"anonymous"`` / ``"free"`` and have them OVERRIDE the parent.

Defect (pre-fix, token_police/context.py):
  The nested branch distinguished "provided" from "default" by comparing the
  argument against the literal default (``user_id if user_id != "anonymous"
  else existing.user_id``), so a nested scope could NEVER set exactly those
  values — the parent's value silently won for those three literals.

Fix: the public entry points default ``name``/``user_id``/``paid_plan`` to
``None`` (the "omitted" sentinel); the nested branch inherits only on ``None``
and overrides on any non-None value; the root branch resolves ``None`` back to
the legacy default literal so root-scope behavior is byte-identical.
"""
import token_police as tp
from token_police import get_current_session


# ── Case 1: nested scope explicitly setting the default literals OVERRIDES ────

def test_nested_explicit_default_literals_override_parent():
    with tp.session(name="outer", user_id="user_1", paid_plan="pro"):
        with tp.session(
            name="default_workflow", user_id="anonymous", paid_plan="free"
        ) as inner:
            assert inner.user_id == "anonymous"
            assert inner.paid_plan == "free"
            assert inner.workflow_name == "default_workflow"
            # Same session grouping is still inherited.
            assert get_current_session().user_id == "anonymous"


def test_nested_explicit_default_literals_override_across_entry_points():
    with tp.agent(name="outer", user_id="user_1", paid_plan="pro"):
        with tp.chain(name="default_workflow", user_id="anonymous", paid_plan="free") as inner:
            assert inner.user_id == "anonymous"
            assert inner.paid_plan == "free"
            assert inner.workflow_name == "default_workflow"


# ── Case 2: nested scope OMITTING them INHERITS the parent (unchanged) ────────

def test_nested_omitted_inherits_parent():
    with tp.session(name="outer", user_id="user_1", paid_plan="pro"):
        with tp.session(name="inner") as inner:
            # name given → its own; user_id/paid_plan omitted → inherited.
            assert inner.workflow_name == "inner"
            assert inner.user_id == "user_1"
            assert inner.paid_plan == "pro"


def test_nested_fully_omitted_inherits_all():
    with tp.session(name="outer", user_id="user_1", paid_plan="pro"):
        with tp.session() as inner:
            assert inner.workflow_name == "outer"
            assert inner.user_id == "user_1"
            assert inner.paid_plan == "pro"


# ── Case 3: root-scope defaults unchanged when nothing is provided ───────────

def test_root_defaults_unchanged():
    with tp.session() as s:
        assert s.user_id == "anonymous"
        assert s.paid_plan == "free"
        assert s.workflow_name == "default_workflow"


def test_root_explicit_values_unchanged():
    with tp.session(name="wf", user_id="u1", paid_plan="pro") as s:
        assert s.user_id == "u1"
        assert s.paid_plan == "pro"
        assert s.workflow_name == "wf"


# ── workflow() decorator parity: nested workflow can set the literals ─────────

def test_nested_workflow_can_set_default_literals():
    @tp.workflow(name="default_workflow", user_id="anonymous", paid_plan="free")
    def inner_fn():
        s = get_current_session()
        return (s.user_id, s.paid_plan, s.workflow_name)

    with tp.session(name="outer", user_id="user_1", paid_plan="pro"):
        assert inner_fn() == ("anonymous", "free", "default_workflow")


def test_nested_workflow_omitting_inherits_parent():
    # A bare @tp.workflow() nested in a parent must still INHERIT the parent's
    # user_id/paid_plan (the decorator no longer bakes the literal defaults).
    @tp.workflow(name="inner_wf")
    def inner_fn():
        s = get_current_session()
        return (s.user_id, s.paid_plan)

    with tp.session(name="outer", user_id="user_1", paid_plan="pro"):
        assert inner_fn() == ("user_1", "pro")
