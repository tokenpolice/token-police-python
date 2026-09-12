"""Packaging-metadata guards for pyproject.toml (§3-py-12).

These parse the real pyproject.toml (resolved relative to this test, so they
travel with the package) and assert the invariants that keep the published
metadata correct as extras drift over time:

  * pyproject parses at all;
  * the umbrella `all` extra is a SUPERSET of every standalone extra except
    the two documented exceptions in EXEMPT_FROM_ALL — so
    `pip install token-police[all]` never silently omits an integration;
  * `all` has no duplicate requirements;
  * every extra referenced from `all` that also stands alone (openai-agents,
    xai, agno, voyageai, ...) actually has its own extra;
  * the [project.urls] and classifiers blocks exist and are well-formed.
"""
import os
import tomllib

_PYPROJECT = os.path.join(os.path.dirname(__file__), os.pardir, "pyproject.toml")

# Extras deliberately excluded from the umbrella `all`, with the reason each is
# excluded. Both are documented in the comment above `all` in pyproject.toml.
# Adding a name here must be a considered decision, never a way to silence this
# test — `all` omitting an integration is exactly the bug it exists to catch.
EXEMPT_FROM_ALL = {
    # crewai >=1.6.0 declares no Python 3.14 support; bundling it made
    # `pip install "token-police[all]"` fail the pip resolver on 3.14 for an
    # integration that is not launch-critical. Opt-in via [crewai] instead.
    "crewai": "no Python 3.14 support; opt-in extra",
    # Test toolchain, not an integration.
    "dev": "developer test toolchain",
}


def _load():
    with open(_PYPROJECT, "rb") as f:
        return tomllib.load(f)


def test_pyproject_parses():
    data = _load()
    assert data["project"]["name"] == "token-police"


def test_all_extra_is_superset_of_standalone_extras():
    opt = _load()["project"]["optional-dependencies"]
    all_set = set(opt["all"])
    violations = {}
    for name, deps in opt.items():
        if name == "all" or name in EXEMPT_FROM_ALL:
            continue
        missing = [d for d in deps if d not in all_set]
        if missing:
            violations[name] = missing
    assert not violations, f"`all` extra is missing standalone deps: {violations}"


def test_crewai_is_opt_in_and_not_in_all():
    """`all` must install on Python 3.14, so crewai stays out of it.

    crewai keeps its own standalone extra — `pip install "token-police[crewai]"`
    — which is the install line that legitimately fails on 3.14, with CrewAI's
    own resolver error.
    """
    opt = _load()["project"]["optional-dependencies"]
    assert not any(d.startswith("crewai") for d in opt["all"]), (
        "crewai is back in `all` — that breaks `pip install token-police[all]` "
        "on Python 3.14"
    )
    assert any(d.startswith("crewai") for d in opt["crewai"]), (
        "the standalone [crewai] extra must still install crewai"
    )


def test_dev_extra_provides_the_test_toolchain():
    opt = _load()["project"]["optional-dependencies"]
    assert "dev" in opt
    for pkg in ("pytest", "pytest-asyncio"):
        assert any(d.split("[")[0].split(">")[0].split("=")[0].strip() == pkg
                   for d in opt["dev"]), f"{pkg} missing from [dev]"
    # dev is a toolchain, not an integration — customers must not get pytest
    # from `token-police[all]`.
    assert not any(d.startswith("pytest") for d in opt["all"])


def test_all_extra_has_no_duplicates():
    all_deps = _load()["project"]["optional-dependencies"]["all"]
    dups = [d for d in set(all_deps) if all_deps.count(d) > 1]
    assert not dups, f"duplicate requirements in `all`: {dups}"


def test_previously_missing_integrations_are_in_all():
    all_deps = _load()["project"]["optional-dependencies"]["all"]
    # These three (openai-agents, xai, agno) had standalone extras but were
    # omitted from `all` before this fix; voyageai was in `all` with no
    # standalone extra. All four must now be present.
    for req_prefix in ("openai-agents", "xai-sdk", "agno", "voyageai"):
        assert any(d.startswith(req_prefix) for d in all_deps), (
            f"{req_prefix} missing from `all` extra"
        )


def test_voyageai_has_standalone_extra():
    opt = _load()["project"]["optional-dependencies"]
    assert "voyageai" in opt
    assert any(d.startswith("voyageai") for d in opt["voyageai"])


def test_project_urls_present():
    urls = _load()["project"]["urls"]
    for key in ("Homepage", "Repository", "Documentation", "Issues"):
        assert key in urls, f"missing [project.urls] {key}"
        assert urls[key].startswith("https://")
    # Repository is the public one-way mirror of this directory — the monorepo
    # is private, so its URL would 404 for every customer.
    assert urls["Repository"] == "https://github.com/tokenpolice/token-police-python"
    assert urls["Documentation"] == "https://tokenpolice.ai/docs"


def test_license_is_pep639_expression():
    proj = _load()["project"]
    # PEP 639: SPDX expression + license-files. A `License ::` classifier next
    # to a License-Expression is the one combination Metadata 2.4 rejects.
    assert proj["license"] == "Apache-2.0"
    assert proj["license-files"] == ["LICENSE"]
    assert not any(c.startswith("License ::") for c in proj["classifiers"])
    assert "Typing :: Typed" in proj["classifiers"]


def test_classifiers_cover_supported_pythons():
    proj = _load()["project"]
    classifiers = proj["classifiers"]
    # requires-python floor is 3.10 → classifiers must list 3.10 and up.
    # 3.14 included: the base package and `[all]` both install and import there
    # (verified 2026-09-02); only the opt-in [crewai] extra does not.
    for ver in ("3.10", "3.11", "3.12", "3.13", "3.14"):
        assert f"Programming Language :: Python :: {ver}" in classifiers, ver
    assert "Development Status :: 5 - Production/Stable" in classifiers
    assert any(c.startswith("Intended Audience ::") for c in classifiers)
    assert any(c.startswith("Topic ::") for c in classifiers)
