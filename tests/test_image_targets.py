"""OpenAI Images instrumentation targets.

Introspects the authoritative `_TARGET_METHODS` install list and asserts the SDK
instruments exactly `{generate}` for `openai.resources.images` (one sync `Images`,
one async `AsyncImages`) — and NOT the nonexistent `create` method.

Pre-fix (with the two dead `create` entries present) test (a)/(c) FAIL; post-fix
they PASS. See contract the regression contract .md assertions 4-6.
"""
import pytest

from token_police.enforcer import _TARGET_METHODS

_IMAGES_MODULE = "openai.resources.images"


def _images_entries():
    return [e for e in _TARGET_METHODS if e.get("module") == _IMAGES_MODULE]


def test_openai_images_methods_are_generate_only():
    """(a) Instrumented method set == {"generate"}; (c) no `create` entry."""
    entries = _images_entries()
    methods = {e["method"] for e in entries}
    assert methods == {"generate"}, (
        f"expected only {{'generate'}} for {_IMAGES_MODULE}, got {methods}"
    )
    # (c) explicit: no dead `create` target survives.
    assert not any(e["method"] == "create" for e in entries), (
        "openai.resources.images must not target the nonexistent `create` method"
    )


def test_openai_images_one_sync_one_async_generate():
    """(b) Exactly one sync (Images) and one async (AsyncImages) `generate` entry."""
    entries = [e for e in _images_entries() if e["method"] == "generate"]
    assert len(entries) == 2, f"expected 2 generate entries, got {len(entries)}"

    sync = [e for e in entries if e["async"] is False]
    asyncs = [e for e in entries if e["async"] is True]
    assert len(sync) == 1 and sync[0]["object"] == "Images"
    assert len(asyncs) == 1 and asyncs[0]["object"] == "AsyncImages"


def test_openai_images_survivors_keep_shape_and_modality():
    """Assertion 6 — surviving entries keep manual/modality/shape untouched."""
    entries = _images_entries()
    assert entries, "expected surviving openai.resources.images entries"
    for e in entries:
        assert e["manual"] is True
        assert e["modality"] == "image_gen"
        assert e["shape"] == "openai_images"


def test_create_is_not_a_real_openai_images_method():
    """Assertion 4 — verify the inversion: `create` is not a real attr on the
    installed openai Images class, so dropping its target loses zero coverage.
    Skips cleanly if openai isn't installed."""
    images = pytest.importorskip("openai.resources.images")
    assert not hasattr(images.Images, "create")
    assert not hasattr(images.AsyncImages, "create")
    # `generate` (the real, retained method) IS present.
    assert hasattr(images.Images, "generate")
    assert hasattr(images.AsyncImages, "generate")
