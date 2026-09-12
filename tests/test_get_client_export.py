"""§6-6 — `get_client` is a public, barrel-exported symbol.

It was importable but absent from `__all__` (Node barrel-exports its
equivalent). This guards both the import and the `__all__` listing.
"""
import token_police
from token_police import get_client


def test_get_client_importable():
    assert callable(get_client)


def test_get_client_in_all():
    assert "get_client" in token_police.__all__
