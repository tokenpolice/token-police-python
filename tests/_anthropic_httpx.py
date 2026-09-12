"""Resolve the httpx flavour the *installed* ``anthropic`` release is built on.

``anthropic`` 1.3.0 moved off ``httpx`` onto ``httpx2`` and its client
constructor type-checks the ``http_client`` argument against that package::

    TypeError: Invalid `http_client` argument; Expected an instance of
               `httpx2.Client` but got <class 'httpx.Client'>

A test that hands a ``MockTransport`` to a real ``anthropic.Anthropic`` must
therefore not name a transport package: which one is correct is a property of
the installed release, and the SDK pins no ``anthropic`` version by design (the
customer owns it, and the enforcer instruments whatever is installed). Ask the
package instead of guessing, so the same test file runs on an old anthropic
(plain ``httpx``) and on 1.3.0+ (``httpx2``) unchanged.

``anthropic.DefaultHttpxClient`` / ``DefaultAsyncHttpxClient`` are public
exports whose whole purpose is to subclass the ``Client`` / ``AsyncClient`` of
whichever transport package the release ships against, so their MRO names that
package on every release. Walk past anthropic's own subclass and take the
top-level module of the first foreign base.

Import ``anthropic_httpx`` and use it wherever the file would have said
``httpx.`` — ``Client``, ``AsyncClient``, ``MockTransport``, ``Response``,
``Request`` are all present under both names.

This is *only* for clients handed to ``anthropic``. Code exercising the
TokenPolice SDK's own transport must keep importing ``httpx`` directly: that is
a declared dependency of this package (see ``pyproject.toml``) and is not tied
to the customer's anthropic version.
"""
from __future__ import annotations

import importlib
from types import ModuleType

import anthropic


def _transport_package(client_cls: type) -> ModuleType:
    """Top-level module of the httpx-alike ``client_cls`` inherits its Client from."""
    for base in client_cls.__mro__[1:]:
        root = base.__module__.split(".")[0]
        if root != "anthropic":
            return importlib.import_module(root)
    raise RuntimeError(
        f"cannot resolve the httpx package behind {client_cls!r}: its MRO "
        f"{[f'{c.__module__}.{c.__qualname__}' for c in client_cls.__mro__]} "
        "contains no non-anthropic base. The anchor used by "
        "tests/_anthropic_httpx.py no longer holds — find a new one rather "
        "than hardcoding a package name."
    )


anthropic_httpx = _transport_package(anthropic.DefaultHttpxClient)

# The sync and async defaults must agree; a client built from one package's
# transport and handed to a constructor type-checking against the other would
# fail exactly the way this helper exists to prevent.
_async_pkg = _transport_package(anthropic.DefaultAsyncHttpxClient)
if _async_pkg is not anthropic_httpx:
    raise RuntimeError(
        "anthropic's sync and async default clients disagree on their httpx "
        f"package ({anthropic_httpx.__name__} vs {_async_pkg.__name__})"
    )

__all__ = ["anthropic_httpx"]
