"""Runtime capability/shape detection helpers.

The enforcer adapts to provider/framework SDKs by **detecting what is actually
in front of it at runtime** rather than branching on version numbers (which rot
immediately, miss patch releases/forks, and don't compose). These helpers
centralize the detection patterns that were previously open-coded across the
wrappers, so every wrapper reuses one definition.

Two rules of thumb:
  * A stream object may support the **sync** protocol, the **async** protocol,
    or **both** (e.g. litellm's ``CustomStreamWrapper`` exposes ``__iter__`` AND
    ``__anext__``). Never assume; detect.
  * When a method is patched, ``self`` must reach the underlying SDK via the
    descriptor protocol (some SDKs — llama-index 0.14+ — wrap methods with a
    strict ``wrapt`` dispatcher that ``inspect.signature().bind()``s the call
    and rejects a ``self`` passed positionally); route calls through the
    descriptor protocol (see ``_li_call_original``).
"""

import inspect
from typing import Any, Callable


def is_sync_iterator(obj: Any) -> bool:
    """True if ``obj`` can be driven with a plain ``for`` loop."""
    return obj is not None and hasattr(obj, "__next__")


def is_async_iterator(obj: Any) -> bool:
    """True if ``obj`` can be driven with ``async for``."""
    return obj is not None and hasattr(obj, "__anext__")


def is_stream(obj: Any) -> bool:
    """True if ``obj`` is a streaming response (sync OR async iterator)."""
    return is_sync_iterator(obj) or is_async_iterator(obj)


def is_dual_protocol(obj: Any) -> bool:
    """True if ``obj`` exposes BOTH the sync and async iterator protocols.

    litellm's ``CustomStreamWrapper`` is the canonical example: probing for
    ``__anext__`` alone wrongly classifies the sync-intended stream as async.
    """
    return is_sync_iterator(obj) and is_async_iterator(obj)


def has_param(fn: Callable, name: str) -> bool:
    """True if callable ``fn`` declares a parameter named ``name``.

    Used to decide whether to pass an argument that only newer SDK versions
    accept — feature detection instead of a version check. Fail-open: returns
    False if the signature can't be introspected (builtins, C extensions).
    """
    try:
        return name in inspect.signature(fn).parameters
    except (ValueError, TypeError):
        return False
