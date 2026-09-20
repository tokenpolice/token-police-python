"""Suite-wide safety net: the SDK's own tests must never reach a real collector.

Why this file exists
--------------------
``tp.init(...)`` resolves ``base_url`` from ``TOKENPOLICE_BASE_URL`` and falls
back to the PRODUCTION collector. In ``firewall='enforce'``/``'dry_run'`` +
``deployment='daemon'`` (the default pairing) ``init()`` also starts a real SSE
reader thread against that URL. So any test that inits without an explicit
``base_url`` used to open a live connection to prod from the test runner.

That is not just untidy, it is a race that fails tests. The bogus ``tp_sk_...``
keys tests use earn an HTTP 401, and a terminal status makes the reader call
``state.invalidate_pack()`` — correct product behaviour (a stale pack must never
outlive a revoked key). If that lands between a test's ``apply_snapshot()`` and
the enforcer's ``is_cache_healthy()`` read, local rule evaluation is skipped for
that call, the mocked inline ``/check`` returns ``allowed``, and any assertion
about a locally-evaluated REROUTE/BLOCK fails. It flaked exactly this way in the
nightly provider-drift run on 2026-09-14 (``test_reroute_stash_session_thread``),
which is a CI box close enough to the collector for the 401 to win the race.

Before the collector was deployed the same tests were safe by accident: the
hostname did not resolve, and a connect failure is TRANSIENT, which does not
invalidate the pack. Going live turned a no-op into a race.

Two independent protections, because either alone can be defeated
-----------------------------------------------------------------
1. Default every test's collector to a closed loopback port, restoring the old
   "connect fails, transient, pack untouched" behaviour deterministically.
2. Refuse, and loudly fail the test, if a TokenPolice control-plane request is
   aimed at anything but loopback anyway — e.g. a future test that hardcodes a
   production ``base_url``. Without this, protection 1 silently stops covering
   new tests and the flake comes back.

The fix lives here and NOT in ``token_police/``: invalidating the pack on a
terminal stream status is the behaviour we want to ship. The tests were wrong.
"""

from __future__ import annotations

import threading

import httpx
import pytest

#: Discard port on loopback. Nothing listens, so a connect is refused
#: immediately — fast, offline, and (unlike a 401 from a real collector) a
#: TRANSIENT failure that leaves the rule pack alone. Tests that want a
#: different URL still pass ``base_url=`` explicitly; that wins over the env var.
DEAD_COLLECTOR_URL = "http://127.0.0.1:9"

#: Hosts a test is allowed to send TokenPolice control-plane traffic to. Several
#: suites stand up their own stub collector on 127.0.0.1/localhost.
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Only TokenPolice's own endpoints are policed. Provider SDK tests legitimately
#: point mock transports at ``api.anthropic.com``, ``api.minimax.io`` and
#: friends, and must pass through untouched.
_GUARD_PATH_PREFIX = "/v1/guard/"

_escapes: list[str] = []
_escapes_lock = threading.Lock()


def _is_escaping(request: httpx.Request) -> bool:
    """True for a TokenPolice control-plane request leaving the local machine."""
    url = request.url
    if not str(url.path).startswith(_GUARD_PATH_PREFIX):
        return False
    return (url.host or "") not in _LOOPBACK_HOSTS


def _record(request: httpx.Request) -> None:
    with _escapes_lock:
        _escapes.append(f"{request.method} {request.url}")


def _drain() -> list[str]:
    with _escapes_lock:
        found = list(_escapes)
        _escapes.clear()
    return found


@pytest.fixture(autouse=True)
def _no_real_collector(monkeypatch):
    monkeypatch.setenv("TOKENPOLICE_BASE_URL", DEAD_COLLECTOR_URL)

    _real_send = httpx.Client.send
    _real_asend = httpx.AsyncClient.send

    def _guarded_send(self, request, *args, **kwargs):
        if _is_escaping(request):
            _record(request)
            # Raise the same class a closed port produces, so the SDK's
            # fail-open paths behave as they would offline. The test still fails
            # at teardown — the SDK swallows this, so a raise alone is invisible.
            raise httpx.ConnectError(
                "blocked by tests/conftest.py: TokenPolice request left loopback",
                request=request,
            )
        return _real_send(self, request, *args, **kwargs)

    async def _guarded_asend(self, request, *args, **kwargs):
        if _is_escaping(request):
            _record(request)
            raise httpx.ConnectError(
                "blocked by tests/conftest.py: TokenPolice request left loopback",
                request=request,
            )
        return await _real_asend(self, request, *args, **kwargs)

    # Patch ``send`` rather than ``post``/``stream``: every httpx entry point,
    # including the module-level ``httpx.stream()`` the SSE reader uses, funnels
    # through ``Client.send`` / ``AsyncClient.send``.
    monkeypatch.setattr(httpx.Client, "send", _guarded_send, raising=True)
    monkeypatch.setattr(httpx.AsyncClient, "send", _guarded_asend, raising=True)

    yield

    escaped = _drain()
    if escaped:
        pytest.fail(
            "TokenPolice request(s) addressed a non-loopback host during this "
            "test:\n  " + "\n  ".join(sorted(set(escaped))) + "\n"
            "Pass base_url= to init(), or point it at "
            f"{DEAD_COLLECTOR_URL}. Tests must never reach a real collector "
            "(see the module docstring in tests/conftest.py). Note: a daemon SSE "
            "thread left running by an EARLIER test can surface here instead."
        )
