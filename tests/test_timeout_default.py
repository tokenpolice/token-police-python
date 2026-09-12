"""
Regression pin: the default remote-call timeout is 2.0 SECONDS.

0.5 s was a LAN/sidecar number. Public-internet round trip to the
Cloudflare-fronted collector measures ~0.44 s warm and 0.9-1.5 s cold from a
distant region, so 0.5 s timed out intermittently -- and a `/check` timeout
fails open, i.e. the old default silently disabled enforcement for distant
customers. Pin both declaration sites (constructor + init()).
"""
import inspect

from token_police.client import TokenPolice, init


def test_default_timeout_is_two_seconds():
    client = TokenPolice(api_key="tp_sk_test_timeout_default")
    try:
        assert client.timeout == 2.0
    finally:
        client.close_sync()

    assert inspect.signature(init).parameters["timeout"].default == 2.0
