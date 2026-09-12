"""
`stream_stale_grace_seconds` clamp — TokenPolice constructor (C-14).

Clamped to [0, 3600]; non-numeric / bool / NaN / inf falls back to the 60s
default rather than silently disabling (or unbounding) the gate. `bool` is
checked FIRST because `isinstance(True, int)` is True in Python — a bare
`isinstance(x, (int, float))` check would silently accept True/False as 1/0.

Direct construction (bypassing init()) never starts a background StreamClient
thread (that only happens inside the module-level init() wrapper), so these
tests need no network mocking or cleanup.

Sibling: token-police-node/tests/streamStaleGraceOption.test.ts pins the same
matrix against the Node SDK.
"""
import math

import pytest

from token_police.client import TokenPolice


@pytest.mark.parametrize(
    "value, expected",
    [
        (0, 0),
        (-5, 0),
        (99999, 3600),
        ("abc", 60),
        (None, 60),
        (math.inf, 60),
        (-math.inf, 60),
        (math.nan, 60),
        (True, 60),
        (False, 60),
    ],
)
def test_stream_stale_grace_seconds_clamp(value, expected):
    client = TokenPolice(api_key="tp_sk_test", firewall="off", stream_stale_grace_seconds=value)
    assert client.stream_stale_grace_seconds == expected


def test_omitted_defaults_to_60():
    client = TokenPolice(api_key="tp_sk_test", firewall="off")
    assert client.stream_stale_grace_seconds == 60


def test_in_range_value_preserved_unchanged():
    client = TokenPolice(api_key="tp_sk_test", firewall="off", stream_stale_grace_seconds=120)
    assert client.stream_stale_grace_seconds == 120


def test_boundary_3600_preserved_3601_clamps_down():
    at = TokenPolice(api_key="tp_sk_test", firewall="off", stream_stale_grace_seconds=3600)
    assert at.stream_stale_grace_seconds == 3600

    over = TokenPolice(api_key="tp_sk_test", firewall="off", stream_stale_grace_seconds=3601)
    assert over.stream_stale_grace_seconds == 3600
