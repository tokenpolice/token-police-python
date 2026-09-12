"""§4-py-l — TOKENPOLICE_API_KEY env fallback is reachable.

`init(api_key=...)` used to be positional-required, so the documented env-var
fallback could never trigger when the argument was omitted. `api_key` is now
`Optional[str] = None`, making the fallback reachable while keeping the
missing-key ValueError contract.
"""
import pytest

import token_police as tp
from token_police.state import get_client, set_client


@pytest.fixture(autouse=True)
def _reset_client():
    yield
    # Tear down whatever init() installed so tests don't leak a client.
    c = get_client()
    if c:
        try:
            c.close_sync()
        except Exception:
            pass
    set_client(None)


def test_env_fallback_used_when_arg_omitted(monkeypatch):
    monkeypatch.setenv("TOKENPOLICE_API_KEY", "tp_sk_from_env")
    client = tp.init()  # no api_key argument at all
    assert client.api_key == "tp_sk_from_env"


def test_env_fallback_used_when_arg_none(monkeypatch):
    monkeypatch.setenv("TOKENPOLICE_API_KEY", "tp_sk_from_env")
    client = tp.init(api_key=None)
    assert client.api_key == "tp_sk_from_env"


def test_explicit_key_wins_over_env(monkeypatch):
    monkeypatch.setenv("TOKENPOLICE_API_KEY", "tp_sk_from_env")
    client = tp.init(api_key="tp_sk_explicit")
    assert client.api_key == "tp_sk_explicit"


def test_missing_everywhere_raises_value_error(monkeypatch):
    monkeypatch.delenv("TOKENPOLICE_API_KEY", raising=False)
    with pytest.raises(ValueError):
        tp.init()
    with pytest.raises(ValueError):
        tp.init(api_key=None)
