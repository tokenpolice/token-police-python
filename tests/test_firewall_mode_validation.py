"""
Firewall mode string validation (Python parity).

An unrecognized `firewall` string must fall back to the SAFE default
"dry_run" (never "enforce"), byte-for-byte and case-sensitively, so a typo can
never silently enable live blocking. Valid modes pass through unchanged.
"""
import logging

import pytest

from token_police.client import TokenPolice

API_KEY = "tp_sk_test123"


def make_client(**kwargs) -> TokenPolice:
    """Construct a client directly (no network, no instrumentation) and ensure
    it is torn down after the test."""
    return TokenPolice(api_key=API_KEY, **kwargs)


@pytest.fixture
def client_factory():
    created = []

    def _make(**kwargs):
        c = make_client(**kwargs)
        created.append(c)
        return c

    yield _make
    for c in created:
        c.close_sync()


@pytest.mark.parametrize("bad", ["dryrun", "Enforce", "shadow", ""])
def test_unknown_mode_falls_back_to_dry_run(client_factory, bad):
    # Item 13 (mirrors Node items 1-4): unknown / wrong-case / non-member /
    # empty → dry_run, and must not raise.
    c = client_factory(firewall=bad)
    assert c.firewall == "dry_run"


@pytest.mark.parametrize("good", ["enforce", "off", "dry_run"])
def test_valid_modes_pass_through_unchanged(client_factory, good):
    # Item 14 (mirrors Node items 5-7): canonical literals preserved exactly.
    c = client_factory(firewall=good)
    assert c.firewall == good


def test_default_stays_dry_run(client_factory):
    # Item 14: neither firewall nor enforce → dry_run.
    c = client_factory()
    assert c.firewall == "dry_run"


def test_legacy_enforce_alias_unchanged(client_factory):
    # Item 15 (mirrors Node item 9): true→enforce, false→off, explicit wins.
    assert client_factory(enforce=True).firewall == "enforce"
    assert client_factory(enforce=False).firewall == "off"
    assert client_factory(enforce=True, firewall="off").firewall == "off"
    # Bogus explicit firewall still wins over the enforce alias → dry_run.
    assert client_factory(enforce=True, firewall="nope").firewall == "dry_run"


def test_warning_gated_on_log_errors(client_factory, caplog):
    # Item 16 (mirrors Node items 10-11): warn only when log_errors=True; the
    # record names the bad value + fallback; logger.warning is total (no raise).
    with caplog.at_level(logging.WARNING, logger="token_police"):
        # log_errors default False → no firewall warning.
        client_factory(firewall="dryrun")
        assert not any("firewall mode" in r.message for r in caplog.records)

        caplog.clear()
        # log_errors True → exactly one firewall warning naming value + dry_run.
        client_factory(firewall="dryrun", log_errors=True)
        fw_records = [r for r in caplog.records if "firewall mode" in r.message]
        assert len(fw_records) == 1
        msg = fw_records[0].getMessage()
        assert "dryrun" in msg
        assert "dry_run" in msg

        caplog.clear()
        # Valid mode under log_errors=True → no firewall warning.
        client_factory(firewall="enforce", log_errors=True)
        assert not any("firewall mode" in r.message for r in caplog.records)


@pytest.mark.parametrize("variant", [" enforce ", "enforce\n", " off "])
def test_whitespace_variant_falls_back_to_dry_run(client_factory, variant):
    # Item 22 (mirrors Node item 21): NO strip / NO casefold — whitespace or
    # newline variants of a valid mode must NOT reach enforce.
    c = client_factory(firewall=variant)
    assert c.firewall == "dry_run"
