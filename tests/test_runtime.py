"""
Unit tests for deployment-mode detection (token_police/runtime.py).

The broad ``VERCEL`` / ``VERCEL_ENV`` markers were removed from
``_SERVERLESS_MARKERS`` because Vercel sets them in EVERY deployment context
(including long-lived Python/Node servers), so any such process was mis-forced
to "serverless" — silently disabling the SSE Decision-Pack stream. Real Vercel
serverless runs on AWS Lambda (AWS_LAMBDA_FUNCTION_NAME / LAMBDA_TASK_ROOT) and
Vercel Edge sets VERCEL_EDGE_REGION, so genuine serverless/edge stays classified.

Mirrors token-police-node/tests/runtime.test.ts for cross-SDK parity.

Env hygiene: the ``clean_env`` fixture removes every known marker before each
test via monkeypatch.delenv, so a VERCEL var leaking from the CI runner cannot
pollute cases or other suites (monkeypatch auto-restores on teardown).
"""
import os

import pytest

from token_police import runtime
from token_police.runtime import detect_deployment_mode, resolve_deployment

_ALL_MARKERS = (
    "AWS_LAMBDA_FUNCTION_NAME",
    "LAMBDA_TASK_ROOT",
    "VERCEL",
    "VERCEL_ENV",
    "NETLIFY",
    "FUNCTIONS_WORKER_RUNTIME",
    "FUNCTION_TARGET",
    "K_SERVICE",
    "CF_PAGES",
    "VERCEL_EDGE_REGION",
    "DENO_DEPLOY",
)


@pytest.fixture
def clean_env(monkeypatch):
    """Start each test from a known-clean baseline (all markers unset)."""
    for m in _ALL_MARKERS:
        monkeypatch.delenv(m, raising=False)
    return monkeypatch


# (a) source assertion — VERCEL / VERCEL_ENV gone; rest intact; edge untouched.
def test_serverless_markers_source_removal():
    assert "VERCEL" not in runtime._SERVERLESS_MARKERS
    assert "VERCEL_ENV" not in runtime._SERVERLESS_MARKERS
    # K_SERVICE removed too (fires in long-lived Cloud Run *services*);
    # true GCP FaaS still detected via the retained FUNCTION_TARGET marker.
    assert "K_SERVICE" not in runtime._SERVERLESS_MARKERS
    assert runtime._SERVERLESS_MARKERS == (
        "AWS_LAMBDA_FUNCTION_NAME",
        "LAMBDA_TASK_ROOT",
        "NETLIFY",
        "FUNCTIONS_WORKER_RUNTIME",
        "FUNCTION_TARGET",
    )
    # EDGE markers untouched (VERCEL_EDGE_REGION is a distinct string, kept).
    assert runtime._EDGE_MARKERS == (
        "CF_PAGES",
        "VERCEL_EDGE_REGION",
        "DENO_DEPLOY",
    )


# (b) VERCEL-only → daemon
def test_vercel_only_is_daemon(clean_env):
    clean_env.setenv("VERCEL", "1")
    clean_env.setenv("VERCEL_ENV", "production")
    assert detect_deployment_mode() == "daemon"


# (c) VERCEL + Lambda → serverless (genuine Vercel serverless via Lambda marker)
def test_vercel_plus_lambda_is_serverless(clean_env):
    clean_env.setenv("VERCEL", "1")
    clean_env.setenv("AWS_LAMBDA_FUNCTION_NAME", "my-fn")
    assert detect_deployment_mode() == "serverless"


# (c) Vercel Edge marker → edge
def test_vercel_edge_region_is_edge(clean_env):
    clean_env.setenv("VERCEL", "1")
    clean_env.setenv("VERCEL_EDGE_REGION", "iad1")
    assert detect_deployment_mode() == "edge"


# (c) untouched serverless markers still work
def test_netlify_is_serverless(clean_env):
    clean_env.setenv("NETLIFY", "1")
    assert detect_deployment_mode() == "serverless"


# RED→GREEN pin: K_SERVICE alone means a long-lived Cloud Run *service*,
# which should behave like a daemon (regain SSE), not serverless.
def test_k_service_alone_is_daemon(clean_env):
    clean_env.setenv("K_SERVICE", "svc")
    assert detect_deployment_mode() == "daemon"


# Over-removal guard: a true Cloud Run *function* / Cloud Function sets BOTH
# K_SERVICE and FUNCTION_TARGET — still serverless via the retained marker.
def test_k_service_plus_function_target_is_serverless(clean_env):
    clean_env.setenv("K_SERVICE", "svc")
    clean_env.setenv("FUNCTION_TARGET", "handler")
    assert detect_deployment_mode() == "serverless"


# FaaS-intact: the retained FUNCTION_TARGET marker independently classifies
# GCP Cloud Functions as serverless.
def test_function_target_alone_is_serverless(clean_env):
    clean_env.setenv("FUNCTION_TARGET", "handler")
    assert detect_deployment_mode() == "serverless"


# (c) bare non-cloud env → daemon
def test_bare_env_is_daemon(clean_env):
    assert detect_deployment_mode() == "daemon"


# edge takes precedence over serverless
def test_edge_over_serverless_precedence(clean_env):
    clean_env.setenv("VERCEL_EDGE_REGION", "iad1")
    clean_env.setenv("NETLIFY", "1")
    assert detect_deployment_mode() == "edge"


# (e) no-throw: detection stays try/except → daemon
def test_detection_never_throws(clean_env):
    # Should not raise for any input; returns a valid mode.
    assert detect_deployment_mode() in ("daemon", "serverless", "edge")


# explicit override still honored
def test_explicit_override_wins(clean_env):
    clean_env.setenv("NETLIFY", "1")  # would auto-detect serverless
    assert resolve_deployment("daemon") == "daemon"
    assert resolve_deployment("auto") == "serverless"
