"""
Deployment-mode detection for the SDK.

The SDK behaves differently depending on where it's running:

- daemon long-lived process (server, container, K8s, CLI). SSE cache
             + local evaluation is safe.
- serverless single-invocation function. Each instance is short-lived and
             can't reliably maintain an SSE connection; inline /check only.
- edge Cloudflare Workers / Vercel Edge / Deno Deploy. No background
             threads, no long-lived sockets. Inline /check only.

`detect_deployment_mode()` looks at well-known environment variables. Any
explicit `deployment=` argument to `init()` wins over auto-detection.
"""
from __future__ import annotations

import os
import logging

logger = logging.getLogger("token_police")

_SERVERLESS_MARKERS = (
    "AWS_LAMBDA_FUNCTION_NAME",
    "LAMBDA_TASK_ROOT",
    "NETLIFY",
    "FUNCTIONS_WORKER_RUNTIME",  # Azure Functions
    "FUNCTION_TARGET",            # GCP Cloud Functions
)

_EDGE_MARKERS = (
    "CF_PAGES",
    "VERCEL_EDGE_REGION",
    "DENO_DEPLOY",
)


def detect_deployment_mode() -> str:
    """Return one of 'daemon', 'serverless', 'edge'."""
    try:
        for marker in _EDGE_MARKERS:
            if os.environ.get(marker):
                return "edge"
        for marker in _SERVERLESS_MARKERS:
            if os.environ.get(marker):
                return "serverless"
        return "daemon"
    except Exception:  # pragma: no cover — env access cannot reasonably fail
        return "daemon"


def resolve_deployment(explicit: str | None) -> str:
    """Resolve the final deployment mode honoring an explicit override."""
    if explicit is None or explicit == "auto":
        return detect_deployment_mode()
    if explicit in ("daemon", "serverless", "edge"):
        return explicit
    logger.warning(
        "TokenPolice: unknown deployment=%r, falling back to auto-detect",
        explicit,
    )
    return detect_deployment_mode()
