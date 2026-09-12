"""No-op reroute detection.

A REROUTE rule written against a model alias (``anthropic/claude-haiku-4-5``)
also matches calls that pin the dated snapshot of the SAME model
(``claude-haiku-4-5-20251001``). Applying the swap there rewrites the customer's
pinned snapshot for zero cost delta and emits a phantom REQUEST_REROUTED event,
so both SDK apply paths treat it as a no-op instead.

Pure functions, no imports beyond ``re`` — kept separate so the server /check
guard and the Node SDK can mirror the exact same semantics.

Bedrock note (``prefer_requested_model`` only). A Bedrock model id is itself the
billing identity: ``[<region>.]<vendor>.<model>[-v<n>[:<m>]]``. The provider
echoes only the bare ``<model>`` part on the streaming taps, so a call made
against ``us.anthropic.claude-haiku-4-5-20251001-v1:0`` comes back as
``claude-haiku-4-5-20251001`` — the SAME family, which is why the echo is
treated as a family match and the requested id wins. The requested id must be
reported VERBATIM: a cross-region inference profile carries its own
region-specific rate (64 of 99 CRIS cards differ from their base card, from
-68.8% to +31.4%), so reporting the base id — or any normalized form — would
misprice the call in BOTH directions. The candidates built below exist solely
to decide family equality; none of them is ever reported or priced.
"""
from __future__ import annotations

import re

# Exactly one trailing provider-dated snapshot suffix. Deliberately narrow:
# -20251001 anthropic (claude-haiku-4-5-20251001)
# -2024-08-13 openai (gpt-4o-2024-08-13)
# @20240229 vertex (claude-3-sonnet@20240229)
# Nothing else is a date — `-latest`, `-002`, `-v1:0` etc. are real model
# identity and must never be stripped.
_DATE_SUFFIX_RE = re.compile(r"(?:-20\d{6}|-20\d{2}-\d{2}-\d{2}|@20\d{6})$")


def strip_date_suffix(model: str) -> str:
    """Remove exactly one trailing dated-snapshot suffix from ``model``.

    Returns the input unchanged when it is not a str or carries no such suffix.
    """
    if not isinstance(model, str):
        return model
    return _DATE_SUFFIX_RE.sub("", model, count=1)


def is_noop_reroute(requested_model, target_model) -> bool:
    """True when a reroute's target resolves to the model already requested.

    Direction-sensitive on purpose: the date suffix is stripped from the
    REQUESTED side only, because an explicitly dated *target* is a deliberate
    pin (alias → dated is a real reroute, as is dated → a different date).
    """
    if not isinstance(requested_model, str) or not isinstance(target_model, str):
        return False
    requested = requested_model.strip().lower()
    target = target_model.strip().lower()
    if not requested or not target:
        return False
    if requested == target:
        return True
    return strip_date_suffix(requested) == target


# Bedrock id shape, used ONLY to test family equality in
# ``prefer_requested_model`` — never to build a string that is reported or priced.
# Region prefix of a cross-region inference profile (CRIS).
# KEEP IN SYNC — this same region list is duplicated, with no shared contract and
# no parity test, at:
#     token-police-node/src/rerouteNoop.ts
#     collector-server/src/lib/cost-calculator.js  (CRIS region-strip tier)
# A new AWS region prefix added in only one of the three degrades silently:
# under-billed no-op reroutes here, $0 ``unknown_model`` rows in the collector.
_BEDROCK_REGION_PREFIX = re.compile(r"^(?:us-gov|us|eu|apac|global)\.")
# Vendor namespace every Bedrock model id carries after the optional region.
_BEDROCK_VENDOR_NS = re.compile(
    r"^(?:ai21|amazon|anthropic|cohere|deepseek|luma|meta|minimax|mistral"
    r"|moonshotai|nvidia|openai|qwen|stability|twelvelabs|writer)\."
)
# Bedrock's own version tag — model identity, not a date. Two real shapes:
# ``-v1:0`` / ``-v0:2`` (explicit ``v``), and a BARE numeric tag ``-1:0``
# (``us.openai.gpt-oss-120b-1:0``, live in the catalog today). So the ``v`` is
# optional WHEN a ``:<n>`` part is present; without a colon an explicit ``v`` is
# required. That second clause is the guard that keeps a dated snapshot suffix
# (``-20251001``) out — a bare ``-\d+$`` would eat it and corrupt the family gate.
_BEDROCK_VERSION_SUFFIX = re.compile(r"(?:-v?\d+:\d+|-v\d+)$")


def _bedrock_family_candidates(model):
    """Forms a provider may echo for a Bedrock id, one wrapper peeled at a time.

    Region-less, then vendor-less, then version-less. Empty when ``model`` is not
    shaped like a Bedrock id — the vendor namespace is REQUIRED after the
    optional region, which is what keeps the widening tightly shape-gated:
    anything not unmistakably a Bedrock id gets no family widening at all.
    ``model`` is the already stripped+lowercased requested string.
    """
    no_region = _BEDROCK_REGION_PREFIX.sub("", model, count=1)
    if not _BEDROCK_VENDOR_NS.match(no_region):
        return []
    no_vendor = _BEDROCK_VENDOR_NS.sub("", no_region, count=1)
    no_version = _BEDROCK_VERSION_SUFFIX.sub("", no_vendor, count=1)
    # ``model`` itself is already covered by the exact-match branch above.
    return [c for c in (no_region, no_vendor, no_version) if c and c != model]


def prefer_requested_model(requested_model, extracted_model):
    """Pick the model string to report when the response echoes another one.

    The auto-instrumented span path reports the (post-rewrite) REQUEST model
    while the manual taps read the provider's echo, so a single rerouted model
    splits into two rows in cost-by-model. Canonical form is the requested
    model — but only when the echo is the SAME family, i.e. the request plus a
    dated snapshot suffix, or a Bedrock id whose echo dropped the region/vendor
    wrapper. A genuinely different served model (OpenRouter auto-routing, a
    gateway substitution) fails those gates and keeps its echo.
    """
    if not isinstance(requested_model, str) or not isinstance(extracted_model, str):
        return extracted_model
    requested = requested_model.strip().lower()
    extracted = extracted_model.strip().lower()
    if not requested or not extracted:
        return extracted_model
    if requested == extracted:
        return extracted_model
    if strip_date_suffix(extracted) == requested:
        return requested_model
    # Bedrock: the echo is the requested id with one or more of its wrappers
    # (region, vendor, version tag) peeled off, ± a dated snapshot suffix. Same
    # family → report the requested id verbatim, because the region-qualified
    # profile id is what AWS actually bills.
    for candidate in _bedrock_family_candidates(requested):
        if candidate == extracted or strip_date_suffix(extracted) == candidate:
            return requested_model
    return extracted_model
