"""
Canonical span_kind taxonomy + derivation (kept in sync with the server's
authoritative derivation).

span_kind describes what a span row *is*:

  - Structural anchors (no usage / no cost): ``agent`` (workflow / sub-agent root),
    ``tool`` (tool / function execution), ``chain`` (composite/chain anchor).
  - Billable model calls: ``llm`` (text chat), plus per-modality kinds derived from
    the call's ``operation``: embedding / image / tts / stt / video / ocr / rerank /
    moderation.

span_kind is re-derived authoritatively server-side from ``operation`` regardless of
what the SDK sends, so this is a best-effort, fail-safe convenience: it must never
raise into customer code — any failure falls back to ``"llm"``.
"""

# Structural, no-usage anchor rows. Trusted verbatim; never re-derived.
STRUCTURAL_SPAN_KINDS = frozenset({"agent", "tool", "chain"})

# operation -> span_kind. chat/unknown collapse to the text-chat default `llm`.
_OPERATION_TO_SPAN_KIND = {
    "chat": "llm",
    "unknown": "llm",
    "embedding": "embedding",
    "image_gen": "image",
    "audio_tts": "tts",
    "audio_stt": "stt",
    "video_gen": "video",
    "ocr": "ocr",
    "rerank": "rerank",
    "moderation": "moderation",
}


def span_kind_for(operation, incoming_kind=None):
    """Resolve the modality-aware span_kind.

    Structural kinds the caller already set (agent/tool/chain) are returned
    verbatim — those rows carry the default ``operation="chat"`` which must NOT
    turn them into ``llm``. Everything else derives from ``operation``,
    defaulting to ``llm``. Never raises.
    """
    try:
        if incoming_kind and incoming_kind in STRUCTURAL_SPAN_KINDS:
            return incoming_kind
        return _OPERATION_TO_SPAN_KIND.get(operation, "llm")
    except Exception:
        return "llm"
