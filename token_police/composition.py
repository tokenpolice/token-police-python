"""
Prompt & Response Composition Parser.

Privacy-preserving decomposition of LLM payloads into structural metadata.
Extracts role, type, length, and hash of each message segment WITHOUT storing
the actual text content.

3-Tier Fallback Strategy:
  1. Supported provider (OpenAI, Anthropic, Google GenAI) — exact per-message parsing
  2. OpenAI-compatible format — try messages[] as best-effort
  3. Complete fallback — single entry with role="complete_prompt"
"""
import hashlib
import logging
import math
import re
import types
from typing import Any, Dict, List, Optional

logger = logging.getLogger("token_police")


def _fast_hash(text: str) -> str:
    """SHA-1 hash truncated to 16 hex chars. Fast and sufficient for fingerprinting."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()[:16]


# ── Canonical JSON serializer ─────────────────────────────────
# One serializer routed through EVERY object-valued tool-arg site so the
# hashed STRING (and therefore the composition `hash`) is byte-identical to
# the Node SDK's `canonicalJson` for the same logical value. We do NOT delegate
# to `json.dumps` because its defaults diverge from the Node side (e.g. floats
# render `1.0` / `1e-05`, `NaN`/`Infinity` are emitted as invalid literals) and
# we need an integer-valued-float / non-finite / plain-decimal number rule plus
# the pinned escape table below.
#
# PURE + TOTAL + NO-THROW: it does no I/O; any value handling that could raise
# (a throwing __str__, an unconvertible object) is caught by the top-level guard
# which degrades to a string form. Tool args reaching the emit sites are JSON
# that arrived over the wire, so the non-JSON-native fallback is only a safety
# net. Exported-by-convention so the tool-span sink can reuse it.

# PINNED escape table = exact intersection of JS `JSON.stringify` and CPython
# `json.dumps(ensure_ascii=False)`. `/`, U+2028 and U+2029 are NOT escaped; all
# non-ASCII (incl. astral) is emitted literally.
_CANON_ESCAPE = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


# Character class covering EXACTLY what the loop below escapes: the two
# structural chars `"` (U+0022) and `\` (U+005C), plus every control char
# U+0000–U+001F (which subsumes the \b \t \n \f \r keys of _CANON_ESCAPE, all of
# which are < 0x20). If a string contains NONE of these, the loop would emit it
# verbatim, so we can wrap it in quotes directly — byte-for-byte identical output
# with no per-char Python loop. Keep this class in lock-step with the loop; the
# canonical bytes feed cross-SDK hashes, so any drift is a hard-contract break.
_CANON_NEEDS_ESCAPE_RE = re.compile(r'[\x00-\x1f"\\]')


def _canon_encode_string(s: str) -> str:
    # Fast path: no char needs escaping -> return the raw string quoted. This is
    # the overwhelmingly common case (clean UTF-8 text) and avoids the per-char
    # loop entirely.
    if not _CANON_NEEDS_ESCAPE_RE.search(s):
        return '"' + s + '"'
    out = ['"']
    for ch in s:
        esc = _CANON_ESCAPE.get(ch)
        if esc is not None:
            out.append(esc)
        elif ord(ch) < 0x20:
            out.append("\\u%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def _canon_expand_decimal(r: str) -> str:
    """Re-emit a shortest-round-trip decimal string (``repr(float)``) as a
    canonical PLAIN decimal — never exponential, no trailing fraction zeros —
    matching JS number formatting. Operates on the STRING representation (does
    not reimplement float->digits), so Python and Node converge on the same
    significant digits."""
    sign = ""
    if r.startswith("-"):
        sign = "-"
        r = r[1:]
    mantissa = r
    exp = 0
    idx = -1
    for i, ch in enumerate(r):
        if ch in "eE":
            idx = i
            break
    if idx >= 0:
        mantissa = r[:idx]
        exp = int(r[idx + 1:])
    if "." in mantissa:
        int_part, frac = mantissa.split(".", 1)
    else:
        int_part, frac = mantissa, ""
    digits = int_part + frac
    k = len(int_part) + exp
    if k <= 0:
        out = "0." + ("0" * (-k)) + digits
    elif k >= len(digits):
        out = digits + ("0" * (k - len(digits)))
    else:
        out = digits[:k] + "." + digits[k:]
    if "." in out:
        out = out.rstrip("0")
        if out.endswith("."):
            out = out[:-1]
    if sign and any(c in "123456789" for c in out):
        out = sign + out
    return out


def _canon_encode_number(v: float) -> str:
    if math.isnan(v) or math.isinf(v):
        return "null"  # NaN / +-inf -> null (JSON has no such literal)
    if v == 0:
        return "0"  # collapse 0.0 and -0.0
    return _canon_expand_decimal(repr(v))


def _canon_encode(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):  # MUST precede int (bool is an int subclass)
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)  # exact (arbitrary precision; >2^53 is the named residual)
    if isinstance(value, float):
        return _canon_encode_number(value)
    if isinstance(value, str):
        return _canon_encode_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_canon_encode(v) for v in value) + "]"
    if isinstance(value, dict):
        items = sorted((str(k), v) for k, v in value.items())
        return "{" + ",".join(_canon_encode_string(k) + ":" + _canon_encode(v) for k, v in items) + "}"
    # Non-JSON-native (datetime / custom) values fall back to a language-local
    # string form; deterministic within this SDK and guaranteed not to throw,
    # but NOT guaranteed byte-identical to the Node SDK's equivalent
    # serialization.
    return _canon_encode_string(str(value))


def _canonical_json(value: Any) -> str:
    """Canonical JSON serializer — byte-identical to the Node SDK's
    ``canonicalJson`` for JSON-native values. PURE, TOTAL and NO-THROW: an
    unconvertible / hostile value is caught here and degraded to a string form
    rather than raising."""
    try:
        return _canon_encode(value)
    except Exception:
        try:
            return _canon_encode_string(str(value))
        except Exception:
            return '""'


def _hash_len(val: Any):
    """(sha1-16 hex, length) for a tool arg/result value. Raw content is never
    kept. THE single home for this pair — previously duplicated verbatim in
    ``telemetry.py`` and ``openai_agents.py``, and shared with the pydantic_ai
    tool-row emit (``enforcer._emit_pydantic_ai_tool_row``). It lives here beside
    the canonical serializer to avoid an import cycle, mirroring the Node SDK's
    ``hashLen`` in the Node SDK's ``composition.ts``.

    String args are hashed byte-unchanged (preserves the OTel hash baseline);
    non-string args route through the canonical serializer (total/no-throw) for
    cross-SDK byte-identity with the Node SDK. ``len()`` already counts Unicode
    code points, matching Node's code-point length. TOTAL + NO-THROW."""
    if val is None:
        return "", 0
    s = val if isinstance(val, str) else _canonical_json(val)
    if not s:
        return "", 0
    return hashlib.sha1(s.encode("utf-8", errors="replace")).hexdigest()[:16], len(s)


def _text_entry(role: str, content: Any, name: Optional[str] = None) -> Dict[str, Any]:
    """Build a text composition entry.

    Length + hash are computed over the whitespace-trimmed content so the SAME
    logical text fingerprints identically across calls even when a framework
    reformats surrounding whitespace between turns (e.g. CrewAI rstrips an
    assistant message before feeding it into the next call). Trimming is safe —
    these are privacy-preserving fingerprints, never the stored text.

    This helper is TOTAL: a malformed (non-string) text part must degrade to an
    empty-text entry, never abort the whole composition — otherwise one bad part
    would collapse the entire call down to a single coarse fallback entry,
    silently losing per-message granularity. ``None`` → empty text; any other
    non-string is routed through the canonical serializer so Python and Node
    fingerprint the same JSON-shaped value identically (a bare ``str()`` would
    diverge on floats/bools/dicts), with an empty-text fallback if that fails.
    Covered by ``test_text_entry_malformed_never_raises`` /
    ``test_null_text_part_keeps_per_message_composition`` /
    ``test_non_string_text_parity``.
    """
    if isinstance(content, str):
        text = content
    elif content is None:
        text = ""
    else:
        try:
            text = _canonical_json(content)
        except Exception:
            text = ""
    normalized = text.strip()
    entry: Dict[str, Any] = {
        "role": role,
        "type": "text",
        "length": len(normalized),
        "hash": _fast_hash(normalized),
    }
    if name:
        entry["name"] = name
    return entry


def _non_text_entry(role: str, media_type: str, name: Optional[str] = None) -> Dict[str, Any]:
    """Build a non-text (image/audio/etc) composition entry."""
    entry: Dict[str, Any] = {"role": role, "type": media_type}
    if name:
        entry["name"] = name
    return entry


def _parse_embedding_input(provider_lower: str, kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build composition for an embeddings call.

    Embeddings have only input (no role/turn structure). Role is fixed to
    ``"input"``. Accepts:
      * string → 1 text entry
      * list of strings → N text entries
      * list of integers (OpenAI pre-tokenized) → 1 text entry with length only
      * Cohere multimodal segments → typed entries (text / image)

    Provider-specific kwarg keys checked: ``input`` (OpenAI/Together),
    ``texts`` (Cohere/Voyage), ``inputs`` (Mistral/Cohere multimodal),
    ``contents`` (Google), positional ``text`` (HuggingFace — passed as
    kwargs by the wrapper).
    """
    if not isinstance(kwargs, dict):
        return []
    payload = (
        kwargs.get("input")
        or kwargs.get("texts")
        or kwargs.get("inputs")
        or kwargs.get("contents")
        or kwargs.get("text")
    )
    if payload is None:
        return []

    out: List[Dict[str, Any]] = []
    if isinstance(payload, str):
        out.append(_text_entry("input", payload))
        return out
    if isinstance(payload, (list, tuple)):
        # list[int] = pre-tokenized single input (OpenAI accepts this shape).
        if payload and all(isinstance(t, int) for t in payload):
            out.append({"role": "input", "type": "text", "length": len(payload)})
            return out
        for item in payload:
            if isinstance(item, str):
                out.append(_text_entry("input", item))
            elif isinstance(item, dict):
                # Cohere multimodal / Mistral content-chunk shape.
                t = (item.get("type") or "").lower()
                if t in ("image", "image_url"):
                    out.append(_non_text_entry("input", "image"))
                elif t == "text":
                    text = item.get("text") or ""
                    out.append(_text_entry("input", text if isinstance(text, str) else str(text)))
                else:
                    # Unknown chunk type — record presence without leaking content.
                    out.append({"role": "input", "type": t or "unknown"})
            elif isinstance(item, (list, tuple)) and item and all(isinstance(t, int) for t in item):
                # list[list[int]] — N pre-tokenized inputs.
                out.append({"role": "input", "type": "text", "length": len(item)})
            else:
                # Fallback to a structural marker so callers know N inputs landed.
                out.append({"role": "input", "type": "unknown"})
        return out
    # Anything else (rare) — emit a single fallback entry rather than nothing.
    return [{"role": "input", "type": "unknown"}]


def _try_modality_prompt(kwargs: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Best-effort prompt composition for non-text generation calls.

    Matches when kwargs looks like an image/audio/video call (no ``messages``
    or ``contents`` field, but a ``prompt`` / ``input`` / ``text`` / ``file``).
    Returns ``None`` so the regular chat parsers stay in charge for text calls.
    """
    if not isinstance(kwargs, dict):
        return None
    if "messages" in kwargs or "contents" in kwargs or "input" in kwargs and isinstance(kwargs.get("input"), list):
        # `input` as a list belongs to the Responses API — skip.
        if "messages" in kwargs or "contents" in kwargs or isinstance(kwargs.get("input"), list):
            return None
    out: List[Dict[str, Any]] = []
    prompt = kwargs.get("prompt")
    if isinstance(prompt, str) and prompt:
        out.append(_text_entry("user", prompt))
    text = kwargs.get("input") or kwargs.get("text")
    if not out and isinstance(text, str) and text:
        out.append(_text_entry("user", text))
    audio_file = kwargs.get("file") or kwargs.get("audio")
    if audio_file is not None:
        name = None
        if isinstance(audio_file, str):
            name = audio_file
        elif hasattr(audio_file, "name") and isinstance(getattr(audio_file, "name"), str):
            name = audio_file.name
        out.append(_non_text_entry("user", "audio", name=name))
    return out or None


# Audio MIME types we recognise as "this response is binary audio, NOT text".
_AUDIO_MIME_PREFIXES = ("audio/", "application/octet-stream")
# Class names that are known binary audio wrappers across providers.
_BINARY_AUDIO_CLASS_HINTS = (
    "HttpxBinaryResponseContent",      # openai._legacy_response
    "BinaryAPIResponse",                # openai._response
    "AsyncBinaryAPIResponse",           # openai._response
    "StreamedBinaryAPIResponse",        # openai._response
    "AsyncStreamedBinaryAPIResponse",   # openai._response
)
# Module prefixes that wrap binary audio content (positive identification —
# safer than just substring-matching the class name).
_BINARY_AUDIO_MODULE_HINTS = (
    "openai._legacy_response",
    "openai._response",
)


def _looks_like_binary_audio(response: Any) -> bool:
    """Positive identification of a binary-audio response wrapper.

    Layered checks (any one is sufficient) — every accessor is wrapped to make
    sure an unrelated AttributeError can't bypass the check:

      1. Raw bytes / bytearray.
      2. Class name in the known-wrappers list (HttpxBinaryResponseContent,
         BinaryAPIResponse, …).
      3. Module path under ``openai._legacy_response`` / ``openai._response``.
      4. Bytes-iterator shape: ``read`` + ``iter_bytes`` + (``aiter_bytes`` or
         ``stream_to_file`` or ``content`` or ``write_to_file``).
      5. The underlying ``response.headers['content-type']`` reports audio /
         octet-stream (handles the case where the SDK exposes the raw
         ``httpx.Response`` directly).
      6. ``Binary`` / ``HttpxBinary`` substring in the class name as a final
         catch-all for vendor-private wrappers we haven't seen yet.

    Every branch swallows its own exceptions — this helper must never raise.
    """
    try:
        if isinstance(response, (bytes, bytearray, memoryview)):
            return True
    except Exception:
        pass
    cls = None
    try:
        cls = type(response)
    except Exception:
        return False
    try:
        cls_name = cls.__name__ or ""
    except Exception:
        cls_name = ""
    try:
        cls_module = cls.__module__ or ""
    except Exception:
        cls_module = ""
    try:
        if cls_name in _BINARY_AUDIO_CLASS_HINTS:
            return True
    except Exception:
        pass
    try:
        if any(cls_module.startswith(m) for m in _BINARY_AUDIO_MODULE_HINTS):
            return True
    except Exception:
        pass
    # Bytes-iterator duck-typing: needs `iter_bytes` (or async variant) AND
    # `read` (or `content`/`stream_to_file`) — chat / responses-style objects
    # never expose this combination.
    try:
        has_iter = hasattr(response, "iter_bytes") or hasattr(response, "aiter_bytes")
        has_read = (hasattr(response, "read") or hasattr(response, "aread")
                    or hasattr(response, "content") or hasattr(response, "stream_to_file")
                    or hasattr(response, "write_to_file"))
        if has_iter and has_read:
            return True
    except Exception:
        pass
    # httpx.Response (or a thin wrapper) — sniff the content-type header.
    try:
        headers = getattr(response, "headers", None)
        if headers is not None:
            ct = None
            # Most httpx-like header containers support .get()
            try:
                ct = headers.get("content-type")
            except Exception:
                try:
                    ct = headers["content-type"]
                except Exception:
                    ct = None
            if isinstance(ct, str):
                ct_lower = ct.lower().split(";", 1)[0].strip()
                if any(ct_lower.startswith(p) for p in _AUDIO_MIME_PREFIXES):
                    return True
    except Exception:
        pass
    # Last-resort substring sniff for vendor wrappers (e.g. some forks rename
    # the class). Keeps backward-compat with the previous behaviour.
    try:
        if "Binary" in cls_name or "HttpxBinary" in cls_name:
            return True
    except Exception:
        pass
    return False


def _is_structured_chat_response(response: Any) -> bool:
    """True when ``response`` is a chat/completion body that may also expose a
    convenience ``.text`` getter.

    Used to skip the transcription-style ``.text`` short-circuit in
    ``_try_modality_response`` so multi-part bodies reach their real parsers.
    Mirrors Node ``isStructuredChat`` (candidates / choices / content /
    output / message) and also treats a list ``parts`` as structured so
    pydantic_ai ``ModelResponse`` (``.text`` + ``.parts``) is not swallowed
    .

    List-only for container fields: e.g. xAI ``.content`` is a str and must
    not count. Checked BEFORE reading ``.text`` — google-genai's multi-part
    ``.text`` getter can warn on the console.
    """
    def _field(key: str) -> Any:
        if isinstance(response, dict):
            return response.get(key)
        return getattr(response, key, None)

    for key in ("candidates", "choices", "content", "output", "parts"):
        if isinstance(_field(key), list):
            return True
    if _field("message") is not None:
        return True
    return False


def _try_modality_response(response: Any) -> Optional[List[Dict[str, Any]]]:
    """Best-effort response composition for non-text generation calls.

    Order matters: the binary-audio check runs FIRST so it can't be bypassed
    by an attribute access raising on the response object (which would jump
    the outer try/except and let a downstream parser reach for ``.text``
    — that ``.text`` on ``HttpxBinaryResponseContent`` returns the raw audio
    bytes decoded as a ~100KB string and was the original source of the
    misclassified text rows in production).
    """
    if response is None:
        return None
    # Binary-audio detection FIRST and isolated from the rest of the checks
    # so an AttributeError elsewhere can't accidentally skip it.
    try:
        if _looks_like_binary_audio(response):
            return [_non_text_entry("assistant", "audio")]
    except Exception:
        pass
    try:
        # OpenAI images.generate / DALL-E / Together images
        data = None
        if isinstance(response, dict):
            data = response.get("data")
        else:
            data = getattr(response, "data", None)
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                if "url" in first or "b64_json" in first:
                    return [_non_text_entry("assistant", "image") for _ in data]
            else:
                if hasattr(first, "url") or hasattr(first, "b64_json"):
                    return [_non_text_entry("assistant", "image") for _ in data]
        # Google Imagen
        images = getattr(response, "generated_images", None) or getattr(response, "images", None)
        if isinstance(images, list) and images and not isinstance(images[0], (str, int, float)):
            # Image objects from google-genai.
            if hasattr(images[0], "image") or hasattr(images[0], "_image") or hasattr(images[0], "to_dict"):
                return [_non_text_entry("assistant", "image") for _ in images]
        # Veo / video gen — operation-style result with a `video` or `videos` field.
        videos = getattr(response, "generated_videos", None) or getattr(response, "videos", None)
        if isinstance(videos, list) and videos:
            return [_non_text_entry("assistant", "video") for _ in videos]
        # OpenAI / Mistral / HF transcription — text under `.text` only.
        # NOTE: this branch is the historical source of the TTS misclassification
        # bug because `HttpxBinaryResponseContent.text` decodes the audio bytes
        # as a string. The `_looks_like_binary_audio` check above short-circuits
        # such responses, but we keep an extra defensive guard here: only treat
        # `.text` as a transcription if the response also exposes the typical
        # transcription attributes (e.g. `.words`, `.segments`, `.language`,
        # `.duration`) OR carries no bytes-iterator interface at all. Without
        # this, any future SDK wrapper that adds a `.text` property without
        # being caught by the binary check would re-introduce the regression.
        #
        # Structured chat bodies (Gemini GenerateContentResponse, OpenAI chat,
        # Anthropic Message, Responses API, pydantic_ai ModelResponse) often
        # expose a convenience `.text` that concatenates text parts. Matching
        # that here used to swallow sibling function_call / tool_use parts
        # . Whisper/HF transcription bodies carry none of these
        # containers. Checked BEFORE touching `.text` (Node parity; google-genai
        # multi-part `.text` can log a console warning).
        if _is_structured_chat_response(response):
            return None
        text = getattr(response, "text", None)
        if isinstance(text, str) and text:
            # Heuristic: a real transcription has at least one of these.
            looks_like_transcription = any(
                hasattr(response, attr) for attr in (
                    "words", "segments", "language", "duration", "task",
                )
            )
            # ANY bytes-iterator surface → almost certainly audio bytes, not text.
            looks_like_binary = (
                hasattr(response, "iter_bytes") or hasattr(response, "aiter_bytes")
                or hasattr(response, "stream_to_file") or hasattr(response, "write_to_file")
            )
            if looks_like_transcription and not looks_like_binary:
                return [_text_entry("assistant", text)]
            if looks_like_binary:
                return [_non_text_entry("assistant", "audio")]
            # Otherwise (plain object exposing only `.text`) treat as text —
            # this is the HF/Mistral transcription shape.
            return [_text_entry("assistant", text)]
    except Exception:
        return None
    return None


def _ocr_composition(response: Any) -> List[Dict[str, Any]]:
    """OCR response composition (Mistral OCR and any future ``mistral_ocr``-
    shaped body). Privacy-preserving structural markers only — the per-page
    text is NEVER read. One ``{"role":"assistant","type":"ocr_page"}`` per
    billed page (capped at 100), or a single
    ``{"role":"assistant","type":"ocr_document"}`` when the page count is
    unknown/zero (still records that an OCR happened).

    Page count comes from ``usage_info.pages_processed`` (the value the SDK
    bills as ``ocr_pages``), falling back to ``len(pages)`` only when ``pages``
    is a real list/tuple. Total/no-throw: any error returns the single-document
    fallback.
    """
    document_fallback = [_non_text_entry("assistant", "ocr_document")]
    try:
        if response is None:
            return document_fallback

        # usage_info: dict-or-object safe.
        if isinstance(response, dict):
            usage_info = response.get("usage_info")
        else:
            usage_info = getattr(response, "usage_info", None)

        n = 0
        # Primary source: usage_info.pages_processed → int.
        raw_pages = None
        if isinstance(usage_info, dict):
            raw_pages = usage_info.get("pages_processed")
        elif usage_info is not None:
            raw_pages = getattr(usage_info, "pages_processed", None)
        try:
            if isinstance(raw_pages, bool):
                parsed = None  # bool is an int subclass — reject it explicitly.
            elif isinstance(raw_pages, int):
                parsed = raw_pages
            elif isinstance(raw_pages, (float, str)):
                parsed = int(raw_pages)
            else:
                parsed = None
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and parsed > 0:
            n = parsed
        else:
            # Fallback: length of the `pages` sequence (only if it really is one).
            if isinstance(response, dict):
                pages = response.get("pages")
            else:
                pages = getattr(response, "pages", None)
            if isinstance(pages, (list, tuple)):
                n = len(pages)

        if n > 0:
            count = min(n, 100)
            return [_non_text_entry("assistant", "ocr_page") for _ in range(count)]
        return document_fallback
    except Exception:
        return document_fallback


def _as_dict(obj: Any) -> Any:
    """Coerce a Pydantic model (e.g. an SDK response content block) to a dict.
    Also coerces ``types.SimpleNamespace`` — used by raw-HTTP wrappers built
    via ``tp.protect(..., manual=True)`` to expose parsed JSON as attribute-
    accessible objects. Returns the object unchanged if it is none of these."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, types.SimpleNamespace):
        return obj.__dict__
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return obj


def _jsonable(obj: Any, _depth: int = 0) -> Any:
    """Recursively coerce namespace-shaped objects (``types.SimpleNamespace``
    or anything carrying ``__dict__``) into plain JSON-encodable structures.

    Raw-HTTP callers (e.g. the Gemini REST apps) decode the JSON body with
    ``object_hook=SimpleNamespace``, so a functionCall's ``args`` is itself a
    SimpleNamespace (recursively) — ``dict(args)`` raises TypeError on it.
    Depth-capped and fail-soft: anything unconvertible becomes ``str(obj)``.
    """
    if _depth > 8:
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, _depth + 1) for v in obj]
    if isinstance(obj, types.SimpleNamespace):
        return {str(k): _jsonable(v, _depth + 1) for k, v in vars(obj).items()}
    # Mapping-like (e.g. protobuf MapComposite from google.generativeai) —
    # keep this BEFORE the generic __dict__ probe so proto internals don't leak.
    try:
        return {str(k): _jsonable(v, _depth + 1) for k, v in dict(obj).items()}
    except Exception:
        pass
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        try:
            return {str(k): _jsonable(v, _depth + 1) for k, v in d.items()}
        except Exception:
            pass
    return str(obj)


def _serialize_tool_args(args: Any) -> str:
    """Serialize a tool/function-call ``args`` payload (dict, SimpleNamespace,
    str, proto map, ...) to a JSON-ish string for hashing/length only.
    Never raises — falls back to ``str(args)``, then ""."""
    try:
        if isinstance(args, str):
            return args
        return _canonical_json(_jsonable(args))
    except Exception:
        try:
            return str(args)
        except Exception:
            return ""


# ═══════════════════════════════════════════════════════════════════
# Tier 1: Provider-specific parsers
# ═══════════════════════════════════════════════════════════════════

# xai-sdk uses protobuf objects (chat_pb2.Message / Content / ToolCall / Response).
# Role is an enum int — look up the symbolic name via chat_pb2.MessageRole.Name()
# when the proto is importable, else fall back to a fixed integer map matching
# the proto definition in xai_sdk.proto.chat_pb2.
_XAI_ROLE_INT_FALLBACK = {
    0: "unknown", 1: "system", 2: "user", 3: "assistant", 4: "tool",
    5: "developer",
}


def _xai_role_name(role_value: Any) -> str:
    """Convert an xai_sdk proto role (int or enum) to an OpenAI-style role string."""
    if isinstance(role_value, str):
        v = role_value.upper().removeprefix("ROLE_").lower()
        return v or "user"
    try:
        from xai_sdk.proto import chat_pb2  # type: ignore
        name = chat_pb2.MessageRole.Name(int(role_value))
        return name.removeprefix("ROLE_").lower()
    except Exception:
        pass
    try:
        return _XAI_ROLE_INT_FALLBACK.get(int(role_value), "user")
    except Exception:
        return "user"


def _parse_xai_content_block(role: str, content_block: Any) -> List[Dict[str, Any]]:
    """One xai_sdk chat_pb2.Content → one or more composition entries."""
    if content_block is None:
        return []
    # Text content
    text = getattr(content_block, "text", None)
    if isinstance(text, str) and text:
        return [_text_entry(role, text)]
    # Image content
    if hasattr(content_block, "image_url"):
        iu = content_block.image_url
        # Protobuf optional sub-message: check via HasField when available; the
        # attribute exists but is "empty" when not set, so probe a sub-field.
        if iu is not None and (getattr(iu, "image_url", "") or getattr(iu, "detail", 0)):
            return [_non_text_entry(role, "image")]
    # File content
    if hasattr(content_block, "file"):
        f = content_block.file
        if f is not None and (
            getattr(f, "file_id", "") or getattr(f, "url", "") or getattr(f, "data", b"")
        ):
            return [_non_text_entry(role, "file")]
    return []


def _parse_xai_messages(messages: list) -> List[Dict[str, Any]]:
    """Parse xai_sdk's repeated chat_pb2.Message list into composition entries.

    Mirrors `_parse_openai_messages` semantics: text content → text_entry,
    image/file → non_text_entry, tool_calls → tool_call entries, ROLE_TOOL →
    tool_result entry.
    """
    result: List[Dict[str, Any]] = []
    for msg in messages or []:
        role = _xai_role_name(getattr(msg, "role", "user"))
        content = getattr(msg, "content", None) or []
        # `content` is a repeated Content field — iterate parts.
        if role == "tool":
            tool_text_parts: List[str] = []
            for block in content:
                t = getattr(block, "text", None)
                if isinstance(t, str) and t:
                    tool_text_parts.append(t)
            tool_call_id = getattr(msg, "tool_call_id", "") or ""
            result.append(_text_entry(
                "tool_result", "".join(tool_text_parts), name=tool_call_id or None,
            ))
        else:
            for block in content:
                result.extend(_parse_xai_content_block(role, block))
            # Assistant reasoning trace (only set for reasoning models)
            reasoning = getattr(msg, "reasoning_content", None)
            if isinstance(reasoning, str) and reasoning:
                result.append(_text_entry(role, reasoning))
            # Assistant tool calls
            tool_calls = list(getattr(msg, "tool_calls", []) or [])
            for tc in tool_calls:
                fn = getattr(tc, "function", None)
                fn_name = getattr(fn, "name", "") if fn is not None else ""
                fn_args = getattr(fn, "arguments", "") if fn is not None else ""
                entry = _text_entry("tool_call", fn_args or "", name=fn_name or "")
                entry["type"] = "tool_call"
                result.append(entry)
    return result


def _parse_xai_response(response: Any) -> List[Dict[str, Any]]:
    """Parse an xai_sdk Response object (non-streaming) into response composition."""
    if response is None:
        return []
    result: List[Dict[str, Any]] = []
    # Reasoning trace (optional)
    try:
        reasoning = response.reasoning_content
        if isinstance(reasoning, str) and reasoning:
            result.append(_text_entry("assistant", reasoning))
    except Exception:
        pass
    # Main content
    try:
        content = response.content
        if isinstance(content, str) and content:
            result.append(_text_entry("assistant", content))
    except Exception:
        pass
    # Tool calls
    try:
        for tc in (response.tool_calls or []):
            fn = getattr(tc, "function", None)
            fn_name = getattr(fn, "name", "") if fn is not None else ""
            fn_args = getattr(fn, "arguments", "") if fn is not None else ""
            entry = _text_entry("tool_call", fn_args or "", name=fn_name or "")
            entry["type"] = "tool_call"
            result.append(entry)
    except Exception:
        pass
    return result


def _parse_openai_messages(messages: list) -> List[Dict[str, Any]]:
    """Parse OpenAI-format messages array."""
    result = []
    # tool_call_id -> tool name. OpenAI/Cohere tool messages carry only the
    # tool_call_id (no name); resolve the human-readable name from the matching
    # assistant tool_call so `tool_result` segments are attributable.
    tool_names: Dict[str, str] = {}
    for msg in messages:
        # Skip null/primitive/non-object elements (a stray None, int, or bare
        # string in the array): vars(msg) below raises TypeError on a scalar, and
        # the outer catch would then collapse the WHOLE composition to a coarse
        # single-entry fallback, losing per-message granularity for every valid
        # message. Mirrors the Node parser, which emits no entry for such
        # elements. A None or scalar has no attributes to project, so there is
        # nothing to parse.
        # Convert Pydantic models to dict if necessary
        if hasattr(msg, "model_dump"):
            msg = msg.model_dump()
        elif hasattr(msg, "dict"):
            msg = msg.dict()
        elif hasattr(msg, "to_dict"):
            msg = msg.to_dict()
        elif not isinstance(msg, dict):
            if msg is None or not hasattr(msg, "__dict__"):
                continue
            msg = vars(msg)

        role = msg.get("role", "unknown")
        content = msg.get("content")

        # String content — most common case
        if isinstance(content, str) and role != "tool":
            result.append(_text_entry(role, content))
        # Multimodal content array (e.g. vision, audio)
        elif isinstance(content, list) and role != "tool":
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "text")
                    if part_type == "text":
                        result.append(_text_entry(role, part.get("text", "")))
                    elif part_type == "image_url":
                        result.append(_non_text_entry(role, "image"))
                    elif part_type == "input_audio":
                        result.append(_non_text_entry(role, "audio"))
                    else:
                        result.append(_non_text_entry(role, part_type))
                elif isinstance(part, str):
                    result.append(_text_entry(role, part))
        elif content is None:
            # Tool call messages often have null content
            pass

        # Cohere v2 assistant messages carry a `tool_plan` — the model's
        # chain-of-thought reflection before it emits tool calls. OpenAI-shaped
        # SDKs never set this field, so this is a no-op for them.
        tool_plan = msg.get("tool_plan")
        if isinstance(tool_plan, str) and tool_plan:
            result.append(_text_entry(role, tool_plan))

        # Tool calls in assistant messages
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                fn_name = fn.get("name", "")
                fn_args = fn.get("arguments", "")
                tc_id = tc.get("id", "") or ""
                if tc_id:
                    tool_names[tc_id] = fn_name
                entry = _text_entry("tool_call", fn_args, name=fn_name)
                entry["type"] = "tool_call"
                result.append(entry)

        # Tool result message
        if role == "tool":
            tool_content = content if isinstance(content, str) else _canonical_json(content or "")
            name = msg.get("name") or tool_names.get(msg.get("tool_call_id", ""), "")
            result.append(_text_entry("tool_result", tool_content, name=name))

    return result


def _parse_openai_responses_input(
    input_payload: Any,
    instructions: Any = None,
) -> List[Dict[str, Any]]:
    """Parse OpenAI Responses API ``input`` (+ top-level ``instructions``).

    The Responses input is either a single string (treated as a user message)
    or a list of items. Each item is one of:
      - ``{"role": "user"|"assistant"|"system"|"developer", "content": str | list}``
        where list elements are ``{"type": "input_text"|"output_text", "text": "..."}``
        or ``{"type": "input_image", ...}`` parts.
      - ``{"type": "function_call", "name": "...", "arguments": "<json>",
            "call_id": "..."}`` — assistant tool call.
      - ``{"type": "function_call_output", "call_id": "...", "output": "..."}``
        — tool result, name resolved via the matching prior function_call.
      - ``{"type": "reasoning", "summary": [...]}`` — emitted as a non-text entry.
    """
    result: List[Dict[str, Any]] = []
    # call_id -> tool name. function_call_output blocks only carry the call_id;
    # resolve the human-readable tool name from the matching function_call item.
    tool_names: Dict[str, str] = {}

    if instructions:
        if isinstance(instructions, str):
            result.append(_text_entry("system", instructions))
        elif isinstance(instructions, list):
            for block in instructions:
                block = _as_dict(block)
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                    if isinstance(text, str):
                        result.append(_text_entry("system", text))

    if input_payload is None:
        return result

    if isinstance(input_payload, str):
        if input_payload:
            result.append(_text_entry("user", input_payload))
        return result

    if not isinstance(input_payload, list):
        return result

    for item in input_payload:
        item = _as_dict(item)
        if isinstance(item, str):
            if item:
                result.append(_text_entry("user", item))
            continue
        if not isinstance(item, dict):
            continue

        itype = item.get("type")
        if itype == "function_call":
            name = item.get("name", "") or ""
            call_id = item.get("call_id", "") or ""
            if call_id:
                tool_names[call_id] = name
            entry = _text_entry("tool_call", item.get("arguments", "") or "", name=name)
            entry["type"] = "tool_call"
            result.append(entry)
            continue
        if itype == "function_call_output":
            call_id = item.get("call_id", "") or ""
            name = tool_names.get(call_id, call_id)
            output = item.get("output", "")
            if not isinstance(output, str):
                output = _canonical_json(output)
            result.append(_text_entry("tool_result", output, name=name))
            continue
        if itype == "reasoning":
            # Reasoning summaries carry no plaintext we want to hash; emit a
            # structural marker so the consumer knows the model thought.
            result.append(_non_text_entry("assistant", "reasoning"))
            continue

        role = item.get("role") or itype or "user"
        content = item.get("content")
        if isinstance(content, str):
            if content:
                result.append(_text_entry(role, content))
        elif isinstance(content, list):
            for part in content:
                part = _as_dict(part)
                if isinstance(part, dict):
                    ptype = part.get("type", "text") or "text"
                    if ptype in ("input_text", "output_text", "text", "summary_text"):
                        text = part.get("text", "") or ""
                        if text:
                            result.append(_text_entry(role, text))
                    elif ptype in ("input_image", "image_url", "image"):
                        result.append(_non_text_entry(role, "image"))
                    elif ptype in ("input_audio", "audio"):
                        result.append(_non_text_entry(role, "audio"))
                    else:
                        result.append(_non_text_entry(role, ptype))
                elif isinstance(part, str):
                    if part:
                        result.append(_text_entry(role, part))

    return result


def _parse_openai_responses_response(response: Any) -> List[Dict[str, Any]]:
    """Parse an OpenAI Responses API response (or accumulated synthetic dict).

    Walks ``response.output`` which is a list of items:
      - ``message`` items carry ``content`` of ``output_text`` parts.
      - ``function_call`` items carry ``name``, ``arguments`` (json string),
        ``call_id``.
      - ``reasoning`` items are reported as a structural marker.
      - ``image_generation_call`` items are reported as assistant/image
        (built-in image_generation tool output).
    """
    result: List[Dict[str, Any]] = []
    if isinstance(response, dict):
        output = response.get("output")
    else:
        output = getattr(response, "output", None)
    if not isinstance(output, list):
        return result

    for item in output:
        item = _as_dict(item)
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content", []) or []:
                part = _as_dict(part)
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype in ("output_text", "text"):
                    text = part.get("text", "") or ""
                    if text:
                        result.append(_text_entry("assistant", text))
                elif ptype == "refusal":
                    refusal = part.get("refusal", "") or ""
                    if refusal:
                        result.append(_text_entry("assistant", refusal))
                else:
                    result.append(_non_text_entry("assistant", ptype or "unknown"))
        elif itype == "function_call":
            name = item.get("name", "") or ""
            args = item.get("arguments", "") or ""
            if not isinstance(args, str):
                args = _canonical_json(args)
            entry = _text_entry("tool_call", args, name=name)
            entry["type"] = "tool_call"
            result.append(entry)
        elif itype == "reasoning":
            result.append(_non_text_entry("assistant", "reasoning"))
        elif itype == "image_generation_call":
            # Built-in image_generation tool output — a real billed image.
            # Without this entry the generated image is invisible in the
            # composition (only surrounding assistant text shows). Mirrors
            # Node composition.ts.
            result.append(_non_text_entry("assistant", "image"))

    return result


def _parse_anthropic_messages(messages: list, system: Any = None) -> List[Dict[str, Any]]:
    """Parse Anthropic-format messages (with top-level system param)."""
    result = []
    # tool_use_id -> tool name. Anthropic tool_result blocks only carry the
    # tool_use_id, so we resolve the human-readable name from the matching
    # tool_use block (which always precedes its result in the messages array).
    tool_names: Dict[str, str] = {}

    # Anthropic system prompt is a top-level parameter, not in messages[]
    if system:
        if isinstance(system, str):
            result.append(_text_entry("system", system))
        elif isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    result.append(_text_entry("system", block.get("text", "")))

    for msg in messages:
        msg = _as_dict(msg)
        # Skip elements that didn't coerce to a mapping (None, ints, bare
        # strings): calling `.get` on one raises (None.get -> AttributeError) and
        # the outer catch would collapse the WHOLE composition to a coarse
        # fallback, losing per-message granularity for every valid message.
        # Mirrors the Node parser, which emits no entry for such elements.
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "unknown")
        content = msg.get("content")

        if isinstance(content, str):
            result.append(_text_entry(role, content))
        elif isinstance(content, list):
            for block in content:
                block = _as_dict(block)
                if isinstance(block, dict):
                    block_type = block.get("type", "text")
                    if block_type == "text":
                        result.append(_text_entry(role, block.get("text", "")))
                    elif block_type == "image":
                        result.append(_non_text_entry(role, "image"))
                    elif block_type == "tool_use":
                        tool_name = block.get("name", "")
                        tool_id = block.get("id", "")
                        if tool_id:
                            tool_names[tool_id] = tool_name
                        inp = _canonical_json(block.get("input", {}))
                        entry = _text_entry("tool_call", inp, name=tool_name)
                        entry["type"] = "tool_call"
                        result.append(entry)
                    elif block_type == "tool_result":
                        tool_use_id = block.get("tool_use_id", "")
                        resolved_name = tool_names.get(tool_use_id, tool_use_id)
                        tr_content = block.get("content", "")
                        if isinstance(tr_content, list):
                            for sub in tr_content:
                                sub = _as_dict(sub)
                                if isinstance(sub, dict) and sub.get("type") == "text":
                                    result.append(_text_entry("tool_result", sub.get("text", ""), name=resolved_name))
                        elif isinstance(tr_content, str):
                            result.append(_text_entry("tool_result", tr_content, name=resolved_name))
                    else:
                        result.append(_non_text_entry(role, block_type))
                elif isinstance(block, str):
                    result.append(_text_entry(role, block))

    return result


def _parse_google_contents(contents: Any, system_instruction: Any = None) -> List[Dict[str, Any]]:
    """Parse Google GenAI ``contents`` format.

    Handles plain strings/dicts as well as ``google-genai`` Pydantic models
    (``Content`` / ``Part``), which the agent loop appends to ``contents`` when
    feeding a model response back in."""
    result = []

    # System instruction
    if system_instruction:
        if isinstance(system_instruction, str):
            result.append(_text_entry("system", system_instruction))
        else:
            sys_obj = _as_dict(system_instruction)
            if isinstance(sys_obj, dict):
                parts = sys_obj.get("parts", [])
                for p in parts:
                    p = _as_dict(p)
                    if isinstance(p, dict) and p.get("text"):
                        result.append(_text_entry("system", p["text"]))
                    elif isinstance(p, str):
                        result.append(_text_entry("system", p))

    if isinstance(contents, str):
        result.append(_text_entry("user", contents))
        return result

    if isinstance(contents, list):
        for content in contents:
            if isinstance(content, str):
                result.append(_text_entry("user", content))
                continue
            content = _as_dict(content)
            if not isinstance(content, dict):
                continue

            role = content.get("role", "user")
            parts = content.get("parts", [])
            for part in parts:
                if isinstance(part, str):
                    result.append(_text_entry(role, part))
                    continue
                part = _as_dict(part)
                if isinstance(part, dict):
                    # model_dump() of a google-genai Part includes every field
                    # (most as None), so check truthiness, not key presence.
                    # The google-genai Pydantic SDK uses snake_case keys; the
                    # native REST format (and apps hand-building `contents`)
                    # uses camelCase. Accept both, or tool-call / tool-result
                    # turns get silently dropped and the composition collapses
                    # to system+first-user — which can falsely trigger loop
                    # detection (identical-composition heuristics).
                    inline_data = _as_dict(part.get("inline_data") or part.get("inlineData"))
                    function_call = _as_dict(part.get("function_call") or part.get("functionCall"))
                    function_response = _as_dict(part.get("function_response") or part.get("functionResponse"))
                    if part.get("text") is not None:
                        result.append(_text_entry(role, part["text"]))
                    elif inline_data:
                        mime = inline_data.get("mime_type") or inline_data.get("mimeType") or "unknown"
                        media = "image" if "image" in mime else "audio" if "audio" in mime else mime
                        result.append(_non_text_entry(role, media))
                    elif function_call:
                        args = _canonical_json(function_call.get("args", {}) or {})
                        entry = _text_entry("tool_call", args, name=function_call.get("name", ""))
                        entry["type"] = "tool_call"
                        result.append(entry)
                    elif function_response:
                        resp = _canonical_json(function_response.get("response", {}) or {})
                        result.append(_text_entry("tool_result", resp, name=function_response.get("name", "")))

    return result


def _parse_bedrock_converse(messages: list, system: Any = None) -> List[Dict[str, Any]]:
    """Parse AWS Bedrock Converse API messages (with top-level system param).

    Converse content is always a list of typed blocks: {text}, {image},
    {document}, {video}, {toolUse}, {toolResult}.
    """
    result = []
    # toolUseId -> tool name, resolved from the toolUse block that precedes the
    # result (Converse toolResult blocks only carry the toolUseId).
    tool_names: Dict[str, str] = {}

    # Converse system prompt is a top-level list of content blocks.
    if system:
        if isinstance(system, str):
            result.append(_text_entry("system", system))
        elif isinstance(system, list):
            for block in system:
                block = _as_dict(block)
                if isinstance(block, dict) and "text" in block:
                    result.append(_text_entry("system", block.get("text", "")))

    for msg in messages:
        msg = _as_dict(msg)
        role = msg.get("role", "unknown")
        content = msg.get("content")

        if isinstance(content, str):
            result.append(_text_entry(role, content))
        elif isinstance(content, list):
            for block in content:
                block = _as_dict(block)
                if isinstance(block, str):
                    result.append(_text_entry(role, block))
                    continue
                if not isinstance(block, dict):
                    continue
                if "text" in block:
                    result.append(_text_entry(role, block.get("text", "")))
                elif "image" in block:
                    result.append(_non_text_entry(role, "image"))
                elif "document" in block:
                    result.append(_non_text_entry(role, "document"))
                elif "video" in block:
                    result.append(_non_text_entry(role, "video"))
                elif "toolUse" in block:
                    tu = _as_dict(block.get("toolUse")) or {}
                    tool_name = tu.get("name", "")
                    tool_id = tu.get("toolUseId", "")
                    if tool_id:
                        tool_names[tool_id] = tool_name
                    inp = _canonical_json(tu.get("input", {}))
                    entry = _text_entry("tool_call", inp, name=tool_name)
                    entry["type"] = "tool_call"
                    result.append(entry)
                elif "toolResult" in block:
                    tr = _as_dict(block.get("toolResult")) or {}
                    tool_use_id = tr.get("toolUseId", "")
                    resolved_name = tool_names.get(tool_use_id, tool_use_id)
                    tr_content = tr.get("content", [])
                    if isinstance(tr_content, list):
                        for sub in tr_content:
                            sub = _as_dict(sub)
                            if not isinstance(sub, dict):
                                continue
                            if "text" in sub:
                                result.append(_text_entry("tool_result", sub.get("text", ""), name=resolved_name))
                            elif "json" in sub:
                                result.append(_text_entry("tool_result", _canonical_json(sub.get("json", {})), name=resolved_name))
                            else:
                                result.append(_non_text_entry("tool_result", "content", name=resolved_name))
                    elif isinstance(tr_content, str):
                        result.append(_text_entry("tool_result", tr_content, name=resolved_name))
                else:
                    keys = list(block.keys())
                    result.append(_non_text_entry(role, keys[0] if keys else "unknown"))

    return result


def _parse_bedrock_converse_response(response: Any) -> List[Dict[str, Any]]:
    """Parse an AWS Bedrock Converse API response dict (output.message.content)."""
    result = []
    response = _as_dict(response)
    if not isinstance(response, dict):
        return result
    output = _as_dict(response.get("output"))
    if not isinstance(output, dict):
        return result
    message = _as_dict(output.get("message")) or {}
    content = message.get("content")
    if not isinstance(content, list):
        return result
    for block in content:
        block = _as_dict(block)
        if not isinstance(block, dict):
            continue
        if "text" in block:
            result.append(_text_entry("assistant", block.get("text", "")))
        elif "toolUse" in block:
            tu = _as_dict(block.get("toolUse")) or {}
            inp = _canonical_json(tu.get("input", {}))
            entry = _text_entry("tool_call", inp, name=tu.get("name", ""))
            entry["type"] = "tool_call"
            result.append(entry)
        elif "reasoningContent" in block:
            result.append(_non_text_entry("assistant", "reasoning"))
    return result


def _parse_cohere_response(response: Any) -> List[Dict[str, Any]]:
    """Parse a Cohere v2 chat response.

    The assistant output lives under ``response.message`` — a content block
    list ({type: "text", text}), an optional ``tool_plan`` string, and
    ``tool_calls`` ({id, type: "function", function: {name, arguments}}).
    """
    result = []
    message = getattr(response, "message", None)
    if message is None:
        return result
    message = _as_dict(message)
    if not isinstance(message, dict):
        return result

    # tool_plan: chain-of-thought reflection the model emits before tool calls.
    tool_plan = message.get("tool_plan")
    if isinstance(tool_plan, str) and tool_plan:
        result.append(_text_entry("assistant", tool_plan))

    content = message.get("content")
    if isinstance(content, str):
        result.append(_text_entry("assistant", content))
    elif isinstance(content, list):
        for block in content:
            block = _as_dict(block)
            if isinstance(block, dict):
                block_type = block.get("type", "text")
                if block_type == "text":
                    result.append(_text_entry("assistant", block.get("text", "")))
                else:
                    result.append(_non_text_entry("assistant", block_type))
            elif isinstance(block, str):
                result.append(_text_entry("assistant", block))

    for tc in message.get("tool_calls") or []:
        tc = _as_dict(tc)
        if not isinstance(tc, dict):
            continue
        fn = _as_dict(tc.get("function")) or {}
        fn_args = fn.get("arguments", "")
        if not isinstance(fn_args, str):
            fn_args = _canonical_json(fn_args)
        entry = _text_entry("tool_call", fn_args, name=fn.get("name", ""))
        entry["type"] = "tool_call"
        result.append(entry)

    return result


# ── LangChain ──────────────────────────────────────────────────────
# LangChain calls flow through BaseChatModel.{generate,agenerate,stream,
# astream}. The prompt is a payload of LangChain BaseMessage objects (or a
# string / PromptValue for stream); the response is an LLMResult whose
# generations carry AIMessage objects.

_LANGCHAIN_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "tool": "tool_result",
    "function": "tool_result",
    # Streaming chunks. LangChain's *MessageChunk classes override `.type` with
    # the class name (e.g. AIMessageChunk.type == "AIMessageChunk") instead of
    # inheriting the lowercase parent type — without these we'd surface
    # "AIMessageChunk" as the role in agent / streaming flows that fold chunks
    # back into the message history.
    "AIMessageChunk": "assistant",
    "HumanMessageChunk": "user",
    "SystemMessageChunk": "system",
    "ToolMessageChunk": "tool_result",
    "FunctionMessageChunk": "tool_result",
}


def _parse_langchain_message_obj(msg: Any) -> List[Dict[str, Any]]:
    """Parse a single LangChain BaseMessage (or chunk) into composition entries."""
    out: List[Dict[str, Any]] = []
    msg_type = getattr(msg, "type", None) or ""
    role = _LANGCHAIN_ROLE_MAP.get(msg_type, msg_type or "user")
    name = getattr(msg, "name", None) or None
    content = getattr(msg, "content", None)
    # A tool-calling AIMessage exposes its calls in BOTH content[] (as tool_use
    # blocks) AND the normalized `.tool_calls` list. Track whether we already
    # emitted them from content[] so the `.tool_calls` loop below doesn't double
    # them (same guard the LlamaIndex parser uses).
    tool_call_blocks_seen = False

    if isinstance(content, str):
        if content:
            if role == "tool_result":
                out.append(_text_entry("tool_result", content, name=name))
            else:
                out.append(_text_entry(role, content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                if part:
                    out.append(_text_entry(role, part))
            elif isinstance(part, dict):
                ptype = part.get("type", "text")
                if ptype == "text":
                    out.append(_text_entry(role, part.get("text", "")))
                elif ptype in ("image_url", "image"):
                    out.append(_non_text_entry(role, "image"))
                elif ptype in ("input_audio", "audio"):
                    out.append(_non_text_entry(role, "audio"))
                elif ptype == "tool_use":
                    inp = _canonical_json(part.get("input", {}))
                    entry = _text_entry("tool_call", inp, name=part.get("name", ""))
                    entry["type"] = "tool_call"
                    out.append(entry)
                    tool_call_blocks_seen = True
                else:
                    out.append(_non_text_entry(role, str(ptype)))

    # AIMessage.tool_calls — normalized [{name, args, id, type}] list. Skip when
    # the same calls were already emitted as content[] tool_use blocks above,
    # otherwise each call lands twice.
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls and not tool_call_blocks_seen:
        for tc in tool_calls:
            tc = _as_dict(tc)
            if not isinstance(tc, dict):
                continue
            fn_name = tc.get("name", "")
            fn_args = tc.get("args", {})
            args_str = fn_args if isinstance(fn_args, str) else _canonical_json(fn_args)
            entry = _text_entry("tool_call", args_str, name=fn_name)
            entry["type"] = "tool_call"
            out.append(entry)

    return out


def _coerce_langchain_messages(payload: Any) -> list:
    """Flatten a LangChain prompt payload into a flat list of message-like objects.

    Accepts: str, a single BaseMessage, list[BaseMessage], list[list[BaseMessage]]
    (the shape BaseChatModel.generate receives), a PromptValue, or list of
    (role, content) tuples / role-dicts.
    """
    if payload is None:
        return []
    # PromptValue → list[BaseMessage]
    if hasattr(payload, "to_messages"):
        try:
            return list(payload.to_messages())
        except Exception:
            pass
    if isinstance(payload, str):
        return [payload]
    # A single BaseMessage
    if hasattr(payload, "type") and hasattr(payload, "content"):
        return [payload]
    if isinstance(payload, list):
        flat = []
        for item in payload:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [payload]


def _parse_langchain_messages(payload: Any) -> List[Dict[str, Any]]:
    """Parse a LangChain prompt payload into composition entries."""
    result: List[Dict[str, Any]] = []
    for msg in _coerce_langchain_messages(payload):
        if isinstance(msg, str):
            if msg:
                result.append(_text_entry("user", msg))
        elif isinstance(msg, (list, tuple)) and len(msg) == 2 and isinstance(msg[0], str):
            role, content = msg
            text = content if isinstance(content, str) else _canonical_json(content)
            result.append(_text_entry(role, text))
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            text = content if isinstance(content, str) else _canonical_json(content)
            result.append(_text_entry(str(msg.get("role", "user")), text))
        elif hasattr(msg, "type"):
            result.extend(_parse_langchain_message_obj(msg))
    return result


def _parse_langchain_response(response: Any) -> List[Dict[str, Any]]:
    """Parse a LangChain LLMResult (returned by BaseChatModel.generate/agenerate)
    OR a bare message object (an AIMessageChunk folded from a stream / astream)."""
    result: List[Dict[str, Any]] = []
    generations = getattr(response, "generations", None)
    if not generations:
        # Streamed calls hand us the accumulated AIMessageChunk directly (it has
        # `.type`/`.content` but no `.generations`) — parse it as a message.
        if response is not None and hasattr(response, "type"):
            return _parse_langchain_message_obj(response)
        return result
    for gen_list in generations:
        if not isinstance(gen_list, (list, tuple)):
            gen_list = [gen_list]
        for gen in gen_list:
            msg = getattr(gen, "message", None)
            if msg is not None and hasattr(msg, "type"):
                result.extend(_parse_langchain_message_obj(msg))
            else:
                text = getattr(gen, "text", None)
                if text:
                    result.append(_text_entry("assistant", text))
    return result


# ── LlamaIndex ─────────────────────────────────────────────────────
# LlamaIndex calls flow through the provider LLM classes
# (llama_index.llms.{openai,anthropic,google_genai}) .{chat,achat,stream_chat,
# astream_chat}. Prompts arrive as List[ChatMessage]; responses are
# ChatResponse with a `.message` ChatMessage attached. Streamed chunks may
# carry incremental `.delta` text; the final/full chunk has `.message`.

_LLAMAINDEX_ROLE_MAP = {
    "user":      "user",
    "human":     "user",
    "assistant": "assistant",
    "ai":        "assistant",
    "model":     "assistant",
    "chatbot":   "assistant",
    "system":    "system",
    "developer": "system",
    "tool":      "tool_result",
    "function":  "tool_result",
}


def _llamaindex_role(msg: Any) -> str:
    """Resolve a LlamaIndex ChatMessage role (handles MessageRole enum + str)."""
    role = getattr(msg, "role", None)
    if role is None:
        return "user"
    # MessageRole enum: prefer .value, fall back to str()
    raw = getattr(role, "value", None)
    if not raw:
        raw = str(role)
    raw = raw.lower().strip()
    return _LLAMAINDEX_ROLE_MAP.get(raw, raw or "user")


def _parse_llamaindex_message_obj(msg: Any) -> List[Dict[str, Any]]:
    """Parse a single LlamaIndex ChatMessage into composition entries."""
    out: List[Dict[str, Any]] = []
    role = _llamaindex_role(msg)
    name = None
    additional = getattr(msg, "additional_kwargs", None) or {}
    if isinstance(additional, dict):
        name = additional.get("name") or additional.get("tool_call_id")

    # Newer LlamaIndex: msg.blocks (TextBlock / ImageBlock / AudioBlock /
    # DocumentBlock / CitableBlock / ToolCallBlock). The msg.content property
    # is derived from these.
    blocks = getattr(msg, "blocks", None)
    handled_blocks = False
    tool_call_blocks_seen = False
    if isinstance(blocks, list) and blocks:
        for block in blocks:
            block_type = (getattr(block, "block_type", "") or
                          type(block).__name__).lower()
            if "text" in block_type:
                text = getattr(block, "text", "") or ""
                if text:
                    if role == "tool_result":
                        out.append(_text_entry("tool_result", text, name=name))
                    else:
                        out.append(_text_entry(role, text))
                handled_blocks = True
            elif "image" in block_type:
                out.append(_non_text_entry(role, "image"))
                handled_blocks = True
            elif "audio" in block_type:
                out.append(_non_text_entry(role, "audio"))
                handled_blocks = True
            elif "document" in block_type or "file" in block_type:
                out.append(_non_text_entry(role, "document"))
                handled_blocks = True
            elif "toolcall" in block_type or "tool_call" in block_type:
                tc_name = getattr(block, "tool_name", "") or getattr(block, "name", "")
                tc_args = getattr(block, "tool_kwargs", None) or getattr(block, "arguments", {})
                args_str = tc_args if isinstance(tc_args, str) else _canonical_json(tc_args)
                entry = _text_entry("tool_call", args_str, name=tc_name)
                entry["type"] = "tool_call"
                out.append(entry)
                handled_blocks = True
                tool_call_blocks_seen = True

    if not handled_blocks:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content:
            if role == "tool_result":
                out.append(_text_entry("tool_result", content, name=name))
            else:
                out.append(_text_entry(role, content))

    # additional_kwargs.tool_calls — OpenAI-shaped tool_calls list. Newer
    # LlamaIndex versions ALSO mirror these into msg.blocks as ToolCallBlock
    # items (handled above) for backward compat — skip if we already parsed
    # tool calls from blocks, otherwise we'd emit each call twice.
    tool_calls = additional.get("tool_calls") if isinstance(additional, dict) else None
    if tool_calls and not tool_call_blocks_seen:
        for tc in tool_calls:
            tc = _as_dict(tc)
            if not isinstance(tc, dict):
                continue
            fn = _as_dict(tc.get("function")) or {}
            fn_name = fn.get("name", "") or tc.get("name", "")
            fn_args = fn.get("arguments", "")
            if not isinstance(fn_args, str):
                fn_args = _canonical_json(fn_args)
            entry = _text_entry("tool_call", fn_args, name=fn_name)
            entry["type"] = "tool_call"
            out.append(entry)

    return out


def _coerce_llamaindex_messages(payload: Any) -> list:
    """Flatten a LlamaIndex prompt payload into a flat list of message-like objects."""
    if payload is None:
        return []
    if isinstance(payload, str):
        return [payload]
    # Single ChatMessage
    if hasattr(payload, "role") and (hasattr(payload, "content") or hasattr(payload, "blocks")):
        return [payload]
    if isinstance(payload, (list, tuple)):
        flat = []
        for item in payload:
            if isinstance(item, (list, tuple)) and not (
                len(item) == 2 and isinstance(item[0], str)
            ):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [payload]


def _parse_llamaindex_messages(payload: Any) -> List[Dict[str, Any]]:
    """Parse a LlamaIndex prompt payload (List[ChatMessage] or string) into composition entries."""
    result: List[Dict[str, Any]] = []
    for msg in _coerce_llamaindex_messages(payload):
        if isinstance(msg, str):
            if msg:
                result.append(_text_entry("user", msg))
        elif isinstance(msg, dict):
            content = msg.get("content", "")
            text = content if isinstance(content, str) else _canonical_json(content)
            result.append(_text_entry(str(msg.get("role", "user")), text))
        elif hasattr(msg, "role"):
            result.extend(_parse_llamaindex_message_obj(msg))
    return result


def _parse_llamaindex_response(response: Any) -> List[Dict[str, Any]]:
    """Parse a LlamaIndex ChatResponse (or CompletionResponse / streamed chunk)."""
    result: List[Dict[str, Any]] = []
    # Primary path: ChatResponse.message — a ChatMessage.
    msg = getattr(response, "message", None)
    if msg is not None and hasattr(msg, "role"):
        result.extend(_parse_llamaindex_message_obj(msg))
        if result:
            return result
    # Streaming chunk fallback: .delta is the incremental text.
    delta = getattr(response, "delta", None)
    if isinstance(delta, str) and delta:
        result.append(_text_entry("assistant", delta))
        return result
    # CompletionResponse fallback: .text
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        result.append(_text_entry("assistant", text))
    return result


# ─── pydantic_ai parsers ───────────────────────────────────────────────────
# pydantic_ai normalizes every provider's request/response into ModelMessage
# objects: ModelRequest(parts=[...], instructions=str|None) and
# ModelResponse(parts=[...], usage, model_name, provider_name). The parts are
# dataclasses discriminated by `part_kind`:
# system-prompt / user-prompt / tool-return / retry-prompt (request side)
# text / thinking / tool-call / builtin-tool-call / builtin-tool-return
# (response side)

def _pa_part_kind(part: Any) -> str:
    return str(getattr(part, "part_kind", "") or "")


def _pa_user_part_entries(part: Any) -> List[Dict[str, Any]]:
    """A UserPromptPart's content is either a string or a Sequence of
    UserContent objects (str / ImageUrl / AudioUrl / DocumentUrl / VideoUrl /
    BinaryContent). Map to (role=user) text or non-text entries."""
    entries: List[Dict[str, Any]] = []
    content = getattr(part, "content", None)
    if isinstance(content, str):
        if content:
            entries.append(_text_entry("user", content))
        return entries
    # Sequence — walk individual UserContent items.
    try:
        items = list(content) if content is not None else []
    except TypeError:
        return entries
    for item in items:
        if isinstance(item, str):
            if item:
                entries.append(_text_entry("user", item))
            continue
        kind = getattr(item, "kind", None)
        if kind in ("image-url", "image_url", "image"):
            entries.append(_non_text_entry("user", "image"))
        elif kind in ("audio-url", "audio_url", "audio"):
            entries.append(_non_text_entry("user", "audio"))
        elif kind in ("document-url", "document_url", "document"):
            entries.append(_non_text_entry("user", "document"))
        elif kind in ("video-url", "video_url", "video"):
            entries.append(_non_text_entry("user", "video"))
        elif kind == "binary":
            media = str(getattr(item, "media_type", "") or "")
            if media.startswith("image/"):
                entries.append(_non_text_entry("user", "image"))
            elif media.startswith("audio/"):
                entries.append(_non_text_entry("user", "audio"))
            elif media.startswith("video/"):
                entries.append(_non_text_entry("user", "video"))
            else:
                entries.append(_non_text_entry("user", "document"))
        else:
            # Unknown UserContent subtype — emit a generic non-text marker.
            entries.append(_non_text_entry("user", str(kind or "unknown")))
    return entries


def _pa_tool_return_text(part: Any) -> str:
    """Serialize a ToolReturnPart's content as text for hashing."""
    content = getattr(part, "content", None)
    if isinstance(content, str):
        return content
    try:
        return _canonical_json(content)
    except Exception:
        return str(content)


def _pa_retry_text(part: Any) -> str:
    """Serialize a RetryPromptPart's content as text."""
    content = getattr(part, "content", None)
    if isinstance(content, str):
        return content
    try:
        return _canonical_json(content)
    except Exception:
        return str(content)


def _pa_args_text(args: Any) -> str:
    if args is None:
        return ""
    if isinstance(args, str):
        return args
    try:
        return _canonical_json(args)
    except Exception:
        return str(args)


def _parse_pydantic_ai_request_message(msg: Any) -> List[Dict[str, Any]]:
    """Parse one ModelRequest message. ModelRequest also carries top-level
    `.instructions` — surfaced as a system entry by the caller."""
    entries: List[Dict[str, Any]] = []
    for part in getattr(msg, "parts", None) or []:
        pk = _pa_part_kind(part)
        if pk == "system-prompt":
            content = getattr(part, "content", "") or ""
            if content:
                entries.append(_text_entry("system", str(content)))
        elif pk == "user-prompt":
            entries.extend(_pa_user_part_entries(part))
        elif pk in ("tool-return", "builtin-tool-return"):
            text = _pa_tool_return_text(part)
            entries.append(_text_entry(
                "tool_result", text, name=getattr(part, "tool_name", "") or ""
            ))
        elif pk == "retry-prompt":
            text = _pa_retry_text(part)
            tool_name = getattr(part, "tool_name", None)
            if tool_name:
                entry = _text_entry("tool_result", text, name=tool_name)
            else:
                entry = _text_entry("user", text)
            entries.append(entry)
        else:
            # Unknown request-side part — fall through silently.
            pass
    return entries


def _parse_pydantic_ai_response_message(msg: Any) -> List[Dict[str, Any]]:
    """Parse one ModelResponse message's parts."""
    entries: List[Dict[str, Any]] = []
    for part in getattr(msg, "parts", None) or []:
        pk = _pa_part_kind(part)
        if pk == "text":
            content = getattr(part, "content", "") or ""
            if content:
                entries.append(_text_entry("assistant", str(content)))
        elif pk == "thinking":
            content = getattr(part, "content", "") or ""
            entry = _text_entry("assistant", str(content))
            entry["type"] = "reasoning"
            entries.append(entry)
        elif pk in ("tool-call", "builtin-tool-call"):
            tool_name = getattr(part, "tool_name", "") or ""
            args_text = _pa_args_text(getattr(part, "args", None))
            entry = _text_entry("tool_call", args_text, name=tool_name)
            entry["type"] = "tool_call"
            entries.append(entry)
        else:
            pass
    return entries


def _parse_pydantic_ai_messages(messages: Any) -> List[Dict[str, Any]]:
    """Parse a pydantic_ai request payload — `list[ModelMessage]` passed into
    `Model.request` / `Model.request_stream`.

    Each message is a `ModelRequest` or `ModelResponse`; `ModelRequest` also
    carries top-level `.instructions` (the agent's system prompt) — emit it as
    a system entry once when encountered."""
    entries: List[Dict[str, Any]] = []
    if not messages:
        return entries
    try:
        items = list(messages)
    except TypeError:
        return entries
    seen_instructions: Optional[str] = None
    for msg in items:
        kind = str(getattr(msg, "kind", "") or "")
        if kind == "request":
            instr = getattr(msg, "instructions", None)
            if isinstance(instr, str) and instr and instr != seen_instructions:
                entries.append(_text_entry("system", instr))
                seen_instructions = instr
            entries.extend(_parse_pydantic_ai_request_message(msg))
        elif kind == "response":
            entries.extend(_parse_pydantic_ai_response_message(msg))
        else:
            # Unknown message — try parts regardless, defaulting to response shape.
            entries.extend(_parse_pydantic_ai_response_message(msg))
    return entries


def _parse_pydantic_ai_response(response: Any) -> List[Dict[str, Any]]:
    """Parse a pydantic_ai `ModelResponse` (or a StreamedResponse.get() result,
    which is also a ModelResponse synthesized from accumulated parts)."""
    if response is None:
        return []
    # ModelResponse has the same `.parts` shape as we parse on the request side.
    return _parse_pydantic_ai_response_message(response)


def _google_system_instruction(kwargs: Dict[str, Any]) -> Any:
    """Resolve the Google GenAI system instruction.

    The legacy SDK accepted a top-level ``system_instruction`` kwarg; the new
    ``google-genai`` SDK nests it inside ``config`` (a dict or a
    ``GenerateContentConfig`` model)."""
    direct = kwargs.get("system_instruction")
    if direct:
        return direct
    config = kwargs.get("config")
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get("system_instruction")
    return getattr(config, "system_instruction", None)


# ═══════════════════════════════════════════════════════════════════
# Tier 3: Complete fallback
# ═══════════════════════════════════════════════════════════════════

def _fallback_composition(payload: Any, role_name: str = "complete_prompt") -> List[Dict[str, Any]]:
    """Tier 3 fallback: serialize entire payload into a single entry."""
    try:
        if isinstance(payload, str):
            text = payload
        else:
            text = _canonical_json(payload)
        return [_text_entry(role_name, text)]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════

def build_prompt_composition(
    provider: str,
    kwargs: Dict[str, Any],
    operation: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build prompt composition from LLM call kwargs.

    3-tier fallback:
      1. Supported provider → exact parsing
      2. Unknown provider with messages[] → try OpenAI-compatible
      3. Cannot parse → complete_prompt fallback

    When ``operation == "embedding"`` the kwargs carry an embedding input
    (str / list-of-strings / list of typed multimodal segments) rather than
    chat messages — route through the dedicated embedding parser.
    """
    try:
        provider_lower = (provider or "").lower().strip()

        # Embedding operation — input shape is fundamentally different from
        # chat (no role/turn structure). Route through the embedding parser
        # which emits role="input" entries.
        if operation == "embedding":
            return _parse_embedding_input(provider_lower, kwargs)

        # Tier 0: non-text generation (image / audio / video) — detect by
        # request shape and short-circuit before any chat parser runs.
        modality_comp = _try_modality_prompt(kwargs)
        if modality_comp is not None:
            return modality_comp

        # Tier 1: Supported providers
        # LangChain — the prompt is a payload of BaseMessage objects (generate)
        # or a string / PromptValue (stream).
        if provider_lower == "langchain":
            comp = _parse_langchain_messages(kwargs.get("messages"))
            if comp:
                return comp

        # LlamaIndex — the prompt is a payload of ChatMessage objects.
        if provider_lower == "llamaindex":
            comp = _parse_llamaindex_messages(kwargs.get("messages"))
            if comp:
                return comp

        # pydantic_ai — the prompt is a list[ModelMessage] (ModelRequest /
        # ModelResponse with .parts and .instructions). Capture from the
        # unified shape rather than the inner provider's request shape.
        if provider_lower == "pydantic_ai":
            comp = _parse_pydantic_ai_messages(kwargs.get("messages"))
            if comp:
                return comp

        # OpenAI Responses API — structurally distinct from Chat Completions.
        # `input` is either a string or a list of items (role+content or
        # function_call/function_call_output/reasoning); `instructions` is the
        # system prompt.
        if provider_lower == "openai_responses":
            comp = _parse_openai_responses_input(
                kwargs.get("input"), kwargs.get("instructions")
            )
            if comp:
                return comp

        # xai-sdk uses protobuf chat_pb2.Message objects — the enforcer hands
        # them to us under "messages" via `_build_xai_pseudo_kwargs(args)`.
        if provider_lower == "xai":
            messages = kwargs.get("messages")
            if messages:
                return _parse_xai_messages(messages)

        # Cohere v2, HuggingFace, and Mistral use the OpenAI-compatible messages[] format.
        if provider_lower in ("openai", "grok", "openrouter", "litellm", "cerebras", "together", "cohere", "huggingface", "mistral", ""):
            messages = kwargs.get("messages")
            if messages and isinstance(messages, list):
                return _parse_openai_messages(messages)

        if provider_lower == "anthropic":
            messages = kwargs.get("messages")
            system = kwargs.get("system")
            if messages and isinstance(messages, list):
                return _parse_anthropic_messages(messages, system)

        if provider_lower in ("google", "google_genai", "gemini"):
            contents = kwargs.get("contents")
            system_instruction = _google_system_instruction(kwargs)
            if contents is not None:
                return _parse_google_contents(contents, system_instruction)

        if provider_lower == "bedrock":
            messages = kwargs.get("messages")
            if messages and isinstance(messages, list):
                return _parse_bedrock_converse(messages, kwargs.get("system"))

        # Tier 2: Try OpenAI-compatible format as fallback
        messages = kwargs.get("messages")
        if messages and isinstance(messages, list):
            return _parse_openai_messages(messages)

        # Try Google-style contents
        contents = kwargs.get("contents")
        if contents is not None:
            return _parse_google_contents(contents, _google_system_instruction(kwargs))

        # Tier 3: Complete fallback
        # Serialize the entire kwargs (minus non-serializable keys)
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in ("stream", "timeout")}
        return _fallback_composition(safe_kwargs, "complete_prompt")

    except Exception as e:
        logger.debug(f"TokenPolice: prompt composition failed (fallback): {e}")
        try:
            return _fallback_composition(kwargs, "complete_prompt")
        except Exception:
            return []


# Map of usage.shape → modality entry type. When the caller has already
# resolved an authoritative usage shape (e.g. via the intercept table in
# enforcer.py), it short-circuits the heuristic parsers below — no `.text`
# read on a binary-audio body can sneak through.
_SHAPE_TO_MEDIA: Dict[str, str] = {
    "openai_audio_tts":       "audio",
    "huggingface_audio_tts":  "audio",
    # mistralai>=2 `client.audio.speech.complete` → SpeechResponse.audio_data,
    # a base64 audio blob. Without this entry it falls to the Tier-3
    # hash+length fallback and is recorded as if it were response text.
    "mistral_audio_tts":      "audio",
    "google_tts":             "audio",
    "openai_images":          "image",
    "google_imagen":          "image",
    "together_image":         "image",
    "huggingface_image":      "image",
    "xai_image":              "image",
    "google_veo":             "video",
    "bedrock_image":          "image",
}

# Embedding shapes — response is a vector, never displayable. Authoritative
# short-circuit in build_response_composition so the parsers never inspect
# the response body looking for `.text` / `.choices`.
_EMBEDDING_SHAPES = frozenset({
    "openai_embeddings",
    "google_genai_embeddings",
    "cohere_embed",
    "mistral_embed",
    "voyage_embed",
    "bedrock_titan_embed",
    "huggingface_embed",
    "together_embed",
})

# OCR shapes — response is a page-structured document (e.g. Mistral OCR). It
# carries none of `.text`/`.choices`/`.content`, so every heuristic parser
# misses and it falls to the Tier-3 hash+length fallback. Authoritative
# short-circuit in build_response_composition emits privacy-preserving
# per-page markers instead.
_OCR_SHAPES = frozenset({"mistral_ocr"})


def build_response_composition(
    provider: str,
    response: Any,
    usage_shape: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Build response composition from LLM response object.

    3-tier fallback:
      1. Supported provider → exact parsing
      2. Unknown provider → try common patterns
      3. Cannot parse → complete_response fallback

    ``usage_shape`` is the authoritative shape enum shared with the
    TokenPolice service. When the caller already knows the modality (e.g. the
    enforcer resolved ``openai_audio_tts`` from the intercept table) we trust
    it as a hard override — no parser is allowed to mis-read the response
    body as text. This is defense-in-depth against TTS misclassification,
    where ``HttpxBinaryResponseContent.text`` would otherwise decode the
    audio bytes into a ~100KB "assistant text" entry.
    """
    try:
        # Authoritative override from the caller. Skips every heuristic.
        if usage_shape:
            # Embeddings — return an empty composition (output is a vector,
            # not displayable content). Skips every parser.
            if usage_shape in _EMBEDDING_SHAPES:
                return []
            media = _SHAPE_TO_MEDIA.get(usage_shape)
            if media is not None:
                return [_non_text_entry("assistant", media)]
            # OCR — page-structured document. Emit one non-text marker per
            # billed page; never inspect the (text-bearing) page bodies.
            if usage_shape in _OCR_SHAPES:
                return _ocr_composition(response)
        provider_lower = (provider or "").lower().strip()

        # Defense-in-depth: even without a usage_shape hint, a bytes-iterator
        # or raw-bytes body should NEVER be processed by a text parser.
        try:
            if _looks_like_binary_audio(response):
                return [_non_text_entry("assistant", "audio")]
        except Exception:
            pass

        # Tier 0: non-text response (image / audio / video / transcription).
        modality_comp = _try_modality_response(response)
        if modality_comp is not None:
            return modality_comp

        # LangChain — response is an LLMResult whose generations carry AIMessages.
        if provider_lower == "langchain":
            result = _parse_langchain_response(response)
            if result:
                return result

        # LlamaIndex — response is a ChatResponse with `.message` (a ChatMessage).
        if provider_lower == "llamaindex":
            result = _parse_llamaindex_response(response)
            if result:
                return result

        # pydantic_ai — response is a ModelResponse with `.parts`.
        if provider_lower == "pydantic_ai":
            result = _parse_pydantic_ai_response(response)
            if result:
                return result

        # Anthropic — response is a Message with role=assistant + content list
        # of text / tool_use blocks. No `.choices`, so it would otherwise fall
        # through to the Tier-3 fallback. Wrap as a single-message list and
        # reuse `_parse_anthropic_messages` so tool_use blocks become
        # tool_call entries (rather than being silently dropped).
        if provider_lower == "anthropic":
            try:
                content = getattr(response, "content", None)
                if content is None and isinstance(response, dict):
                    content = response.get("content")
                if content is not None:
                    result = _parse_anthropic_messages([
                        {"role": "assistant", "content": content},
                    ])
                    if result:
                        return result
            except Exception:
                pass

        # xai-sdk Response object has `.content` (str) + `.tool_calls`. A dict
        # response is the streaming-fallback shape (OpenAI-compat) — fall
        # through to the existing dict/choices branch in that case.
        if provider_lower == "xai" and not isinstance(response, dict):
            result = _parse_xai_response(response)
            if result:
                return result

        # OpenAI Responses API — response.output is a list of items
        # (message/function_call/reasoning). Checked before Bedrock Converse
        # below because Bedrock's response.output is a DICT, while Responses'
        # output is a LIST — but be explicit anyway.
        if provider_lower == "openai_responses":
            result = _parse_openai_responses_response(response)
            if result:
                return result
        else:
            # Provider not specified but response carries a Responses-shaped
            # output list — match it before the generic dict-output check.
            output_list = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
            if isinstance(output_list, list) and output_list:
                first = output_list[0]
                first = first if isinstance(first, dict) else (
                    first.model_dump() if hasattr(first, "model_dump") else (
                        first.dict() if hasattr(first, "dict") else None
                    )
                )
                if isinstance(first, dict) and first.get("type") in (
                    "message", "function_call", "reasoning"
                ):
                    result = _parse_openai_responses_response(response)
                    if result:
                        return result

        # AWS Bedrock Converse — response is a dict shaped {output: {message: ...}}
        if isinstance(response, dict) and isinstance(response.get("output"), dict):
            result = _parse_bedrock_converse_response(response)
            if result:
                return result

        # Cohere v2 — response carries an assistant `message` object
        # (no `choices`/`candidates`; content lives under `.message`).
        if hasattr(response, "message") and not hasattr(response, "choices"):
            result = _parse_cohere_response(response)
            if result:
                return result

        # OpenAI-style response. Handles SDK response objects (getattr-based)
        # AND plain dicts — the latter is how the HuggingFace manual-telemetry
        # stream wrapper hands back a synthetic accumulated response, since a
        # raw ChatCompletionStreamOutput chunk only carries `.choices[].delta`,
        # not `.choices[].message`.
        choices = None
        if isinstance(response, dict) and isinstance(response.get("choices"), list):
            choices = response["choices"]
        elif hasattr(response, "choices"):
            choices = response.choices
        if choices is not None:
            result = []
            for choice in choices:
                if isinstance(choice, dict):
                    msg = choice.get("message")
                else:
                    msg = getattr(choice, "message", None)
                if msg:
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        tool_calls = msg.get("tool_calls")
                    else:
                        content = getattr(msg, "content", None)
                        tool_calls = getattr(msg, "tool_calls", None)
                    if content:
                        result.append(_text_entry("assistant", content))
                    if tool_calls:
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                fn = tc.get("function")
                            else:
                                fn = getattr(tc, "function", None)
                            if fn:
                                if isinstance(fn, dict):
                                    fn_name = fn.get("name", "")
                                    fn_args = fn.get("arguments", "")
                                else:
                                    fn_name = getattr(fn, "name", "")
                                    fn_args = getattr(fn, "arguments", "")
                                entry = _text_entry("tool_call", fn_args, name=fn_name)
                                entry["type"] = "tool_call"
                                result.append(entry)
            if result:
                return result

        # Anthropic-style response
        if hasattr(response, "content") and isinstance(response.content, list):
            result = []
            for block in response.content:
                block_type = getattr(block, "type", "text")
                if block_type == "text":
                    result.append(_text_entry("assistant", getattr(block, "text", "")))
                elif block_type == "tool_use":
                    inp = _canonical_json(getattr(block, "input", {}))
                    entry = _text_entry("tool_call", inp, name=getattr(block, "name", ""))
                    entry["type"] = "tool_call"
                    result.append(entry)
            if result:
                return result

        # Google GenAI-style response
        if hasattr(response, "candidates"):
            result = []
            for candidate in response.candidates:
                content = getattr(candidate, "content", None)
                if content:
                    parts = getattr(content, "parts", [])
                    for part in parts:
                        # Per-part isolation: a single hostile part must not
                        # abort its siblings — previously one functionCall part
                        # whose args wasn't dict-coercible (e.g. recursively
                        # SimpleNamespace-decoded REST bodies) threw and dropped
                        # the WHOLE response to the Tier-3 complete_response blob.
                        try:
                            # google-genai Pydantic uses snake_case (`function_call`);
                            # native REST responses (decoded to SimpleNamespace) carry
                            # camelCase (`functionCall`). Accept both so tool-call
                            # responses aren't dropped to the string fallback.
                            # Gemini TTS / native-audio returns audio as
                            # inline_data / inlineData (mime audio/*) — without
                            # this branch those parts are dropped and the whole
                            # response collapses to Tier-3 complete_response.
                            fc = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
                            inline = getattr(part, "inline_data", None) or getattr(part, "inlineData", None)
                            if hasattr(part, "text") and part.text:
                                result.append(_text_entry("assistant", part.text))
                            elif fc:
                                # _serialize_tool_args handles dict / namespace /
                                # str / proto-map args without raising.
                                args = _serialize_tool_args(getattr(fc, "args", {}))
                                entry = _text_entry("tool_call", args, name=getattr(fc, "name", ""))
                                entry["type"] = "tool_call"
                                result.append(entry)
                            elif inline is not None:
                                inline_d = _as_dict(inline) if not isinstance(inline, dict) else inline
                                if not isinstance(inline_d, dict):
                                    inline_d = {}
                                    try:
                                        mime = (
                                            getattr(inline, "mime_type", None)
                                            or getattr(inline, "mimeType", None)
                                            or "unknown"
                                        )
                                    except Exception:
                                        mime = "unknown"
                                else:
                                    mime = (
                                        inline_d.get("mime_type")
                                        or inline_d.get("mimeType")
                                        or "unknown"
                                    )
                                mime_s = str(mime or "unknown")
                                media = (
                                    "image" if "image" in mime_s
                                    else "audio" if "audio" in mime_s
                                    else mime_s
                                )
                                result.append(_non_text_entry("assistant", media))
                        except Exception:
                            continue
            if result:
                return result

        # Google GenAI-style response as a PLAIN DICT — manual REST callers
        # (tp.protect / tp.log over raw httpx) pass the decoded JSON body
        # {"candidates":[{"content":{"parts":[...]}}]}. `hasattr(...)` above
        # never matches a dict key, so without this branch tool-call turns
        # collapse into the Tier-3 complete_response blob. Reuse
        # _parse_google_contents (already dict-aware, camelCase + snake_case)
        # on the candidates' content list, then normalize the role to
        # "assistant" (the response role is "model" in Google's format).
        # Gated: only a dict carrying a "candidates" list enters here.
        if isinstance(response, dict) and isinstance(response.get("candidates"), list):
            contents = []
            for candidate in response["candidates"]:
                candidate = _as_dict(candidate)
                if isinstance(candidate, dict) and candidate.get("content") is not None:
                    contents.append(candidate["content"])
            result = []
            for entry in _parse_google_contents(contents):
                if entry.get("role") not in ("tool_call", "tool_result"):
                    entry = {**entry, "role": "assistant"}
                result.append(entry)
            if result:
                return result

        # Tier 3: Complete fallback
        response_str = str(response)
        return _fallback_composition(response_str, "complete_response")

    except Exception as e:
        logger.debug(f"TokenPolice: response composition failed (fallback): {e}")
        try:
            return _fallback_composition(str(response), "complete_response")
        except Exception:
            return []


def _tool_call_id_name(tc: Any) -> tuple:
    """Best-effort ``(id, name)`` from dict / Pydantic / SimpleNamespace / protobuf.

    Dict path uses ``_as_dict`` then ``function.name`` (OpenAI/xAI shape) with a
    top-level ``name`` fallback (LangChain AIMessage shape). Non-dict elements
    (e.g. xai-sdk protobuf ToolCall) fall through to ``getattr`` — ``_as_dict``
    does not convert protos, so the previous isinstance(dict) gate dropped them
    and native-xAI tool rows lost ``tool_call_id``. Never invents ids.
    """
    d = _as_dict(tc)
    if isinstance(d, dict):
        fn = _as_dict(d.get("function")) or {}
        name = fn.get("name", "") if isinstance(fn, dict) else ""
        if not name:
            name = d.get("name") or ""
        return str(d.get("id") or d.get("call_id") or ""), str(name or "")
    fn = getattr(tc, "function", None)
    name = (getattr(fn, "name", "") if fn is not None else "") or ""
    if not name:
        name = getattr(tc, "name", "") or ""
    cid = getattr(tc, "id", None) or getattr(tc, "call_id", None) or ""
    return str(cid or ""), str(name or "")


def extract_pending_tool_calls(provider: str, response: Any) -> List[Dict[str, str]]:
    """Return an ORDERED list of ``{"id":..., "name":...}`` for tool calls in the
    LLM ``response`` that carry a non-empty provider tool-call id.

    Used to auto-correlate the model's requested tool-call ids to the subsequent
    ``@tp.tool`` / ``tool_span()`` executions (consumed in context.py). Provider-
    agnostic and best-effort: returns ``[]`` on anything it can't parse and never
    raises — the caller runs on the customer's hot path. Holds id + name only,
    never args/results (privacy). Gemini/Google ``FunctionCall.id`` is optional —
    when the provider populates it we stash it; when omitted we contribute nothing
    (never invent ids).
    """
    try:
        out: List[Dict[str, str]] = []

        def _add(cid: Any, nm: Any) -> None:
            cid = str(cid or "")
            if cid:
                out.append({"id": cid, "name": str(nm or "")})

        # OpenAI Responses API — output[] `function_call` items carry `call_id`.
        output = response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
        if isinstance(output, list):
            for item in output:
                item = _as_dict(item)
                if isinstance(item, dict) and item.get("type") == "function_call":
                    _add(item.get("call_id"), item.get("name"))
            if out:
                return out

        # Anthropic — top-level content[] of text / tool_use blocks (id + name).
        content = getattr(response, "content", None)
        if content is None and isinstance(response, dict):
            content = response.get("content")
        if isinstance(content, list):
            for block in content:
                b = _as_dict(block)
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    _add(b.get("id"), b.get("name"))
            if out:
                return out

        # OpenAI chat-completions shape — choices[].message.tool_calls[]
        # (openai, mistral, together, cerebras, openrouter, huggingface, groq…).
        choices = None
        if isinstance(response, dict) and isinstance(response.get("choices"), list):
            choices = response["choices"]
        elif hasattr(response, "choices"):
            choices = response.choices
        if isinstance(choices, list):
            for choice in choices:
                msg = choice.get("message") if isinstance(choice, dict) else getattr(choice, "message", None)
                if not msg:
                    continue
                tool_calls = msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
                for tc in tool_calls or []:
                    cid, nm = _tool_call_id_name(tc)
                    _add(cid, nm)
            if out:
                return out

        # Cohere v2 — assistant `message.tool_calls[]` ({id, function:{name}}).
        message = getattr(response, "message", None)
        if message is None and isinstance(response, dict):
            message = response.get("message")
        if message is not None:
            m = _as_dict(message)
            if isinstance(m, dict):
                for tc in m.get("tool_calls") or []:
                    cid, nm = _tool_call_id_name(tc)
                    _add(cid, nm)
            if out:
                return out

        # LlamaIndex — the ChatResponse's assistant message carries tool calls as
        # ToolCallBlock entries in `message.blocks` (id in `tool_call_id`) and/or
        # mirrored into `message.additional_kwargs["tool_calls"]` (OpenAI shape).
        # _parse_llamaindex_message reads those same two sources to build the
        # composition but deliberately drops the id; this is where the id is kept.
        # Blocks first, returning on whichever source yields an id, so a version
        # that mirrors the same calls into both can't double-count them.
        if message is not None:
            blocks = getattr(message, "blocks", None)
            if blocks is None and isinstance(m, dict):
                blocks = m.get("blocks")
            if isinstance(blocks, list):
                for block in blocks:
                    try:
                        # Classify BEFORE serializing (same idiom as the parser at
                        # :1573): a message carries image/document/audio blocks too,
                        # and model_dump()-ing every one of them on the hot path is
                        # both wasteful and a way to lose the whole turn's ids to a
                        # single unserializable block.
                        if isinstance(block, dict):
                            btype = str(block.get("block_type") or type(block).__name__).lower()
                        else:
                            btype = str(getattr(block, "block_type", "") or type(block).__name__).lower()
                        if "toolcall" not in btype and "tool_call" not in btype:
                            continue
                        b = block if isinstance(block, dict) else _as_dict(block)
                        if isinstance(b, dict):
                            _add(b.get("tool_call_id") or b.get("id"),
                                 b.get("tool_name") or b.get("name"))
                        else:
                            _add(getattr(block, "tool_call_id", None) or getattr(block, "id", None),
                                 getattr(block, "tool_name", None) or getattr(block, "name", None))
                    except Exception:
                        # One malformed block must not cost the turn its other ids.
                        continue
                if out:
                    return out
            additional = getattr(message, "additional_kwargs", None)
            if additional is None and isinstance(m, dict):
                additional = m.get("additional_kwargs")
            if isinstance(additional, dict):
                for tc in additional.get("tool_calls") or []:
                    cid, nm = _tool_call_id_name(tc)
                    _add(cid, nm)
                if out:
                    return out

        # xai-sdk / LangChain AIMessage — top-level `.tool_calls[]`.
        # Elements may be protobuf messages; _tool_call_id_name falls back to attrs.
        # LangChain shape is `{id, name, args}` (name top-level, not under .function).
        tcs = getattr(response, "tool_calls", None)
        if tcs is None and isinstance(response, dict):
            tcs = response.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                cid, nm = _tool_call_id_name(tc)
                _add(cid, nm)
            if out:
                return out

        # LangChain LLMResult — generate/agenerate returns generations[][].message
        # (AIMessage with .tool_calls). Stream path passes a bare AIMessage and is
        # covered by the top-level branch above.
        generations = getattr(response, "generations", None)
        if generations is None and isinstance(response, dict):
            generations = response.get("generations")
        if generations:
            for gen_list in generations:
                if not isinstance(gen_list, (list, tuple)):
                    gen_list = [gen_list]
                for gen in gen_list:
                    msg = (
                        gen.get("message") if isinstance(gen, dict)
                        else getattr(gen, "message", None)
                    )
                    if msg is None:
                        continue
                    tool_calls = (
                        msg.get("tool_calls") if isinstance(msg, dict)
                        else getattr(msg, "tool_calls", None)
                    )
                    for tc in tool_calls or []:
                        cid, nm = _tool_call_id_name(tc)
                        _add(cid, nm)
            if out:
                return out

        # Gemini / Google GenAI — candidates[].content.parts[].function_call
        # (snake_case Pydantic) or functionCall (camelCase REST). FunctionCall.id
        # is optional; only stash when the provider populated it.
        candidates = getattr(response, "candidates", None)
        if candidates is None and isinstance(response, dict):
            candidates = response.get("candidates")
        if isinstance(candidates, list):
            for cand in candidates:
                cand = _as_dict(cand) if not isinstance(cand, dict) else cand
                content = None
                if isinstance(cand, dict):
                    content = cand.get("content")
                else:
                    content = getattr(cand, "content", None)
                content = _as_dict(content) if content is not None and not isinstance(content, dict) else content
                parts = None
                if isinstance(content, dict):
                    parts = content.get("parts")
                elif content is not None:
                    parts = getattr(content, "parts", None)
                if not isinstance(parts, list):
                    continue
                for part in parts:
                    part_d = _as_dict(part) if not isinstance(part, dict) else part
                    fc = None
                    if isinstance(part_d, dict):
                        fc = part_d.get("function_call") or part_d.get("functionCall")
                    if fc is None and not isinstance(part, dict):
                        fc = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
                    if fc is None:
                        continue
                    fc_d = _as_dict(fc) if not isinstance(fc, dict) else fc
                    if isinstance(fc_d, dict):
                        _add(fc_d.get("id"), fc_d.get("name"))
                    else:
                        _add(getattr(fc, "id", None), getattr(fc, "name", None))
            if out:
                return out

        return out
    except Exception:
        return []
