"""
Active Enforcement Engine.
Applies extremely lightweight pre-flight monkey-patches to target SDKs.
These patches only run tp.check() and block the call if the budget is 0.
They DO NOT parse the response or handle telemetry.
"""
import sys
import asyncio
import uuid
import importlib
import functools
import re
import threading
import types as _types  # SimpleNamespace for synthetic provider responses
import inspect as _inspect  # coroutine detection when discarding a rebuilt stream manager
from urllib.parse import urlsplit as _urlsplit
from datetime import datetime, timezone
from ._safe import fail_safe
from ._detect import is_sync_iterator, is_async_iterator
from .state import get_client
from . import state as _state
from . import local_evaluator as _local_evaluator
from ._classify import build_call_outcome
from .reroute_noop import is_noop_reroute as _is_noop_reroute
from .reroute_noop import prefer_requested_model as _prefer_requested_model
import time as _time
from .exceptions import TokenPoliceBlockedError
from .context import (
    get_current_session,
    consume_pending_span_name,
    manual_span_ids,
    _session_parent_span_id,
    random_hex16,
    in_langchain,
    _in_langchain,
    enter_langchain_trace,
    leave_langchain_trace,
    in_litellm,
    _in_litellm,
    in_llamaindex,
    _in_llamaindex,
    in_pydantic_ai,
    _in_pydantic_ai,
    in_agno,
    _in_agno,
    set_pending_tool_calls,
    reserve_span_order,
    reset_span_order,
    arm_anthropic_stream_span_window,
    disarm_anthropic_stream_span_window,
)
from .composition import (
    build_prompt_composition,
    build_response_composition,
    extract_pending_tool_calls,
    _as_dict,
)
from .openai_agents import maybe_register_openai_agents_tracing, _hash_len
import logging

logger = logging.getLogger("token_police")

_is_instrumented = False

# Dictionary of modules we want to protect with pre-flight checks.
# We only need to protect the outermost execution methods.
_TARGET_METHODS = [
    # OpenAI
    { "module": "openai.resources.chat.completions", "object": "Completions", "method": "create", "async": False },
    { "module": "openai.resources.chat.completions", "object": "AsyncCompletions", "method": "create", "async": True },
    { "module": "openai.resources.completions", "object": "Completions", "method": "create", "async": False },
    { "module": "openai.resources.completions", "object": "AsyncCompletions", "method": "create", "async": True },
    # OpenAI Responses API — used by the openai-agents SDK (Runner.run* invokes
    # responses.create internally) and by direct Responses callers. We run on
    # the manual telemetry path even though opentelemetry-instrumentation-openai
    # ships a Responses wrapper: that wrapper crashes for openai-agents'
    # streaming path because the SDK calls `.with_raw_response.create()` and
    # the OpenLLMetry wrapper assumes a parsed Response, accessing `.id` on the
    # raw AsyncAPIResponse object. telemetry._auto_instrument() unwraps
    # OpenLLMetry's Responses hooks immediately after install so our manual
    # wrapper is the only thing patched and there's no double-logging.
    { "module": "openai.resources.responses", "object": "Responses", "method": "create", "async": False, "manual": True },
    { "module": "openai.resources.responses", "object": "AsyncResponses", "method": "create", "async": True, "manual": True },
    # Anthropic
    { "module": "anthropic.resources.messages", "object": "Messages", "method": "create", "async": False },
    { "module": "anthropic.resources.messages", "object": "AsyncMessages", "method": "create", "async": True },
    # Anthropic BETA surface. `anthropic.resources.beta.messages.messages.
    # Messages` is a direct `SyncAPIResource` child — a SIBLING of the
    # non-beta Messages, NOT a subclass — so the rows above give
    # `client.beta.messages.*` ZERO pre-flight enforcement (no block, no
    # reroute, no budget gate). Same OTel-covered (non-manual) kind as the
    # non-beta rows. `count_tokens` / `parse` / `batches` are deliberately
    # NOT listed: count_tokens is a free metadata endpoint — a /check there
    # could block a token-counting call for zero revenue protection.
    { "module": "anthropic.resources.beta.messages.messages", "object": "Messages", "method": "create", "async": False },
    { "module": "anthropic.resources.beta.messages.messages", "object": "AsyncMessages", "method": "create", "async": True },
    # Bedrock beta twins. `anthropic.lib.bedrock._beta_messages` ALIASES the
    # class attribute (`create = FirstPartyMessagesAPI.create`) bound at
    # import; `import anthropic` imports it EAGERLY (0.105.2), so the alias
    # captures the pre-TP function and there is no double-wrap today —
    # _wrap_method's `_tp_preflight_wrapper` identity guard keeps a future
    # lazy-import refactor (alias capturing our wrapper) from double-billing.
    { "module": "anthropic.lib.bedrock._beta_messages", "object": "Messages", "method": "create", "async": False },
    { "module": "anthropic.lib.bedrock._beta_messages", "object": "AsyncMessages", "method": "create", "async": True },
    # Google GenAI
    { "module": "google.genai.models", "object": "Models", "method": "generate_content", "async": False },
    { "module": "google.genai.models", "object": "Models", "method": "generate_content_stream", "async": False },
    { "module": "google.genai.models", "object": "AsyncModels", "method": "generate_content", "async": True },
    { "module": "google.genai.models", "object": "AsyncModels", "method": "generate_content_stream", "async": True },
    # Cohere v2 (cohere.ClientV2 / AsyncClientV2). Runs on the MANUAL telemetry
    # path: OpenLLMetry's opentelemetry-instrumentation-cohere only supports
    # `cohere <6` (no release covers cohere 6.x / 7.x), so we extract usage
    # ourselves from the Cohere v2 response / stream — version-resilient across
    # cohere 5/6/7 (see _extract_cohere_chat_usage). usage.tokens.{input,output}_
    # tokens (non-stream) and the `message-end` event's .delta.usage (stream).
    # chat_stream is async=False on BOTH clients: the call returns an (async)
    # iterator synchronously, it is not a coroutine.
    { "module": "cohere.client_v2", "object": "ClientV2", "method": "chat", "async": False, "manual": True },
    { "module": "cohere.client_v2", "object": "AsyncClientV2", "method": "chat", "async": True, "manual": True },
    { "module": "cohere.client_v2", "object": "ClientV2", "method": "chat_stream", "async": False, "manual": True },
    { "module": "cohere.client_v2", "object": "AsyncClientV2", "method": "chat_stream", "async": False, "manual": True, "async_iter": True },
    # ── Mistral AI (mistralai 1.x AND 2.x) ──
    # Runs on the manual-telemetry path. We deliberately never register
    # `opentelemetry-instrumentation-mistralai`: it imports `mistralai.models`,
    # a module that does NOT exist on mistralai>=2 (the import raises at
    # instrumentation time), and the mistralai SDK's own native TracingHook
    # (mistralai/client/_hooks/tracing.py on 2.x; mistralai/_hooks/tracing.py
    # on 1.x) fires its httpx `after_success` hook BEFORE the stream is
    # consumed — so for `stream` / `stream_async` its span carries no usage at
    # all. Extracting usage ourselves from the response / accumulated stream
    # chunks is both correct and symmetric with the Node SDK. Streaming chunks
    # are `CompletionEvent { data: CompletionChunk }` on both majors — the
    # `.data` unwrap is handled in the manual-mode helpers when provider ==
    # "mistral". TokenPoliceSpanProcessor drops the Mistral SDK's native spans
    # so they never double-log; that guard is keyed on the instrumentation
    # SCOPE NAME "mistralai_sdk_tracer" (stable across both majors), NOT on a
    # `mistral_ai.operation.id` span attribute — 2.x no longer emits one.
    #
    # DUAL MODULE PATHS (mistralai 2.0 breaking change): 2.x turned the
    # distribution into a namespace package and moved every resource module
    # under `mistralai.client.*` (`from mistralai.client import Mistral`).
    # `_wrap_method` returns silently on ImportError, so a 1.x-only row on a
    # 2.x install wraps NOTHING — no /check, no /log, no error: un-enforced
    # and un-metered spend. Both paths are therefore registered; the one that
    # is not installed is skipped. Keep BOTH: 1.x is still widely deployed.
    # Regression pin: tests/test_real_provider_seams_mistral.py resolves every
    # row below against the REAL installed package on either major.
    { "module": "mistralai.chat", "object": "Chat", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai.chat", "object": "Chat", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai.chat", "object": "Chat", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai.chat", "object": "Chat", "method": "stream_async",   "async": True,  "manual": True },
    { "module": "mistralai.client.chat", "object": "Chat", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai.client.chat", "object": "Chat", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai.client.chat", "object": "Chat", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai.client.chat", "object": "Chat", "method": "stream_async",   "async": True,  "manual": True },
    # `Chat.parse` / `parse_async` / `parse_stream` / `parse_stream_async`
    # (structured outputs) are DELIBERATELY absent: on both majors they call
    # `self.complete` / `self.complete_async` / `self.stream` /
    # `self.stream_async` internally, so the rows above already cover them.
    # Registering them would run the pre-flight check TWICE and emit TWO /log
    # rows for one billable call. `ParsedChatCompletionResponse` subclasses
    # `ChatCompletionResponse`, so the complete-wrapper's extractor handles the
    # parsed response unchanged.
    #
    # Mistral FIM (Codestral fill-in-the-middle, `client.fim.complete(...)`) —
    # billable chat-like spend on a DIFFERENT endpoint (/v1/fim/completions)
    # and a DIFFERENT class, so the Chat rows never covered it. The request
    # carries `prompt`/`suffix` instead of `messages` (composition capture
    # degrades to the fallback and never raises); `FIMCompletionResponse`
    # carries an OpenAI-shaped `.usage` + `.choices[].message`, so the chat
    # extractors and the stream accumulator handle it unchanged.
    { "module": "mistralai.fim", "object": "Fim", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai.fim", "object": "Fim", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai.fim", "object": "Fim", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai.fim", "object": "Fim", "method": "stream_async",   "async": True,  "manual": True },
    { "module": "mistralai.client.fim", "object": "Fim", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai.client.fim", "object": "Fim", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai.client.fim", "object": "Fim", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai.client.fim", "object": "Fim", "method": "stream_async",   "async": True,  "manual": True },
    # Mistral on Azure AI (`MistralAzure`) and Google Vertex (`MistralGCP`).
    # These are DISTINCT classes shipped in parallel client trees, not
    # re-exports of the ones above — a wrapper on `Chat` never reaches
    # `azure...Chat`, so without these rows managed-deployment traffic was
    # un-checked and un-logged. Both majors ship both trees, at different
    # paths: top-level `mistralai_azure` / `mistralai_gcp` packages on 1.x,
    # `mistralai.azure.client.*` / `mistralai.gcp.client.*` on 2.x. Azure
    # exposes chat + OCR; GCP exposes chat + FIM. `_detect_provider` maps every
    # one of these to "mistral" (substring match), so shapes/extractors are
    # identical to the direct-API rows.
    { "module": "mistralai_azure.chat", "object": "Chat", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai_azure.chat", "object": "Chat", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai_azure.chat", "object": "Chat", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai_azure.chat", "object": "Chat", "method": "stream_async",   "async": True,  "manual": True },
    { "module": "mistralai.azure.client.chat", "object": "Chat", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai.azure.client.chat", "object": "Chat", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai.azure.client.chat", "object": "Chat", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai.azure.client.chat", "object": "Chat", "method": "stream_async",   "async": True,  "manual": True },
    { "module": "mistralai_gcp.chat", "object": "Chat", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai_gcp.chat", "object": "Chat", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai_gcp.chat", "object": "Chat", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai_gcp.chat", "object": "Chat", "method": "stream_async",   "async": True,  "manual": True },
    { "module": "mistralai.gcp.client.chat", "object": "Chat", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai.gcp.client.chat", "object": "Chat", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai.gcp.client.chat", "object": "Chat", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai.gcp.client.chat", "object": "Chat", "method": "stream_async",   "async": True,  "manual": True },
    { "module": "mistralai_gcp.fim", "object": "Fim", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai_gcp.fim", "object": "Fim", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai_gcp.fim", "object": "Fim", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai_gcp.fim", "object": "Fim", "method": "stream_async",   "async": True,  "manual": True },
    { "module": "mistralai.gcp.client.fim", "object": "Fim", "method": "complete",       "async": False, "manual": True },
    { "module": "mistralai.gcp.client.fim", "object": "Fim", "method": "complete_async", "async": True,  "manual": True },
    { "module": "mistralai.gcp.client.fim", "object": "Fim", "method": "stream",         "async": False, "manual": True },
    { "module": "mistralai.gcp.client.fim", "object": "Fim", "method": "stream_async",   "async": True,  "manual": True },
    # NOT covered (deliberate, revisit when demand appears): the Mistral
    # Agents API (`client.agents.complete/stream`) and Conversations API
    # (`client.conversations.start/append/restart`). They are billable, but
    # their request/response envelopes are not chat-shaped, so a row here
    # would /check and /log with mis-parsed composition rather than no row.
    # LiteLLM. Framework — internally calls the underlying provider SDK
    # (openai, anthropic, ...) which is ALSO patched. The wrapper runs the
    # SINGLE pre-flight check, captures composition from the OpenAI-shaped
    # messages/response, and holds the _in_litellm guard so nested provider
    # wrappers pass straight through. Telemetry is manual since
    # opentelemetry-instrumentation-litellm does not exist on PyPI; LiteLLM
    # responses are OpenAI-shaped so _extract_openai_compatible_usage handles
    # them. Streaming uses stream=True on the same method — returns a
    # CustomStreamWrapper of OpenAI-shaped chunks; final chunk carries usage
    # when stream_options={"include_usage": True} is passed.
    { "module": "litellm", "object": None, "method": "completion",  "async": False, "manual": True, "framework": "litellm" },
    { "module": "litellm", "object": None, "method": "acompletion", "async": True,  "manual": True, "framework": "litellm" },
    # AWS Bedrock. No manual flag: Bedrock Converse chat token usage is captured
    # by opentelemetry-instrumentation-bedrock (converse_usage_record), which
    # requires >=0.61.0 for the Converse usage attributes — earlier releases
    # (0.52.4–0.60.x) silently undercount (pinned in pyproject.toml). Only
    # Bedrock embedding (InvokeModel) is handled on the manual path.
    { "module": "botocore.client", "object": "BaseClient", "method": "_make_api_call", "async": False },
    { "module": "aiobotocore.client", "object": "AioBaseClient", "method": "_make_api_call", "async": True },
    # OpenRouter (native SDK). No OpenLLMetry instrumentor exists, so these
    # wrappers extract token usage manually (manual=True).
    { "module": "openrouter.chat", "object": "Chat", "method": "send", "async": False, "manual": True },
    { "module": "openrouter.chat", "object": "Chat", "method": "send_async", "async": True, "manual": True },
    # Cerebras (native cerebras_cloud_sdk). No OpenLLMetry instrumentor exists,
    # so these wrappers extract token usage manually (manual=True). The SDK is
    # OpenAI-compatible — usage is OpenAI-shaped.
    { "module": "cerebras.cloud.sdk.resources.chat.completions", "object": "CompletionsResource", "method": "create", "async": False, "manual": True },
    { "module": "cerebras.cloud.sdk.resources.chat.completions", "object": "AsyncCompletionsResource", "method": "create", "async": True, "manual": True },
    # Together.ai (native together-python SDK). opentelemetry-instrumentation-
    # together (Traceloop) exists but its current release imports
    # `together.types.completions.CompletionResponse`, a symbol the together
    # 2.x package no longer exposes — the import errors at instrumentation
    # time. Fall back to manual telemetry: Together is OpenAI-compatible so
    # _extract_openai_compatible_usage and the OpenAI-shaped stream
    # accumulator handle it without further plumbing.
    { "module": "together.resources.chat.completions", "object": "CompletionsResource", "method": "create", "async": False, "manual": True },
    { "module": "together.resources.chat.completions", "object": "AsyncCompletionsResource", "method": "create", "async": True, "manual": True },
    # Groq (native `groq` SDK). Groq ships its own Stainless-generated package
    # (`from groq import Groq`) with a private httpx client — it is NOT the
    # `openai` package, so the `openai` OpenLLMetry instrumentor never patches
    # it and a native-groq call would be un-metered AND un-enforced. Runs on the
    # manual-telemetry path (manual=True), mirroring cerebras/together. Groq is
    # OpenAI-compatible: `groq.resources.chat.completions.{Completions,
    # AsyncCompletions}.create` returns an OpenAI-shaped response whose
    # non-streaming `.usage` is top-level OpenAI-shaped. STREAMING usage is
    # nested under `chunk.x_groq.usage` on the final chunk (top-level
    # `chunk.usage` is absent/None) — unwrapped in _chunk_usage /
    # _extract_openai_compatible_usage / _extract_raw_usage.
    { "module": "groq.resources.chat.completions", "object": "Completions", "method": "create", "async": False, "manual": True },
    { "module": "groq.resources.chat.completions", "object": "AsyncCompletions", "method": "create", "async": True, "manual": True },
    # HuggingFace (huggingface_hub InferenceClient / AsyncInferenceClient).
    # No OpenLLMetry instrumentor exists, so these wrappers extract token usage
    # manually (manual=True). `chat_completion` is a regular method on both
    # classes and is OpenAI-compatible — the response carries an OpenAI-shaped
    # `.usage`. Streaming (`stream=True`) returns a (sync/async) generator of
    # ChatCompletionStreamOutput chunks; the final chunk MAY carry `.usage`.
    { "module": "huggingface_hub", "object": "InferenceClient", "method": "chat_completion", "async": False, "manual": True },
    { "module": "huggingface_hub", "object": "AsyncInferenceClient", "method": "chat_completion", "async": True, "manual": True },
    # xAI native xai-sdk. No OpenLLMetry instrumentor exists, so these wrappers
    # extract token usage manually (manual=True). The shape is unusual:
    # `client.chat.create(model=...)` returns a `Chat` instance, then the call
    # site invokes `.sample()` / `.stream()` with NO args — messages accumulate
    # on the Chat instance via `.append()`. The manual wrapper therefore reads
    # composition from `self` (args[0]) via `_build_xai_pseudo_kwargs`; the
    # Response's `.usage` is OpenAI-shaped (`prompt_tokens` / `completion_tokens`).
    # Streaming yields `(Response, Chunk)` tuples — `_chunk_usage`,
    # `_accumulate_stream_chunk`, and `_extract_openai_compatible_usage` unwrap
    # the tuple. `stream` is declared async=False on BOTH sync and async Chat:
    # the sync method returns a generator and the async method is an async
    # generator function (calling it returns the async iterator without
    # awaiting) — same pattern as Cohere chat_stream above.
    { "module": "xai_sdk.sync.chat", "object": "Chat", "method": "sample", "async": False, "manual": True },
    { "module": "xai_sdk.sync.chat", "object": "Chat", "method": "stream", "async": False, "manual": True },
    { "module": "xai_sdk.aio.chat",  "object": "Chat", "method": "sample", "async": True,  "manual": True },
    { "module": "xai_sdk.aio.chat",  "object": "Chat", "method": "stream", "async": False, "manual": True, "async_iter": True },
    # LangChain (langchain_core.language_models.chat_models.BaseChatModel).
    # LangChain is a framework — it calls the underlying provider SDK (openai,
    # anthropic, ...) internally, and that SDK is ALSO patched above. The
    # LangChain wrappers run the single pre-flight check + capture composition,
    # then hold the _in_langchain guard so the nested provider wrapper stays
    # inert (no double check). Telemetry flows through the OpenLLMetry LangChain
    # instrumentor span — see telemetry.py. ChatOpenAI / ChatAnthropic /
    # ChatGoogleGenerativeAI all inherit these methods from BaseChatModel.
    # `astream` is an async-generator function: calling it returns the async
    # iterator synchronously (not a coroutine), so async=False — same as the
    # Cohere chat_stream entries above.
    { "module": "langchain_core.language_models.chat_models", "object": "BaseChatModel", "method": "generate", "async": False, "langchain": "sync" },
    { "module": "langchain_core.language_models.chat_models", "object": "BaseChatModel", "method": "agenerate", "async": True, "langchain": "async" },
    { "module": "langchain_core.language_models.chat_models", "object": "BaseChatModel", "method": "stream", "async": False, "langchain": "stream" },
    { "module": "langchain_core.language_models.chat_models", "object": "BaseChatModel", "method": "astream", "async": False, "langchain": "astream" },
    # LlamaIndex (llama_index.llms.{openai,anthropic,google_genai}). Framework —
    # each provider class internally calls the underlying SDK (openai, anthropic,
    # google.genai), which is ALSO patched above. The LlamaIndex wrappers run the
    # single pre-flight check + capture composition from LlamaIndex's ChatMessage
    # objects, then hold the _in_llamaindex guard so the nested provider wrapper
    # stays inert (no double check). Telemetry flows through the inner provider's
    # OpenLLMetry span. stream_chat returns its iterator synchronously, so
    # async=False; astream_chat is `async def` (unlike LangChain astream) and is
    # served by the "async" kind — li_async awaits it and returns the guarded
    # _drain() async generator. There is intentionally no "astream" kind for
    # LlamaIndex (see _set_llamaindex_wrapper).
    { "module": "llama_index.llms.openai", "object": "OpenAI", "method": "chat",         "async": False, "llamaindex": "sync" },
    { "module": "llama_index.llms.openai", "object": "OpenAI", "method": "achat",        "async": True,  "llamaindex": "async" },
    { "module": "llama_index.llms.openai", "object": "OpenAI", "method": "stream_chat",  "async": False, "llamaindex": "stream" },
    { "module": "llama_index.llms.openai", "object": "OpenAI", "method": "astream_chat", "async": True, "llamaindex": "async" },
    # OpenAIResponses (B1) — a SIBLING of OpenAI, not a subclass: it derives
    # from FunctionCallingLLM and defines its own chat/achat/stream_chat/
    # astream_chat against /v1/responses. Without its own entries the guard was
    # never taken for it, so its inner `client.responses.create` ran the raw
    # manual wrapper's OWN pre-flight and applied a reroute the framework guard
    # promises to suppress. Same four kinds as OpenAI above; _wrap_method
    # resolves the class with a 3-arg getattr and returns when it is absent, so
    # an older llama-index-llms-openai without this class degrades silently.
    { "module": "llama_index.llms.openai", "object": "OpenAIResponses", "method": "chat",         "async": False, "llamaindex": "sync" },
    { "module": "llama_index.llms.openai", "object": "OpenAIResponses", "method": "achat",        "async": True,  "llamaindex": "async" },
    { "module": "llama_index.llms.openai", "object": "OpenAIResponses", "method": "stream_chat",  "async": False, "llamaindex": "stream" },
    { "module": "llama_index.llms.openai", "object": "OpenAIResponses", "method": "astream_chat", "async": True, "llamaindex": "async" },
    { "module": "llama_index.llms.anthropic", "object": "Anthropic", "method": "chat",         "async": False, "llamaindex": "sync" },
    { "module": "llama_index.llms.anthropic", "object": "Anthropic", "method": "achat",        "async": True,  "llamaindex": "async" },
    { "module": "llama_index.llms.anthropic", "object": "Anthropic", "method": "stream_chat",  "async": False, "llamaindex": "stream" },
    { "module": "llama_index.llms.anthropic", "object": "Anthropic", "method": "astream_chat", "async": True, "llamaindex": "async" },
    { "module": "llama_index.llms.google_genai", "object": "GoogleGenAI", "method": "chat",         "async": False, "llamaindex": "sync" },
    { "module": "llama_index.llms.google_genai", "object": "GoogleGenAI", "method": "achat",        "async": True,  "llamaindex": "async" },
    { "module": "llama_index.llms.google_genai", "object": "GoogleGenAI", "method": "stream_chat",  "async": False, "llamaindex": "stream" },
    { "module": "llama_index.llms.google_genai", "object": "GoogleGenAI", "method": "astream_chat", "async": True, "llamaindex": "async" },
    # ── Non-text modality endpoints (image / audio / video / OCR) ──
    # All run on the manual-telemetry path. The service dispatches on the
    # explicit `shape` enum; items/duration are produced by the registered
    # handler in _MODALITY_HANDLERS.
    # OpenAI images.generate (DALL-E 2/3, gpt-image-1/2)
    { "module": "openai.resources.images", "object": "Images",       "method": "generate", "async": False, "manual": True, "modality": "image_gen", "shape": "openai_images" },
    { "module": "openai.resources.images", "object": "AsyncImages",  "method": "generate", "async": True,  "manual": True, "modality": "image_gen", "shape": "openai_images" },
    # OpenAI audio.speech.create (tts-1, gpt-4o-mini-tts)
    { "module": "openai.resources.audio.speech", "object": "Speech",      "method": "create", "async": False, "manual": True, "modality": "audio_tts", "shape": "openai_audio_tts" },
    { "module": "openai.resources.audio.speech", "object": "AsyncSpeech", "method": "create", "async": True,  "manual": True, "modality": "audio_tts", "shape": "openai_audio_tts" },
    # OpenAI audio.transcriptions / translations (whisper-1, gpt-4o-transcribe)
    { "module": "openai.resources.audio.transcriptions", "object": "Transcriptions",      "method": "create", "async": False, "manual": True, "modality": "audio_stt", "shape": "openai_audio_stt" },
    { "module": "openai.resources.audio.transcriptions", "object": "AsyncTranscriptions", "method": "create", "async": True,  "manual": True, "modality": "audio_stt", "shape": "openai_audio_stt" },
    { "module": "openai.resources.audio.translations",   "object": "Translations",        "method": "create", "async": False, "manual": True, "modality": "audio_stt", "shape": "openai_audio_stt" },
    { "module": "openai.resources.audio.translations",   "object": "AsyncTranslations",   "method": "create", "async": True,  "manual": True, "modality": "audio_stt", "shape": "openai_audio_stt" },
    # Google Imagen (generate_images) + Veo (generate_videos)
    { "module": "google.genai.models", "object": "Models",      "method": "generate_images", "async": False, "manual": True, "modality": "image_gen", "shape": "google_imagen" },
    { "module": "google.genai.models", "object": "AsyncModels", "method": "generate_images", "async": True,  "manual": True, "modality": "image_gen", "shape": "google_imagen" },
    { "module": "google.genai.models", "object": "Models",      "method": "generate_videos", "async": False, "manual": True, "modality": "video_gen", "shape": "google_veo" },
    { "module": "google.genai.models", "object": "AsyncModels", "method": "generate_videos", "async": True,  "manual": True, "modality": "video_gen", "shape": "google_veo" },
    # xAI Grok image generation (xai-sdk: `client.image.sample(prompt, model, ...)`
    # returns one ImageResponse; `sample_batch(prompt, model, n, ...)` returns N.)
    { "module": "xai_sdk.sync.image", "object": "Client", "method": "sample",       "async": False, "manual": True, "modality": "image_gen", "shape": "xai_image" },
    { "module": "xai_sdk.sync.image", "object": "Client", "method": "sample_batch", "async": False, "manual": True, "modality": "image_gen", "shape": "xai_image" },
    { "module": "xai_sdk.aio.image",  "object": "Client", "method": "sample",       "async": True,  "manual": True, "modality": "image_gen", "shape": "xai_image" },
    { "module": "xai_sdk.aio.image",  "object": "Client", "method": "sample_batch", "async": True,  "manual": True, "modality": "image_gen", "shape": "xai_image" },
    # Mistral OCR + audio transcription (Voxtral) + audio speech (TTS).
    # Both 1.x (`mistralai.*`) and 2.x (`mistralai.client.*`) module paths are
    # registered — see the dual-path note on the Mistral chat rows above.
    # `mistralai.client.speech` (`client.audio.speech.complete`) is
    # mistralai>=2 ONLY; 1.x shipped no TTS surface, so that row simply never
    # resolves there.
    #
    # NOT covered (deliberate): `Transcriptions.stream` / `stream_async`.
    # The modality wrapper is non-streaming by construction — it logs at the
    # moment the vendor call returns (see `_set_manual_wrapper`: the
    # `if modality:` branch extracts + logs + returns immediately). A
    # transcription stream returns an un-drained `EventStream` whose usage
    # (including `prompt_audio_seconds`) only arrives on the final
    # `transcription.done` event, so a row here would log 0 audio-seconds on
    # every streamed transcription — worse than no row, because it would look
    # like measured free traffic. Covering it needs a draining proxy in the
    # modality path (the machinery the chat path has and the modality path
    # deliberately does not); left as a known gap. `Speech.complete` is safe
    # to register even though it too accepts `stream=True`: TTS is billed on
    # the REQUEST's character count, which the extractor reads from kwargs
    # and is therefore exact whether or not the response is drained.
    { "module": "mistralai.ocr",            "object": "Ocr",            "method": "process",        "async": False, "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai.ocr",            "object": "Ocr",            "method": "process_async",  "async": True,  "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai.client.ocr",     "object": "Ocr",            "method": "process",        "async": False, "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai.client.ocr",     "object": "Ocr",            "method": "process_async",  "async": True,  "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai_azure.ocr",      "object": "Ocr",            "method": "process",        "async": False, "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai_azure.ocr",      "object": "Ocr",            "method": "process_async",  "async": True,  "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai.azure.client.ocr", "object": "Ocr",          "method": "process",        "async": False, "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai.azure.client.ocr", "object": "Ocr",          "method": "process_async",  "async": True,  "manual": True, "modality": "ocr",       "shape": "mistral_ocr" },
    { "module": "mistralai.transcriptions", "object": "Transcriptions", "method": "complete",       "async": False, "manual": True, "modality": "audio_stt", "shape": "mistral_audio_stt" },
    { "module": "mistralai.transcriptions", "object": "Transcriptions", "method": "complete_async", "async": True,  "manual": True, "modality": "audio_stt", "shape": "mistral_audio_stt" },
    { "module": "mistralai.client.transcriptions", "object": "Transcriptions", "method": "complete",       "async": False, "manual": True, "modality": "audio_stt", "shape": "mistral_audio_stt" },
    { "module": "mistralai.client.transcriptions", "object": "Transcriptions", "method": "complete_async", "async": True,  "manual": True, "modality": "audio_stt", "shape": "mistral_audio_stt" },
    { "module": "mistralai.client.speech",  "object": "Speech",         "method": "complete",       "async": False, "manual": True, "modality": "audio_tts", "shape": "mistral_audio_tts" },
    { "module": "mistralai.client.speech",  "object": "Speech",         "method": "complete_async", "async": True,  "manual": True, "modality": "audio_tts", "shape": "mistral_audio_tts" },
    # Together images.generate (FLUX, SD3, Qwen-Image). Together's SDK class
    # names are *Resource, distinct from OpenAI's plain *.
    { "module": "together.resources.images", "object": "ImagesResource",      "method": "generate", "async": False, "manual": True, "modality": "image_gen", "shape": "together_image" },
    { "module": "together.resources.images", "object": "AsyncImagesResource", "method": "generate", "async": True,  "manual": True, "modality": "image_gen", "shape": "together_image" },
    # HuggingFace InferenceClient — text-to-image / text-to-speech / ASR
    { "module": "huggingface_hub", "object": "InferenceClient",      "method": "text_to_image",                "async": False, "manual": True, "modality": "image_gen", "shape": "huggingface_image" },
    { "module": "huggingface_hub", "object": "AsyncInferenceClient", "method": "text_to_image",                "async": True,  "manual": True, "modality": "image_gen", "shape": "huggingface_image" },
    { "module": "huggingface_hub", "object": "InferenceClient",      "method": "text_to_speech",               "async": False, "manual": True, "modality": "audio_tts", "shape": "huggingface_audio_tts" },
    { "module": "huggingface_hub", "object": "AsyncInferenceClient", "method": "text_to_speech",               "async": True,  "manual": True, "modality": "audio_tts", "shape": "huggingface_audio_tts" },
    { "module": "huggingface_hub", "object": "InferenceClient",      "method": "automatic_speech_recognition", "async": False, "manual": True, "modality": "audio_stt", "shape": "huggingface_audio_stt" },
    { "module": "huggingface_hub", "object": "AsyncInferenceClient", "method": "automatic_speech_recognition", "async": True,  "manual": True, "modality": "audio_stt", "shape": "huggingface_audio_stt" },
    # CrewAI: no dedicated entry. CrewAI's `crewai.llm.LLM.call` invokes
    # `litellm.completion(**params)` directly, so the existing LiteLLM wrapper
    # above already does the right thing for every CrewAI LLM call:
    # pre-flight check, OpenAI-shaped prompt composition, token extraction,
    # manual /log, streaming accumulator. The OpenLLMetry CrewAI instrumentor
    # uses its own TracerProvider so its spans never reach our SpanProcessor,
    # and adding our own CrewAI wrapper would only re-check the budget that
    # LiteLLM already checks one frame deeper. End users still get per-call
    # span_order under the @tp.workflow() they decorate, with model + usage
    # reported through the LiteLLM telemetry path.
    # ── Embeddings ──
    # All embedding wrappers run on the manual-telemetry path. No
    # OpenLLMetry instrumentor reliably emits embedding spans across
    # providers, and embedding usage is structurally distinct from chat
    # (input-only, vector output) — the dedicated _extract_embedding_usage
    # path and composition.operation="embedding" parser handle both ends.
    # Embedding traffic is exempt from the server's loop/anomaly detection,
    # so bulk RAG ingest is never throttled.
    # OpenAI
    { "module": "openai.resources.embeddings", "object": "Embeddings",      "method": "create", "async": False, "manual": True, "operation": "embedding", "shape": "openai_embeddings" },
    { "module": "openai.resources.embeddings", "object": "AsyncEmbeddings", "method": "create", "async": True,  "manual": True, "operation": "embedding", "shape": "openai_embeddings" },
    # Google GenAI (text-embedding-004, gemini-embedding-001)
    { "module": "google.genai.models", "object": "Models",      "method": "embed_content", "async": False, "manual": True, "operation": "embedding", "shape": "google_genai_embeddings" },
    { "module": "google.genai.models", "object": "AsyncModels", "method": "embed_content", "async": True,  "manual": True, "operation": "embedding", "shape": "google_genai_embeddings" },
    # Cohere v2 (embed-english-v3.0, embed-multilingual-v3.0, embed-v4.0 multimodal)
    { "module": "cohere.client_v2", "object": "ClientV2",      "method": "embed", "async": False, "manual": True, "operation": "embedding", "shape": "cohere_embed" },
    { "module": "cohere.client_v2", "object": "AsyncClientV2", "method": "embed", "async": True,  "manual": True, "operation": "embedding", "shape": "cohere_embed" },
    # Mistral (mistral-embed). The Mistral SDK's embeddings surface lives at
    # `mistralai.embeddings` (class `Embeddings`) — not `mistralai.embed` —
    # so target the real module, or the wrapper silently fails to install
    # (never resolves at import time). Customer code calls
    # `Mistral().embeddings.create(...)` which delegates here. mistralai>=2
    # moved the same class to `mistralai.client.embeddings`; both are
    # registered so either major is covered (see the Mistral chat rows).
    { "module": "mistralai.embeddings", "object": "Embeddings", "method": "create",       "async": False, "manual": True, "operation": "embedding", "shape": "mistral_embed" },
    { "module": "mistralai.embeddings", "object": "Embeddings", "method": "create_async", "async": True,  "manual": True, "operation": "embedding", "shape": "mistral_embed" },
    { "module": "mistralai.client.embeddings", "object": "Embeddings", "method": "create",       "async": False, "manual": True, "operation": "embedding", "shape": "mistral_embed" },
    { "module": "mistralai.client.embeddings", "object": "Embeddings", "method": "create_async", "async": True,  "manual": True, "operation": "embedding", "shape": "mistral_embed" },
    # Together (OpenAI-shaped usage; `together_embed` is an accepted alias shape)
    { "module": "together.resources.embeddings", "object": "EmbeddingsResource",      "method": "create", "async": False, "manual": True, "operation": "embedding", "shape": "together_embed" },
    { "module": "together.resources.embeddings", "object": "AsyncEmbeddingsResource", "method": "create", "async": True,  "manual": True, "operation": "embedding", "shape": "together_embed" },
    # HuggingFace feature_extraction (no usage object — SDK approximates token count)
    { "module": "huggingface_hub", "object": "InferenceClient",      "method": "feature_extraction", "async": False, "manual": True, "operation": "embedding", "shape": "huggingface_embed" },
    { "module": "huggingface_hub", "object": "AsyncInferenceClient", "method": "feature_extraction", "async": True,  "manual": True, "operation": "embedding", "shape": "huggingface_embed" },
    # LiteLLM (gateway — OpenAI-shaped usage; framework guard for nested provider pass-through)
    { "module": "litellm", "object": None, "method": "embedding",  "async": False, "manual": True, "framework": "litellm", "operation": "embedding", "shape": "openai_embeddings" },
    { "module": "litellm", "object": None, "method": "aembedding", "async": True,  "manual": True, "framework": "litellm", "operation": "embedding", "shape": "openai_embeddings" },
    # Voyage AI (the Anthropic-recommended embedding provider). voyageai
    # exports `Client` / `AsyncClient`; both expose `embed` and `multimodal_embed`.
    # voyageai is added as a peer dep in pyproject.toml `all` extra — the SDK
    # is small (Pydantic + httpx) and well-shaped.
    { "module": "voyageai", "object": "Client",      "method": "embed",            "async": False, "manual": True, "operation": "embedding", "shape": "voyage_embed" },
    { "module": "voyageai", "object": "AsyncClient", "method": "embed",            "async": True,  "manual": True, "operation": "embedding", "shape": "voyage_embed" },
    { "module": "voyageai", "object": "Client",      "method": "multimodal_embed", "async": False, "manual": True, "operation": "embedding", "shape": "voyage_embed" },
    { "module": "voyageai", "object": "AsyncClient", "method": "multimodal_embed", "async": True,  "manual": True, "operation": "embedding", "shape": "voyage_embed" },
]

_originals = {}

# Providers whose OpenLLMetry instrumentor does NOT emit a gen_ai.system span
# attribute. The telemetry SpanProcessor would otherwise log these with
# provider="" — so the enforcer always stashes a provider override for them.
_OTEL_PROVIDERS_WITHOUT_SYSTEM_ATTR = {"cohere"}


def _is_bedrock_runtime(obj) -> bool:
    """Safely extracts service_name from a botocore/aiobotocore client to check if it's bedrock-runtime."""
    try:
        return obj.meta.service_model.service_name == 'bedrock-runtime'
    except AttributeError:
        return False


# Bedrock model-id prefixes that identify an embedding call. The botocore
# wrapper checks against these BEFORE consuming the response stream so we
# never read .body unless we're going to manually log embedding usage.
_BEDROCK_EMBEDDING_MODEL_PREFIXES = (
    "amazon.titan-embed",  # amazon.titan-embed-text-v2:0, amazon.titan-embed-image-v1
    "cohere.embed",        # cohere.embed-english-v3, cohere.embed-multilingual-v3
    "voyage.voyage",       # voyage.voyage-3-large, voyage.voyage-multimodal-3, etc.
)


def _is_bedrock_embedding_invoke(args) -> bool:
    """True iff the botocore call is `InvokeModel` for an embedding model.

    `_make_api_call(self, operation_name, api_params)` — so args[1] is the
    operation name and args[2] is the request params dict carrying `modelId`.
    """
    try:
        if len(args) < 3:
            return False
        if args[1] != "InvokeModel":
            return False
        api_params = args[2]
        if not isinstance(api_params, dict):
            return False
        model_id = api_params.get("modelId", "")
        if not isinstance(model_id, str):
            return False
        return any(model_id.startswith(p) for p in _BEDROCK_EMBEDDING_MODEL_PREFIXES)
    except Exception:
        return False


# The four bedrock-runtime LLM operations that carry a `modelId` in their
# request params. Same allowlist `_stash_attempt_context` uses for failure
# rows — any other op (ApplyGuardrail, ListFoundationModels, ...) has no model
# to speak of and must keep reaching /check with none.
_BEDROCK_MODELED_OPS = ("Converse", "ConverseStream",
                        "InvokeModel", "InvokeModelWithResponseStream")


def _bedrock_model_hint(module_path, args):
    """Matching-only model id for the botocore/aiobotocore call shape.

    botocore invokes `_make_api_call(self, operation_name, api_params)`
    POSITIONALLY, so the wrapper's `kwargs` is EMPTY and the model lives at
    `args[2]["modelId"]`. Without this the Bedrock chat pre-flight reached
    /check with no model at all: a model-scoped rule could not match, so an
    ENFORCE BLOCK silently failed open (and the call was billed) and budget
    group-bys bucketed to unknown.

    Surfaced as a HINT, never merged into kwargs — `_make_api_call` takes no
    `model` keyword, so writing one back would raise TypeError inside the
    customer's call, and handing the check a throwaway `{"model": ...}` body
    would let _apply_reroute "apply" a swap that never reaches AWS (phantom
    _tp_routing + false REQUEST_REROUTED). The hint feeds matching + audit
    only; an enforce-mode REROUTE here still rejects as
    `unappliable_call_shape`, now with `reroute.from.model` populated. Mirrors
    Node's `modelHint` in enforcer.ts. Total: never raises.

    The `module_path` gate is load-bearing: the two chat pre-flight call sites
    are shared by every provider (openai, anthropic, google, ...), whose
    `args` mean something else entirely.
    """
    try:
        if module_path not in ("botocore.client", "aiobotocore.client"):
            return None
        if len(args) < 3:
            return None
        if args[1] not in _BEDROCK_MODELED_OPS:
            return None
        api_params = args[2]
        if not isinstance(api_params, dict):
            return None
        model_id = api_params.get("modelId")
        return model_id if isinstance(model_id, str) and model_id else None
    except Exception:
        return None


def _extract_bedrock_embedding_usage(model_id: str, parsed_body) -> dict:
    """Map a parsed Bedrock InvokeModel response body → (input_tokens, raw_usage).

    Per design §5.2 Bedrock — per-provider response shape table:
      amazon.titan-embed-* → inputTextTokenCount
      cohere.embed-* → meta.billed_units.{input_tokens, images}
      voyage.voyage-* → usage.total_tokens
    """
    if not isinstance(parsed_body, dict):
        return {"input_tokens": 0, "raw_usage": None, "shape": "unknown"}
    mid = (model_id or "").lower()
    if mid.startswith("amazon.titan-embed"):
        tokens = int(parsed_body.get("inputTextTokenCount") or 0)
        return {
            "input_tokens": tokens,
            # Forward Titan's native field verbatim under its own shape — the
            # server owns the (shape, raw) → billable-units mapping; the SDK
            # only identifies the shape.
            "raw_usage": {"inputTextTokenCount": tokens},
            "shape": "bedrock_titan_embed",
        }
    if mid.startswith("cohere.embed"):
        billed = ((parsed_body.get("meta") or {}).get("billed_units") or {})
        tokens = int(billed.get("input_tokens") or 0)
        images = int(billed.get("images") or 0)
        return {
            "input_tokens": tokens,
            "raw_usage": {"meta": {"billed_units": {"input_tokens": tokens, "images": images}}},
            "shape": "cohere_embed",
        }
    if mid.startswith("voyage.voyage"):
        usage = parsed_body.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        return {
            "input_tokens": tokens,
            "raw_usage": {"total_tokens": tokens},
            "shape": "voyage_embed",
        }
    return {"input_tokens": 0, "raw_usage": None, "shape": "openai_embeddings"}


def _bedrock_embedding_original_provider(model_id: str) -> str:
    """Recover the upstream provider from a Bedrock model id prefix."""
    mid = (model_id or "").lower()
    if mid.startswith("amazon.titan-embed"): return "amazon"
    if mid.startswith("cohere.embed"):       return "cohere"
    if mid.startswith("voyage.voyage"):      return "voyage"
    return "bedrock"


def _bedrock_embedding_kwargs(api_params: dict) -> dict:
    """Synthesize a kwargs dict that the embedding composition parser can read.

    botocore InvokeModel calls bury the customer's input inside `body` as a
    JSON-encoded string/bytes. We decode it once here so the composition
    layer's `_parse_embedding_input` (which looks for `input` / `texts` /
    `inputs`) finds something. Failure to decode is non-fatal — composition
    is best-effort.
    """
    out = {"model": api_params.get("modelId")}
    body = api_params.get("body")
    try:
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            import json as _json
            parsed = _json.loads(body)
        elif isinstance(body, dict):
            parsed = body
        else:
            parsed = None
        if isinstance(parsed, dict):
            for k in ("inputText", "texts", "input", "inputs", "input_type"):
                if k in parsed and k not in ("input_type",):
                    out[k] = parsed[k]
            # Titan uses `inputText` (singular). Surface it under `input` so
            # the embedding parser picks it up.
            if "inputText" in parsed and "input" not in out:
                out["input"] = parsed["inputText"]
    except Exception:
        pass
    return out


def _bedrock_restore_response_body(response, raw_bytes: bytes):
    """Replace the consumed StreamingBody with a fresh one wrapping the same
    bytes so customer code calling `response["body"].read()` still works.
    """
    try:
        from io import BytesIO
        from botocore.response import StreamingBody
        response["body"] = StreamingBody(BytesIO(raw_bytes), len(raw_bytes))
    except Exception:
        # If botocore.response isn't importable for any reason, fall back to
        # a BytesIO — `.read()` still works, and customer code that only
        # calls `.read()` is none the wiser.
        try:
            from io import BytesIO
            response["body"] = BytesIO(raw_bytes)
        except Exception:
            pass


@fail_safe
def _log_bedrock_embedding(model_id: str, api_params: dict, parsed_body, span_name, start_time,
                           obs_key=None):
    """Emit a /log row for a successfully-completed Bedrock InvokeModel
    embedding call. Mirrors what _log_manual does for the registry-based
    embedding wrappers (operation="embedding", per-provider shape, etc.)
    so the dashboard segments the row correctly.

    ``obs_key`` is the caller's key captured right after its check; it keys
    both the routing stamp and the observation drain below. Plain ``None``
    default (not the ``_OBS_KEY_CURRENT`` sentinel — that is defined later in
    this module, so it cannot be a def-time default here); both callers pass
    the captured key explicitly, and ``None`` claims untagged + stale only.
    """
    tp = get_client()
    if tp is None:
        return
    session = get_current_session()
    extracted = _extract_bedrock_embedding_usage(model_id, parsed_body)
    input_tokens = extracted["input_tokens"]
    raw_usage = extracted["raw_usage"]
    shape = extracted["shape"]
    original_provider = _bedrock_embedding_original_provider(model_id)

    # Bedrock-Cohere quirk (AWS-documented): the Embed route strips Cohere's
    # native meta.billed_units block, so the extractor's input_tokens lands
    # at 0 even on successful calls (verified against
    # docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html).
    # Approximate from the decoded request body — same rule-of-thumb the HF
    # and Google-mldev embedding paths use. Applied generically (any shape)
    # so a future Bedrock route with the same gap would also recover.
    if input_tokens == 0 and isinstance(raw_usage, dict):
        approx = _approximate_bedrock_embedding_tokens(api_params)
        if approx > 0:
            input_tokens = approx
            raw_usage = {**raw_usage, "approx_input_tokens": approx, "approximated": True}

    order = session.next_span_order()

    # Parent onto the chain/agent root directly, NOT via manual_span_ids().
    # The Bedrock OTel instrumentor wraps invoke_model, so at this point the
    # active OTel span is the instrumentor's span — which on_end later drops as
    # an embedding duplicate (see _is_instrumentor_embedding_span). Parenting to
    # that span would orphan the embedding row (its parent span is never reported).
    # The instrumentor span's own parent IS the chain span, so the anchored
    # session.root_span_id yields exactly the link a correct nesting would
    # produce. Unscoped throwaways are unanchored → parent "" (no phantom root).
    span_obj = {
        "trace_id": session.trace_id,
        "span_id": random_hex16(),
        "parent_span_id": _session_parent_span_id(session),
        "span_kind": "llm",
        "span_name": span_name or model_id,
        "span_order": order,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    metadata = {"workflow_name": session.workflow_name}
    if session.session_id:
        metadata["session_id"] = session.session_id
    # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
    # session metadata WITHOUT it, then re-add it only when this row belongs
    # to the call that was actually rerouted (exact obs-key match).
    _copy_session_metadata(metadata, session)
    # The threaded key drives BOTH the stamp and the drain below (the callers
    # capture it right after their check; this frame's live var may be stale).
    _stamp_routing_marker(metadata, session, obs_key)

    # Build prompt composition from the decoded request body.
    comp_kwargs = _bedrock_embedding_kwargs(api_params)
    try:
        prompt_comp = build_prompt_composition(original_provider, comp_kwargs, operation="embedding")
    except Exception:
        prompt_comp = []

    # Keyed observation drain (observations only — this seam never owns a
    # keyed local_decision; a claim could only steal a sibling's via the
    # untagged fallback). Own try: fail-open.
    _pending_obs = None
    try:
        _pending_obs = _state.drain_observations(obs_key)
    except Exception:
        _pending_obs = None

    tp.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model=model_id,
        provider="bedrock",
        input_tokens=input_tokens,
        output_tokens=0,
        cached_tokens=0,
        metadata=metadata,
        span=span_obj,
        prompt_composition=prompt_comp,
        response_composition=[],
        usage={"shape": shape, "raw": raw_usage},
        model_extras={"original_provider": original_provider},
        operation="embedding",
        observations=_pending_obs or None,
    )


def _bedrock_embedding_failure_span(session, span_name, start_time):
    """Span dict for a FAILED Bedrock embedding row, mirroring
    _log_bedrock_embedding's success construction (same trace id, fresh
    span id, `_session_parent_span_id` parent, span order). The default
    _emit_call_failure_log span parents onto the ACTIVE OTel span — here
    that is the instrumentor's invoke_model span, which the SpanProcessor
    unconditionally drops (_is_instrumentor_embedding_span), so that parent
    would dangle. Returns None on any failure so the caller degrades to the
    default span. Never raises."""
    try:
        return {
            "trace_id": session.trace_id,
            "span_id": random_hex16(),
            "parent_span_id": _session_parent_span_id(session),
            "span_kind": "llm",
            "span_name": span_name or session.workflow_name,
            "span_order": session.next_span_order(),
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        return None


async def _handle_bedrock_embedding_async(original, args, kwargs):
    """Dedicated handler for `InvokeModel` against a Bedrock embedding model
    on the aiobotocore async path. Runs the pre-flight check, calls the
    original, reads + parses the response body to extract usage, then
    restores the body so customer code can still .read() it.
    """
    # 1. Pre-flight check with embedding intent hint.
    try:
        if not in_agno():
            await _run_async_check(kwargs={"model": (args[2] or {}).get("modelId") if len(args) > 2 else None},
                                   provider="bedrock",
                                   intent={"kind": "embedding"},
                                   # Throwaway {model} body; real InvokeModel
                                   # re-reads *args unchanged, so a reroute swap could
                                   # never reach the provider. Suppress reroute
                                   # (block/allow still enforce) to avoid a phantom
                                   # _tp_routing / misreported savings.
                                   can_reroute=False)
    except TokenPoliceBlockedError:
        raise

    session = get_current_session()
    # Capture this call's obs key right after the check (in-Agno the var
    # holds the Agent-level key — the right owner for that run's entries).
    _obs_key = _state.get_current_obs_key()
    api_params = args[2] if len(args) > 2 else {}
    model_id = (api_params or {}).get("modelId", "")
    _stash_attempt_context(session, "bedrock", "botocore.client", args, kwargs,
                           operation="embedding")
    start_time = datetime.now(timezone.utc)
    span_name = consume_pending_span_name()
    _call_start = _time.monotonic()

    try:
        response = await original(*args, **kwargs)
    except Exception as _exc:
        elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
        try:
            session._call_outcome = build_call_outcome(_exc, elapsed_ms)
        except Exception:
            pass
        _emit_call_failure_log(get_client(), session, obs_key=_obs_key,
                               span_override=_bedrock_embedding_failure_span(
                                   session, span_name, start_time))
        raise

    # 2. Read + restore body so customer code can still consume it. Once the
    # stream has been drained the customer's response is broken until we put
    # the bytes back — they own the response — so the restore MUST run even if
    # parsing the telemetry fails. Keyed on a successful read via try/finally.
    parsed = None
    try:
        body = response.get("body") if isinstance(response, dict) else None
        if body is not None:
            raw_bytes = await body.read() if hasattr(body, "read") and callable(getattr(body, "read")) and asyncio.iscoroutinefunction(body.read) else body.read()
            try:
                import json as _json
                parsed = _json.loads(raw_bytes)
            except Exception:
                parsed = None
            finally:
                # Restore is guaranteed once raw_bytes was obtained; the helper
                # swallows its own errors so it can never break the return path.
                _bedrock_restore_response_body(response, raw_bytes)
    except Exception:
        parsed = None

    # 3. Manual log — operation="embedding".
    _log_bedrock_embedding(model_id, api_params, parsed, span_name, start_time,
                           obs_key=_obs_key)
    return response


def _handle_bedrock_embedding_sync(original, args, kwargs):
    """Sync sibling of _handle_bedrock_embedding_async (botocore.client path)."""
    try:
        if not in_agno():
            _run_sync_check(kwargs={"model": (args[2] or {}).get("modelId") if len(args) > 2 else None},
                            provider="bedrock",
                            intent={"kind": "embedding"},
                            # Throwaway {model} body; real InvokeModel re-reads
                            # *args unchanged, so a reroute swap could never reach the
                            # provider. Suppress reroute (block/allow still enforce) to
                            # avoid a phantom _tp_routing / misreported savings.
                            can_reroute=False)
    except TokenPoliceBlockedError:
        raise

    session = get_current_session()
    # Capture this call's obs key right after the check (see async sibling).
    _obs_key = _state.get_current_obs_key()
    api_params = args[2] if len(args) > 2 else {}
    model_id = (api_params or {}).get("modelId", "")
    _stash_attempt_context(session, "bedrock", "botocore.client", args, kwargs,
                           operation="embedding")
    start_time = datetime.now(timezone.utc)
    span_name = consume_pending_span_name()
    _call_start = _time.monotonic()

    try:
        response = original(*args, **kwargs)
    except Exception as _exc:
        elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
        try:
            session._call_outcome = build_call_outcome(_exc, elapsed_ms)
        except Exception:
            pass
        _emit_call_failure_log(get_client(), session, obs_key=_obs_key,
                               span_override=_bedrock_embedding_failure_span(
                                   session, span_name, start_time))
        raise

    # Once the stream has been drained the customer's response is broken until
    # we put the bytes back — they own the response — so the restore MUST run
    # even if parsing the telemetry fails. Keyed on a successful read.
    parsed = None
    try:
        body = response.get("body") if isinstance(response, dict) else None
        if body is not None:
            raw_bytes = body.read()
            try:
                import json as _json
                parsed = _json.loads(raw_bytes)
            except Exception:
                parsed = None
            finally:
                # Restore is guaranteed once raw_bytes was obtained; the helper
                # swallows its own errors so it can never break the return path.
                _bedrock_restore_response_body(response, raw_bytes)
    except Exception:
        parsed = None

    _log_bedrock_embedding(model_id, api_params, parsed, span_name, start_time,
                           obs_key=_obs_key)
    return response


def _is_litellm_seam(framework, module_path) -> bool:
    """True for the LiteLLM observation seam only.

    Gated on the REGISTRATION (auto-instrument entries carry
    ``framework="litellm"``; ``tp.protect("litellm", ...)`` carries only the
    module path and may name any ``provider=`` slug, e.g. "openai_responses"),
    never on the provider label and never globally. Prefixed ids on OTHER seams
    — OpenRouter's ``openai/gpt-4o-mini``, HuggingFace's ``org/model`` — ARE the
    model identity and must stay byte-untouched.
    """
    try:
        if framework == "litellm":
            return True
        root = str(module_path or "").split(".", 1)[0].strip().lower()
        return root == "litellm"
    except Exception:
        return False


def _litellm_route_context(framework, module_path, kwargs):
    """Per-call LiteLLM route context, or None when it can't be resolved.

    LiteLLM addresses non-OpenAI vendors as ``"<vendor>/<model>"``
    (``gemini/gemini-2.5-flash``), but every other surface — the dashboard, the
    /log row (LiteLLM echoes the bare name), the rule the customer writes —
    carries the BARE name, so a model-scoped rule can never match the verbatim
    request string. Resolve the bare name once per call here; the seam's
    matching/telemetry sites then use it while the customer's call keeps the
    route.

    Returns ``{"request_model", "bare_model", "route_head", "vendor"}``:
      - ``bare_model``/``vendor`` come from LiteLLM's own resolver, which is
        authoritative (two-slash ids, bare names in its registry, and the
        bare-model + ``custom_llm_provider=`` kwarg shape all resolve here).
      - ``route_head`` is the LITERAL text before the first "/" of the REQUEST,
        kept only when stripping it yields exactly ``bare_model``.

    Any failure (not the seam, LiteLLM absent, resolver raising on an unknown
    id, non-str model) yields None and EVERY downstream site keeps today's
    verbatim behavior. The no-op fallback is deliberate: a wrong bare name on a
    request REWRITE breaks the customer's call, which is worse than the bug —
    so this never guesses with a head-split of its own.
    """
    if not _is_litellm_seam(framework, module_path):
        return None
    try:
        model = kwargs.get("model") if isinstance(kwargs, dict) else None
        if not isinstance(model, str) or not model.strip():
            return None
        # Lazy off sys.modules — we never import litellm ourselves. Same access
        # pattern as _resolve_litellm_original_provider; the call CAN raise
        # (BadRequestError on an unknown model id).
        _litellm = sys.modules.get("litellm")
        get_provider = getattr(_litellm, "get_llm_provider", None)
        if get_provider is None:
            return None
        custom = kwargs.get("custom_llm_provider")
        if isinstance(custom, str) and custom.strip():
            resolved = get_provider(model, custom_llm_provider=custom)
        else:
            resolved = get_provider(model)
        if not isinstance(resolved, (tuple, list)) or len(resolved) < 2:
            return None
        bare = resolved[0]
        if not isinstance(bare, str) or not bare:
            return None
        vendor = resolved[1].strip() if isinstance(resolved[1], str) else ""
        route_head = None
        if "/" in model:
            head, rest = model.split("/", 1)
            if head and rest == bare:
                route_head = head
        return {
            "request_model": model,
            "bare_model": bare,
            "route_head": route_head,
            "vendor": vendor,
        }
    except Exception:
        return None


def _litellm_bare(route_ctx, model):
    """Bare form of ``model`` under this call's LiteLLM route context.

    Maps the request string to its resolved bare name, and — because the
    write-back re-prefixes with the same literal head — any post-swap value
    carrying that head. Everything else (no context, unrelated string) passes
    through untouched.
    """
    if not route_ctx or not isinstance(model, str) or not model:
        return model
    try:
        if model == route_ctx.get("request_model"):
            return route_ctx.get("bare_model") or model
        head = route_ctx.get("route_head")
        if head and model.startswith(head + "/"):
            return model[len(head) + 1:] or model
    except Exception:
        pass
    return model


def _litellm_route_target(route_ctx, target_model):
    """The value to write into ``kwargs["model"]`` for a rerouted LiteLLM call.

    Mirrors the REQUEST's own format: a prefixed request gets the same literal
    head back, a bare request (vendor came from ``custom_llm_provider=`` or
    LiteLLM's registry) stays bare. The head is NEVER derived from the
    directive's provider — the collector canonicalizes ``gemini``→``google`` and
    ``google/…`` is a different LiteLLM route than ``gemini/…``, so that would
    break the customer's call.

    A target that ALREADY carries this route's head is written verbatim: rule
    targets reaching Redis outside the backend's authoring route (seed scripts,
    direct Prisma writes) can be authored prefixed, and double-prefixing it
    (``gemini/gemini/…``) would raise BadRequestError into the customer's app.
    The check is head-specific, so the legitimate gateway case — an
    ``openrouter/…`` request whose target is ``anthropic/claude-3-haiku`` —
    still gets the head.
    """
    if not route_ctx or not isinstance(target_model, str) or not target_model:
        return target_model
    try:
        head = route_ctx.get("route_head")
        if head and not target_model.startswith(head + "/"):
            return f"{head}/{target_model}"
    except Exception:
        pass
    return target_model


def _apply_reroute(result, kwargs, provider, serving_unverified: bool = False,
                   model_hint=None, route_ctx=None):
    """If the /check response carries a REROUTE directive in ENFORCE mode,
    mutate `kwargs['model']` to the target model and stash original/actual
    metadata in the session so /log can report routing details.

    Shadow-mode reroute directives are ignored on the SDK side — the
    service has already audited the would-be reroute.

    Returns ``"applied"`` | ``"rejected"`` | ``"noop"`` for audit wiring.
    On reject (unappliable call shape / cross-provider / serving-unverified),
    pushes a ``reroute_rejected`` observation so /log can emit
    REROUTE_REJECTED even when local eval did not run (State B). Fail-safe:
    never throws into the customer's LLM call.

    ``serving_unverified``: unrecognized custom base_url — refuse the
    swap even when the module slug would otherwise equal the target (server
    may still return a REROUTE based on the module provider in the check payload).

    ``model_hint``: the pre-flight's matching-only model hint (kwargs-less
    call shapes). Audit context ONLY — used for ``from.model`` on a rejection
    observation; never a swap target.

    ``route_ctx``: the LiteLLM seam's per-call route context (None everywhere
    else). Present, it makes the cross-provider gate compare against the
    underlying vendor instead of the framework slug, the no-op check and every
    audit field read the BARE model, and the write-back re-applies the request's
    own route head — see _litellm_route_context.
    """
    try:
        reroute = result.get("reroute") if isinstance(result, dict) else None
        if not reroute or not isinstance(reroute, dict):
            return "noop"
        if reroute.get("mode") != "enforce":
            return "noop"
        target_model = reroute.get("model")
        if not target_model or not isinstance(target_model, str):
            return "noop"

        def _push_rejected(reason: str) -> None:
            try:
                from_model = ""
                if isinstance(kwargs, dict):
                    from_model = _litellm_bare(route_ctx, kwargs.get("model")) or ""
                if not from_model and isinstance(model_hint, str):
                    from_model = model_hint
                obs = {
                    "rule_id": reroute.get("rule_id"),
                    "outcome": "reroute_rejected",
                    "mode": "enforce",
                    "rejection_reason": reason,
                    "reroute": {
                        "from": {
                            "provider": (reroute.get("original") or {}).get("provider") or provider or "",
                            "model": from_model,
                        },
                        "to": {
                            "provider": reroute.get("provider") or "",
                            "model": target_model,
                        },
                    },
                }
                # Omit rule_name when missing/empty so Python does not emit
                # `"rule_name": null` (Node JSON.stringify already drops undefined).
                rn = reroute.get("rule_name")
                if isinstance(rn, str) and rn:
                    obs["rule_name"] = rn
                _state.push_observation(obs)
            except Exception:
                pass

        # A live ENFORCE directive on a call shape whose kwargs carry no
        # top-level "model" (framework hint-only paths) can never be applied
        # here — record the refusal instead of resolving silently. Ordered
        # FIRST so an unappliable shape is reported as exactly that, not
        # misattributed to serving_unverified or (when the framework provider
        # couldn't be derived) cross_provider_unsupported.
        if not isinstance(kwargs, dict) or "model" not in kwargs:
            _push_rejected("unappliable_call_shape")
            return "rejected"
        if serving_unverified:
            _push_rejected("serving_unverified")
            return "rejected"
        # Skip cross-provider reroutes so this apply-path matches the local
        # evaluator's State A rejection (`cross_provider_unsupported`). Run BOTH
        # the reroute target provider and the call provider through the SAME
        # effective_provider() helper the evaluator uses, so trim/case and alias
        # slugs (e.g. together_ai vs together) canonicalize identically on both
        # sides. Fires only when a target provider is present and differs;
        # absent/falsy target provider preserves the existing fallback.
        # On the LiteLLM seam ``provider`` is the FRAMEWORK slug ("litellm"), so
        # a vendor target could never equal it and every reroute would reject.
        # Compare against the underlying vendor the router resolved instead —
        # a genuinely cross-vendor target (gemini request, anthropic target)
        # still rejects.
        call_provider = provider or ""
        if route_ctx and route_ctx.get("vendor"):
            call_provider = route_ctx["vendor"]
        reroute_provider = reroute.get("provider")
        if reroute_provider and (
            _local_evaluator.effective_provider(reroute_provider)
            != _local_evaluator.effective_provider(call_provider)
        ):
            _push_rejected("cross_provider_unsupported")
            return "rejected"
        # Defense-in-depth (unreachable: the unappliable_call_shape reject
        # above already returned) — mirrors Node's retained swap gate.
        if not isinstance(kwargs, dict) or "model" not in kwargs:
            return "noop"
        # An alias-targeting rule also matches the dated snapshot of the
        # same model — rewriting there unpins the customer's snapshot for zero
        # cost delta and emits a phantom REQUEST_REROUTED. Nothing applied, so
        # no observation either (this is not a rejection).
        # Both sides bared: a target authored WITH this route's head
        # (`gemini/gemini-2.5-flash` for a `gemini/gemini-2.5-flash` request) is
        # the same model, so it must land here as a clean no-op rather than a
        # phantom self-rewrite. Only the inputs change — is_noop_reroute's own
        # semantics are byte-pinned across both SDKs and the server.
        if _is_noop_reroute(_litellm_bare(route_ctx, kwargs.get("model")),
                            _litellm_bare(route_ctx, target_model)):
            return "noop"
        original_model = _litellm_bare(route_ctx, kwargs.get("model"))
        # The ONLY place the prefixed string survives post-fix: what LiteLLM
        # is actually handed. Bare here would send the call to LiteLLM's
        # OpenAI default and break it.
        kwargs["model"] = _litellm_route_target(route_ctx, target_model)
        try:
            session = get_current_session()
            session.metadata = dict(session.metadata or {})
            # Omit rule_name when missing/empty so Python does not emit
            # `"rule_name": null` (Node JSON.stringify already drops undefined).
            routing = {
                "rule_id": reroute.get("rule_id"),
                "mode": "enforce",
                "original_model": original_model,
                "actual_model": target_model,
                "original_provider": (reroute.get("original") or {}).get("provider") or provider,
                "actual_provider": reroute.get("provider") or provider,
            }
            rn = reroute.get("rule_name")
            if isinstance(rn, str) and rn:
                routing["rule_name"] = rn
            session.metadata["_tp_routing"] = routing
            # The session write above stays exactly as it was — it is the
            # /check-payload input rules may match on, and its shape is pinned
            # by tests. But session metadata is copied into EVERY row emitted
            # afterwards, so on its own it painted this call's marker onto tool
            # spans, agent/chain anchors and unrelated sibling calls. Stash it
            # per-call too: the row side strips `_tp_routing` from every copy
            # and re-adds it only for a model-call row whose own obs key
            # matches this one. Keyed exactly like the `local_decision` stash —
            # this runs DURING check execution, so the contextvar read is this
            # call's own key.
            _stash_routing_marker(session, routing, _state.get_current_obs_key())
        except Exception:
            pass
        return "applied"
    except Exception:
        return "noop"


def _local_evaluate(tp, session, kwargs, provider, intent=None, serving_unverified=False,
                    model_hint=None, route_ctx=None):
    """Run the local evaluator if daemon mode + healthy cache, else None.

    Returns (decision_dict, observations_list, armable_miss) on a usable cache
    hit; (None, [], False) when the SDK should fall through to inline /check.
    ``armable_miss`` is evaluator metadata (never part of the decision): an
    entity-gated directive matched but its tag wasn't on the streamed list —
    see _stale_stream_needs_check.

    ``serving_unverified``: unrecognized custom base_url — REROUTE refuse
    only; ``provider`` stays the module/serving slug for matchConditions/groupBy.

    ``model_hint`` supplies the model for surfaces that don't carry it in
    kwargs (xai proto, framework `self`); matching/audit ONLY — never merged into
    ``kwargs``, so it can't become an appliable reroute target.

    ``route_ctx``: LiteLLM seam route context (None elsewhere) — matches on the
    bare model and hands the evaluator the underlying vendor for its own
    cross-provider gate. ``provider`` stays the framework slug so match/groupBy
    tags line up with /check and /log.
    """
    if tp is None or tp.deployment != "daemon":
        return None, [], False
    pack = _state.get_pack()
    if pack is None or not _state.is_cache_healthy():
        return None, [], False
    target_model = (kwargs or {}).get("model") if isinstance(kwargs, dict) else None
    if not isinstance(target_model, str) or not target_model:
        target_model = model_hint if isinstance(model_hint, str) else None
    target_model = _litellm_bare(route_ctx, target_model)
    try:
        ctx = {
            "model": target_model if isinstance(target_model, str) else "",
            "provider": provider or "",
            "trace_id": session.trace_id,
        }
        # Not a matchable selector field — SDK-only signal so the evaluator's
        # REROUTE cross-provider gate can compare vendors rather than the
        # framework slug (mirrors _apply_reroute).
        if route_ctx and route_ctx.get("vendor"):
            ctx["route_vendor"] = route_ctx["vendor"]
        if serving_unverified:
            ctx["serving_unverified"] = True
        if intent:
            ctx["intent"] = intent
            kind = intent.get("kind") if isinstance(intent, dict) else None
            if isinstance(kind, str) and kind:
                ctx["modality"] = kind
        ev = _local_evaluator.evaluate(pack, session, ctx)
        return (
            ev.get("decision"),
            list(ev.get("observations") or []),
            ev.get("armable_miss") is True,
        )
    except Exception:
        # Local evaluator threw — drop to inline /check.
        return None, [], False


def _stale_stream_needs_check(tp, armable_miss: bool) -> bool:
    """Should a locally-ALLOWED call still be verified with an inline /check?

    Entity arming (``entity_blocked``/``entity_rerouted``) reaches the pack ONLY
    over SSE, so while the stream is down an entity the collector has already
    blocked keeps evaluating to "allowed" locally — the pack itself stays
    perfectly healthy, so no existing signal catches it. Verify only when BOTH
    hold: the allow hinged on an entity-list miss, and the stream has been down
    longer than the configured grace. Everything else — healthy stream,
    within-grace windows, unguarded traffic, local block/reroute — keeps its
    exact prior path. Pure local reads, fully guarded: on any doubt it returns
    False (today's zero-round-trip behavior).
    """
    if not armable_miss:
        return False
    try:
        grace = getattr(tp, "stream_stale_grace_seconds", 60)
        if isinstance(grace, bool) or not isinstance(grace, (int, float)):
            grace = 60
        return not _state.is_stream_fresh(grace)
    except Exception:
        return False


# Sentinel default for the drain-key parameters below: "read the current
# contextvar at drain time". Safe only when the drain runs in the same
# context as the call's own check with no intervening check (a later check
# in the same context overwrites the var — that is exactly why wrappers
# capture the key into a LOCAL right after their check and pass it down for
# any deferred/stream drain). Distinct from an explicit None, which means
# "unknown claimant: untagged + stale entries only".
_OBS_KEY_CURRENT = object()


def _resolve_obs_key(obs_key):
    """Map the sentinel to the live contextvar; pass through explicit values.
    Never raises."""
    if obs_key is _OBS_KEY_CURRENT:
        return _state.get_current_obs_key()
    return obs_key


@fail_safe
def _emit_call_failure_log(tp, session, obs_key=_OBS_KEY_CURRENT, span_override=None):
    """When the original LLM call raises, dispatch a /log so the failure
    is recorded as a failed-call row with the classified
    error_kind/http_status. Also drains this call's observations (keyed on
    ``obs_key`` — captured by the wrapper right after its check; the
    sentinel default falls back to the live contextvar) so shadow rules
    that fired before the call still get audited.

    Pulls the attempted model + provider stashed by the wrapper before the
    call so the row has real context (not `model='unknown'` / `provider=''`).
    Also pulls any prompt_composition captured pre-call from
    `_pending_compositions` so failed rows still carry forensic context.

    ``span_override``: caller-built span dict used verbatim instead of the
    default manual_span_ids() construction. The default parents onto the
    ACTIVE OTel span — wrong for the Bedrock embedding failure path, where
    the active span is the instrumentor's invoke_model span that the
    SpanProcessor unconditionally drops (the row would carry a dangling
    parent_span_id). Those callers pass a span mirroring their success
    twin's construction; every other call site leaves this None
    (byte-identical behavior)."""
    if tp is None or session is None:
        return
    call_outcome = getattr(session, "_call_outcome", None)
    _call_obs_key = _resolve_obs_key(obs_key)
    observations = _state.drain_observations(_call_obs_key)
    # Keyed claim (this call's own decision only — see the store above). The
    # claim is destructive, but the early return below requires it to be
    # absent, so nothing is dropped on that path.
    local_decision = _claim_local_decision(session, _call_obs_key)
    if not call_outcome and not observations and not local_decision:
        return

    attempted_model = getattr(session, "_attempted_model", None) or "unknown"
    attempted_provider = getattr(session, "_attempted_provider", None) or ""
    # `_attempted_operation` is stashed by manual / embedding wrappers so
    # failure rows from embedding calls don't mis-default to operation="chat".
    attempted_operation = getattr(session, "_attempted_operation", None) or "chat"
    attempted_shape = getattr(session, "_attempted_shape", None)
    attempted_wire_key = getattr(session, "_attempted_wire_key", None)

    # Pull pre-call prompt composition. _capture_prompt_for_call stashes it
    # under the key `{trace_id}:{span_counter}` BEFORE on_start runs, so we
    # use the same key here.
    prompt_composition = None
    try:
        pending = getattr(session, "_pending_compositions", None) or {}
        comp_key = f"{session.trace_id}:{session._span_counter}"
        prompt_composition = (pending.get(comp_key) or {}).get("prompt")
        if prompt_composition:
            # Don't leak this composition into the next successful call.
            try:
                pending.pop(comp_key, None)
            except Exception:
                pass
        if not prompt_composition:
            # Narrow fallback: manual-path wrappers reserve their span order via
            # next_span_order() BEFORE stashing the prompt, so by the time the
            # call raises, _span_counter has already advanced one past the
            # stash key and the lookup above misses. Only when the primary key
            # held no prompt, look one order back — a previous span's stash
            # can't be sitting there (it is popped when that span logs).
            try:
                prev_order = int(getattr(session, "_span_counter", 0) or 0) - 1
                if prev_order >= 0:
                    prev_key = f"{session.trace_id}:{prev_order}"
                    prev_prompt = (pending.get(prev_key) or {}).get("prompt")
                    if prev_prompt:
                        prompt_composition = prev_prompt
                        pending.pop(prev_key, None)
            except Exception:
                pass
    except Exception:
        prompt_composition = None

    # N4 / without an explicit usage block the client synthesizes
    # `openai_compatible_chat`, stamping an OpenAI CHAT shape onto every failed
    # row regardless of the provider that was actually attempted. N4 fixed the
    # EMBEDDING case; generalizes it to chat + every modality so a failed
    # cohere row reads `cohere_chat` (and a failed image row `openai_images`)
    # exactly like its successful siblings. Counts stay zero (a failed call has
    # no usage), so the row is still unmeasured either way.
    #
    # Priority: the registry's explicit shape override (what the success path
    # would have passed as `shape_override`) → the embedding table → the chat
    # table keyed on the serving provider.
    #
    # Wire-key gate: keeps the wire shape on the MODULE client, so an
    # Anthropic-SDK call remapped to a MiniMax base_url logs `anthropic_messages`
    # on success. The gate engages ONLY when the serving slug is absent from the
    # chat table (i.e. we would otherwise fall to the synth default), so no
    # in-table provider's shape changes.
    failure_usage = None
    try:
        if attempted_shape:
            _failure_shape = attempted_shape
        elif attempted_operation == "embedding":
            _failure_shape = _resolve_embedding_shape(attempted_provider)
        else:
            _failure_shape = _resolve_usage_shape(attempted_provider)
            if ((attempted_provider or "").lower() not in _USAGE_SHAPE_BY_PROVIDER
                    and (attempted_wire_key or "").lower() in _USAGE_SHAPE_BY_PROVIDER):
                _failure_shape = _resolve_usage_shape(attempted_wire_key)
        failure_usage = {
            "shape": _failure_shape,
            "raw": {"prompt_tokens": 0, "total_tokens": 0},
        }
    except Exception:
        failure_usage = None

    # A failed LiteLLM call logs provider="litellm" (a framework, not a
    # provider), so the row would carry no deployer hint at all. No response
    # object exists on this path; resolve the underlying vendor from the
    # attempted (request) model alone. Gated on provider == "litellm" — every
    # other failure row's payload is byte-identical (model_extras stays None).
    failure_extras = None
    try:
        if (attempted_provider or "").lower() == "litellm":
            # Prefer the vendor the wrapper already resolved through LiteLLM's
            # own router — `attempted_model` is now stashed BARE, so the
            # request-prefix fallback inside the resolver has nothing to read.
            _orig_provider = (getattr(session, "_attempted_route_vendor", None)
                              or _resolve_litellm_original_provider(attempted_model, None))
            if _orig_provider:
                failure_extras = {"original_provider": _orig_provider}
    except Exception:
        failure_extras = None

    tp.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model=attempted_model,
        provider=attempted_provider,
        # B4: row metadata, not the live session object — a sibling call's
        # applied reroute must not ride this failure row (and this row must
        # never mutate the session's own metadata dict). Keyed to the failed
        # call, so a rerouted call's own failure row DOES keep `_tp_routing`:
        # one of the multi-row cases the store's peek-many semantics exist for.
        metadata=_row_metadata_from_session(session, _call_obs_key),
        span=(span_override if span_override is not None
              else {**manual_span_ids(session), "span_name": session.workflow_name}),
        prompt_composition=prompt_composition,
        local_decision=local_decision,
        observations=observations or None,
        call_outcome=call_outcome,
        usage=failure_usage,
        model_extras=failure_extras,
        operation=attempted_operation,
    )
    session._call_outcome = None
    # No slot to clear — the claim above already removed this call's entry.
    session._attempted_model = None
    session._attempted_provider = None
    session._attempted_operation = None
    session._attempted_shape = None
    session._attempted_wire_key = None
    session._attempted_route_vendor = None


# Bedrock op name -> the per-client method the OTel instrumentor patches.
_BEDROCK_LLM_OPS = {
    "Converse": "converse",
    "ConverseStream": "converse_stream",
    "InvokeModel": "invoke_model",
    "InvokeModelWithResponseStream": "invoke_model_with_response_stream",
}


def _bedrock_chat_failure_handoff(session, module_path, args, obs_key, is_async):
    """On a failed Bedrock CHAT call, hand the enforcer's failure payload to
    the instrumentor span's on_end instead of emitting a second row.

    Unlike every other provider, the OTel Bedrock instrumentor patches the
    per-client methods (converse/...) OUTSIDE this wrapper, so its span is
    still open when this except path runs and ends errored right after the
    re-raise — on_end then emits the canonical failure row (correct
    span_name + parent). Emitting here too is what produced the duplicate.
    Stashes the classified outcome + this call's observations on the session
    for telemetry's on_end merge and returns True — the caller then skips
    _emit_call_failure_log. Any doubt returns False (today's behavior; worst
    case is today's duplicate, never a lost row). Total: never raises.
    """
    try:
        if module_path not in ("botocore.client", "aiobotocore.client"):
            return False
        op = args[1] if len(args) > 1 else None
        meth_name = _BEDROCK_LLM_OPS.get(op)
        if meth_name is None:
            # Non-LLM bedrock-runtime ops (e.g. ApplyGuardrail) get no
            # instrumentor span — their failure rows must keep emitting here.
            return False
        if is_async and op in ("ConverseStream", "InvokeModelWithResponseStream"):
            # aiobotocore streaming: the instrumentor never ends its span on
            # error, so the enforcer row is the ONLY row — do not suppress.
            return False
        # Instrumentor-present detector. The patch closures are plain
        # functions defined in opentelemetry/instrumentation/bedrock/
        # __init__.py, stored as client instance attributes; @wraps copies
        # __module__/__qualname__ from the wrapped botocore method, so the
        # defining module is only visible via the closure's __globals__.
        # A client created before instrumentation (or an absent/old
        # instrumentor) fails this check and keeps today's emit.
        meth = getattr(args[0] if args else None, meth_name, None)
        g = getattr(meth, "__globals__", None)
        if not isinstance(g, dict) or g.get("__name__") != "opentelemetry.instrumentation.bedrock":
            return False
        if session is None:
            return False
        try:
            observations = _state.drain_observations(_resolve_obs_key(obs_key))
        except Exception:
            observations = []
        # This call's pending-composition key (the bedrock capture stashed it
        # at _span_counter - 1: on_start had already consumed the order before
        # the wrapper ran). The merge path pops via on_end's own key; this
        # copy serves only the unconsumed-stash fallback row.
        try:
            comp_key = f"{session.trace_id}:{max(0, int(session._span_counter) - 1)}"
        except Exception:
            comp_key = None
        # Drain observations NOW with the call-scoped key: the instrumentor
        # span's recorded obs key predates this call's /check (the span opens
        # before the wrapper runs), so on_end's own drain cannot claim them.
        session._pending_bedrock_failure = {
            "op": op,
            "call_outcome": getattr(session, "_call_outcome", None),
            "observations": observations,
            "model": getattr(session, "_attempted_model", None),
            "provider": getattr(session, "_attempted_provider", None),
            "comp_key": comp_key,
            # CLAIMED here (keyed to this call), not peeked: the instrumentor
            # span's recorded obs key predates this call's /check, so on_end
            # can no longer claim this call's keyed decision itself. Whichever
            # consumer wins the stash — telemetry's merge or the unconsumed
            # fallback below — ships it; a concurrent sibling's decision stays
            # in the store for its own row.
            "local_decision": _claim_local_decision(
                session, _resolve_obs_key(obs_key)),
        }
        # Mirror _emit_call_failure_log's cleanup for the state on_end does
        # NOT read: _call_outcome must not leak into a later
        # _flush_deferred_spans; _attempted_* must not leak into the next
        # failure row. The pending prompt composition is deliberately left in
        # place — on_end reads it (peek-not-pop on errored spans) and pops it
        # on the merge path.
        session._call_outcome = None
        session._attempted_model = None
        session._attempted_provider = None
        session._attempted_operation = None
        session._attempted_shape = None
        session._attempted_wire_key = None
        session._attempted_route_vendor = None
        return True
    except Exception:
        return False


def _flush_unconsumed_bedrock_failure(session):
    """Safety net for _bedrock_chat_failure_handoff: if the instrumentor span
    never emitted (detector wrong about span behavior, sampling, instrumentor
    version drift), the stashed failure would be silently lost. Emit it as a
    standalone failure row at the next wrapped call's entry — a late row
    beats a lost row. O(1) when no stash exists (a single attr check).
    Never raises."""
    try:
        stash = getattr(session, "_pending_bedrock_failure", None)
        if not stash:
            return
        session._pending_bedrock_failure = None
        tp = get_client()
        if tp is None or not isinstance(stash, dict):
            return
        call_outcome = stash.get("call_outcome")
        observations = stash.get("observations")
        local_decision = stash.get("local_decision")
        if not call_outcome and not observations and not local_decision:
            return
        # No session-side clear here any more: the handoff CLAIMED the failed
        # call's decision out of the keyed store when it built this stash, so
        # the stash owns the only copy and a concurrent call's freshly stashed
        # decision was never reachable from here in the first place.
        # Pop the failed call's pending composition (on_end never ran to pop
        # it) and carry the prompt on the late row like every failure row does.
        prompt_composition = None
        try:
            comp_key = stash.get("comp_key")
            pending = getattr(session, "_pending_compositions", None)
            if comp_key and isinstance(pending, dict):
                prompt_composition = (pending.pop(comp_key, None) or {}).get("prompt")
        except Exception:
            prompt_composition = None
        tp.log_sync(
            user_id=session.user_id,
            paid_plan=session.paid_plan,
            plan_source=getattr(session, "plan_source", None),
            workflow_name=session.workflow_name,
            session_id=session.session_id,
            model=stash.get("model") or "unknown",
            provider=stash.get("provider") or "bedrock",
            # B4: this row is emitted at the NEXT wrapped call's entry, so the
            # live obs key belongs to a DIFFERENT call — stamping from it could
            # misattribute. Strip only, never re-add (`stamp=False`). Costless
            # in practice: bedrock's kwargs-less call shape is unappliable, so
            # a bedrock call never carries an applied reroute to begin with.
            metadata=_row_metadata_from_session(session, None, stamp=False),
            # The original active span is long gone — parent onto the
            # anchored session root (never manual_span_ids).
            span={
                "trace_id": session.trace_id,
                "span_id": random_hex16(),
                "parent_span_id": _session_parent_span_id(session),
                "span_name": session.workflow_name,
            },
            prompt_composition=prompt_composition,
            local_decision=local_decision,
            observations=observations or None,
            call_outcome=call_outcome,
            usage={
                "shape": _resolve_usage_shape(stash.get("provider") or "bedrock"),
                "raw": {"prompt_tokens": 0, "total_tokens": 0},
            },
            operation="chat",
        )
    except Exception:
        pass


def _push_would_reroute_observation(rule_id, from_provider, from_model, to_provider, to_model,
                                    mode="enforce", rule_name=None):
    """SDK dry_run dial suppressed an ENFORCE reroute — ship would_reroute
    so /log can emit WOULD_REROUTE (try-before-enforce KPI). Fail-open.

    ``mode`` is the **rule** executionMode (typically ``"enforce"``), not the
    SDK dial. The dial is stamped only as ``sdk_firewall_mode`` on /log.
    """
    try:
        rule_mode = mode if isinstance(mode, str) and mode else "enforce"
        obs = {
            "rule_id": rule_id,
            "outcome": "would_reroute",
            "mode": rule_mode,
            "reroute": {
                "from": {"provider": from_provider or "", "model": from_model or ""},
                "to": {"provider": to_provider or "", "model": to_model or ""},
            },
        }
        if isinstance(rule_name, str) and rule_name:
            obs["rule_name"] = rule_name
        _state.push_observation(obs)
    except Exception:
        pass


# ── Per-call keyed store for the applied `local_decision` audit stash ────────
#
# WHY: `local_decision` is the SOLE provenance the collector turns into a
# `REQUEST_REROUTED` audit event — there is no server-side fallback. It used to
# live in ONE flat slot on the session (`session._local_decision`), so N
# concurrent calls overwrote each other (N-1 decisions lost) and the first call
# to finish drained the survivor: one applied-reroute event per burst, stamped
# on an arbitrary row carrying an arbitrary sibling's rule/from/to. Worse than
# the under-count, in a heterogeneous burst an UN-rerouted call's /log could
# carry a sibling's decision (a false-positive reroute event).
#
# The store mirrors the observation queue's per-call keying (PR #345) — same key
# source (the call's obs key), same 300s window, same lock discipline — with
# three deliberate differences, because for a local decision MIS-ATTRIBUTION is
# worse than stranding:
#   1. a claim returns AT MOST ONE record (never "everything claimable");
#   2. there is no drain-all / greedy-flush mode — a terminal flush must never
#      stamp an orphaned reroute onto an arbitrary row;
#   3. an expired entry is SWEPT, not claimed — the obs queue ships its stale
#      orphans late on an arbitrary /log; this store drops them.
#
# SCOPE: entries live on the SESSION object (``session._local_decisions``), not
# in module state, so the untagged-claim fallback can never reach across
# sessions — a scoping property the old single slot had and this fix must not
# weaken.
#
# GOLDEN RULE: every helper below swallows its own failures. A failure degrades
# to "no decision" (the event is lost, exactly as it is lost today when a drain
# misses) — never to a wrong attribution, and never to a raise into customer
# code.

# EXPIRY window for stashed-but-unclaimed decisions. Deliberately the same 300s
# the observation queue uses (``_state.OBS_STALE_SECONDS``): the window must
# exceed the longest plausible LLM call duration, or a concurrent /log could
# retire a slow/streaming call's decision while that call is still running.
#
# The obs queue CLAIMS its stale entries (for observations, late delivery beats
# loss). This store deliberately does the opposite: an expired entry is SWEPT —
# dropped silently on the next stash/claim — and never handed to a stranger's
# row. An orphan is either a call whose /log never fired or a call that outlived
# the window; decorating an unrelated row with its rule/from/to would be a
# false-positive REQUEST_REROUTED, and D3's own rationale is that for a local
# decision misattribution is worse than stranding. Cost: a stream that runs
# longer than 300s loses its applied-reroute event.
LOCAL_DECISION_STALE_SECONDS = 300.0

# Hard per-session cap (drop-oldest). Bounds the list even if the clock is
# unreadable and the sweep below can never run.
_LOCAL_DECISION_CAP = 64

# Guards every mutation of every session's ``_local_decisions`` list. A
# ``TPSession`` is shared across threads (see its own ``_counter_lock``), and
# these helpers read an index and then pop it — un-synchronized that races into
# lost own-key claims and orphaned entries. Mirrors ``state._observations_lock``
# (the obs queue this store is modelled on). One module-level lock rather than
# one per session: the critical sections are pure in-memory list scans, so
# contention is irrelevant, and it can never be forgotten on a new session.
# NEVER call back into the store from a ``predicate`` — the lock is not
# reentrant.
_local_decisions_lock = threading.Lock()


def _local_decision_entries(session):
    """The session's entry list, or None when absent/corrupt. Never raises.

    Lock-free primitive — every caller below holds ``_local_decisions_lock``.
    """
    try:
        if session is None:
            return None
        entries = getattr(session, "_local_decisions", None)
        return entries if isinstance(entries, list) else None
    except Exception:
        return None


def _sweep_local_decisions(entries):
    """Drop expired entries in place. Caller holds the lock; never raises.

    Runs at the top of stash AND claim, so an expired decision dies silently
    instead of riding a later, unrelated /log (see the constant above). A clock
    that cannot be read sweeps nothing — the cap still bounds the list.
    """
    try:
        now = _time.monotonic()
    except Exception:
        return
    try:
        for i in range(len(entries) - 1, -1, -1):
            e = entries[i]
            if (not isinstance(e, dict)
                    or (now - (e.get("ts") or 0.0)) > LOCAL_DECISION_STALE_SECONDS):
                entries.pop(i)
    except Exception:
        pass


def _stash_local_decision_entry(session, ld, key):
    """Stash one call's applied decision under its obs key.

    Same key => REPLACE: repeated stashes within one logical call keep today's
    last-wins semantics and never leave a duplicate behind for a sibling to
    claim. A None key (no obs key current — degraded/mocked paths) always
    appends: two untagged entries may belong to two different calls, and
    dropping one would lose an event the old slot would also have lost.
    """
    try:
        if session is None or not ld:
            return
        with _local_decisions_lock:
            entries = _local_decision_entries(session)
            if entries is None:
                entries = []
                session._local_decisions = entries
            _sweep_local_decisions(entries)
            try:
                ts = _time.monotonic()
            except Exception:
                ts = 0.0
            if key:
                for i in range(len(entries) - 1, -1, -1):
                    e = entries[i]
                    if isinstance(e, dict) and e.get("key") == key:
                        entries[i] = {"ld": ld, "key": key, "ts": ts}
                        return
            entries.append({"ld": ld, "key": key, "ts": ts})
            while len(entries) > _LOCAL_DECISION_CAP:
                entries.pop(0)
    except Exception:
        pass  # fail-open: a decision we could not stash is simply not audited


def _claim_local_decision(session, key):
    """Claim AT MOST ONE decision for a /log about to be dispatched.

    Expired entries are swept first (see ``_sweep_local_decisions``), so what
    remains is only live decisions. Precedence — own key -> untagged:
      1. the entry stashed under ``key`` (this call's own decision; the newest
         when several share the key, and every same-key entry is consumed so a
         superseded one can never resurface);
      2. the newest untagged entry (degraded paths with no obs key — reproduces
         the old single-slot behavior exactly).
    Anything else stays put for its OWN call's drain — that is the whole fix.
    There is deliberately NO stale-claim arm: an orphan is swept, never
    attached to a stranger's row.
    """
    try:
        with _local_decisions_lock:
            entries = _local_decision_entries(session)
            if not entries:
                return None
            _sweep_local_decisions(entries)
            if key:
                found = None
                for i in range(len(entries) - 1, -1, -1):
                    e = entries[i]
                    if isinstance(e, dict) and e.get("key") == key:
                        # Newest wins; every same-key entry is removed either way.
                        if found is None:
                            found = e.get("ld")
                        entries.pop(i)
                if found is not None:
                    return found
            for i in range(len(entries) - 1, -1, -1):
                e = entries[i]
                if isinstance(e, dict) and e.get("key") is None:
                    entries.pop(i)
                    return e.get("ld")
            return None
    except Exception:
        return None


def _drop_local_decision(session, key, predicate=None):
    """Deliberately DISCARD a call's own decision (the applied action was
    undone — e.g. an anthropic stream manager that could not be rebuilt around
    the swap, so the request on the wire carries the ORIGINAL model).

    Strictly narrower than a claim: removes entries under ``key`` only (or,
    with a None key, the newest untagged entry — the degraded equivalent).
    A drop is destructive and must never be able to delete a sibling call's
    pending decision. ``predicate`` narrows further; it must not re-enter the
    store (the lock is not reentrant).
    """
    try:
        with _local_decisions_lock:
            entries = _local_decision_entries(session)
            if not entries:
                return

            def _matches(ld):
                if predicate is None:
                    return True
                try:
                    return bool(predicate(ld))
                except Exception:
                    return False

            if key:
                for i in range(len(entries) - 1, -1, -1):
                    e = entries[i]
                    if isinstance(e, dict) and e.get("key") == key and _matches(e.get("ld")):
                        entries.pop(i)
                return
            for i in range(len(entries) - 1, -1, -1):
                e = entries[i]
                if isinstance(e, dict) and e.get("key") is None and _matches(e.get("ld")):
                    entries.pop(i)
                    return
    except Exception:
        pass  # fail-open: worst case the decision is claimed later, as today


def _stash_local_decision(session, outcome: str, rule_id, verified_by_check: bool, reroute=None,
                          rule_name=None):
    if session is None:
        return
    decision = {
        "outcome": outcome,
        "rule_id": rule_id,
        "mode": "enforce",
        "verified_by_check": verified_by_check,
    }
    # Surface rule_name on /log applied rows (audit/routing UIs).
    rn = rule_name if isinstance(rule_name, str) and rule_name else None
    if rn is None and isinstance(reroute, dict):
        cand = reroute.get("rule_name")
        if isinstance(cand, str) and cand:
            rn = cand
    if rn:
        decision["rule_name"] = rn
    if reroute is not None:
        decision["reroute"] = reroute
    # Keyed per-call stash (replaces the old flat `session._local_decision`
    # slot, which N concurrent calls overwrote). Every stash site runs DURING
    # `_run_sync_check`/`_run_async_check` execution, which mints this call's
    # obs key at entry, so the contextvar read here is this call's own key —
    # the same key its /log drain claims with. None (no key current / partial
    # mock) stashes untagged, which any claim takes: exactly the old
    # single-slot behavior on degraded paths.
    _stash_local_decision_entry(session, decision, _state.get_current_obs_key())


# -- Per-call keyed store for the applied-reroute marker (`_tp_routing`) ------
#
# WHY: `_apply_reroute` records the swap it just made on the SESSION
# (``session.metadata["_tp_routing"]``), and session metadata is copied into
# EVERY row the SDK emits from then on. So one rerouted call painted its marker
# onto every later row of that session: tool spans (which execute no model call
# at all), agent/chain anchors, and unrelated sibling calls in another modality
# (a TTS row carrying an image rule's marker was observed live). Per-call
# routing provenance in the trace display could not be trusted.
#
# The session write itself is UNCHANGED — it is the /check-payload input rules
# may match on, and both SDKs' shapes are pinned by tests. What changed is the
# ROW side: every copy of session metadata into a /log row drops
# ``_tp_routing`` (``_copy_session_metadata``), and only a MODEL-CALL row whose
# own obs key matches the rerouted call's re-adds it (``_stamp_routing_marker``).
#
# Mirrors the local-decision store above — same key source (the call's obs
# key), same 300s window, same session scoping, same lock discipline — with ONE
# deliberate difference:
#
#   PEEK-MANY. A read NEVER removes the record. One rerouted call can emit
#   several rows (a stream-failure row plus a framework manual row, batch
#   result rows, ...) and every one of them is that call's row, so every one
#   must carry the marker. A `local_decision` is claimed at most once because
#   it is an AUDIT EVENT (emitting it twice double-counts); `_tp_routing` is
#   pure display provenance with no server-side reader, so duplication across a
#   call's own rows is correct, not a hazard.
#
# Attribution is EXACT — deliberately stricter than ``_claim_local_decision``,
# which falls back to an untagged record for a keyed claimant:
#   - a keyed row matches only a record stashed under that same key;
#   - a keyless row matches only an untagged record.
# A near-miss degrades to "no marker on the row" (what the row looked like
# before any reroute existed) instead of stamping a stranger's from->to onto
# it — the exact failure this fix exists to remove.
#
# NEVER filter these rows on model equality instead: the served model
# legitimately differs from ``actual_model`` on gateway / dated-snapshot echoes
# (HuggingFace, Anthropic), and live rows prove that filter wrong.
#
# GOLDEN RULE: every helper below swallows its own failures and degrades to
# "no marker" — never a raise into customer code, never a wrong attribution.

#: The metadata key carrying applied-reroute provenance.
_TP_ROUTING_KEY = "_tp_routing"

#: The span attribute the OTel paths mirror ``_TP_ROUTING_KEY`` into. Derived,
#: never re-spelled: telemetry.py's three read-back loops strip by this exact
#: name, and a drifted literal there would silently reopen the leak. Lives here
#: (not state.py) so it travels with the store, and telemetry.py already
#: lazy-imports from enforcer inside functions — no module-level cycle, and no
#: new export for a partial ``state`` mock to trip over.
_TP_ROUTING_ATTR = "tp.meta." + _TP_ROUTING_KEY

#: Same 300s window as the local-decision store: it must exceed the longest
#: plausible LLM call duration, or a slow/streaming call's own rows would lose
#: the marker while the call is still running.
ROUTING_MARKER_STALE_SECONDS = 300.0

#: Hard per-session cap (drop-oldest), for an unreadable clock.
_ROUTING_MARKER_CAP = 64

#: Guards every mutation of every session's ``_routing_markers`` list. A
#: ``TPSession`` is shared across threads; a separate lock from
#: ``_local_decisions_lock`` so the two stores can never contend or deadlock on
#: each other. NEVER call back into the store while holding it (not reentrant).
_routing_markers_lock = threading.Lock()


def _routing_marker_entries(session):
    """The session's entry list, or None when absent/corrupt. Never raises.

    Lock-free primitive — every caller below holds ``_routing_markers_lock``.
    """
    try:
        if session is None:
            return None
        entries = getattr(session, "_routing_markers", None)
        return entries if isinstance(entries, list) else None
    except Exception:
        return None


def _sweep_routing_markers(entries):
    """Drop expired entries in place. Caller holds the lock; never raises.

    Runs at the top of stash AND peek, so an expired marker dies silently
    instead of decorating a much later row of the same session.
    """
    try:
        now = _time.monotonic()
    except Exception:
        return
    try:
        for i in range(len(entries) - 1, -1, -1):
            e = entries[i]
            if (not isinstance(e, dict)
                    or (now - (e.get("ts") or 0.0)) > ROUTING_MARKER_STALE_SECONDS):
                entries.pop(i)
    except Exception:
        pass


def _stash_routing_marker(session, routing, key):
    """Stash the marker ``_apply_reroute`` just wrote onto the session, under
    the rerouted call's own obs key.

    Same key => REPLACE (a call that re-decides keeps last-wins and never
    leaves a superseded marker behind). A None key always appends: two untagged
    entries may belong to two different calls.
    """
    try:
        if session is None or not routing:
            return
        with _routing_markers_lock:
            entries = _routing_marker_entries(session)
            if entries is None:
                entries = []
                session._routing_markers = entries
            _sweep_routing_markers(entries)
            try:
                ts = _time.monotonic()
            except Exception:
                ts = 0.0
            if key:
                for i in range(len(entries) - 1, -1, -1):
                    e = entries[i]
                    if isinstance(e, dict) and e.get("key") == key:
                        entries[i] = {"routing": routing, "key": key, "ts": ts}
                        return
            entries.append({"routing": routing, "key": key, "ts": ts})
            while len(entries) > _ROUTING_MARKER_CAP:
                entries.pop(0)
    except Exception:
        pass  # fail-open: an un-stashed marker just means the row shows no reroute


def _peek_routing_marker(session, key):
    """Read (WITHOUT removing) the marker belonging to the call identified by
    ``key``. Returns None when this call was not the rerouted one.

    Exact match in BOTH directions — a keyed row never takes an untagged
    record, and a keyless row never takes a keyed one.
    """
    try:
        with _routing_markers_lock:
            entries = _routing_marker_entries(session)
            if not entries:
                return None
            _sweep_routing_markers(entries)
            want = key or None
            for i in range(len(entries) - 1, -1, -1):
                e = entries[i]
                # Newest wins; the record STAYS for this call's other rows.
                if isinstance(e, dict) and e.get("key") == want and e.get("routing"):
                    return e.get("routing")
            return None
    except Exception:
        return None


def _drop_routing_marker(session, key):
    """Deliberately DISCARD a call's own marker (the applied swap was undone —
    see ``_rebuild_after_reroute``: the request on the wire carries the
    ORIGINAL model, so no row of that call may claim a reroute).

    Strictly narrower than a peek: removes entries under ``key`` only (or, with
    a None key, the newest untagged entry). A drop is destructive and must
    never delete a sibling call's pending marker.
    """
    try:
        with _routing_markers_lock:
            entries = _routing_marker_entries(session)
            if not entries:
                return
            if key:
                for i in range(len(entries) - 1, -1, -1):
                    e = entries[i]
                    if isinstance(e, dict) and e.get("key") == key:
                        entries.pop(i)
                return
            for i in range(len(entries) - 1, -1, -1):
                e = entries[i]
                if isinstance(e, dict) and e.get("key") is None:
                    entries.pop(i)
                    return
    except Exception:
        pass  # fail-open: worst case the marker expires unread


def _copy_session_metadata(dest, session):
    """Copy session metadata into a row's /log metadata, MINUS ``_tp_routing``.

    Every row-metadata build goes through here (or the equivalent
    ``tp.meta.*`` read-back skip in telemetry.py), so the marker is absent by
    default and only the rerouted call's own model-call rows put it back via
    ``_stamp_routing_marker``. Tool and agent/chain rows never do. Never
    raises; the marker can only ever be omitted, never added, by a failure.
    """
    try:
        md = getattr(session, "metadata", None)
        # `dest is None`, never `not dest`: the destination is normally an
        # EMPTY dict (falsy) at this point, and a truthiness test here would
        # silently drop every customer metadata key from the row.
        if dest is None or not md:
            return
        for k, v in md.items():
            if k == _TP_ROUTING_KEY:
                continue
            dest[k] = v
    except Exception:
        pass


def _stamp_routing_marker(dest, session, key, serialize=False):
    """Stamp a MODEL-CALL row's metadata with this call's own routing marker,
    if it had one. No-op for every other row.

    ``serialize`` matches the OTel paths, whose metadata values are JSON
    strings (the session-metadata -> span-attribute hop json.dumps() objects);
    manual emitters pass the dict through unchanged, exactly as before.
    """
    try:
        if dest is None:
            return
        routing = _peek_routing_marker(session, key)
        if not routing:
            return
        if not serialize:
            dest[_TP_ROUTING_KEY] = routing
            return
        try:
            import json as _json_local
            dest[_TP_ROUTING_KEY] = _json_local.dumps(routing)
        except Exception:
            pass  # un-serializable => leave the row unmarked
    except Exception:
        pass


def _row_metadata_from_session(session, key, stamp=True):
    """Build a row's /log metadata from the session's metadata alone (no
    ``workflow_name``/``session_id`` injection) for the call sites that used to
    hand ``session.metadata`` itself to ``log_sync`` by reference. Returns a
    fresh dict; the session's own metadata object is never mutated.

    ``key`` is the emitting call's obs key, so a rerouted call's own failure /
    block row keeps its marker while a sibling's row does not. ``stamp=False``
    for rows emitted OUTSIDE their own call's context, where no key can be
    trusted (see ``_flush_unconsumed_bedrock_failure``).
    """
    out = {}
    try:
        _copy_session_metadata(out, session)
        if stamp:
            _stamp_routing_marker(out, session, key)
    except Exception:
        # Never fall back to the raw session metadata here: that is exactly the
        # leak this store exists to close.
        pass
    return out


def _emit_local_block_log(tp, session):
    """Dispatch a /log for a locally-blocked call (LLM call never happened).
    Drains observations + local_decision so the attempt is audited server-side.
    Every call site is inside _run_sync_check/_run_async_check execution, so
    the contextvar still holds THIS call's freshly-minted obs key — reading
    it directly here is safe (no later check can have overwritten it yet).
    Fail-safe: any failure is swallowed.
    """
    try:
        _obs_key = _state.get_current_obs_key()
        local_decision = _claim_local_decision(session, _obs_key)
        observations = _state.drain_observations(_obs_key)
        if not local_decision and not observations:
            return
        tp.log_sync(
            user_id=session.user_id,
            paid_plan=session.paid_plan,
            plan_source=getattr(session, "plan_source", None),
            workflow_name=session.workflow_name,
            session_id=session.session_id,
            # B4: keyed row metadata (see `_emit_call_failure_log`) — a
            # sibling's applied reroute never rides this blocked row.
            metadata=_row_metadata_from_session(session, _obs_key),
            span={**manual_span_ids(session), "span_name": session.workflow_name},
            local_decision=local_decision,
            observations=observations or None,
        )
        # No slot to clear — the claim above already removed this call's entry.
    except Exception:
        pass


@fail_safe
def _run_sync_check(kwargs=None, provider=None, intent=None, can_reroute: bool = True,
                    session=None, serving_unverified: bool = False, model_hint=None,
                    route_ctx=None):
    # Mint a FRESH per-call obs key at entry, ALWAYS (overwrite is fine): an
    # inline awaited/sync call shares the caller's context, so the wrapper
    # frame reads this key right after the check to key its later drain;
    # concurrent asyncio Tasks copy context at creation, so there is no
    # cross-task pollution. Nested delegation (framework wrapper → provider
    # wrapper) overwrites to the inner key — the outer wrapper's own entries
    # then ride the staleness fallback; acceptable (same trace).
    _state.mint_obs_key()
    tp = get_client()
    if not tp:
        return
    # Active for enforce AND dry_run in ALL deployments. dry_run is a faithful
    # preview of enforce: it runs the exact same path (local-eval + inline
    # /check) and differs only in never acting on the final decision. Only
    # 'off' skips the pre-flight entirely.
    active = tp.firewall != "off"
    if not active:
        return
    # dry_run mirrors enforce but suppresses the final block/reroute.
    is_dry_run = tp.firewall == "dry_run"

    # The caller MAY thread in the exact session it will later flush, so a
    # REROUTE decision's ``_local_decision`` audit stash lands on the SAME
    # object (outside an open context every get_current_session() would mint a
    # fresh throwaway — see context.get_current_session). Backward-compatible:
    # callers that don't pass one resolve it here exactly as before.
    session = session if session is not None else get_current_session()
    target_model = (kwargs or {}).get("model") if isinstance(kwargs, dict) else None
    # Surfaces that keep the model off kwargs (xai proto, framework `self`)
    # pass it as ``model_hint``. kwargs wins when it has one. Matching + audit
    # context only — the hint is NEVER written back into kwargs, so
    # _apply_reroute still refuses to swap a body that carries no "model" key
    # (the enforce-mode refusal is recorded as REROUTE_REJECTED
    # `unappliable_call_shape`).
    if not isinstance(target_model, str) or not target_model:
        target_model = model_hint if isinstance(model_hint, str) else None
    # LiteLLM seam only: /check carries the BARE model (what /log records and
    # what the customer's rule names). `provider` deliberately stays "litellm"
    # so /check and /log build the same groupBy tags.
    target_model = _litellm_bare(route_ctx, target_model)

    # ── State A: SSE healthy → local eval is authoritative for the
    # allow path; BLOCK/REROUTE still verify via inline /check.
    # dry_run computes the REAL decision so it reaches /check exactly like
    # enforce; the action is suppressed below.
    local_decision, observations, armable_miss = _local_evaluate(
        tp, session, kwargs, provider, intent, serving_unverified=serving_unverified,
        model_hint=model_hint, route_ctx=route_ctx,
    )
    for obs in observations:
        _state.push_observation(obs)

    # Allow-path: zero round-trip — unless the allow rests on an entity list a
    # stale stream may no longer be delivering, in which case fall through to
    # the SAME inline /check State B uses (bounded timeout, fail-open).
    if (
        local_decision
        and local_decision.get("status") == "allowed"
        and not _stale_stream_needs_check(tp, armable_miss)
    ):
        return

    # Inline /check — used for State A verify AND State B fallback.
    result = tp.check_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        metadata=session.metadata,
        trace_id=session.trace_id,
        model=target_model if isinstance(target_model, str) else None,
        provider=provider,
        intent=intent,
    )
    check_fail_open = bool(result.get("fail_open"))

    # ── State A verify branches ──────────────────────────────────────
    if local_decision and local_decision.get("status") == "blocked":
        if is_dry_run:
            # dry_run: the pre-flight request itself is audited server-side as a
            # dry-run decision; never act or stash locally.
            return
        # Golden-Rule exception (by design): during a server outage (check_fail_open),
        # an entity already present in the last-known-good streamed blocked set still
        # blocks (local_decision == "blocked"). This is not a throw-on-connectivity
        # failure — it is the server's own explicit prior block decision. A non-blocked
        # entity on outage never reaches here and is allowed (fail-open).
        verified = (not check_fail_open) and result.get("status") == "blocked"
        if verified or check_fail_open:
            _stash_local_decision(session, "blocked", local_decision.get("rule_id"),
                                  verified_by_check=not check_fail_open)
            _emit_local_block_log(tp, session)
            raise TokenPoliceBlockedError(
                f"TokenPolice: Budget exceeded — {result.get('reason', 'Policy Violation')}",
                reason=result.get("reason"),
                rule_id=result.get("ruleId") or local_decision.get("rule_id"),
                kind=result.get("detail") or ("budget" if result.get("status") == "blocked" else None),
                trace_id=result.get("traceId"),
            )
        # /check disagrees → the cache was stale. Drop the local decision and proceed.
        return

    # Gate the rerouted branch with `can_reroute`. Throwaway-body
    # callers (Bedrock embedding InvokeModel) pass can_reroute=False so no reroute
    # is applied AND no phantom _tp_routing is stashed on a call whose mutation
    # could never reach the provider. Suppressing reroute here falls through to
    # State B, which still enforces block; only the reroute action is skipped.
    if local_decision and local_decision.get("status") == "rerouted" and can_reroute:
        if is_dry_run:
            # dry_run dial: never apply. Ship would_reroute so /log can emit
            # WOULD_REROUTE (ENFORCE rule + dry_run dial = try-before-enforce KPI).
            rr = local_decision.get("reroute") or {}
            fr = rr.get("from") or {}
            to = rr.get("to") or {}
            _push_would_reroute_observation(
                local_decision.get("rule_id"),
                fr.get("provider") or provider,
                fr.get("model"),
                to.get("provider"),
                to.get("model"),
                mode=local_decision.get("mode") or "enforce",
                rule_name=local_decision.get("rule_name"),
            )
            return
        if check_fail_open or result.get("status") == "allowed":
            # Trust the cache (or /check agrees-by-omission) and rewrite kwargs.
            reroute = local_decision.get("reroute") or {}
            target = (reroute.get("to") or {})
            # Pass rule_name from local decision (directive pack `name`).
            synthetic_reroute = {
                "mode": "enforce",
                "model": target.get("model"),
                "provider": target.get("provider"),
                "rule_id": local_decision.get("rule_id"),
                "original": reroute.get("from") or {},
            }
            rn = local_decision.get("rule_name")
            if isinstance(rn, str) and rn:
                synthetic_reroute["rule_name"] = rn
            synthetic = {"reroute": synthetic_reroute}
            # Only claim applied when the swap landed (refuse pushes an
            # observation inside _apply_reroute — including kwargs-less call
            # shapes, rejected as unappliable_call_shape).
            apply_status = _apply_reroute(synthetic, kwargs, provider,
                                          serving_unverified=serving_unverified,
                                          model_hint=model_hint,
                                          route_ctx=route_ctx)
            if apply_status == "applied":
                _stash_local_decision(session, "rerouted", local_decision.get("rule_id"),
                                      verified_by_check=not check_fail_open,
                                      reroute=local_decision.get("reroute"),
                                      rule_name=local_decision.get("rule_name"))
            return
        # /check returned an explicit different verdict (block or its own reroute).
        if result.get("status") == "blocked":
            _stash_local_decision(session, "blocked", local_decision.get("rule_id"),
                                  verified_by_check=True)
            _emit_local_block_log(tp, session)
            raise TokenPoliceBlockedError(
                f"TokenPolice: Budget exceeded — {result.get('reason', 'Policy Violation')}",
                reason=result.get("reason"),
                rule_id=result.get("ruleId") or local_decision.get("rule_id"),
                kind=result.get("detail") or "budget",
                trace_id=result.get("traceId"),
            )
        apply_status = _apply_reroute(result, kwargs, provider,
                                      serving_unverified=serving_unverified,
                                      model_hint=model_hint,
                                      route_ctx=route_ctx)
        # Only stash applied when the swap landed; rejects already
        # pushed an observation inside _apply_reroute (including kwargs-less
        # call shapes, rejected as unappliable_call_shape).
        if apply_status == "applied":
            rr = result.get("reroute") or {}
            orig = rr.get("original") or {}
            _stash_local_decision(
                session,
                "rerouted",
                result.get("ruleId") or rr.get("rule_id") or (local_decision or {}).get("rule_id"),
                verified_by_check=True,
                reroute={
                    "from": {
                        "provider": orig.get("provider") or provider or "",
                        "model": orig.get("model") or "",
                    },
                    "to": {
                        "provider": rr.get("provider"),
                        "model": rr.get("model"),
                    },
                },
                rule_name=rr.get("rule_name") or (local_decision or {}).get("rule_name"),
            )
        return

    # ── State B: pack absent or unhealthy → original /check flow ─────
    if is_dry_run:
        # dry_run dial: never act. ENFORCE reroute directive still needs a
        # would_reroute observation (check only issues intent for ENFORCE).
        rr = result.get("reroute") if isinstance(result, dict) else None
        if can_reroute and isinstance(rr, dict) and rr.get("mode") == "enforce":
            orig = rr.get("original") or {}
            _push_would_reroute_observation(
                result.get("ruleId") or rr.get("rule_id"),
                orig.get("provider") or provider,
                orig.get("model") or _litellm_bare(
                    route_ctx, (kwargs or {}).get("model") if isinstance(kwargs, dict) else None),
                rr.get("provider"),
                rr.get("model"),
                mode=rr.get("mode") or "enforce",
                rule_name=rr.get("rule_name"),
            )
        return
    if result.get("status") == "blocked":
        _stash_local_decision(session, "blocked", result.get("ruleId"), verified_by_check=True)
        _emit_local_block_log(tp, session)
        raise TokenPoliceBlockedError(
            f"TokenPolice: Budget exceeded — {result.get('reason', 'Policy Violation')}",
            reason=result.get("reason"),
            rule_id=result.get("ruleId"),
            kind=result.get("detail") or "budget",
            trace_id=result.get("traceId"),
        )
    # State-B reroute apply is likewise gated so throwaway-body callers
    # (can_reroute=False) never mutate a kwargs dict that isn't dispatched.
    # Stash applied local_decision so /log emits REQUEST_REROUTED;
    # rejects push observations inside _apply_reroute (kwargs-less call
    # shapes are rejected there as unappliable_call_shape).
    if can_reroute:
        apply_status = _apply_reroute(result, kwargs, provider,
                                      serving_unverified=serving_unverified,
                                      model_hint=model_hint,
                                      route_ctx=route_ctx)
        if apply_status == "applied":
            rr = result.get("reroute") or {}
            orig = rr.get("original") or {}
            _stash_local_decision(
                session,
                "rerouted",
                result.get("ruleId") or rr.get("rule_id"),
                verified_by_check=True,
                reroute={
                    "from": {
                        "provider": orig.get("provider") or provider or "",
                        "model": (orig.get("model")
                                  or _litellm_bare(route_ctx, kwargs.get("model")) or ""),
                    },
                    "to": {
                        "provider": rr.get("provider"),
                        "model": rr.get("model"),
                    },
                },
                rule_name=rr.get("rule_name"),
            )


@fail_safe
async def _run_async_check(kwargs=None, provider=None, intent=None, can_reroute: bool = True,
                           session=None, serving_unverified: bool = False, model_hint=None,
                           route_ctx=None):
    # Mint a FRESH per-call obs key at entry, ALWAYS — see _run_sync_check
    # for the context-sharing rationale and the nested-delegation caveat.
    _state.mint_obs_key()
    tp = get_client()
    if not tp:
        return
    # Active for enforce AND dry_run in ALL deployments — dry_run runs the
    # exact same path as enforce and only suppresses the final action.
    active = tp.firewall != "off"
    if not active:
        return
    is_dry_run = tp.firewall == "dry_run"

    # See _run_sync_check: the caller may thread in the session it later flushes
    # so a REROUTE audit stash lands on that same object. Backward-compatible.
    session = session if session is not None else get_current_session()
    target_model = (kwargs or {}).get("model") if isinstance(kwargs, dict) else None
    # See _run_sync_check: hint fills the model for kwargs-less surfaces,
    # for matching + audit only; never merged into kwargs (an enforce-mode
    # REROUTE there is rejected as unappliable_call_shape).
    if not isinstance(target_model, str) or not target_model:
        target_model = model_hint if isinstance(model_hint, str) else None
    # See _run_sync_check: LiteLLM seam only — /check carries the bare model.
    target_model = _litellm_bare(route_ctx, target_model)

    # dry_run computes the REAL decision so it reaches /check exactly like
    # enforce; the action is suppressed below.
    local_decision, observations, armable_miss = _local_evaluate(
        tp, session, kwargs, provider, intent, serving_unverified=serving_unverified,
        model_hint=model_hint, route_ctx=route_ctx,
    )
    for obs in observations:
        _state.push_observation(obs)

    # See _run_sync_check: a stale-stream armable allow falls through to /check.
    if (
        local_decision
        and local_decision.get("status") == "allowed"
        and not _stale_stream_needs_check(tp, armable_miss)
    ):
        return

    result = await tp.check(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        metadata=session.metadata,
        trace_id=session.trace_id,
        model=target_model if isinstance(target_model, str) else None,
        provider=provider,
        intent=intent,
    )
    check_fail_open = bool(result.get("fail_open"))

    if local_decision and local_decision.get("status") == "blocked":
        if is_dry_run:
            # dry_run: the pre-flight request itself is audited server-side as a
            # dry-run decision; never act or stash locally.
            return
        # Golden-Rule exception (by design): during a server outage (check_fail_open),
        # an entity already present in the last-known-good streamed blocked set still
        # blocks (local_decision == "blocked"). This is not a throw-on-connectivity
        # failure — it is the server's own explicit prior block decision. A non-blocked
        # entity on outage never reaches here and is allowed (fail-open).
        verified = (not check_fail_open) and result.get("status") == "blocked"
        if verified or check_fail_open:
            _stash_local_decision(session, "blocked", local_decision.get("rule_id"),
                                  verified_by_check=not check_fail_open)
            _emit_local_block_log(tp, session)
            raise TokenPoliceBlockedError(
                f"TokenPolice: Budget exceeded — {result.get('reason', 'Policy Violation')}",
                reason=result.get("reason"),
                rule_id=result.get("ruleId") or local_decision.get("rule_id"),
                kind=result.get("detail") or ("budget" if result.get("status") == "blocked" else None),
                trace_id=result.get("traceId"),
            )
        return

    # Gate the rerouted branch with `can_reroute`. Throwaway-body
    # callers (Bedrock embedding InvokeModel) pass can_reroute=False so no reroute
    # is applied AND no phantom _tp_routing is stashed on a call whose mutation
    # could never reach the provider. Suppressing reroute here falls through to
    # State B, which still enforces block; only the reroute action is skipped.
    if local_decision and local_decision.get("status") == "rerouted" and can_reroute:
        if is_dry_run:
            # dry_run dial: never apply. Ship would_reroute so /log can emit
            # WOULD_REROUTE (ENFORCE rule + dry_run dial = try-before-enforce KPI).
            rr = local_decision.get("reroute") or {}
            fr = rr.get("from") or {}
            to = rr.get("to") or {}
            _push_would_reroute_observation(
                local_decision.get("rule_id"),
                fr.get("provider") or provider,
                fr.get("model"),
                to.get("provider"),
                to.get("model"),
                mode=local_decision.get("mode") or "enforce",
                rule_name=local_decision.get("rule_name"),
            )
            return
        if check_fail_open or result.get("status") == "allowed":
            reroute = local_decision.get("reroute") or {}
            target = (reroute.get("to") or {})
            # Pass rule_name from local decision (directive pack `name`).
            synthetic_reroute = {
                "mode": "enforce",
                "model": target.get("model"),
                "provider": target.get("provider"),
                "rule_id": local_decision.get("rule_id"),
                "original": reroute.get("from") or {},
            }
            rn = local_decision.get("rule_name")
            if isinstance(rn, str) and rn:
                synthetic_reroute["rule_name"] = rn
            synthetic = {"reroute": synthetic_reroute}
            # Only claim applied when the swap landed (refuse pushes an
            # observation inside _apply_reroute — including kwargs-less call
            # shapes, rejected as unappliable_call_shape).
            apply_status = _apply_reroute(synthetic, kwargs, provider,
                                          serving_unverified=serving_unverified,
                                          model_hint=model_hint,
                                          route_ctx=route_ctx)
            if apply_status == "applied":
                _stash_local_decision(session, "rerouted", local_decision.get("rule_id"),
                                      verified_by_check=not check_fail_open,
                                      reroute=local_decision.get("reroute"),
                                      rule_name=local_decision.get("rule_name"))
            return
        if result.get("status") == "blocked":
            _stash_local_decision(session, "blocked", local_decision.get("rule_id"),
                                  verified_by_check=True)
            _emit_local_block_log(tp, session)
            raise TokenPoliceBlockedError(
                f"TokenPolice: Budget exceeded — {result.get('reason', 'Policy Violation')}",
                reason=result.get("reason"),
                rule_id=result.get("ruleId") or local_decision.get("rule_id"),
                kind=result.get("detail") or "budget",
                trace_id=result.get("traceId"),
            )
        apply_status = _apply_reroute(result, kwargs, provider,
                                      serving_unverified=serving_unverified,
                                      model_hint=model_hint,
                                      route_ctx=route_ctx)
        # Only stash applied when the swap landed; rejects already pushed an
        # observation inside _apply_reroute (including kwargs-less call
        # shapes, rejected as unappliable_call_shape).
        if apply_status == "applied":
            rr = result.get("reroute") or {}
            orig = rr.get("original") or {}
            _stash_local_decision(
                session,
                "rerouted",
                result.get("ruleId") or rr.get("rule_id") or (local_decision or {}).get("rule_id"),
                verified_by_check=True,
                reroute={
                    "from": {
                        "provider": orig.get("provider") or provider or "",
                        "model": orig.get("model") or "",
                    },
                    "to": {
                        "provider": rr.get("provider"),
                        "model": rr.get("model"),
                    },
                },
                rule_name=rr.get("rule_name") or (local_decision or {}).get("rule_name"),
            )
        return

    if is_dry_run:
        rr = result.get("reroute") if isinstance(result, dict) else None
        if can_reroute and isinstance(rr, dict) and rr.get("mode") == "enforce":
            orig = rr.get("original") or {}
            _push_would_reroute_observation(
                result.get("ruleId") or rr.get("rule_id"),
                orig.get("provider") or provider,
                orig.get("model") or _litellm_bare(
                    route_ctx, (kwargs or {}).get("model") if isinstance(kwargs, dict) else None),
                rr.get("provider"),
                rr.get("model"),
                mode=rr.get("mode") or "enforce",
                rule_name=rr.get("rule_name"),
            )
        return
    if result.get("status") == "blocked":
        _stash_local_decision(session, "blocked", result.get("ruleId"), verified_by_check=True)
        _emit_local_block_log(tp, session)
        raise TokenPoliceBlockedError(
            f"TokenPolice: Budget exceeded — {result.get('reason', 'Policy Violation')}",
            reason=result.get("reason"),
            rule_id=result.get("ruleId"),
            kind=result.get("detail") or "budget",
            trace_id=result.get("traceId"),
        )
    # State-B reroute apply is likewise gated so throwaway-body callers
    # (can_reroute=False) never mutate a kwargs dict that isn't dispatched.
    # Stash applied local_decision so /log emits REQUEST_REROUTED; rejects
    # push observations inside _apply_reroute (kwargs-less call shapes are
    # rejected there as unappliable_call_shape).
    if can_reroute:
        apply_status = _apply_reroute(result, kwargs, provider,
                                      serving_unverified=serving_unverified,
                                      model_hint=model_hint,
                                      route_ctx=route_ctx)
        if apply_status == "applied":
            rr = result.get("reroute") or {}
            orig = rr.get("original") or {}
            _stash_local_decision(
                session,
                "rerouted",
                result.get("ruleId") or rr.get("rule_id"),
                verified_by_check=True,
                reroute={
                    "from": {
                        "provider": orig.get("provider") or provider or "",
                        "model": (orig.get("model")
                                  or _litellm_bare(route_ctx, kwargs.get("model")) or ""),
                    },
                    "to": {
                        "provider": rr.get("provider"),
                        "model": rr.get("model"),
                    },
                },
                rule_name=rr.get("rule_name"),
            )


def _wrap_method(target: dict, override_module=None):
    """Wraps a specific method to inject a pre-flight check.

    ``override_module``: a live class/container object passed to
    ``protect()`` via ``target_object=`` — parity with Node ``options.module``.
    When provided, the class is resolved DIRECTLY from it and ``importlib`` is
    skipped, so an in-app / ``__main__`` / closure-defined class that has no
    importable dotted path can still be protected.
    """
    module_path = target["module"]
    class_name = target["object"]
    method_name = target["method"]
    is_async = target["async"]
    is_manual = target.get("manual", False)
    lc_kind = target.get("langchain")
    li_kind = target.get("llamaindex")
    framework = target.get("framework")  # "litellm" → wrapper sets _in_litellm guard
    modality = target.get("modality")  # image_gen / audio_tts / audio_stt / video_gen / ocr
    shape = target.get("shape")        # explicit usage-shape enum, e.g. "openai_images"
    operation = target.get("operation")  # "embedding" → embedding extraction + composition + log
    async_iter = target.get("async_iter")  # an async-client stream method registered
                                            # sync (async=False) whose consumer still
                                            # iterates with `async for` — bridge the
                                            # protocols so the customer's loop works either way

    if override_module is not None:
        # Resolve the class directly from the caller-supplied live object;
        # no importlib. The 3-arg getattr mirrors the import path below so a
        # missing/wrong attr yields None (→ `if not cls: return`) rather than
        # raising an AttributeError into the customer's protect() call at
        # app-setup time (this function is called with no surrounding try/except).
        if class_name:
            cls = getattr(override_module, class_name, None)
        else:
            cls = override_module
    else:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            return  # SDK not installed, skip

        if class_name:
            cls = getattr(module, class_name, None)
        else:
            cls = module

    if not cls:
        return
        
    original = getattr(cls, method_name, None)
    if not original:
        return

    # Identity guard: never wrap our own wrapper. A module can alias another
    # class's method as a plain class attribute (anthropic.lib.bedrock
    # ._beta_messages: `create = FirstPartyMessagesAPI.create`, bound at
    # import). Today that alias captures the pre-TP function (`import
    # anthropic` imports the module eagerly), but under a future lazy-import
    # refactor it would capture the wrapper already installed on the aliased
    # class — wrapping it again would run the pre-flight check and telemetry
    # TWICE per call (double-billing). The per-(cls, method) `_originals` key
    # check below can't catch this: the alias lives on a DIFFERENT class.
    if getattr(original, "_tp_preflight_wrapper", False):
        return

    key = (cls, method_name)
    if key in _originals:
        return

    _originals[key] = original

    # Detect provider from module path
    provider = _detect_provider(module_path)

    # LangChain framework methods — see _set_langchain_wrapper.
    if lc_kind:
        _set_langchain_wrapper(cls, method_name, original, lc_kind)
        return

    # LlamaIndex framework methods — see _set_llamaindex_wrapper.
    if li_kind:
        _set_llamaindex_wrapper(cls, method_name, original, li_kind)
        return

    # SDKs with no OpenLLMetry instrumentor (e.g. the native OpenRouter SDK):
    # the wrapper does the pre-flight check AND extracts/logs token usage
    # itself, since the telemetry SpanProcessor never observes these calls.
    if is_manual:
        _set_manual_wrapper(cls, method_name, original, provider, is_async,
                            framework=framework, modality=modality, shape=shape,
                            operation=operation, async_iter=async_iter,
                            module_path=module_path)
        return

    if is_async:
        @functools.wraps(original)
        async def async_wrapper(*args, **kwargs):
            # Inside a framework that logs at its own level (LangChain /
            # LiteLLM / LlamaIndex / Pydantic AI) → pass through. Those
            # frameworks ran the single pre-flight check and emit their own
            # framework-level /log entry.
            if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai():
                return await original(*args, **kwargs)

            # Bedrock Specific Safety
            if module_path in ("botocore.client", "aiobotocore.client"):
                if not _is_bedrock_runtime(args[0] if args else None):
                    return await original(*args, **kwargs)
                # InvokeModel for embedding models gets a dedicated manual
                # handler — OpenLLMetry's Bedrock instrumentor doesn't emit
                # usage spans for InvokeModel embeddings, and the response
                # body is a stream we have to read+restore.
                if _is_bedrock_embedding_invoke(args):
                    return await _handle_bedrock_embedding_async(original, args, kwargs)

            # Serving provider from the bound client's base_url (e.g. OpenAI SDK
            # → openrouter.ai reports "openrouter"; Anthropic SDK →
            # api.minimax.io reports "minimax"). Unrecognized custom hosts keep
            # the module provider for match/groupBy and set serving_unverified
            # so REROUTE still refuses.
            # Wire/parse key stays on the module client (OpenAI-shaped
            # bytes stay OpenAI-parsed even when serving remaps to minimax/xai).
            _serving = _resolve_serving_provider(provider, args)
            eff_provider = _serving["provider"]
            _serving_unverified = _serving["serving_unverified"]
            wire_key = _wire_parse_key(provider)

            # 1. Pre-flight check — skipped inside Agno because the Agent.run
            # wrapper already ran the single per-agent check. The rest of
            # the wrapper (composition capture, defer-telemetry plumbing)
            # still runs so each inner LLM call gets its own row with
            # proper prompt/response composition.
            # The check may mutate kwargs["model"] when a REROUTE rule
            # fires in enforce mode.
            # Resolve the session ONCE, up front, and thread it into the check
            # so a REROUTE decision's audit stash lands on the SAME object this
            # wrapper later flushes (outside an open context every
            # get_current_session() would otherwise mint a fresh throwaway).
            session = get_current_session()
            # Gemini TTS reuses generate_content — request-side AUDIO modality
            # must reach /check as audio_tts so operation-scoped rules match
            # pre- and post-call (check/log mirror). Fail-open → None.
            _tts_intent = _google_tts_intent_if_wanted(eff_provider, kwargs)
            # Bedrock chat: botocore passes api_params positionally, so kwargs
            # is empty and the model only exists in args (see
            # _bedrock_model_hint). None — hence a byte-identical /check
            # payload — for every other provider.
            _model_hint = _bedrock_model_hint(module_path, args)
            if not in_agno():
                await _run_async_check(
                    kwargs=kwargs, provider=eff_provider, session=session,
                    intent=_tts_intent, serving_unverified=_serving_unverified,
                    model_hint=_model_hint,
                )
            # Capture THIS call's obs key into a local immediately after the
            # check (inside Agno the var holds the Agent-level check's key —
            # the right owner for that run's observations). Threaded into
            # every later drain of this call: a deferred flush or stream
            # finalize can run after ANOTHER call's check overwrote the var.
            _obs_key = _state.get_current_obs_key()

            # Reserve this call's LLM-span order up front so the instrumentor's
            # on_start consumes this exact order and concurrent calls on one
            # session cannot cross-attribute their post-call stashes. Bedrock's
            # span is created before this wrapper runs, so it keeps the derived
            # order path (no reservation). See tests/test_concurrent_attribution.py.
            _reserved_order = None
            _order_token = None
            if module_path not in ("botocore.client", "aiobotocore.client"):
                try:
                    _reserved_order = session.next_span_order()
                    _order_token = reserve_span_order(_reserved_order)
                except Exception:
                    _reserved_order = None
                    _order_token = None

            # 2. Capture prompt composition (fail-safe, never blocks).
            # Wire key = module client (not serving).
            _capture_prompt_for_call(wire_key, module_path, args, kwargs, order=_reserved_order)
            # Request-side Gemini TTS → stash operation for Mode A deferred /log.
            if _tts_intent:
                _stash_google_tts_hints(session, eff_provider, _reserved_order, kwargs=kwargs)

            # Request-config service tier (Google GenAI takes `service_tier` on the
            # request config and never echoes it in usageMetadata, so the
            # response-side extraction is inert for Gemini). Stash it as a fallback
            # the response tier still overrides. Google-only, fail-open.
            _stash_request_service_tier(session, eff_provider, kwargs, order=_reserved_order)

            # Tell on_end to defer logging to avoid race condition with response_composition.
            # Stash override on recognized remap OR for OTEL providers that never
            # emit gen_ai.system (cohere) — use module provider when remapped is
            # the same (unrecognized keeps module; never blank rows).
            if (
                (eff_provider and eff_provider != provider)
                or provider in _OTEL_PROVIDERS_WITHOUT_SYSTEM_ATTR
            ):
                _stash_provider_override(eff_provider, session, order=_reserved_order)
                # Gateway case only (provider remap, e.g. openai->openrouter):
                # also stash the verbatim vendor-prefixed model slug +
                # original_provider so the instrumented span keeps them.
                if eff_provider != provider:
                    _stash_gateway_request_model(session, kwargs, order=_reserved_order)
            # Forward the serving endpoint so the service can resolve the
            # serving provider (e.g. MiniMax via an Anthropic-compatible base_url).
            _stash_api_base(session, _extract_base_url(args), order=_reserved_order)
            prev_defer = getattr(session, '_defer_telemetry', False)
            session._defer_telemetry = True

            # Stash model/provider so _emit_call_failure_log can populate them
            # if the original LLM call raises (e.g. invalid key, wrong model).
            # When request wants audio out, failure rows also use audio_tts.
            # `wire_key` (module client slug) lets the failure row keep the
            # wire shape a success row would carry on a remapped base_url.
            _stash_attempt_context(
                session, eff_provider, module_path, args, kwargs,
                operation=("audio_tts" if _tts_intent else None),
                wire_key=wire_key,
            )

            _call_start = _time.monotonic()
            # Chat streams opened without include_usage get it injected so
            # streamed spend isn't silently lost (synthetic chunk stripped below).
            _usage_injected = _inject_stream_usage_option(module_path, kwargs)
            try:
                # 3. Call original (OpenLLMetry will handle telemetry internally here)
                try:
                    result = await original(*args, **kwargs)
                except Exception as _exc:
                    # Strip-and-retry only when the rejection is plausibly caused
                    # by the injected option (a strict-compat server answering
                    # 400/422 or naming the param) — refused before generation, so
                    # a retry cannot double-generate. A rate-limit or auth failure
                    # must never trigger a second provider request. The injection
                    # can never fail a call that would otherwise have succeeded.
                    if _usage_injected and _should_retry_without_injection(_exc):
                        _restore_stream_usage_option(kwargs, _usage_injected)
                        _usage_injected = None
                        result = await original(*args, **kwargs)
                    else:
                        raise
            except Exception as _exc:
                elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                try:
                    session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                except Exception:
                    pass
                session._defer_telemetry = prev_defer
                # Failed bedrock CHAT call: the instrumentor span (still open,
                # ends after the re-raise) emits the canonical failure row via
                # on_end — hand our payload to it instead of emitting a
                # duplicate. False (non-bedrock op, async streaming, detector
                # miss) keeps today's emit.
                if not _bedrock_chat_failure_handoff(session, module_path, args,
                                                     _obs_key, is_async=True):
                    _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                raise
            else:
                elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                try:
                    session._call_outcome = build_call_outcome(None, elapsed_ms)
                except Exception:
                    pass
            finally:
                session._defer_telemetry = prev_defer
                # on_start (which runs inside the provider call above) has
                # consumed the reservation by now; the stream/non-stream captures
                # below thread the captured local order, so drop the contextvar so
                # a later unrelated span can't consume a stale reservation.
                if _order_token is not None:
                    reset_span_order(_order_token)

            # 4a. Streaming → wrap the iterator so we can accumulate the
            # assistant response across chunks, keep `_defer_telemetry` set
            # while iteration runs (so the OTel on_end queues rather than
            # dispatches), and then capture composition + flush after the
            # last chunk. Without this, response_composition is empty for
            # every streamed Mode A call.
            # Mode-A parse key MUST be wire (module), never serving —
            # remapped minimax/xai had no Mode-A accumulator branch.
            if _is_stream(result):
                return _wrap_mode_a_async_stream(result, wire_key, session, prev_defer,
                                                 req_start_mono=_call_start,
                                                 suppress_usage_chunk=bool(_usage_injected),
                                                 order=_reserved_order,
                                                 obs_key=_obs_key)

            # 4b. Non-streaming → capture composition from the materialized
            # response object directly. Wire key (module), not serving.
            _capture_response_composition(wire_key, result, order=_reserved_order)

            if not session._defer_telemetry:
                _flush_deferred_spans(session, obs_key=_obs_key)

            return result

        # Identity marker for _wrap_method's alias guard (see there).
        async_wrapper._tp_preflight_wrapper = True
        setattr(cls, method_name, async_wrapper)
    else:
        @functools.wraps(original)
        def sync_wrapper(*args, **kwargs):
            # See async_wrapper above for the rationale on which framework
            # guards short-circuit vs. fall through.
            if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai():
                return original(*args, **kwargs)

            # Bedrock Specific Safety
            if module_path in ("botocore.client", "aiobotocore.client"):
                if not _is_bedrock_runtime(args[0] if args else None):
                    return original(*args, **kwargs)
                if _is_bedrock_embedding_invoke(args):
                    return _handle_bedrock_embedding_sync(original, args, kwargs)

            # Serving provider from the bound client's base_url (see async_wrapper).
            # Wire/parse key stays on the module client.
            _serving = _resolve_serving_provider(provider, args)
            eff_provider = _serving["provider"]
            _serving_unverified = _serving["serving_unverified"]
            wire_key = _wire_parse_key(provider)

            # 1. Pre-flight check (skipped inside Agno — see async_wrapper).
            # The check may mutate kwargs["model"] when a REROUTE rule
            # fires in enforce mode.
            # Resolve the session ONCE, up front, and thread it into the check
            # (see async_wrapper) so a REROUTE audit stash lands on the SAME
            # object this wrapper later flushes.
            session = get_current_session()
            # Gemini TTS reuses generate_content — request-side AUDIO modality
            # must reach /check as audio_tts so operation-scoped rules match
            # pre- and post-call (check/log mirror). Fail-open → None.
            _tts_intent = _google_tts_intent_if_wanted(eff_provider, kwargs)
            # Bedrock chat model lives in positional args, not kwargs (see
            # async_wrapper / _bedrock_model_hint). None elsewhere.
            _model_hint = _bedrock_model_hint(module_path, args)
            if not in_agno():
                _run_sync_check(
                    kwargs=kwargs, provider=eff_provider, session=session,
                    intent=_tts_intent, serving_unverified=_serving_unverified,
                    model_hint=_model_hint,
                )
            # Capture THIS call's obs key into a local immediately after the
            # check (see async_wrapper) — threaded into every later drain.
            _obs_key = _state.get_current_obs_key()

            # Reserve this call's LLM-span order up front (see async_wrapper) so
            # the instrumentor's on_start consumes it and concurrent calls on one
            # session cannot cross-attribute their post-call stashes. Bedrock is
            # excluded (its span predates this wrapper).
            _reserved_order = None
            _order_token = None
            if module_path not in ("botocore.client", "aiobotocore.client"):
                try:
                    _reserved_order = session.next_span_order()
                    _order_token = reserve_span_order(_reserved_order)
                except Exception:
                    _reserved_order = None
                    _order_token = None

            # 2. Capture prompt composition (fail-safe, never blocks).
            # Wire key = module client (not serving).
            _capture_prompt_for_call(wire_key, module_path, args, kwargs, order=_reserved_order)
            # Request-side Gemini TTS → stash operation for Mode A deferred /log.
            if _tts_intent:
                _stash_google_tts_hints(session, eff_provider, _reserved_order, kwargs=kwargs)

            # Request-config service tier (Google GenAI takes `service_tier` on the
            # request config and never echoes it in usageMetadata, so the
            # response-side extraction is inert for Gemini). Stash it as a fallback
            # the response tier still overrides. Google-only, fail-open.
            _stash_request_service_tier(session, eff_provider, kwargs, order=_reserved_order)

            # Tell on_end to defer logging (see async_wrapper for OTEL gate).
            if (
                (eff_provider and eff_provider != provider)
                or provider in _OTEL_PROVIDERS_WITHOUT_SYSTEM_ATTR
            ):
                _stash_provider_override(eff_provider, session, order=_reserved_order)
                # Gateway case only (provider remap, e.g. openai->openrouter):
                # also stash the verbatim vendor-prefixed model slug +
                # original_provider so the instrumented span keeps them.
                if eff_provider != provider:
                    _stash_gateway_request_model(session, kwargs, order=_reserved_order)
            # Forward the serving endpoint so the service can resolve the
            # serving provider (e.g. MiniMax via an Anthropic-compatible base_url).
            _stash_api_base(session, _extract_base_url(args), order=_reserved_order)
            prev_defer = getattr(session, '_defer_telemetry', False)
            session._defer_telemetry = True

            # Stash model/provider so _emit_call_failure_log can populate them
            # if the original LLM call raises (e.g. invalid key, wrong model).
            # When request wants audio out, failure rows also use audio_tts.
            # `wire_key` (module client slug) lets the failure row keep the
            # wire shape a success row would carry on a remapped base_url.
            _stash_attempt_context(
                session, eff_provider, module_path, args, kwargs,
                operation=("audio_tts" if _tts_intent else None),
                wire_key=wire_key,
            )

            _call_start = _time.monotonic()
            # Chat streams opened without include_usage get it injected so
            # streamed spend isn't silently lost (synthetic chunk stripped below).
            _usage_injected = _inject_stream_usage_option(module_path, kwargs)
            try:
                # 3. Call original (OpenLLMetry will handle telemetry internally here)
                try:
                    result = original(*args, **kwargs)
                except Exception as _exc:
                    # See async_wrapper — strip-and-retry once only when the
                    # rejection is plausibly caused by the injected option.
                    if _usage_injected and _should_retry_without_injection(_exc):
                        _restore_stream_usage_option(kwargs, _usage_injected)
                        _usage_injected = None
                        result = original(*args, **kwargs)
                    else:
                        raise
            except Exception as _exc:
                elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                try:
                    session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                except Exception:
                    pass
                session._defer_telemetry = prev_defer
                # See async_wrapper: failed bedrock CHAT calls hand off to the
                # instrumentor span's on_end row instead of duplicating it.
                if not _bedrock_chat_failure_handoff(session, module_path, args,
                                                     _obs_key, is_async=False):
                    _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                raise
            else:
                elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                try:
                    session._call_outcome = build_call_outcome(None, elapsed_ms)
                except Exception:
                    pass
            finally:
                session._defer_telemetry = prev_defer
                # See async_wrapper: on_start consumed the reservation during the
                # provider call; drop the contextvar so a later unrelated span
                # can't consume a stale reservation (the captures below thread the
                # captured local order).
                if _order_token is not None:
                    reset_span_order(_order_token)

            # 4a. Streaming → wrap the iterator (see async_wrapper above).
            # An async-client stream method can be registered
            # async=False (it returns its iterator without awaiting). When
            # the consumer iterates with `async for` (async_iter hint) or the
            # result is async-only, hand back an async generator;
            # _wrap_mode_a_async_stream bridges a sync underlying so the
            # customer's loop works regardless of the underlying protocol.
            # Mode-A parse key MUST be wire (module), never serving.
            if _is_stream(result):
                if async_iter or (is_async_iterator(result) and not is_sync_iterator(result)):
                    return _wrap_mode_a_async_stream(result, wire_key, session, prev_defer,
                                                     req_start_mono=_call_start,
                                                     suppress_usage_chunk=bool(_usage_injected),
                                                     order=_reserved_order,
                                                     obs_key=_obs_key)
                return _wrap_mode_a_sync_stream(result, wire_key, session, prev_defer,
                                                req_start_mono=_call_start,
                                                suppress_usage_chunk=bool(_usage_injected),
                                                order=_reserved_order,
                                                obs_key=_obs_key)

            # 4b. Non-streaming → capture composition from the materialized
            # response object directly. Wire key (module), not serving.
            _capture_response_composition(wire_key, result, order=_reserved_order)

            if not session._defer_telemetry:
                _flush_deferred_spans(session, obs_key=_obs_key)

            return result

        # Identity marker for _wrap_method's alias guard (see there).
        sync_wrapper._tp_preflight_wrapper = True
        setattr(cls, method_name, sync_wrapper)


def _detect_provider(module_path: str) -> str:
    """Map module path to provider name for composition parsing."""
    mp = module_path.lower()
    if "openrouter" in mp:
        return "openrouter"
    if "cerebras" in mp:
        return "cerebras"
    if "together" in mp:
        return "together"  # together-python — OpenAI-compatible messages/usage
    if "groq" in mp:
        return "groq"  # native groq SDK — OpenAI-compatible; stream usage under x_groq.usage
    if "huggingface" in mp:
        return "huggingface"  # huggingface_hub — OpenAI-compatible messages format
    # Responses API is structurally different from Chat Completions
    # (response.output instead of response.choices, input instead of messages)
    # — must be matched before the generic "openai" branch.
    if "openai.resources.responses" in mp:
        return "openai_responses"
    if "openai" in mp:
        return "openai"
    if "anthropic" in mp:
        return "anthropic"
    if "google" in mp or "genai" in mp:
        return "google"
    if "cohere" in mp:
        return "cohere"  # Cohere v2 uses OpenAI-compatible messages format
    if "mistral" in mp:
        return "mistral"  # mistralai SDK is OpenAI-compatible
    if "xai" in mp:
        return "xai"  # xai-sdk (proto-backed Chat instance — see _build_xai_pseudo_kwargs)
    if "voyageai" in mp:
        return "voyage"  # Anthropic-recommended embedding provider
    if "litellm" in mp:
        return "litellm"  # reported as its own provider; messages/usage are OpenAI-shaped
    if "bedrock" in mp or "botocore" in mp:
        return "bedrock"
    if "langchain" in mp:
        return "langchain"
    return ""


@fail_safe
def _capture_prompt_composition(provider: str, kwargs: dict, order: int = None):
    """Extract prompt composition from call kwargs and store on session.

    ``order`` is the LLM-span order reserved by the wrapper BEFORE the provider
    call, so concurrent calls on one session can't cross-attribute (see
    tests/test_concurrent_attribution.py). When it's None the legacy peek is
    used: the enforcer runs BEFORE telemetry on_start, so _span_counter is still
    at the value that on_start will consume via next_span_order() — i.e. THIS
    call's LLM-span order. Either way the matching response capture (which runs
    AFTER the call, when _span_counter has already advanced past this span in
    agent / streaming modes) keys onto the SAME span instead of `_span_counter
    - 1`. Without this, response composition is stranded on the wrong key for
    every LangChain AgentExecutor and every streamed call. The one-slot mailbox
    below stays as a dormant fallback for callers that pass no explicit order.
    """
    session = get_current_session()
    if order is None:
        order = session._span_counter
    session._mode_a_prompt_order = order
    comp = build_prompt_composition(provider, kwargs)
    if comp:
        if not hasattr(session, '_pending_compositions'):
            session._pending_compositions = {}
        comp_key = f"{session.trace_id}:{order}"
        session._pending_compositions.setdefault(comp_key, {})["prompt"] = comp


@fail_safe
def _stash_request_service_tier(session, provider: str, kwargs: dict, order: int = None):
    """Capture a REQUEST-config service tier for providers that take the tier on
    the request but never echo it in the response. Google GenAI accepts
    ``service_tier`` on the GenerateContentConfig (dict OR object) and its
    response usageMetadata omits it, so the response-side _extract_service_tier
    is inert for Gemini and priority traffic would otherwise bill at standard
    rates. Stashed under the call's comp_key as a FALLBACK: the post-call
    _capture_response_composition only writes the tier when the response actually
    carries one, so a real response tier still wins. Google-only. Fail-open — the
    @fail_safe wrapper swallows a raising attribute/item access so a hostile
    config can never break the customer app."""
    if provider not in ("google", "gemini"):
        return
    config = kwargs.get("config")
    if config is None:
        return
    if isinstance(config, dict):
        raw = config.get("service_tier")
    else:
        raw = getattr(config, "service_tier", None)
    if raw is None:
        return
    # The value may be an enum (``str(raw) == "ServiceTier.PRIORITY"``) or a plain
    # string; take the tail after any "." then lower/trim before canonicalizing.
    token = str(raw).rsplit(".", 1)[-1].strip().lower()
    if not token:
        return
    tier = _SERVICE_TIER_CANONICAL.get(token, "")  # standard/default/auto -> "" (dropped)
    if not tier:
        return
    if order is None:
        order = session._span_counter
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    comp_key = f"{session.trace_id}:{order}"
    session._pending_compositions.setdefault(comp_key, {})["service_tier"] = tier


@fail_safe
def _stash_attempt_context(session, eff_provider: str, module_path: str, args, kwargs,
                           operation: str = None, shape: str = None, wire_key: str = None,
                           route_ctx=None):
    """Stash the attempted model + provider + operation on the session so the
    failure /log can populate `model`, `provider`, and `operation` if the
    original LLM call raises before any OTel span runs. The `operation` hint
    comes from the registry entry (e.g. `"embedding"` for embedding wrappers)
    and prevents the failure-log default of `"chat"` mis-tagging failed
    embedding calls.

    `shape` is the registry's explicit usage-shape override (the same value the
    success path passes as `shape_override`, e.g. `"openai_images"`); `wire_key`
    is the module client's wire slug (`_wire_parse_key`). Both let the failure
    /log reproduce the shape a successful sibling would have carried instead of
    falling back to the client's `openai_compatible_chat` synth.

    `route_ctx` (LiteLLM seam only) strips the route prefix so a failed call
    lands on the SAME model row as its successful siblings, which log LiteLLM's
    bare response echo."""
    if session is None:
        return
    # A bedrock failure handed off to an instrumentor span that never emitted
    # would be lost — flush it as a late standalone row now. O(1) no-op when
    # no stash exists (the overwhelmingly common case).
    if getattr(session, "_pending_bedrock_failure", None) is not None:
        _flush_unconsumed_bedrock_failure(session)
    session._attempted_provider = eff_provider or ""
    session._attempted_operation = operation
    session._attempted_shape = shape
    session._attempted_wire_key = wire_key
    model = None
    if module_path in ("botocore.client", "aiobotocore.client"):
        # All four LLM ops (and the embedding InvokeModel) carry modelId in
        # api_params — without it a failed InvokeModel/embedding row logs
        # model="unknown".
        if len(args) > 1 and args[1] in ("Converse", "ConverseStream",
                                         "InvokeModel", "InvokeModelWithResponseStream"):
            api_params = args[2] if len(args) > 2 else {}
            if isinstance(api_params, dict):
                model = api_params.get("modelId")
    elif isinstance(kwargs, dict):
        model = _litellm_bare(route_ctx, kwargs.get("model"))
    session._attempted_model = model
    # The vendor LiteLLM's router resolved for this call. Stashed (and cleared
    # on every non-LiteLLM call) because the failure row's deployer hint used to
    # be recoverable from the prefix on `_attempted_model`, which is now bare.
    session._attempted_route_vendor = (route_ctx or {}).get("vendor") or None


@fail_safe
def _capture_prompt_for_call(provider: str, module_path: str, args, kwargs, order: int = None):
    """Capture prompt composition, handling the botocore _make_api_call signature.

    ``order`` (when supplied by the wrapper) is this call's reserved LLM-span
    order for the OpenAI/Anthropic-style path. The bedrock path derives its own
    order (its span is created before this runs) and ignores it.

    botocore calls _make_api_call(self, operation_name, api_params), so the LLM
    request lives in args[2] (api_params), not kwargs. Only Converse carries an
    LLM payload; other bedrock-runtime operations are skipped.

    The bedrock OTel instrumentor wraps the client's `converse` method (outside
    `_make_api_call`), so by the time this enforcer wrapper runs, on_start has
    already consumed this span's order — it is _span_counter - 1, unlike the
    OpenAI/Anthropic path where the enforcer runs before on_start.
    """
    if module_path in ("botocore.client", "aiobotocore.client"):
        op = args[1] if len(args) > 1 else None
        if op in ("Converse", "ConverseStream"):
            api_params = args[2] if len(args) > 2 else {}
            session = get_current_session()
            order = max(0, session._span_counter - 1)
            _capture_composition_at(provider, api_params, order, is_response=False)
            _stash_model_at(api_params.get("modelId"), session, order)
        return
    _capture_prompt_composition(provider, kwargs, order=order)


@fail_safe
def _stash_model_at(model: str, session, order: int):
    """Record the full Bedrock model id for the LLM span at the given order.

    The OpenLLMetry bedrock instrumentor strips the vendor prefix from
    gen_ai.request.model (amazon.nova-lite-v1:0 -> nova-lite-v1:0), which breaks
    server-side price resolution. Stash the full Converse modelId so the
    telemetry SpanProcessor can restore it in on_end.
    """
    if not model:
        return
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    comp_key = f"{session.trace_id}:{order}"
    session._pending_compositions.setdefault(comp_key, {})["model"] = model


# Provider-reported service tier -> canonical tier name ("" = the provider's
# default tier or an unrecognized value — never guess a discount). OpenAI
# reports `service_tier` at the response/chunk top level ("default" | "flex" |
# "priority" | legacy "scale"); Anthropic inside the usage object ("standard" |
# "batch" | "priority_tier"). Canonical names match the price table's
# tier_modifiers keys / when:{tier} rules.
_SERVICE_TIER_CANONICAL = {
    "flex": "flex",
    "priority": "priority",
    "priority_tier": "priority",
    "scale": "priority",
    "batch": "batch",
}


def _extract_service_tier(result) -> str:
    """Canonical service tier from a response/chunk, or "". Fail-safe."""
    try:
        raw = getattr(result, "service_tier", None)
        if raw is None and isinstance(result, dict):
            raw = result.get("service_tier")
        if raw is None:
            usage = getattr(result, "usage", None)
            if usage is None and isinstance(result, dict):
                usage = result.get("usage")
            if usage is not None:
                raw = getattr(usage, "service_tier", None)
                if raw is None and isinstance(usage, dict):
                    raw = usage.get("service_tier")
        if not raw:
            return ""
        return _SERVICE_TIER_CANONICAL.get(str(raw).lower(), "")
    except Exception:
        return ""


def _modalities_include_audio(mods) -> bool:
    """True if a response_modalities list/enum sequence names AUDIO. Fail-open."""
    try:
        if mods is None:
            return False
        # Single enum/str value (not a sequence of modalities).
        if isinstance(mods, str):
            return mods.upper() == "AUDIO"
        try:
            iter(mods)
        except TypeError:
            val = getattr(mods, "value", None) or str(mods)
            return str(val).upper() == "AUDIO"
        for m in mods:
            if isinstance(m, str):
                if m.upper() == "AUDIO":
                    return True
            else:
                val = getattr(m, "value", None) or str(m)
                if str(val).upper() == "AUDIO":
                    return True
        return False
    except Exception:
        return False


def _wants_google_audio_out(kwargs) -> bool:
    """Request-side Gemini TTS / native-audio intent.

    True when ``config.response_modalities`` (snake or camel) includes AUDIO, or
    when ``speech_config`` / ``speechConfig`` is present. Walks dict + Pydantic
    config objects. Fail-open — never raises.
    """
    try:
        if not isinstance(kwargs, dict):
            return False
        config = kwargs.get("config")
        if config is None:
            mods = kwargs.get("response_modalities") or kwargs.get("responseModalities")
            speech = kwargs.get("speech_config") or kwargs.get("speechConfig")
        elif isinstance(config, dict):
            mods = config.get("response_modalities") or config.get("responseModalities")
            speech = config.get("speech_config") or config.get("speechConfig")
        else:
            mods = (
                getattr(config, "response_modalities", None)
                or getattr(config, "responseModalities", None)
            )
            speech = (
                getattr(config, "speech_config", None)
                or getattr(config, "speechConfig", None)
            )
        if speech:
            return True
        return _modalities_include_audio(mods)
    except Exception:
        return False


def _is_google_audio_output(result) -> bool:
    """Response-side evidence that Gemini returned audio output.

    Looks at ``usage_metadata.candidates_tokens_details`` (snake + camel) for an
    AUDIO modality with tokenCount > 0, and falls back to response parts whose
    ``inline_data`` / ``inlineData`` MIME is audio/*. Fail-open — never raises.
    """
    try:
        if result is None:
            return False
        um = None
        if isinstance(result, dict):
            um = result.get("usage_metadata") or result.get("usageMetadata")
        else:
            um = getattr(result, "usage_metadata", None) or getattr(result, "usageMetadata", None)
        if um is not None:
            if isinstance(um, dict):
                dets = um.get("candidates_tokens_details") or um.get("candidatesTokensDetails")
            else:
                dets = (
                    getattr(um, "candidates_tokens_details", None)
                    or getattr(um, "candidatesTokensDetails", None)
                )
            if dets:
                for d in dets:
                    try:
                        if isinstance(d, dict):
                            mod = d.get("modality")
                            tc = d.get("token_count")
                            if tc is None:
                                tc = d.get("tokenCount")
                        else:
                            mod = getattr(d, "modality", None)
                            tc = getattr(d, "token_count", None)
                            if tc is None:
                                tc = getattr(d, "tokenCount", None)
                        if str(mod or "").upper() == "AUDIO" and float(tc or 0) > 0:
                            return True
                    except Exception:
                        continue
        # Optional: audio MIME on response parts (TTS returns inline audio blobs).
        cands = (
            result.get("candidates")
            if isinstance(result, dict)
            else getattr(result, "candidates", None)
        )
        if not cands:
            return False
        for cand in cands:
            try:
                content = (
                    cand.get("content")
                    if isinstance(cand, dict)
                    else getattr(cand, "content", None)
                )
                if not content:
                    continue
                parts = (
                    content.get("parts")
                    if isinstance(content, dict)
                    else getattr(content, "parts", None)
                ) or []
                for part in parts:
                    if isinstance(part, dict):
                        inline = part.get("inline_data") or part.get("inlineData")
                    else:
                        inline = (
                            getattr(part, "inline_data", None)
                            or getattr(part, "inlineData", None)
                        )
                    if not inline:
                        continue
                    if isinstance(inline, dict):
                        mime = inline.get("mime_type") or inline.get("mimeType") or ""
                    else:
                        mime = (
                            getattr(inline, "mime_type", None)
                            or getattr(inline, "mimeType", None)
                            or ""
                        )
                    if "audio" in str(mime).lower():
                        return True
            except Exception:
                continue
        return False
    except Exception:
        return False


def _google_tts_intent_if_wanted(provider: str, kwargs) -> dict | None:
    """Return ``{kind: audio_tts}`` when Google request wants audio out; else None."""
    try:
        if (provider or "").lower() not in ("google", "gemini"):
            return None
        if _wants_google_audio_out(kwargs):
            return {"kind": "audio_tts"}
    except Exception:
        return None
    return None


@fail_safe
def _stash_google_verbatim_usage(session, provider: str, order, result=None):
    """Stash the verbatim google-genai ``usage_metadata`` for the Mode A
    deferred log (G3-O1).

    The google instrumentor only emits flat prompt/completion attrs, so the
    OTel-synth ``usage.raw`` drops ``thoughts_token_count`` /
    ``cached_content_token_count`` — and Google's ``candidates_token_count``
    EXCLUDES thoughts, so thinking tokens go entirely unbilled without this.
    Mirrors the TTS stash below but is UNCONDITIONAL on audio evidence.
    Never overwrites an existing stash (the stream wrappers' ``usage_raw=``
    forward has precedence), and only stashes when the serialized dict carries
    a positive prompt or candidates count — so a zero/absent usage block can
    never clobber good synth counts, and LangChain results are structurally
    excluded (their usage_metadata is keyed input_tokens/output_tokens).
    """
    if session is None or result is None:
        return
    if (provider or "").lower() not in ("google", "gemini"):
        return
    if order is None:
        order = getattr(session, "_span_counter", 0)
    if not hasattr(session, "_pending_compositions"):
        session._pending_compositions = {}
    comp_key = f"{session.trace_id}:{order}"
    # Peek only — the slot is created at write time below, so a guard-rejected
    # call leaves no new empty slot behind.
    existing = session._pending_compositions.get(comp_key)
    if isinstance(existing, dict) and existing.get("usage_raw"):
        return
    try:
        import json as _json
        raw = _extract_raw_usage("google", result)
        if raw is None:
            return
        serialized = _json.loads(_json.dumps(
            _as_dict(raw) if not isinstance(raw, dict) else raw,
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
        ))
        if not (isinstance(serialized, dict) and serialized):
            return
        if (int(serialized.get("prompt_token_count") or 0) > 0
                or int(serialized.get("candidates_token_count") or 0) > 0):
            session._pending_compositions.setdefault(comp_key, {})["usage_raw"] = serialized
    except Exception:
        pass


@fail_safe
def _stash_google_tts_hints(session, provider: str, order, kwargs=None, result=None):
    """Stash operation=audio_tts (+ optional raw usage) for Mode A deferred log.

    Mode A builds the deferred payload without seeing the full response object,
    so reclass + candidatesTokensDetails must be merged at flush from this
    stash. Shape stays ``google_genai`` — never switch to google_tts.
    """
    if session is None:
        return
    if (provider or "").lower() not in ("google", "gemini"):
        return
    wants = bool(kwargs is not None and _wants_google_audio_out(kwargs))
    is_out = bool(result is not None and _is_google_audio_output(result))
    if not (wants or is_out):
        return
    if order is None:
        order = getattr(session, "_span_counter", 0)
    if not hasattr(session, "_pending_compositions"):
        session._pending_compositions = {}
    comp_key = f"{session.trace_id}:{order}"
    slot = session._pending_compositions.setdefault(comp_key, {})
    slot["operation"] = "audio_tts"
    # Forward real usageMetadata so mapGoogleGenAI can split audio_output_tokens
    # (Mode A OTel synth drops candidates_tokens_details). Fail-open serialize.
    if result is not None:
        try:
            import json as _json
            raw = _extract_raw_usage("google", result)
            if raw is not None:
                serialized = _json.loads(_json.dumps(
                    _as_dict(raw) if not isinstance(raw, dict) else raw,
                    default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
                ))
                if isinstance(serialized, dict) and serialized:
                    slot["usage_raw"] = serialized
        except Exception:
            pass


@fail_safe
def _capture_response_composition(provider: str, response, service_tier: str = None,
                                  order: int = None, usage_raw=None):
    """Extract response composition and store on session.

    ``service_tier`` lets streaming wrappers pass the tier harvested from the
    chunks (the synthetic end-of-stream response doesn't carry it); when absent
    it is read off the response itself. Forwarded as ``usage.tier`` so
    batch/flex/priority pricing is applied server-side.

    ``order`` lets an interleave-safe caller (the LangChain stream guards) pass
    the LLM-span order it snapshotted at prompt-capture time, so the response
    fingerprint keys onto this stream's span even if another interleaved stream
    has since clobbered the shared ``_mode_a_prompt_order`` mailbox.
    When ``order is None`` (all non-stream callers) the legacy mailbox path is
    used, byte-for-byte unchanged.

    ``usage_raw`` (G3-14-1) lets the Mode A OpenAI-wire stream wrappers pass
    the already-serialized verbatim final-chunk usage; it is stashed under the
    same ``comp_key`` (required for concurrency attribution) and merged into
    ``payload["usage"]["raw"]`` by ``_flush_deferred_spans``. Callers passing
    nothing are unaffected.
    """
    session = get_current_session()
    # Always read-and-clear the one-slot mailbox so the lc-stream path (which
    # passes an explicit ``order``) never leaves a stale last-writer order
    # lingering for the next capture. Only the use of the mailbox value is
    # gated: an explicit ``order`` wins; otherwise fall back to the stashed
    # mailbox order, then to `_span_counter - 1` (non-langchain paths).
    mailbox = getattr(session, '_mode_a_prompt_order', None)
    session._mode_a_prompt_order = None
    if order is None:
        order = mailbox if mailbox is not None else (session._span_counter - 1)
    comp_key = f"{session.trace_id}:{order}"
    tier = service_tier or _extract_service_tier(response)
    if tier:
        if not hasattr(session, '_pending_compositions'):
            session._pending_compositions = {}
        session._pending_compositions.setdefault(comp_key, {})["service_tier"] = tier
    # G3-O1: the stream wrappers may call with response=None (parts-less turn
    # whose terminal-chunk usage still needs forwarding) — no response object
    # means no composition, never a fabricated complete_response entry.
    comp = build_response_composition(provider, response) if response is not None else None
    if comp:
        if not hasattr(session, '_pending_compositions'):
            session._pending_compositions = {}
        session._pending_compositions.setdefault(comp_key, {})["response"] = comp
    # G3-14-1: verbatim stream-usage forward (openai wire only; the wrappers
    # gate + serialize). Streamed spans never get reasoning-token attrs from
    # the instrumentor, so the OTel-synth usage.raw would silently drop
    # completion_tokens_details.reasoning_tokens without this.
    if usage_raw:
        if not hasattr(session, '_pending_compositions'):
            session._pending_compositions = {}
        session._pending_compositions.setdefault(comp_key, {})["usage_raw"] = usage_raw
    # G3-O1: forward the verbatim google usage_metadata unconditionally (the
    # OTel synth drops thoughts/cached counts). Skips when the stream
    # wrappers' ``usage_raw=`` stash above already filled the slot.
    _stash_google_verbatim_usage(session, provider, order, result=response)
    # Gemini TTS / native-audio: response evidence (or earlier request stash)
    # reclass operation + forward real usageMetadata for audio_output_tokens.
    _stash_google_tts_hints(session, provider, order, result=response)
    # Stash the response's tool-call ids (ordered, id+name only) so the manual
    # @tp.tool / tool_span path can auto-correlate them by name. REPLACE on every
    # capture (even with []) so a no-tool response clears stale ids from a prior
    # agent-loop iteration. Best-effort; both calls are independently fail-open.
    try:
        set_pending_tool_calls(extract_pending_tool_calls(provider, response))
    except Exception:
        pass


@fail_safe
def _flush_deferred_spans(session, *, obs_key=_OBS_KEY_CURRENT):
    """Flush any spans that were deferred by the telemetry processor.

    ``obs_key`` keys the observations drain to the flushing call (wrappers
    capture it right after their check and thread it here — a flush can run
    after ANOTHER call's check has overwritten the contextvar, e.g. a stream
    drained late). The sentinel default keeps legacy callers on the live
    contextvar."""
    if not hasattr(session, '_deferred_spans') or not session._deferred_spans:
        return

    tp = get_client()
    if not tp:
        return

    # Drain v2 extras ONCE per call attempt. Attach them only to the FIRST
    # deferred span (typically there's only one anyway); subsequent spans in
    # the same workflow shouldn't double-audit the same local decision.
    _flush_obs_key = _resolve_obs_key(obs_key)
    # CLAIMED up front, not read-then-cleared-at-the-end: the loop below makes
    # blocking HTTP calls, and the old trailing `session._local_decision = None`
    # destroyed whatever a CONCURRENT call stashed during that window. The
    # claim is scoped to this call's key, so a sibling's decision is untouched.
    pending_local_decision = _claim_local_decision(session, _flush_obs_key)
    pending_call_outcome = getattr(session, "_call_outcome", None)
    pending_latency = getattr(session, "_latency_metrics", None)
    pending_observations = _state.drain_observations(_flush_obs_key)
    extras_attached = False

    for payload in session._deferred_spans:
        # Merge any newly captured composition into the payload
        span_obj = payload.get("span", {})
        trace_id = span_obj.get("trace_id")
        span_order = span_obj.get("span_order")
        comp_key = f"{trace_id}:{span_order}"

        if hasattr(session, '_pending_compositions') and comp_key in session._pending_compositions:
            comp_data = session._pending_compositions.pop(comp_key, {})
            # The enforcer captures composition from the real request/response
            # objects — it is authoritative over the span-attribute fallback
            # that on_end placed in the payload (the fallback cannot see
            # tool_use blocks, so it would mislabel them as plain assistant text).
            if comp_data.get("prompt"):
                payload["prompt_composition"] = comp_data["prompt"]
            if comp_data.get("response"):
                payload["response_composition"] = comp_data["response"]
            # Provider-reported service tier stashed by the post-call capture
            # (after on_end built this payload) → usage.tier so
            # batch/flex/priority pricing is applied server-side.
            if comp_data.get("service_tier") and isinstance(payload.get("usage"), dict):
                payload["usage"]["tier"] = comp_data["service_tier"]
            # Gemini TTS reclass: Mode A deferred payloads default to
            # operation=chat; inject audio_tts when request/response evidence
            # stashed it. Also replace OTel-synth usage.raw with real
            # usageMetadata so candidatesTokensDetails → audio_output_tokens.
            if comp_data.get("operation"):
                payload["operation"] = comp_data["operation"]
            if comp_data.get("usage_raw") and isinstance(payload.get("usage"), dict):
                payload["usage"]["raw"] = comp_data["usage_raw"]
            # G3-O1: LangChain reasoning merge — the LC instrumentor never
            # emits gen_ai.usage.reasoning_tokens, so the Mode-A synth raw
            # lacks it. Inject the stashed count under the shape's native key.
            # Fully guarded: any failure leaves the payload untouched.
            try:
                _lc_r = int(comp_data.get("lc_reasoning_tokens") or 0)
                _usage = payload.get("usage")
                if _lc_r > 0 and isinstance(_usage, dict):
                    _raw = _usage.get("raw")
                    _shape = _usage.get("shape")
                    if isinstance(_raw, dict):
                        _out = int(_raw.get("completion_tokens") or 0)
                        if _shape in ("openai_chat", "openai_compatible_chat"):
                            # Subset semantics (text = completion − reasoning);
                            # clamp so reasoning never exceeds output. Never
                            # override a nonzero value already present.
                            _det = _raw.get("completion_tokens_details")
                            _has = (isinstance(_det, dict)
                                    and int(_det.get("reasoning_tokens") or 0) > 0)
                            _lc_r = min(_lc_r, _out)
                            if _lc_r > 0 and not _has:
                                _raw.setdefault(
                                    "completion_tokens_details", {}
                                )["reasoning_tokens"] = _lc_r
                        elif _shape == "google_genai":
                            # LC's output_tokens is thoughts-INCLUSIVE while
                            # the google mapper reads candidates_token_count
                            # as EXCLUSIVE-of-thoughts (additive) — emit the
                            # split so thoughts bill at the reasoning rate
                            # without changing total output. Skip when the raw
                            # already carries real google usage keys.
                            _has = ("thoughts_token_count" in _raw
                                    or "thoughtsTokenCount" in _raw
                                    or "candidates_token_count" in _raw
                                    or "candidatesTokenCount" in _raw)
                            _lc_r = min(_lc_r, _out)
                            if _lc_r > 0 and _out > 0 and not _has:
                                _raw["thoughts_token_count"] = _lc_r
                                _raw["candidates_token_count"] = max(0, _out - _lc_r)
            except Exception:
                pass

        if not extras_attached:
            if pending_local_decision:
                payload["local_decision"] = pending_local_decision
            if pending_observations:
                payload["observations"] = pending_observations
            if pending_call_outcome:
                payload["call_outcome"] = pending_call_outcome
            if pending_latency:
                payload["latency"] = pending_latency
            extras_attached = True

        tp.log_sync(**payload)

    session._deferred_spans.clear()
    # No local_decision clear here — it was claimed (removed) above.
    session._call_outcome = None
    session._latency_metrics = None


# ── Serving-provider identification (host → provider) ────────────────────────
# Mirrors the server-side host→provider identity tables (HOST_EXACT / HOST_PATTERNS)
# (kept inline — SDKs do not read the shared test-only JSON). Used so the
# REROUTE cross-provider guard compares the *serving* host, not just the
# client SDK module.
_HOST_EXACT = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "google",
    "api.mistral.ai": "mistral",
    "codestral.mistral.ai": "mistral",
    "api.x.ai": "xai",
    "api.deepseek.com": "deepseek",
    "api.moonshot.ai": "moonshot",
    "api.moonshot.cn": "moonshot",
    "api.minimax.io": "minimax",
    "api.minimaxi.com": "minimax",
    "api.perplexity.ai": "perplexity",
    "api.cohere.com": "cohere",
    "api.cohere.ai": "cohere",
    "open.bigmodel.cn": "zhipu",
    "api.z.ai": "zhipu",
    "ai-gateway.vercel.sh": "vercel-gateway",
    "openrouter.ai": "openrouter",
    "api.together.xyz": "together",
    "api.together.ai": "together",
    "api.fireworks.ai": "fireworks",
    "api.deepinfra.com": "deepinfra",
    "api.novita.ai": "novita",
    "api.groq.com": "groq",
    "api.cerebras.ai": "cerebras",
    "api.studio.nebius.com": "nebius",
    "api.studio.nebius.ai": "nebius",
}

_HOST_PATTERNS = (
    (re.compile(r"\.openai\.azure\.com$", re.I), "azure-openai"),
    (re.compile(r"\.services\.ai\.azure\.com$", re.I), "azure-ai"),
    (re.compile(r"\.inference\.ai\.azure\.com$", re.I), "azure-ai"),
    (re.compile(r"^bedrock-runtime\..*\.amazonaws\.com$", re.I), "bedrock"),
    (re.compile(r"aiplatform\.googleapis\.com$", re.I), "vertex-ai"),
    (re.compile(r"^(localhost|127\.0\.0\.1|0\.0\.0\.0)$", re.I), "self_hosted"),
)


def _extract_host(api_base: str) -> str:
    """Extract a lowercase hostname from a base URL. Fail-safe — "" on error."""
    if not api_base or not isinstance(api_base, str):
        return ""
    try:
        with_scheme = api_base if "://" in api_base else f"http://{api_base}"
        parts = _urlsplit(with_scheme)
        return (parts.hostname or "").lower()
    except Exception:
        try:
            h = re.sub(r"^[a-z]+:\/\/", "", api_base, flags=re.I)
            h = h.split("/")[0].split("?")[0]
            if "@" in h:
                h = h.rsplit("@", 1)[-1]
            return h.split(":")[0].lower()
        except Exception:
            return ""


def _match_host_to_provider(host: str):
    """Map a hostname to a canonical serving-provider slug, or None if unknown."""
    if not host:
        return None
    exact = _HOST_EXACT.get(host)
    if exact:
        return exact
    for pattern, provider in _HOST_PATTERNS:
        if pattern.search(host):
            return provider
    return None


def resolve_serving_from_base_url(api_base: str) -> dict:
    """
    Resolve serving provider from a raw base-URL string.

    Returns one of:
      {"kind": "absent"}
      {"kind": "recognized", "provider": "<slug>"}
      {"kind": "unrecognized"}
    """
    try:
        if not api_base or not isinstance(api_base, str) or not api_base.strip():
            return {"kind": "absent"}
        host = _extract_host(api_base)
        if not host:
            return {"kind": "unrecognized"}
        provider = _match_host_to_provider(host)
        if provider:
            return {"kind": "recognized", "provider": provider}
        return {"kind": "unrecognized"}
    except Exception:
        return {"kind": "absent"}


def _resolve_serving_provider(module_provider: str, args) -> dict:
    """
    Resolve serving identity for pre-flight check + REROUTE.

    Returns ``{"provider": str, "serving_unverified": bool}``:
    - recognized host → remapped provider; serving_unverified=False
    - absent base_url → module provider; serving_unverified=False
    - unrecognized custom host → **module provider kept** for matchConditions /
      groupBy /check payload, serving_unverified=True so REROUTE refuses

    Fail-safe: any error → module provider, verified.

    Serving provider and wire/parse key are **two axes**. This function
    returns the billing/rules vendor only. Never pass its result into stream
    accumulators, Mode-A composition, or message parsers — those must use
    ``_wire_parse_key(module_provider)`` (the client library's wire shape), or
    OpenAI-compatible gateways lose composition / tool ids (and Node loses
    streamed rows entirely).
    """
    if not args:
        return {"provider": module_provider, "serving_unverified": False}
    try:
        base_url = _extract_base_url(args)
        if not base_url:
            client = getattr(args[0], "_client", None) if args else None
            base_url = str(getattr(client, "base_url", "") or "") if client is not None else ""
            if not base_url:
                base_url = str(getattr(args[0], "base_url", "") or "")
        resolved = resolve_serving_from_base_url(base_url)
        kind = resolved.get("kind")
        if kind == "recognized":
            return {"provider": resolved["provider"], "serving_unverified": False}
        if kind == "unrecognized":
            return {"provider": module_provider, "serving_unverified": True}
    except Exception:
        pass
    return {"provider": module_provider, "serving_unverified": False}


def _wire_parse_key(module_provider: str) -> str:
    """
    Wire/parse key for stream accumulation and composition.

    Equals the **module** (client-library) provider slug — never the
    host-remapped serving vendor. An OpenAI SDK client always produces
    OpenAI-shaped bytes regardless of ``base_url`` (api.minimax.io, api.x.ai,
    …). Using the serving slug here re-opens (empty composition / wrong
    protobuf parsers / no stream accumulator).
    """
    return module_provider


def _effective_provider(provider: str, args) -> str:
    """
    Resolve the provider actually being billed / serving the call.
    See ``_resolve_serving_provider`` for unrecognized-host semantics (returns
    the module provider — never an empty sentinel that would blank match/groupBy).
    """
    return _resolve_serving_provider(provider, args)["provider"]


@fail_safe
def _stash_provider_override(provider: str, session, order: int = None):
    """
    Record a provider override for the next LLM span on this session.

    The telemetry SpanProcessor derives the provider from the OpenLLMetry
    ``gen_ai.system`` attribute, which only knows about the physical SDK
    (``openai``). For OpenRouter-routed calls the enforcer stashes the real
    provider here, keyed by span order, and ``on_end`` applies it.

    ``order`` is the wrapper's reserved LLM-span order; when None, fall back to
    peeking _span_counter (the enforcer runs BEFORE on_start, so it still holds
    the value on_start will consume via next_span_order()).
    """
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    if order is None:
        order = session._span_counter
    comp_key = f"{session.trace_id}:{order}"
    session._pending_compositions.setdefault(comp_key, {})["provider"] = provider


@fail_safe
def _stash_gateway_request_model(session, kwargs, order: int = None):
    """Gateway-routed SDK call (e.g. the OpenAI SDK pointed at OpenRouter):
    record the FULL request model slug and its vendor head for the next LLM
    span on this session.

    Traceloop's openai instrumentor strips the vendor prefix from
    gen_ai.request.model ("openai/gpt-4.1-nano" -> "gpt-4.1-nano") before
    telemetry on_end sees it, so the instrumented-path row lost the gateway
    identity (model stripped, original_provider=""). Stash the customer's
    verbatim slug ("model") plus its vendor head ("original_provider" — the
    segment before "/", "" when un-prefixed) on the same pending-composition
    key _stash_provider_override uses; on_end forwards them. Mirrors the Node
    telemetry.ts pending-stash pattern. Only called on the detected-gateway
    path — non-gateway spans never carry these keys.
    """
    model = kwargs.get("model") if isinstance(kwargs, dict) else None
    if not isinstance(model, str) or not model:
        return
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    # Same key as _stash_provider_override: the wrapper's reserved order, or a
    # peek of _span_counter (the enforcer runs BEFORE on_start consumes it).
    if order is None:
        order = session._span_counter
    comp_key = f"{session.trace_id}:{order}"
    entry = session._pending_compositions.setdefault(comp_key, {})
    entry["model"] = model
    entry["original_provider"] = (
        model.split("/", 1)[0].strip().lower() if "/" in model else ""
    )


def _extract_base_url(args) -> str:
    """Best-effort read of the bound client's base URL as host+path (no userinfo,
    no query). Strips embedded credentials so only host metadata leaves the
    process. Mirrors ``_effective_provider``'s client lookup; the service maps
    the host to a serving provider (e.g. ``api.minimax.io`` -> ``minimax``).
    Fail-safe — returns "" on any error; api_base is advisory.
    """
    try:
        client = getattr(args[0], "_client", None) if args else None
        base_url = str(getattr(client, "base_url", "") or "")
        if not base_url:
            return ""
        parts = _urlsplit(base_url)
        host = parts.hostname or ""
        if not host:
            # Not a parseable URL (e.g. bare host, no scheme) — best-effort:
            # drop fragment + query, and strip any userinfo credentials so a
            # `user:pass@host/path` never leaks into telemetry. Userinfo lives in
            # the authority only: the last '@' that occurs BEFORE the first '/'.
            # An '@' inside the path (e.g. `host/a@b`) is NOT credentials.
            cleaned = base_url.split("#", 1)[0].split("?", 1)[0]
            slash = cleaned.find("/")
            authority_end = len(cleaned) if slash == -1 else slash
            at = cleaned.rfind("@", 0, authority_end)
            if at != -1:
                cleaned = cleaned[at + 1:]
            return cleaned
        if parts.port:
            host = f"{host}:{parts.port}"
        # scheme + host (incl. port, EXCL. user:pass) + path — drops credentials,
        # query, and fragment. Only host metadata leaves the process.
        return f"{parts.scheme}://{host}{parts.path}" if parts.scheme else f"{host}{parts.path}"
    except Exception:
        return ""


@fail_safe
def _stash_api_base(session, base_url: str, order: int = None):
    """Stash the serving endpoint for the next LLM span (read by on_end).

    ``order`` is the wrapper's reserved LLM-span order; None → peek _span_counter.
    """
    if not base_url:
        return
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    if order is None:
        order = session._span_counter
    comp_key = f"{session.trace_id}:{order}"
    session._pending_compositions.setdefault(comp_key, {})["api_base"] = base_url


# ═══════════════════════════════════════════════════════════════════
# Manual telemetry path — for SDKs with no OpenLLMetry instrumentor
# (the native OpenRouter SDK and the Cerebras SDK). The wrapper extracts
# token usage from the response itself and logs it directly via
# tp.log_sync(). Both SDKs return OpenAI-shaped usage objects.
# ═══════════════════════════════════════════════════════════════════

def _is_stream(obj) -> bool:
    """True if obj is a streaming response (sync or async iterator)."""
    return obj is not None and (hasattr(obj, "__next__") or hasattr(obj, "__anext__"))


@fail_safe
def _capture_composition_at(provider: str, payload, order: int, is_response: bool,
                            usage_shape: str = None,
                            operation: str = None):
    """Capture prompt or response composition at an explicit span order.

    ``usage_shape`` (response side) is the authoritative usage-shape
    enum — passed in by modality wrappers so the composition builder
    can short-circuit binary/image/video bodies into a non-text entry without
    risking a parser reading ``.text`` off a binary response. For embedding
    shapes ``build_response_composition`` returns [] (response is a vector).

    ``operation`` (prompt side) selects the prompt parser: "embedding"
    routes through ``_parse_embedding_input`` which emits role="input"
    entries; anything else uses the existing chat-message tier ladder.
    """
    session = get_current_session()
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    comp_key = f"{session.trace_id}:{order}"
    if is_response:
        comp = build_response_composition(provider, payload, usage_shape=usage_shape)
        key = "response"
    else:
        comp = build_prompt_composition(provider, payload, operation=operation)
        key = "prompt"
    if comp:
        session._pending_compositions.setdefault(comp_key, {})[key] = comp
    if is_response:
        # Stash the response's tool-call ids (ordered, id+name only) so the
        # manual @tp.tool / tool_span path can auto-correlate them by name. This
        # is the manual-telemetry capture site (cohere, xai, cerebras, openrouter,
        # together, huggingface, mistral_rest…); the OpenLLMetry-instrumented path
        # (openai/anthropic SDKs) stashes in _capture_response_composition. REPLACE
        # on every response capture (even []) so a no-tool response clears stale
        # ids from a prior loop iteration. Best-effort; independently fail-open.
        try:
            set_pending_tool_calls(extract_pending_tool_calls(provider, payload))
        except Exception:
            pass


def _build_xai_pseudo_kwargs(args) -> dict:
    """Build a synthetic kwargs dict for xai-sdk calls.

    Two surfaces:
      * Chat: ``Chat.sample()/.stream()`` accumulates messages on the Chat
        instance via .append(); the call methods take no message kwargs.
        Pull ``messages`` (a repeated protobuf field) and ``model`` from
        ``self._proto`` so the existing composition pipeline + ``_log_manual``
        model fallback work without further plumbing.
      * Image: ``image.Client.sample(prompt, model, ...)`` and
        ``sample_batch(prompt, model, n, ...)`` take positional prompt + model.
        Surface them as ``{prompt, model, n}`` so the modality-aware
        composition helper picks the prompt up as a text entry.
    """
    if not args:
        return {}
    self_obj = args[0]
    # Image client — no _proto; prompt + model are positional.
    self_cls = type(self_obj).__name__ if self_obj is not None else ""
    self_mod = type(self_obj).__module__ if self_obj is not None else ""
    if self_cls == "Client" and "image" in self_mod:
        prompt = args[1] if len(args) > 1 and isinstance(args[1], str) else ""
        model = args[2] if len(args) > 2 and isinstance(args[2], str) else ""
        n = args[3] if len(args) > 3 and isinstance(args[3], int) else None
        out = {"prompt": prompt or None, "model": model or None}
        if n is not None:
            out["n"] = n
        return out
    proto = getattr(self_obj, "_proto", None)
    if proto is None:
        return {}
    try:
        messages = list(getattr(proto, "messages", []) or [])
    except Exception:
        messages = []
    model = ""
    try:
        model = str(getattr(proto, "model", "") or "")
    except Exception:
        model = ""
    return {"messages": messages, "model": model or None}


def _xai_model_hint(args):
    """Xai-sdk keeps the model on ``self._proto`` (chat) or a positional
    arg (image), never in kwargs — so the pre-flight used to evaluate with an
    empty model. Surface it as a hint for matching + audit only; it must not
    reach kwargs (see _run_sync_check). Total: never raises."""
    try:
        model = (_build_xai_pseudo_kwargs(args) or {}).get("model")
        return model if isinstance(model, str) and model else None
    except Exception:
        return None


def _xai_usage_response(obj):
    """If `obj` is an xai-sdk stream tuple (Response, Chunk), return the first
    element (the Response carrying .usage on the final iteration). Otherwise
    return obj unchanged."""
    if isinstance(obj, tuple) and len(obj) == 2:
        return obj[0]
    return obj


def _extract_openai_compatible_usage(obj):
    """
    Extract (model, input_tokens, output_tokens, cached_tokens) from an
    OpenAI-shaped response or streaming chunk — used by the native OpenRouter
    SDK and the Cerebras SDK, both of which return OpenAI-compatible usage.

    Also handles the OpenAI Responses API shape (``input_tokens`` /
    ``output_tokens`` instead of ``prompt_tokens`` / ``completion_tokens``;
    ``input_tokens_details.cached_tokens`` /
    ``output_tokens_details.reasoning_tokens``). When a Responses
    ``response.completed`` stream event is passed, unwrap once via ``.response``
    so we read usage from the inner Response.

    ``prompt_tokens`` already includes the cached prompt tokens, so subtract
    them to avoid double-counting cost. ``completion_tokens`` already includes
    ``completion_tokens_details.reasoning_tokens``, so they must not be added
    again.

    Mistral wraps each streaming chunk as ``CompletionEvent { data: CompletionChunk }``
    — when the caller hands us such a wrapped chunk, unwrap once so we read
    usage/model from the inner CompletionChunk. Non-streaming Mistral responses
    are OpenAI-shaped at the top level (no `.data` wrapper).

    xai-sdk streams yield `(Response, Chunk)` tuples; unwrap to the Response
    side, which carries the OpenAI-shaped `.usage` (only filled on the final
    iteration). Non-streaming xai-sdk returns a Response directly.
    """
    # xai-sdk stream tuple — read usage from the Response part.
    obj = _xai_usage_response(obj)
    # Responses stream `response.completed` event wraps the Response under
    # `.response`; non-streaming Response itself has `.usage` directly.
    if getattr(obj, "type", None) == "response.completed":
        inner = getattr(obj, "response", None)
        if inner is not None:
            obj = inner
    inner = getattr(obj, "data", None)
    if inner is not None and getattr(inner, "usage", None) is not None:
        obj = inner
    model = str(getattr(obj, "model", None) or "unknown")
    usage = getattr(obj, "usage", None)
    if not usage:
        # Groq streaming final chunk nests usage under `chunk.x_groq.usage`
        # (top-level `.usage` absent/None). `x_groq` is a groq-only field, so
        # this fallback never affects any other OpenAI-compatible provider.
        xg = getattr(obj, "x_groq", None)
        if xg is not None:
            usage = getattr(xg, "usage", None)
    if not usage:
        return model, 0, 0, 0

    # Responses API: usage.input_tokens / usage.output_tokens.
    resp_input = getattr(usage, "input_tokens", None)
    resp_output = getattr(usage, "output_tokens", None)
    if resp_input is not None or resp_output is not None:
        input_t = int(resp_input or 0)
        output_t = int(resp_output or 0)
        cached = 0
        itd = getattr(usage, "input_tokens_details", None)
        if itd:
            cached = int(getattr(itd, "cached_tokens", 0) or 0)
        # Mirror Chat Completions accounting: input_tokens reported by the
        # Responses API already INCLUDES cached_tokens — subtract to avoid
        # double-counting. Reasoning tokens (output_tokens_details.reasoning_tokens)
        # are already part of output_tokens, so they are intentionally not read.
        input_tokens = max(0, input_t - cached)
        return model, input_tokens, output_t, cached

    # Chat Completions API shape.
    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    cached = 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd:
        cached = int(getattr(ptd, "cached_tokens", 0) or 0)
    input_tokens = max(0, prompt - cached)
    output_tokens = completion
    return model, input_tokens, output_tokens, cached


_USAGE_SHAPE_BY_PROVIDER = {
    "anthropic": "anthropic_messages",
    "openai": "openai_chat",
    "openai_responses": "openai_responses",
    "google": "google_genai",
    "gemini": "google_genai",
    "bedrock": "bedrock_converse",
    "cohere": "cohere_chat",
    "mistral": "mistral_chat",
    "xai": "xai_chat",
    "together": "together_chat",
    "cerebras": "cerebras_chat",
    "groq": "groq_chat",
    "huggingface": "huggingface_chat",
    "openrouter": "openrouter_routed",
    "litellm": "openai_compatible_chat",
}

# Per-provider default embedding usage shape — used when an embedding wrapper's
# registry entry omits an explicit ``shape`` override.
_EMBEDDING_SHAPE_BY_PROVIDER = {
    "openai": "openai_embeddings",
    "google": "google_genai_embeddings",
    "gemini": "google_genai_embeddings",
    "cohere": "cohere_embed",
    "mistral": "mistral_embed",
    "voyage": "voyage_embed",
    "huggingface": "huggingface_embed",
    "together": "together_embed",
    "litellm": "openai_embeddings",  # LiteLLM forwards OpenAI-shaped usage.
}


def _resolve_usage_shape(provider: str) -> str:
    """Pick the usage-mapper shape for a provider."""
    return _USAGE_SHAPE_BY_PROVIDER.get((provider or "").lower(), "openai_compatible_chat")


def _resolve_embedding_shape(provider: str) -> str:
    """Pick the usage-mapper shape for an embedding call."""
    return _EMBEDDING_SHAPE_BY_PROVIDER.get((provider or "").lower(), "openai_embeddings")


def _extract_embedding_usage(result, provider: str):
    """Extract (model, input_tokens, raw_usage_dict) from an embeddings response.

    Per-provider response shapes are handled inline below (OpenAI/Together/
    LiteLLM, Google GenAI, Cohere, Mistral, Voyage, ...). Fail-safe: any
    extraction failure returns ``("unknown", 0, None)`` so the SDK still
    emits a log row.
    """
    if result is None:
        return "unknown", 0, None
    p = (provider or "").lower()
    try:
        # OpenAI / Together / LiteLLM — usage.prompt_tokens, usage.total_tokens.
        if p in ("openai", "together", "litellm"):
            usage = _attr(result, "usage") or {}
            model = str(_attr(result, "model") or "unknown")
            prompt = int(_attr(usage, "prompt_tokens") or _attr(usage, "total_tokens") or 0)
            raw = {"prompt_tokens": prompt,
                   "total_tokens": int(_attr(usage, "total_tokens") or prompt)}
            return model, prompt, raw

        # Google GenAI — `EmbedContentResponse` has NO usage_metadata (that's
        # only on generate_content). Three possible token sources, in priority:
        # 1. Vertex per-embedding stats: result.embeddings[i].statistics.token_count
        # 2. Vertex char-billed: result.metadata.billable_character_count → chars/4
        # 3. mldev/Gemini API: response carries no usage at all — caller
        # approximates from the request kwargs (handled in _log_manual,
        # mirroring the HuggingFace approach).
        if p in ("google", "gemini"):
            # 1. Vertex per-embedding stats — most precise when present.
            embeddings = _attr(result, "embeddings") or []
            if isinstance(embeddings, (list, tuple)):
                vertex_tokens = 0
                for emb in embeddings:
                    stats = _attr(emb, "statistics")
                    if stats is not None:
                        vertex_tokens += int(_attr(stats, "token_count") or 0)
                if vertex_tokens > 0:
                    return "unknown", vertex_tokens, {
                        "prompt_token_count": vertex_tokens,
                        "total_token_count": vertex_tokens,
                    }
            # 2. Vertex billable characters — approximate tokens at chars / 4.
            metadata = _attr(result, "metadata")
            if metadata is not None:
                chars = int(_attr(metadata, "billable_character_count") or 0)
                if chars > 0:
                    approx = max(1, chars // 4)
                    return "unknown", approx, {
                        "billable_character_count": chars,
                        "approx_input_tokens": approx,
                        "approximated": True,
                    }
            # 3. mldev/Gemini API — no response-side usage. Signal the caller
            # by returning None for raw_usage so _log_manual approximates
            # from kwargs (parallel to the HuggingFace path below).
            return "unknown", 0, None

        # Cohere v2 — response.meta.billed_units.{input_tokens, images}.
        if p == "cohere":
            meta = _attr(result, "meta") or {}
            billed = _attr(meta, "billed_units") or {}
            input_tokens = int(_attr(billed, "input_tokens") or 0)
            images = int(_attr(billed, "images") or 0)
            raw = {"meta": {"billed_units": {"input_tokens": input_tokens, "images": images}}}
            return "unknown", input_tokens, raw

        # Mistral — usage.prompt_tokens, usage.total_tokens.
        if p == "mistral":
            usage = _attr(result, "usage") or {}
            prompt = int(_attr(usage, "prompt_tokens") or _attr(usage, "total_tokens") or 0)
            raw = {"prompt_tokens": prompt,
                   "total_tokens": int(_attr(usage, "total_tokens") or prompt)}
            return _attr(result, "model") or "unknown", prompt, raw

        # Voyage AI — the returned EmbeddingsObject / MultimodalEmbeddingsObject
        # expose token counts as TOP-LEVEL attributes (`total_tokens`), NOT under
        # `.usage` (the SDK rolls the raw API `usage.total_tokens` into the result
        # object's own `.total_tokens`). Read the top level, with `usage` fallback.
        if p == "voyage":
            total = int(
                _attr(result, "total_tokens")
                or _attr(_attr(result, "usage") or {}, "total_tokens")
                or 0
            )
            raw = {"total_tokens": total}
            return _attr(result, "model") or "unknown", total, raw

        # HuggingFace feature_extraction — returns a raw vector with no usage.
        # The wrapper supplies an approximate token count via raw.approx_input_tokens
        # by tokenizer fallback or len(input) heuristic before calling this helper.
        if p == "huggingface":
            return "unknown", 0, {"approx_input_tokens": 0}

        # Unknown provider — best-effort usage.
        usage = _attr(result, "usage") or {}
        prompt = int(_attr(usage, "prompt_tokens") or _attr(usage, "total_tokens") or 0)
        return str(_attr(result, "model") or "unknown"), prompt, {"prompt_tokens": prompt}
    except Exception:
        return "unknown", 0, None


def _approximate_bedrock_embedding_tokens(api_params: dict) -> int:
    """Best-effort token count when Bedrock returns no usable usage.

    Reuses _bedrock_embedding_kwargs to decode the request body once, then
    sums `max(1, len(t) // 4)` over the input strings — same heuristic as
    _approximate_hf_embedding_tokens. Returns 0 on any failure so the caller
    can fall through to "no approximation".

    Triggered by the AWS-documented gap where Bedrock's Cohere Embed route
    strips Cohere's native meta.billed_units field (see
    docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html).
    """
    try:
        decoded = _bedrock_embedding_kwargs(api_params or {})
        texts = decoded.get("texts") or decoded.get("inputs")
        if isinstance(texts, (list, tuple)):
            return sum(max(1, len(str(t)) // 4) for t in texts)
        single = decoded.get("input") or decoded.get("inputText")
        if isinstance(single, str):
            return max(1, len(single) // 4)
    except Exception:
        return 0
    return 0


def _approximate_hf_embedding_tokens(args, kwargs) -> int:
    """Best-effort token count for HuggingFace feature_extraction.

    `feature_extraction` accepts `text: str | List[str]`. We have no usage
    object on return, so approximate via len(text) / 4 — the canonical
    rule of thumb that produces an order-of-magnitude estimate.
    Approximated rows are flagged server-side and surfaced as estimates.
    """
    try:
        text = (kwargs or {}).get("text")
        if text is None and len(args) > 1:
            text = args[1]
        if isinstance(text, str):
            return max(1, len(text) // 4)
        if isinstance(text, (list, tuple)):
            total = 0
            for t in text:
                if isinstance(t, str):
                    total += max(1, len(t) // 4)
            return total
    except Exception:
        pass
    return 0


def _approx_chars_to_tokens(payload) -> int:
    """Sum character lengths in any Google embed_content `contents` shape
    and divide by 4. Recurses into Content objects with `.parts[i].text`
    and dict equivalents. Returns 0 on any failure (fail-safe)."""
    try:
        if payload is None:
            return 0
        if isinstance(payload, str):
            return max(1, len(payload) // 4)
        if isinstance(payload, (list, tuple)):
            total = 0
            for item in payload:
                total += _approx_chars_to_tokens(item)
            return total
        # Content object / dict — look for `parts` then `.text` on each part.
        parts = _attr(payload, "parts")
        if parts is not None:
            return _approx_chars_to_tokens(parts)
        # Single Part with .text
        text = _attr(payload, "text")
        if isinstance(text, str):
            return max(1, len(text) // 4)
    except Exception:
        pass
    return 0


def _approximate_google_embedding_tokens(args, kwargs) -> int:
    """Approximate input tokens for Google embed_content when the response
    carries no usage telemetry (mldev/Gemini API path). Walks the
    `contents` argument — accepts str / list[str] / Content / list[Content]
    — and sums character lengths / 4. Mirrors `_approximate_hf_embedding_tokens`.
    Fail-safe: bad input returns 0."""
    try:
        contents = (kwargs or {}).get("contents")
        if contents is None and args and len(args) > 1:
            contents = args[1]
        return _approx_chars_to_tokens(contents)
    except Exception:
        return 0


def _proto_usage_to_dict(usage):
    """Convert an xai-sdk protobuf usage message into a plain OpenAI-shaped dict.

    xai-sdk returns `usage` as a protobuf message (``xai.api.v1.usage_pb2``).
    The generic ``_as_dict`` + ``json.dumps`` path can't read protobuf fields —
    it serializes the message *class's* ``__dict__`` (DESCRIPTOR / __slots__ /
    ...) — so read the populated scalar fields here instead. Fail-safe: returns
    ``None`` on any error (never throws into customer code).
    """
    if usage is None:
        return None
    try:
        # Protobuf message — ListFields() yields only populated (non-default)
        # fields. Keep scalar ints/floats; skip bools and nested submessages.
        if hasattr(usage, "ListFields"):
            out = {}
            for field, value in usage.ListFields():
                if isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    out[getattr(field, "name", str(field))] = value
            if out:
                return out
        # Fallback: read the OpenAI-shaped scalars by name (covers proto stubs
        # that don't expose ListFields the way we expect).
        out = {}
        for name in ("prompt_tokens", "completion_tokens", "total_tokens",
                     "reasoning_tokens", "cached_prompt_text_tokens"):
            value = getattr(usage, name, None)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                out[name] = int(value)
        return out or None
    except Exception:
        return None


def _serialize_anthropic_raw_usage(usage):
    """Verbatim Anthropic usage object → plain JSON-safe dict, or None if it
    can't serialize. Mirrors the batch/non-stream manual paths' idiom
    (_as_dict + json round-trip with the __dict__/str default) so nested SDK
    objects (e.g. usage.cache_creation.ephemeral_5m_input_tokens) survive.

    Returns None — never {} or a non-dict — so callers can gate on a truthy
    dict: an empty/absent raw under the anthropic_messages shape maps to
    all-zeros server-side, which is worse than the openai_compatible
    fallback. Self-guarded: any failure degrades to None (fail-open) so the
    finalizer still logs via the fallback rather than losing the row."""
    if usage is None:
        return None
    try:
        import json as _json
        raw = _json.loads(_json.dumps(
            _as_dict(usage),
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
        ))
        return raw if isinstance(raw, dict) and raw else None
    except Exception:
        return None


def _extract_raw_usage(provider: str, result):
    """Return the verbatim provider usage object off `result` for forwarding."""
    if result is None:
        return None
    p = (provider or "").lower()
    try:
        if p in ("google", "gemini"):
            # Object path (typed SDK) + dict path (REST / SimpleNamespace-decoded
            # bodies after json.loads). Dict branch is required so Mode A TTS
            # stash can forward candidates_tokens_details for audio unit split.
            if isinstance(result, dict):
                return result.get("usage_metadata") or result.get("usageMetadata")
            return getattr(result, "usage_metadata", None) or getattr(result, "usageMetadata", None)
        if p == "bedrock":
            return _attr(result, "usage")
        if p == "xai":
            # xai-sdk usage is a protobuf message; streams yield (Response, Chunk)
            # tuples so unwrap first. Convert protobuf → plain dict; any other
            # shape (dict / pydantic) is returned unchanged so the existing
            # serialization path behaves exactly as before.
            obj = _xai_usage_response(result)
            usage = _attr(obj, "usage")
            if usage is None:
                return None
            if hasattr(usage, "ListFields"):
                return _proto_usage_to_dict(usage)
            return usage
        if p == "cohere":
            usage = _attr(result, "usage")
            if usage is None:
                delta = _attr(result, "delta")
                if delta is not None:
                    usage = _attr(delta, "usage")
            return usage
        if p == "openai_responses":
            # Streaming `response.completed` nests usage at .response.usage;
            # non-streaming Response carries .usage at the top level.
            # Part 2: do NOT inject image_output_count into chat-span
            # usage. Built-in image_generation_call items are billed on child
            # image spans via `_log_responses_image_children` (openai_images
            # shape + gpt-image-*). Chat usage stays text/cache tokens only.
            resp = _attr(result, "response")
            u = None
            if resp is not None:
                u = _attr(resp, "usage")
            if u is None:
                u = _attr(result, "usage")
            return u
        if p == "mistral":
            usage = _attr(result, "usage")
            if usage is None:
                data = _attr(result, "data")
                if data is not None:
                    usage = _attr(data, "usage")
            return usage
        if p == "groq":
            # Non-stream groq responses carry top-level OpenAI-shaped `.usage`;
            # streaming final chunks nest it under `.x_groq.usage`. Prefer the
            # top-level value, else unwrap the groq-only `x_groq` field.
            usage = _attr(result, "usage")
            if usage is None:
                usage = _attr(_attr(result, "x_groq"), "usage")
            return usage
        return _attr(result, "usage")
    except Exception:
        return None


def _attr(obj, name):
    """Best-effort field accessor — handles dict, object, and Pydantic shapes."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _extract_cohere_chat_usage(obj):
    """Extract (model, input_tokens, output_tokens, cached_tokens) from a Cohere
    v2 chat response OR a streamed message-end event.

    Cohere chat is captured manually (Mode C) rather than via the OpenLLMetry
    cohere instrumentor, which only supports `cohere <6` (no release covers
    cohere 6.x / 7.x). This is version-resilient across cohere 5.x / 6.x / 7.x —
    all use Cohere v2 `ClientV2`, with usage at:
      * non-stream: ``obj.usage``
      * stream: ``obj.delta.usage`` (the ``message-end`` event)
    Tokens come from ``usage.tokens.{input,output}_tokens`` (the actual token
    counts, reported as floats), falling back to ``usage.billed_units.*``. The
    chat response carries no model field, so the model is returned empty and the
    caller falls back to ``kwargs["model"]``. Fail-open: any shape mismatch
    returns zeros — never raises into the customer's call path."""
    inp = out = cached = 0
    try:
        usage = _attr(obj, "usage")
        if usage is None:
            usage = _attr(_attr(obj, "delta"), "usage")
        if usage is not None:
            src = _attr(usage, "tokens") or _attr(usage, "billed_units")
            if src is not None:
                inp = int(_attr(src, "input_tokens") or 0)
                out = int(_attr(src, "output_tokens") or 0)
            cached = int(_attr(usage, "cached_tokens") or 0)
    except Exception:
        return "", 0, 0, 0
    return "", inp, out, cached


def _resolve_litellm_original_provider(request_model, result) -> str:
    """Underlying provider slug for a LiteLLM-routed call, or "" when unknown.

    LiteLLM is a **framework**, not a provider — its rows log
    ``provider="litellm"`` and the server (which types "litellm" as a
    framework) honestly leaves ``original_provider`` blank. The real vendor IS
    knowable inside the wrapper, so resolve it here and ship it as an explicit
    hint (server-side, an explicit hint always wins).

    Resolution order, each step fully guarded:
      (a) ``result._hidden_params["custom_llm_provider"]`` — LiteLLM stamps the
          provider it actually dispatched to on the response / stream chunk;
      (b) ``litellm.get_llm_provider(request_model)[1]`` — the library's own
          resolver ("gpt-4o-mini" -> "openai", "gemini/gemini-2.5-flash" ->
          "gemini"). Accessed lazily off ``sys.modules`` (never imported by us)
          and it CAN raise (BadRequestError on an unknown model);
      (c) the vendor head before "/" in the request model;
      (d) "" — caller omits the hint, i.e. today's behavior.

    The slug is returned VERBATIM (e.g. "gemini", "vertex_ai"). Canonicalization
    ("gemini" -> "google") is the server's job; the SDK stays thin.
    """
    # (a) response / final stream chunk hidden params.
    try:
        hidden = getattr(result, "_hidden_params", None)
        if isinstance(hidden, dict):
            slug = hidden.get("custom_llm_provider")
            if isinstance(slug, str) and slug.strip():
                return slug.strip()
    except Exception:
        pass

    model = request_model if isinstance(request_model, str) else ""
    if not model:
        return ""

    # (b) LiteLLM's own resolver — lazy, never imported by us, may raise.
    try:
        _litellm = sys.modules.get("litellm")
        get_provider = getattr(_litellm, "get_llm_provider", None)
        if get_provider is not None:
            resolved = get_provider(model)
            if isinstance(resolved, (tuple, list)) and len(resolved) > 1:
                slug = resolved[1]
                if isinstance(slug, str) and slug.strip():
                    return slug.strip()
    except Exception:
        pass

    # (c) vendor prefix on the request model ("anthropic/claude-..." -> "anthropic").
    try:
        if "/" in model:
            head = model.split("/", 1)[0].strip().lower()
            if head:
                return head
    except Exception:
        pass

    # (d) unknown — leave the hint absent.
    return ""


@fail_safe
def _log_manual(provider: str, session, kwargs: dict, result, order: int,
                span_name, start_time,
                operation: str = "chat",
                shape_override: str = None,
                args: tuple = None,
                latency: dict = None,
                obs_key=_OBS_KEY_CURRENT,
                route_ctx=None):
    """Extract token usage from a manual-telemetry response and log it.

    ``operation`` selects the extraction shape: "embedding" routes through
    ``_extract_embedding_usage``; anything else uses the OpenAI-compatible
    chat path. ``shape_override`` lets the registry entry pin an explicit
    usage shape (e.g. "openai_embeddings") instead of provider-defaults.
    ``args`` is forwarded so HuggingFace's text-only `feature_extraction`
    surface (no usage object) can approximate input tokens from the input.
    ``route_ctx`` (LiteLLM seam only) applies to the request-model FALLBACK
    below — the success path already reads LiteLLM's bare response echo.
    """
    tp = get_client()
    if not tp:
        return

    if operation == "embedding":
        model, input_tokens, raw_usage = _extract_embedding_usage(result, provider)
        output_tokens = 0
        cached_tokens = 0
        # HuggingFace has no usage object — approximate from the input text.
        if (provider or "").lower() == "huggingface":
            approx = _approximate_hf_embedding_tokens(args or (), kwargs or {})
            input_tokens = approx
            raw_usage = {"approx_input_tokens": approx, "approximated": True}
        # Google mldev/Gemini API: embed_content returns no usage on the
        # response. The Vertex path (handled inside _extract_embedding_usage)
        # already filled tokens from per-embedding stats or
        # billable_character_count; if we still have 0 here, we're on mldev
        # and need to approximate from the request kwargs.
        elif (provider or "").lower() in ("google", "gemini") and input_tokens == 0:
            approx = _approximate_google_embedding_tokens(args or (), kwargs or {})
            if approx > 0:
                input_tokens = approx
                raw_usage = {"approx_input_tokens": approx, "approximated": True}
    elif (provider or "").lower() == "cohere":
        # Cohere v2 chat usage is NOT OpenAI-shaped (usage.tokens.{input,output}_
        # tokens). Manual extraction, version-resilient across cohere 5/6/7.
        model, input_tokens, output_tokens, cached_tokens = _extract_cohere_chat_usage(result)
        raw_usage = None  # extracted below via _extract_raw_usage
    else:
        model, input_tokens, output_tokens, cached_tokens = _extract_openai_compatible_usage(result)
        raw_usage = None  # extracted below via _extract_raw_usage
    if not model or model == "unknown":
        model = str(_litellm_bare(route_ctx, kwargs.get("model")) or "unknown")
    # The auto span path reports the requested model, this path reads the
    # provider's echo — so a dated snapshot echo (`…-4-5-20251001`) split one
    # model across two rows. Family-gated: only an echo that is the request plus
    # a date suffix collapses back; a genuinely different served model wins.
    # The requested side is bared first (LiteLLM seam only) or the route prefix
    # would fail that family gate and split the model across two rows again.
    model = _prefer_requested_model(
        _litellm_bare(route_ctx, (kwargs or {}).get("model")), model)

    span_obj = {
        **manual_span_ids(session),
        "span_kind": "llm",
        "span_name": span_name or model,
        "span_order": order,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    metadata = {"workflow_name": session.workflow_name}
    if session.session_id:
        metadata["session_id"] = session.session_id
    # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
    # session metadata WITHOUT it, then re-add it only when this row belongs
    # to the call that was actually rerouted (exact obs-key match).
    _copy_session_metadata(metadata, session)
    _stamp_routing_marker(metadata, session, _resolve_obs_key(obs_key))

    prompt_comp = []
    response_comp = []
    comp_data = {}
    comp_key = f"{session.trace_id}:{order}"
    if hasattr(session, '_pending_compositions'):
        comp_data = session._pending_compositions.pop(comp_key, {}) or {}
        prompt_comp = comp_data.get("prompt", [])
        response_comp = comp_data.get("response", [])

    # ── Gateway attribution (OpenAI-compatible gateways, e.g. OpenRouter) ──
    # Narrowly gated: model_extras is built ONLY when this call is detectably
    # gateway-routed — (a) a serving-provider override was stashed on this
    # span's comp key (_stash_provider_override), (b) the bound client's
    # base_url resolves to a gateway (_effective_provider, e.g.
    # base_url=openrouter.ai through the OpenAI SDK), or (c) the native
    # OpenRouter SDK (provider == "openrouter"). The model string is forwarded
    # un-stripped (e.g. "google/gemini-2.5-flash-lite") and original_provider
    # carries the vendor slug prefix ("google") so server-side identity
    # resolution prices the underlying model. Every non-gateway call leaves
    # the payload byte-identical (model_extras=None).
    model_extras = None
    try:
        gw_provider = str(comp_data.get("provider") or "")
        gw_api_base = str(comp_data.get("api_base") or "")
        if not gw_provider:
            eff = _effective_provider(provider, args or ())
            if eff != provider:
                gw_provider = eff
        if not gw_provider and (provider or "").lower() == "openrouter":
            gw_provider = "openrouter"
        if gw_provider:
            extras = {}
            api_base = gw_api_base or _extract_base_url(args or ())
            if api_base:
                extras["api_base"] = api_base
            if isinstance(model, str) and "/" in model:
                vendor = model.split("/", 1)[0].strip().lower()
                if vendor:
                    extras["original_provider"] = vendor
            if extras:
                model_extras = extras
    except Exception:
        model_extras = None

    # ── LiteLLM deployer attribution ──
    # Covers all three LiteLLM success paths, which all funnel through here:
    # non-streaming completion/acompletion, streaming (the latched final usage
    # chunk arrives as `result`), and embedding/aembedding. Gated strictly on
    # provider == "litellm", so every other provider's payload stays
    # byte-identical. `setdefault` keeps a gateway-derived hint (built above)
    # authoritative if one somehow exists.
    if (provider or "").lower() == "litellm":
        try:
            _orig_provider = _resolve_litellm_original_provider(
                (kwargs or {}).get("model"), result)
            if _orig_provider:
                if model_extras is None:
                    model_extras = {}
                model_extras.setdefault("original_provider", _orig_provider)
        except Exception:
            pass  # fail-safe: telemetry hint only, never break the log

    # ── Raw provider usage + shape ──
    # Extract verbatim usage from `result` (stream wrappers pass the latched
    # final usage chunk / response.completed event as `result`). Optional
    # session._pending_usage stash is a legacy no-op path (nothing writes it
    # today) kept for fail-open compatibility. The embedding branch above
    # already populated raw_usage from _extract_embedding_usage.
    if operation != "embedding":
        try:
            stash = getattr(session, "_pending_usage", None)
            if stash is not None and comp_key in stash:
                raw_usage = stash.pop(comp_key)
            else:
                raw_usage = _extract_raw_usage(provider, result)
                # Pydantic models / SDK return objects / SimpleNamespace (raw-HTTP
                # manual wrappers) don't JSON-encode directly; coerce to a dict.
                # _as_dict handles the top level; the json round-trip flattens any
                # nested SDK/SimpleNamespace objects (e.g. Gemini usageMetadata's
                # promptTokensDetails list) so the /log payload always serializes.
                if raw_usage is not None:
                    import json as _json
                    raw_usage = _json.loads(_json.dumps(
                        _as_dict(raw_usage),
                        default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
                    ))
        except Exception:
            raw_usage = None

    if operation == "embedding":
        usage_shape = shape_override or _resolve_embedding_shape(provider)
    else:
        usage_shape = shape_override or _resolve_usage_shape(provider)

    usage_block = {
        "shape": usage_shape,
        "raw": raw_usage,
    }
    # Provider-reported service tier (OpenAI response top level / Anthropic
    # usage object) → usage.tier so batch/flex/priority pricing is applied
    # server-side on the manual-telemetry path too.
    _svc_tier = _extract_service_tier(result)
    if _svc_tier:
        usage_block["tier"] = _svc_tier

    # Drain applied local_decision + shadow/reject observations so
    # stream + manual success paths confirm REQUEST_REROUTED / REROUTE_REJECTED
    # on /log (parity with _flush_deferred_spans). Keyed on ``obs_key`` —
    # stream wrappers thread the key captured after their own check, since a
    # late stream drain can run after another call's check overwrote the var.
    pending_local_decision = None
    pending_observations = None
    try:
        _log_obs_key = _resolve_obs_key(obs_key)
        pending_local_decision = _claim_local_decision(session, _log_obs_key)
        pending_observations = _state.drain_observations(_log_obs_key)
    except Exception:
        pending_local_decision = None
        pending_observations = None

    tp.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        metadata=metadata,
        span=span_obj,
        prompt_composition=prompt_comp,
        response_composition=response_comp,
        usage=usage_block,
        operation=operation,
        latency=latency,
        model_extras=model_extras,
        local_decision=pending_local_decision,
        observations=pending_observations or None,
    )

    # Part 2: after the chat row, emit child image spans for Responses
    # built-in image_generation tool calls (stream + non-stream).
    if (provider or "").lower() == "openai_responses":
        try:
            _log_responses_image_children(
                session, args or (), kwargs or {}, result, start_time,
            )
        except Exception:
            pass  # fail-open


def _sanitize_image_dim(v):
    """A usable image dimension, or "" when there is nothing to report.

    'auto' is the provider choosing for us, not an observation, so it must never
    travel as a value — empty string means "not observed"; leave unset rather than inventing a default.
    """
    if not isinstance(v, str):
        return ""
    s = v.strip()
    return s if s and s.lower() != "auto" else ""


def _extract_responses_image_tool_config(args, kwargs):
    """Read image model/size/quality from Responses tools[] config. Fail-open."""
    try:
        body = None
        if args and isinstance(args[0], dict):
            body = args[0]
        elif isinstance(kwargs, dict):
            body = kwargs
        tools = (body or {}).get("tools") if isinstance(body, dict) else None
        if not isinstance(tools, list):
            return {"model": "gpt-image-1", "size": "", "quality": ""}
        for t in tools:
            t = _as_dict(t) if t is not None else None
            if not isinstance(t, dict):
                continue
            if t.get("type") != "image_generation":
                continue
            model = t.get("model") or "gpt-image-1"
            if not isinstance(model, str) or not model:
                model = "gpt-image-1"
            size = t.get("size") or ""
            if not isinstance(size, str):
                size = ""
            if size.lower() == "auto":
                size = ""
            quality = t.get("quality") or ""
            if not isinstance(quality, str):
                quality = ""
            if quality.lower() == "auto":
                quality = ""
            return {"model": model, "size": size, "quality": quality}
    except Exception:
        pass
    return {"model": "gpt-image-1", "size": "", "quality": ""}


def _list_responses_image_calls(result):
    """Completed (or result-bearing) image_generation_call items. Fail-open."""
    try:
        resp = result
        if (
            resp is not None
            and _attr(resp, "type") == "response.completed"
            and _attr(resp, "response") is not None
        ):
            resp = _attr(resp, "response")
        output = _attr(resp, "output")
        if output is None:
            output = _attr(result, "output")
        if not isinstance(output, list):
            return []
        out = []
        for it in output:
            if _attr(it, "type") != "image_generation_call":
                continue
            status = _attr(it, "status")
            if status == "failed":
                continue
            res = _attr(it, "result")
            if status in ("in_progress", "generating"):
                if not isinstance(res, str) or not res:
                    continue
            out.append(it)
        return out
    except Exception:
        return []


@fail_safe
def _log_responses_image_children(session, args, kwargs, result, start_time):
    """Emit one child openai_images span per Responses image_generation_call.

    Mirrors Node ``_logResponsesImageChildren``. Never throws (``@fail_safe``).
    """
    tp = get_client()
    if not tp:
        return
    images = _list_responses_image_calls(result)
    if not images:
        return
    cfg = _extract_responses_image_tool_config(args, kwargs)
    model = cfg.get("model") or "gpt-image-1"

    for item in images:
        try:
            try:
                order = session.next_span_order()
            except Exception:
                order = 0
            b64 = _attr(item, "result")
            if not isinstance(b64, str):
                b64 = ""
            # Reuse cascade via synthetic Images API shape for b64 header.
            synthetic = {"data": [{"b64_json": b64}]} if b64 else {}
            # Opportunistically read the dims off the output item itself.
            # Today the image_generation_call item ships only
            # {id,result,status,type}, so these are always absent and behavior is
            # unchanged — but openai-python sets extra="allow", so any field
            # OpenAI starts echoing is reachable and we capture it for free. The
            # item value wins over the request tool config because it is what was
            # actually produced (the request may have said nothing, or 'auto').
            # Never invent a default: empty string means "not observed".
            item_size = _sanitize_image_dim(_attr(item, "size"))
            item_quality = _sanitize_image_dim(_attr(item, "quality"))
            image_size = _resolve_image_size(
                # Ahead of cfg["size"] deliberately: _resolve_image_size
                # short-circuits on the first usable source, so passing it after
                # the b64/provider-default steps would make it dead code.
                request_size=item_size or cfg.get("size") or None,
                result=synthetic,
                provider="openai",
                model=model,
            )
            image_quality = item_quality or cfg.get("quality") or ""

            span_obj = {
                **manual_span_ids(session),
                "span_kind": "llm",
                "span_name": model,
                "span_order": order,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": datetime.now(timezone.utc).isoformat(),
            }
            metadata = {
                "workflow_name": session.workflow_name,
                "modality": "image_gen",
                "tp_source": "openai_responses_image_generation",
            }
            if session.session_id:
                metadata["session_id"] = session.session_id
            # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
            # session metadata WITHOUT it, then re-add it only when this row belongs
            # to the call that was actually rerouted (exact obs-key match).
            _copy_session_metadata(metadata, session)
            _stamp_routing_marker(metadata, session, _state.get_current_obs_key())

            items = {
                "images_generated": 1,
                "image_model": model,
                "image_size": image_size,
                "image_quality": image_quality,
            }
            usage_block = {
                "shape": "openai_images",
                "raw": {},
                "items": items,
                "duration": {},
            }
            tp.log_sync(
                user_id=session.user_id,
                paid_plan=session.paid_plan,
                plan_source=getattr(session, "plan_source", None),
                workflow_name=session.workflow_name,
                session_id=session.session_id,
                model=str(model),
                provider="openai",
                metadata=metadata,
                span=span_obj,
                prompt_composition=[],
                response_composition=[{"role": "assistant", "type": "image"}],
                usage=usage_block,
                operation="image_gen",
            )
        except Exception:
            pass  # per-item fail-open


# ═══════════════════════════════════════════════════════════════════
# Modality (image / audio / video / OCR) extraction registry.
#
# Each handler returns:
# intent(args, kwargs) -> dict forwarded to /check so rules can match on
# intent.kind / intent.count / etc.
# extract(args, kwargs, result) -> {items, duration, raw} produced after
# the call so the service receives the
# same `usage = { shape, raw, items, duration }`
# contract used for text calls.
#
# Adding a new (provider, modality) pair is one entry in _MODALITY_HANDLERS.
# Adding a new method to instrument is one entry in _TARGET_METHODS with the
# matching `modality` + `shape` fields.
# ═══════════════════════════════════════════════════════════════════

def _safe_int(v, default=0):
    try:
        return int(v) if v is not None else default
    except Exception:
        return default

def _image_size_from_dims(w, h):
    """Format positive width×height as 'WxH' for items.image_size; '' if unknown."""
    width = _safe_int(w, 0)
    height = _safe_int(h, 0)
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return ""


# ── image size resolution (request → response → binary → default) ──────
# Mapper only multiplies parseable "WxH" × count. Request dims are often omitted
# (provider defaults). Resolve from response metadata / in-memory image headers
# before falling back to documented API defaults. Never fetch URLs. Fail-open.
_IMAGE_SIZE_PARSE_RE = re.compile(
    r"^(\d+)\s*(?:[x×*]|-x-)\s*(\d+)$", re.IGNORECASE
)


def _parse_image_size_str(size_str):
    """Return 'WxH' if *size_str* is a parseable pixel size; else ''."""
    if size_str is None:
        return ""
    try:
        s = str(size_str).strip()
    except Exception:
        return ""
    if not s:
        return ""
    m = _IMAGE_SIZE_PARSE_RE.match(s)
    if not m:
        return ""
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0 or w > 10_000_000 or h > 10_000_000:
        return ""
    return f"{w}x{h}"


def _image_dims_from_binary(data):
    """Read width×height from PNG / JPEG / WEBP headers. Returns 'WxH' or ''.

    Operates on already-in-memory bytes only (never network). Bounded reads.
    Fail-open: any error → ''.
    """
    try:
        if data is None:
            return ""
        if isinstance(data, memoryview):
            data = data.tobytes()
        elif isinstance(data, bytearray):
            data = bytes(data)
        elif not isinstance(data, (bytes, bytearray)):
            return ""
        if len(data) < 24:
            return ""
        # PNG: signature + IHDR length/type + width/height (big-endian)
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return _image_size_from_dims(w, h)
        # JPEG: scan SOF0/SOF2 within first 64 KiB
        if data[:2] == b"\xff\xd8":
            i = 2
            limit = min(len(data), 65536)
            while i + 9 < limit:
                if data[i] != 0xFF:
                    i += 1
                    continue
                while i < limit and data[i] == 0xFF:
                    i += 1
                if i >= limit:
                    break
                marker = data[i]
                i += 1
                # Standalone markers without length
                if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
                    continue
                if i + 1 >= limit:
                    break
                seg_len = (data[i] << 8) | data[i + 1]
                if seg_len < 2:
                    break
                # SOF0 / SOF2 (baseline / progressive)
                if marker in (0xC0, 0xC2) and i + 7 < limit:
                    h = (data[i + 3] << 8) | data[i + 4]
                    w = (data[i + 5] << 8) | data[i + 6]
                    return _image_size_from_dims(w, h)
                i += seg_len
            return ""
        # WEBP: RIFF....WEBP + VP8 / VP8L / VP8X
        if (
            len(data) >= 30
            and data[:4] == b"RIFF"
            and data[8:12] == b"WEBP"
        ):
            fourcc = data[12:16]
            if fourcc == b"VP8X" and len(data) >= 30:
                # canvas width/height are 24-bit little-endian minus 1
                w = 1 + (data[24] | (data[25] << 8) | (data[26] << 16))
                h = 1 + (data[27] | (data[28] << 8) | (data[29] << 16))
                return _image_size_from_dims(w, h)
            if fourcc == b"VP8 " and len(data) >= 30:
                # lossy frame: width/height at offset 26 (14-bit LE each)
                w = (data[26] | (data[27] << 8)) & 0x3FFF
                h = (data[28] | (data[29] << 8)) & 0x3FFF
                return _image_size_from_dims(w, h)
            if fourcc == b"VP8L" and len(data) >= 25:
                bits = data[21] | (data[22] << 8) | (data[23] << 16) | (data[24] << 24)
                w = (bits & 0x3FFF) + 1
                h = ((bits >> 14) & 0x3FFF) + 1
                return _image_size_from_dims(w, h)
        return ""
    except Exception:
        return ""


def _b64_prefix_to_bytes(s, max_decoded=96):
    """Decode just enough base64 to read an image header. Fail-open → None.

    Only touches a small prefix of *s* (never strips/allocates the full
    multi-MB payload) so a large b64_json cannot burn CPU on the hot path.
    """
    try:
        if not isinstance(s, str) or not s:
            return None
        # data-URL prefix: data:image/png;base64,....
        if "base64," in s[:80]:
            s = s.split("base64,", 1)[1]
        # Need ~ceil(max_decoded * 4/3) chars; take a little extra then strip
        # whitespace on the *prefix only* so newlines in wrapped b64 still work.
        want = ((max_decoded + 3) * 4) // 3
        prefix = s[: want + 64]
        prefix = "".join(prefix.split())
        if not prefix:
            return None
        n_chars = min(len(prefix), want)
        n_chars -= n_chars % 4
        if n_chars < 24:
            return None
        import base64 as _b64mod
        return _b64mod.b64decode(prefix[:n_chars])
    except Exception:
        return None


def _attr_or_key(obj, name):
    """Read attr or dict key; None on any failure."""
    if obj is None:
        return None
    try:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)
    except Exception:
        return None


def _first_image_payload(result):
    """Yield candidate binary / sized objects from a provider image response.

    Never fetches URLs. Synchronous only. Fail-open.
    """
    if result is None:
        return
    # PIL.Image-like: .size = (w, h)
    try:
        sz = _attr_or_key(result, "size")
        if isinstance(sz, (tuple, list)) and len(sz) >= 2:
            yield ("dims", sz[0], sz[1])
        elif isinstance(sz, dict):
            yield ("dims", sz.get("width"), sz.get("height"))
    except Exception:
        pass
    # Direct bytes / buffer
    if isinstance(result, (bytes, bytearray, memoryview)):
        yield ("bytes", result)
        return
    # OpenAI / Together: result.data[i].b64_json
    data = _attr_or_key(result, "data")
    if data is None and isinstance(result, (list, tuple)):
        data = result
    try:
        # Only iterate real sequences — never walk a huge str/bytes as items.
        if isinstance(data, (list, tuple)):
            for item in data:
                for key in ("b64_json", "base64", "b64"):
                    v = _attr_or_key(item, key)
                    if isinstance(v, str) and v:
                        yield ("b64", v)
                        break
                for key in ("bytes", "image_bytes", "imageBytes", "content"):
                    v = _attr_or_key(item, key)
                    if isinstance(v, (bytes, bytearray, memoryview)):
                        yield ("bytes", v)
                        break
                # nested .image.image_bytes (Google Imagen)
                img = _attr_or_key(item, "image")
                if img is not None:
                    for key in ("image_bytes", "imageBytes", "data", "bytes"):
                        v = _attr_or_key(img, key)
                        if isinstance(v, (bytes, bytearray, memoryview)):
                            yield ("bytes", v)
                            break
                        if isinstance(v, str) and v:
                            yield ("b64", v)
                            break
                w = _attr_or_key(item, "width")
                h = _attr_or_key(item, "height")
                if w is not None and h is not None:
                    yield ("dims", w, h)
                sz = _attr_or_key(item, "size")
                if isinstance(sz, str) and sz:
                    yield ("size_str", sz)
                elif isinstance(sz, (tuple, list)) and len(sz) >= 2:
                    yield ("dims", sz[0], sz[1])
    except Exception:
        pass
    # xAI ImageResponse: .base64 property (may raise if format=url — catch)
    try:
        b64 = _attr_or_key(result, "base64")
        if isinstance(b64, str) and b64:
            yield ("b64", b64)
    except Exception:
        pass
    # Google: result.generated_images
    for coll_name in ("generated_images", "generatedImages", "images"):
        coll = _attr_or_key(result, coll_name)
        if not coll:
            continue
        try:
            for item in coll:
                img = _attr_or_key(item, "image") or item
                for key in ("image_bytes", "imageBytes", "data", "bytes"):
                    v = _attr_or_key(img, key)
                    if isinstance(v, (bytes, bytearray, memoryview)):
                        yield ("bytes", v)
                        break
                    if isinstance(v, str) and v:
                        yield ("b64", v)
                        break
                # PIL nested
                sz = _attr_or_key(img, "size")
                if isinstance(sz, (tuple, list)) and len(sz) >= 2:
                    yield ("dims", sz[0], sz[1])
        except Exception:
            pass
    # Top-level size / width / height
    try:
        sz = _attr_or_key(result, "size")
        if isinstance(sz, str) and sz:
            yield ("size_str", sz)
        w = _attr_or_key(result, "width")
        h = _attr_or_key(result, "height")
        if w is not None and h is not None:
            yield ("dims", w, h)
    except Exception:
        pass


def _image_size_from_result(result):
    """Best-effort size from response payload / in-memory binary. Never throws."""
    try:
        for kind, *rest in _first_image_payload(result):
            if kind == "dims":
                s = _image_size_from_dims(rest[0], rest[1])
                if s:
                    return s
            elif kind == "size_str":
                s = _parse_image_size_str(rest[0])
                if s:
                    return s
            elif kind == "bytes":
                s = _image_dims_from_binary(rest[0])
                if s:
                    return s
            elif kind == "b64":
                raw = _b64_prefix_to_bytes(rest[0])
                if raw:
                    s = _image_dims_from_binary(raw)
                    if s:
                        return s
    except Exception:
        pass
    return ""


def _image_size_provider_default(provider, model, request_size=None):
    """Documented API default when request/response/binary yield nothing.

    Conservative — only where the API pins a single pixel default.
    xAI left empty (aspect + 1k/2k is not unique WxH).
    """
    try:
        p = (provider or "").lower()
        req = (request_size or "").strip().lower() if request_size else ""
        # "auto" is not a pixel size — treat as omitted for default purposes
        if req and req not in ("auto", "null", "none"):
            # Caller passed a non-empty size we couldn't parse (e.g. aspect) —
            # do not invent over an explicit non-pixel request intent.
            if _parse_image_size_str(request_size):
                return _parse_image_size_str(request_size)
            # aspect / tier strings → no default invent
            if ":" in req or req in ("1k", "2k", "4k"):
                return ""
        if p in ("together", "openai", "huggingface", "hf", "google", "gemini"):
            # Documented/common default square when size omitted/auto.
            # model is accepted for future per-model tables; unused today.
            _ = model
            return "1024x1024"
        # xai / ai_sdk / unknown: no invent
        return ""
    except Exception:
        return ""


def _resolve_image_size(
    *,
    request_size=None,
    request_w=None,
    request_h=None,
    result=None,
    provider=None,
    model=None,
    allow_default=True,
):
    """Cascade: request WxH → request dims → response/binary → provider default.

    Returns parseable 'WxH' or ''. Never throws. Never fetches URLs.
    """
    try:
        # 1. Request size string (parseable only)
        s = _parse_image_size_str(request_size)
        if s:
            return s
        # 2. Request numeric dims
        s = _image_size_from_dims(request_w, request_h)
        if s:
            return s
        # 3–4. Response metadata + binary header
        s = _image_size_from_result(result)
        if s:
            return s
        # 5. Documented default
        if allow_default:
            return _image_size_provider_default(provider, model, request_size)
        return ""
    except Exception:
        return ""


def _dump_raw_usage(obj):
    if obj is None:
        return {}
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}

def _intent_openai_images(args, kwargs):
    n = _safe_int(kwargs.get("n"), 1)
    return {
        "kind": "image_generation",
        "count": n,
        "size": str(kwargs.get("size") or "") or None,
        "quality": str(kwargs.get("quality") or "") or None,
    }

def _extract_openai_images(args, kwargs, result):
    n = _safe_int(kwargs.get("n"), 1)
    raw = _dump_raw_usage(_attr(result, "usage"))
    data = _attr(result, "data") or []
    try:
        actual = len(data) if data else n
    except Exception:
        actual = n
    model = str(kwargs.get("model") or "")
    image_size = _resolve_image_size(
        request_size=kwargs.get("size"),
        result=result,
        provider="openai",
        model=model,
    )
    return {
        "items": {
            "images_generated": actual,
            "image_size": image_size,
            "image_quality": str(kwargs.get("quality") or ""),
            "image_model": model,
        },
        "duration": {},
        "raw": raw,
    }

def _intent_openai_audio_tts(args, kwargs):
    text = kwargs.get("input") or ""
    return {
        "kind": "audio_speech",
        "character_count": len(text) if isinstance(text, str) else 0,
        "voice": kwargs.get("voice") or None,
    }

def _extract_openai_audio_tts(args, kwargs, result):
    text = kwargs.get("input") or ""
    chars = len(text) if isinstance(text, str) else 0
    # When the caller opted into stream_format="sse",
    # _maybe_tap_openai_tts_sse(_sync|_async) attached the captured usage
    # here. Otherwise (binary path) the OpenAI SDK returns a binary content
    # wrapper with no `.usage` — _dump_raw_usage returns {} and the row is
    # reported without usage, so its cost is marked unmeasured.
    captured = getattr(result, "_tp_captured_usage", None)
    raw = captured if captured is not None else _dump_raw_usage(_attr(result, "usage"))
    return {
        "items": {
            "tts_characters": chars,
            "audio_model": str(kwargs.get("model") or ""),
            "voice": str(kwargs.get("voice") or ""),
        },
        "duration": {},
        "raw": raw,
    }


# ════════════════════════════════════════════════════════════════════════════
# Best-effort OpenAI TTS SSE tap (Python).
#
# Mirror of the Node SDK's equivalent helper. When the caller passes
# stream_format="sse" on a TTS-capable model (gpt-4o-mini-tts and its dated
# snapshots), drain the SSE bytes, parse the terminal speech.audio.done
# event for `usage`, reassemble the audio chunks, and return a duck-typed
# replacement object so the customer's downstream code keeps seeing a
# normal binary response (`.read()` returns mp3 bytes, not SSE text).
#
# Never auto-injected. Streaming playback is surrendered when sse is opted
# into — SSE opt-in already forgoes incremental playback, so buffering here
# does not remove any capability the caller had.
#
# Note: this model allowlist is release-pinned; new TTS models require an
# SDK update until capability templates are delivered dynamically.
# ════════════════════════════════════════════════════════════════════════════
import base64 as _b64
import json as _json
import re as _re

_OPENAI_TTS_SSE_CAPABLE_MODELS = _re.compile(r"^gpt-4o(-mini)?-tts")


def _parse_openai_tts_sse_buffer(buf):
    """Returns (audio_bytes, usage_dict_or_None)."""
    audio_chunks = []
    usage = None
    if isinstance(buf, str):
        buf = buf.encode("utf-8", errors="replace")
    text = buf.replace(b"\r\n", b"\n").decode("utf-8", errors="replace")
    for frame in text.split("\n\n"):
        for line in frame.split("\n"):
            if not line.startswith("data:"):
                continue
            payload = line[5:].lstrip()
            if payload == "[DONE]":
                continue
            try:
                obj = _json.loads(payload)
            except Exception:
                continue
            audio_field = obj.get("audio") if isinstance(obj, dict) else None
            if isinstance(audio_field, str):
                try:
                    audio_chunks.append(_b64.b64decode(audio_field))
                except Exception:
                    pass
            if isinstance(obj, dict) and obj.get("type") == "speech.audio.done" and isinstance(obj.get("usage"), dict):
                usage = obj["usage"]
    return b"".join(audio_chunks), usage


class _TokenPoliceBufferedTtsResponse:
    """Duck-typed stand-in for openai._legacy_response.HttpxBinaryResponseContent.

    Owns reassembled audio bytes. The methods most customer code touches —
    `.read()`, `.aread()`, `.iter_bytes()`, `.content`, `.text`,
    `.write_to_file()`, `.close()` — return the buffered audio. Anything we
    don't override delegates to the original (`__getattr__`) so a customer
    poking at uncommon methods still gets the underlying response object's
    behavior."""

    def __init__(self, audio_bytes, original, captured_usage):
        # Underscore prefix so __getattr__ delegation isn't shadowed.
        object.__setattr__(self, "_tp_audio", audio_bytes)
        object.__setattr__(self, "_tp_original", original)
        object.__setattr__(self, "_tp_captured_usage", captured_usage)

    def __getattr__(self, name):
        # Only invoked if `name` isn't found on self.
        return getattr(self._tp_original, name)

    @property
    def content(self):
        return self._tp_audio

    @property
    def text(self):
        # Mirror httpx semantics; mp3 bytes won't decode meaningfully but
        # this keeps the property type stable for code that reads it.
        try:
            return self._tp_audio.decode("utf-8")
        except Exception:
            return ""

    def read(self):
        return self._tp_audio

    async def aread(self):
        return self._tp_audio

    def iter_bytes(self, chunk_size=None):
        if chunk_size and chunk_size > 0:
            for i in range(0, len(self._tp_audio), chunk_size):
                yield self._tp_audio[i:i + chunk_size]
        else:
            yield self._tp_audio

    def iter_raw(self, chunk_size=None):
        return self.iter_bytes(chunk_size)

    def iter_text(self, chunk_size=None):
        for chunk in self.iter_bytes(chunk_size):
            yield chunk.decode("utf-8", errors="replace")

    def iter_lines(self):
        for line in self._tp_audio.split(b"\n"):
            yield line.decode("utf-8", errors="replace")

    async def aiter_bytes(self, chunk_size=None):
        async def _gen():
            for chunk in self.iter_bytes(chunk_size):
                yield chunk
        return _gen()

    async def aiter_raw(self, chunk_size=None):
        return await self.aiter_bytes(chunk_size)

    async def aiter_text(self, chunk_size=None):
        async def _gen():
            for chunk in self.iter_text(chunk_size):
                yield chunk
        return _gen()

    async def aiter_lines(self):
        async def _gen():
            for line in self.iter_lines():
                yield line
        return _gen()

    def write_to_file(self, file):
        with open(file, "wb") as f:
            f.write(self._tp_audio)

    def stream_to_file(self, file, *, chunk_size=None):
        self.write_to_file(file)

    def close(self):
        try:
            self._tp_original.close()
        except Exception:
            pass


def _should_tap_openai_tts(provider, modality, kwargs, result):
    if provider != "openai" or modality != "audio_tts":
        return False
    if not isinstance(kwargs, dict):
        return False
    if kwargs.get("stream_format") != "sse":
        return False
    model = kwargs.get("model") or ""
    if not isinstance(model, str) or not _OPENAI_TTS_SSE_CAPABLE_MODELS.match(model):
        return False
    # The OpenAI Python SDK exposes both `.read()` (sync) and `.aread()`
    # (async) on HttpxBinaryResponseContent. Anything else (e.g. customer
    # already used with_streaming_response or with_raw_response) lacks
    # this contract — bail.
    return result is not None and (hasattr(result, "read") or hasattr(result, "aread"))


def _tts_body_bytes(audio, raw_bytes):
    """Pick the body bytes for the rebuilt TTS response.

    Invariant: never hand the customer an empty rebuilt body when the provider
    actually sent bytes. If audio extraction came up empty but the raw response
    was non-empty (e.g. the per-frame audio field was renamed/moved), fall back
    to the original bytes verbatim rather than a silently-empty body — the
    customer opted into SSE, so this returns exactly what the provider sent. The
    wrapper delegates headers to the original, so its content-type is preserved
    on the fallback path; and metering is unaffected because usage rides its own
    frame. A genuinely empty 200 (raw_bytes empty) keeps prior behavior.
    Covered by the fallback + happy-path TTS tap tests.
    """
    if not audio and raw_bytes:
        return raw_bytes
    return audio


def _maybe_tap_openai_tts_sse_sync(provider, modality, kwargs, result):
    try:
        if not _should_tap_openai_tts(provider, modality, kwargs, result):
            return result
        raw_bytes = result.read()
        audio, usage = _parse_openai_tts_sse_buffer(raw_bytes)
        return _TokenPoliceBufferedTtsResponse(_tts_body_bytes(audio, raw_bytes), result, usage)
    except Exception:
        # Fail-safe: never let the tap break the customer's audio call.
        return result


async def _maybe_tap_openai_tts_sse_async(provider, modality, kwargs, result):
    try:
        if not _should_tap_openai_tts(provider, modality, kwargs, result):
            return result
        if hasattr(result, "aread"):
            raw_bytes = await result.aread()
        else:
            raw_bytes = result.read()
        audio, usage = _parse_openai_tts_sse_buffer(raw_bytes)
        return _TokenPoliceBufferedTtsResponse(_tts_body_bytes(audio, raw_bytes), result, usage)
    except Exception:
        return result

def _audio_file_seconds(file_obj):
    """Best-effort duration extraction from an audio file handle. Returns 0 on
    any failure — pricing degrades gracefully to per-token if seconds are
    missing. Soft-dependency on `mutagen`."""
    if file_obj is None:
        return 0.0
    path = None
    if isinstance(file_obj, str):
        path = file_obj
    elif hasattr(file_obj, "name") and isinstance(getattr(file_obj, "name"), str):
        path = file_obj.name
    if not path:
        return 0.0
    try:
        import mutagen  # type: ignore
        info = mutagen.File(path)
        if info is not None and getattr(info, "info", None) is not None:
            return float(info.info.length or 0)
    except Exception:
        pass
    try:
        import wave
        with wave.open(path, "rb") as wf:
            return float(wf.getnframes()) / float(wf.getframerate() or 1)
    except Exception:
        return 0.0

def _intent_openai_audio_stt(args, kwargs):
    secs = _audio_file_seconds(kwargs.get("file"))
    return {
        "kind": "audio_transcription",
        "expected_seconds": secs or None,
    }

def _extract_openai_audio_stt(args, kwargs, result):
    raw = _dump_raw_usage(_attr(result, "usage"))
    secs = float(raw.get("duration") if isinstance(raw, dict) and raw.get("duration") else 0) or _audio_file_seconds(kwargs.get("file"))
    return {
        "items": {"audio_model": str(kwargs.get("model") or "")},
        "duration": {"audio_seconds": float(secs or 0)},
        "raw": raw,
    }

def _intent_google_imagen(args, kwargs):
    cfg = kwargs.get("config") or {}
    n = _safe_int(_attr(cfg, "number_of_images") if cfg else kwargs.get("number_of_images"), 1)
    return {"kind": "image_generation", "count": n}

def _extract_google_imagen(args, kwargs, result):
    images = _attr(result, "generated_images") or _attr(result, "images") or []
    try:
        count = len(images)
    except Exception:
        count = 0
    cfg = kwargs.get("config") or {}
    if not count:
        count = _safe_int(_attr(cfg, "number_of_images") if cfg else kwargs.get("number_of_images"), 1)
    w = _attr(cfg, "width") if cfg else None
    if w is None:
        w = kwargs.get("width")
    h = _attr(cfg, "height") if cfg else None
    if h is None:
        h = kwargs.get("height")
    # Aspect / tier strings are not pixel dims — pass through so the default
    # path refuses to invent over an explicit non-pixel size intent.
    aspect = (
        _attr(cfg, "aspect_ratio") if cfg else None
    ) or kwargs.get("aspect_ratio") or kwargs.get("aspectRatio")
    image_size_cfg = (
        _attr(cfg, "image_size") if cfg else None
    ) or kwargs.get("image_size") or kwargs.get("imageSize")
    request_size = None
    if aspect:
        request_size = str(aspect)
    elif image_size_cfg is not None:
        request_size = str(image_size_cfg)
    model = str(kwargs.get("model") or "")
    image_size = _resolve_image_size(
        request_size=request_size,
        request_w=w,
        request_h=h,
        result=result,
        provider="google",
        model=model,
    )
    return {
        "items": {
            "images_generated": count,
            "image_model": model,
            "image_size": image_size,
        },
        "duration": {},
        "raw": {},
    }

def _intent_google_veo(args, kwargs):
    cfg = kwargs.get("config") or {}
    secs = _safe_int(_attr(cfg, "duration_seconds") if cfg else kwargs.get("duration_seconds"), 0)
    return {"kind": "video_generation", "expected_seconds": secs or None}

def _extract_google_veo(args, kwargs, result):
    cfg = kwargs.get("config") or {}
    secs = _safe_int(_attr(cfg, "duration_seconds") if cfg else kwargs.get("duration_seconds"), 0)
    # Some Veo responses surface duration in the operation metadata; fall back to request.
    op_meta = _attr(result, "metadata") or {}
    if isinstance(op_meta, dict) and not secs:
        secs = _safe_int(op_meta.get("video_duration_seconds") or op_meta.get("duration_seconds"), 0)
    return {
        "items": {"video_model": str(kwargs.get("model") or "")},
        "duration": {"video_seconds": float(secs or 0)},
        "raw": {},
    }

def _xai_image_call_info(args, kwargs):
    """Extract (model, prompt, n) from a xai-sdk image call.

    Signature is `Client.sample(self, prompt, model, *, ...)` or
    `Client.sample_batch(self, prompt, model, n, *, ...)`, so the positional
    args carry prompt + model + (optional) n.
    """
    prompt = ""
    model = str(kwargs.get("model") or "")
    n = _safe_int(kwargs.get("n"), 0)
    if len(args) > 1 and isinstance(args[1], str):
        prompt = args[1]
    if not model and len(args) > 2 and isinstance(args[2], str):
        model = args[2]
    if n == 0 and len(args) > 3:
        n = _safe_int(args[3], 0)
    return model, prompt, n or 1

def _intent_xai_image(args, kwargs):
    _, _, n = _xai_image_call_info(args, kwargs)
    return {"kind": "image_generation", "count": n}

def _extract_xai_image(args, kwargs, result):
    model, _, n = _xai_image_call_info(args, kwargs)
    raw = _dump_raw_usage(_attr(result, "usage"))
    # sample() returns one ImageResponse; sample_batch() returns Sequence[ImageResponse].
    actual = 1
    if isinstance(result, (list, tuple)):
        try:
            actual = len(result)
        except Exception:
            actual = n
    elif _attr(result, "images") is not None:
        try:
            actual = len(_attr(result, "images") or [])
        except Exception:
            actual = n
    # xAI uses aspect_ratio + resolution ("1k"/"2k") — not unique WxH. Capture
    # binary dims when format=base64; never invent a default (allow_default=False).
    image_size = _resolve_image_size(
        result=result,
        provider="xai",
        model=model,
        allow_default=False,
    )
    return {
        "items": {
            "images_generated": actual,
            "image_model": model,
            "image_size": image_size,
        },
        "duration": {},
        "raw": raw,
    }

def _intent_mistral_audio_tts(args, kwargs):
    text = kwargs.get("input") or kwargs.get("text") or ""
    return {
        "kind": "audio_speech",
        "character_count": len(text) if isinstance(text, str) else 0,
    }

def _extract_mistral_audio_tts(args, kwargs, result):
    text = kwargs.get("input") or kwargs.get("text") or ""
    chars = len(text) if isinstance(text, str) else 0
    return {
        "items": {"tts_characters": chars, "audio_model": str(kwargs.get("model") or "")},
        "duration": {},
        "raw": _dump_raw_usage(_attr(result, "usage")),
    }

def _intent_mistral_audio_stt(args, kwargs):
    secs = _audio_file_seconds(kwargs.get("file") or kwargs.get("audio"))
    return {"kind": "audio_transcription", "expected_seconds": secs or None}

def _extract_mistral_audio_stt(args, kwargs, result):
    raw = _dump_raw_usage(_attr(result, "usage_info") or _attr(result, "usage"))
    # Voxtral STT is billed per AUDIO SECOND, and the response reports it as
    # `usage.prompt_audio_seconds` (a `UsageInfo` field on both mistralai
    # majors) — there is no `duration` key, so reading only that one always
    # fell through to the local-file fallback, which is structurally 0 for the
    # `file_url=` / `file_id=` call forms (nothing local was ever read).
    # Result: every URL/id transcription billed as 0 seconds. `duration` is
    # kept as a second probe in case a future release adds one.
    secs = 0.0
    if isinstance(raw, dict):
        for key in ("prompt_audio_seconds", "duration"):
            try:
                secs = float(raw.get(key) or 0)
            except (TypeError, ValueError):
                secs = 0.0
            if secs:
                break
    if not secs:
        secs = _audio_file_seconds(kwargs.get("file") or kwargs.get("audio"))
    return {
        "items": {"audio_model": str(kwargs.get("model") or "")},
        "duration": {"audio_seconds": float(secs or 0)},
        "raw": raw,
    }

def _intent_mistral_ocr(args, kwargs):
    return {"kind": "ocr"}

def _extract_mistral_ocr(args, kwargs, result):
    raw = _dump_raw_usage(_attr(result, "usage_info") or _attr(result, "usage"))
    pages = _safe_int(raw.get("pages_processed") if isinstance(raw, dict) else 0, 0)
    return {
        "items": {"ocr_pages": pages},
        "duration": {},
        "raw": raw,
    }

def _intent_together_image(args, kwargs):
    n = _safe_int(kwargs.get("n"), 1)
    return {"kind": "image_generation", "count": n}

def _extract_together_image(args, kwargs, result):
    data = _attr(result, "data") or []
    try:
        actual = len(data) if data else _safe_int(kwargs.get("n"), 1)
    except Exception:
        actual = _safe_int(kwargs.get("n"), 1)
    model = str(kwargs.get("model") or "")
    # Cascade: request dims → response b64 header → API default 1024×1024.
    image_size = _resolve_image_size(
        request_w=kwargs.get("width"),
        request_h=kwargs.get("height"),
        result=result,
        provider="together",
        model=model,
    )
    return {
        "items": {
            "images_generated": actual,
            "image_model": model,
            "image_size": image_size,
        },
        "duration": {},
        "raw": _dump_raw_usage(_attr(result, "usage")),
    }

def _intent_hf_image(args, kwargs):
    return {"kind": "image_generation", "count": 1}

def _extract_hf_image(args, kwargs, result):
    params = kwargs.get("parameters") or {}
    # HF text-to-image often carries width/height under parameters (or top-level).
    # Response is commonly a PIL.Image with .size = (w, h) — preferred measured source.
    w = _attr(params, "width")
    if w is None:
        w = kwargs.get("width")
    h = _attr(params, "height")
    if h is None:
        h = kwargs.get("height")
    model = str(kwargs.get("model") or "")
    image_size = _resolve_image_size(
        request_w=w,
        request_h=h,
        result=result,
        provider="huggingface",
        model=model,
    )
    return {
        "items": {
            "images_generated": 1,
            "image_model": model,
            "image_size": image_size,
        },
        "duration": {},  # filled by wrapper from wall-clock if needed
        "raw": {},
    }

def _intent_hf_audio_tts(args, kwargs):
    text = (args[1] if len(args) > 1 else None) or kwargs.get("text") or ""
    return {
        "kind": "audio_speech",
        "character_count": len(text) if isinstance(text, str) else 0,
    }

def _extract_hf_audio_tts(args, kwargs, result):
    text = (args[1] if len(args) > 1 else None) or kwargs.get("text") or ""
    chars = len(text) if isinstance(text, str) else 0
    return {
        "items": {"tts_characters": chars, "audio_model": str(kwargs.get("model") or "")},
        "duration": {},
        "raw": {},
    }

def _intent_hf_audio_stt(args, kwargs):
    audio = (args[1] if len(args) > 1 else None) or kwargs.get("audio")
    secs = _audio_file_seconds(audio)
    return {"kind": "audio_transcription", "expected_seconds": secs or None}

def _extract_hf_audio_stt(args, kwargs, result):
    audio = (args[1] if len(args) > 1 else None) or kwargs.get("audio")
    secs = _audio_file_seconds(audio)
    return {
        "items": {"audio_model": str(kwargs.get("model") or "")},
        "duration": {"audio_seconds": float(secs or 0)},
        "raw": {},
    }


_MODALITY_HANDLERS = {
    ("openai",        "image_gen"):    {"intent": _intent_openai_images,    "extract": _extract_openai_images},
    ("openai",        "audio_tts"):    {"intent": _intent_openai_audio_tts, "extract": _extract_openai_audio_tts},
    ("openai",        "audio_stt"):    {"intent": _intent_openai_audio_stt, "extract": _extract_openai_audio_stt},
    ("google",        "image_gen"):    {"intent": _intent_google_imagen,    "extract": _extract_google_imagen},
    ("google",        "video_gen"):    {"intent": _intent_google_veo,       "extract": _extract_google_veo},
    ("xai",           "image_gen"):    {"intent": _intent_xai_image,        "extract": _extract_xai_image},
    ("mistral",       "audio_tts"):    {"intent": _intent_mistral_audio_tts,"extract": _extract_mistral_audio_tts},
    ("mistral",       "audio_stt"):    {"intent": _intent_mistral_audio_stt,"extract": _extract_mistral_audio_stt},
    ("mistral",       "ocr"):          {"intent": _intent_mistral_ocr,      "extract": _extract_mistral_ocr},
    ("together",      "image_gen"):    {"intent": _intent_together_image,   "extract": _extract_together_image},
    ("huggingface",   "image_gen"):    {"intent": _intent_hf_image,         "extract": _extract_hf_image},
    ("huggingface",   "audio_tts"):    {"intent": _intent_hf_audio_tts,     "extract": _extract_hf_audio_tts},
    ("huggingface",   "audio_stt"):    {"intent": _intent_hf_audio_stt,     "extract": _extract_hf_audio_stt},
}


@fail_safe
def _log_modality(provider: str, modality: str, shape: str, session, kwargs, args,
                  result, order: int, span_name, start_time, elapsed_seconds: float = 0.0,
                  obs_key=_OBS_KEY_CURRENT):
    """Modality-aware sibling of `_log_manual`. Builds a usage block with the
    explicit usage-shape enum (e.g. ``openai_images``) and items/duration
    derived from the call inputs and response."""
    tp = get_client()
    if not tp:
        return
    handler = _MODALITY_HANDLERS.get((provider, modality))
    if not handler:
        return
    try:
        extracted = handler["extract"](args, kwargs, result) or {}
    except Exception:
        extracted = {}
    items = extracted.get("items") or {}
    duration = extracted.get("duration") or {}
    raw = extracted.get("raw") or {}
    # Fall back to wall-clock for HF text-to-* where the API exposes no timing.
    if elapsed_seconds and not duration.get("audio_seconds") and not duration.get("video_seconds"):
        duration["compute_seconds"] = float(elapsed_seconds)

    model = (
        (kwargs.get("model") if isinstance(kwargs, dict) else None)
        or items.get("image_model")
        or items.get("audio_model")
        or items.get("video_model")
        or "unknown"
    )

    span_obj = {
        **manual_span_ids(session),
        "span_kind": "llm",
        "span_name": span_name or model,
        "span_order": order,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    metadata = {"workflow_name": session.workflow_name}
    if session.session_id:
        metadata["session_id"] = session.session_id
    # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
    # session metadata WITHOUT it, then re-add it only when this row belongs
    # to the call that was actually rerouted (exact obs-key match).
    _copy_session_metadata(metadata, session)
    _stamp_routing_marker(metadata, session, _resolve_obs_key(obs_key))
    metadata["modality"] = modality

    prompt_comp = []
    response_comp = []
    comp_key = f"{session.trace_id}:{order}"
    if hasattr(session, '_pending_compositions'):
        comp_data = session._pending_compositions.pop(comp_key, {})
        prompt_comp = comp_data.get("prompt", [])
        response_comp = comp_data.get("response", [])

    usage_block = {"shape": shape, "raw": raw, "items": items, "duration": duration}

    # Same success-path drain as _log_manual — modality wrappers run
    # preflight and may stash applied local_decision / reject observations.
    # Keyed on ``obs_key`` (threaded from the wrapper's post-check capture).
    pending_local_decision = None
    pending_observations = None
    try:
        _log_obs_key = _resolve_obs_key(obs_key)
        pending_local_decision = _claim_local_decision(session, _log_obs_key)
        pending_observations = _state.drain_observations(_log_obs_key)
    except Exception:
        pending_local_decision = None
        pending_observations = None

    tp.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model=str(model),
        provider=provider,
        metadata=metadata,
        span=span_obj,
        prompt_composition=prompt_comp,
        response_composition=response_comp,
        usage=usage_block,
        # modality (image_gen / audio_tts / audio_stt / video_gen / ocr) is a
        # canonical `operation` value. Passing it segments spend-by-operation
        # correctly AND lets the modality-aware span_kind derive (image/tts/...).
        operation=modality,
        local_decision=pending_local_decision,
        observations=pending_observations or None,
    )


def _build_intent(provider: str, modality: str, args, kwargs):
    """Best-effort intent extraction. Always fail-safe — returns an empty
    fallback so the /check round-trip still happens."""
    if not modality:
        return None
    handler = _MODALITY_HANDLERS.get((provider, modality))
    if not handler:
        return {"kind": modality}
    try:
        intent = handler["intent"](args, kwargs) or {}
    except Exception:
        intent = {"kind": modality}
    if "kind" not in intent:
        intent["kind"] = modality
    return intent


# Providers whose streaming chunks follow the OpenAI ChatCompletionChunk shape
# (`choices[0].delta.{content,tool_calls}`). Manual-telemetry wrappers
# accumulate these chunks themselves; the Mode A stream wrapper also reuses the
# same accumulator (see _wrap_mode_a_sync_stream / _async variant). "openai" is
# included because the native OpenAI SDK also emits this shape — but for it
# (Mode A) the chunks pass through the existing OpenLLMetry instrumentor too;
# we accumulate ONLY to recover response composition, never tokens.
# Cerebras is OpenAI-compatible for stream deltas (choices[0].delta.*); usage
# extraction already handles it — membership here is composition-only.
_OPENAI_SHAPED_STREAM_PROVIDERS = {
    "openai", "huggingface", "litellm", "mistral", "together", "groq", "cerebras",
}


def _chunk_usage(chunk):
    """Return the usage object on a streaming chunk, looking through Mistral's
    ``CompletionEvent { data: CompletionChunk }`` wrapper when present.

    For the OpenAI Responses API, usage arrives ONLY on the
    ``response.completed`` stream event — under ``event.response.usage``.

    xai-sdk streams yield ``(Response, Chunk)`` tuples; usage is on the
    Response's `.usage` and is only filled on the final iteration. Return it
    only when at least one of prompt/completion tokens is non-zero so the
    caller's ``last`` tracker latches the right tuple.
    """
    if isinstance(chunk, tuple) and len(chunk) == 2:
        response_obj = chunk[0]
        u = getattr(response_obj, "usage", None)
        if u is not None and (
            int(getattr(u, "prompt_tokens", 0) or 0)
            or int(getattr(u, "completion_tokens", 0) or 0)
        ):
            return u
        return None
    if getattr(chunk, "type", None) == "response.completed":
        inner = getattr(chunk, "response", None)
        if inner is not None:
            return getattr(inner, "usage", None)
    u = getattr(chunk, "usage", None)
    if u is not None:
        return u
    # Groq streaming: usage is nested under `chunk.x_groq.usage` on the final
    # chunk (top-level `chunk.usage` is absent/None). `x_groq` is a groq-only
    # field, so this unwrap never affects any other provider.
    xg = getattr(chunk, "x_groq", None)
    if xg is not None:
        xu = getattr(xg, "usage", None)
        if xu is not None:
            return xu
    inner = getattr(chunk, "data", None)
    if inner is not None:
        return getattr(inner, "usage", None)
    # Cohere v2 stream: usage arrives only on the `message-end` event, under
    # `.delta.usage` — latch that event so the manual wrapper logs its tokens.
    delta = getattr(chunk, "delta", None)
    if delta is not None:
        return getattr(delta, "usage", None)
    return None


def _new_stream_accumulator(provider: str):
    """Per-stream accumulator for building response composition from streamed
    chunks. Returns None for providers not accumulated here (the wrapper then
    falls back to the last usage chunk for composition)."""
    if provider in _OPENAI_SHAPED_STREAM_PROVIDERS:
        return {"text_parts": [], "tool_calls": {}}
    if provider == "openai_responses":
        # OpenAI Responses API stream — accumulate by output_item index.
        # items[idx] = {"type": "message", "text_parts": [...]} or
        # {"type": "function_call", "name": "...", "call_id": "...",
        # "arg_parts": [...]}
        return {"items": {}, "order": []}
    if provider == "xai":
        # xai-sdk streams yield (Response, Chunk) tuples. The Response side
        # accumulates content/tool_calls server-side and is the authoritative
        # source by the final iteration; track it directly so
        # `_stream_accumulator_to_response` can hand the proto Response back
        # to the xai branch in build_response_composition.
        return {"_final_response": None, "text_parts": []}
    if provider == "cohere":
        # Cohere v2 stream events (content-delta / tool-plan-delta /
        # tool-call-start / tool-call-delta). Node parity 5.
        return {"text_parts": [], "tool_plan_parts": [], "tool_calls": {}}
    return None


def _accumulate_stream_chunk(provider: str, acc, chunk) -> None:
    """Fold one streamed chunk into the accumulator (provider-aware)."""
    if not acc or chunk is None:
        return
    if provider == "openai_responses":
        _accumulate_responses_stream_chunk(acc, chunk)
        return
    if provider == "xai":
        # Each iteration yields (Response, Chunk). Hold the latest Response —
        # it accumulates content/tool_calls server-side — and also stash delta
        # text from Chunk.content as a fallback for synthesizing composition
        # when the Response side returns empty.
        try:
            if isinstance(chunk, tuple) and len(chunk) == 2:
                response_obj, chunk_obj = chunk
                if response_obj is not None:
                    acc["_final_response"] = response_obj
                if chunk_obj is not None:
                    delta = getattr(chunk_obj, "content", None)
                    if isinstance(delta, str) and delta:
                        acc["text_parts"].append(delta)
        except Exception:
            pass  # fail-open: composition is best-effort
        return
    if provider == "cohere":
        # Cohere v2 stream events. Python cohere SDK may expose attrs or dict-
        # like chunks; accept both (and camelCase + snake_case field names,
        # matching Node). Fail-open on any shape drift.
        try:
            etype = getattr(chunk, "type", None)
            if etype is None and isinstance(chunk, dict):
                etype = chunk.get("type")
            delta = getattr(chunk, "delta", None)
            if delta is None and isinstance(chunk, dict):
                delta = chunk.get("delta")
            message = getattr(delta, "message", None) if delta is not None else None
            if message is None and isinstance(delta, dict):
                message = delta.get("message")
            if etype == "content-delta":
                content = None
                if message is not None:
                    content = getattr(message, "content", None)
                    if content is None and isinstance(message, dict):
                        content = message.get("content")
                frag = None
                if content is not None:
                    frag = getattr(content, "text", None)
                    if frag is None and isinstance(content, dict):
                        frag = content.get("text")
                if frag:
                    acc["text_parts"].append(frag)
            elif etype == "tool-plan-delta":
                frag = None
                if message is not None:
                    frag = (
                        getattr(message, "tool_plan", None)
                        or getattr(message, "toolPlan", None)
                    )
                    if frag is None and isinstance(message, dict):
                        frag = message.get("tool_plan") or message.get("toolPlan")
                if frag:
                    acc["tool_plan_parts"].append(frag)
            elif etype == "tool-call-start":
                idx = getattr(chunk, "index", None)
                if idx is None and isinstance(chunk, dict):
                    idx = chunk.get("index")
                if idx is None:
                    idx = 0
                tc = None
                if message is not None:
                    tc = (
                        getattr(message, "tool_calls", None)
                        or getattr(message, "toolCalls", None)
                    )
                    if tc is None and isinstance(message, dict):
                        tc = message.get("tool_calls") or message.get("toolCalls")
                if tc is not None:
                    tc_id = getattr(tc, "id", None)
                    if tc_id is None and isinstance(tc, dict):
                        tc_id = tc.get("id")
                    fn = getattr(tc, "function", None)
                    if fn is None and isinstance(tc, dict):
                        fn = tc.get("function")
                    fn_name = ""
                    fn_args = ""
                    if fn is not None:
                        fn_name = getattr(fn, "name", None) or (
                            fn.get("name") if isinstance(fn, dict) else None
                        ) or ""
                        fn_args = getattr(fn, "arguments", None) or (
                            fn.get("arguments") if isinstance(fn, dict) else None
                        ) or ""
                    acc["tool_calls"][idx] = {
                        "id": tc_id or "",
                        "type": "function",
                        "function": {"name": fn_name, "arguments": fn_args},
                    }
            elif etype == "tool-call-delta":
                idx = getattr(chunk, "index", None)
                if idx is None and isinstance(chunk, dict):
                    idx = chunk.get("index")
                if idx is None:
                    idx = 0
                tcd = None
                if message is not None:
                    tcd = (
                        getattr(message, "tool_calls", None)
                        or getattr(message, "toolCalls", None)
                    )
                    if tcd is None and isinstance(message, dict):
                        tcd = message.get("tool_calls") or message.get("toolCalls")
                frag = None
                if tcd is not None:
                    fn = getattr(tcd, "function", None)
                    if fn is None and isinstance(tcd, dict):
                        fn = tcd.get("function")
                    if fn is not None:
                        frag = getattr(fn, "arguments", None)
                        if frag is None and isinstance(fn, dict):
                            frag = fn.get("arguments")
                if frag and idx in acc["tool_calls"]:
                    acc["tool_calls"][idx]["function"]["arguments"] += frag
        except Exception:
            pass  # fail-open: composition is best-effort
        return
    if provider not in _OPENAI_SHAPED_STREAM_PROVIDERS:
        return
    try:
        # OpenAI-shaped delta chunks: choices[0].delta.content and
        # choices[0].delta.tool_calls (each delta tool call carries an `index`).
        # LiteLLM normalizes Anthropic/Gemini/etc. streams into this shape too.
        # Mistral wraps each chunk as `CompletionEvent { data: CompletionChunk }`
        # — unwrap once when present so the same OpenAI-compat parsing applies.
        inner = getattr(chunk, "data", None)
        if inner is not None and getattr(inner, "choices", None):
            chunk = inner
        choices = getattr(chunk, "choices", None)
        if not choices:
            return
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return
        content = getattr(delta, "content", None)
        # Mistral SDK uses an `Unset` sentinel instead of None — its
        # `__bool__` returns False, so `if content:` correctly filters it.
        if content and isinstance(content, str):
            acc["text_parts"].append(content)
        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                idx = getattr(tc, "index", 0) or 0
                slot = acc["tool_calls"].setdefault(
                    idx, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                )
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    slot["id"] = tc_id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    fn_name = getattr(fn, "name", None)
                    if fn_name:
                        slot["function"]["name"] = fn_name
                    fn_args = getattr(fn, "arguments", None)
                    if fn_args:
                        slot["function"]["arguments"] += fn_args
    except Exception:
        pass  # fail-open: composition is best-effort


def _stream_accumulator_to_response(provider: str, acc):
    """Build a synthetic OpenAI-shaped plain-dict response from the accumulator
    so build_response_composition can parse it. Returns None when nothing was
    accumulated."""
    if not acc:
        return None
    if provider == "openai_responses":
        return _responses_stream_accumulator_to_response(acc)
    if provider == "xai":
        # Prefer the final Response object — it carries .content + .tool_calls
        # natively and the xai branch in composition.py reads them directly.
        final = acc.get("_final_response")
        if final is not None:
            return final
        # Fallback: synthesize an OpenAI-compat dict from the delta text.
        text = "".join(acc.get("text_parts", []))
        if not text:
            return None
        return {"choices": [{"message": {"role": "assistant", "content": text}}]}
    if provider == "cohere":
        # Synthetic Cohere-shaped response object so build_response_composition
        # hits `_parse_cohere_response` (requires `.message` attr, not a plain
        # dict key). Node returns `{ message }`; Python composition uses
        # getattr — use SimpleNamespace for parity with SDK objects.
        message = {}
        text = "".join(acc.get("text_parts", []))
        if text:
            message["content"] = [{"type": "text", "text": text}]
        tool_plan = "".join(acc.get("tool_plan_parts", []))
        if tool_plan:
            message["tool_plan"] = tool_plan
        tool_calls = [acc["tool_calls"][k] for k in sorted(acc["tool_calls"])]
        if tool_calls:
            message["tool_calls"] = tool_calls
        if not message:
            return None
        return _types.SimpleNamespace(message=message)
    if provider not in _OPENAI_SHAPED_STREAM_PROVIDERS:
        return None
    message = {"role": "assistant"}
    text = "".join(acc["text_parts"])
    if text:
        message["content"] = text
    tool_calls = [acc["tool_calls"][k] for k in sorted(acc["tool_calls"])]
    if tool_calls:
        message["tool_calls"] = tool_calls
    if "content" not in message and "tool_calls" not in message:
        return None
    return {"choices": [{"message": message}]}


# ── Latency capture (TTFT + streaming throughput) ──────────────────────────
# The stream wrappers timestamp the first *content* chunk (skipping role-only /
# lifecycle priming chunks) to mark Time-To-First-Token, anchored at the
# monotonic provider-call start threaded in from the wrapper. All deltas use
# the monotonic clock and are clamped non-negative. Every helper is best-effort:
# a failure costs the latency metric, never the customer's stream.

def _stream_acc_has_content(provider: str, acc) -> bool:
    """True once the manual-mode accumulator holds renderable content (assistant
    text or a tool-call delta). Reuses the accumulator→response builder so it
    can never drift from the per-provider parsing. When the provider isn't
    accumulated (acc is None) we cannot introspect, so we best-effort treat the
    first chunk as first content."""
    if acc is None:
        return True
    try:
        return _stream_accumulator_to_response(provider, acc) is not None
    except Exception:
        return False


def _mode_a_acc_has_content(provider: str, acc) -> bool:
    """Mode-A counterpart of _stream_acc_has_content (reuses the Mode-A
    synthetic-response builder)."""
    if acc is None:
        return True
    try:
        return _mode_a_synthetic_response(provider, acc) is not None
    except Exception:
        return False


def _build_stream_latency(req_start_mono, ttft_mono, end_mono):
    """Assemble the SDK latency primitives for a streamed call from monotonic
    timestamps. Returns None on any failure or when the anchor is missing
    (telemetry loss, never a raised error). output_tokens is left None — the
    service prefers its own mapped token count and only falls
    back to this field."""
    try:
        if req_start_mono is None or end_mono is None:
            return None
        total_ms = int(round(max(0.0, (end_mono - req_start_mono) * 1000.0)))
        ttft_ms = None
        generation_ms = None
        if ttft_mono is not None:
            ttft_ms = int(round(max(0.0, (ttft_mono - req_start_mono) * 1000.0)))
            generation_ms = int(round(max(0.0, (end_mono - ttft_mono) * 1000.0)))
        return {
            "is_streaming": True,
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "generation_ms": generation_ms,
            "output_tokens": None,
            "clock": "monotonic",
        }
    except Exception:
        return None


def _build_non_stream_latency(req_start_mono, end_mono):
    """Latency primitives for a non-streaming manual call. Mirrors
    ``_build_stream_latency``'s key set with the streaming-only fields null —
    the first token is not observable separately from the full response, so
    ttft_ms / generation_ms stay None; TTFT is only aggregated for streamed
    rows. Returns None on any failure or
    missing anchor (telemetry loss, never a raised error)."""
    try:
        if req_start_mono is None or end_mono is None:
            return None
        total_ms = int(round(max(0.0, (end_mono - req_start_mono) * 1000.0)))
        return {
            "is_streaming": False,
            "ttft_ms": None,
            "total_ms": total_ms,
            "generation_ms": None,
            "output_tokens": None,
            "clock": "monotonic",
        }
    except Exception:
        return None


def _accumulate_responses_stream_chunk(acc, chunk) -> None:
    """Fold one Responses API stream event into the accumulator.

    Relevant event types (the rest are ignored — they're lifecycle markers):
      response.output_item.added → register the new output item under its
                                        output_index; capture function_call name
                                        and call_id (text items get text_parts).
      response.output_text.delta → append delta text to the matching item.
      response.function_call_arguments.delta
                                      → append delta JSON to the matching
                                        function_call's arg_parts.
      response.completed → ignored here; usage is read separately.
    """
    try:
        etype = getattr(chunk, "type", None) or ""
        if etype == "response.output_item.added":
            item = getattr(chunk, "item", None)
            idx = getattr(chunk, "output_index", None)
            if item is None or idx is None:
                return
            itype = getattr(item, "type", None) or ""
            if itype == "function_call":
                slot = {
                    "type": "function_call",
                    "name": getattr(item, "name", "") or "",
                    "call_id": getattr(item, "call_id", "") or "",
                    "arg_parts": [],
                }
            elif itype == "message":
                slot = {"type": "message", "text_parts": []}
            elif itype == "reasoning":
                slot = {"type": "reasoning"}
            elif itype == "image_generation_call":
                # Part 2: retain id/status/result for composition + children.
                slot = {
                    "type": "image_generation_call",
                    "id": getattr(item, "id", "") or "",
                    "status": getattr(item, "status", "") or "",
                    "result": getattr(item, "result", None),
                }
            else:
                slot = {"type": itype}
            acc["items"][idx] = slot
            if idx not in acc["order"]:
                acc["order"].append(idx)
            return
        # Full image item (incl. b64 result) often arrives on output_item.done.
        if etype == "response.output_item.done":
            idx = getattr(chunk, "output_index", None)
            item = getattr(chunk, "item", None)
            if idx is None or item is None:
                return
            if getattr(item, "type", None) == "image_generation_call":
                acc["items"][idx] = {
                    "type": "image_generation_call",
                    "id": getattr(item, "id", "") or "",
                    "status": getattr(item, "status", "") or "completed",
                    "result": getattr(item, "result", None),
                }
                if idx not in acc["order"]:
                    acc["order"].append(idx)
            return
        if etype == "response.output_text.delta":
            idx = getattr(chunk, "output_index", None)
            delta = getattr(chunk, "delta", None)
            if idx is None or not isinstance(delta, str) or not delta:
                return
            slot = acc["items"].setdefault(idx, {"type": "message", "text_parts": []})
            slot.setdefault("text_parts", []).append(delta)
            return
        if etype == "response.function_call_arguments.delta":
            idx = getattr(chunk, "output_index", None)
            delta = getattr(chunk, "delta", None)
            if idx is None or not isinstance(delta, str) or not delta:
                return
            slot = acc["items"].setdefault(
                idx, {"type": "function_call", "name": "", "call_id": "", "arg_parts": []}
            )
            slot.setdefault("arg_parts", []).append(delta)
            return
    except Exception:
        pass  # fail-open: composition is best-effort


def _responses_stream_accumulator_to_response(acc):
    """Build a synthetic Responses-shaped object that
    _parse_openai_responses_response in composition.py can consume."""
    out_items = []
    for idx in acc.get("order", []) or sorted(acc.get("items", {}).keys()):
        slot = acc["items"].get(idx)
        if not slot:
            continue
        st = slot.get("type")
        if st == "message":
            text = "".join(slot.get("text_parts", []))
            if text:
                out_items.append({
                    "type": "message",
                    "content": [{"type": "output_text", "text": text}],
                })
        elif st == "function_call":
            out_items.append({
                "type": "function_call",
                "name": slot.get("name", ""),
                "call_id": slot.get("call_id", ""),
                "arguments": "".join(slot.get("arg_parts", [])),
            })
        elif st == "reasoning":
            out_items.append({"type": "reasoning"})
        elif st == "image_generation_call":
            # Part 2: surface built-in image tool output on stream path.
            out_items.append({
                "type": "image_generation_call",
                "id": slot.get("id", ""),
                "status": slot.get("status") or "completed",
                "result": slot.get("result"),
            })
    if not out_items:
        return None
    return {"output": out_items}


# ─── Mode A stream accumulation (response composition only) ────────────────
# The native OpenAI / Anthropic / Google streaming responses don't have their
# `gen_ai.completion.*` span attributes populated by OpenLLMetry until AFTER
# the customer fully drains the iterator — but TokenPoliceSpanProcessor.on_end
# may fire either before our enforcer wrapper has a chance to capture response
# composition (non-streaming runs through that path fine; streaming doesn't).
# To recover the assistant text + tool_calls for the streaming case we wrap the
# returned iterator, accumulate chunks per provider, and stash a synthetic
# response on the session before letting `_flush_deferred_spans` dispatch the
# queued payload. Token usage continues to come from the OpenLLMetry span
# attributes — these helpers only build response composition.


def _mode_a_new_accumulator(provider: str):
    """Per-stream accumulator state for Mode A providers."""
    if provider == "openai":
        return {"text_parts": [], "tool_calls": {}}
    if provider == "openai_responses":
        # Reuse the manual-mode Responses-API accumulator shape (items by
        # output_index). The Mode-A wrapper rebuilds a synthetic response via
        # _responses_stream_accumulator_to_response below.
        return {"items": {}, "order": []}
    if provider == "anthropic":
        # blocks[idx] = {"type": "text", "text_parts": [...]} or
        # {"type": "tool_use", "id": "...", "name": "...", "input_json_parts": [...]}
        return {"blocks": {}}
    if provider == "google":
        return {"text_parts": [], "function_calls": []}
    return None


def _mode_a_accumulate(provider: str, acc, chunk) -> None:
    """Fold one streamed chunk into the Mode A accumulator."""
    if not acc or chunk is None:
        return
    try:
        if provider == "openai":
            # Reuse the existing OpenAI-shaped accumulator (LiteLLM/HuggingFace
            # use the same logic).
            _accumulate_stream_chunk(provider, acc, chunk)
            return

        if provider == "openai_responses":
            _accumulate_responses_stream_chunk(acc, chunk)
            return

        if provider == "anthropic":
            # Anthropic stream events:
            # ContentBlockStartEvent: index, content_block.{type, name?, id?}
            # ContentBlockDeltaEvent: index, delta.{type: "text_delta", text}
            # or {type: "input_json_delta", partial_json}
            etype = getattr(chunk, "type", None)
            if etype == "content_block_start":
                idx = getattr(chunk, "index", 0)
                cb = getattr(chunk, "content_block", None)
                cb_type = getattr(cb, "type", "text")
                if cb_type == "text":
                    acc["blocks"][idx] = {"type": "text", "text_parts": []}
                elif cb_type == "tool_use":
                    acc["blocks"][idx] = {
                        "type": "tool_use",
                        "id": getattr(cb, "id", ""),
                        "name": getattr(cb, "name", ""),
                        "input_json_parts": [],
                    }
                else:
                    acc["blocks"][idx] = {"type": cb_type or "text", "text_parts": []}
            elif etype == "content_block_delta":
                idx = getattr(chunk, "index", 0)
                delta = getattr(chunk, "delta", None)
                dtype = getattr(delta, "type", None)
                slot = acc["blocks"].setdefault(idx, {"type": "text", "text_parts": []})
                if dtype == "text_delta":
                    txt = getattr(delta, "text", "") or ""
                    if txt:
                        slot.setdefault("text_parts", []).append(txt)
                elif dtype == "input_json_delta":
                    pj = getattr(delta, "partial_json", "") or ""
                    if pj:
                        slot.setdefault("input_json_parts", []).append(pj)
            return

        if provider == "google":
            # google-genai stream chunks expose `.text` (concat of all text
            # parts) and `.candidates[0].content.parts` with text or
            # function_call. Walking the parts gives us tool_call entries too.
            candidates = getattr(chunk, "candidates", None) or []
            for cand in candidates:
                content = getattr(cand, "content", None)
                if content is None:
                    continue
                for part in getattr(content, "parts", None) or []:
                    txt = getattr(part, "text", None)
                    if isinstance(txt, str) and txt:
                        acc["text_parts"].append(txt)
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        try:
                            args = dict(getattr(fc, "args", {}) or {})
                        except Exception:
                            args = {}
                        acc["function_calls"].append(
                            {"name": getattr(fc, "name", "") or "", "args": args}
                        )
            return
    except Exception:
        pass  # fail-open: composition is best-effort


def _mode_a_synthetic_response(provider: str, acc):
    """Build a synthetic response object that build_response_composition can
    parse for this provider. Returns None when nothing was accumulated."""
    if not acc:
        return None
    try:
        if provider == "openai":
            return _stream_accumulator_to_response("openai", acc)

        if provider == "openai_responses":
            # The Mode-A wrapper hands the synthetic response to
            # _capture_response_composition with provider="openai_responses",
            # which routes through build_response_composition →
            # _parse_openai_responses_response. Build the dict shape that
            # parser consumes.
            return _responses_stream_accumulator_to_response(acc)

        if provider == "anthropic":
            # Anthropic parser does `for block in response.content:` with
            # getattr-based field access, so emit a SimpleNamespace per block.
            blocks_out = []
            for idx in sorted(acc["blocks"].keys()):
                b = acc["blocks"][idx]
                if b["type"] == "text":
                    text = "".join(b.get("text_parts", []))
                    if text:
                        blocks_out.append(_types.SimpleNamespace(type="text", text=text))
                elif b["type"] == "tool_use":
                    inp_raw = "".join(b.get("input_json_parts", []))
                    try:
                        import json as _j
                        inp = _j.loads(inp_raw) if inp_raw else {}
                    except Exception:
                        inp = {}
                    blocks_out.append(_types.SimpleNamespace(
                        type="tool_use",
                        id=b.get("id", ""),
                        name=b.get("name", ""),
                        input=inp,
                    ))
            if not blocks_out:
                return None
            return _types.SimpleNamespace(content=blocks_out)

        if provider == "google":
            parts = []
            text = "".join(acc.get("text_parts", []))
            if text:
                parts.append(_types.SimpleNamespace(text=text, function_call=None))
            for fc in acc.get("function_calls", []):
                parts.append(_types.SimpleNamespace(
                    text=None,
                    function_call=_types.SimpleNamespace(
                        name=fc.get("name", ""), args=fc.get("args", {}),
                    ),
                ))
            if not parts:
                return None
            return _types.SimpleNamespace(
                candidates=[_types.SimpleNamespace(
                    content=_types.SimpleNamespace(parts=parts),
                )]
            )
    except Exception:
        return None
    return None


# ── Stream include_usage injection ──
# An OpenAI-wire chat-completions stream opened WITHOUT
# stream_options.include_usage carries no usage chunk → the call's cost is
# silently lost. Inject the option (openai SDK modules only — custom base_urls
# like MiniMax/Groq/vLLM ride the same client; native-protocol SDKs already
# carry usage on their terminal events) and strip the synthetic usage-only
# terminal chunk from the customer's iterator so their code sees exactly the
# chunks it asked for. Disable with capture_stream_usage=False /
# TP_CAPTURE_STREAM_USAGE=0.

_INJECT_ABSENT = object()  # sentinel: stream_options key did not exist


def _inject_stream_usage_option(module_path: str, kwargs: dict):
    """Injects include_usage when appropriate. Returns a restore token
    (truthy) when injected, else None. Fail-safe."""
    try:
        if not str(module_path or "").startswith("openai.resources"):
            return None
        return _do_inject_stream_usage(kwargs)
    except Exception:
        return None


# Manual-telemetry providers whose chat-stream path gets the include_usage
# injection. Together (G5-1): the API is OpenAI-wire and honors the option;
# without it, whether a stream carries a usage payload AT ALL is
# replica-dependent on Together's side (some serving backends attach usage to
# the final chunk unasked, others omit it entirely), so the streamed row was
# silently lost 30–50% of the time. Deliberately narrow — other manual
# providers (groq/cerebras/mistral/cohere) deliver stream usage on their own
# terminal events without an opt-in.
_STREAM_USAGE_INJECT_PROVIDERS = ("together",)


def _inject_stream_usage_option_for_provider(provider: str, kwargs: dict):
    """Manual-path twin of :func:`_inject_stream_usage_option`, keyed on the
    resolved provider slug (the manual wrapper closure has no module_path).

    Injects via ``extra_body`` — NOT a top-level ``stream_options`` kwarg:
    together's Stainless-typed ``create(*, messages, model, ...)`` has no
    ``stream_options`` parameter, so a top-level kwarg raises TypeError before
    any HTTP; ``extra_body`` is the SDK's own documented channel for
    additional API parameters and merges into the wire body. Same customer
    opt-in / stream gates otherwise; fail-safe."""
    try:
        if (provider or "").lower() not in _STREAM_USAGE_INJECT_PROVIDERS:
            return None
        tp = get_client()
        if not tp or not getattr(tp, "capture_stream_usage", False):
            return None
        if kwargs.get("stream") is not True or "messages" not in kwargs:
            return None
        prior = kwargs.get("extra_body", _INJECT_ABSENT)
        prior_so = prior.get("stream_options") if isinstance(prior, dict) else None
        if isinstance(prior_so, dict) and prior_so.get("include_usage") is True:
            return None  # customer asked — the usage chunk is theirs
        merged = dict(prior) if isinstance(prior, dict) else {}
        so = dict(prior_so) if isinstance(prior_so, dict) else {}
        so["include_usage"] = True
        merged["stream_options"] = so
        kwargs["extra_body"] = merged
        return ("prior_extra_body", prior)
    except Exception:
        return None


def _do_inject_stream_usage(kwargs: dict):
    """Shared injection core: opt the request into stream usage unless the
    customer already did. Returns a restore token (truthy) when injected."""
    try:
        tp = get_client()
        if not tp or not getattr(tp, "capture_stream_usage", False):
            return None
        if kwargs.get("stream") is not True or "messages" not in kwargs:
            return None
        prior = kwargs.get("stream_options", _INJECT_ABSENT)
        if isinstance(prior, dict) and prior.get("include_usage") is True:
            return None  # customer asked — the usage chunk is theirs
        merged = dict(prior) if isinstance(prior, dict) else {}
        merged["include_usage"] = True
        kwargs["stream_options"] = merged
        return ("prior", prior)
    except Exception:
        return None


def _restore_stream_usage_option(kwargs: dict, token):
    """Undoes _inject_stream_usage_option / the manual-path extra_body twin
    (for the 4xx retry net). Fail-safe."""
    try:
        if not token:
            return
        key = "extra_body" if token[0] == "prior_extra_body" else "stream_options"
        prior = token[1]
        if prior is _INJECT_ABSENT:
            kwargs.pop(key, None)
        else:
            kwargs[key] = prior
    except Exception:
        pass


def _should_retry_without_injection(exc) -> bool:
    """Strip-and-retry only when the rejection is plausibly caused by the
    injected stream_options.include_usage — a strict-compat server rejecting an
    unknown param answers 400/422 or names the param in its message. A
    rate-limit (429) or auth failure (401/403), and any other status, is never
    caused by the injection and must never trigger a second provider request.

    Fully guarded: a hostile error object cannot make this raise, and a failure
    here yields False (no retry → the provider's own error propagates, which is
    the customer's error, not an SDK failure)."""
    try:
        status = getattr(exc, "status_code", None)
        if status is None:
            status = getattr(getattr(exc, "response", None), "status_code", None)
        try:
            status = int(status) if status is not None else None
        except Exception:
            status = None
        if status in (400, 422):
            return True
        try:
            msg = str(exc).lower()
        except Exception:
            msg = ""
        # "extra_body": the manual-path twin injects via extra_body; a
        # (hypothetical) together build without that parameter raises
        # TypeError naming it — strip-and-retry so the injection can never
        # fail a call that would otherwise have succeeded (golden rule).
        return "stream_options" in msg or "include_usage" in msg or "extra_body" in msg
    except Exception:
        return False


def _is_usage_only_chunk(chunk) -> bool:
    """A usage-only terminal chunk (usage set, empty choices) — what
    include_usage appends to an OpenAI-wire chat stream."""
    try:
        if isinstance(chunk, dict):
            usage, choices = chunk.get("usage"), chunk.get("choices")
        else:
            usage = getattr(chunk, "usage", None)
            choices = getattr(chunk, "choices", None)
        return usage is not None and isinstance(choices, (list, tuple)) and len(choices) == 0
    except Exception:
        return False


def _tp_maybe_await(res):
    """If ``res`` is awaitable, await it; otherwise return it. Used so the async
    close-through can drive either a coroutine ``close``/``aclose`` (openai
    ``AsyncStream.close`` is a coroutine) or a plain sync ``close``."""
    if hasattr(res, "__await__"):
        return res  # caller awaits
    return None


class _ModeAStreamProxy:
    """Sync Mode-A stream proxy that restores the provider stream's
    surface a bare metering generator would drop — the context-manager protocol
    (``with stream as s``), ``.response`` / arbitrary provider attributes, and a
    ``close()`` that actually closes the underlying HTTP connection.

    Iteration is driven through the retained metering generator ``_tp_gen``
    (the original generator body, statement-for-statement), so every
    per-chunk tap / ``suppress_usage_chunk`` strip / composition / flush behavior
    is preserved. Unknown attributes delegate to the wrapped provider stream via
    ``__getattr__``; the CM/iterator dunders are defined explicitly on the
    type (``__getattr__`` alone does not restore dunders). On early
    abandonment / CM exit / explicit ``close()`` we close-through to both the
    underlying provider stream (fixes the connection leak) and the metering
    generator (its ``GeneratorExit``-driven ``finally`` still flushes partial
    telemetry) — each step in its own ``try/except`` (fail-open: nothing may
    escape into customer code). Mirrors ``_AnthropicMessageStreamProxy`` and
    the Node SDK's equivalent close-through on abandonment.
    """

    __slots__ = ("_tp_stream", "_tp_gen")

    def __init__(self, stream, gen):
        object.__setattr__(self, "_tp_stream", stream)
        object.__setattr__(self, "_tp_gen", gen)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tp_stream"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_tp_stream"), name, value)

    def __iter__(self):
        return object.__getattribute__(self, "_tp_gen")

    def __next__(self):
        return next(object.__getattribute__(self, "_tp_gen"))

    def _tp_close_through(self):
        # Close the metering generator first so its finally (partial-telemetry
        # flush on abandonment) still runs, then the underlying provider (leak
        # fix). Each step is independently fail-open — a raise from either must
        # never reach customer code. GeneratorExit from _gen.close()
        # hits the metering generator's `yield`; its `except Exception` does not
        # catch it, so its `finally` → _flush_deferred_spans still fires.
        try:
            object.__getattribute__(self, "_tp_gen").close()
        except Exception:
            pass
        try:
            stream = object.__getattribute__(self, "_tp_stream")
            close = getattr(stream, "close", None)
            if callable(close):
                close()
        except Exception:
            pass

    def close(self):
        self._tp_close_through()

    def __enter__(self):
        # Return the proxy (not the underlying) so iteration inside the `with`
        # block still runs through the metering generator.
        return self

    def __exit__(self, exc_type, exc, tb):
        self._tp_close_through()
        return False


class _ModeAAsyncStreamProxy:
    """Async twin of ``_ModeAStreamProxy`` (async Mode-A streaming).

    Restores the async provider stream surface — ``async with``, ``.response`` /
    arbitrary attributes, ``aclose()``. Iteration drives the retained async
    metering generator ``_tp_gen`` (original body verbatim, including the
    sync-underlying/async-consumer bridge for a stream method that may be
    registered sync but iterated with `async for`). Close-through awaits the
    underlying's ``aclose``/``close`` (openai ``AsyncStream.close`` is a
    coroutine) and drives ``_tp_gen.aclose()`` so the metering generator's
    ``finally`` still flushes on abandonment — every step fail-open (nothing
    may escape into customer code).
    """

    __slots__ = ("_tp_stream", "_tp_gen")

    def __init__(self, stream, gen):
        object.__setattr__(self, "_tp_stream", stream)
        object.__setattr__(self, "_tp_gen", gen)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tp_stream"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_tp_stream"), name, value)

    def __aiter__(self):
        return object.__getattribute__(self, "_tp_gen")

    def __anext__(self):
        return object.__getattribute__(self, "_tp_gen").__anext__()

    async def _tp_aclose_through(self):
        # Metering generator first (its finally flushes partial telemetry on
        # abandonment), then the underlying provider (leak fix). Each step
        # fail-open — nothing escapes into customer code.
        try:
            await object.__getattribute__(self, "_tp_gen").aclose()
        except Exception:
            pass
        try:
            stream = object.__getattribute__(self, "_tp_stream")
            aclose = getattr(stream, "aclose", None)
            if callable(aclose):
                awaitable = _tp_maybe_await(aclose())
                if awaitable is not None:
                    await awaitable
            else:
                close = getattr(stream, "close", None)
                if callable(close):
                    awaitable = _tp_maybe_await(close())
                    if awaitable is not None:
                        await awaitable
        except Exception:
            pass

    async def aclose(self):
        await self._tp_aclose_through()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._tp_aclose_through()
        return False


def _serialize_openai_stream_usage(usage):
    """Serialize the verbatim final-chunk ``usage`` from an OpenAI-wire stream.

    The OpenLLMetry openai instrumentor never sets reasoning-token attrs on
    STREAMED spans (it does on non-streamed), so the Mode A deferred payload's
    OTel-synth ``usage.raw`` silently drops
    ``completion_tokens_details.reasoning_tokens``. On xAI ``completion_tokens``
    EXCLUDES reasoning, so the loss is a silent undercharge. The stream
    wrappers already see the provider's verbatim usage chunk — forward it
    through the existing ``usage_raw`` merge seam in ``_flush_deferred_spans``.
    No renaming/netting/re-derivation: the collector's shape mappers expect the
    wire dict verbatim (nested ``completion_tokens_details`` /
    ``prompt_tokens_details``, provider extras like xAI's
    ``cost_in_usd_ticks``). Returns None unless the serialized dict is
    non-empty with a positive prompt or completion count; fail-open (any
    failure returns None, never raises)."""
    try:
        serialized = _json.loads(_json.dumps(
            _as_dict(usage) if not isinstance(usage, dict) else usage,
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
        ))
        if not (isinstance(serialized, dict) and serialized):
            return None
        if (int(serialized.get("prompt_tokens") or 0) > 0
                or int(serialized.get("completion_tokens") or 0) > 0):
            return serialized
        return None
    except Exception:
        return None


def _serialize_google_stream_usage(usage):
    """Serialize the verbatim ``usage_metadata`` from a google-genai stream
    chunk (G3-O1). Google puts the COMPLETE usage on the terminal chunk, so
    the tap keeps last-write-wins — never accumulated or de-cumulated.
    Same forwarding rationale as ``_serialize_openai_stream_usage`` above:
    the instrumentor's streamed attrs drop ``thoughts_token_count`` /
    ``cached_content_token_count``, and Google's ``candidates_token_count``
    EXCLUDES thoughts so thinking tokens are otherwise unbilled. Returns None
    unless the serialized dict is non-empty with a positive prompt or
    candidates count (a zero block must never clobber good synth counts);
    fail-open (any failure returns None, never raises)."""
    try:
        serialized = _json.loads(_json.dumps(
            _as_dict(usage) if not isinstance(usage, dict) else usage,
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
        ))
        if not (isinstance(serialized, dict) and serialized):
            return None
        if (int(serialized.get("prompt_token_count") or 0) > 0
                or int(serialized.get("candidates_token_count") or 0) > 0):
            return serialized
        return None
    except Exception:
        return None


def _serialize_mode_a_stream_usage(provider: str, usage):
    """Route the tapped stream usage to its provider serializer, or None."""
    try:
        if usage is None:
            return None
        if provider == "openai":
            return _serialize_openai_stream_usage(usage)
        if provider in ("google", "gemini"):
            return _serialize_google_stream_usage(usage)
        return None
    except Exception:
        return None


def _wrap_mode_a_sync_stream(stream, provider: str, session, prev_defer: bool,
                             req_start_mono=None, suppress_usage_chunk: bool = False,
                             order: int = None, obs_key=None):
    """Wrap a Mode A sync stream so the deferred span carries response
    composition. Keeps `_defer_telemetry=True` during iteration; on stream end
    builds the synthetic response, captures composition, and flushes.

    Iteration errors (network drop, mid-stream 5xx, malformed chunk) are
    classified into a `call_outcome` and dispatched via `_emit_call_failure_log`
    so the failure is recorded as a failed-call row. The original
    exception is re-raised so customer code sees it verbatim.

    ``req_start_mono`` anchors TTFT/total latency to the provider-call start; the
    captured metrics are stashed on the session and attached to the first
    deferred span by ``_flush_deferred_spans``.

    Returns a ``_ModeAStreamProxy`` wrapping the underlying provider
    stream + this metering generator so the customer keeps the provider's
    surface (CM protocol, ``.response``, ``close()``) and abandonment closes the
    underlying connection. The metering generator body below is unchanged
    (only wrapped in ``_gen`` and driven by the proxy)."""
    def _gen():
        acc = _mode_a_new_accumulator(provider)
        session._defer_telemetry = True
        _stream_start = _time.monotonic()
        _ttft_mono = None
        _stream_failed = False
        _svc_tier = ""
        _stream_usage = None
        try:
            for chunk in stream:
                try:
                    _mode_a_accumulate(provider, acc, chunk)
                    if _ttft_mono is None and _mode_a_acc_has_content(provider, acc):
                        _ttft_mono = _time.monotonic()
                    if not _svc_tier:
                        _svc_tier = _extract_service_tier(chunk)
                    # G3-14-1: keep a reference to the latest chunk usage
                    # (last-write-wins — covers a usage-only terminal chunk AND
                    # providers that attach usage to the last content chunk;
                    # interim cumulative usage is overwritten by the final one).
                    # Gated to the openai wire shape: Anthropic chunk
                    # usage has different/cumulative semantics. Serialized once
                    # at clean stream end, never per chunk.
                    if provider == "openai":
                        u = getattr(chunk, "usage", None)
                        if u is None and isinstance(chunk, dict):
                            u = chunk.get("usage")
                        if u is not None:
                            _stream_usage = u
                    # G3-O1: google-genai puts the COMPLETE usage_metadata on
                    # the terminal chunk — last-write-wins here too, never
                    # accumulate/de-cumulate.
                    elif provider in ("google", "gemini"):
                        u = getattr(chunk, "usage_metadata", None)
                        if u is None and isinstance(chunk, dict):
                            u = chunk.get("usage_metadata") or chunk.get("usageMetadata")
                        if u is not None:
                            _stream_usage = u
                except Exception:
                    pass  # fail-open: tap failure never breaks the stream
                # SDK-injected include_usage → strip the synthetic usage-only
                # terminal chunk; the customer sees exactly what they asked for.
                if suppress_usage_chunk and _is_usage_only_chunk(chunk):
                    continue
                yield chunk
        except Exception as _exc:
            _stream_failed = True
            elapsed_ms = int((_time.monotonic() - _stream_start) * 1000)
            try:
                session._call_outcome = build_call_outcome(_exc, elapsed_ms)
            except Exception:
                pass
            session._defer_telemetry = prev_defer
            # ``obs_key`` was captured by the wrapper right after this call's
            # check — the drain here runs at consumer-iteration time, when the
            # contextvar may already hold a LATER call's key.
            _emit_call_failure_log(get_client(), session, obs_key=obs_key)
            raise
        finally:
            session._defer_telemetry = prev_defer
            if not _stream_failed:
                try:
                    session._latency_metrics = _build_stream_latency(
                        req_start_mono, _ttft_mono, _time.monotonic())
                except Exception:
                    pass
                try:
                    synthetic = _mode_a_synthetic_response(provider, acc)
                    # G3-14-1 / G3-O1: streamed spans get no reasoning/thoughts
                    # attrs from the instrumentor — forward the verbatim
                    # final-chunk usage via the usage_raw merge seam. Must not
                    # be gated on `synthetic` (a parts-less google turn returns
                    # None but the terminal chunk's usage still needs to land).
                    _usage_raw = _serialize_mode_a_stream_usage(provider, _stream_usage)
                    if synthetic is not None or _usage_raw:
                        # The order was reserved before the provider call (see
                        # tests/test_concurrent_attribution.py) so this stream's
                        # response keys onto its own span even under concurrency;
                        # order=None keeps the legacy mailbox/-1 fallback.
                        _capture_response_composition(
                            provider, synthetic, service_tier=_svc_tier, order=order,
                            usage_raw=_usage_raw)
                except Exception:
                    pass
                try:
                    _flush_deferred_spans(session, obs_key=obs_key)
                except Exception:
                    pass
    return _ModeAStreamProxy(stream, _gen())


def _wrap_mode_a_async_stream(stream, provider: str, session, prev_defer: bool,
                              req_start_mono=None, suppress_usage_chunk: bool = False,
                              order: int = None, obs_key=None):
    """Async variant of _wrap_mode_a_sync_stream.

    Bridges a sync underlying iterator: a provider's async client may hand back
    a plain sync generator, but the consumer iterates with `async for`,
    so we present an async generator and drive whichever protocol the underlying
    actually exposes."""
    async def _agen():
        acc = _mode_a_new_accumulator(provider)
        session._defer_telemetry = True
        _stream_start = _time.monotonic()
        _ttft_mono = None
        _stream_failed = False
        _svc_tier = ""
        _stream_usage = None

        def _tap(chunk):
            nonlocal _ttft_mono, _svc_tier, _stream_usage
            try:
                _mode_a_accumulate(provider, acc, chunk)
                if _ttft_mono is None and _mode_a_acc_has_content(provider, acc):
                    _ttft_mono = _time.monotonic()
                if not _svc_tier:
                    _svc_tier = _extract_service_tier(chunk)
                # G3-14-1: latest chunk usage, last-write-wins; openai wire
                # shape only — see the sync variant above for the rationale.
                if provider == "openai":
                    u = getattr(chunk, "usage", None)
                    if u is None and isinstance(chunk, dict):
                        u = chunk.get("usage")
                    if u is not None:
                        _stream_usage = u
                # G3-O1: google terminal-chunk usage_metadata, last-write-wins
                # — see the sync variant above.
                elif provider in ("google", "gemini"):
                    u = getattr(chunk, "usage_metadata", None)
                    if u is None and isinstance(chunk, dict):
                        u = chunk.get("usage_metadata") or chunk.get("usageMetadata")
                    if u is not None:
                        _stream_usage = u
            except Exception:
                pass  # fail-open: tap failure never breaks the stream

        try:
            if hasattr(stream, "__aiter__") or is_async_iterator(stream):
                async for chunk in stream:
                    _tap(chunk)
                    if suppress_usage_chunk and _is_usage_only_chunk(chunk):
                        continue
                    yield chunk
            else:
                for chunk in stream:  # sync underlying, async consumer
                    _tap(chunk)
                    if suppress_usage_chunk and _is_usage_only_chunk(chunk):
                        continue
                    yield chunk
        except Exception as _exc:
            _stream_failed = True
            elapsed_ms = int((_time.monotonic() - _stream_start) * 1000)
            try:
                session._call_outcome = build_call_outcome(_exc, elapsed_ms)
            except Exception:
                pass
            session._defer_telemetry = prev_defer
            # Captured-at-wrapper obs key — see the sync variant above.
            _emit_call_failure_log(get_client(), session, obs_key=obs_key)
            raise
        finally:
            session._defer_telemetry = prev_defer
            if not _stream_failed:
                try:
                    session._latency_metrics = _build_stream_latency(
                        req_start_mono, _ttft_mono, _time.monotonic())
                except Exception:
                    pass
                try:
                    synthetic = _mode_a_synthetic_response(provider, acc)
                    # G3-14-1 / G3-O1 usage_raw forwarding — see the sync
                    # variant above (not gated on `synthetic`).
                    _usage_raw = _serialize_mode_a_stream_usage(provider, _stream_usage)
                    if synthetic is not None or _usage_raw:
                        # Reserved-order attribution (see the sync variant above /
                        # tests/test_concurrent_attribution.py); order=None keeps
                        # the legacy mailbox/-1 fallback.
                        _capture_response_composition(
                            provider, synthetic, service_tier=_svc_tier, order=order,
                            usage_raw=_usage_raw)
                except Exception:
                    pass
                try:
                    _flush_deferred_spans(session, obs_key=obs_key)
                except Exception:
                    pass
    return _ModeAAsyncStreamProxy(stream, _agen())


def _suppress_chunk_safe(chunk) -> bool:
    """Guarded :func:`_is_usage_only_chunk` — a hostile chunk object must
    never break the customer's iteration (used by the suppress paths)."""
    try:
        return _is_usage_only_chunk(chunk)
    except Exception:
        return False


def _approximate_tokens_from_chars(text) -> int:
    """Best-effort chars→tokens estimate (len // 4, the canonical
    order-of-magnitude rule the embedding approximations already use). Used by
    the G5-1 no-usage-chunk stream fallback; always paired with
    ``approximated: True`` in the raw usage so the collector reports
    cost_status='approximated', never an exact 'measured'. Fail-safe."""
    try:
        length = len(text) if isinstance(text, str) else 0
        return -(-length // 4) if length > 0 else 0  # ceil division
    except Exception:
        return 0


def _log_stream_no_usage_fallback(provider, session, kwargs, acc, order, span_name,
                                  start_time, latency, obs_key=None):
    """G5-1 fallback: the stream drained cleanly but carried NO usage payload
    at all (replica-dependent on Together, and still reachable with injection
    on: capture_stream_usage=False, the strip-and-retry path, or a replica
    that accepts the option and omits the chunk anyway). Losing the row
    silently is the worst outcome — a customer under-sees real spend with no
    signal anywhere. Log the row with token counts approximated from the
    request messages + accumulated response (chars/4) and
    ``approximated: True`` in raw usage, so the collector prices it as
    cost_status='approximated' — never a fake exact 'measured', never a
    silent zero. Gated to together by the caller: other manual providers'
    usage extraction is not OpenAI-shaped, and only together has evidenced
    this omission class. Fail-safe throughout (caller swallows)."""
    import json as _json
    response_for_comp = _stream_accumulator_to_response(provider, acc)
    if response_for_comp is not None:
        _capture_composition_at(provider, response_for_comp, order, True)
    try:
        est_in = _approximate_tokens_from_chars(
            _json.dumps(kwargs.get("messages", ""), default=str))
    except Exception:
        est_in = 0
    est_out = 0
    try:
        choices = _attr(response_for_comp, "choices") or []
        message = _attr(choices[0], "message") if choices else None
        if message is not None:
            est_out = _approximate_tokens_from_chars(
                _json.dumps(message, default=str))
    except Exception:
        est_out = 0
    synthetic = {
        "model": kwargs.get("model"),
        "usage": {
            "prompt_tokens": est_in,
            "completion_tokens": est_out,
            "approximated": True,
        },
    }
    _log_manual(provider, session, kwargs, synthetic, order, span_name,
                start_time, latency=latency, obs_key=obs_key)


def _wrap_sync_stream(stream, provider, session, kwargs, order, span_name, start_time,
                      framework=None, req_start_mono=None, obs_key=None,
                      suppress_usage_chunk: bool = False, route_ctx=None):
    """Wrap a sync streaming response — log the final chunk's usage on completion.

    Response composition is accumulated from the streamed chunks (the final
    usage chunk alone does not carry the assistant message content).

    When ``framework='litellm'`` the wrapper holds the ``_in_litellm`` guard
    across iteration so nested provider wrappers (driven lazily as chunks are
    pulled from the underlying SDK) stay inert.

    ``req_start_mono`` is the monotonic timestamp taken right before the provider
    call (threaded from the enforcing wrapper). It anchors TTFT/total latency to
    true request initiation rather than the lazy first-pull of this generator.
    """
    def _gen():
        last = None
        acc = _new_stream_accumulator(provider)
        if framework == "litellm":
            _in_litellm.set(True)
        _stream_start = _time.monotonic()
        _ttft_mono = None
        _stream_failed = False
        try:
            for chunk in stream:
                # Tap is best-effort: a malformed chunk (provider version drift
                # in chunk shape) must cost telemetry, never abort the customer's
                # iteration. Only a genuine error from the underlying `stream`
                # itself reaches the except branch below and re-raises.
                try:
                    if _chunk_usage(chunk):
                        last = chunk
                    _accumulate_stream_chunk(provider, acc, chunk)
                    # Mark TTFT at the first chunk carrying renderable content.
                    if _ttft_mono is None and _stream_acc_has_content(provider, acc):
                        _ttft_mono = _time.monotonic()
                except Exception:
                    pass  # fail-open: tap failure never breaks the stream
                # The SDK injected include_usage (the customer didn't ask) —
                # strip the synthetic usage-only terminal chunk after tapping
                # it; a usage payload riding a CONTENT chunk is never
                # stripped. Guarded: a check failure yields the chunk.
                if suppress_usage_chunk:
                    try:
                        if _is_usage_only_chunk(chunk):
                            continue
                    except Exception:
                        pass
                yield chunk
        except Exception as _exc:
            _stream_failed = True
            elapsed_ms = int((_time.monotonic() - _stream_start) * 1000)
            try:
                session._call_outcome = build_call_outcome(_exc, elapsed_ms)
            except Exception:
                pass
            # Captured-at-wrapper obs key: this drain runs at consumer-
            # iteration time, when the contextvar may hold a later call's key.
            _emit_call_failure_log(get_client(), session, obs_key=obs_key)
            raise
        finally:
            if framework == "litellm":
                _in_litellm.set(False)
            if not _stream_failed:
                try:
                    if last is not None:
                        _latency = _build_stream_latency(req_start_mono, _ttft_mono, _time.monotonic())
                        response_for_comp = _stream_accumulator_to_response(provider, acc) or last
                        _capture_composition_at(provider, response_for_comp, order, True)
                        _log_manual(provider, session, kwargs, last, order, span_name, start_time,
                                    latency=_latency, obs_key=obs_key,
                                    route_ctx=route_ctx)
                    elif (provider or "").lower() == "together":
                        # G5-1: no usage chunk arrived — log an approximated
                        # row instead of silently losing the call.
                        _latency = _build_stream_latency(req_start_mono, _ttft_mono, _time.monotonic())
                        _log_stream_no_usage_fallback(provider, session, kwargs, acc,
                                                      order, span_name, start_time,
                                                      _latency, obs_key=obs_key)
                except Exception:
                    pass  # fail-open: never throw out of a wrapped stream
    # Wrap the metering generator in the sync stream proxy (same proxy reused
    # on the manual-telemetry stream wrappers) so the customer keeps the
    # manual-path provider's surface (CM protocol, `.response`,
    # `close()`) and early abandonment closes the underlying connection. The
    # `_gen()` body above is unchanged; the proxy only drives iteration + dual
    # close-through (metering gen first → its `finally`/litellm guard reset/
    # `_log_manual`, then the underlying).
    return _ModeAStreamProxy(stream, _gen())


def _wrap_async_stream(stream, provider, session, kwargs, order, span_name, start_time,
                       framework=None, req_start_mono=None, obs_key=None,
                       suppress_usage_chunk: bool = False, route_ctx=None):
    """Wrap an async streaming response — log the final chunk's usage on completion.

    Response composition is accumulated from the streamed chunks (the final
    usage chunk alone does not carry the assistant message content).

    When ``framework='litellm'`` the wrapper holds the ``_in_litellm`` guard
    across iteration so nested provider wrappers stay inert.

    ``req_start_mono`` anchors TTFT/total latency to the provider-call start
    (see ``_wrap_sync_stream``).
    """
    async def _agen():
        last = None
        acc = _new_stream_accumulator(provider)
        if framework == "litellm":
            _in_litellm.set(True)
        _stream_start = _time.monotonic()
        _ttft_mono = None
        _stream_failed = False

        def _tap(chunk):
            # Best-effort: a tap failure (chunk-shape drift) must never abort
            # the customer's iteration — degrade to telemetry loss.
            nonlocal last, _ttft_mono
            try:
                if _chunk_usage(chunk):
                    last = chunk
                _accumulate_stream_chunk(provider, acc, chunk)
                # Mark TTFT at the first chunk carrying renderable content.
                if _ttft_mono is None and _stream_acc_has_content(provider, acc):
                    _ttft_mono = _time.monotonic()
            except Exception:
                pass

        try:
            # Bridge a sync underlying iterator: an async-client stream method
            # (async_iter hint) may hand back a plain sync generator,
            # but the consumer iterates with `async for`. Drive whichever
            # protocol the underlying actually exposes.
            if hasattr(stream, "__aiter__") or is_async_iterator(stream):
                async for chunk in stream:
                    _tap(chunk)
                    if suppress_usage_chunk and _suppress_chunk_safe(chunk):
                        continue
                    yield chunk
            else:
                for chunk in stream:
                    _tap(chunk)
                    if suppress_usage_chunk and _suppress_chunk_safe(chunk):
                        continue
                    yield chunk
        except Exception as _exc:
            _stream_failed = True
            elapsed_ms = int((_time.monotonic() - _stream_start) * 1000)
            try:
                session._call_outcome = build_call_outcome(_exc, elapsed_ms)
            except Exception:
                pass
            # Captured-at-wrapper obs key: this drain runs at consumer-
            # iteration time, when the contextvar may hold a later call's key.
            _emit_call_failure_log(get_client(), session, obs_key=obs_key)
            raise
        finally:
            if framework == "litellm":
                _in_litellm.set(False)
            if not _stream_failed:
                try:
                    if last is not None:
                        _latency = _build_stream_latency(req_start_mono, _ttft_mono, _time.monotonic())
                        response_for_comp = _stream_accumulator_to_response(provider, acc) or last
                        _capture_composition_at(provider, response_for_comp, order, True)
                        _log_manual(provider, session, kwargs, last, order, span_name, start_time,
                                    latency=_latency, obs_key=obs_key,
                                    route_ctx=route_ctx)
                    elif (provider or "").lower() == "together":
                        # G5-1: no usage chunk arrived — log an approximated
                        # row instead of silently losing the call.
                        _latency = _build_stream_latency(req_start_mono, _ttft_mono, _time.monotonic())
                        _log_stream_no_usage_fallback(provider, session, kwargs, acc,
                                                      order, span_name, start_time,
                                                      _latency, obs_key=obs_key)
                except Exception:
                    pass  # fail-open: never throw out of a wrapped stream
    # Async twin — wrap in the async stream proxy (same proxy reused on the
    # manual-telemetry stream wrappers), which restores
    # `async with`, `.response`, `aclose()` + close-through on abandonment. The
    # `_agen()` body above (incl. the sync-underlying/async-consumer bridge)
    # is unchanged; the proxy only drives `__aiter__`/`__anext__` + dual
    # close-through.
    return _ModeAAsyncStreamProxy(stream, _agen())


class _ResponsesRawStreamProxy:
    """Proxies an openai AsyncAPIResponse / APIResponse for the Responses-API
    streaming path used by the openai-agents SDK.

    `client.responses.with_streaming_response.create(...)` sets the
    `X-Stainless-Raw-Response: stream` header and returns an AsyncAPIResponse
    object wrapping the underlying HTTP response — the caller then drains the
    SSE stream via `await api_response.parse()`. The raw response itself
    carries no usage or content; usage arrives only on the `response.completed`
    event emitted while iterating the parsed stream. We proxy every attribute
    of the wrapped response and intercept `.parse()` (sync / async) so the
    returned iterator runs through `_wrap_async_stream` /
    `_wrap_sync_stream`, which accumulates composition + dispatches `log_sync`
    once the consumer finishes draining.
    """

    __slots__ = (
        "_tp_wrapped", "_tp_provider", "_tp_session", "_tp_kwargs",
        "_tp_order", "_tp_span_name", "_tp_start_time", "_tp_framework",
        "_tp_req_start_mono", "_tp_obs_key", "_tp_route_ctx",
    )

    def __init__(self, wrapped, provider, session, kwargs, order, span_name,
                 start_time, framework, req_start_mono=None, obs_key=None,
                 route_ctx=None):
        object.__setattr__(self, "_tp_wrapped", wrapped)
        object.__setattr__(self, "_tp_provider", provider)
        object.__setattr__(self, "_tp_session", session)
        object.__setattr__(self, "_tp_kwargs", kwargs)
        object.__setattr__(self, "_tp_order", order)
        object.__setattr__(self, "_tp_span_name", span_name)
        object.__setattr__(self, "_tp_start_time", start_time)
        object.__setattr__(self, "_tp_framework", framework)
        object.__setattr__(self, "_tp_req_start_mono", req_start_mono)
        # Captured-at-wrapper obs key — parse() is drained later, when the
        # contextvar may hold another call's key.
        object.__setattr__(self, "_tp_obs_key", obs_key)
        # LiteLLM seam route context (None elsewhere) — the parsed stream is
        # drained later and still logs this call's model.
        object.__setattr__(self, "_tp_route_ctx", route_ctx)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_tp_wrapped"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_tp_wrapped"), name, value)

    async def parse(self, *args, **kwargs):
        stream = await object.__getattribute__(self, "_tp_wrapped").parse(
            *args, **kwargs
        )
        return _wrap_async_stream(
            stream,
            object.__getattribute__(self, "_tp_provider"),
            object.__getattribute__(self, "_tp_session"),
            object.__getattribute__(self, "_tp_kwargs"),
            object.__getattribute__(self, "_tp_order"),
            object.__getattribute__(self, "_tp_span_name"),
            object.__getattribute__(self, "_tp_start_time"),
            framework=object.__getattribute__(self, "_tp_framework"),
            req_start_mono=object.__getattribute__(self, "_tp_req_start_mono"),
            obs_key=object.__getattribute__(self, "_tp_obs_key"),
            route_ctx=object.__getattribute__(self, "_tp_route_ctx"),
        )


def _is_raw_stream_response(provider: str, kwargs: dict, result) -> bool:
    """True if this Responses-API call was made through
    `with_streaming_response.create(...)` — i.e. extra_headers carries the raw
    stream sentinel and the SDK returned an AsyncAPIResponse / APIResponse."""
    if provider != "openai_responses":
        return False
    if not result:
        return False
    headers = kwargs.get("extra_headers") if isinstance(kwargs, dict) else None
    if not isinstance(headers, dict):
        return False
    if headers.get("X-Stainless-Raw-Response") != "stream":
        return False
    # Anything that has a `.parse` method and *isn't* a stream is treated as a
    # raw API response carrying a streaming HTTP body. The proxy delegates
    # everything else.
    return hasattr(result, "parse") and not _is_stream(result)


def _set_manual_wrapper(cls, method_name, original, provider, is_async, framework=None,
                        modality=None, shape=None, operation=None, async_iter=None,
                        module_path=None):
    """Install a manual-telemetry wrapper (pre-flight check + manual token extraction).

    ``framework`` is set for SDKs that internally call other patched SDKs
    (currently "litellm"). The wrapper holds a guard ContextVar across the
    inner call so the nested provider wrapper stays inert (avoids double
    pre-flight checks and double logging).

    ``modality`` + ``shape`` activate the modality (image/audio/video/ocr)
    code path: pre-flight check carries an intent hint, telemetry uses an
    explicit usage-shape enum with items/duration extracted by the registered
    handler in ``_MODALITY_HANDLERS``.

    ``module_path`` is the registration's module — with ``framework`` it
    identifies the LiteLLM seam (``tp.protect("litellm", ...)`` supplies no
    framework and may name any provider slug), which resolves a per-call route
    context so matching/telemetry use LiteLLM's bare model name.
    """
    if is_async:
        @functools.wraps(original)
        async def manual_async_wrapper(*args, **kwargs):
            # Inside a framework that logs at its own level (LangChain /
            # LiteLLM / LlamaIndex / Pydantic AI) → pass through entirely;
            # those frameworks emit the /log themselves.
            if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai():
                return await original(*args, **kwargs)

            # OpenAI Agents SDK runs tool functions outside any patched call and
            # records them only in its own (non-OTel) tracing — register our
            # tool-span processor once the SDK is in use (cheap idempotent guard).
            maybe_register_openai_agents_tracing()

            # 1. Pre-flight check (may raise TokenPoliceBlockedError —
            # intended). Skipped inside Agno because the Agent.run wrapper
            # already ran the single per-agent check; we still proceed to
            # composition + token extraction + /log so every inner LLM
            # call lands as its own row.
            # The check may mutate kwargs["model"] when a REROUTE rule
            # fires in enforce mode.
            intent = _build_intent(provider, modality, args, kwargs) if modality else None
            # LiteLLM addresses vendors as "<vendor>/<model>"; resolve the bare
            # name ONCE, before the check can rewrite kwargs. None on every
            # other seam (byte-identical behavior there).
            _route_ctx = _litellm_route_context(framework, module_path, kwargs)
            if not in_agno():
                await _run_async_check(kwargs=kwargs, provider=provider, intent=intent,
                                       model_hint=(_xai_model_hint(args)
                                                   if provider == "xai" else None),
                                       route_ctx=_route_ctx)
            # Capture THIS call's obs key into a local right after the check —
            # threaded into every later drain (failure log / stream finalize /
            # manual log), which may run after another call's check overwrote
            # the contextvar.
            _obs_key = _state.get_current_obs_key()

            # 2. Reserve span order + name up front (no OTel span exists here,
            # so telemetry.on_start never runs to consume the counter).
            order = 0
            span_name = None
            session = None
            start_time = datetime.now(timezone.utc)
            # xai-sdk's sample()/stream() receive no message kwargs; messages
            # live on `self._proto` (args[0]). Synthesize kwargs once so
            # composition + _log_manual model fallback work uniformly.
            comp_kwargs = _build_xai_pseudo_kwargs(args) if provider == "xai" else kwargs
            try:
                session = get_current_session()
                order = session.next_span_order()
                span_name = consume_pending_span_name()
                # Default span name to "agent_step_N" inside Agno so each
                # inner LLM call surfaces as a distinct, recognizable step
                # in the dashboard instead of just the model id.
                if not span_name and in_agno():
                    span_name = f"agent_step_{order + 1}"
                _capture_composition_at(provider, comp_kwargs, order, False,
                                        operation=operation)
            except Exception:
                pass  # fail-safe

            # Stash attempt context so the failure log carries model/provider/
            # operation if the call raises (e.g. invalid api key, wrong model id).
            # Prefer registry `operation` (e.g. "embedding"); fall back to
            # `modality` (image_gen/audio_tts/...) so failed modality rows
            # classify like successful ones instead of defaulting to "chat".
            # `shape` is the registry's usage-shape override (the same value
            # the success path passes as `shape_override`) so failed modality
            # rows carry e.g. `openai_images`, not the chat synth.
            _stash_attempt_context(session, provider, "", args, comp_kwargs,
                                   operation=(operation or modality), shape=shape,
                                   route_ctx=_route_ctx)

            # 3. Call original — hold the framework guard so nested provider
            # wrappers (e.g. openai/anthropic patched layer) pass through.
            if framework == "litellm":
                _in_litellm.set(True)
            _call_start = _time.monotonic()
            # G5-1: chat streams on injection-listed manual providers
            # (together) get include_usage injected so the final usage chunk
            # is deterministic instead of replica-dependent; the synthetic
            # usage-only chunk is stripped below (suppress_usage_chunk).
            # Strip-and-retry mirrors the Mode-A path: only a 400/422 or a
            # message naming the param retries — refused before generation,
            # so a retry cannot double-generate.
            _usage_injected = _inject_stream_usage_option_for_provider(provider, kwargs)
            try:
                try:
                    result = await original(*args, **kwargs)
                except Exception as _exc:
                    if _usage_injected and _should_retry_without_injection(_exc):
                        _restore_stream_usage_option(kwargs, _usage_injected)
                        _usage_injected = None
                        result = await original(*args, **kwargs)
                    else:
                        raise
            except Exception as _exc:
                elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                try:
                    session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                except Exception:
                    pass
                if framework == "litellm":
                    _in_litellm.set(False)
                _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                raise
            finally:
                # For non-streaming we reset immediately. For streaming the
                # stream wrapper re-enters the guard for the duration of
                # iteration; resetting here is safe since the wrapper sets it
                # again before yielding the first chunk.
                if framework == "litellm":
                    _in_litellm.set(False)

            # 4. Streaming → wrap; non-streaming → extract usage + log now.
            # The Responses-API "with_streaming_response.create(...)" path
            # (used by openai-agents) returns an AsyncAPIResponse that the
            # caller drains via `await api_response.parse()`; intercept it
            # via _ResponsesRawStreamProxy so the parsed stream still flows
            # through _wrap_async_stream.
            try:
                # Modality calls (image/audio/video) never stream and have
                # their own item/duration extractors. Pass the resolved shape
                # so build_response_composition can short-circuit binary bodies
                # into a non-text entry (prevents TTS audio bytes from being
                # decoded into a fake "assistant text" via `.text`).
                if modality:
                    # Best-effort: harvest usage from the SSE stream when the
                    # caller opted into stream_format="sse" on a TTS-capable
                    # model. No-op for every other case.
                    result = await _maybe_tap_openai_tts_sse_async(provider, modality, kwargs, result)
                    _capture_composition_at(provider, result, order, True,
                                            usage_shape=shape)
                    elapsed_s = max(0.0, _time.monotonic() - _call_start)
                    _log_modality(provider, modality, shape, session, comp_kwargs, args,
                                  result, order, span_name, start_time, elapsed_s,
                                  obs_key=_obs_key)
                    return result
                if _is_stream(result):
                    # This is the async wrapper — the customer will iterate with
                    # `async for`, so prefer the async tap for any async-capable
                    # result (including dual-protocol streams, e.g. litellm's
                    # CustomStreamWrapper exposing both __iter__ and __anext__ —
                    # must not be coerced to the wrong protocol). Only cross to
                    # the sync tap for a sync-only result. Mirror of the
                    # equivalent fix in the sync wrapper below.
                    if is_sync_iterator(result) and not is_async_iterator(result):
                        return _wrap_sync_stream(result, provider, session, comp_kwargs,
                                                 order, span_name, start_time,
                                                 framework=framework, req_start_mono=_call_start,
                                                 obs_key=_obs_key,
                                                 suppress_usage_chunk=bool(_usage_injected),
                                                 route_ctx=_route_ctx)
                    return _wrap_async_stream(result, provider, session, comp_kwargs,
                                              order, span_name, start_time,
                                              framework=framework, req_start_mono=_call_start,
                                              obs_key=_obs_key,
                                              suppress_usage_chunk=bool(_usage_injected),
                                              route_ctx=_route_ctx)
                if _is_raw_stream_response(provider, kwargs, result):
                    return _ResponsesRawStreamProxy(
                        result, provider, session, comp_kwargs, order, span_name,
                        start_time, framework, req_start_mono=_call_start,
                        obs_key=_obs_key, route_ctx=_route_ctx,
                    )
                # Embedding response is a vector; use the shape override so
                # build_response_composition returns [] via _EMBEDDING_SHAPES.
                resp_shape = shape if operation == "embedding" else None
                _capture_composition_at(provider, result, order, True,
                                        usage_shape=resp_shape)
                _log_manual(provider, session, comp_kwargs, result, order, span_name, start_time,
                            operation=operation or "chat",
                            shape_override=shape,
                            args=args,
                            latency=_build_non_stream_latency(_call_start, _time.monotonic()),
                            obs_key=_obs_key,
                            route_ctx=_route_ctx)
            except Exception:
                pass  # fail-open
            return result

        # Identity marker for _wrap_method's alias guard (see there).
        manual_async_wrapper._tp_preflight_wrapper = True
        setattr(cls, method_name, manual_async_wrapper)
    else:
        @functools.wraps(original)
        def manual_sync_wrapper(*args, **kwargs):
            # See manual_async_wrapper above for the rationale on which
            # framework guards short-circuit vs. fall through.
            if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai():
                return original(*args, **kwargs)

            # OpenAI Agents SDK runs tool functions outside any patched call and
            # records them only in its own (non-OTel) tracing — register our
            # tool-span processor once the SDK is in use (cheap idempotent guard).
            maybe_register_openai_agents_tracing()

            # 1. Pre-flight check (skipped inside Agno — Agent.run already
            # ran it once for the entire agent loop).
            # The check may mutate kwargs["model"] when a REROUTE rule
            # fires in enforce mode.
            intent = _build_intent(provider, modality, args, kwargs) if modality else None
            # See manual_async_wrapper — resolved before the check may rewrite kwargs.
            _route_ctx = _litellm_route_context(framework, module_path, kwargs)
            if not in_agno():
                _run_sync_check(kwargs=kwargs, provider=provider, intent=intent,
                                model_hint=(_xai_model_hint(args)
                                            if provider == "xai" else None),
                                route_ctx=_route_ctx)
            # Capture THIS call's obs key right after the check (see
            # manual_async_wrapper) — threaded into every later drain.
            _obs_key = _state.get_current_obs_key()

            # 2. Reserve span order + name up front.
            order = 0
            span_name = None
            session = None
            start_time = datetime.now(timezone.utc)
            comp_kwargs = _build_xai_pseudo_kwargs(args) if provider == "xai" else kwargs
            try:
                session = get_current_session()
                order = session.next_span_order()
                span_name = consume_pending_span_name()
                # Default span name to "agent_step_N" inside Agno so each
                # inner LLM call surfaces as a distinct, recognizable step
                # in the dashboard instead of just the model id.
                if not span_name and in_agno():
                    span_name = f"agent_step_{order + 1}"
                _capture_composition_at(provider, comp_kwargs, order, False,
                                        operation=operation)
            except Exception:
                pass  # fail-safe

            # Stash attempt context so the failure log carries model/provider/
            # operation if the call raises (e.g. invalid api key, wrong model id).
            # Prefer registry `operation` (e.g. "embedding"); fall back to
            # `modality` (image_gen/audio_tts/...) so failed modality rows
            # classify like successful ones instead of defaulting to "chat".
            # `shape` is the registry's usage-shape override (the same value
            # the success path passes as `shape_override`) so failed modality
            # rows carry e.g. `openai_images`, not the chat synth.
            _stash_attempt_context(session, provider, "", args, comp_kwargs,
                                   operation=(operation or modality), shape=shape,
                                   route_ctx=_route_ctx)

            # 3. Call original — hold the framework guard so nested provider
            # wrappers pass through.
            if framework == "litellm":
                _in_litellm.set(True)
            _call_start = _time.monotonic()
            # G5-1 include_usage injection — see manual_async_wrapper above.
            _usage_injected = _inject_stream_usage_option_for_provider(provider, kwargs)
            try:
                try:
                    result = original(*args, **kwargs)
                except Exception as _exc:
                    if _usage_injected and _should_retry_without_injection(_exc):
                        _restore_stream_usage_option(kwargs, _usage_injected)
                        _usage_injected = None
                        result = original(*args, **kwargs)
                    else:
                        raise
            except Exception as _exc:
                elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                try:
                    session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                except Exception:
                    pass
                if framework == "litellm":
                    _in_litellm.set(False)
                _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                raise
            finally:
                if framework == "litellm":
                    _in_litellm.set(False)

            # 4. Streaming → wrap; non-streaming → extract usage + log now.
            # (Sync Responses-API streaming path also goes through
            # _ResponsesRawStreamProxy — its `parse()` is sync but the
            # proxy returns the wrapped stream the same way.)
            # Some sync-declared methods (Cohere AsyncClientV2.chat_stream,
            # xai_sdk.aio.chat.Chat.stream) return an async iterator
            # synchronously — pick the right wrapper based on iterator
            # protocol so `async for chunk in result:` keeps working.
            try:
                if modality:
                    # Best-effort SSE tap when caller opted in. No-op otherwise.
                    result = _maybe_tap_openai_tts_sse_sync(provider, modality, kwargs, result)
                    # Pass shape as authoritative override — see async_wrapper
                    # above for the TTS-misclassification rationale.
                    _capture_composition_at(provider, result, order, True,
                                            usage_shape=shape)
                    elapsed_s = max(0.0, _time.monotonic() - _call_start)
                    _log_modality(provider, modality, shape, session, comp_kwargs, args,
                                  result, order, span_name, start_time, elapsed_s,
                                  obs_key=_obs_key)
                    return result
                if _is_stream(result):
                    # This is the sync wrapper — the customer called the
                    # sync method and (unless this is an async-client method
                    # registered async=False, the `async_iter` hint) will iterate
                    # with a plain `for`. Hand back an iterator of the protocol
                    # they expect. A dual-protocol stream (litellm's
                    # CustomStreamWrapper exposes both __iter__ and __anext__)
                    # must not be coerced to an async_generator; prefer the sync
                    # tap whenever the result is sync-capable, crossing to the
                    # async tap only for an async-only result or the async_iter
                    # hint (e.g. xAI's aio Chat.stream — the same sync-registered/
                    # async-consumer situation as the async wrapper above).
                    if async_iter or (is_async_iterator(result) and not is_sync_iterator(result)):
                        return _wrap_async_stream(result, provider, session, comp_kwargs,
                                                  order, span_name, start_time,
                                                  framework=framework, req_start_mono=_call_start,
                                                  obs_key=_obs_key,
                                                  suppress_usage_chunk=bool(_usage_injected),
                                                  route_ctx=_route_ctx)
                    return _wrap_sync_stream(result, provider, session, comp_kwargs,
                                             order, span_name, start_time,
                                             framework=framework, req_start_mono=_call_start,
                                             obs_key=_obs_key,
                                             suppress_usage_chunk=bool(_usage_injected),
                                             route_ctx=_route_ctx)
                if _is_raw_stream_response(provider, kwargs, result):
                    return _ResponsesRawStreamProxy(
                        result, provider, session, comp_kwargs, order, span_name,
                        start_time, framework, req_start_mono=_call_start,
                        obs_key=_obs_key, route_ctx=_route_ctx,
                    )
                # Embedding response is a vector; use the shape override so
                # build_response_composition returns [] via _EMBEDDING_SHAPES.
                resp_shape = shape if operation == "embedding" else None
                _capture_composition_at(provider, result, order, True,
                                        usage_shape=resp_shape)
                _log_manual(provider, session, comp_kwargs, result, order, span_name, start_time,
                            operation=operation or "chat",
                            shape_override=shape,
                            args=args,
                            latency=_build_non_stream_latency(_call_start, _time.monotonic()),
                            obs_key=_obs_key,
                            route_ctx=_route_ctx)
            except Exception:
                pass  # fail-open
            return result

        # Identity marker for _wrap_method's alias guard (see there).
        manual_sync_wrapper._tp_preflight_wrapper = True
        setattr(cls, method_name, manual_sync_wrapper)


# ═══════════════════════════════════════════════════════════════════
# LangChain wrappers — for langchain_core.language_models.chat_models
# .BaseChatModel.{generate,agenerate,stream,astream}.
#
# LangChain is a framework: it calls the underlying provider SDK (openai,
# anthropic, ...) internally, and that SDK is also patched by the enforcer.
# These wrappers run the SINGLE pre-flight check, capture prompt/response
# composition from LangChain's own message objects, and hold the _in_langchain
# guard so the nested provider wrapper passes straight through. Telemetry flows
# through the OpenLLMetry LangChain instrumentor span — see telemetry.py.
# ═══════════════════════════════════════════════════════════════════

@fail_safe
def _capture_langchain_prompt(args):
    """Capture prompt composition for a LangChain generate/stream call.

    BaseChatModel.generate / agenerate receive messages as the positional arg
    args[1] (list[list[BaseMessage]]); stream / astream receive a single
    `input` (str / list[BaseMessage] / PromptValue) as args[1]. The "langchain"
    composition parser handles all of these shapes.
    """
    if len(args) < 2:
        return
    payload = args[1]
    session = get_current_session()
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    # The enforcer runs BEFORE telemetry on_start, so _span_counter is still at
    # the value the LLM span's on_start will consume via next_span_order().
    order = session._span_counter
    # Stash it so the response capture (which runs at stream-end, after the
    # counter has advanced) keys onto this same LLM span. See
    # _capture_response_composition.
    session._mode_a_prompt_order = order
    comp = build_prompt_composition("langchain", {"messages": payload})
    if comp:
        comp_key = f"{session.trace_id}:{order}"
        session._pending_compositions.setdefault(comp_key, {})["prompt"] = comp
    # Return the snapshotted order so the stream guards can thread it into the
    # response capture (interleave-safe keying) even if a later interleaved
    # stream clobbers the shared mailbox. `@fail_safe` returns None on any
    # internal raise, which the guards treat as "no snapshot" → mailbox fallback.
    return order


def _accumulate_langchain_chunk(acc, chunk):
    """Fold a streamed LangChain chunk into a running accumulator. Chunks are
    AIMessageChunks (summable via +); a ChatGenerationChunk exposes the message
    on `.message`. Fail-open — a bad chunk just isn't accumulated."""
    try:
        msg = getattr(chunk, "message", None)
        if msg is None:
            msg = chunk
        return msg if acc is None else (acc + msg)
    except Exception:
        return acc


def _lc_reasoning_from_usage_metadata(um) -> int:
    """Reasoning-token count from a LangChain ``usage_metadata`` dict
    (``output_token_details.reasoning``), or 0. Never raises."""
    try:
        if not isinstance(um, dict):
            return 0
        otd = um.get("output_token_details")
        if not isinstance(otd, dict):
            return 0
        return max(0, int(otd.get("reasoning") or 0))
    except Exception:
        return 0


def _lc_reasoning_from_llmresult(result) -> int:
    """Sum reasoning tokens across an LLMResult's outer generations lists
    (one per provider call). Within a list only the FIRST usage-bearing
    candidate is read — n>1 candidates carry the same duplicated full-call
    usage_metadata, so summing within a list would over-count. 0 on any
    failure (never raises)."""
    try:
        total = 0
        gens = getattr(result, "generations", None)
        if not isinstance(gens, list):
            return 0
        for gen_list in gens:
            if not isinstance(gen_list, list):
                gen_list = [gen_list]
            for cand in gen_list:
                msg = getattr(cand, "message", None)
                um = getattr(msg, "usage_metadata", None) if msg is not None else None
                if not isinstance(um, dict):
                    continue
                total += _lc_reasoning_from_usage_metadata(um)
                break
        return total
    except Exception:
        return 0


@fail_safe
def _stash_lc_reasoning_tokens(session, reasoning, order=None):
    """Stash a LangChain reasoning-token count on the pending-composition slot
    (G3-O1) so ``_flush_deferred_spans`` can merge it into the deferred
    payload's usage raw. The LC instrumentor never emits
    ``gen_ai.usage.reasoning_tokens``, so the Mode-A synth raw drops it.

    Must run BEFORE ``_capture_response_composition`` (which clears the
    one-slot mailbox) — the ``order is None`` fallback mirrors that
    function's mailbox → ``_span_counter - 1`` resolution so both writes key
    onto the same slot."""
    if session is None:
        return
    r = int(reasoning or 0)
    if r <= 0:
        return
    if order is None:
        mailbox = getattr(session, "_mode_a_prompt_order", None)
        order = mailbox if mailbox is not None else (session._span_counter - 1)
    if not hasattr(session, "_pending_compositions"):
        session._pending_compositions = {}
    comp_key = f"{session.trace_id}:{order}"
    session._pending_compositions.setdefault(comp_key, {})["lc_reasoning_tokens"] = r


def _finalize_langchain_stream(session, prev_defer, acc, order=None,
                               obs_key=_OBS_KEY_CURRENT):
    """Capture response composition from the folded stream + flush the deferred
    LLM span. Mirrors lc_sync/lc_async's post-call capture so streamed and
    AgentExecutor-driven calls get response composition too (they otherwise only
    carried the prompt). Fail-open at every step.

    ``order`` is this stream's own LLM-span order (snapshotted at prompt capture)
    so the response fingerprint keys onto this stream even if an interleaved
    stream has since overwritten the shared one-slot mailbox.

    ``obs_key`` is the stream's own obs key, captured right after its check —
    this finalize runs at consumer-drain time, when the contextvar may hold
    an interleaved call's key."""
    session._defer_telemetry = prev_defer
    try:
        if acc is not None:
            # G3-O1: reasoning from the concat'd chunk's usage_metadata.
            # concat has already summed output_token_details.reasoning across
            # chunks (LC-OpenAI only stamps the final chunk, so the sum IS the
            # value; LC-Gemini de-cumulates upstream) — do NOT re-derive.
            # Stashed before the response capture (which clears the mailbox).
            _stash_lc_reasoning_tokens(
                session,
                _lc_reasoning_from_usage_metadata(getattr(acc, "usage_metadata", None)),
                order=order)
            _capture_response_composition("langchain", acc, order=order)
    except Exception:
        pass
    try:
        if not session._defer_telemetry:
            _flush_deferred_spans(session, obs_key=obs_key)
    except Exception:
        pass


def _lc_trace_id(session) -> str:
    """Session.trace_id for the process-global LC registry. Never raises."""
    try:
        tid = getattr(session, "trace_id", None) if session is not None else None
        return tid if isinstance(tid, str) and tid else ""
    except Exception:
        return ""


def _guard_sync_stream(stream, order=None, session=None, obs_key=_OBS_KEY_CURRENT):
    """Yield from a LangChain stream while holding the _in_langchain guard (so the
    underlying provider call stays inert) AND folding the streamed chunks so the
    deferred LLM span carries response composition (not just the prompt).

    The guard is held True only around each inner pull and restored to its
    true prior value before the customer-facing ``yield`` (a plain sync generator
    shares the caller's contextvars.Context, so a whole-iteration hold would leak
    into an interleaved stream and make it skip enforcement + metering). The
    restore primitive is ``_in_langchain.set(prev)`` — ``ContextVar.set`` has no
    failure mode, so fail-openness never depends on the try/except wrap. A
    stream-local depth counter (``_lc_stream_depth``) keeps ``_defer_telemetry``
    truthy across an interleave and flushes the shared deferred-span buffer once,
    at the outermost unwind, from a single baseline stashed at the outermost entry
    (``_lc_stream_defer_prev``) — otherwise a non-LIFO interleaved drain would
    clobber the other stream's in-flight span and pin ``_defer_telemetry``.

    The process-global ``enter_langchain_trace``/``leave_langchain_trace``
    pair follows the same pull-only window so interleaved streams keep their
    Invariant (no whole-stream registry hold across yield).

    ``session`` is threaded from lc_stream (which resolved it before the check)
    so the REROUTE audit the check stashed lands on the object finalized here;
    absent it, resolve as before (backward-compatible).

    ``obs_key`` is threaded from lc_stream too (captured right after its
    check) so the finalize-time observations drain claims THIS stream's
    entries even when an interleaved call has since overwritten the
    contextvar."""
    session = session if session is not None else get_current_session()
    tid = _lc_trace_id(session)
    # Failure-outcome anchor: no request-start mono is in scope here (the
    # wrapper already created `stream` before handing it over), so guard entry
    # (= first pull, generators are lazy) is the closest available anchor.
    _t0 = _time.monotonic()
    # True only when the pull-failure handler below stamped _call_outcome —
    # gates the stale-outcome clear in the outermost finally.
    outcome_set = False
    depth = getattr(session, '_lc_stream_depth', 0)
    if depth == 0:                                    # remember TRUE baseline ONCE (outermost)
        session._lc_stream_defer_prev = getattr(session, '_defer_telemetry', False)
    session._lc_stream_depth = depth + 1
    session._defer_telemetry = True
    acc = None
    it = iter(stream)
    try:
        while True:
            prev = _in_langchain.get()                # (A) true prior value (nested-safe)
            try:
                _in_langchain.set(True)               # guard True ONLY around the pull
            except Exception:                         # set() has no failure mode; guarded defensively anyway
                pass
            enter_langchain_trace(tid)                # Backstop (process-global)
            try:
                try:
                    chunk = next(it)
                except StopIteration:
                    break
                except Exception as _exc:
                    # Provider failure surfacing on a pull (401/400/429 —
                    # the request only fires on iteration): stamp the failed
                    # outcome so the deferred ERROR span the LC callback
                    # queued flushes WITH call_outcome instead of landing as
                    # a success-defaulted row. Scoped to the pull ONLY — the
                    # yield stays outside so GeneratorExit / consumer-break
                    # semantics are untouched (Exception never catches
                    # BaseException). The customer's original exception is
                    # re-raised by identity (golden rule).
                    try:
                        session._call_outcome = build_call_outcome(
                            _exc, int((_time.monotonic() - _t0) * 1000))
                        outcome_set = True
                    except Exception:
                        pass
                    raise
            finally:
                leave_langchain_trace(tid)
                try:
                    _in_langchain.set(prev)           # restore BEFORE yield; set() cannot raise
                except Exception:
                    pass
            acc = _accumulate_langchain_chunk(acc, chunk)
            yield chunk                               # customer holds A here with guard = prior
    finally:                                          # (B) unwind ONE interleave level
        try:
            d = getattr(session, '_lc_stream_depth', 1) - 1
            session._lc_stream_depth = d if d > 0 else 0
        except Exception:
            d = 0
        if d <= 0:                                    # outermost: restore baseline + flush ONCE
            _finalize_langchain_stream(
                session, getattr(session, '_lc_stream_defer_prev', False), acc, order,
                obs_key=obs_key)
            # Stale-outcome clear (regression guard): _flush_deferred_spans
            # early-returns WITHOUT clearing _call_outcome when the deferred
            # buffer is empty (e.g. the ERROR span was gate-dropped) — a
            # leftover failed outcome would mislabel the session's NEXT row.
            # Outermost level only, and only when the flush actually ran
            # (defer restored False): under an outer defer window the payload
            # flushes later and still needs the outcome.
            try:
                if (outcome_set and not getattr(session, '_defer_telemetry', False)
                        and getattr(session, '_call_outcome', None) is not None):
                    session._call_outcome = None
            except Exception:
                pass
        else:                                         # inner level still open: capture own comp only
            try:
                if acc is not None:
                    _capture_response_composition("langchain", acc, order=order)
            except Exception:
                pass


def _instance_model_attr(instance, *names):
    """First non-empty string attribute among ``names`` on ``instance``, else
    None. Framework SDKs keep the model on the instance; reads it for the
    pre-flight context. Never raises — hostile properties just yield None."""
    if instance is None:
        return None
    for name in names:
        try:
            value = getattr(instance, name, None)
        except Exception:
            continue
        if isinstance(value, str) and value:
            return value
    return None


# Canonical provider slug per LangChain integration package. Slugs match
# the published provider-slug table (the local evaluator canonicalizes aliases, but
# emitting the canonical form keeps /check and /log bucketing identical).
_LC_CHAT_MODULE_PROVIDERS = {
    "langchain_openai": "openai",
    "langchain_anthropic": "anthropic",
    "langchain_google_genai": "google",
    "langchain_google_vertexai": "vertex-ai",
    "langchain_mistralai": "mistral",
    "langchain_cohere": "cohere",
    "langchain_groq": "groq",
    "langchain_aws": "bedrock",
}

# Ordered class-name fallback for out-of-tree subclasses. "vertex"/"claude"
# precede the broader needles so a ChatVertexAI subclass isn't caught by
# "google" first.
_LC_CHAT_CLASS_PROVIDERS = (
    ("vertex", "vertex-ai"),
    ("bedrock", "bedrock"),
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("gemini", "google"),
    ("google", "google"),
    ("mistral", "mistral"),
    ("cohere", "cohere"),
    ("groq", "groq"),
    ("openai", "openai"),
)


def _lc_provider_from_instance(instance):
    """Canonical provider slug for a LangChain chat-model instance, or None.

    Returns None when underivable — the framework slug "langchain" must NEVER
    be used as a chat-path check provider: it would make every provider-targeted
    REROUTE look cross-provider. Never raises."""
    try:
        cls = type(instance)
        root = (cls.__module__ or "").split(".")[0]
        hit = _LC_CHAT_MODULE_PROVIDERS.get(root)
        if hit:
            return hit
        name = (cls.__name__ or "").lower()
    except Exception:
        return None
    for needle, slug in _LC_CHAT_CLASS_PROVIDERS:
        if needle in name:
            return slug
    return None


# The Google API resource-name prefix langchain_google_genai bolts onto .model.
_GOOGLE_MODELS_PREFIX = "models/"


def _lc_google_bare_model(instance, model):
    """Bare model id for a ``langchain_google_genai`` instance, else ``model``.

    Upstream defect, not ours: ChatGoogleGenerativeAI /
    GoogleGenerativeAIEmbeddings rewrite their own ``.model`` field inside the
    pydantic ``validate_environment`` validator —
    ``if not self.model.startswith("models/"): self.model = f"models/{self.model}"``
    — so by the time we read the instance the attribute is the Google API
    *resource name* ``models/gemini-2.5-flash``, not the model id the customer
    typed. Every other LangChain integration leaves ``.model`` alone.

    That matters because /check matches rule conditions on this string
    EXACTLY and the collector normalizes nothing, so a rule
    ``model EQ gemini-2.5-flash`` silently misses the ``models/``-prefixed hint:
    an ENFORCE BLOCK never fires and a model-scoped REROUTE stays silent —
    the worst failure mode we have, since the customer sees no error at all.
    The defect is invisible in ``generations`` because the /log path already
    strips the prefix in telemetry.py before pricing; the two strips are
    independent and both idempotent (stripping a bare id is a no-op).

    The gate is deliberately the *library* — some class in the instance's MRO
    defined under ``langchain_google_genai`` — and not the provider slug: only
    the package that carries the bug is normalized, so no other provider,
    framework or hand-rolled integration can be perturbed by this. Walking the
    whole MRO (not just the concrete class) keeps out-of-tree subclasses of
    ChatGoogleGenerativeAI covered.

    The instance is NEVER mutated: LangChain must keep sending ``models/<id>``
    to Google, which is the correct wire format. Only our own copy of the
    string — used for rule matching, audit and logging — is normalized.
    Never raises; returns ``model`` untouched on any failure."""
    try:
        # Cheap guard first: one isinstance + one startswith for every non-Google
        # call in the process, so nothing else walks the MRO on the hot path. It
        # sits INSIDE the try because a str subclass could override startswith to
        # raise, and _lc_check_ctx is called from wrapper bodies that are not
        # themselves @fail_safe — nothing here may reach the customer's app.
        if not isinstance(model, str) or not model.startswith(_GOOGLE_MODELS_PREFIX):
            return model
        for klass in type(instance).__mro__:
            if (getattr(klass, "__module__", "") or "").split(".")[0] == "langchain_google_genai":
                return model[len(_GOOGLE_MODELS_PREFIX):]
    except Exception:
        return model
    return model


def _lc_check_ctx(args):
    """``(model_hint, provider)`` from the bound ChatModel (``args[0]``).
    LangChain keeps the model on the instance, never in the call kwargs, so the
    pre-flight used to evaluate with an empty model AND no provider. Hint is for
    matching + audit only — it never reaches kwargs. Never raises."""
    instance = args[0] if args else None
    if instance is None:
        return None, None
    # langchain_google_genai stores the Google resource name ("models/<id>") on
    # .model; normalize OUR copy so model-scoped rules can match. See
    # _lc_google_bare_model. Provider derivation is unaffected.
    return (_lc_google_bare_model(instance,
                                  _instance_model_attr(instance, "model", "model_name")),
            _lc_provider_from_instance(instance))


async def _guard_async_stream(original, args, kwargs):
    """Run the async pre-flight check, then yield from a LangChain astream while
    holding the _in_langchain guard and folding chunks for response composition.
    astream() returns its async iterator synchronously, so the check is deferred
    into this wrapping async generator where it can run in async context.

    Async twin of _guard_sync_stream: same-Task interleaving shares the
    Task's contextvars.Context, so the guard is held True only around each
    ``await it.__anext__()`` and restored to its prior value before the yield;
    the same depth-counter coordination flushes the shared deferred buffer once
    at the outermost unwind. registry window matches the pull-only guard."""
    # Resolve the session once and thread it into the check so a REROUTE audit
    # stash lands on the same object finalized (flushed) below.
    session = get_current_session()
    tid = _lc_trace_id(session)
    _hint, _prov = _lc_check_ctx(args)
    await _run_async_check(session=session, model_hint=_hint, provider=_prov)
    # Capture this stream's obs key right after the check — the finalize
    # below runs at consumer-drain time, when an interleaved call may have
    # overwritten the contextvar.
    _obs_key = _state.get_current_obs_key()
    # Snapshot this stream's own LLM-span order synchronously (before any await
    # on iteration), so an interleaved stream that has since overwritten the
    # shared one-slot mailbox can't mis-key this stream's response fingerprint.
    order = _capture_langchain_prompt(args)
    depth = getattr(session, '_lc_stream_depth', 0)
    if depth == 0:                                    # remember TRUE baseline ONCE (outermost)
        session._lc_stream_defer_prev = getattr(session, '_defer_telemetry', False)
    session._lc_stream_depth = depth + 1
    session._defer_telemetry = True
    acc = None
    # Failure-outcome anchor: post-check, pre-provider-call — the closest
    # in-scope stand-in for request start (no mono anchor is threaded here).
    _t0 = _time.monotonic()
    # True only when the pull-failure handler below stamped _call_outcome —
    # gates the stale-outcome clear in the outermost finally.
    outcome_set = False
    try:
        try:
            astream = original(*args, **kwargs)
        except Exception as _exc:
            # Construction-time rejection (before any pull): the outer finally
            # still finalizes/flushes, so stamp the failed outcome now or the
            # deferred ERROR span would land success-defaulted — same stamp as
            # the pull-failure handler below. Identity re-raise (golden rule).
            try:
                session._call_outcome = build_call_outcome(
                    _exc, int((_time.monotonic() - _t0) * 1000))
                outcome_set = True
            except Exception:
                pass
            raise
        it = astream.__aiter__()
        while True:
            prev = _in_langchain.get()                # (A) true prior value (nested-safe)
            try:
                _in_langchain.set(True)               # guard True ONLY around the pull
            except Exception:                         # set() has no failure mode; guarded defensively anyway
                pass
            enter_langchain_trace(tid)                # Backstop (process-global)
            try:
                try:
                    chunk = await it.__anext__()
                except StopAsyncIteration:
                    break
                except Exception as _exc:
                    # Provider failure surfacing on a pull — see the sync twin
                    # (_guard_sync_stream): stamp the failed outcome for the
                    # deferred ERROR span's flush; pull-scoped so
                    # GeneratorExit / consumer-break stay untouched; identity
                    # re-raise (golden rule).
                    try:
                        session._call_outcome = build_call_outcome(
                            _exc, int((_time.monotonic() - _t0) * 1000))
                        outcome_set = True
                    except Exception:
                        pass
                    raise
            finally:
                leave_langchain_trace(tid)
                try:
                    _in_langchain.set(prev)           # restore BEFORE yield; set() cannot raise
                except Exception:
                    pass
            acc = _accumulate_langchain_chunk(acc, chunk)
            yield chunk                               # customer holds here with guard = prior
    finally:                                          # (B) unwind ONE interleave level
        try:
            d = getattr(session, '_lc_stream_depth', 1) - 1
            session._lc_stream_depth = d if d > 0 else 0
        except Exception:
            d = 0
        if d <= 0:                                    # outermost: restore baseline + flush ONCE
            _finalize_langchain_stream(
                session, getattr(session, '_lc_stream_defer_prev', False), acc, order,
                obs_key=_obs_key)
            # Stale-outcome clear — see the sync twin (_guard_sync_stream):
            # an empty-buffer flush early-returns without clearing, and a
            # leftover failed outcome would mislabel the session's NEXT row.
            # Outermost + flush-actually-ran only.
            try:
                if (outcome_set and not getattr(session, '_defer_telemetry', False)
                        and getattr(session, '_call_outcome', None) is not None):
                    session._call_outcome = None
            except Exception:
                pass
        else:                                         # inner level still open: capture own comp only
            try:
                if acc is not None:
                    _capture_response_composition("langchain", acc, order=order)
            except Exception:
                pass


def _set_langchain_wrapper(cls, method_name, original, kind: str):
    """Install a LangChain wrapper. `kind` is one of sync/async/stream/astream."""
    provider = "langchain"

    if kind == "sync":
        @functools.wraps(original)
        def lc_sync(*args, **kwargs):
            if in_langchain() or in_litellm() or in_pydantic_ai() or in_agno():
                return original(*args, **kwargs)
            # Resolve the session once and thread it into the check so a REROUTE
            # audit stash lands on the SAME object flushed below.
            session = get_current_session()
            tid = _lc_trace_id(session)
            _hint, _prov = _lc_check_ctx(args)
            _run_sync_check(session=session, model_hint=_hint, provider=_prov)
            # This call's obs key, captured right after the check.
            _obs_key = _state.get_current_obs_key()
            _capture_langchain_prompt(args)
            prev_defer = getattr(session, '_defer_telemetry', False)
            session._defer_telemetry = True
            # Failure-outcome anchor: post-check, pre-provider-call.
            _t0 = _time.monotonic()
            _in_langchain.set(True)
            enter_langchain_trace(tid)                # Process-global twin of contextvar
            try:
                try:
                    result = original(*args, **kwargs)
                finally:
                    leave_langchain_trace(tid)
                    _in_langchain.set(False)
                    session._defer_telemetry = prev_defer
            except Exception as _exc:
                # Provider rejection on the eager path (e.g. 401): the raw
                # provider wrapper short-circuited under our guard, so this
                # wrapper is the SOLE emitter — without a failure row the call
                # vanishes (L-2). State is already restored by the inner
                # finally, so the emission below cannot be swallowed by a
                # defer/guard seam. Outermost defer level ONLY: under an outer
                # defer window the exception propagates and the outermost
                # owner emits exactly once. Emission is fully fail-open;
                # identity re-raise (golden rule).
                try:
                    if not prev_defer:
                        session._call_outcome = build_call_outcome(
                            _exc, int((_time.monotonic() - _t0) * 1000))
                        try:
                            if getattr(session, '_deferred_spans', None):
                                # The instrumentor queued an ERROR span —
                                # flush it as the canonical failed row (the
                                # outcome rides on the first flushed payload).
                                _flush_deferred_spans(session, obs_key=_obs_key)
                            else:
                                # No deferred span (rejection pre-
                                # instrumentor) — synthetic failed row via the
                                # raw-path machinery. LangChain keeps the
                                # model on the instance, never in kwargs:
                                # thread the check context through a synthetic
                                # kwargs so the row isn't model="unknown"; the
                                # framework tag backstops an unresolvable
                                # provider so the row is never provider-blank.
                                _stash_attempt_context(
                                    session, _prov or provider, "langchain",
                                    (), {"model": _hint})
                                _emit_call_failure_log(
                                    get_client(), session, obs_key=_obs_key)
                        finally:
                            # Stale-outcome guard: an empty-buffer flush
                            # early-returns WITHOUT clearing (and a throwing
                            # flush/emit must not skip the clear) — a leftover
                            # failed outcome would mislabel the session's
                            # NEXT row.
                            if getattr(session, '_call_outcome', None) is not None:
                                session._call_outcome = None
                except Exception:
                    pass
                raise
            # G3-O1: stash reasoning BEFORE the response capture (which clears
            # the mailbox the order fallback reads).
            _stash_lc_reasoning_tokens(session, _lc_reasoning_from_llmresult(result))
            _capture_response_composition(provider, result)
            if not session._defer_telemetry:
                _flush_deferred_spans(session, obs_key=_obs_key)
            return result
        setattr(cls, method_name, lc_sync)

    elif kind == "async":
        @functools.wraps(original)
        async def lc_async(*args, **kwargs):
            if in_langchain() or in_litellm() or in_pydantic_ai() or in_agno():
                return await original(*args, **kwargs)
            # Resolve the session once and thread it into the check so a REROUTE
            # audit stash lands on the SAME object flushed below.
            session = get_current_session()
            tid = _lc_trace_id(session)
            _hint, _prov = _lc_check_ctx(args)
            await _run_async_check(session=session, model_hint=_hint, provider=_prov)
            # This call's obs key, captured right after the check.
            _obs_key = _state.get_current_obs_key()
            _capture_langchain_prompt(args)
            prev_defer = getattr(session, '_defer_telemetry', False)
            session._defer_telemetry = True
            # Failure-outcome anchor: post-check, pre-provider-call.
            _t0 = _time.monotonic()
            _in_langchain.set(True)
            enter_langchain_trace(tid)                # Process-global twin of contextvar
            try:
                try:
                    result = await original(*args, **kwargs)
                finally:
                    leave_langchain_trace(tid)
                    _in_langchain.set(False)
                    session._defer_telemetry = prev_defer
            except Exception as _exc:
                # Provider rejection on the eager path — see the sync twin
                # (lc_sync): this wrapper is the SOLE emitter (L-2). State is
                # already restored by the inner finally; outermost defer level
                # only; fully fail-open; identity re-raise (golden rule).
                try:
                    if not prev_defer:
                        session._call_outcome = build_call_outcome(
                            _exc, int((_time.monotonic() - _t0) * 1000))
                        try:
                            if getattr(session, '_deferred_spans', None):
                                _flush_deferred_spans(session, obs_key=_obs_key)
                            else:
                                _stash_attempt_context(
                                    session, _prov or provider, "langchain",
                                    (), {"model": _hint})
                                _emit_call_failure_log(
                                    get_client(), session, obs_key=_obs_key)
                        finally:
                            # Stale-outcome guard — see the sync twin
                            # (unconditional: a throwing flush/emit must not
                            # skip the clear).
                            if getattr(session, '_call_outcome', None) is not None:
                                session._call_outcome = None
                except Exception:
                    pass
                raise
            # G3-O1: stash reasoning BEFORE the response capture — see lc_sync.
            _stash_lc_reasoning_tokens(session, _lc_reasoning_from_llmresult(result))
            _capture_response_composition(provider, result)
            if not session._defer_telemetry:
                _flush_deferred_spans(session, obs_key=_obs_key)
            return result
        setattr(cls, method_name, lc_async)

    elif kind == "stream":
        @functools.wraps(original)
        def lc_stream(*args, **kwargs):
            if in_langchain() or in_litellm() or in_pydantic_ai() or in_agno():
                return original(*args, **kwargs)
            # Resolve the session once and thread it into BOTH the check and the
            # stream guard (which flushes via _finalize_langchain_stream) so a
            # REROUTE audit stash lands on the SAME object that gets flushed.
            session = get_current_session()
            _hint, _prov = _lc_check_ctx(args)
            _run_sync_check(session=session, model_hint=_hint, provider=_prov)
            # This stream's obs key, captured right after the check — the
            # guard finalizes (and drains) at consumer-drain time.
            _obs_key = _state.get_current_obs_key()
            # Snapshot this stream's own LLM-span order so an interleaved stream
            # that has since overwritten the shared mailbox can't mis-key its response.
            order = _capture_langchain_prompt(args)
            return _guard_sync_stream(original(*args, **kwargs), order, session=session,
                                      obs_key=_obs_key)
        setattr(cls, method_name, lc_stream)

    elif kind == "astream":
        @functools.wraps(original)
        def lc_astream(*args, **kwargs):
            if in_langchain() or in_litellm() or in_pydantic_ai() or in_agno():
                return original(*args, **kwargs)
            # astream() returns an async iterator synchronously — defer the
            # async check + iteration into a wrapping async generator.
            return _guard_async_stream(original, args, kwargs)
        setattr(cls, method_name, lc_astream)


# ═══════════════════════════════════════════════════════════════════
# LlamaIndex wrappers — for llama_index.llms.{openai,anthropic,google_genai}.
# {OpenAI,Anthropic,GoogleGenAI}.{chat,achat,stream_chat,astream_chat}.
#
# LlamaIndex is a framework: each provider class internally calls the underlying
# provider SDK (openai, anthropic, google.genai) which is ALSO patched. These
# wrappers run the SINGLE pre-flight check, capture prompt/response composition
# from LlamaIndex's own ChatMessage objects, and hold the _in_llamaindex guard
# so the nested provider wrapper passes straight through. Telemetry flows
# through the inner provider's OpenLLMetry span — no LlamaIndex-specific
# OpenLLMetry instrumentor is required.
# ═══════════════════════════════════════════════════════════════════

@fail_safe
def _capture_llamaindex_prompt(args, kwargs):
    """Capture prompt composition for a LlamaIndex chat/stream_chat call.

    LlamaIndex chat/achat/stream_chat/astream_chat receive `messages` as the
    first positional arg after `self` (args[1]) — a List[ChatMessage]. The
    "llamaindex" composition parser handles this shape (and falls back for
    str / single ChatMessage if seen elsewhere).
    """
    if len(args) >= 2:
        payload = args[1]
    elif "messages" in kwargs:
        payload = kwargs["messages"]
    else:
        return
    session = get_current_session()
    if not hasattr(session, '_pending_compositions'):
        session._pending_compositions = {}
    # Enforcer runs BEFORE the inner provider's OpenLLMetry on_start, so
    # _span_counter is still at the value next_span_order() will consume.
    order = session._span_counter
    comp = build_prompt_composition("llamaindex", {"messages": payload})
    if comp:
        comp_key = f"{session.trace_id}:{order}"
        session._pending_compositions.setdefault(comp_key, {})["prompt"] = comp


def _li_provider_from_instance(instance) -> str:
    """Map a LlamaIndex provider LLM instance to a normalized provider name."""
    cls = type(instance).__name__.lower() if instance is not None else ""
    if "anthropic" in cls:
        return "anthropic"
    if "google" in cls or "gemini" in cls:
        return "google"
    # B1: OpenAIResponses is a SIBLING of the OpenAI class (its own
    # chat/stream_chat against /v1/responses) with a different usage shape —
    # map it to the Responses pseudo-provider so the pre-flight, the usage
    # extractor and the row all agree. Serving identity stays openai
    # (collector provider-identity.js maps openai_responses → openai).
    if "responses" in cls:
        return "openai_responses"
    return "openai"  # OpenAI default — also covers OpenAI-compatible subclasses


def _li_check_ctx(args):
    """``(model_hint, provider)`` from the bound LlamaIndex LLM
    (``args[0]``). The model lives on the instance, never in the call kwargs,
    so the pre-flight used to evaluate with an empty model and no provider.
    Hint is for matching + audit only — it never reaches kwargs. Never raises."""
    instance = args[0] if args else None
    if instance is None:
        return None, None
    try:
        provider = _li_provider_from_instance(instance)
    except Exception:
        provider = None
    return _instance_model_attr(instance, "model"), provider


@fail_safe
def _extract_li_usage_py(instance, last_response):
    """Extract token usage from a LlamaIndex ChatResponse (streamed last chunk
    or non-stream response). Provider-aware: handles the multiple shapes
    LlamaIndex normalizes usage into across openai / anthropic / google.

    Returns (model, input_tokens, output_tokens, cached_tokens).
    """
    fallback_model = str(getattr(instance, "model", None) or "unknown")
    if last_response is None:
        return fallback_model, 0, 0, 0
    provider = _li_provider_from_instance(instance)

    msg_kwargs = {}
    msg = getattr(last_response, "message", None)
    if msg is not None:
        msg_kwargs = getattr(msg, "additional_kwargs", None) or {}
    resp_kwargs = getattr(last_response, "additional_kwargs", None) or {}
    raw = getattr(last_response, "raw", None)

    if provider == "anthropic":
        # LlamaIndex Anthropic stream_chat sets
        # message.additional_kwargs["usage"] = {input_tokens, output_tokens,
        # cache_read_input_tokens?, ...} on every chunk; the last chunk has
        # the final values. Non-stream response: raw is an Anthropic Message
        # with .usage.input_tokens / output_tokens.
        usage = (msg_kwargs.get("usage") if isinstance(msg_kwargs, dict) else None) or {}
        if not usage and raw is not None:
            u = getattr(raw, "usage", None)
            if u is not None:
                usage = {
                    "input_tokens": getattr(u, "input_tokens", 0) or 0,
                    "output_tokens": getattr(u, "output_tokens", 0) or 0,
                    "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                }
        cached = int(usage.get("cache_read_input_tokens", 0) or 0)
        inp = int(usage.get("input_tokens", 0) or 0)
        out = int(usage.get("output_tokens", 0) or 0)
        model = str(getattr(raw, "model", None) or fallback_model)
        return model, inp, out, cached

    if provider == "google":
        # google.genai response: usage_metadata = {prompt_token_count,
        # candidates_token_count, cached_content_token_count, ...}
        um = None
        if raw is not None:
            um = getattr(raw, "usage_metadata", None)
            if um is None and isinstance(raw, dict):
                um = raw.get("usage_metadata")
        if um is None:
            um = msg_kwargs.get("usage_metadata") if isinstance(msg_kwargs, dict) else None
        um = um or {}
        if not isinstance(um, dict):
            um = {
                "prompt_token_count": getattr(um, "prompt_token_count", 0) or 0,
                "candidates_token_count": getattr(um, "candidates_token_count", 0) or 0,
                "cached_content_token_count": getattr(um, "cached_content_token_count", 0) or 0,
                "thoughts_token_count": getattr(um, "thoughts_token_count", 0) or 0,
            }
        prompt = int(um.get("prompt_token_count", 0) or 0)
        cached = int(um.get("cached_content_token_count", 0) or 0)
        inp = max(0, prompt - cached)
        out = int(um.get("candidates_token_count", 0) or 0) + int(
            um.get("thoughts_token_count", 0) or 0
        )
        model = str(
            (getattr(raw, "model_version", None) if raw is not None else None)
            or fallback_model
        )
        return model, inp, out, cached

    if provider == "openai_responses":
        # B1 — OpenAI Responses API via LlamaIndex's OpenAIResponses. It stows
        # the verbatim `ResponseUsage` OBJECT (not loose ints) on the
        # ChatResponse's OWN additional_kwargs on every path: non-stream
        # chat/achat set it from `response.usage`, and the stream accumulates
        # it from the terminal `response.completed` event. `raw` is the
        # `Response` on the non-stream paths but the last stream EVENT on the
        # streamed one, where the Response nests under `.response` — so unwrap
        # that first and read the model off whichever object carries usage.
        # `input_tokens` is cache-INCLUSIVE (subtract input_tokens_details
        # .cached_tokens); `output_tokens` already includes reasoning_tokens,
        # so they are deliberately not added again. Mirrors Node's
        # openai_responses branch and the direct-SDK mapping above, including
        # its degrade-to-zeros catch: a hostile shape must still produce a row
        # (the caller unpacks a 4-tuple), never lose the call to an exception.
        def _resp_attr(obj, key):
            if obj is None:
                return None
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        try:
            resp_obj = _resp_attr(raw, "response")
            if _resp_attr(resp_obj, "usage") is None:
                resp_obj = raw
            usage = resp_kwargs.get("usage") if isinstance(resp_kwargs, dict) else None
            if usage is None:
                usage = _resp_attr(resp_obj, "usage")
            total_in = int(_resp_attr(usage, "input_tokens") or 0)
            cached = int(_resp_attr(_resp_attr(usage, "input_tokens_details"),
                                    "cached_tokens") or 0)
            inp = max(0, total_in - cached)
            out = int(_resp_attr(usage, "output_tokens") or 0)
            model = str(_resp_attr(resp_obj, "model") or fallback_model)
            return model, inp, out, cached
        except Exception:
            return fallback_model, 0, 0, 0

    # OpenAI default — LlamaIndex OpenAI stream_chat puts token counts on
    # ChatResponse.additional_kwargs (via _get_response_token_counts).
    # Non-streaming: raw is a ChatCompletion with .usage.{prompt|completion}_tokens.
    prompt = int(resp_kwargs.get("prompt_tokens", 0) or 0)
    out = int(resp_kwargs.get("completion_tokens", 0) or 0)
    cached = 0
    if (prompt == 0 and out == 0) and raw is not None:
        u = getattr(raw, "usage", None)
        if u is not None:
            prompt = int(getattr(u, "prompt_tokens", 0) or 0)
            out = int(getattr(u, "completion_tokens", 0) or 0)
            ptd = getattr(u, "prompt_tokens_details", None)
            if ptd is not None:
                cached = int(getattr(ptd, "cached_tokens", 0) or 0)
    inp = max(0, prompt - cached)
    model = str(
        (getattr(raw, "model", None) if raw is not None else None) or fallback_model
    )
    return model, inp, out, cached


def _li_google_verbatim_usage_py(instance, last_response):
    """Build a verbatim ``google_genai`` usage block for a LlamaIndex Gemini call,
    or ``None`` to fall back to the positional counts.

    ``mapGoogleGenAI`` treats ``usage_metadata`` as cache-INCLUSIVE (it subtracts
    ``cached_content_token_count`` itself), so forwarding the raw metadata avoids
    the double cache-subtraction the pre-netted positional counts would otherwise
    trigger — the F1 under-bill. Fail-safe: any missing/hostile shape returns
    ``None`` and ``_log_li_py`` keeps today's positional behavior.

    Object metadata is forwarded verbatim too (F7): a dict passes through, and a
    non-dict object is generically serialized (``_as_dict`` + json round-trip)
    so ``prompt_tokens_details`` / ``cache_tokens_details`` /
    ``tool_use_prompt_token_count`` survive — folding those out would lose the
    AUDIO premium at the server. Only an exotic object that can't serialize to
    a dict falls back to the 4-scalar snake_case rebuild.

    The forwarded raw is a JSON snapshot, not a reference: the log POST
    serializes in a background thread, where a nested non-serializable value
    (pydantic object, set) would drop the whole row — snapshotting here surfaces
    that failure NOW and falls back to the positional counts instead. It also
    freezes the block against later mutation of the source dict.
    """
    try:
        if last_response is None:
            return None
        raw = getattr(last_response, "raw", None)
        msg = getattr(last_response, "message", None)
        msg_kwargs = getattr(msg, "additional_kwargs", None) if msg is not None else None
        um = None
        if raw is not None:
            um = getattr(raw, "usage_metadata", None)
            if um is None and isinstance(raw, dict):
                um = raw.get("usage_metadata")
        if um is None and isinstance(msg_kwargs, dict):
            um = msg_kwargs.get("usage_metadata")
        if um is None:
            return None
        if isinstance(um, dict):
            # snake_case keys are mapper-known → forward verbatim.
            block_raw = um
        else:
            # Object metadata → serialize verbatim so modality/cache details
            # (prompt_tokens_details, cache_tokens_details,
            # tool_use_prompt_token_count, any future fields) survive; without
            # them the server folds all prompt tokens into text and loses the
            # AUDIO premium. The real type's model_dump() emits snake_case and
            # the server reads snake_case first, so no casing translation is
            # needed. Mirrors _serialize_anthropic_raw_usage's idiom (_as_dict +
            # json round-trip with the __dict__/str default).
            block_raw = None
            try:
                serialized = _json.loads(_json.dumps(
                    _as_dict(um),
                    default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
                ))
                if isinstance(serialized, dict) and serialized:
                    block_raw = serialized
            except Exception:
                block_raw = None
            if block_raw is None:
                # Last resort for exotic objects generic serialization can't
                # turn into a dict (e.g. slotted / no __dict__): rebuild the
                # snake_case scalars the extractor uses. Never worse than today.
                block_raw = {
                    "prompt_token_count": int(getattr(um, "prompt_token_count", 0) or 0),
                    "candidates_token_count": int(
                        getattr(um, "candidates_token_count", 0) or 0
                    ),
                    "cached_content_token_count": int(
                        getattr(um, "cached_content_token_count", 0) or 0
                    ),
                    "thoughts_token_count": int(
                        getattr(um, "thoughts_token_count", 0) or 0
                    ),
                }
        # Only forward when at least one token count is positive; an empty /
        # all-zero block maps to zeros server-side, so fall back instead.
        positive = False
        for v in block_raw.values():
            try:
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                    positive = True
                    break
            except (TypeError, ValueError):
                continue
        if not positive:
            return None
        # Snapshot: non-serializable nested values raise here (→ None → positional
        # fallback) instead of killing the background POST and losing the row.
        block_raw = _json.loads(_json.dumps(block_raw))
        return {"shape": "google_genai", "raw": block_raw}
    except Exception:
        return None


def _anthropic_usage_has_positive(raw_dict):
    """True if a serialized anthropic usage dict carries any positive token
    count (top-level input/output/cache totals or the nested cache_creation
    5m/1h split). An all-zero block maps to zeros server-side — worse than the
    positional fallback — so callers gate on this before forwarding."""
    if not isinstance(raw_dict, dict):
        return False
    for k in ("input_tokens", "output_tokens",
              "cache_read_input_tokens", "cache_creation_input_tokens"):
        v = raw_dict.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return True
    cc = raw_dict.get("cache_creation")
    if isinstance(cc, dict):
        for v in cc.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return True
    return False


def _li_anthropic_usage_from_chunk(chunk):
    """Return a JSON-safe anthropic usage dict from a LlamaIndex Anthropic
    stream chunk's underlying event (``chunk.raw``), or ``None``.

    LlamaIndex's Anthropic stream_chat sets message.additional_kwargs["usage"]
    to ONLY {input_tokens, output_tokens} and drops cache_read/cache_creation
    entirely, so the cache-write tokens survive nowhere but the raw event it
    carries on each chunk as ``raw=dict(event)``: message_start's
    ``raw["message"].usage`` holds the input-side usage (incl. the nested
    cache_creation 5m/1h split), message_delta's ``raw["usage"]`` the final
    output_tokens. Every other event type has no usage → ``None``. Handles
    ``raw`` as either the ``dict(event)`` LlamaIndex builds or a raw event
    object, defensively across versions. Fully fail-open."""
    try:
        raw = getattr(chunk, "raw", None)
        if raw is None:
            return None

        def _field(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        etype = _field(raw, "type")
        usage_obj = None
        msg = _field(raw, "message")
        if etype == "message_start" or msg is not None:
            if msg is not None:
                usage_obj = _field(msg, "usage")
        if usage_obj is None and (etype == "message_delta"
                                  or _field(raw, "usage") is not None):
            usage_obj = _field(raw, "usage")
        if usage_obj is None:
            return None
        return _serialize_anthropic_raw_usage(usage_obj)
    except Exception:
        return None


def _merge_li_anthropic_stream_usage(acc, chunk):
    """Field-wise merge the anthropic usage on a LlamaIndex stream chunk into
    ``acc`` (mutated in place) so the full cache-aware usage is reconstructed
    across the message_start (input side + cache) and message_delta (final
    output) chunks. Merge semantics: numerics take the max (a later chunk that
    re-sends the same field never inflates it), nested dicts merge key-wise,
    other non-null values overwrite. Fully fail-open — a hostile chunk never
    breaks stream iteration or corrupts prior accumulation, and it never
    consumes/alters the chunk (pass-through iteration stays byte-identical)."""
    try:
        src = _li_anthropic_usage_from_chunk(chunk)
        _merge_anthropic_usage_dict(acc, src)
    except Exception:
        pass


def _merge_anthropic_usage_dict(acc, src):
    """Recursive field-wise merge helper for _merge_li_anthropic_stream_usage:
    numerics take the max, nested dicts merge key-wise, other non-null values
    overwrite. ``acc`` is mutated in place; a non-dict ``src`` is ignored."""
    if not isinstance(acc, dict) or not isinstance(src, dict):
        return
    for k, v in src.items():
        if v is None:
            continue
        if isinstance(v, dict):
            existing = acc.get(k)
            if not isinstance(existing, dict):
                existing = {}
            _merge_anthropic_usage_dict(existing, v)
            acc[k] = existing
        elif isinstance(v, bool):
            acc[k] = v
        elif isinstance(v, (int, float)):
            prev = acc.get(k)
            if isinstance(prev, (int, float)) and not isinstance(prev, bool):
                acc[k] = max(prev, v)
            else:
                acc[k] = v
        else:
            acc[k] = v


def _li_anthropic_verbatim_usage_py(last_response, stream_usage=None):
    """Build a verbatim ``anthropic_messages`` usage block for a LlamaIndex
    Anthropic call, or ``None`` to fall back to the positional counts.

    LlamaIndex's Anthropic path drops cache-write tokens: its
    additional_kwargs["usage"] carries only input/output, so pricing the
    positional counts bills cache writes (cache_creation_input_tokens, split
    5m/1h) at $0 — the F4 under-bill. Forwarding the verbatim anthropic usage
    keeps input_tokens cache-EXCLUSIVE and prices cache writes at the write
    rate, mirroring the native anthropic stream path and the F1 google
    precedent.

    Source priority: (a) the merged stream usage dict reconstructed across the
    message_start/message_delta chunks (the only place cache writes survive on
    a stream), (b) the stream last-chunk message.additional_kwargs["usage"]
    dict, (c) the non-stream response's ``raw.usage`` object (or ``raw["usage"]``
    when raw is a dict). Fail-open: any failure, or usage that is empty/all-zero,
    returns ``None`` and the positional counts stand. ``_serialize_anthropic_raw_usage``
    JSON-snapshots the block, so a non-serializable nested value surfaces here
    (→ ``None``) instead of dropping the whole row in the background POST."""
    try:
        usage_obj = None
        if isinstance(stream_usage, dict) and stream_usage:
            usage_obj = stream_usage
        if usage_obj is None and last_response is not None:
            msg = getattr(last_response, "message", None)
            msg_kwargs = getattr(msg, "additional_kwargs", None) if msg is not None else None
            if isinstance(msg_kwargs, dict):
                u = msg_kwargs.get("usage")
                if isinstance(u, dict) and u:
                    usage_obj = u
        if usage_obj is None and last_response is not None:
            raw = getattr(last_response, "raw", None)
            if raw is not None:
                u = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
                if u is not None:
                    usage_obj = u
        if usage_obj is None:
            return None
        raw_dict = _serialize_anthropic_raw_usage(usage_obj)
        if not raw_dict or not _anthropic_usage_has_positive(raw_dict):
            return None
        return {"shape": "anthropic_messages", "raw": raw_dict}
    except Exception:
        return None


def _openai_usage_has_positive(raw_dict):
    """True if a serialized OpenAI-shaped usage dict carries a positive
    prompt/completion count. An all-zero block maps to zeros server-side — worse
    than the positional fallback — so callers gate on this before forwarding."""
    if not isinstance(raw_dict, dict):
        return False
    for k in ("prompt_tokens", "completion_tokens"):
        v = raw_dict.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return True
    return False


def _li_openai_verbatim_usage_py(last_response):
    """Build a verbatim ``openai_compatible_chat`` usage block for a LlamaIndex
    OpenAI (or OpenAI-compatible) call, or ``None`` to fall back to the
    positional counts.

    OpenAI ``prompt_tokens`` is cache-INCLUSIVE and ``mapOpenAICompatChat``
    subtracts ``prompt_tokens_details.cached_tokens`` itself. LlamaIndex's OpenAI
    extractor surfaces only loose token ints on additional_kwargs and never
    surfaces ``prompt_tokens_details``, so with no verbatim block the cache reads
    vanish and all input bills at the full rate (over-bill); and the synthesised
    fallback would net an already-netted positional input a SECOND time (latent
    double-subtract). Forwarding the raw OpenAI usage verbatim cures both, so the
    mapper nets cached exactly once — mirrors the F1 google / F4 anthropic
    helpers and Node's F1 openai branch.

    Source: the non-stream response's / stream last chunk's ``raw.usage`` object
    (or ``raw["usage"]`` when raw is a dict); as a fallback, a full usage OBJECT
    under ``message.additional_kwargs["usage"]`` — LlamaIndex's own loose
    token-count ints there are NOT a usage object and are never fabricated into
    one. Fail-open: any missing / hostile / all-zero shape returns ``None`` and
    the positional counts stand. ``_serialize_anthropic_raw_usage`` is a generic
    JSON snapshotter (not anthropic-specific): it preserves nested
    prompt/completion_tokens_details verbatim and surfaces a non-serializable
    nested value HERE (→ ``None`` → positional fallback) instead of dropping the
    whole row in the background POST."""
    try:
        if last_response is None:
            return None
        usage_obj = None
        raw = getattr(last_response, "raw", None)
        if raw is not None:
            usage_obj = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
        if usage_obj is None:
            msg = getattr(last_response, "message", None)
            msg_kwargs = getattr(msg, "additional_kwargs", None) if msg is not None else None
            if isinstance(msg_kwargs, dict):
                usage_obj = msg_kwargs.get("usage")
        if usage_obj is None:
            return None
        raw_dict = _serialize_anthropic_raw_usage(usage_obj)
        if not raw_dict or not _openai_usage_has_positive(raw_dict):
            return None
        return {"shape": "openai_compatible_chat", "raw": raw_dict}
    except Exception:
        return None


def _openai_responses_usage_has_positive(raw_dict):
    """True if a serialized OpenAI Responses usage dict carries a positive
    input/output count. An all-zero block maps to zeros server-side — worse than
    the positional fallback — so callers gate on this before forwarding."""
    if not isinstance(raw_dict, dict):
        return False
    for k in ("input_tokens", "output_tokens"):
        v = raw_dict.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
            return True
    return False


def _li_openai_responses_verbatim_usage_py(last_response):
    """Build a verbatim ``openai_responses`` usage block for a LlamaIndex
    OpenAIResponses call (B1), or ``None`` to fall back to the positional counts.

    Responses ``input_tokens`` is cache-INCLUSIVE and ``mapOpenAIResponses``
    subtracts ``input_tokens_details.cached_tokens`` itself, so forwarding the
    raw usage is what makes the server net cached exactly ONCE — the positional
    input the extractor returns is already netted and would otherwise be netted
    a second time. Forwarding verbatim also preserves
    ``output_tokens_details.reasoning_tokens`` and any future detail fields,
    which the 4-scalar positional row cannot carry.

    Source priority: the ChatResponse's own ``additional_kwargs["usage"]`` (the
    ``ResponseUsage`` object LlamaIndex sets on every path), then the terminal
    stream event's ``raw.response.usage``, then a non-stream ``raw.usage``.
    Fail-open: any missing / hostile / all-zero shape returns ``None`` and the
    positional counts stand. ``_serialize_anthropic_raw_usage`` is a generic
    JSON snapshotter (not anthropic-specific): it preserves the nested
    input/output ``_tokens_details`` verbatim and surfaces a non-serializable
    nested value HERE (→ ``None`` → positional fallback) instead of dropping the
    whole row in the background POST."""
    try:
        if last_response is None:
            return None
        usage_obj = None
        resp_kwargs = getattr(last_response, "additional_kwargs", None)
        if isinstance(resp_kwargs, dict):
            usage_obj = resp_kwargs.get("usage")
        if usage_obj is None:
            raw = getattr(last_response, "raw", None)
            nested = (raw.get("response") if isinstance(raw, dict)
                      else getattr(raw, "response", None)) if raw is not None else None
            for src in (nested, raw):
                if src is None:
                    continue
                u = src.get("usage") if isinstance(src, dict) else getattr(src, "usage", None)
                if u is not None:
                    usage_obj = u
                    break
        if usage_obj is None:
            return None
        raw_dict = _serialize_anthropic_raw_usage(usage_obj)
        if not raw_dict or not _openai_responses_usage_has_positive(raw_dict):
            return None
        return {"shape": "openai_responses", "raw": raw_dict}
    except Exception:
        return None


@fail_safe
def _log_li_py(instance, last_response, order, span_name, start_time,
               stream_usage=None, latency=None, call_outcome=None,
               observations=None, local_decision=None,
               obs_key=_OBS_KEY_CURRENT):
    """Manual log for a LlamaIndex call. Used when we cannot rely on the inner
    provider's OpenLLMetry span (currently: streaming, where OpenLLMetry's
    anthropic instrumentor often fails to surface usage). Drops any spans
    deferred during the call so we don't double-log.

    ``stream_usage`` (anthropic streams only) is the cache-aware usage dict the
    drain merged across message_start/message_delta chunks — the verbatim block
    prefers it since the LlamaIndex last chunk drops cache writes.

    ``obs_key`` is the emitting call's obs key, used ONLY to key this row's
    ``_tp_routing`` stamp. The streaming callers run this from a generator
    ``finally`` in the CONSUMER's context, where the contextvar can still hold a
    SIBLING call's key (``mint_obs_key`` sets it with no reset token) — that
    would stamp a stranger's reroute onto this row, the exact B4 failure. They
    pass the key captured at wrapper entry instead. The sentinel default keeps
    the live contextvar, correct for the in-context callers.

    ``latency`` is the ``_build_stream_latency`` dict from the streaming callers
    — the SOLE source of is_streaming/ttft_ms on the /log payload (the deferred
    inner span that would otherwise carry them is dropped). None on the
    non-streaming callers and whenever the anchor is missing.

    ``call_outcome`` is the streaming guards' failure classification (the
    inner instrumentor's ERROR span is dropped above, so this row is the only
    carrier). None (the default) keeps the payload byte-identical to before —
    the client omits an absent call_outcome and the collector stamps
    success.

    ``observations`` / ``local_decision`` (B1) are the v2 extras
    ``_flush_deferred_spans`` normally attaches to the first deferred span. The
    LlamaIndex OpenAIResponses non-stream paths never reach that flush (their
    inner Responses.create emits no span), so they drain + pop them and hand
    them here instead — otherwise this call's `unappliable_call_shape`
    observation would sit in the queue until OBS_STALE_SECONDS and then be
    swept onto an unrelated later call's row. Both default None, which the
    client truthiness-gates exactly like the flush does, so every existing
    caller's payload stays byte-identical."""
    tp = get_client()
    if not tp:
        return
    session = get_current_session()
    provider = _li_provider_from_instance(instance)
    model, input_tokens, output_tokens, cached_tokens = _extract_li_usage_py(
        instance, last_response
    )
    # Gemini usage_metadata is cache-INCLUSIVE; forward it verbatim so the mapper
    # nets cached_content_token_count once (the positional input_tokens above is
    # already netted, which the mapper would net again → the F1 under-bill).
    usage_block = None
    if provider == "google":
        usage_block = _li_google_verbatim_usage_py(instance, last_response)
    elif provider == "anthropic":
        # Anthropic input_tokens is cache-EXCLUSIVE and LlamaIndex drops cache
        # writes from its usage dict — forward the verbatim shape so writes are
        # priced at the write rate (F4 under-bill). When it builds, reconcile
        # the positional counts from the same raw so cached_tokens is reads-only
        # (writes billed via the block) and input stays exclusive — mirrors the
        # pydantic_ai and native anthropic paths.
        usage_block = _li_anthropic_verbatim_usage_py(last_response, stream_usage)
        if usage_block is not None:
            raw = usage_block.get("raw") or {}
            input_tokens = int(raw.get("input_tokens", input_tokens) or 0)
            output_tokens = int(raw.get("output_tokens", output_tokens) or 0)
            cached_tokens = int(raw.get("cache_read_input_tokens", 0) or 0)
    elif provider == "openai_responses":
        # B1 — Responses input_tokens is cache-INCLUSIVE and the
        # openai_responses mapper subtracts input_tokens_details.cached_tokens
        # itself. Must be checked BEFORE the openai else-branch: that branch
        # reads a prompt_tokens/completion_tokens shape the Responses usage
        # never has, so it would forward nothing and bill 0. Reconcile the
        # positional counts from the SAME raw so the mapper nets cached once —
        # mirrors the anthropic/openai branches above.
        usage_block = _li_openai_responses_verbatim_usage_py(last_response)
        if usage_block is not None:
            raw = usage_block.get("raw") or {}
            total_in = int(raw.get("input_tokens", 0) or 0)
            itd = raw.get("input_tokens_details")
            cached = int(itd.get("cached_tokens", 0) or 0) if isinstance(itd, dict) else 0
            input_tokens = max(0, total_in - cached)
            output_tokens = int(raw.get("output_tokens", output_tokens) or 0)
            cached_tokens = cached
    else:
        # OpenAI / OpenAI-compatible: prompt_tokens is cache-INCLUSIVE and the
        # mapper subtracts prompt_tokens_details.cached_tokens itself. LlamaIndex
        # drops that detail (cache reads billed at full rate) and the synthesised
        # fallback would net an already-netted input twice — forward the verbatim
        # raw usage and reconcile the positional counts from the same raw so the
        # mapper nets cached once. Mirrors Node's F1 openai branch.
        usage_block = _li_openai_verbatim_usage_py(last_response)
        if usage_block is not None:
            raw = usage_block.get("raw") or {}
            prompt = int(raw.get("prompt_tokens", 0) or 0)
            ptd = raw.get("prompt_tokens_details")
            cached = int(ptd.get("cached_tokens", 0) or 0) if isinstance(ptd, dict) else 0
            input_tokens = max(0, prompt - cached)
            output_tokens = int(raw.get("completion_tokens", output_tokens) or 0)
            cached_tokens = cached

    span_obj = {
        **manual_span_ids(session),
        "span_kind": "llm",
        "span_name": span_name or model,
        "span_order": order,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }
    metadata = {"workflow_name": session.workflow_name}
    if session.session_id:
        metadata["session_id"] = session.session_id
    # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
    # session metadata WITHOUT it, then re-add it only when this row belongs
    # to the call that was actually rerouted (exact obs-key match).
    _copy_session_metadata(metadata, session)
    _stamp_routing_marker(metadata, session, _resolve_obs_key(obs_key))

    prompt_comp = []
    response_comp = []
    comp_key = f"{session.trace_id}:{order}"
    if hasattr(session, "_pending_compositions"):
        comp_data = session._pending_compositions.pop(comp_key, {})
        prompt_comp = comp_data.get("prompt", [])
        response_comp = comp_data.get("response", [])

    tp.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model=model,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        metadata=metadata,
        span=span_obj,
        prompt_composition=prompt_comp,
        response_composition=response_comp,
        usage=usage_block,
        latency=latency,
        call_outcome=call_outcome,
        observations=observations,
        local_decision=local_decision,
    )


@fail_safe
def _capture_llamaindex_prompt_at(args, kwargs, order):
    """Capture LlamaIndex prompt composition at an explicit span order. Used by
    the streaming path which reserves order up front."""
    if len(args) >= 2:
        payload = args[1]
    elif "messages" in kwargs:
        payload = kwargs["messages"]
    else:
        return
    session = get_current_session()
    if not hasattr(session, "_pending_compositions"):
        session._pending_compositions = {}
    comp = build_prompt_composition("llamaindex", {"messages": payload})
    if comp:
        comp_key = f"{session.trace_id}:{order}"
        session._pending_compositions.setdefault(comp_key, {})["prompt"] = comp


@fail_safe
def _capture_response_composition_at(provider, response, order):
    """Capture response composition at an explicit span order."""
    session = get_current_session()
    if not hasattr(session, "_pending_compositions"):
        session._pending_compositions = {}
    comp = build_response_composition(provider, response)
    if comp:
        comp_key = f"{session.trace_id}:{order}"
        session._pending_compositions.setdefault(comp_key, {})["response"] = comp
    # The stream paths capture here instead of _capture_response_composition,
    # so without this the pending tool-call ids were never stashed and every
    # streamed LlamaIndex tool row lost its id. Same REPLACE-on-capture contract
    # as:2268 — stash even when empty so a no-tool turn clears stale ids.
    try:
        set_pending_tool_calls(extract_pending_tool_calls(provider, response))
    except Exception:
        pass


def _li_chunk_marks_ttft(chunk):
    """First chunk carrying renderable payload — mirrors Node's
    (typeof chunk?.delta === "string" || chunk?.options) TTFT gate."""
    try:
        if isinstance(getattr(chunk, "delta", None), str):
            return True
        msg = getattr(chunk, "message", None)
        return bool(getattr(msg, "additional_kwargs", None))
    except Exception:
        return False


def _drop_deferred_spans(session, keep_ids=None, order=None, trace_id=None):
    """Drop spans queued via _defer_telemetry without logging them.

    Default (no scoping args) → clear the ENTIRE deferred buffer. Used by
    callers that own the whole deferred window and manually log instead.

    Scoped (any of keep_ids/order/trace_id given) → drop ONLY entries that
    belong to the caller's own call, so spans queued by OTHER concurrent calls
    sharing this session survive:
      * keep_ids: object ids of entries already present when the caller began —
        these are always kept (they predate the caller's window).
      * order: the caller's own reserved span_order — entries carrying it are
        dropped regardless of usage (the caller logs that row manually).
      * trace_id: entries on this trace that carry NO usage queued in the
        caller's window are dropped — dud rows with no tokens/cost. Entries
        carrying real usage, or on other orders/traces, are never dropped.
        Whether the inner instrumentor's anthropic stream span carries
        gen_ai.usage.* depends on its version (0.60 no, 0.61 yes) — never
        branch on it. This usage-less drop rule is deliberately unchanged: it
        is the best-effort FALLBACK behind the per-call suppression window
        (claim_anthropic_stream_span), which stops the span from landing at
        all; widening the drop to usage-carrying entries would risk dropping
        a legitimate concurrent call's row.
    """
    try:
        spans = getattr(session, "_deferred_spans", None)
        if not spans:
            return
        # Unscoped legacy behaviour — clear everything.
        if keep_ids is None and order is None and trace_id is None:
            spans.clear()
            return

        keep = keep_ids or set()

        def _is_empty(p):
            try:
                return (int(p.get("input_tokens", 0) or 0) == 0
                        and int(p.get("output_tokens", 0) or 0) == 0
                        and int(p.get("cached_tokens", 0) or 0) == 0)
            except Exception:
                return False

        kept = []
        for p in spans:
            # Entries present before the caller's window → always survive.
            if id(p) in keep:
                kept.append(p)
                continue
            sp = p.get("span") or {}
            so = sp.get("span_order")
            tid = sp.get("trace_id")
            # This call's own reserved order → drop (manually logged elsewhere).
            if order is not None and so == order:
                continue
            # This call's window dud: a usage-less span on our trace → drop.
            if trace_id is not None and tid == trace_id and _is_empty(p):
                continue
            kept.append(p)
        # Mutate in place so any other holder of the list sees the result.
        spans[:] = kept
    except Exception:
        pass


def _guard_sync_stream_li(stream, instance, args, kwargs, session,
                          order, span_name, start_time, prev_defer,
                          req_start_mono=None, obs_key=None):
    """Yield from a LlamaIndex stream while holding the _in_llamaindex guard;
    track the last chunk so we can do a manual log on completion (OpenLLMetry's
    inner-provider span often fails to surface usage for streamed LlamaIndex
    calls). Drops any spans queued during the stream to avoid double-logging.

    ``req_start_mono`` is the monotonic timestamp taken right before the provider
    call, so TTFT/total anchor to true request initiation rather than the lazy
    first-pull of this generator.

    ``obs_key`` is the wrapper's own key, captured right after its check. The
    manual log below runs in the CONSUMER's context, where the live contextvar
    may hold a sibling's key — see ``_log_li_py``."""
    _in_llamaindex.set(True)
    last = None
    _ttft_mono = None
    # Anthropic-only: merge cache-aware usage across message_start/message_delta
    # chunks — LlamaIndex's last chunk drops cache writes. Provider-gated and
    # fully fail-open; never consumes/alters chunks.
    _li_is_anthr = False
    try:
        _li_is_anthr = _li_provider_from_instance(instance) == "anthropic"
    except Exception:
        _li_is_anthr = False
    stream_usage = {}
    # Failure classification handed to the finally's manual log. A LOCAL by
    # construction (one generator = one call), so no session-global stale
    # hazard exists here. None → today's success payload, byte-identical.
    _fail_outcome = None
    try:
        for chunk in stream:
            last = chunk
            if _li_is_anthr:
                _merge_li_anthropic_stream_usage(stream_usage, chunk)
            # Own try: a hostile chunk must cost TTFT telemetry, never abort
            # the customer's iteration.
            try:
                if _ttft_mono is None and _li_chunk_marks_ttft(chunk):
                    _ttft_mono = _time.monotonic()
            except Exception:
                pass
            yield chunk
    except Exception as _exc:
        # Provider failure during iteration (the inner instrumentor's ERROR
        # span is dropped in the finally, so this row is the only carrier).
        # A consumer-injected gen.throw(e) resuming at the yield IS caught and
        # treated as call failure — same accepted semantics as the Node
        # LangChain streamIterator catch; GeneratorExit is BaseException and
        # stays untouched, so a consumer break still lands the partial success
        # row. Identity re-raise (golden rule).
        try:
            elapsed = (int(max(0.0, _time.monotonic() - req_start_mono) * 1000)
                       if req_start_mono is not None else 0)
            _fail_outcome = build_call_outcome(_exc, elapsed)
        except Exception:
            pass
        raise
    finally:
        _in_llamaindex.set(False)
        session._defer_telemetry = prev_defer
        try:
            # Drop only THIS call's inner span (it reserved `order`), so a
            # concurrent stream sharing the session keeps its own queued spans.
            _drop_deferred_spans(session, order=order)
            # The inner OpenLLMetry on_end already popped our prompt stash
            # (before queuing the deferred span we just dropped) — re-capture
            # it so our manual log includes prompt composition.
            _capture_llamaindex_prompt_at(args, kwargs, order)
            if last is not None:
                _capture_response_composition_at("llamaindex", last, order)
            latency = _build_stream_latency(req_start_mono, _ttft_mono, _time.monotonic())
            # Keyed observation drain (observations only — a body-less seam
            # never owns a keyed local_decision; a claim could only steal a
            # sibling's via the untagged fallback). Own try: fail-open.
            _pending_obs = None
            try:
                _pending_obs = _state.drain_observations(obs_key)
            except Exception:
                _pending_obs = None
            # On failure the row keeps any observed usage (the provider billed
            # those tokens) but lands status=failed via _fail_outcome; a
            # zero-pull failure lands model-from-instance, 0/0, unmeasured.
            _log_li_py(instance, last, order, span_name, start_time, stream_usage,
                       latency=latency, call_outcome=_fail_outcome,
                       observations=_pending_obs,
                       obs_key=obs_key)
        except Exception:
            pass  # fail-open


def _li_call_original(original, args, kwargs):
    """Invoke a LlamaIndex original method, handling its wrapt-based dispatcher.

    LlamaIndex 0.13+ decorates provider chat methods with `dispatcher.span`,
    which uses `@wrapt.decorator`. wrapt's FunctionWrapper distinguishes
    "called as bound method" vs "called as plain function" via the descriptor
    protocol — calling it as `original(self, ...)` does NOT count as bound,
    so its inner `inspect.signature(func).bind(*args, **kwargs)` then fails
    on the function's `_self` parameter. Re-bind via `__get__(self)` to
    invoke correctly.
    """
    if args and hasattr(original, "__get__"):
        bound = original.__get__(args[0], type(args[0]))
        return bound(*args[1:], **kwargs)
    return original(*args, **kwargs)


async def _li_acall_original(original, args, kwargs):
    """Async variant of _li_call_original."""
    if args and hasattr(original, "__get__"):
        bound = original.__get__(args[0], type(args[0]))
        return await bound(*args[1:], **kwargs)
    return await original(*args, **kwargs)


def _set_llamaindex_wrapper(cls, method_name, original, kind: str):
    """Install a LlamaIndex wrapper. `kind` is one of sync/async/stream.

    There is deliberately no "astream" kind: LlamaIndex `astream_chat` is an
    `async def` and is served by the "async" kind, whose wrapper returns the
    guarded `_drain()` async generator. An unknown kind installs nothing.
    """
    provider = "llamaindex"

    if kind == "sync":
        @functools.wraps(original)
        def li_sync(*args, **kwargs):
            if in_llamaindex() or in_pydantic_ai() or in_agno():
                return _li_call_original(original, args, kwargs)
            # Resolve the session once and thread it into the check so a REROUTE
            # audit stash lands on the SAME object flushed below.
            session = get_current_session()
            _hint, _prov = _li_check_ctx(args)
            _run_sync_check(session=session, model_hint=_hint, provider=_prov)
            # This call's obs key, captured right after the check.
            _obs_key = _state.get_current_obs_key()
            # B1: OpenAIResponses' inner call is openai.resources.responses
            # .Responses.create — a MANUAL-wrapper provider that emits no
            # OpenLLMetry span (telemetry.py deliberately unwraps OpenLLMetry's
            # Responses hooks) and now passes straight through under our guard.
            # So nothing queues a deferred span for this call and nothing
            # consumes the span counter: the flush in the success tail would
            # no-op (ZERO rows) and every call would reuse span_order 0. This
            # wrapper therefore reserves the order itself and logs the row
            # manually, exactly like the streaming paths do. The other three
            # LlamaIndex providers always have an inner span, so they keep the
            # flush path byte-for-byte.
            _is_responses = (_prov == "openai_responses")
            _resp_order = None
            _resp_span_name = None
            _resp_start = None
            if _is_responses:
                _resp_order = session.next_span_order()
                _resp_span_name = consume_pending_span_name()
                _resp_start = datetime.now(timezone.utc)
                _capture_llamaindex_prompt_at(args, kwargs, _resp_order)
            else:
                _capture_llamaindex_prompt(args, kwargs)
            prev_defer = getattr(session, '_defer_telemetry', False)
            session._defer_telemetry = True
            # Failure-outcome anchor: post-check, pre-provider-call.
            _t0 = _time.monotonic()
            _in_llamaindex.set(True)
            try:
                try:
                    result = _li_call_original(original, args, kwargs)
                finally:
                    _in_llamaindex.set(False)
                    session._defer_telemetry = prev_defer
            except Exception as _exc:
                # Provider rejection on the eager path — see lc_sync: the raw
                # provider wrapper short-circuited under our guard, so this
                # wrapper is the SOLE emitter (L-2). State already restored by
                # the inner finally; outermost defer level only; fully
                # fail-open; identity re-raise (golden rule).
                try:
                    if not prev_defer:
                        session._call_outcome = build_call_outcome(
                            _exc, int((_time.monotonic() - _t0) * 1000))
                        try:
                            if getattr(session, '_deferred_spans', None):
                                _flush_deferred_spans(session, obs_key=_obs_key)
                            else:
                                # Model lives on the instance, never in
                                # kwargs — thread the check context via a
                                # synthetic kwargs; the framework tag
                                # backstops an unresolvable provider so the
                                # row is never provider-blank.
                                _stash_attempt_context(
                                    session, _prov or provider, "llamaindex",
                                    (), {"model": _hint})
                                _emit_call_failure_log(
                                    get_client(), session, obs_key=_obs_key)
                        finally:
                            # Stale-outcome guard — see lc_sync
                            # (unconditional: a throwing flush/emit must not
                            # skip the clear).
                            if getattr(session, '_call_outcome', None) is not None:
                                session._call_outcome = None
                except Exception:
                    pass
                raise
            if _is_responses:
                # B1 manual emission — see the note above. Capture keyed at OUR
                # reserved order so _log_li_py's pop finds this call's prompt +
                # response composition; the composition provider stays
                # "llamaindex" because `result` is a LlamaIndex ChatResponse,
                # not a raw Responses `Response`.
                # The scoped drop covers ONLY a deferred span carrying our own
                # reserved order — it is not general double-log insurance (a
                # live inner Responses span would consume order+1 and slip
                # past it). What actually guarantees no inner span exists is
                # telemetry.py's unconditional _unwrap_openai_responses_hooks
                # (telemetry.py:1888), which removes OpenLLMetry's Responses
                # wrappers at instrumentation time.
                # We also hand over the v2 extras the flush would normally
                # attach: without draining them here this call's observations
                # (e.g. PR #482's unappliable_call_shape) and local_decision
                # would linger and be swept onto an unrelated later row.
                # Fully fail-open.
                try:
                    _drop_deferred_spans(session, order=_resp_order)
                    _capture_response_composition_at(provider, result, _resp_order)
                    # Claim-before-log, exactly like the flush — a throwing log
                    # must not leave a stale decision for the next call to
                    # inherit. Keyed to THIS call, so a concurrent sibling's
                    # decision stays in the store for its own row.
                    _pending_obs = _state.drain_observations(_obs_key)
                    _pending_ld = _claim_local_decision(session, _obs_key)
                    _log_li_py(args[0] if args else None, result, _resp_order,
                               _resp_span_name, _resp_start,
                               observations=_pending_obs,
                               local_decision=_pending_ld,
                               obs_key=_obs_key)
                except Exception:
                    pass  # fail-open
            else:
                _capture_response_composition(provider, result)
                if not session._defer_telemetry:
                    _flush_deferred_spans(session, obs_key=_obs_key)
            return result
        setattr(cls, method_name, li_sync)

    elif kind == "async":
        @functools.wraps(original)
        async def li_async(*args, **kwargs):
            if in_llamaindex() or in_pydantic_ai() or in_agno():
                return await _li_acall_original(original, args, kwargs)
            # Resolve the session once and thread it into the check so a REROUTE
            # audit stash lands on the SAME object flushed below (non-stream path).
            session = get_current_session()
            _hint, _prov = _li_check_ctx(args)
            await _run_async_check(session=session, model_hint=_hint, provider=_prov)
            # This call's obs key, captured right after the check.
            _obs_key = _state.get_current_obs_key()
            # B1: see li_sync — OpenAIResponses' inner Responses.create is a
            # manual-wrapper provider with no OpenLLMetry span, so nothing
            # downstream consumes the span counter and nothing ever lands in
            # the deferred buffer. RESERVE the order here (rather than
            # snapshotting a counter that will never advance) and consume the
            # pending span name once, for whichever branch below emits.
            _is_responses = (_prov == "openai_responses")
            _resp_span_name = None
            _resp_start = None
            if _is_responses:
                order_snapshot = session.next_span_order()
                _resp_span_name = consume_pending_span_name()
                _resp_start = datetime.now(timezone.utc)
                _capture_llamaindex_prompt_at(args, kwargs, order_snapshot)
            else:
                # Snapshot the current order BEFORE any inner OpenLLMetry
                # on_start can fire. This is the order our prompt-capture and
                # (for streaming path) manual log will use.
                order_snapshot = session._span_counter
                _capture_llamaindex_prompt(args, kwargs)
            prev_defer = getattr(session, '_defer_telemetry', False)
            session._defer_telemetry = True
            _in_llamaindex.set(True)
            # Anchor before the provider call — the streaming branch below uses
            # it for TTFT/total; harmless on the non-streaming branch.
            req_start_mono = _time.monotonic()
            try:
                result = await _li_acall_original(original, args, kwargs)
            except Exception as _exc:
                _in_llamaindex.set(False)
                session._defer_telemetry = prev_defer
                # Provider rejection before a stream/response existed — see
                # lc_sync: this wrapper is the SOLE emitter (L-2). State is
                # restored above, so the emission cannot be swallowed by a
                # defer/guard seam. Outermost defer level only; fully
                # fail-open; identity re-raise (golden rule).
                try:
                    if not prev_defer:
                        session._call_outcome = build_call_outcome(
                            _exc, int(max(0.0, _time.monotonic()
                                          - req_start_mono) * 1000))
                        try:
                            if getattr(session, '_deferred_spans', None):
                                _flush_deferred_spans(session, obs_key=_obs_key)
                            else:
                                # Model lives on the instance, never in
                                # kwargs — thread the check context via a
                                # synthetic kwargs; the framework tag
                                # backstops an unresolvable provider so the
                                # row is never provider-blank.
                                _stash_attempt_context(
                                    session, _prov or provider, "llamaindex",
                                    (), {"model": _hint})
                                _emit_call_failure_log(
                                    get_client(), session, obs_key=_obs_key)
                        finally:
                            # Stale-outcome guard — see lc_sync
                            # (unconditional: a throwing flush/emit must not
                            # skip the clear).
                            if getattr(session, '_call_outcome', None) is not None:
                                session._call_outcome = None
                except Exception:
                    pass
                raise
            # astream_chat returns an async-iterator from an async function —
            # wrap to keep the guard for the duration of consumption AND do
            # a manual log (OpenLLMetry's inner-provider span often fails to
            # surface usage on streamed LlamaIndex calls).
            if hasattr(result, "__aiter__"):
                instance = args[0] if args else None
                order = order_snapshot
                # B1: the responses branch already consumed both above (it has
                # to, to key its prompt capture) — reuse them rather than
                # consuming a second, now-empty pending name.
                span_name = (_resp_span_name if _is_responses
                             else consume_pending_span_name())
                start_time = (_resp_start if _is_responses
                              else datetime.now(timezone.utc))
                # Anthropic-only cache-aware usage merge — see _guard_sync_stream_li.
                _li_is_anthr = False
                try:
                    _li_is_anthr = _li_provider_from_instance(instance) == "anthropic"
                except Exception:
                    _li_is_anthr = False
                async def _drain():
                    last = None
                    _ttft_mono = None
                    stream_usage = {}
                    # Failure classification handed to the finally's manual
                    # log — see _guard_sync_stream_li (a LOCAL, so no stale
                    # hazard by construction).
                    _fail_outcome = None
                    try:
                        async for chunk in result:
                            last = chunk
                            if _li_is_anthr:
                                _merge_li_anthropic_stream_usage(stream_usage, chunk)
                            # Own try — see _guard_sync_stream_li.
                            try:
                                if _ttft_mono is None and _li_chunk_marks_ttft(chunk):
                                    _ttft_mono = _time.monotonic()
                            except Exception:
                                pass
                            yield chunk
                    except Exception as _exc:
                        # Provider failure during iteration — see
                        # _guard_sync_stream_li: gen.throw at the yield IS
                        # caught (accepted); GeneratorExit (BaseException) is
                        # not, so consumer break keeps the partial success
                        # row. Identity re-raise (golden rule).
                        try:
                            elapsed = (int(max(0.0, _time.monotonic()
                                               - req_start_mono) * 1000)
                                       if req_start_mono is not None else 0)
                            _fail_outcome = build_call_outcome(_exc, elapsed)
                        except Exception:
                            pass
                        raise
                    finally:
                        _in_llamaindex.set(False)
                        session._defer_telemetry = prev_defer
                        try:
                            # Scope the drop to THIS call's order so a
                            # concurrent stream keeps its own queued spans.
                            _drop_deferred_spans(session, order=order)
                            # Inner OpenLLMetry on_end already popped our
                            # prompt stash — re-capture so the manual log
                            # includes prompt composition.
                            _capture_llamaindex_prompt_at(args, kwargs, order)
                            if last is not None:
                                _capture_response_composition_at(
                                    provider, last, order
                                )
                            latency = _build_stream_latency(
                                req_start_mono, _ttft_mono, _time.monotonic())
                            # Keyed observation drain — observations only;
                            # see _guard_sync_stream_li. Own try: fail-open.
                            _pending_obs = None
                            try:
                                _pending_obs = _state.drain_observations(_obs_key)
                            except Exception:
                                _pending_obs = None
                            # Failure keeps observed usage but lands
                            # status=failed — see _guard_sync_stream_li.
                            _log_li_py(instance, last, order, span_name,
                                       start_time, stream_usage,
                                       latency=latency,
                                       call_outcome=_fail_outcome,
                                       observations=_pending_obs,
                                       obs_key=_obs_key)
                        except Exception:
                            pass  # fail-open
                return _drain()
            # Non-streaming response — capture + flush immediately.
            _in_llamaindex.set(False)
            session._defer_telemetry = prev_defer
            if _is_responses:
                # B1 manual emission — see li_sync's twin: no inner span means
                # the flush below would be a no-op and the call would land ZERO
                # rows. Keyed at our reserved order; composition provider stays
                # "llamaindex" (result is a ChatResponse). The scoped drop
                # covers only a span at OUR order, not a hypothetical live
                # inner Responses span (that would take order+1) — what rules
                # one out is telemetry.py's unconditional
                # _unwrap_openai_responses_hooks (telemetry.py:1888). The v2
                # extras are drained + popped here because the flush that
                # normally attaches them never runs. Fully fail-open.
                try:
                    _drop_deferred_spans(session, order=order_snapshot)
                    _capture_response_composition_at(provider, result, order_snapshot)
                    # Claim-before-log, keyed to this call — see li_sync's twin.
                    _pending_obs = _state.drain_observations(_obs_key)
                    _pending_ld = _claim_local_decision(session, _obs_key)
                    _log_li_py(args[0] if args else None, result, order_snapshot,
                               _resp_span_name, _resp_start,
                               observations=_pending_obs,
                               local_decision=_pending_ld,
                               obs_key=_obs_key)
                except Exception:
                    pass  # fail-open
            else:
                _capture_response_composition(provider, result)
                if not session._defer_telemetry:
                    _flush_deferred_spans(session, obs_key=_obs_key)
            return result
        setattr(cls, method_name, li_async)

    elif kind == "stream":
        @functools.wraps(original)
        def li_stream(*args, **kwargs):
            if in_llamaindex() or in_pydantic_ai() or in_agno():
                return _li_call_original(original, args, kwargs)
            # Resolve the session BEFORE the check and thread it in (parity with
            # li_sync/li_async) so a REROUTE audit stash lands on the SAME
            # object this call flushes.
            session = get_current_session()
            _hint, _prov = _li_check_ctx(args)
            _run_sync_check(session=session, model_hint=_hint, provider=_prov)
            # This call's own obs key, read while the check's context is still
            # current — both emit paths below run later, in the consumer's
            # context, where the contextvar may hold a sibling's key.
            _obs_key = _state.get_current_obs_key()
            # Snapshot the current counter as the order for OUR manual log.
            # The inner provider's OpenLLMetry on_start will call
            # next_span_order() → consumes this same value (so the counter
            # advances by exactly 1 per call, matching the non-stream path).
            # B1 exception: OpenAIResponses' inner Responses.create is a
            # manual-wrapper provider with NO OpenLLMetry span, so nothing
            # would ever consume the counter and every call in the workflow
            # would reuse the same span_order — reserve it here instead.
            if _prov == "openai_responses":
                order = session.next_span_order()
            else:
                order = session._span_counter
            span_name = consume_pending_span_name()
            start_time = datetime.now(timezone.utc)
            _capture_llamaindex_prompt_at(args, kwargs, order)
            instance = args[0] if args else None
            # Defer telemetry so the inner OpenLLMetry span (if any) is queued
            # rather than logged — we manually log instead from the stream
            # accumulator below to ensure consistent token extraction.
            prev_defer = getattr(session, "_defer_telemetry", False)
            session._defer_telemetry = True
            # Anchor before the provider call so TTFT/total measure the request,
            # not the customer's lazy first pull on the returned generator.
            req_start_mono = _time.monotonic()
            # B1: hold the guard ACROSS the dispatch. LlamaIndex's Anthropic
            # stream_chat fires the HTTP request EAGERLY (it calls
            # `messages.create(..., stream=True)` before returning its
            # generator), so the inner provider wrapper used to run here with
            # the guard unset — running its own pre-flight (applying a reroute
            # the framework guard promises to suppress) AND emitting its own
            # row, so every streamed call landed twice at two different models.
            # OpenAI/OpenAIResponses dispatch lazily inside the returned
            # generator, which is why only anthropic duplicated. Released in
            # the inner finally before we return: _guard_sync_stream_li re-takes
            # the guard at first pull and resets in its own finally, and the
            # inner span produced during this eager window is deferred (above)
            # then dropped order-scoped there. Mirrors li_sync's structure —
            # note only the GUARD is released here, not `_defer_telemetry`,
            # which the stream still needs and its guard restores.
            _in_llamaindex.set(True)
            try:
                try:
                    stream = _li_call_original(original, args, kwargs)
                finally:
                    _in_llamaindex.set(False)
            except Exception as _exc:
                # The provider call raised synchronously (e.g. an auth/quota
                # error) before returning a stream — restore the defer flag we
                # just set so this session's later telemetry keeps flushing
                # (leaving it True would silently buffer every subsequent span),
                # then re-raise the customer's own provider error unchanged.
                session._defer_telemetry = prev_defer
                # This wrapper is the SOLE emitter for the call (L-2) — emit
                # the manual failed row exactly like the async twin's
                # zero-pull failure path (li_async's _drain() finally): model
                # from instance, 0/0 tokens, unmeasured. Fully fail-open.
                try:
                    _fail_outcome = None
                    try:
                        _fail_outcome = build_call_outcome(
                            _exc, int(max(0.0, _time.monotonic()
                                          - req_start_mono) * 1000))
                    except Exception:
                        pass
                    # Scope the drop to THIS call's order — see
                    # _guard_sync_stream_li.
                    _drop_deferred_spans(session, order=order)
                    _capture_llamaindex_prompt_at(args, kwargs, order)
                    latency = _build_stream_latency(
                        req_start_mono, None, _time.monotonic())
                    # Keyed observation drain — observations only; see
                    # _guard_sync_stream_li. Own try: fail-open.
                    _pending_obs = None
                    try:
                        _pending_obs = _state.drain_observations(_obs_key)
                    except Exception:
                        _pending_obs = None
                    _log_li_py(instance, None, order, span_name, start_time,
                               {}, latency=latency, call_outcome=_fail_outcome,
                               observations=_pending_obs,
                               obs_key=_obs_key)
                except Exception:
                    pass
                raise
            return _guard_sync_stream_li(
                stream, instance, args, kwargs, session,
                order, span_name, start_time, prev_defer, req_start_mono,
                obs_key=_obs_key,
            )
        setattr(cls, method_name, li_stream)

    # There is intentionally no "astream" kind: LlamaIndex `astream_chat` is
    # `async def` (a coroutine, unlike LangChain's `astream`), so it is served
    # by the "async" kind above — `li_async` awaits the original and returns
    # the guarded `_drain()` async generator. A sync-return wrapper here would
    # break `await llm.astream_chat(...)` with a TypeError.


# ═══════════════════════════════════════════════════════════════════
# pydantic_ai wrappers — for the concrete Model subclasses' `request` /
# `request_stream` methods.
#
# pydantic_ai is a framework: each Model subclass (OpenAIChatModel /
# AnthropicModel / GoogleModel / GeminiModel / MistralModel /
# OpenAIResponsesModel) internally calls the underlying provider SDK (openai,
# anthropic, google-genai, mistralai) — which is ALSO patched by the enforcer.
# These wrappers run the SINGLE pre-flight check, capture prompt/response
# composition from pydantic_ai's unified ModelMessage / ModelResponse shape,
# and hold the _in_pydantic_ai guard so:
# (a) the nested provider wrapper passes straight through
# (b) the inner provider's OpenLLMetry span is dropped in TokenPoliceSpan
# Processor (it would otherwise be a duplicate /log)
#
# No OpenLLMetry pydantic_ai instrumentor exists — telemetry is manual,
# extracted from ModelResponse.usage / StreamedResponse.usage(). Provider name
# comes from `model.system`, model id from `model.model_name`.
# ═══════════════════════════════════════════════════════════════════

# Each entry → wrap `cls.request` (async) and `cls.request_stream` (async
# context manager) on the concrete model class.
_PYDANTIC_AI_TARGETS = [
    ("pydantic_ai.models.openai",    "OpenAIChatModel"),
    ("pydantic_ai.models.openai",    "OpenAIResponsesModel"),
    ("pydantic_ai.models.anthropic", "AnthropicModel"),
    ("pydantic_ai.models.google",    "GoogleModel"),
    ("pydantic_ai.models.gemini",    "GeminiModel"),
    ("pydantic_ai.models.mistral",   "MistralModel"),
]


def _pa_provider(model) -> str:
    """Normalize pydantic_ai's `model.system` to TokenPolice's provider keys."""
    raw = ""
    try:
        raw = str(getattr(model, "system", "") or "")
    except Exception:
        raw = ""
    pl = raw.lower()
    if "gemini" in pl or "google" in pl:
        return "google"
    if "anthropic" in pl or "claude" in pl:
        return "anthropic"
    if "mistral" in pl:
        return "mistral"
    if "openai" in pl:
        return "openai"
    return pl or "pydantic_ai"


def _pa_model_name(model, fallback: str = "unknown") -> str:
    try:
        return str(getattr(model, "model_name", None) or fallback)
    except Exception:
        return fallback


def _pa_check_ctx(model):
    """``(model_hint, provider)`` for a pydantic_ai pre-flight, from the
    same `model_name` / `system` fields the /log row uses. The "pydantic_ai"
    fallback is dropped to None — a framework slug in the check context would
    make every provider-targeted REROUTE look cross-provider. Never raises."""
    try:
        name = _pa_model_name(model, fallback="")
    except Exception:
        name = ""
    try:
        provider = _pa_provider(model)
    except Exception:
        provider = ""
    if provider == "pydantic_ai":
        provider = ""
    return (name or None), (provider or None)


def _pa_usage_to_counts(usage) -> tuple:
    """Pull (input_tokens, output_tokens, cached_tokens) from a pydantic_ai
    RequestUsage. RequestUsage.details may carry provider-specific extras
    including cache-read / cache-write tokens — best-effort.

    `input_tokens` from RequestUsage already EXCLUDES cached tokens for the
    OpenAI-shaped providers (pydantic_ai mirrors the Anthropic accounting) so
    don't subtract again. Reasoning tokens are folded into output by
    pydantic_ai already.
    """
    if usage is None:
        return 0, 0, 0
    # New names; tolerate deprecated request_tokens / response_tokens too.
    inp = getattr(usage, "input_tokens", None)
    if inp is None:
        inp = getattr(usage, "request_tokens", 0) or 0
    out = getattr(usage, "output_tokens", None)
    if out is None:
        out = getattr(usage, "response_tokens", 0) or 0
    cached = 0
    details = getattr(usage, "details", None)
    if isinstance(details, dict):
        # Common detail keys across providers; fall back to 0.
        for k in ("cache_read_input_tokens", "cached_tokens",
                  "cached_content_token_count", "cache_creation_input_tokens"):
            v = details.get(k)
            if v:
                try:
                    cached += int(v)
                except Exception:
                    pass
    try:
        return int(inp or 0), int(out or 0), int(cached)
    except Exception:
        return 0, 0, 0


def _pa_reasoning_from_details(usage, output_tokens):
    """Reasoning-token count from a pydantic_ai ``RequestUsage.details``
    (G3-O1), clamped to ``output_tokens`` — pydantic_ai folds reasoning into
    output, so a larger value can only be provider noise. 0 on any
    failure/absence (never raises)."""
    try:
        details = getattr(usage, "details", None)
        if not isinstance(details, dict):
            return 0
        r = int(details.get("reasoning_tokens") or 0)
        if r <= 0:
            return 0
        return min(r, int(output_tokens or 0))
    except Exception:
        return 0


@fail_safe
def _capture_pydantic_ai_prompt_at(model, messages, order: int):
    """Stash pydantic_ai prompt composition at an explicit span order."""
    session = get_current_session()
    if not hasattr(session, "_pending_compositions"):
        session._pending_compositions = {}
    comp = build_prompt_composition("pydantic_ai", {"messages": messages})
    if comp:
        comp_key = f"{session.trace_id}:{order}"
        session._pending_compositions.setdefault(comp_key, {})["prompt"] = comp


@fail_safe
def _capture_pydantic_ai_response_at(model, response, order: int):
    """Stash pydantic_ai response composition at an explicit span order."""
    session = get_current_session()
    if not hasattr(session, "_pending_compositions"):
        session._pending_compositions = {}
    comp = build_response_composition("pydantic_ai", response)
    if comp:
        comp_key = f"{session.trace_id}:{order}"
        session._pending_compositions.setdefault(comp_key, {})["response"] = comp


def _pa_anthropic_usage_from_details(usage):
    """For an Anthropic pydantic_ai `RequestUsage` that recorded cache-WRITE
    tokens, return `(usage_block, input_excl, cached_read)` so the caller can
    forward the verbatim Anthropic usage shape and keep `cached_tokens` as
    reads-only. Returns `None` for every other case (no cache write present,
    missing / non-dict details, or any error) so the caller stays on today's
    path unchanged.

    `RequestUsage.details` carries the native Anthropic keys with `input_tokens`
    EXCLUSIVE of cache (cache reads and writes are counted separately), whereas
    the top-level `RequestUsage.input_tokens` is cache-INCLUSIVE. Building the
    raw block from `details` keeps `input_tokens` exclusive so cache reads and
    cache writes are each billed once, at their own rate — without it the write
    tokens get folded into the cheaper cache-READ bucket and under-billed.
    Self-guarded (fail-safe): any failure degrades to `None`, so the row is
    still logged via the caller's existing fallback path rather than lost."""
    try:
        details = getattr(usage, "details", None)
        if not isinstance(details, dict):
            return None
        cache_write = int(details.get("cache_creation_input_tokens") or 0)
        if cache_write <= 0:
            # No cache-WRITE tokens → nothing is mis-bucketed; stay byte-identical.
            return None
        cache_read = int(details.get("cache_read_input_tokens") or 0)
        input_excl = int(details.get("input_tokens") or 0)
        output = int(details.get("output_tokens") or 0)
        raw = {
            "input_tokens": input_excl,
            "output_tokens": output,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_write,
        }
        return {"shape": "anthropic_messages", "raw": raw}, input_excl, cache_read
    except Exception:
        return None


@fail_safe
def _log_pydantic_ai(model, session, messages, response, order, span_name,
                    start_time, latency=None, call_outcome=None,
                    observations=None, obs_key=_OBS_KEY_CURRENT):
    """Manual /log dispatch for a pydantic_ai Model.request / request_stream
    completion. `response` is either a ModelResponse (non-stream) or the
    `StreamedResponse.get()` synthetic ModelResponse (stream).

    `latency` is an optional SDK latency-primitives dict (built via
    `_build_stream_latency` / `_build_non_stream_latency`). When None the wire
    payload is byte-identical to the pre-latency behavior (client gates on
    `if latency:`).

    ``call_outcome`` is the wrappers' failure classification (the inner
    provider's OTel span is hard-dropped under the pydantic_ai guard, so this
    row is the only carrier). None (the default) keeps the payload
    byte-identical to before — the client omits an absent call_outcome and the
    collector stamps success.

    ``obs_key`` is the emitting call's obs key, threaded from the wrapper that
    captured it right after its check. It keys BOTH this row's ``_tp_routing``
    stamp and the keyed observation drain below — never a live contextvar read
    here, since ``__aexit__``/``_finalize`` frames run in the consumer's
    context where the var may hold a sibling's key. ``observations`` lets a
    caller hand in pre-drained entries; they are merged with the internal
    drain's result."""
    tp = get_client()
    if tp is None or session is None:
        return

    provider = _pa_provider(model)
    model_name = _pa_model_name(model, fallback="unknown")
    # ModelResponse carries .model_name + .usage. StreamedResponse.get() also
    # returns a ModelResponse with those fields populated from the stream.
    if response is not None:
        resp_model = getattr(response, "model_name", None)
        if isinstance(resp_model, str) and resp_model:
            model_name = resp_model
        resp_provider = getattr(response, "provider_name", None)
        if isinstance(resp_provider, str) and resp_provider:
            # Re-normalize the provider name reported by pydantic_ai (e.g.
            # "openai" / "anthropic" / "google-gla") to our canonical key.
            pl = resp_provider.lower()
            if "gemini" in pl or "google" in pl:
                provider = "google"
            elif "anthropic" in pl or "claude" in pl:
                provider = "anthropic"
            elif "mistral" in pl:
                provider = "mistral"
            elif "openai" in pl:
                provider = "openai"

    usage = getattr(response, "usage", None) if response is not None else None
    input_tokens, output_tokens, cached_tokens = _pa_usage_to_counts(usage)

    # Anthropic-only, cache-WRITE-only override: forward the verbatim Anthropic
    # usage shape so cache-WRITE tokens are billed at the write rate. Otherwise
    # `input_tokens`/`cached_tokens` stay exactly as computed above and no usage
    # shape is forwarded (byte-identical to the prior behavior). See the helper.
    usage_block = None
    if provider == "anthropic":
        _pa_cache_usage = _pa_anthropic_usage_from_details(usage)
        if _pa_cache_usage is not None:
            usage_block, input_tokens, cached_tokens = _pa_cache_usage

    # G3-O1: OpenAI-arm reasoning forward. pydantic_ai folds reasoning into
    # output_tokens but only exposes the count at
    # ``usage.details['reasoning_tokens']``; the client's synthesized wire
    # block carries no details key, so it was silently dropped. Byte-mirror
    # that synth (client.py) PLUS the details key — rows without reasoning
    # keep ``usage=None`` → wire payload byte-identical to today. openai
    # ONLY: xai's mapper treats nested reasoning as EXCLUSIVE of
    # completion_tokens (would double-count; pydantic_ai xai arms report
    # provider "openai", so screen the raw provider_name too), and the
    # google/anthropic details keys have unverified output-inclusion
    # semantics (follow-up). Fully fail-open — a raise here would kill the
    # whole row via @fail_safe.
    try:
        if usage_block is None and provider == "openai":
            _resp_prov = str(getattr(response, "provider_name", "") or "").lower() \
                if response is not None else ""
            if "xai" not in _resp_prov and "grok" not in _resp_prov:
                _r = _pa_reasoning_from_details(usage, output_tokens)
                if _r > 0:
                    _raw = {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "completion_tokens_details": {"reasoning_tokens": _r},
                    }
                    if cached_tokens:
                        _raw["prompt_tokens_details"] = {"cached_tokens": cached_tokens}
                    usage_block = {"shape": "openai_compatible_chat", "raw": _raw}
    except Exception:
        pass

    span_obj = {
        **manual_span_ids(session),
        "span_kind": "llm",
        "span_name": span_name or model_name,
        "span_order": order,
        "start_time": start_time.isoformat() if start_time else None,
        "end_time": datetime.now(timezone.utc).isoformat(),
    }

    metadata = {"workflow_name": session.workflow_name}
    if session.session_id:
        metadata["session_id"] = session.session_id
    # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
    # session metadata WITHOUT it, then re-add it only when this row belongs
    # to the call that was actually rerouted (exact obs-key match).
    _copy_session_metadata(metadata, session)
    # Resolved once — the SAME key drives the stamp and the drain below.
    _row_obs_key = _resolve_obs_key(obs_key)
    _stamp_routing_marker(metadata, session, _row_obs_key)

    prompt_comp = []
    response_comp = []
    comp_key = f"{session.trace_id}:{order}"
    if hasattr(session, "_pending_compositions"):
        comp_data = session._pending_compositions.pop(comp_key, {})
        prompt_comp = comp_data.get("prompt", []) or []
        response_comp = comp_data.get("response", []) or []

    # Keyed observation drain (observations only — a body-less seam never owns
    # a keyed local_decision; a claim could only steal a sibling's via the
    # untagged fallback). Own try: fail-open, degrades to no observations.
    pending_observations = observations
    try:
        _drained = _state.drain_observations(_row_obs_key)
        if _drained:
            pending_observations = list(observations or []) + _drained
    except Exception:
        pending_observations = observations

    tp.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model=model_name,
        provider=provider,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        # None outside the anthropic cache-write case → client.log_sync treats
        # it exactly like an omitted usage= (it gates on `if usage:`), so the
        # wire payload is unchanged for every other path.
        usage=usage_block,
        metadata=metadata,
        span=span_obj,
        prompt_composition=prompt_comp,
        response_composition=response_comp,
        latency=latency,
        call_outcome=call_outcome,
        observations=pending_observations or None,
    )


def _pa_extract_messages(args, kwargs):
    """Pull the messages list from a Model.request / request_stream call.
    Signature: `(self, messages, model_settings, model_request_parameters,
    run_context=None)`."""
    if len(args) >= 2:
        return args[1]
    return kwargs.get("messages")


class _PydanticAIAsyncStreamMgr:
    """Async context-manager wrapper around `Model.request_stream(...)`.

    pydantic_ai's `request_stream` is decorated with `@asynccontextmanager`, so
    calling it returns an `AbstractAsyncContextManager` whose `__aenter__`
    yields a `StreamedResponse`. We delegate to that manager while:
      - running the pre-flight check on `__aenter__`
      - capturing prompt composition from the unified messages list
      - holding `_in_pydantic_ai=True` across the entire iteration window so
        the inner provider SDK call and its OpenLLMetry span both pass through
      - tapping `StreamedResponse._get_event_iterator` (instance override) so
        the first content event stamps TTFT without replacing the stream object
      - on `__aexit__`, building a synthetic ModelResponse from the drained
        StreamedResponse via `.get()` and dispatching the manual /log
    """

    def __init__(self, mgr, model, messages):
        self._mgr = mgr
        self._model = model
        self._messages = messages
        self._stream = None
        self._session = None
        self._order = 0
        self._span_name = None
        self._start_time = None
        self._start_mono = None
        # Marked by the _get_event_iterator tap on first content event; stays
        # None if the consumer never iterates (honest null TTFT — never fabricated).
        self._ttft_mono = None
        # This call's obs key — captured in __aenter__ right after the check
        # (the check runs there, not here). None until then / on pass-through.
        self._obs_key = None

    def _mark_ttft(self, event):
        """One-shot TTFT marker for pydantic_ai ModelResponseStreamEvent items.

        Duck-types PartStartEvent / PartDeltaEvent (and .part / .delta carriers)
        so we do not hard-import pydantic_ai message classes. Tool-only turns
        often never emit text deltas — PartStartEvent is the first served token
        (mirrors Anthropic content_block_start). Fail-safe: probe errors cost
        only the metric.
        """
        if self._ttft_mono is not None:
            return
        try:
            name = type(event).__name__
            if name in ("PartStartEvent", "PartDeltaEvent"):
                self._ttft_mono = _time.monotonic()
                return
            if getattr(event, "part", None) is not None or getattr(event, "delta", None) is not None:
                self._ttft_mono = _time.monotonic()
        except Exception:
            pass

    def _tap_stream_for_ttft(self, stream):
        """Instance-tap StreamedResponse._get_event_iterator for TTFT.

        Base StreamedResponse.__aiter__ builds its event pipeline from
        ``self._get_event_iterator()`` once. Special methods like ``__aiter__``
        are type-looked-up, so an instance override of ``__aiter__`` is a no-op;
        overriding the normal method ``_get_event_iterator`` works and keeps
        object identity / type (no proxy). On ANY setup failure return the raw
        stream — golden rule: TTFT loss only, app stream unaffected.
        """
        try:
            if stream is None:
                return stream
            orig = getattr(stream, "_get_event_iterator", None)
            if orig is None or not callable(orig):
                return stream

            mgr = self

            async def _tapped_get_event_iterator():
                async for event in orig():
                    try:
                        mgr._mark_ttft(event)
                    except Exception:
                        pass
                    yield event

            stream._get_event_iterator = _tapped_get_event_iterator
            return stream
        except Exception:
            return stream

    async def __aenter__(self):
        if (in_langchain() or in_litellm() or in_llamaindex()
                or in_pydantic_ai() or in_agno()):
            # Inside another framework / nested pydantic_ai — pass through.
            self._stream = await self._mgr.__aenter__()
            return self._stream

        try:
            _hint, _prov = _pa_check_ctx(self._model)
            # may raise TokenPoliceBlockedError
            await _run_async_check(model_hint=_hint, provider=_prov)
        except TokenPoliceBlockedError:
            # Nothing to clean up: pydantic_ai creates its httpx stream lazily
            # in __aenter__, which we have not entered yet — just propagate.
            raise

        # This call's obs key, read while the check's context is still current
        # — __aexit__/_finalize run in the consumer's context, where the
        # contextvar may hold a sibling's key (mirrors
        # _AnthropicStreamMgrWrapper). Fail-safe.
        try:
            self._obs_key = _state.get_current_obs_key()
        except Exception:
            self._obs_key = None

        try:
            self._session = get_current_session()
            self._order = self._session.next_span_order()
            self._span_name = consume_pending_span_name()
            self._start_time = datetime.now(timezone.utc)
            self._start_mono = _time.monotonic()
            _capture_pydantic_ai_prompt_at(
                self._model, self._messages, self._order
            )
            _in_pydantic_ai.set(True)
        except Exception:
            pass  # fail-safe

        try:
            self._stream = await self._mgr.__aenter__()
        except Exception as _exc:
            # Restore guard before re-raising so customer's error handling
            # path doesn't run with _in_pydantic_ai still set.
            try:
                _in_pydantic_ai.set(False)
            except Exception:
                pass
            # The provider HTTP request fires inside the vendor manager's
            # __aenter__ (e.g. a 401 surfaces here), and Python skips
            # __aexit__ when enter raises — without this the rejection
            # produces ZERO rows (mirrors
            # _AnthropicStreamMgrWrapper._emit_enter_failure). The inner
            # provider's OTel span is hard-dropped under the guard, so this
            # manual row is the only carrier. Fully fail-open; identity
            # re-raise (golden rule).
            try:
                elapsed = (max(0, int((_time.monotonic()
                                       - self._start_mono) * 1000))
                           if self._start_mono is not None else 0)
                _log_pydantic_ai(
                    self._model, self._session, self._messages, None,
                    self._order, self._span_name, self._start_time,
                    call_outcome=build_call_outcome(_exc, elapsed),
                    obs_key=self._obs_key,
                )
            except Exception:
                pass
            raise
        # / T4b: observe first content event for TTFT without replacing
        # the StreamedResponse object the customer / agent graph holds.
        self._stream = self._tap_stream_for_ttft(self._stream)
        return self._stream

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._finalize(exc_type is None, exc_val)
        except Exception:
            pass  # fail-safe — never throw out of the customer's `async with`
        # Keep `_in_pydantic_ai` True through the underlying manager's exit so
        # the bundled Anthropic OTel stream proxy's span.end() (often deferred
        # until stream close inside __aexit__) still sees the framework guard.
        # Pair with telemetry on_start stamping tp.suppress under the guard so
        # even a late on_end after the guard resets cannot emit a ghost row.
        try:
            return await self._mgr.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            try:
                _in_pydantic_ai.set(False)
            except Exception:
                pass

    async def _finalize(self, ok: bool, exc_val=None) -> None:
        if self._session is None or self._stream is None:
            return
        if not ok:
            # Failure inside the customer's `async with` window. Only a
            # genuine Exception gets a failed row — BaseException teardown
            # (CancelledError / GeneratorExit / KeyboardInterrupt) keeps
            # today's no-row behavior untouched. The inner provider's OTel
            # span is hard-dropped under the guard, so this manual row is the
            # only carrier (L-2): it keeps whatever usage the stream
            # accumulated (the provider billed those tokens); a zero-pull
            # failure lands 0/0. No latency on a broken stream (matches the
            # success path's early-return placement). Never suppresses the
            # exception — __aexit__ still returns the vendor manager's result.
            if not isinstance(exc_val, Exception):
                return
            try:
                elapsed = (max(0, int((_time.monotonic()
                                       - self._start_mono) * 1000))
                           if self._start_mono is not None else 0)
                try:
                    partial_response = self._stream.get()
                except Exception:
                    partial_response = None
                if partial_response is not None:
                    _capture_pydantic_ai_response_at(
                        self._model, partial_response, self._order
                    )
                _log_pydantic_ai(
                    self._model, self._session, self._messages,
                    partial_response, self._order, self._span_name,
                    self._start_time,
                    call_outcome=build_call_outcome(exc_val, elapsed),
                    obs_key=self._obs_key,
                )
            except Exception:
                pass  # fail-open
            return
        # is_streaming telemetry: shared stream-latency builder. _ttft_mono is
        # stamped by the _get_event_iterator tap on first content;
        # stays None when the consumer never iterates → honest null TTFT, not
        # fabricated. Returns None if the start anchor is missing → is_streaming
        # falls back to 0 (acceptable, never a crash). Built AFTER the early-
        # return guard so a broken/errored stream logs no latency.
        latency = _build_stream_latency(
            self._start_mono, self._ttft_mono, _time.monotonic()
        )
        # `StreamedResponse.get()` builds a ModelResponse from the parts the
        # consumer has accumulated by draining the stream. If the consumer
        # didn't drain, the synthesized response is incomplete (documented
        # caveat — applies to all manual-stream wrappers).
        try:
            final_response = self._stream.get()
        except Exception:
            final_response = None
        try:
            if final_response is not None:
                _capture_pydantic_ai_response_at(
                    self._model, final_response, self._order
                )
            _log_pydantic_ai(
                self._model, self._session, self._messages, final_response,
                self._order, self._span_name, self._start_time,
                latency=latency,
                obs_key=self._obs_key,
            )
        except Exception:
            pass  # fail-open


def _make_pydantic_ai_request_wrapper(original):
    """Build the async wrapper for `Model.request`."""
    @functools.wraps(original)
    async def request_wrapper(self, *args, **kwargs):
        # Inside any framework guard → pass through. The outer framework
        # wrapper handles the single check + composition for this LLM call.
        if (in_langchain() or in_litellm() or in_llamaindex()
                or in_pydantic_ai() or in_agno()):
            return await original(self, *args, **kwargs)

        _hint, _prov = _pa_check_ctx(self)
        # may raise TokenPoliceBlockedError
        await _run_async_check(model_hint=_hint, provider=_prov)
        # This call's obs key, captured right after the check while the
        # contextvar is still ours; threaded into both emit paths below.
        _obs_key = _state.get_current_obs_key()

        # Bound args: (self, messages, model_settings, model_request_parameters)
        messages = args[0] if args else kwargs.get("messages")
        session = None
        order = 0
        span_name = None
        start_time = datetime.now(timezone.utc)
        start_mono = _time.monotonic()
        try:
            session = get_current_session()
            order = session.next_span_order()
            span_name = consume_pending_span_name()
            _capture_pydantic_ai_prompt_at(self, messages, order)
        except Exception:
            pass  # fail-safe

        _in_pydantic_ai.set(True)
        try:
            response = await original(self, *args, **kwargs)
        except Exception as _exc:
            # Provider rejection: the inner provider's OTel span is
            # hard-dropped under the pydantic_ai guard, so this manual row is
            # the ONLY carrier (L-2). Guard restored first (the finally's
            # repeat set is idempotent); 0/0 usage (response=None);
            # model/provider from the instance. Fully fail-open; identity
            # re-raise (golden rule).
            try:
                _in_pydantic_ai.set(False)
                _log_pydantic_ai(
                    self, session, messages, None, order, span_name,
                    start_time,
                    latency=_build_non_stream_latency(
                        start_mono, _time.monotonic()),
                    call_outcome=build_call_outcome(
                        _exc, int(max(0.0, _time.monotonic()
                                      - start_mono) * 1000)),
                    obs_key=_obs_key,
                )
            except Exception:
                pass
            raise
        finally:
            _in_pydantic_ai.set(False)

        try:
            if response is not None:
                _capture_pydantic_ai_response_at(self, response, order)
            _log_pydantic_ai(
                self, session, messages, response, order, span_name, start_time,
                # is_streaming=False + total_ms via the shared non-stream builder
                # (ttft/generation null). None on missing anchor → is_streaming 0.
                latency=_build_non_stream_latency(start_mono, _time.monotonic()),
                obs_key=_obs_key,
            )
        except Exception:
            pass  # fail-open
        return response

    return request_wrapper


def _make_pydantic_ai_request_stream_wrapper(original):
    """Build the wrapper for `Model.request_stream`.

    `request_stream` is decorated with `@asynccontextmanager`, so calling it
    returns an `AbstractAsyncContextManager` synchronously (it is NOT itself a
    coroutine). The wrapper must therefore be `def`, not `async def` — same
    pattern as the Anthropic Messages.stream wrapper.
    """
    @functools.wraps(original)
    def request_stream_wrapper(self, *args, **kwargs):
        if (in_langchain() or in_litellm() or in_llamaindex()
                or in_pydantic_ai() or in_agno()):
            return original(self, *args, **kwargs)
        messages = args[0] if args else kwargs.get("messages")
        mgr = original(self, *args, **kwargs)
        return _PydanticAIAsyncStreamMgr(mgr, self, messages)

    return request_stream_wrapper


# ── pydantic_ai tool spans ──────────────────────────────────────────
# pydantic_ai runs `@agent.tool` / `@agent.tool_plain` functions itself via
# `ToolManager.handle_call(call)` — outside any patched provider call. It only
# emits an OTel span for the tool when the app enables pydantic_ai/logfire
# instrumentation, which most apps don't, so TokenPolice never saw tool calls.
# Wrap `handle_call` to emit a zero-usage `tool` row into the active session's
# trace. Transparent: returns/raises the original verbatim (never breaks the app).

def _emit_pydantic_ai_tool_row(call, result, started_at, failed, err_msg):
    """Emit a TokenPolice `tool` row for a pydantic_ai tool execution.

    Mirrors telemetry._log_tool_span; raw args/results are reduced to
    (sha1-16, length) and never stored. Fail-open."""
    client = get_client()
    if client is None:
        return
    session = get_current_session()
    ids = manual_span_ids(session)

    try:
        args_str = call.args_as_json_str()
    except Exception:
        args_str = getattr(call, "args", None)
    param_hash, param_len = _hash_len(args_str)
    result_hash, result_len = _hash_len(result)
    name = str(getattr(call, "tool_name", "") or "tool")

    metadata = {"workflow_name": session.workflow_name}
    if session.session_id:
        metadata["session_id"] = session.session_id
    # B4: a `tool` row executes no model call, so it can never own reroute
    # provenance — it must not inherit the marker a sibling LLM call left on
    # the session metadata. Copy without it; never re-add.
    _copy_session_metadata(metadata, session)

    duration_ms = 0
    end_at = datetime.now(timezone.utc)
    try:
        from ._classify import to_duration_ms
        duration_ms = to_duration_ms(
            (end_at - started_at).total_seconds() * 1000.0
        )
    except Exception:
        pass
    call_outcome = {"status": "failed" if failed else "success", "duration_ms": duration_ms}
    if failed and err_msg:
        # Route the raw value through the central scrub helper (no
        # pre-stringify); the default 'redacted' mode ships a hash, never the raw error string.
        from ._classify import scrub_error_message, resolve_error_detail
        call_outcome.update(scrub_error_message(err_msg, resolve_error_detail()))

    span_obj = {
        "trace_id": ids["trace_id"],
        "span_id": ids["span_id"],
        "parent_span_id": ids["parent_span_id"],
        "span_kind": "tool",
        "span_name": name,
        "span_order": 0,
        "start_time": started_at.isoformat() if started_at else None,
        "end_time": end_at.isoformat(),
    }

    client.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model="",
        provider="",
        input_tokens=0,
        output_tokens=0,
        cached_tokens=0,
        metadata=metadata,
        span=span_obj,
        prompt_composition=[],
        response_composition=[],
        tool={
            "name": name,
            "type": "function",
            "call_id": str(getattr(call, "tool_call_id", "") or ""),
            "param_hash": param_hash,
            "param_length": param_len,
            "result_hash": result_hash,
            "result_length": result_len,
        },
        call_outcome=call_outcome,
    )


def _make_pydantic_ai_tool_wrapper(original, extract_call=lambda first: first):
    """Wrap pydantic_ai's tool-execution method to emit a TokenPolice tool row.

    `extract_call` normalizes the wrapped method's first positional arg to the
    `ToolCallPart`: identity for 0.x `ToolManager.handle_call(call, …)`, and
    `validated.call` for 1.x `ToolManager.execute_tool_call(validated, …)`
    (1.x split execution into validate/execute and the agent drives
    `execute_tool_call`; `handle_call` became an unused convenience wrapper).

    Transparent: the original's return value / exception is propagated
    unchanged. Emits only for real (non-partial) executions of non-`output`
    tools — output tools are pydantic_ai's structured-output mechanism, not user
    tools, and `allow_partial=True` calls (0.x only) are streaming arg-validation
    passes.
    """
    @functools.wraps(original)
    async def tool_exec_wrapper(self, first, *args, **kwargs):
        # Signature varies across pydantic_ai versions (0.x positional
        # `allow_partial`; 1.x keyword-only `approved`/`metadata`, no
        # `allow_partial`). Stay signature-agnostic — forward *args/**kwargs
        # untouched — and derive allow_partial defensively (absent ⇒ False ⇒ a
        # real, non-partial execution ⇒ emit).
        allow_partial = kwargs.get("allow_partial", args[0] if args else False)
        call = None
        emit = False
        try:
            if allow_partial is False:
                call = extract_call(first)
                tools = getattr(self, "tools", None) or {}
                tool = tools.get(getattr(call, "tool_name", None))
                kind = getattr(getattr(tool, "tool_def", None), "kind", None)
                emit = kind != "output"
        except Exception:
            emit = False

        if not emit:
            return await original(self, first, *args, **kwargs)

        started_at = datetime.now(timezone.utc)
        failed = False
        err_msg = ""
        result = None
        try:
            result = await original(self, first, *args, **kwargs)
            return result
        except Exception as e:
            failed = True
            err_msg = str(e)
            raise
        finally:
            try:
                _emit_pydantic_ai_tool_row(call, result, started_at, failed, err_msg)
            except Exception:
                pass  # fail-open: emission must never affect the tool call

    return tool_exec_wrapper


def _instrument_pydantic_ai():
    """Patch every concrete pydantic_ai Model subclass's `request` and
    `request_stream`. Silently skip classes whose module isn't importable."""
    for module_path, class_name in _PYDANTIC_AI_TARGETS:
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if cls is None:
            continue

        # request — async coroutine
        orig_request = getattr(cls, "request", None)
        if orig_request is not None and (cls, "request") not in _originals:
            _originals[(cls, "request")] = orig_request
            try:
                setattr(cls, "request", _make_pydantic_ai_request_wrapper(orig_request))
            except Exception as e:
                logger.debug(
                    f"TokenPolice: failed to wrap {class_name}.request: {e}"
                )

        # request_stream — sync method returning an async context manager
        orig_stream = getattr(cls, "request_stream", None)
        if orig_stream is not None and (cls, "request_stream") not in _originals:
            _originals[(cls, "request_stream")] = orig_stream
            try:
                setattr(
                    cls, "request_stream",
                    _make_pydantic_ai_request_stream_wrapper(orig_stream),
                )
            except Exception as e:
                logger.debug(
                    f"TokenPolice: failed to wrap {class_name}.request_stream: {e}"
                )

    # Tool execution: wrap the ToolManager method the agent actually drives so
    # tool/function calls land as `tool` rows (pydantic_ai runs tools outside any
    # patched provider call). Version differences:
    # - 0.x: module `pydantic_ai._tool_manager`; agent calls
    # `handle_call(call, …)` — wrap it, the call IS the ToolCallPart.
    # - 1.x: module renamed to public `pydantic_ai.tool_manager`; execution was
    # split and the agent calls `execute_tool_call(validated, …)` while
    # `handle_call` became an unused convenience wrapper — wrap
    # `execute_tool_call` and reach the ToolCallPart via `validated.call`.
    for tm_path in ("pydantic_ai.tool_manager", "pydantic_ai._tool_manager"):
        try:
            tm_module = importlib.import_module(tm_path)
        except ImportError:
            continue  # not this layout (or pydantic_ai absent) — try the next
        ToolManager = getattr(tm_module, "ToolManager", None)
        if ToolManager is None:
            continue
        if getattr(ToolManager, "execute_tool_call", None) is not None:
            method_name, extract = "execute_tool_call", lambda v: getattr(v, "call", v)
        elif getattr(ToolManager, "handle_call", None) is not None:
            method_name, extract = "handle_call", lambda c: c
        else:
            break  # found the module but no known execution method
        if (ToolManager, method_name) not in _originals:
            try:
                orig_exec = getattr(ToolManager, method_name)
                _originals[(ToolManager, method_name)] = orig_exec
                setattr(ToolManager, method_name,
                        _make_pydantic_ai_tool_wrapper(orig_exec, extract))
            except Exception as e:
                logger.debug(f"TokenPolice: failed to wrap ToolManager.{method_name}: {e}")
        break  # found the module — don't also wrap the legacy path


# ═══════════════════════════════════════════════════════════════════
# Agno wrappers — for `agno.agent.Agent.run` / `Agent.arun`.
#
# Agno is an outer-loop framework: each `Agent.run` / `arun` invokes the
# underlying provider SDK (openai, anthropic, google.genai, ...) N times,
# once per tool-call iteration. The provider SDK is ALSO patched, and its
# OpenLLMetry instrumentor emits a span per inner call. Our wrappers run:
# - ONE pre-flight `/check` at the Agno boundary (so the customer isn't
# blocked mid-agent-run), and
# - Hold the `_in_agno` guard across the entire run so the nested provider
# enforcer wrappers short-circuit (no duplicate pre-flight check, no
# duplicate prompt-composition capture).
# The inner provider OTel spans flow through unchanged — each one becomes
# its own /log row with per-step composition (system + accumulated messages
# + tool calls + tool results) and per-step token usage. The result: N rows
# per `Agent.run`, properly nested under the workflow span.
#
# Agno is Python-only; no Node SDK mirror exists.
# ═══════════════════════════════════════════════════════════════════


class _AgnoSyncStreamIter:
    """Proxy iterator around Agno's sync RunOutputEvent stream.

    Holds `_in_agno=True` across iteration so the inner provider enforcer
    wrappers short-circuit and don't re-run the pre-flight check. Per-step
    telemetry comes from the inner provider OTel spans (one per tool-call
    iteration). The guard is reset on exhaustion or iterator abandonment."""

    def __init__(self, inner, agent):
        self._inner = inner
        self._agent = agent
        self._done = False

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self._inner)
        except StopIteration:
            self._finalize()
            raise
        except BaseException:
            # A mid-stream provider error (or teardown such as
            # GeneratorExit/CancelledError) must also reset the guard —
            # otherwise `_in_agno` leaks True and a subsequent Agent.run in
            # the same context skips the pre-flight /check (under-enforcement).
            # Reset fires only on the exception path, never on a normal
            # per-event return (a `finally` would break cross-event
            # enforcement). Defensively total: the reset can never replace the
            # original exception, which is always re-raised verbatim.
            try:
                self._finalize()
            except BaseException:
                pass
            raise

    def __del__(self):
        try:
            if not self._done:
                _in_agno.set(False)
        except Exception:
            pass

    def _finalize(self):
        if self._done:
            return
        self._done = True
        try:
            _in_agno.set(False)
        except Exception:
            pass


class _AgnoLazyAsyncStreamIter:
    """Proxy async iterator around Agno's async RunOutputEvent stream.

    Constructed synchronously from the wrapper (so the caller can do
    `async for x in agent.arun(..., stream=True)` without an extra `await`),
    but defers the async pre-flight check + `_in_agno` guard set until the
    first `__anext__`. Per-step telemetry comes from the inner provider OTel
    spans (one per tool-call iteration) — this iterator only manages the
    pre-flight check + framework guard. The guard is reset on stream
    exhaustion or iterator abandonment.
    """

    def __init__(self, original, agent, args, kwargs):
        self._original = original
        self._agent = agent
        self._args = args
        self._kwargs = kwargs
        self._inner = None
        self._started = False
        self._done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._started:
            await self._start()
            self._started = True
        try:
            return await self._inner.__anext__()
        except StopAsyncIteration:
            self._finalize()
            raise
        except BaseException:
            # Mirror of the sync path — a mid-stream provider error or
            # teardown (CancelledError/GeneratorExit) must reset the guard so a
            # subsequent Agent.arun in the same task still runs the pre-flight
            # /check. Reset fires only on the exception path (never a normal
            # per-event return); defensively total so the original exception is
            # always re-raised verbatim.
            try:
                self._finalize()
            except BaseException:
                pass
            raise

    async def _start(self):
        # Pre-flight check happens before we call the original. If it raises
        # TokenPoliceBlockedError, the stream never starts and the customer's
        # `async for` exits with that exception — by design.
        _hint, _prov = _agno_check_ctx(self._agent)
        await _run_async_check(model_hint=_hint, provider=_prov)

        _in_agno.set(True)
        try:
            self._inner = self._original(
                self._agent, *self._args, **self._kwargs
            )
            # Tolerate Agno releases where arun_dispatch returns an awaitable
            # rather than the iterator directly.
            if hasattr(self._inner, "__await__"):
                self._inner = await self._inner
        except BaseException:
            # Widened from `except Exception` so a CancelledError /
            # GeneratorExit during first-event startup (after _in_agno.set(True)
            # above) also resets the guard before propagating — otherwise the
            # flag leaks True for the rest of the task. Reset then re-raise
            # verbatim.
            _in_agno.set(False)
            raise

    def __del__(self):
        try:
            if self._started and not self._done:
                _in_agno.set(False)
        except Exception:
            pass

    def _finalize(self):
        if self._done:
            return
        self._done = True
        try:
            _in_agno.set(False)
        except Exception:
            pass


def _agno_streaming(instance, kwargs) -> bool:
    """Resolve whether an Agno run is streaming.

    An explicit ``stream=`` kwarg always wins (even ``stream=False`` overrides an
    instance-level ``stream=True``). When the kwarg is absent, fall back to the
    Agent instance's own ``stream`` attribute, which Agno also honors. Guarded so
    attribute access on an exotic instance can never break the wrapped call."""
    try:
        if "stream" in kwargs:
            return bool(kwargs.get("stream"))
        return bool(getattr(instance, "stream", False))
    except Exception:
        return False


# Agno's concrete Model classes declare a canonical `provider` string
# ("OpenAI"/"Anthropic"/"Google"/...) plus the model id on `.id`. Matched
# EXACTLY (lowercased), never by substring: agno also ships CerebrasOpenAI /
# LlamaOpenAI / LiteLLMOpenAI / AzureOpenAI, which a substring match would
# mislabel as openai — and a wrong slug makes every provider-targeted REROUTE
# look cross-provider. Unknown → None (the previous behaviour: provider omitted).
_AGNO_PROVIDER_NAMES = {
    "openai": "openai", "openaichat": "openai", "openairesponses": "openai",
    "anthropic": "anthropic", "claude": "anthropic",
    "google": "google", "gemini": "google",
    "mistral": "mistral", "mistralchat": "mistral",
    "groq": "groq",
    "cohere": "cohere",
}


def _agno_check_ctx(agent):
    """Best-effort ``(model_hint, provider)`` from an agno Agent for the
    pre-flight. Conservative: unknown model class → ``(model_id, None)``.
    Never raises — a hostile/exotic Agent just degrades to a bare check."""
    try:
        model = getattr(agent, "model", None)
    except Exception:
        return None, None
    if model is None:
        return None, None
    try:
        mid = getattr(model, "id", None)
        model_hint = mid if isinstance(mid, str) and mid else None
    except Exception:
        model_hint = None
    provider = None
    try:
        raw = getattr(model, "provider", None)
        if isinstance(raw, str):
            provider = _AGNO_PROVIDER_NAMES.get(raw.strip().lower())
    except Exception:
        provider = None
    if provider is None:
        try:
            provider = _AGNO_PROVIDER_NAMES.get(type(model).__name__.strip().lower())
        except Exception:
            provider = None
    return model_hint, provider


def _make_agno_run_wrapper(original):
    """Build the sync wrapper for `Agent.run`.

    Agno is an outer-loop framework: each `Agent.run` invokes the provider SDK
    (openai/anthropic/google) N times (one per tool-call iteration). Rather
    than collapse those into a single rollup, we run ONE pre-flight check at
    the Agent boundary and let the inner provider OTel spans emit normally —
    that gives N rows under the workflow, each with its own composition
    (including tool calls / tool results) and per-step token counts. The
    `_in_agno` guard only short-circuits the inner enforcer wrappers so they
    don't re-run the pre-flight check or duplicate prompt-composition capture.
    """
    @functools.wraps(original)
    def run_wrapper(self, *args, **kwargs):
        if (in_langchain() or in_litellm() or in_llamaindex()
                or in_pydantic_ai() or in_agno()):
            return original(self, *args, **kwargs)

        _hint, _prov = _agno_check_ctx(self)
        # may raise TokenPoliceBlockedError
        _run_sync_check(model_hint=_hint, provider=_prov)

        streaming = _agno_streaming(self, kwargs)

        _in_agno.set(True)
        try:
            result = original(self, *args, **kwargs)
        except Exception:
            _in_agno.set(False)
            raise

        if streaming:
            # Stream: keep the guard set across iterator consumption; the
            # wrapper resets it on exhaustion / abandonment.
            return _AgnoSyncStreamIter(result, self)

        try:
            _in_agno.set(False)
        except Exception:
            pass
        return result

    return run_wrapper


def _make_agno_arun_wrapper(original):
    """Build the wrapper for `Agent.arun`.

    Agno's `arun` is NOT `async def`: it's a regular function that internally
    dispatches to either `_arun(...)` (a coroutine — caller must `await`) or
    `_arun_stream(...)` (an async-generator — caller does `async for`). The
    wrapper preserves that calling convention so existing Agno code keeps
    working without an extra `await` around the streaming form.

      - `await agent.arun(prompt)` → RunOutput
      - `async for x in agent.arun(prompt, stream=True)` → events

    See `_make_agno_run_wrapper` for why we only enforce + guard at this
    boundary and let inner provider OTel spans handle per-iteration telemetry.
    """
    @functools.wraps(original)
    def arun_wrapper(self, *args, **kwargs):
        if (in_langchain() or in_litellm() or in_llamaindex()
                or in_pydantic_ai() or in_agno()):
            return original(self, *args, **kwargs)

        streaming = _agno_streaming(self, kwargs)

        if streaming:
            return _AgnoLazyAsyncStreamIter(original, self, args, kwargs)

        async def _coro():
            _hint, _prov = _agno_check_ctx(self)
            # may raise TokenPoliceBlockedError
            await _run_async_check(model_hint=_hint, provider=_prov)
            _in_agno.set(True)
            try:
                return await original(self, *args, **kwargs)
            finally:
                try:
                    _in_agno.set(False)
                except Exception:
                    pass

        return _coro()

    return arun_wrapper


def _instrument_agno():
    """Patch `agno.agent.Agent.run` and `Agent.arun`. Silently skip if the
    agno package isn't installed."""
    try:
        module = importlib.import_module("agno.agent")
    except ImportError:
        return
    cls = getattr(module, "Agent", None)
    if cls is None:
        return

    # run — sync method (may return RunOutput or Iterator when stream=True)
    orig_run = getattr(cls, "run", None)
    if orig_run is not None and (cls, "run") not in _originals:
        _originals[(cls, "run")] = orig_run
        try:
            setattr(cls, "run", _make_agno_run_wrapper(orig_run))
        except Exception as e:
            logger.debug(f"TokenPolice: failed to wrap Agent.run: {e}")

    # arun — async coroutine (returns RunOutput or AsyncIterator on stream=True)
    orig_arun = getattr(cls, "arun", None)
    if orig_arun is not None and (cls, "arun") not in _originals:
        _originals[(cls, "arun")] = orig_arun
        try:
            setattr(cls, "arun", _make_agno_arun_wrapper(orig_arun))
        except Exception as e:
            logger.debug(f"TokenPolice: failed to wrap Agent.arun: {e}")


# ═══════════════════════════════════════════════════════════════════
# Anthropic Messages.stream wrapper — context-manager streaming.
#
# `client.messages.stream(**params)` returns a `MessageStreamManager` used as
# with client.messages.stream(...) as stream:
# for event in stream: ...
# final = stream.get_final_message()
#
# CrewAI's native AnthropicCompletion uses this pattern (it does NOT use
# `messages.create(stream=True)`). The OpenLLMetry anthropic instrumentor
# wraps Messages.stream and emits a span; whether it populates gen_ai.usage.*
# on that span depends on the instrumentor version (0.60 did not, 0.61 does)
# — NEVER branch on it. To guarantee tokens AND response composition
# regardless, we wrap Messages.stream ourselves: pre-flight check + capture
# prompt at call time, then on the context-manager exit extract usage /
# response from the SDK's `final_message` and manually log. The duplicate
# OpenLLMetry span is suppressed via a per-call ContextVar window (armed
# around the statements where the instrumentor starts it — see
# claim_anthropic_stream_span in context.py); the `_defer_telemetry` +
# drop-on-exit machinery below stays as a best-effort fallback.
# ═══════════════════════════════════════════════════════════════════

class _AnthropicStreamMgrWrapper:
    """Sync wrapper around an Anthropic MessageStreamManager.

    Preserves the customer's `with client.messages.stream(...) as stream:` API
    while running our pre-flight check and capturing usage/composition on
    `__exit__` via `stream.get_final_message()`.
    """

    def __init__(self, mgr, kwargs, base_url="", serving=None, span_window=None):
        self._mgr = mgr
        self._kwargs = kwargs
        # Serving endpoint of the bound client (e.g. https://api.minimax.io/anthropic).
        # Forwarded in the manual log so the service can remap the provider from
        # "anthropic" to the actual serving provider (host -> provider). The
        # `.create` wrapper does this via _stash_api_base; `.stream` bypasses
        # `.create`, so we capture it here. "" when unknown (fail-safe).
        self._base_url = base_url
        # {"provider", "serving_unverified"} as resolved by the stream()
        # wrapper's _resolve_serving_provider — the SAME value its pre-flight
        # used, so an enter-failure row classifies its provider exactly like
        # its non-stream failure siblings (api.minimax.io -> minimax). {} when
        # not threaded (fail-safe -> "anthropic").
        self._serving = serving or {}
        # The W1 suppression record the stream() wrapper armed around the
        # wrapped `stream()` call (see claim_anthropic_stream_span). Threaded
        # in so __enter__/__aenter__ can shape-detect whether the instrumentor
        # already started its span there (record["seen"]) and arm the
        # manager-enter window only when it did not. None when the wrapper is
        # constructed directly (several unit tests do) — then no window is
        # ever armed here and behaviour is identical to an unarmed call.
        self._span_window = span_window
        self._session = None
        self._order = 0
        self._span_name = None
        self._start_time = None
        self._prev_defer = False
        self._stream = None
        # Object ids of the deferred spans already queued (by other in-flight
        # calls) at the moment we start deferring in __enter__. On exit we drop
        # only spans NOT in this set that belong to THIS call — never another
        # call's queued telemetry. None until __enter__ completes; None means
        # "enter never finished deferring", so exit drops nothing of ours.
        self._enter_deferred_ids = None
        # Latency anchors (monotonic). _req_start_mono is set here (wrapper is
        # created right after messages.stream() returns) and refreshed at
        # __enter__ together with _start_time (span wall) — the anthropic SDK
        # fires the HTTP request inside the manager's __enter__, so both share
        # the true request-start anchor. _ttft_mono is marked by
        # _mark_ttft() on the first content event the customer pulls through
        # the iteration proxy; stays None if iteration is never tapped
        # (latency then carries total_ms only — never a fabricated TTFT).
        self._req_start_mono = _time.monotonic()
        self._ttft_mono = None
        # This call's obs key. The SYNC path's check runs in sync_stream_wrapper
        # immediately before this constructor (same context, no intervening
        # check), so the contextvar still holds it here. The async subclass
        # re-captures in __aenter__ (its check runs there instead). _finalize
        # drains with this captured value — never a re-read, since the stream
        # is drained later, when the var may hold another call's key.
        try:
            self._obs_key = _state.get_current_obs_key()
        except Exception:
            self._obs_key = None

    def _mark_ttft(self, event):
        """One-shot TTFT marker — called per yielded stream item. Fail-safe."""
        if self._ttft_mono is not None:
            return
        try:
            if isinstance(event, str):
                # text_stream yields plain text chunks.
                if event:
                    self._ttft_mono = _time.monotonic()
                return
            etype = getattr(event, "type", "") or ""
            if etype in ("content_block_start", "content_block_delta", "text"):
                self._ttft_mono = _time.monotonic()
        except Exception:
            pass

    def _wrap_stream_for_ttft(self, stream):
        """Return a transparent iteration proxy over the SDK MessageStream so
        the first content event timestamps TTFT. On ANY failure the raw stream
        is returned unchanged — the customer's API never degrades."""
        try:
            return _AnthropicMessageStreamProxy(stream, self._mark_ttft)
        except Exception:
            return stream

    def _emit_enter_failure(self, exc):
        """Emit a failure row when the VENDOR manager's __enter__/__aenter__
        raises — the provider HTTP request fires inside it, so e.g. a 401
        surfaces here. Python skips __(a)exit__ when enter raises, so
        _finalize never runs, and the orphaned OpenLLMetry stream span never
        reaches on_end — without this handler the rejection produces ZERO
        rows and leaks the pre-flight observations / _local_decision. (The
        span-suppression window needs no cleanup here: its `finally` disarm
        at the enter site already ran.) Mirrors the generic non-stream failure path
        (_stash_attempt_context → build_call_outcome → _emit_call_failure_log,
        which drains observations and _local_decision).

        Shared by the async subclass (it inherits this method; its __aenter__
        re-captures _obs_key and refreshes _req_start_mono BEFORE the guarded
        enter, so both are correct here). Deliberately does NOT touch the
        orphaned span, _defer_telemetry, or self._mgr.__(a)exit__. Every step
        is individually fail-safe, and the caller additionally swallows
        anything raised here so the provider error always propagates by
        identity."""
        session = get_current_session()
        if session is None:
            return
        try:
            elapsed_ms = max(0, int((_time.monotonic() - self._req_start_mono) * 1000))
        except Exception:
            elapsed_ms = 0
        # Serving provider exactly as this wrapper's pre-flight resolved it
        # (api.minimax.io -> minimax); wire_key stays the module slug
        # ("anthropic") like the generic wrapper's stash, so the failure row's
        # provider + usage shape match its non-stream failure siblings.
        try:
            provider = (getattr(self, "_serving", None) or {}).get("provider") or "anthropic"
        except Exception:
            provider = "anthropic"
        try:
            _stash_attempt_context(
                session, provider, "anthropic.resources.messages", (),
                self._kwargs, wire_key="anthropic",
            )
        except Exception:
            pass
        # Forensics: stash the prompt at a freshly-reserved order so
        # _emit_call_failure_log's one-order-back fallback attaches it (the
        # same reserve-then-stash pattern that fallback documents for
        # manual-path wrappers). Keyed by trace_id:order, so a concurrent
        # call's stash is never consumed by mistake here.
        try:
            comp = build_prompt_composition("anthropic", self._kwargs)
            if comp:
                order = session.next_span_order()
                if not hasattr(session, '_pending_compositions'):
                    session._pending_compositions = {}
                session._pending_compositions.setdefault(
                    f"{session.trace_id}:{order}", {}
                )["prompt"] = comp
        except Exception:
            pass
        try:
            session._call_outcome = build_call_outcome(exc, elapsed_ms)
        except Exception:
            pass
        # Drains this call's keyed observations and claims this call's keyed
        # local_decision (both consumed inside), like every other failure path.
        _emit_call_failure_log(get_client(), session, obs_key=self._obs_key)

    def __enter__(self):
        try:
            # The HTTP request fires inside the manager's __enter__ — anchor
            # TTFT/total latency AND span start here (request start), not at
            # wrapper construction and not after the handshake. stamping
            # _start_time after __enter__ excluded handshake from duration_ms
            # while TTFT still included it → ttft > duration on slow gateways.
            self._req_start_mono = _time.monotonic()
            self._start_time = datetime.now(timezone.utc)
        except Exception:
            pass
        # The provider HTTP request fires inside the vendor __enter__, and
        # Python does NOT call __exit__ when __enter__ raises — so a
        # request-time rejection (e.g. 401) would skip _finalize AND never
        # reach the instrumentor's on_end: zero rows plus leaked pre-flight
        # state. The guard wraps ONLY this one statement.
        #
        # W2 suppression window: armed ONLY when the W1 record (armed by the
        # stream() wrapper around the wrapped `stream()` call) reports
        # seen == 0 — i.e. the installed instrumentor did NOT start its
        # stream span inside `stream()`, so a FUTURE instrumentor that starts
        # it at manager-enter instead is still caught here. Pure runtime
        # shape-detection — never version-detection. That shape-detection is
        # what earns the guard its place (it also avoids needlessly re-arming
        # a spent record over an outer call's). Note it is NOT here to
        # prevent a double suppression slot: arm() reuses this record without
        # resetting `seen` and `limit` is 1, so unconditional arming would be
        # behaviourally identical today — do not delete the guard on
        # "discovering" that equivalence. This
        # per-call window replaces the old session-wide one-shot suppress
        # flag, which never matched on this seam (the span starts inside
        # `stream()`, before this wrapper exists) and, unconsumed, leaked
        # onto the NEXT anthropic call's telemetry. The `finally`
        # disarm makes an enter-failure leak structurally impossible.
        _w2 = None
        try:
            if isinstance(self._span_window, dict) and self._span_window.get("seen") == 0:
                _w2 = arm_anthropic_stream_span_window(self._span_window)
        except Exception:
            _w2 = None  # fail-open: unarmed → worst case an extra row
        try:
            self._stream = self._mgr.__enter__()
        except TokenPoliceBlockedError:
            # Defense-in-depth: the vendor enter can't realistically raise a
            # TP denial, but one must never be reclassified as a provider
            # failure row.
            raise
        except Exception as exc:
            try:
                self._emit_enter_failure(exc)
            except Exception:
                pass  # fail-safe: telemetry must never mask the provider error
            raise
        finally:
            disarm_anthropic_stream_span_window(_w2)
        try:
            self._session = get_current_session()
            self._order = self._session.next_span_order()
            self._span_name = consume_pending_span_name()
            if not self._span_name and in_agno():
                self._span_name = f"agent_step_{self._order + 1}"
            # Stash prompt composition keyed by the order WE just reserved (the
            # OpenLLMetry span's on_start has already fired by now and consumed
            # an earlier order — that span will be dropped on exit, so we want
            # OUR manual log to carry the prompt).
            comp = build_prompt_composition("anthropic", self._kwargs)
            if comp:
                if not hasattr(self._session, '_pending_compositions'):
                    self._session._pending_compositions = {}
                self._session._pending_compositions.setdefault(
                    f"{self._session.trace_id}:{self._order}", {}
                )["prompt"] = comp
            # Defer the OpenLLMetry anthropic stream span so we can drop it
            # on exit — best-effort FALLBACK behind the suppression window
            # (whether that span carries usage depends on the instrumentor
            # version; never branch on it — the window suppresses it at
            # on_start either way, and the scoped drop below only catches a
            # usage-less dud the window happened to miss).
            self._prev_defer = getattr(self._session, '_defer_telemetry', False)
            self._session._defer_telemetry = True
            # Snapshot the ids already in the deferred buffer NOW (before our
            # own dud can be queued) so exit drops only spans this call adds —
            # never another concurrent call's queued telemetry.
            existing = getattr(self._session, '_deferred_spans', None) or []
            self._enter_deferred_ids = {id(p) for p in existing}
        except Exception:
            pass  # fail-safe
        return self._wrap_stream_for_ttft(self._stream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self._finalize()
        except Exception:
            pass  # fail-safe
        try:
            return self._mgr.__exit__(exc_type, exc_val, exc_tb)
        finally:
            # The inner instrumentor's on_end fires inside `self._mgr.__exit__()`
            # above. Only NOW is it safe to reset defer + drop the queued (empty)
            # OTel dud — doing it inside `_finalize` (before the inner __exit__)
            # would let that span emit a duplicate empty row. Drop ONLY this
            # call's own entries (our reserved order + any usage-less dud queued
            # in our window on our trace); spans other concurrent calls queued
            # survive. Covered by test_anthropic_stream_deferred_buffer.py.
            try:
                if self._session is not None:
                    self._session._defer_telemetry = self._prev_defer
                    if self._enter_deferred_ids is not None:
                        _drop_deferred_spans(
                            self._session,
                            keep_ids=self._enter_deferred_ids,
                            order=self._order,
                            trace_id=getattr(self._session, 'trace_id', None),
                        )
            except Exception:
                pass

    def _finalize(self):
        session = self._session
        if session is None:
            return
        tp = get_client()
        if tp is None or self._stream is None:
            return
        try:
            final_msg = self._stream.get_final_message()
        except Exception:
            return
        if final_msg is None:
            return

        # This context-manager path bypasses .create entirely and logs by
        # hand, so neither pending-id producer ever ran and every streamed tool
        # row lost its id. The final message's content[] carries the tool_use
        # blocks with ids. Stashed before anything below can fail; REPLACE on
        # every capture (even empty) so a no-tool streamed turn clears stale ids,
        # matching the non-streamed path.
        try:
            set_pending_tool_calls(extract_pending_tool_calls("anthropic", final_msg))
        except Exception:
            pass

        usage = getattr(final_msg, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        # Forward the verbatim anthropic usage shape when it serializes: the
        # `anthropic_messages` usage shape then prices cache writes
        # (cache_creation_input_tokens, split 5m/1h) at the write rate and keeps
        # input_tokens exclusive of cache. Without it, log_sync synthesises an
        # openai_compatible shape that folds writes into the read bucket and
        # under-bills cache writes ~4-10x. Gated: on any serialization
        # failure fall back to today's behavior — an anthropic_messages shape
        # with empty raw maps to all-zeros server-side, which is worse.
        raw_usage = _serialize_anthropic_raw_usage(usage)
        if raw_usage:
            cached_tokens = cache_read  # reads only; writes priced via raw
            usage_block = {"shape": "anthropic_messages", "raw": raw_usage}
        else:
            cached_tokens = cache_read + cache_creation
            usage_block = None
        model = str(getattr(final_msg, "model", "") or self._kwargs.get("model") or "unknown")
        # Family-gated collapse of the provider's dated echo to the requested
        # (post-reroute) alias so one logical model stays one cost-by-model row.
        model = _prefer_requested_model((self._kwargs or {}).get("model"), model)
        response_comp = build_response_composition("anthropic", final_msg) or []

        # Pull our prompt stash.
        prompt_comp = []
        comp_key = f"{session.trace_id}:{self._order}"
        if hasattr(session, '_pending_compositions'):
            comp_data = session._pending_compositions.pop(comp_key, {})
            prompt_comp = comp_data.get("prompt", []) or prompt_comp

        span_obj = {
            **manual_span_ids(session),
            "span_kind": "llm",
            "span_name": self._span_name or model,
            "span_order": self._order,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }
        metadata = {"workflow_name": session.workflow_name}
        if session.session_id:
            metadata["session_id"] = session.session_id
        # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
        # session metadata WITHOUT it, then re-add it only when this row belongs
        # to the call that was actually rerouted (exact obs-key match).
        _copy_session_metadata(metadata, session)
        _stamp_routing_marker(metadata, session, getattr(self, "_obs_key", None))

        # Drain applied local_decision + shadow/reject observations so the
        # anthropic .stream() manual log confirms REQUEST_REROUTED / REROUTE_REJECTED
        # on /log (parity with _log_manual / _flush_deferred_spans). Keyed on the
        # obs key captured at wrapper construction (the stream is drained later,
        # when the contextvar may hold another call's key). NOTE: unlike the
        # Node SDK (whose anthropic .stream() delegates to create({stream:true}) and
        # therefore runs TWO pre-flights, needing _pushObservationOnce dedup), the
        # Python anthropic SDK's stream() posts directly and runs exactly ONE
        # pre-flight per call — a plain drain cannot double-emit; do not add a dedup latch.
        pending_local_decision = None
        pending_observations = None
        try:
            _stream_obs_key = getattr(self, "_obs_key", None)
            pending_local_decision = _claim_local_decision(session, _stream_obs_key)
            pending_observations = _state.drain_observations(_stream_obs_key)
        except Exception:
            pending_local_decision = None
            pending_observations = None

        tp.log_sync(
            user_id=session.user_id,
            paid_plan=session.paid_plan,
            plan_source=getattr(session, "plan_source", None),
            workflow_name=session.workflow_name,
            session_id=session.session_id,
            model=model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            # None in the fallback branch → client.log_sync treats it exactly
            # like an omitted usage= (it gates on `if usage:`), preserving today.
            usage=usage_block,
            metadata=metadata,
            span=span_obj,
            prompt_composition=prompt_comp,
            response_composition=response_comp,
            # Serving endpoint (e.g. api.minimax.io) so the service remaps the
            # provider from anthropic to the actual serving provider (minimax).
            model_extras={"api_base": self._base_url} if self._base_url else None,
            # Streamed call → TTFT/total anchored at the manager __enter__
            # (request start). Mirrors the OpenAI streaming-tap pattern;
            # _build_stream_latency returns None on any anomaly (key absent).
            latency=_build_stream_latency(self._req_start_mono, self._ttft_mono,
                                          _time.monotonic()),
            local_decision=pending_local_decision,
            observations=pending_observations or None,
        )


class _AnthropicMessageStreamProxy:
    """Transparent iteration proxy over an anthropic ``MessageStream`` /
    ``AsyncMessageStream`` used purely to observe the first content event for
    TTFT. Every attribute (``get_final_message``, ``until_done``, ``close``,
    ``current_message_snapshot``, …) passes straight through to the wrapped
    stream; ``text_stream`` is re-wrapped lazily so text-only consumers also
    mark TTFT. The tap callback is fail-safe — a probe failure costs only the
    metric, never the customer's iteration.
    """

    __slots__ = ("_tp_stream", "_tp_on_event")

    def __init__(self, stream, on_event):
        object.__setattr__(self, "_tp_stream", stream)
        object.__setattr__(self, "_tp_on_event", on_event)

    def __getattr__(self, name):
        val = getattr(object.__getattribute__(self, "_tp_stream"), name)
        if name == "text_stream":
            # Serve text_stream from OUR event iteration (which marks TTFT on
            # every content_block_start/_delta, regardless of block type)
            # rather than tapping the SDK's pre-filtered text output. The SDK
            # filter yields ZERO chunks on tool-call-only turns (the events are
            # tool_use content_block_start + input_json_delta), so a tap on its
            # output never observed the first served token and those turns
            # logged no TTFT. The filter below matches the SDK's
            # __stream_text__ (only text_delta deltas carry `.text`), so the
            # customer-visible text is byte-identical. Fail-open: any setup
            # problem falls back to the previous output-tap, then to the raw
            # value.
            try:
                if hasattr(val, "__aiter__") and not (
                    hasattr(val, "__iter__") or hasattr(val, "__next__")
                ):
                    return _tp_text_stream_from_events_async(self)
                if hasattr(val, "__iter__") or hasattr(val, "__next__"):
                    return _tp_text_stream_from_events_sync(self)
            except Exception:
                pass
            try:
                return _tp_tap_iterable(val, object.__getattribute__(self, "_tp_on_event"))
            except Exception:
                return val
        return val

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_tp_stream"), name, value)

    def __iter__(self):
        stream = object.__getattribute__(self, "_tp_stream")
        on_event = object.__getattribute__(self, "_tp_on_event")
        for event in stream:
            try:
                on_event(event)
            except Exception:
                pass
            yield event

    def __aiter__(self):
        stream = object.__getattribute__(self, "_tp_stream")
        on_event = object.__getattribute__(self, "_tp_on_event")

        async def _agen():
            async for event in stream:
                try:
                    on_event(event)
                except Exception:
                    pass
                yield event

        return _agen()


def _tp_event_text(event):
    """The text chunk a MessageStream ``text_stream`` consumer would receive
    for ``event``, or None. Mirrors the anthropic SDK's ``__stream_text__``
    filter (only ``text_delta`` deltas carry ``.text``). Never raises."""
    try:
        if getattr(event, "type", "") == "content_block_delta":
            text = getattr(getattr(event, "delta", None), "text", None)
            if isinstance(text, str) and text:
                return text
    except Exception:
        pass
    return None


def _tp_text_stream_from_events_sync(proxy):
    """text_stream equivalent built over the proxy's EVENT iteration, so TTFT
    is marked on the first content event of ANY block type (tool_use included),
    not just on the first text chunk."""
    for event in proxy:
        text = _tp_event_text(event)
        if text is not None:
            yield text


async def _tp_text_stream_from_events_async(proxy):
    """Async twin of _tp_text_stream_from_events_sync."""
    async for event in proxy:
        text = _tp_event_text(event)
        if text is not None:
            yield text


def _tp_tap_iterable(it, on_event):
    """Wrap a sync OR async iterable so each yielded item is observed by
    ``on_event`` (fail-safe). Returns the iterable unchanged when its protocol
    can't be determined — never alters what the customer receives."""
    if hasattr(it, "__aiter__"):
        async def _atap():
            async for item in it:
                try:
                    on_event(item)
                except Exception:
                    pass
                yield item
        return _atap()
    if hasattr(it, "__iter__") or hasattr(it, "__next__"):
        def _tap():
            for item in it:
                try:
                    on_event(item)
                except Exception:
                    pass
                yield item
        return _tap()
    return it


# ═══════════════════════════════════════════════════════════════════
# Anthropic Message Batches — results-retrieval telemetry. Batch spend is
# otherwise invisible: the submit call carries no usage, and results never
# pass through the instrumented .create path. Wrapping .results() logs one
# llm row per succeeded entry with shape='anthropic_batch' + tier='batch'
# (the batch discount is applied server-side), yielding every entry verbatim
# to the customer.
# ═══════════════════════════════════════════════════════════════════

# In-process dedup: results pages are idempotent reads and may be re-read
# (retries, multiple consumers). Only the first read in this process logs a
# given (batch_id, custom_id). Cross-process re-reads produce rows with the
# SAME deterministic span_id, so they stay identifiable/dedupable downstream.
_BATCH_RESULTS_LOGGED = {}
_BATCH_RESULTS_LOGGED_MAX = 50000
# Guards the whole check-set-evict sequence below: concurrent .results()
# consumers across threads must not both log the same (batch_id, custom_id).
_BATCH_RESULTS_LOGGED_LOCK = threading.Lock()


def _batch_result_already_logged(key: str) -> bool:
    with _BATCH_RESULTS_LOGGED_LOCK:
        if key in _BATCH_RESULTS_LOGGED:
            return True
        _BATCH_RESULTS_LOGGED[key] = True
        if len(_BATCH_RESULTS_LOGGED) > _BATCH_RESULTS_LOGGED_MAX:
            _BATCH_RESULTS_LOGGED.pop(next(iter(_BATCH_RESULTS_LOGGED)))
        return False


@fail_safe
def _log_anthropic_batch_entry(entry, batch_id: str, api_base: str):
    """Log one Message Batch result entry. Fail-safe — never breaks iteration."""
    tp = get_client()
    if not tp:
        return
    result = getattr(entry, "result", None)
    if getattr(result, "type", "") != "succeeded":
        return  # errored / canceled / expired entries carry no usage
    custom_id = str(getattr(entry, "custom_id", "") or "")
    if _batch_result_already_logged(f"{batch_id}:{custom_id}"):
        return
    message = getattr(result, "message", None)
    usage = getattr(message, "usage", None)
    if usage is None:
        return
    model = str(getattr(message, "model", "") or "unknown")
    import hashlib as _hashlib
    import json as _json
    try:
        raw_usage = _json.loads(_json.dumps(
            _as_dict(usage),
            default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o),
        ))
    except Exception:
        raw_usage = None

    session = get_current_session()
    order = session.next_span_order()
    now_iso = datetime.now(timezone.utc).isoformat()
    span_obj = {
        **manual_span_ids(session),
        # Deterministic span id: a re-read from another process emits the SAME
        # id for the same batch entry, so duplicates are detectable downstream.
        "span_id": _hashlib.sha1(
            f"anthropic_batch:{batch_id}:{custom_id}".encode("utf-8")
        ).hexdigest()[:16],
        "span_kind": "llm",
        "span_name": f"batch:{custom_id or batch_id}",
        "span_order": order,
        "start_time": now_iso,
        "end_time": now_iso,
    }
    metadata = {
        "workflow_name": session.workflow_name,
        "batch_id": batch_id,
        "batch_custom_id": custom_id,
    }
    if session.session_id:
        metadata["session_id"] = session.session_id
    # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
    # session metadata WITHOUT it, then re-add it only when this row belongs
    # to the call that was actually rerouted (exact obs-key match).
    _copy_session_metadata(metadata, session)
    # ACCEPTED B4 RESIDUAL — batch reroute provenance is intentionally dropped
    # here. A reroute (if any) was applied on the batch CREATE call; this row is
    # emitted while RETRIEVING results, often in another process entirely (the
    # dedupe registry above is result-time only — nothing correlates create back
    # to retrieve, by design, since results are re-readable). So the key read
    # here belongs to the retrieval call and never matches: the row simply
    # carries no `_tp_routing`. Omission only — a batch row can never show a
    # stranger's reroute, which is the property that matters. Do not "fix" this
    # by falling back to an unkeyed peek.
    _stamp_routing_marker(metadata, session, _state.get_current_obs_key())

    model_extras = {"endpoint": "messages/batches/results"}
    if api_base:
        model_extras["api_base"] = api_base

    tp.log_sync(
        user_id=session.user_id,
        paid_plan=session.paid_plan,
        plan_source=getattr(session, "plan_source", None),
        workflow_name=session.workflow_name,
        session_id=session.session_id,
        model=model,
        provider="anthropic",
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        metadata=metadata,
        span=span_obj,
        usage={"shape": "anthropic_batch", "raw": raw_usage, "tier": "batch"},
        model_extras=model_extras,
    )


def _instrument_anthropic_batches():
    """Patch messages.batches.{Batches,AsyncBatches}.results — see the section
    comment above. Entries are yielded verbatim; logging rides alongside and is
    fail-open everywhere.
    """
    try:
        from anthropic.resources.messages.batches import Batches
    except ImportError:
        return

    if (Batches, "results") not in _originals:
        orig_results = getattr(Batches, "results", None)
        if orig_results is not None:
            _originals[(Batches, "results")] = orig_results

            @functools.wraps(orig_results)
            def results_wrapper(self, message_batch_id, **kwargs):
                api_base = _extract_base_url((self,))
                page = orig_results(self, message_batch_id, **kwargs)

                def _gen():
                    for entry in page:
                        try:
                            _log_anthropic_batch_entry(entry, str(message_batch_id), api_base)
                        except Exception:
                            pass  # fail-open: telemetry never breaks iteration
                        yield entry

                # Wrap in the Mode-A proxy so the customer keeps the page's
                # surface (`with`/`close()`/`.response`) — a bare generator drops
                # it and leaks the HTTP connection on abandonment. Entries yielded
                # verbatim through the metering generator; behavior otherwise
                # identical.
                return _ModeAStreamProxy(page, _gen())

            setattr(Batches, "results", results_wrapper)

    try:
        from anthropic.resources.messages.batches import AsyncBatches
    except ImportError:
        AsyncBatches = None
    if AsyncBatches is not None and (AsyncBatches, "results") not in _originals:
        orig_aresults = getattr(AsyncBatches, "results", None)
        if orig_aresults is not None:
            _originals[(AsyncBatches, "results")] = orig_aresults

            @functools.wraps(orig_aresults)
            async def aresults_wrapper(self, message_batch_id, **kwargs):
                api_base = _extract_base_url((self,))
                page = await orig_aresults(self, message_batch_id, **kwargs)

                async def _agen():
                    async for entry in page:
                        try:
                            _log_anthropic_batch_entry(entry, str(message_batch_id), api_base)
                        except Exception:
                            pass  # fail-open
                        yield entry

                # Async twin of the sync path: preserve the page's `async with` /
                # `aclose()` / `.response` surface instead of a bare async
                # generator. Entries yielded verbatim; behavior identical.
                return _ModeAAsyncStreamProxy(page, _agen())

            setattr(AsyncBatches, "results", aresults_wrapper)


def _instrument_anthropic_stream():
    """Patch `.stream` on every anthropic Messages surface that exposes it so
    Anthropic context-manager streaming yields full telemetry (see the section
    comment above _AnthropicStreamMgrWrapper for why): the non-beta classes
    (anthropic.resources.messages), their BETA siblings
    (anthropic.resources.beta.messages.messages — direct `SyncAPIResource`
    children, NOT subclasses of the non-beta classes, so wrapping non-beta
    alone left `client.beta.messages.stream` un-enforced and logging
    0/0-token rows), and the Bedrock beta twins
    (anthropic.lib.bedrock._beta_messages — no `stream` attribute today; the
    getattr guard skips them until anthropic adds one). One shared wrapper
    factory per protocol so every surface — beta included — inherits the
    suppression-window arming below identically; a beta-specific wrapper
    without the window would reproduce the double-row/leak bug the window
    fixes.
    """

    def _wrap_sync_stream(cls):
        orig_stream = getattr(cls, "stream", None)
        if orig_stream is None or (cls, "stream") in _originals:
            return
        if getattr(orig_stream, "_tp_preflight_wrapper", False):
            return  # already ours (e.g. aliased from an already-wrapped class)
        _originals[(cls, "stream")] = orig_stream

        @functools.wraps(orig_stream)
        def sync_stream_wrapper(self, *args, **kwargs):
            # Frameworks that log at their own level (LangChain / LiteLLM /
            # LlamaIndex / Pydantic AI) → pass through entirely; their wrapper
            # owns telemetry for this call.
            if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai():
                return orig_stream(self, *args, **kwargs)
            # Serving provider from the bound client's base_url (api.minimax.io ->
            # minimax), same axis the generic wrapper resolves. the pre-flight
            # used to run BARE — no kwargs, no provider — so the local evaluator
            # built its context with model=""/provider="" and treated EVERY
            # provider-targeted REROUTE as cross-provider (a blank-field
            # `reroute_rejected` on every streamed call, a same-provider reroute
            # that could never apply, and model-conditioned BLOCK rules that could
            # never match).
            _serving = _resolve_serving_provider("anthropic", (self,))
            # Inside Agno we still wrap so composition + token extraction run
            # per inner LLM call; we only skip the duplicate pre-flight check.
            if not in_agno():
                try:
                    # Runs BEFORE orig_stream builds the manager, so an applied
                    # reroute's in-place kwargs["model"] swap reaches the wire.
                    # Session is deliberately NOT threaded: the wrapper resolves the
                    # session it flushes later, in __enter__ — inside an open
                    # context both resolve the same object, outside one neither
                    # survives, so threading a construction-time session would only
                    # add divergence.
                    _run_sync_check(kwargs=kwargs, provider=_serving["provider"],
                                    serving_unverified=_serving["serving_unverified"])
                except TokenPoliceBlockedError:
                    raise
                except Exception:
                    pass  # fail-safe (matches @fail_safe behavior)
            # W1 suppression window: the OTel instrumentor's wrapped stream
            # method starts its `anthropic.chat` span INSIDE this call —
            # before our wrapper object even exists — which is why the old
            # session-wide one-shot flag (armed later, in __enter__) never
            # matched it: outside a workflow the span double-logged, inside
            # one the unconsumed flag leaked and ate the NEXT anthropic
            # call's telemetry. The per-call record armed here is claimed by
            # TokenPoliceSpanProcessor.on_start; threaded into the wrapper so
            # its manager-enter (W2) window can shape-detect `seen`.
            _w1 = arm_anthropic_stream_span_window()
            try:
                mgr = orig_stream(self, *args, **kwargs)
            finally:
                disarm_anthropic_stream_span_window(_w1)
            # `serving` threaded so an enter-failure row carries the same resolved
            # provider the pre-flight above used (see _emit_enter_failure).
            return _AnthropicStreamMgrWrapper(
                mgr, kwargs, _extract_base_url((self,)), serving=_serving,
                span_window=_w1[0] if _w1 is not None else None,
            )

        sync_stream_wrapper._tp_preflight_wrapper = True
        setattr(cls, "stream", sync_stream_wrapper)

    def _wrap_async_stream(cls):
        # AsyncMessages.stream — same context-manager pattern, just `async with`.
        orig_async = getattr(cls, "stream", None)
        if orig_async is None or (cls, "stream") in _originals:
            return
        if getattr(orig_async, "_tp_preflight_wrapper", False):
            return  # already ours (e.g. aliased from an already-wrapped class)
        _originals[(cls, "stream")] = orig_async

        @functools.wraps(orig_async)
        def async_stream_wrapper(self, *args, **kwargs):
            if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai():
                return orig_async(self, *args, **kwargs)
            # The pre-flight check runs inside the wrapper's __aenter__ (awaited
            # on the caller's event loop) — NOT here — so a blocking sync HTTP
            # round-trip never stalls the customer's loop. Covered by
            # tests/test_anthropic_async_stream_check.py.
            # Resolve serving identity here (sync + cheap) and hand the
            # wrapper a rebuild handle. The anthropic SDK freezes the request
            # body when stream() constructs the manager but fires HTTP only in
            # __aenter__, so an applied reroute (which mutates kwargs after
            # construction) needs the manager rebuilt from the mutated kwargs.
            _serving = _resolve_serving_provider("anthropic", (self,))
            # W1 suppression window — see sync_stream_wrapper above. `stream()`
            # is a plain sync method on the async client too (it returns the
            # manager without awaiting), so the window is await-free. The
            # rebuild lambda runs LATER, outside this window —
            # _rebuild_after_reroute arms its own record around it and folds
            # the claim count back into this one.
            _w1 = arm_anthropic_stream_span_window()
            try:
                mgr = orig_async(self, *args, **kwargs)
            finally:
                disarm_anthropic_stream_span_window(_w1)
            return _AnthropicAsyncStreamMgrWrapper(
                mgr, kwargs, _extract_base_url((self,)),
                rebuild=lambda: orig_async(self, *args, **kwargs),
                serving=_serving,
                span_window=_w1[0] if _w1 is not None else None,
            )

        async_stream_wrapper._tp_preflight_wrapper = True
        setattr(cls, "stream", async_stream_wrapper)

    # Anthropic not installed at all → nothing to do (matches _wrap_method's
    # ImportError contract). Beta / Bedrock-beta modules are guarded
    # per-module: either may be absent on older/newer anthropic versions.
    try:
        from anthropic.resources import messages as _msgs_mod
    except ImportError:
        return
    _stream_modules = [_msgs_mod]
    try:
        from anthropic.resources.beta.messages import messages as _beta_msgs_mod
        _stream_modules.append(_beta_msgs_mod)
    except ImportError:
        pass
    try:
        from anthropic.lib.bedrock import _beta_messages as _bedrock_beta_mod
        _stream_modules.append(_bedrock_beta_mod)
    except ImportError:
        pass

    for _mod in _stream_modules:
        _sync_cls = getattr(_mod, "Messages", None)
        if _sync_cls is not None:
            _wrap_sync_stream(_sync_cls)
        _async_cls = getattr(_mod, "AsyncMessages", None)
        if _async_cls is not None:
            _wrap_async_stream(_async_cls)


class _AnthropicAsyncStreamMgrWrapper(_AnthropicStreamMgrWrapper):
    """Async variant — overrides the context manager protocol to use async.

    The pre-flight budget/anomaly check runs here in `__aenter__`, awaited on
    the caller's event loop, so it never blocks the loop with a synchronous HTTP
    round-trip. A blocked call raises when the stream is entered — the provider
    request never fires, because the underlying manager is never entered.
    Covered by tests/test_anthropic_async_stream_check.py.
    """

    def __init__(self, mgr, kwargs, base_url="", rebuild=None, serving=None,
                 span_window=None):
        super().__init__(mgr, kwargs, base_url, span_window=span_window)
        # Zero-arg factory that rebuilds the underlying manager from the
        # (possibly reroute-mutated) kwargs; None when unavailable.
        self._rebuild = rebuild
        # {"provider", "serving_unverified"} resolved at stream() time.
        self._serving = serving or {}

    def _rebuild_after_reroute(self, model_before):
        """Swap in a manager built from the reroute-mutated kwargs.

        No-op unless the pre-flight actually changed ``kwargs["model"]`` and a
        rebuild handle exists. Total and fail-open: a failed rebuild keeps the
        original manager (so the customer's `async with` still works) and drops
        the applied-reroute claim — ``_tp_routing`` and a ``rerouted``
        ``_local_decision`` — because the request that fires will carry the
        ORIGINAL model and must never be audited as rerouted.
        """
        rebuild = getattr(self, "_rebuild", None)
        try:
            model_after = (self._kwargs or {}).get("model")
        except Exception:
            return
        if rebuild is None or model_after == model_before:
            return
        # W1-rebuild suppression window: `rebuild()` re-invokes the wrapped
        # `stream()`, so the instrumentor can start a SECOND stream span here
        # exactly like it did during the original W1 call. Its own fresh
        # record (never the possibly-exhausted W1 record), with the claim
        # count folded back into W1 afterwards so __aenter__'s W2 decision
        # (`seen == 0`) still reflects whether ANY span fired before
        # manager-enter. Fully fail-open at every step.
        _wr = None
        try:
            _wr = arm_anthropic_stream_span_window()
        except Exception:
            _wr = None
        try:
            new_mgr = rebuild()
        except Exception:
            new_mgr = None
        finally:
            disarm_anthropic_stream_span_window(_wr)
            try:
                if (_wr is not None and isinstance(self._span_window, dict)
                        and int(_wr[0].get("seen", 0)) > 0):
                    self._span_window["seen"] = (
                        int(self._span_window.get("seen", 0)) + int(_wr[0].get("seen", 0))
                    )
            except Exception:
                pass  # fold-back is best-effort, like every window step
        if new_mgr is not None:
            old_mgr, self._mgr = self._mgr, new_mgr
            # The discarded manager holds an un-awaited request coroutine
            # (anthropic builds it in stream(), awaits it in __aenter__).
            # Close it or the customer's log gets a "coroutine was never
            # awaited" RuntimeWarning. Nothing is sent by closing it.
            try:
                for _v in list(vars(old_mgr).values()):
                    if _inspect.iscoroutine(_v):
                        _v.close()
            except Exception:
                pass
            return
        try:
            sess = get_current_session()
            md = getattr(sess, "metadata", None)
            if isinstance(md, dict) and "_tp_routing" in md:
                md = dict(md)
                md.pop("_tp_routing", None)
                sess.metadata = md
            # Same withdrawal, keyed store side: the rows this call emits must
            # not display reroute provenance either. The session-metadata pop
            # above only covers copies taken from here on; the per-call record
            # is what the row side actually stamps from, so it has to go too.
            _drop_routing_marker(sess, getattr(self, "_obs_key", None))
            # Drop THIS call's rerouted decision only (keyed to the obs key
            # re-captured in __aenter__ right after our own check): a
            # concurrent sibling's applied reroute must still reach its own
            # /log row.
            _drop_local_decision(
                sess,
                getattr(self, "_obs_key", None),
                predicate=lambda ld: (isinstance(ld, dict)
                                      and ld.get("outcome") == "rerouted"),
            )
        except Exception:
            pass

    async def __aenter__(self):
        # Pre-flight check on entry (skipped inside an Agno-instrumented call —
        # the outer wrapper owns the single check per run). A denial raises
        # TokenPoliceBlockedError out of the customer's `async with`; the
        # underlying manager below is never entered, so no provider request
        # fires. Any other failure fails open.
        if not in_agno():
            # The check used to run BARE, so the local evaluator saw
            # model=""/provider="" — every provider-targeted REROUTE looked
            # cross-provider and model-conditioned BLOCK rules could not match.
            try:
                model_before = (self._kwargs or {}).get("model")
            except Exception:
                model_before = None
            try:
                await _run_async_check(
                    kwargs=self._kwargs,
                    provider=self._serving.get("provider") or "anthropic",
                    serving_unverified=bool(self._serving.get("serving_unverified")),
                )
            except TokenPoliceBlockedError:
                raise
            except Exception:
                pass  # fail-open
            # Re-capture the obs key: the async path's check runs HERE (not at
            # construction, where __init__ captured a possibly-stale value).
            try:
                self._obs_key = _state.get_current_obs_key()
            except Exception:
                pass  # keep the construction-time value
            # An applied reroute mutated kwargs["model"] in place, but this
            # manager was built from the PRE-swap body (the SDK freezes it at
            # stream() time and only fires HTTP on entry). Rebuild from the
            # mutated kwargs so the swap reaches the wire; the old manager was
            # never entered, so no request was made against it.
            try:
                self._rebuild_after_reroute(model_before)
            except Exception:
                pass  # fail-open: the original manager is still usable
        try:
            # The HTTP request fires inside the manager's __aenter__ — anchor
            # TTFT/total latency AND span start here (request start). See sync
            # __enter__: both clocks must share the pre-handshake anchor
            # so span duration and ttft_ms measure the same window.
            self._req_start_mono = _time.monotonic()
            self._start_time = datetime.now(timezone.utc)
        except Exception:
            pass
        # See sync __enter__: Python skips __aexit__ when __aenter__ raises, so
        # a request-time rejection inside the vendor enter would otherwise
        # yield zero rows. The guard wraps ONLY this one statement — the
        # pre-flight ABOVE can legitimately raise TokenPoliceBlockedError, and
        # keeping the guard this narrow (plus the explicit re-raise arm) means
        # a denial can never be reclassified as a provider failure.
        #
        # W2 suppression window — see sync __enter__ for the full rationale:
        # armed only when the W1 record saw no span inside the wrapped
        # `stream()` call (runtime shape-detection of a future instrumentor
        # that starts its span at manager-enter; never version-detection),
        # and only after the pre-flight, so a blocked call arms nothing. On
        # the single await inside the window: ContextVars are copied at Task
        # CREATION, so a task created BEFORE the arm holds a pre-arm snapshot
        # and can neither see nor claim the record — that is what keeps the
        # realistic concurrent shapes safe (verified: gather over calls,
        # threads, nesting). A task spawned INSIDE the window is different:
        # its copied Context references the same mutable record, so it CAN
        # consume the budget. That degradation is deliberate and bounded in
        # the safe direction — the parent's real span then lands
        # unsuppressed, i.e. an extra row, never a lost one (matching this
        # fix's ordering: raise into customer code never > silent loss
        # never > extra row acceptable). Note this window replaces the old
        # session-wide one-shot flag (armed here pre-fix) — see sync
        # __enter__ for why the flag was dead on this seam and leaked.
        _w2 = None
        try:
            if isinstance(self._span_window, dict) and self._span_window.get("seen") == 0:
                _w2 = arm_anthropic_stream_span_window(self._span_window)
        except Exception:
            _w2 = None  # fail-open: unarmed → worst case an extra row
        try:
            self._stream = await self._mgr.__aenter__()
        except TokenPoliceBlockedError:
            raise  # defense-in-depth: never turn a denial into a failure row
        except Exception as exc:
            try:
                self._emit_enter_failure(exc)
            except Exception:
                pass  # fail-safe: telemetry must never mask the provider error
            raise
        finally:
            disarm_anthropic_stream_span_window(_w2)
        try:
            self._session = get_current_session()
            self._order = self._session.next_span_order()
            self._span_name = consume_pending_span_name()
            if not self._span_name and in_agno():
                self._span_name = f"agent_step_{self._order + 1}"
            comp = build_prompt_composition("anthropic", self._kwargs)
            if comp:
                if not hasattr(self._session, '_pending_compositions'):
                    self._session._pending_compositions = {}
                self._session._pending_compositions.setdefault(
                    f"{self._session.trace_id}:{self._order}", {}
                )["prompt"] = comp
            self._prev_defer = getattr(self._session, '_defer_telemetry', False)
            self._session._defer_telemetry = True
            # Snapshot ids already queued (see sync __enter__) so exit drops
            # only this call's own entries, never a concurrent call's spans.
            existing = getattr(self._session, '_deferred_spans', None) or []
            self._enter_deferred_ids = {id(p) for p in existing}
        except Exception:
            pass
        return self._wrap_stream_for_ttft(self._stream)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self._afinalize()
        except Exception:
            pass
        try:
            return await self._mgr.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            # See sync __exit__ above for why defer reset + dud drop have to
            # happen after the inner __aexit__ runs, and why the drop is scoped
            # to this call's own entries. A blocked-at-enter call never reached
            # the snapshot (_enter_deferred_ids stays None) → we drop nothing.
            try:
                if self._session is not None:
                    self._session._defer_telemetry = self._prev_defer
                    if self._enter_deferred_ids is not None:
                        _drop_deferred_spans(
                            self._session,
                            keep_ids=self._enter_deferred_ids,
                            order=self._order,
                            trace_id=getattr(self._session, 'trace_id', None),
                        )
            except Exception:
                pass

    async def _afinalize(self):
        """Async variant — `AsyncMessageStream.get_final_message()` is async,
        so we can't share the sync `_finalize` body."""
        session = self._session
        if session is None:
            return
        tp = get_client()
        if tp is None or self._stream is None:
            return
        try:
            final_msg = await self._stream.get_final_message()
        except Exception:
            return
        if final_msg is None:
            return

        # This context-manager path bypasses .create entirely and logs by
        # hand, so neither pending-id producer ever ran and every streamed tool
        # row lost its id. The final message's content[] carries the tool_use
        # blocks with ids. Stashed before anything below can fail; REPLACE on
        # every capture (even empty) so a no-tool streamed turn clears stale ids,
        # matching the non-streamed path.
        try:
            set_pending_tool_calls(extract_pending_tool_calls("anthropic", final_msg))
        except Exception:
            pass

        usage = getattr(final_msg, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0) if usage else 0
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage else 0
        # See _finalize (sync) for the full rationale: forward the verbatim
        # anthropic usage shape so cache writes price at the write rate; gated
        # fallback to today's behavior when serialization fails (empty raw would
        # map to all-zeros server-side).
        raw_usage = _serialize_anthropic_raw_usage(usage)
        if raw_usage:
            cached_tokens = cache_read  # reads only; writes priced via raw
            usage_block = {"shape": "anthropic_messages", "raw": raw_usage}
        else:
            cached_tokens = cache_read + cache_creation
            usage_block = None
        model = str(getattr(final_msg, "model", "") or self._kwargs.get("model") or "unknown")
        # Family-gated collapse of the provider's dated echo to the requested
        # (post-reroute) alias so one logical model stays one cost-by-model row.
        model = _prefer_requested_model((self._kwargs or {}).get("model"), model)
        response_comp = build_response_composition("anthropic", final_msg) or []

        prompt_comp = []
        comp_key = f"{session.trace_id}:{self._order}"
        if hasattr(session, '_pending_compositions'):
            comp_data = session._pending_compositions.pop(comp_key, {})
            prompt_comp = comp_data.get("prompt", []) or prompt_comp

        span_obj = {
            **manual_span_ids(session),
            "span_kind": "llm",
            "span_name": self._span_name or model,
            "span_order": self._order,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }
        metadata = {"workflow_name": session.workflow_name}
        if session.session_id:
            metadata["session_id"] = session.session_id
        # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
        # session metadata WITHOUT it, then re-add it only when this row belongs
        # to the call that was actually rerouted (exact obs-key match).
        _copy_session_metadata(metadata, session)
        _stamp_routing_marker(metadata, session, getattr(self, "_obs_key", None))

        # Drain applied local_decision + shadow/reject observations so the
        # anthropic .stream() manual log confirms REQUEST_REROUTED / REROUTE_REJECTED
        # on /log (parity with _log_manual / _flush_deferred_spans). Keyed on the
        # obs key re-captured in __aenter__ right after this call's check (the
        # stream is drained later, when the contextvar may hold another call's
        # key). NOTE: unlike the
        # Node SDK (whose anthropic .stream() delegates to create({stream:true}) and
        # therefore runs TWO pre-flights, needing _pushObservationOnce dedup), the
        # Python anthropic SDK's stream() posts directly and runs exactly ONE
        # pre-flight per call — a plain drain cannot double-emit; do not add a dedup latch.
        pending_local_decision = None
        pending_observations = None
        try:
            _stream_obs_key = getattr(self, "_obs_key", None)
            pending_local_decision = _claim_local_decision(session, _stream_obs_key)
            pending_observations = _state.drain_observations(_stream_obs_key)
        except Exception:
            pending_local_decision = None
            pending_observations = None

        tp.log_sync(
            user_id=session.user_id,
            paid_plan=session.paid_plan,
            plan_source=getattr(session, "plan_source", None),
            workflow_name=session.workflow_name,
            session_id=session.session_id,
            model=model,
            provider="anthropic",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            # None in the fallback branch → client.log_sync treats it exactly
            # like an omitted usage= (it gates on `if usage:`), preserving today.
            usage=usage_block,
            metadata=metadata,
            span=span_obj,
            prompt_composition=prompt_comp,
            response_composition=response_comp,
            # Serving endpoint (e.g. api.minimax.io) so the service remaps the
            # provider from anthropic to the actual serving provider (minimax).
            model_extras={"api_base": self._base_url} if self._base_url else None,
            # Streamed call → TTFT/total anchored at the manager __aenter__
            # (request start). _build_stream_latency returns None on anomaly.
            latency=_build_stream_latency(self._req_start_mono, self._ttft_mono,
                                          _time.monotonic()),
            local_decision=pending_local_decision,
            observations=pending_observations or None,
        )


def protect(module_path: str, class_name: str, method_name: str, is_async: bool,
            *, manual: bool = False, provider: str = None, target_object=None):
    """
    Public API to manually apply the pre-flight check to a specific module/class/method.
    Useful for protecting custom internal SDKs or unlisted providers (e.g. raw httpx
    calls to LLM provider REST endpoints).

    ``manual=True`` routes the wrapper through the manual-telemetry path —
    same path used for SDKs without an OpenLLMetry instrumentor (Cerebras,
    HuggingFace, native OpenRouter, …). The wrapper extracts token usage from
    the response itself and logs synchronously, instead of relying on an
    OTel span. Required for any wrapper that bypasses a real provider SDK.

    ``provider`` lets the caller override the substring-based provider
    detection (``_detect_provider``) which yields "" for arbitrary module
    paths like ``__main__``. Composition parsing keys off this string —
    pass e.g. "openai", "openai_responses", "anthropic", "google", "cohere",
    "together", "cerebras", "huggingface", "mistral", "litellm".

    ``target_object`` lets the caller pass a live class/container object
    instead of resolving ``module_path`` by import — parity with the Node SDK's
    ``options.module``. Use it to protect an in-app class that has no importable
    dotted path (defined in ``__main__``, a closure, or created dynamically):
    pass the class directly with ``class_name=""``, or a container object with
    a truthy ``class_name`` naming the attribute on it. ``module_path`` then
    drives provider detection/diagnostics only. When ``None`` (default) the
    behaviour is unchanged and the class is resolved by import.
    """
    # Capture the override BEFORE the local `target` dict is built below —
    # naming the public param `target` would collide with (and be clobbered by)
    # that reassignment, silently dropping the override. Hence `target_object`.
    override = target_object
    target = {
        "module": module_path,
        "object": class_name,
        "method": method_name,
        "async": is_async,
    }
    if manual:
        target["manual"] = True
    # Per-target isolation + fail-open: a poisoned/frozen target (a throwing
    # descriptor, a frozen-attr setattr, or a module whose import-time body
    # raises) must degrade to a no-op with a warning — never throw into the
    # customer's setup code. Mirrors auto_instrument()'s per-target guard.
    try:
        _wrap_method(target, override_module=override)
    except Exception as e:
        logger.warning(
            f"TokenPolice: protect() failed to instrument "
            f"{module_path}.{class_name}.{method_name}: {e}"
        )
        return

    # When the caller supplied an explicit provider, replace the wrapper
    # _wrap_method installed (which used the detected provider — likely "")
    # with one bound to the caller's provider. Only meaningful on the manual
    # path; the non-manual path derives provider from the OTel gen_ai.system
    # attribute anyway.
    if manual and provider:
        try:
            # Honour the same live object so we resolve the identical
            # class `_wrap_method` used (a re-import of a __main__/in-app class
            # would miss the `_originals` key). 3-arg getattr for consistency.
            if override is not None:
                cls = getattr(override, class_name, None) if class_name else override
            else:
                module = importlib.import_module(module_path)
                cls = getattr(module, class_name, None) if class_name else module
            original = _originals.get((cls, method_name))
            if original is not None:
                # Keep module_path so a protect("litellm", ...) registration is
                # still recognized as the LiteLLM seam after the provider
                # override re-wraps it.
                _set_manual_wrapper(cls, method_name, original, provider, is_async,
                                    module_path=module_path)
        except Exception as e:
            logger.debug(f"TokenPolice: provider override failed for {module_path}.{class_name}.{method_name}: {e}")


def _instrument_langchain_embeddings():
    """Patch langchain_core.embeddings.Embeddings.{embed_documents, embed_query,
    aembed_documents, aembed_query} with a framework wrapper that runs a SINGLE
    pre-flight check, holds the _in_langchain guard so the inner provider
    wrapper (e.g. openai.embeddings.create) stays inert, and manually logs
    the call with operation="embedding".

    No OpenLLMetry instrumentor exists for LangChain embeddings, so without
    this we'd either skip logging (with the guard set) or double-check the
    budget (without the guard). The framework wrapper picks "single check +
    single log" — the customer-facing semantic.
    """
    try:
        import langchain_core.embeddings as _lc_emb_mod
        BaseCls = getattr(_lc_emb_mod, "Embeddings", None)
        if BaseCls is None:
            return

        provider = "langchain"

        # Concrete-subclass instrumentation table. Third tuple element is the
        # canonical provider key used for price resolution — it must match the
        # service's canonical provider slugs exactly or the underlying model's
        # price will not resolve. Single source of truth used by both
        # _patch_cls (compile-time) and _derive_lc_provider (runtime fallback
        # for the base-class patch).
        _LC_EMBEDDING_PROVIDERS = [
            ("langchain_openai",       "OpenAIEmbeddings",             "openai"),
            ("langchain_cohere",       "CohereEmbeddings",             "cohere"),
            ("langchain_mistralai",    "MistralAIEmbeddings",          "mistral"),
            ("langchain_google_genai", "GoogleGenerativeAIEmbeddings", "gemini"),
            ("langchain_huggingface",  "HuggingFaceEmbeddings",        "huggingface"),
            ("langchain_voyageai",     "VoyageAIEmbeddings",           "voyage"),
            ("langchain_together",     "TogetherEmbeddings",           "together_ai"),
            ("langchain_aws",          "BedrockEmbeddings",            "bedrock"),
        ]
        _LC_MODULE_TO_PROVIDER = {m: p for (m, _c, p) in _LC_EMBEDDING_PROVIDERS}

        def _derive_lc_provider(self_obj):
            try:
                mod = getattr(type(self_obj), "__module__", "") or ""
            except Exception:
                return None
            for prefix, canonical in _LC_MODULE_TO_PROVIDER.items():
                if mod == prefix or mod.startswith(prefix + "."):
                    return canonical
            return None

        @fail_safe
        def _log_embedding_call(model_hint, args, kwargs, span_name, start_time,
                                original_provider=None, obs_key=_OBS_KEY_CURRENT):
            tp = get_client()
            if tp is None:
                return
            session = get_current_session()
            # Reach for the bound subclass (self) to pull the configured model
            # name. Falls back to "langchain_embedding".
            model = model_hint
            try:
                self_obj = args[0] if args else None
                if self_obj is not None:
                    for attr in ("model", "model_name"):
                        v = getattr(self_obj, attr, None)
                        if isinstance(v, str) and v:
                            # This re-read deliberately overrides the caller's
                            # model_hint, so the Google "models/" strip the
                            # wrapper already applied has to be reapplied here
                            # too — otherwise the SUCCESS row (the one carrying
                            # tokens and cost) lands in a different
                            # generations.model bucket than the /check hint and
                            # the failure row. See _lc_google_bare_model.
                            model = _lc_google_bare_model(self_obj, v)
                            break
            except Exception:
                pass
            order = session.next_span_order()
            span_obj = {
                **manual_span_ids(session),
                "span_kind": "llm",
                "span_name": span_name or model,
                "span_order": order,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": datetime.now(timezone.utc).isoformat(),
            }
            metadata = {"workflow_name": session.workflow_name, "framework": "langchain"}
            if session.session_id:
                metadata["session_id"] = session.session_id
            # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
            # session metadata WITHOUT it, then re-add it only when this row belongs
            # to the call that was actually rerouted (exact obs-key match).
            _copy_session_metadata(metadata, session)
            # Resolved once — the SAME threaded key drives the stamp and the
            # drain below (callers pass the key captured after their check).
            _row_obs_key = _resolve_obs_key(obs_key)
            _stamp_routing_marker(metadata, session, _row_obs_key)
            # Resolve the underlying provider so the cost engine can route the
            # price lookup. Concrete-subclass patches pass it via the closure;
            # the base-class patch path resolves at call time from self's
            # module path so e.g. a custom langchain_openai.OpenAIEmbeddings
            # subclass that doesn't override embed_* still attributes to openai.
            resolved_op = original_provider
            # Kept in its own guard: provider attribution is strictly optional,
            # so it must never be able to cost the row its resolved model name
            # (mirrors the LlamaIndex twin's guard).
            try:
                if not resolved_op and args:
                    resolved_op = _derive_lc_provider(args[0])
            except Exception:
                resolved_op = original_provider
            # Composition: LangChain Embeddings methods take texts (list[str])
            # or text (str) — surface both shapes to the embedding parser.
            comp_kwargs = {}
            if len(args) >= 2:
                comp_kwargs["input"] = args[1]
            if "texts" in kwargs:
                comp_kwargs["texts"] = kwargs["texts"]
            if "text" in kwargs:
                comp_kwargs["input"] = kwargs["text"]
            try:
                prompt_comp = build_prompt_composition(provider, comp_kwargs, operation="embedding")
            except Exception:
                prompt_comp = []
            # We don't have provider-side usage here (the inner call was
            # suppressed by the guard). Approximate input tokens via the
            # composition's length sum / 4 — same rule-of-thumb as the HF
            # approximation. The row's cost is marked as approximated
            # server-side.
            approx_tokens = 0
            try:
                for entry in prompt_comp:
                    approx_tokens += max(1, (entry.get("length") or 0) // 4)
            except Exception:
                approx_tokens = 0
            # Keyed observation drain (observations only — a body-less seam
            # never owns a keyed local_decision; a claim could only steal a
            # sibling's via the untagged fallback). Own try: fail-open.
            _pending_obs = None
            try:
                _pending_obs = _state.drain_observations(_row_obs_key)
            except Exception:
                _pending_obs = None
            tp.log_sync(
                user_id=session.user_id,
                paid_plan=session.paid_plan,
                plan_source=getattr(session, "plan_source", None),
                workflow_name=session.workflow_name,
                session_id=session.session_id,
                model=model,
                provider="langchain",
                input_tokens=approx_tokens,
                output_tokens=0,
                cached_tokens=0,
                metadata=metadata,
                span=span_obj,
                prompt_composition=prompt_comp,
                response_composition=[],
                usage={
                    "shape": "openai_embeddings",
                    "raw": {"approx_input_tokens": approx_tokens, "approximated": True},
                },
                model_extras=(
                    {"framework": "langchain", "original_provider": resolved_op}
                    if resolved_op else
                    {"framework": "langchain"}
                ),
                operation="embedding",
                observations=_pending_obs or None,
            )

        def _make_sync_wrapper(original, method_name, original_provider=None):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                if in_langchain() or in_litellm() or in_pydantic_ai() or in_agno():
                    return original(*args, **kwargs)
                # Resolved BEFORE the check so the pre-flight carries the
                # real model + underlying vendor instead of a bare "langchain".
                # `method_name` stays a LOGGING-only last resort — a method name
                # is not a model, so it never enters the check context. Vendor
                # falls back to the framework slug when the module is unknown.
                # Prefer model, then model_name (parity with success _log_embedding_call).
                _self = args[0] if args else None
                # GoogleGenerativeAIEmbeddings prefixes its own .model with
                # "models/" (see _lc_google_bare_model); strip it at the source
                # so the check hint and the failure-row model carry the bare id.
                # The SUCCESS row re-reads the instance in _log_embedding_call
                # and is stripped there, so all three agree on one bucket.
                _model_attr = _lc_google_bare_model(
                    _self, _instance_model_attr(_self, "model", "model_name"))
                model_hint = _model_attr or method_name
                _run_sync_check(provider=(_derive_lc_provider(_self) or provider),
                                model_hint=_model_attr)
                # This call's obs key, captured right after the check.
                _obs_key = _state.get_current_obs_key()
                span_name = consume_pending_span_name()
                start_time = datetime.now(timezone.utc)
                session = get_current_session()
                # Model lives on `self`, not kwargs — synthetic kwargs for stash.
                _stash_attempt_context(
                    session, provider, "", args, {"model": model_hint},
                    operation="embedding",
                )
                tid = _lc_trace_id(session)
                _call_start = _time.monotonic()
                _in_langchain.set(True)
                enter_langchain_trace(tid)
                try:
                    result = original(*args, **kwargs)
                except Exception as _exc:
                    # Success log sits after the call; without this path a
                    # provider failure emits zero embedding rows. Re-raise the
                    # original exception unchanged (golden rule); emit itself is
                    # @fail_safe so a logging failure cannot mask it.
                    elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                    try:
                        session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                    except Exception:
                        pass
                    _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                    raise
                finally:
                    leave_langchain_trace(tid)
                    _in_langchain.set(False)
                _log_embedding_call(model_hint, args, kwargs, span_name, start_time,
                                    original_provider, obs_key=_obs_key)
                return result
            return wrapper

        def _make_async_wrapper(original, method_name, original_provider=None):
            @functools.wraps(original)
            async def wrapper(*args, **kwargs):
                if in_langchain() or in_litellm() or in_pydantic_ai() or in_agno():
                    return await original(*args, **kwargs)
                # See the sync twin above, including the Google "models/" strip.
                _self = args[0] if args else None
                _model_attr = _lc_google_bare_model(
                    _self, _instance_model_attr(_self, "model", "model_name"))
                model_hint = _model_attr or method_name
                await _run_async_check(provider=(_derive_lc_provider(_self) or provider),
                                       model_hint=_model_attr)
                # This call's obs key, captured right after the check.
                _obs_key = _state.get_current_obs_key()
                span_name = consume_pending_span_name()
                start_time = datetime.now(timezone.utc)
                session = get_current_session()
                _stash_attempt_context(
                    session, provider, "", args, {"model": model_hint},
                    operation="embedding",
                )
                tid = _lc_trace_id(session)
                _call_start = _time.monotonic()
                _in_langchain.set(True)
                enter_langchain_trace(tid)
                try:
                    result = await original(*args, **kwargs)
                except Exception as _exc:
                    # See sync twin above.
                    elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                    try:
                        session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                    except Exception:
                        pass
                    _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                    raise
                finally:
                    leave_langchain_trace(tid)
                    _in_langchain.set(False)
                _log_embedding_call(model_hint, args, kwargs, span_name, start_time,
                                    original_provider, obs_key=_obs_key)
                return result
            return wrapper

        embed_methods = [
            ("embed_documents", False),
            ("embed_query", False),
            ("aembed_documents", True),
            ("aembed_query", True),
        ]

        def _patch_cls(target_cls, original_provider=None):
            """Patch a single Embeddings-shaped class. Idempotent — keyed on
            (class, method) so repeated calls (e.g. auto_instrument re-entry)
            skip already-wrapped methods.

            ``original_provider`` is the canonical provider key of the
            underlying provider (e.g. "openai" for OpenAIEmbeddings). The
            wrapper threads it into model_extras so server-side price
            resolution can resolve the underlying model's price.
            None for the base-class patch — derived at call time instead."""
            for method_name, is_async in embed_methods:
                # Only walk class.__dict__ — `getattr` returns inherited
                # methods which we DON'T want to patch on the subclass (that
                # would double-wrap if the base is also patched). We patch
                # the subclass only when it has its OWN method definition.
                if method_name not in target_cls.__dict__ and target_cls is not BaseCls:
                    continue
                original = target_cls.__dict__.get(method_name) if target_cls is not BaseCls else getattr(BaseCls, method_name, None)
                if not callable(original):
                    continue
                key = (target_cls, method_name)
                if key in _originals:
                    continue
                _originals[key] = original
                wrapped = (
                    _make_async_wrapper(original, method_name, original_provider)
                    if is_async else
                    _make_sync_wrapper(original, method_name, original_provider)
                )
                try:
                    setattr(target_cls, method_name, wrapped)
                except Exception:
                    continue

        # 1. Patch the base class — covers any subclass that DOESN'T override.
        # No original_provider known at patch time; the wrapper derives it
        # from self.__class__.__module__ at call time.
        _patch_cls(BaseCls)

        # 2. Patch concrete provider subclasses directly. Python MRO finds
        # subclass overrides FIRST, so when a subclass redefines
        # embed_documents (every modern LC provider does), the base-class
        # patch is shadowed and never runs. Mirror the strategy used by
        # _instrument_llama_index_embeddings which has the same hazard.
        for module_name, class_name, canonical_provider in _LC_EMBEDDING_PROVIDERS:
            try:
                provider_module = importlib.import_module(module_name)
            except ImportError:
                continue
            ConcreteCls = getattr(provider_module, class_name, None)
            if ConcreteCls is None:
                continue
            try:
                _patch_cls(ConcreteCls, canonical_provider)
            except Exception as e:
                logger.debug(
                    f"TokenPolice: LangChain {class_name} instrumentation skipped: {e}"
                )
    except ImportError:
        return
    except Exception as e:
        logger.debug(f"TokenPolice: LangChain embeddings instrumentation failed: {e}")


def _instrument_llama_index_embeddings():
    """Patch llama_index.core.embeddings.BaseEmbedding.{get_text_embedding,
    get_query_embedding, get_text_embedding_batch, aget_text_embedding,
    aget_query_embedding} with a framework wrapper that runs a SINGLE
    pre-flight check and holds the _in_llamaindex guard so the inner
    provider wrapper (e.g. openai.embeddings.create) stays inert.

    Same rationale as _instrument_langchain_embeddings: no OpenLLMetry
    LlamaIndex Python instrumentor exists, so the framework wrapper
    handles manual logging.
    """
    try:
        # In modern LlamaIndex the base lives at llama_index.core.embeddings.
        try:
            import llama_index.core.embeddings as _li_emb_mod
        except ImportError:
            return
        BaseCls = getattr(_li_emb_mod, "BaseEmbedding", None)
        if BaseCls is None:
            return

        provider = "llamaindex"

        # Underlying-vendor table, mirroring _LC_EMBEDDING_PROVIDERS in
        # _instrument_langchain_embeddings. Values are copied VERBATIM from that
        # table so the two frameworks attribute the same vendor to the same slug
        # (the server canonicalizes the aliases — `gemini` → google,
        # `together_ai` → together — in provider-identity.js). A wrong slug here
        # silently breaks server-side price resolution, so do not invent slugs.
        # Unlike LangChain there is no concrete-subclass patch (only BaseEmbedding
        # is patched), so this is resolved at call time from self's module path.
        _LI_EMBEDDING_MODULE_TO_PROVIDER = {
            "llama_index.embeddings.openai":      "openai",
            "llama_index.embeddings.cohere":      "cohere",
            "llama_index.embeddings.mistralai":   "mistral",
            "llama_index.embeddings.google_genai": "gemini",
            "llama_index.embeddings.gemini":      "gemini",
            "llama_index.embeddings.huggingface": "huggingface",
            "llama_index.embeddings.voyageai":    "voyage",
            "llama_index.embeddings.together":    "together_ai",
            "llama_index.embeddings.bedrock":     "bedrock",
        }

        def _derive_li_provider(self_obj):
            try:
                mod = getattr(type(self_obj), "__module__", "") or ""
            except Exception:
                return None
            for prefix, canonical in _LI_EMBEDDING_MODULE_TO_PROVIDER.items():
                if mod == prefix or mod.startswith(prefix + "."):
                    return canonical
            return None

        @fail_safe
        def _log_embedding_call(model_hint, args, kwargs, span_name, start_time,
                                obs_key=_OBS_KEY_CURRENT):
            tp = get_client()
            if tp is None:
                return
            session = get_current_session()
            model = model_hint
            resolved_op = None
            try:
                self_obj = args[0] if args else None
                if self_obj is not None:
                    for attr in ("model_name", "model"):
                        v = getattr(self_obj, attr, None)
                        if isinstance(v, str) and v:
                            model = v
                            break
            except Exception:
                pass
            # Kept in its own guard: provider attribution is strictly optional,
            # so it must never be able to cost the row its resolved model name
            # (the LangChain sibling resolves it after the model loop too).
            try:
                if args and args[0] is not None:
                    resolved_op = _derive_li_provider(args[0])
            except Exception:
                resolved_op = None
            order = session.next_span_order()
            span_obj = {
                **manual_span_ids(session),
                "span_kind": "llm",
                "span_name": span_name or model,
                "span_order": order,
                "start_time": start_time.isoformat() if start_time else None,
                "end_time": datetime.now(timezone.utc).isoformat(),
            }
            metadata = {"workflow_name": session.workflow_name, "framework": "llamaindex"}
            if session.session_id:
                metadata["session_id"] = session.session_id
            # B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
            # session metadata WITHOUT it, then re-add it only when this row belongs
            # to the call that was actually rerouted (exact obs-key match).
            _copy_session_metadata(metadata, session)
            # Resolved once — the SAME threaded key drives the stamp and the
            # drain below (callers pass the key captured after their check).
            _row_obs_key = _resolve_obs_key(obs_key)
            _stamp_routing_marker(metadata, session, _row_obs_key)
            # Composition: LlamaIndex embedding methods take a str (text/query)
            # or list[str] (batch). args[1] is the payload.
            comp_kwargs = {}
            if len(args) >= 2:
                comp_kwargs["input"] = args[1]
            try:
                prompt_comp = build_prompt_composition(provider, comp_kwargs, operation="embedding")
            except Exception:
                prompt_comp = []
            approx_tokens = 0
            try:
                for entry in prompt_comp:
                    approx_tokens += max(1, (entry.get("length") or 0) // 4)
            except Exception:
                approx_tokens = 0
            # Keyed observation drain (observations only — a body-less seam
            # never owns a keyed local_decision; a claim could only steal a
            # sibling's via the untagged fallback). Own try: fail-open.
            _pending_obs = None
            try:
                _pending_obs = _state.drain_observations(_row_obs_key)
            except Exception:
                _pending_obs = None
            tp.log_sync(
                user_id=session.user_id,
                paid_plan=session.paid_plan,
                plan_source=getattr(session, "plan_source", None),
                workflow_name=session.workflow_name,
                session_id=session.session_id,
                model=model,
                provider="llamaindex",
                input_tokens=approx_tokens,
                output_tokens=0,
                cached_tokens=0,
                metadata=metadata,
                span=span_obj,
                prompt_composition=prompt_comp,
                response_composition=[],
                usage={
                    "shape": "openai_embeddings",
                    "raw": {"approx_input_tokens": approx_tokens, "approximated": True},
                },
                model_extras=(
                    {"framework": "llamaindex", "original_provider": resolved_op}
                    if resolved_op else
                    {"framework": "llamaindex"}
                ),
                operation="embedding",
                observations=_pending_obs or None,
            )

        def _make_sync_wrapper(original, method_name):
            @functools.wraps(original)
            def wrapper(*args, **kwargs):
                # llama-index 0.14+ decorates these embedding methods with
                # a strict wrapt dispatcher (`@dispatcher.span`) whose inner
                # `inspect.signature(func).bind(*args, **kwargs)` rejects the call
                # unless `self` arrives via the descriptor protocol. Calling
                # `original(self, ...)` directly raises TypeError into customer
                # code. `_li_call_original` re-binds via `original.__get__(self)`
                # (a no-op on older lax llama-index), preserving fail-open.
                if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai() or in_agno():
                    return _li_call_original(original, args, kwargs)
                # Resolved BEFORE the check so the pre-flight carries the
                # real model + underlying vendor instead of a bare "llamaindex".
                # `method_name` stays a LOGGING-only last resort — a method name
                # is not a model, so it never enters the check context. Vendor
                # falls back to the framework slug when the module is unknown.
                # Prefer model_name, then model (parity with success _log_embedding_call).
                _self = args[0] if args else None
                _model_attr = _instance_model_attr(_self, "model_name", "model")
                model_hint = _model_attr or method_name
                _run_sync_check(provider=(_derive_li_provider(_self) or provider),
                                model_hint=_model_attr)
                # This call's obs key, captured right after the check.
                _obs_key = _state.get_current_obs_key()
                span_name = consume_pending_span_name()
                start_time = datetime.now(timezone.utc)
                session = get_current_session()
                # Model lives on `self` (model_name), not kwargs — synthetic stash.
                _stash_attempt_context(
                    session, provider, "", args, {"model": model_hint},
                    operation="embedding",
                )
                _call_start = _time.monotonic()
                _in_llamaindex.set(True)
                try:
                    result = _li_call_original(original, args, kwargs)
                except Exception as _exc:
                    # Framework embedding failure must still emit a failed
                    # embedding row (operation=embedding). Re-raise unchanged.
                    elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                    try:
                        session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                    except Exception:
                        pass
                    _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                    raise
                finally:
                    _in_llamaindex.set(False)
                _log_embedding_call(model_hint, args, kwargs, span_name, start_time,
                                    obs_key=_obs_key)
                return result
            return wrapper

        def _make_async_wrapper(original, method_name):
            @functools.wraps(original)
            async def wrapper(*args, **kwargs):
                # See _make_sync_wrapper above — async twin via
                # `_li_acall_original` descriptor re-bind.
                if in_langchain() or in_litellm() or in_llamaindex() or in_pydantic_ai() or in_agno():
                    return await _li_acall_original(original, args, kwargs)
                # See the sync twin above.
                _self = args[0] if args else None
                _model_attr = _instance_model_attr(_self, "model_name", "model")
                model_hint = _model_attr or method_name
                await _run_async_check(provider=(_derive_li_provider(_self) or provider),
                                       model_hint=_model_attr)
                # This call's obs key, captured right after the check.
                _obs_key = _state.get_current_obs_key()
                span_name = consume_pending_span_name()
                start_time = datetime.now(timezone.utc)
                session = get_current_session()
                _stash_attempt_context(
                    session, provider, "", args, {"model": model_hint},
                    operation="embedding",
                )
                _call_start = _time.monotonic()
                _in_llamaindex.set(True)
                try:
                    result = await _li_acall_original(original, args, kwargs)
                except Exception as _exc:
                    # See sync twin above.
                    elapsed_ms = int((_time.monotonic() - _call_start) * 1000)
                    try:
                        session._call_outcome = build_call_outcome(_exc, elapsed_ms)
                    except Exception:
                        pass
                    _emit_call_failure_log(get_client(), session, obs_key=_obs_key)
                    raise
                finally:
                    _in_llamaindex.set(False)
                _log_embedding_call(model_hint, args, kwargs, span_name, start_time,
                                    obs_key=_obs_key)
                return result
            return wrapper

        for method_name, is_async in [
            ("get_text_embedding", False),
            ("get_query_embedding", False),
            ("get_text_embedding_batch", False),
            ("aget_text_embedding", True),
            ("aget_query_embedding", True),
        ]:
            original = getattr(BaseCls, method_name, None)
            if not callable(original):
                continue
            key = (BaseCls, method_name)
            if key in _originals:
                continue
            _originals[key] = original
            wrapped = _make_async_wrapper(original, method_name) if is_async else _make_sync_wrapper(original, method_name)
            try:
                setattr(BaseCls, method_name, wrapped)
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"TokenPolice: LlamaIndex embeddings instrumentation failed: {e}")


def auto_instrument():
    """
    Applies the pre-flight enforcement hook to all known, installed SDKs.
    Call this once at application startup.
    """
    global _is_instrumented
    if _is_instrumented:
        return

    for target in _TARGET_METHODS:
        # PER-TARGET ISOLATION: _wrap_method already swallows ImportError (SDK
        # not installed → silent skip). Any OTHER exception from a module's
        # import-time body / a throwing descriptor / a frozen-attr setattr must
        # NOT escape init() and must NOT abort the loop — log-and-skip this one
        # target so one bad SDK doesn't skip instrumenting all the others.
        try:
            _wrap_method(target)
        except Exception as e:
            logger.debug(f"TokenPolice: instrumentation failed for a target: {e}")
            continue

    # Anthropic Messages.stream is a context-manager method (not in the
    # registry-walked _TARGET_METHODS) — patched separately so its custom
    # exit semantics can extract usage + composition from final_message.
    try:
        _instrument_anthropic_stream()
    except Exception as e:
        logger.debug(f"TokenPolice: anthropic stream instrumentation failed: {e}")

    # Message Batches results retrieval (A5) — one llm row per succeeded entry
    # with tier='batch'; batch spend is otherwise invisible to telemetry.
    try:
        _instrument_anthropic_batches()
    except Exception as e:
        logger.debug(f"TokenPolice: anthropic batches instrumentation failed: {e}")

    # pydantic_ai's concrete Model subclasses are patched via a dedicated
    # function because the registry-walked _TARGET_METHODS path is for
    # registry-friendly module+class+method targets and pydantic_ai needs
    # uniform _in_pydantic_ai guarding plus async-context-manager handling for
    # request_stream. Telemetry is manual — no OpenLLMetry pydantic_ai
    # instrumentor exists.
    try:
        _instrument_pydantic_ai()
    except Exception as e:
        logger.debug(f"TokenPolice: pydantic_ai instrumentation failed: {e}")

    # Agno (github.com/agno-agi/agno). Agent.run / arun call the underlying
    # provider SDK internally; the wrappers run the SINGLE pre-flight check
    # and hold the _in_agno guard so the nested provider wrapper passes
    # straight through. Telemetry is manual — no OpenLLMetry agno
    # instrumentor exists.
    try:
        _instrument_agno()
    except Exception as e:
        logger.debug(f"TokenPolice: agno instrumentation failed: {e}")

    # LangChain + LlamaIndex embedding wrappers — patched on the abstract
    # base class so every concrete subclass (OpenAIEmbeddings,
    # CohereEmbeddings, ...) inherits the wrapped methods. No OpenLLMetry
    # instrumentor exists for embeddings on either framework, so each
    # wrapper handles manual logging with operation="embedding".
    try:
        _instrument_langchain_embeddings()
    except Exception as e:
        logger.debug(f"TokenPolice: LangChain embeddings instrumentation failed: {e}")
    try:
        _instrument_llama_index_embeddings()
    except Exception as e:
        logger.debug(f"TokenPolice: LlamaIndex embeddings instrumentation failed: {e}")

    _is_instrumented = True
    logger.debug("TokenPolice: Pre-flight enforcement hooks applied.")


def uninstrument():
    """
    Restores the original SDK methods, removing TokenPolice pre-flight enforcement.
    Useful for testing or dynamic reconfiguration.
    """
    global _is_instrumented
    if not _is_instrumented:
        return

    for (target_obj, method_name), original in _originals.items():
        try:
            setattr(target_obj, method_name, original)
        except Exception as e:
            logger.debug(f"Failed to uninstrument {method_name} on {target_obj}: {e}")

    _originals.clear()
    _is_instrumented = False

    try:
        from .telemetry import unsetup_opentelemetry
        unsetup_opentelemetry()
    except Exception as e:
        logger.debug(f"Failed to unsetup opentelemetry: {e}")
    logger.debug("TokenPolice: Pre-flight enforcement hooks removed.")
