# TokenPolice — Python SDK

TokenPolice blocks expensive and runaway LLM requests **before** they reach the provider, using
the budget and anomaly rules you set in the dashboard. This SDK is the piece that runs in your app.

- **One install, working capture.** `pip install token-police` captures **OpenAI**, **Anthropic**,
  **AWS Bedrock**, **Google Gemini**, **Cohere** and more out of the box — no extra packages to
  wire up.
- **Fails open, always.** If TokenPolice is unreachable or anything goes wrong inside the SDK, your
  LLM call proceeds normally. The firewall never becomes a hard dependency of your request path.
- **Private by design.** Token counts and session metadata are extracted in your own process. Only
  that metadata is sent to TokenPolice — never prompt or completion text.

Docs: [tokenpolice.ai/docs](https://tokenpolice.ai/docs) · Dashboard: [app.tokenpolice.ai](https://app.tokenpolice.ai)

Two lines to start. Dry-run is the default: nothing is blocked until you create a rule and switch
to `enforce`.

```bash
pip install token-police
```

```python
import token_police as tp
tp.init()  # reads TOKENPOLICE_API_KEY, from app.tokenpolice.ai → API Keys
```

## Installation

Requires Python 3.10 or newer.

```bash
pip install token-police
```

That single package captures tokens for **OpenAI**, **Anthropic**, **AWS Bedrock**, **Google
Gemini** and **Cohere** (chat + embeddings). Mistral, Together.ai, xAI and OpenRouter are covered by
the same install — see the [support matrix](#provider--framework-support).

TokenPolice instruments the provider SDKs **your app already installs**. It never pulls or pins
those versions. If a provider SDK is missing or on a version the SDK doesn't fully support, token
capture for it is skipped and **your app keeps working**.

### Framework extras

Framework integrations are opt-in extras:

```bash
pip install "token-police[langchain]"     # LangChain
pip install "token-police[all]"           # every bundled framework integration
```

Extras: `langchain`, `llamaindex`, `pydantic-ai`, `openai-agents`, `xai`, `voyageai`, `agno`, and
`all` (all of the above).

**CrewAI is opt-in and deliberately not part of `all`** — install it on its own line:

```bash
pip install "token-police[crewai]"
```

CrewAI supports Python 3.10 to 3.13; it does not yet support 3.14. Keeping it out of `all` is what
lets `pip install "token-police[all]"` resolve on 3.14 — the `[crewai]` extra is the one install
that fails there, with CrewAI's own resolver error.

### OpenTelemetry version conflicts

TokenPolice depends on the current OpenTelemetry core with lower bounds only and never caps it. A
few provider SDKs bundle their own OpenTelemetry and cap it. That does not break TokenPolice: pip
resolves the whole environment down to satisfy the cap, or prints a resolver warning and installs
anyway.

| Provider/framework SDK | Problem | What to do |
|---|---|---|
| `mistralai` 2.0 – 2.9.1 (and 1.10.0 / 1.11.1) | caps `opentelemetry-semantic-conventions`, so pip installs an older OpenTelemetry train for the whole environment — a **tolerated** conflict: it installs and TokenPolice works normally | nothing required. Upgrade to **`mistralai>=2.9.2`** (or `1.12.x`), which dropped the cap, if you want the current OpenTelemetry train. Both `mistralai` majors are supported — never pin `<2` |
| `crewai` (all releases) | hard-pins `opentelemetry-api/sdk ~=1.34`, below what TokenPolice asks for — a **tolerated** conflict: pip prints a resolver warning but it works at runtime | no fix needed for normal `pip install`. Optionally set **`CREWAI_DISABLE_TELEMETRY=true`** to stop CrewAI's own telemetry and silence its `Overriding of current TracerProvider is not allowed` log line. A **strict resolver** (uv / poetry / pip-tools) may refuse to co-resolve — allow the conflict or install in two steps |

> Don't reach for `OTEL_SDK_DISABLED=true` to quiet these — it disables OpenTelemetry globally,
> including TokenPolice's own capture. Use the SDK-specific opt-out (e.g. `CREWAI_DISABLE_TELEMETRY`)
> instead.

## Get an API key

Create a key in the dashboard at [app.tokenpolice.ai](https://app.tokenpolice.ai) → **API Keys**.
It looks like `tp_sk_…` and is shown **once**. Put it in the `TOKENPOLICE_API_KEY` environment
variable (recommended) or pass it to `init()`.

## Add it with your coding agent (recommended)

The fastest way to integrate is to let your coding agent do it. Install the TokenPolice skill
once — it works with Claude Code, Cursor, GitHub Copilot, Antigravity and Codex; the install
steps for each are at [Install the skill](https://tokenpolice.ai/docs/get-started/coding-agent/install).
Then ask the agent in plain language. You write no integration code yourself.

In your project:

> Integrate TokenPolice into this app.

The agent reads your code to find your LLM calls and where your user, plan and session values
live; asks you for your API key and confirms those fields; shows you the full plan and changes
nothing until you approve; wires the SDK in `dry_run` mode (nothing blocked yet); and after you
run the app once, checks that TokenPolice received the data. The edits are the same few lines
shown in the manual quick start below.
[Walkthrough](https://tokenpolice.ai/docs/get-started/coding-agent/walkthrough).

## Quick start (manual)

### 1. Initialize once, at startup

```python
import token_police as tp
import openai

tp.init(
    # api_key="tp_sk_...",   # or set TOKENPOLICE_API_KEY
    firewall="enforce",      # 'dry_run' (default) evaluates rules without acting; 'enforce' acts on them; 'off' records usage only
)

client = openai.Client()

# Checked against your rules before it runs; token usage is recorded afterward.
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello world!"}],
)
```

Call `tp.init()` before your first LLM request. The SDK finds the LLM libraries you have installed
and starts capturing their calls.

`enforce` acts on the rules you create in the dashboard — with no rules, nothing is blocked. Start
in `dry_run` to see what *would* happen, then switch to `enforce`.

### 2. Handle a blocked call

In `enforce` mode a blocked call raises `TokenPoliceBlockedError` instead of reaching the provider.
It is the **only** exception the SDK ever raises into your code.

```python
try:
    client.chat.completions.create(...)
except tp.TokenPoliceBlockedError as e:
    print(e.reason, e.rule_id, e.kind, e.trace_id)  # all optional
    return {"error": "Usage limit reached. Please try again later."}
```

### 3. Group by workflow & user

Attribute and budget usage per user, plan, feature or workflow.

```python
# Recommended for request handlers: session()  → an "agent" span by default
with tp.session(name="customer_support", user_id="user_123", paid_plan="pro", metadata={"tenant": "acme"}):
    response = client.chat.completions.create(...)

# Recommended for reusable functions: @workflow  → a "chain" span by default
@tp.workflow(name="rag_pipeline")
def run_pipeline(user_id: str, query: str):
    # user_id is auto-extracted from the function arguments.
    return client.chat.completions.create(...)

run_pipeline(user_id="user_123", query="How do I reset my password?")
```

`user_id`, `paid_plan`, `name` and `metadata` are what rules match on and what the dashboard groups
by. See [Identity](https://tokenpolice.ai/docs/concepts/identity).

#### Agent vs chain spans

The grouping span is one of two kinds:

- **`agent`** — a *dynamic, LLM-driven* loop (the model decides each next step).
  `tp.session(...)` and `tp.agent(...)` emit this.
- **`chain`** — a *static, developer-defined* sequence or glue code (step A → B → C),
  or a root entry point. `tp.chain(...)` and `tp.workflow(...)` emit this.

```python
with tp.chain(name="rag_pipeline", user_id="u1"):
    docs = retrieve(q)                              # static step 1
    response = client.chat.completions.create(...)  # static step 2

with tp.agent(name="research_agent", user_id="u1"):
    ...  # an autonomous loop that calls tools and re-plans
```

Pass `kind="agent"` / `kind="chain"` to `session()` / `workflow()` to override the default.
Grouping spans carry no cost of their own.

#### Thread a conversation with `session_id`

A trace is one run (one turn). Pass the **same** `session_id` across turns to group them into a
conversation, visible under **Conversations** in the dashboard:

```python
# Each turn is a separate request → separate session() call, same session_id.
with tp.session(name="support", user_id="u1", session_id=conversation_id):
    response = client.chat.completions.create(...)
```

Omit it and each run gets its own id. `@workflow` can also pick it up from a `session_id` function
argument at runtime.

### 4. Track tool calls

Capture tool/function executions as tool spans under the active session — useful for tool calls
TokenPolice can't see on its own (CrewAI, plain functions, MCP):

```python
# Decorate a function:
@tp.tool()
def web_search(query: str) -> str:
    return run_search(query)

# Or wrap a single execution inline:
with tp.tool_span("web_search", args=query) as h:
    result = run_search(query)
    h["result"] = result  # optional — recorded as a hash + length, never raw text
```

### 5. Serverless (AWS Lambda, Vercel, …)

Decorate your handler so pending telemetry is sent before the container freezes:

```python
@tp.serverless
def handler(event, context):
    with tp.session(name="lambda_handler", user_id=event["userId"]):
        response = client.chat.completions.create(...)
        return {"statusCode": 200, "body": response.choices[0].message.content}
```

### 6. Name a span

```python
tp.set_span_name("nightly_summarizer")  # applies to the next LLM span in this context
```

### 7. Manual check / log & protecting unsupported SDKs

If you call an LLM through an SDK TokenPolice doesn't recognize, drive enforcement yourself or
register the method for automatic enforcement:

```python
client = tp.get_client()

# Manual pre-flight check (sync)
result = client.check_sync("user_123", "pro", "my_workflow")
if result["status"] == "blocked":
    raise SystemExit  # budget exceeded

# ... make your LLM call ...

# Manual log: (user_id, paid_plan, workflow_name, session_id, model, provider, input_tokens, output_tokens)
client.log_sync("user_123", "pro", "my_workflow", "", "gpt-4o", "openai", 100, 50)

# Or register a method so TokenPolice enforces + logs it automatically:
tp.protect("my_llm_sdk.client", "Client", "generate", is_async=True)
```

### Fail-open guarantee

TokenPolice is designed to **never crash your application**. If the service is unreachable or
anything fails inside the SDK, the SDK fails open and lets your LLM call proceed. The only exception
is a deliberate block in `enforce` mode, which raises `TokenPoliceBlockedError`.
See [Fail-open](https://tokenpolice.ai/docs/concepts/fail-open).

### A note on the capture warning

If token capture for a provider you use can't be enabled, TokenPolice logs **once**:

```
TokenPolice: token capture for openai is disabled — its instrumentor failed to
load; reinstall token-police to enable it. Budgets are still enforced.
```

Budgets keep being enforced — only token *capture* for that provider is affected. You'll never see
this warning for a provider you don't use. To raise SDK log visibility, pass `log_errors=True` to
`tp.init()`.

## Provider & framework support

| Provider / framework | Coverage |
|---|---|
| OpenAI | Base install |
| Anthropic | Base install |
| Cohere (v2, chat + embeddings) | Base install |
| AWS Bedrock | Base install |
| Google Gemini (`google-genai` 1.x and 2.x) | Base install |
| Mistral (`mistralai` 1.x and 2.x — chat, FIM, embeddings, OCR, transcription; speech on 2.x). On 2.x the client import is `from mistralai.client import Mistral` | Base install |
| Together.ai | Base install |
| OpenRouter | Base install |
| xAI (`xai-sdk`) | `[xai]` extra |
| Voyage AI | `[voyageai]` extra |
| LangChain | `[langchain]` extra |
| LlamaIndex | `[llamaindex]` extra |
| PydanticAI | `[pydantic-ai]` extra |
| OpenAI Agents | `[openai-agents]` extra |
| Agno | `[agno]` extra |
| CrewAI | `[crewai]` extra (not in `all`) |

Full matrix and tested versions: [Integrations](https://tokenpolice.ai/docs/integrations/matrix) ·
[Supported versions](https://tokenpolice.ai/docs/sdk/supported-versions).

## Configuration

### `tp.init(...)`

| Argument | Type | Default | Description |
|---|---|---|---|
| `api_key` | `str` | `TOKENPOLICE_API_KEY` | Your TokenPolice API key (`tp_sk_...`) |
| `base_url` | `str` | `TOKENPOLICE_BASE_URL` or `https://collect.tokenpolice.ai` | TokenPolice endpoint. Leave unset unless told otherwise |
| `timeout` | `float` | `2.0` | Max seconds for a call to TokenPolice; on timeout the LLM call proceeds |
| `firewall` | `'enforce' \| 'dry_run' \| 'off'` | `'dry_run'` | `'enforce'` acts on your rules; `'dry_run'` evaluates them and records what would have happened without acting; `'off'` records usage only |
| `enforce` | `bool` | — | **Deprecated** alias for `firewall` (`True` → `'enforce'`, `False` → `'off'`) |
| `log_errors` | `bool` | `False` | Log SDK errors at WARNING level |
| `max_workers` | `int` | `None` | Background thread-pool size for telemetry sends. Defaults to `min(32, cpu_count + 4)` |
| `deployment` | `str` | `"auto"` | Process shape hint (`auto` / `daemon` / `serverless` / `edge`); auto-detected |
| `capture_stream_usage` | `bool` | `None` (on) | Ask the provider for the usage frame on streamed calls so they report token counts (`TP_CAPTURE_STREAM_USAGE=0` disables) |
| `error_detail` | `str` | `"redacted"` | `"none"` / `"redacted"` / `"raw"` — how much of a failed provider call's error text leaves the process. `raw` is opt-in only |
| `sse_reconnect_max_interval_seconds` | `int` | `300` | Cap on the backoff between reconnect attempts of the live rule feed |
| `stream_stale_grace_seconds` | `float` | `60` | Clamped to 0–3600. How long the cached rule view is trusted after the live feed drops before a matching call is re-checked with the server |
| `tracer_provider` | OTel `TracerProvider` | `None` | Attach TokenPolice to an OpenTelemetry provider you already run instead of its own |

### Environment variables

| Variable | Purpose |
|---|---|
| `TOKENPOLICE_API_KEY` | API key, used when `api_key` is not passed |
| `TOKENPOLICE_BASE_URL` | Endpoint override, used when `base_url` is not passed |
| `TP_CAPTURE_STREAM_USAGE` | Set to `0` to stop requesting usage frames on streamed calls |
| `CREWAI_DISABLE_TELEMETRY` | CrewAI's own switch; see the conflicts table above |

### Exports

| Export | Signature | Purpose |
|---|---|---|
| `init` | `init(api_key=None, ...)` | Initialize the SDK. |
| `session` | `session(name=None, user_id=None, paid_plan=None, metadata=None, session_id="", kind="agent")` | Context manager for a session (`agent` span). |
| `agent` | `agent(...)` | Same as `session` — an explicit `agent` span. |
| `chain` | `chain(...)` | Context manager for a `chain` span. |
| `workflow` | `@workflow(name, user_id, paid_plan, ...)` | Decorator that groups a function under one session (auto-extracts `user_id`). |
| `serverless` | `@serverless` | Decorator that auto-flushes telemetry on return. |
| `tool` | `@tool(name=None, tool_type="function")` | Decorator emitting a tool span per call. |
| `tool_span` | `tool_span(name, ...)` | Context manager emitting a tool span around a block. |
| `set_span_name` | `set_span_name(name)` | Name the next LLM span in the current context. |
| `get_current_session` | `get_current_session()` | The active `TPSession`, if any. |
| `TPSession` | class | The session object returned by `get_current_session()`. |
| `protect` | `protect(module_path, class_name, method_name, is_async, ...)` | Register a custom SDK method for enforcement + logging. |
| `uninstrument` | `uninstrument()` | Remove all enforcement hooks. |
| `flush` | `await flush()` | Async-wait for all pending logs. |
| `flush_sync` | `flush_sync()` | Blocking drain — joins the background logging threads. Use it before a script exits. (Python has no `shutdown()`; a normal process exit is covered automatically.) |
| `get_client` | `get_client()` | Get the underlying client for manual `check_sync`/`log_sync`. |
| `TokenPoliceBlockedError` | — | Raised when an enforced call is blocked (`reason`, `rule_id`, `kind`, `trace_id`). |

## What leaves your process

Prompt text and completion content never leave your process. TokenPolice receives token counts,
model and provider names, and the session metadata you attach (`user_id`, `paid_plan`, `name`,
`session_id`, `metadata`). Error text from a failed provider call is redacted by default
(`error_detail`). See [Data privacy](https://tokenpolice.ai/docs/concepts/data-privacy) and the
[privacy policy](https://tokenpolice.ai/privacy).

## Requirements

- Python >= 3.10

## Support

- Docs: [tokenpolice.ai/docs](https://tokenpolice.ai/docs)
- Issues: [github.com/tokenpolice/token-police-python/issues](https://github.com/tokenpolice/token-police-python/issues)
- Email: [support@tokenpolice.ai](mailto:support@tokenpolice.ai) · Security: [security@tokenpolice.ai](mailto:security@tokenpolice.ai)

## License

Apache-2.0
