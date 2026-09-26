# Changelog

All notable changes to `token-police` (Python) are documented here.

## 1.0.1 — 2026-09-26

### Fixed

- `pip install "token-police[all]"` failed with `resolution-too-deep` once
  OpenTelemetry 1.45 shipped. The `all` and `pydantic-ai` extras now floor
  `pydantic-ai-slim` rather than the `pydantic-ai` umbrella, which also pulled in
  logfire (capped below OpenTelemetry 1.45) and `openai>=3.19` (clashing with
  `llama-index-llms-openai`'s `openai<3`). Installs that already have the
  umbrella are unaffected. If you relied on `token-police[pydantic-ai]` to
  install PydanticAI's model providers for you, install `pydantic-ai` yourself.

## 1.0.0 — 2026-09

Initial public release.
