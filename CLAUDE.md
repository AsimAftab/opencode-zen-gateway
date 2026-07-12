# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project uses `uv` (not pip). There is no `requirements.txt`; dependencies live in `pyproject.toml`.

```bash
uv sync                                  # install (dev group included)
uv run python main.py                    # run server -> http://localhost:8000
uv run python main.py --port 9000        # CLI args > env vars > defaults

uv run pytest -v                                                # all tests (~1400)
uv run pytest tests/unit/test_streaming_core.py -v              # single file
uv run pytest tests/unit/test_cache.py::TestModelInfoCache -v   # single class
uv run pytest "tests/unit/test_routes_anthropic.py::TestVerifyAnthropicApiKey::test_valid_x_api_key_returns_true" -v
uv run pytest -k "streaming and not retry" -v                   # by keyword
uv run pytest -s                                                # show the print() narration tests rely on
uv run pytest --cov=opencode_zen --cov-report=term              # coverage

docker compose up -d                     # requires a .env file to exist, or compose fails
```

There is **no linter, formatter, or type checker** configured (no ruff/black/mypy). CI (`.github/workflows/docker.yml`, on push/PR to `main`) runs pytest + coverage, then a Docker build, a Trivy scan, and three inline `docker run` smoke checks. Note the Dockerfile uses `uv sync --frozen`, so a stale `uv.lock` breaks the image build.

## Architecture

A FastAPI proxy that presents **two client-facing API dialects** and forwards to one upstream. Both dialects converge on the same upstream call:

- `/v1/chat/completions` (OpenAI) — `routes_openai.py`, auth via `Authorization: Bearer $PROXY_API_KEY`
- `/v1/messages`, `/v1/messages/count_tokens` (Anthropic) — `routes_anthropic.py`, auth via `x-api-key` **or** `Bearer` (count_tokens is answered locally via `tokenizer.py`; the upstream has no token-counting API)
- `/v1/models`, `/health`, `/` — `routes_openai.py`

**The upstream (OpenCode Zen, `OPENCODE_BASE_URL`, default `https://opencode.ai/zen/v1`) is OpenAI-shaped.** Both routes POST to `{BASE}/chat/completions`. This asymmetry drives the whole design: the OpenAI path is nearly a passthrough, while the Anthropic path must translate in both directions.

**The upstream uses DASH-form Claude model IDs** (`claude-sonnet-4-5`, `claude-haiku-4-5` — verified against the live `/v1/models`). `model_resolver.normalize_model_name` canonicalizes client variants (date suffixes like `-20251001`, dotted `claude-haiku-4.5`, `[1m]` annotations, legacy inverted `claude-3-7-sonnet`) into that dash form. Do not reintroduce dot-form output — it makes every Claude model 400 upstream.

Request flow, both paths:

1. Route rewrites inbound messages for truncation recovery; the Anthropic route also strips server-side tools (e.g. Claude Code's `web_search_20250305`) which the upstream can't execute; `WEB_SEARCH_ENABLED` (opt-in, default false) injects a `web_search` function tool.
2. A format-specific adapter parses the wire format into the canonical internal types — `UnifiedMessage` / `UnifiedTool` / `ThinkingConfig` (dataclasses in `converters_core.py`). Entry points: `converters_openai.build_opencode_payload`, `converters_anthropic.anthropic_to_opencode`. Tool calls travel in **nested OpenAI shape** (`{"id", "type", "function": {...}}`); the core builder accepts nested and flat.
3. Both delegate to the single builder `converters_core.build_opencode_payload`, which emits the OpenAI-shaped upstream payload. It hardcodes `"stream": True` — **the upstream is always streamed, regardless of the client's `stream` flag.**
4. `http_client.OpenCodeHttpClient.request_with_retry` sends it (injects the upstream `OPENCODE_API_KEY`, retries 429/5xx with exponential backoff, and consults `network_errors.classify_network_error` to decide whether a transport error is even retryable).
5. Streaming response: the OpenAI path re-yields upstream bytes verbatim (`aiter_raw` — do not re-frame lines; SSE parsers need the blank-line separators). The Anthropic path runs `streaming_anthropic.stream_openai_to_anthropic`, the only real translator, turning OpenAI deltas into the Anthropic SSE event sequence (`message_start` → `ping` → `content_block_start/delta/stop` → `message_delta` → `message_stop`). Upstream `reasoning_content`/`reasoning` deltas become Anthropic `thinking` blocks. The translator guarantees termination events even if the upstream dies mid-stream.
6. Non-streaming response: since the upstream always streams, `sse_aggregator.aggregate_openai_sse` collects the SSE into one result, rendered as `chat.completion` (OpenAI) or via `collect_anthropic_response` (Anthropic). Upstream error bodies (Anthropic-shaped: `{"type":"error","error":{...}}`) are passed through/re-shaped so clients see the real upstream message.

The `_core` vs `_openai`/`_anthropic` split is the organizing principle: **format-agnostic logic lives in the `_core` module; the format-specific modules are thin adapters.** CONTRIBUTING.md makes this a hard rule — a change must be applied to *both* APIs *and* to both streaming and non-streaming modes. When you touch one adapter, check its twin.

`OpenCodeHttpClient` wraps either the pooled `app.state.http_client` (created in the `main.py` lifespan) or, deliberately, a throwaway per-request client when `stream=True`; the `stream_wrapper` `finally` block closes it. Check `_owns_client` before closing.

### This repo is a fork of a "Kiro" (AWS CodeWhisperer) gateway

Commit `0200cae` migrated it to OpenCode Zen. **A large legacy layer survives, is still fully unit-tested, and is not reachable from any route.** Do not assume a module is live because it exists and has tests. Verify with `grep` that a route actually reaches it before modifying it.

Currently dormant: `streaming_core.py` and `streaming_openai.py` (imported by no route — the live OpenAI path does no conversion at all), `parsers.AwsEventStreamParser`, `mcp_tools.handle_native_web_search` (no route calls it anymore; it still targets the Kiro-era MCP endpoint), `ModelResolver` + `ModelInfoCache` (imported in `main.py` but never instantiated; `/v1/models` proxies upstream directly), `MODEL_ALIASES` / `HIDDEN_FROM_LIST` / `FALLBACK_MODELS`, `payload_guards`, and `opencode_zen_errors.enhance_opencode_zen_error`. Of the model-config dicts, only `HIDDEN_MODELS` is live (via `get_model_id_for_kiro` in both adapters — note the legacy name).

A consequence worth knowing: two README features are **half-wired**. Truncation recovery's *inject* half (`get_tool_truncation` / `get_content_truncation`) is live in both routes, but its *detect-and-persist* half (`save_*_truncation`) sits in the dead `streaming_openai.py`, so nothing ever writes the state the routes read. Likewise `ThinkingParser`/`inject_thinking_tags` (fake reasoning *injection*) are never called by the live path — though upstream models that natively emit `reasoning_content` DO surface as thinking blocks via `streaming_anthropic`/`sse_aggregator`. Treat README's feature list as intent, not as current behavior.

### Subsystems whose purpose isn't obvious from the filename

- **`truncation_state.py` + `truncation_recovery.py`** — The upstream silently truncates large tool-call arguments, and the model then blames itself and retries the identical oversized call, looping forever. The fix spans a *request boundary*: detect truncation during a stream, persist it in a process-global map keyed by something the client will echo back (the `tool_call_id`, or a sha256 of the assistant content), then on the *next* request splice a synthetic `[API Limitation]` / `[System Notice]` message into the history so the model knows it was the API, not itself. The generated wording deliberately avoids saying "break it into steps" — that induces pathological micro-stepping.
- **`thinking_parser.py`** — The upstream has no native extended-thinking API. "Fake reasoning" injects `<thinking>` instructions into the prompt and parses the tags back out of the response with a 3-state FSM (the tag can straddle network chunks, so the parser always retains a tail buffer). The thinking budget is capped because fake reasoning burns *output* tokens — uncapped, the model spends its whole allowance thinking and has none left to answer.
- **`tokenizer.py`** — Anthropic publishes no tokenizer, so this uses tiktoken `cl100k_base` times an empirical 1.15 correction. The correction is applied **once at the end of an aggregate**, never per-sub-call (inner calls pass `apply_claude_correction=False`), or it compounds.
- **`converters_core.process_tools_with_long_descriptions`** — The upstream 400s on long tool descriptions, so over-limit descriptions are swapped for a pointer and the full text is appended to the system prompt.
- **Errors** are three unrelated modules, not a class hierarchy: `exceptions.py` (Pydantic 422 handler), `network_errors.py` (httpx transport errors → retryability + suggested status), `opencode_zen_errors.py` (upstream error *bodies* → plain English).
- **`debug_middleware.py`** runs *before* Pydantic validation specifically so it can capture the raw body of requests that will 422. `DEBUG_MODE=all` wipes and rewrites `debug_logs/` on every request; `errors` buffers in memory and only flushes on failure. The debug logger is a global singleton, so concurrent requests clobber each other's buffers.

## Conventions

- **Config is import-time.** `config.py` is flat module-level constants read via `os.getenv` once at import — settings are frozen at process start. Tests that change env must `importlib.reload` the config module (see `tests/unit/test_config.py`).
- Type hints on all functions; Google-style docstrings; `loguru` at decision points; English only in code, comments, and names.
- **Tests are mandatory per CONTRIBUTING.md**, covering edge cases and errors, not just the happy path. Every `opencode_zen/<mod>.py` has a matching `tests/unit/test_<mod>.py` — add to the existing file rather than creating a new one.
- The test suite enforces **total network isolation** via a session-scoped autouse fixture in `conftest.py` that patches `httpx.AsyncClient`; an unmocked network call is meant to fail the test. Mocking is plain `unittest.mock` plus hand-rolled async generators — no `respx`/`pytest-httpx`.
- `pytest-asyncio` runs in **strict mode**, so every async test needs an explicit `@pytest.mark.asyncio`.
- Conventional Commits (`fix(scope): ...`). Don't mix formatting sweeps with functional changes.

Beware two stale artifacts: `tests/README.md` (references deleted modules and `pip install -r requirements.txt`) and `hypothesis` in the dev deps (declared but no property tests exist).
