# -*- coding: utf-8 -*-

"""
FastAPI routes for OpenCode Zen Gateway.

Contains all API endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions
"""

import json
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from opencode_zen.config import (
    PROXY_API_KEY,
    APP_VERSION,
    OPENCODE_BASE_URL,
    WEB_SEARCH_ENABLED,
    FALLBACK_MODELS,
)
from opencode_zen.models_openai import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
)
from opencode_zen.converters_openai import build_opencode_payload
from opencode_zen.http_client import OpenCodeHttpClient
from opencode_zen.sse_aggregator import aggregate_openai_sse, build_openai_completion
from opencode_zen.utils import generate_conversation_id

try:
    from opencode_zen.debug_logger import debug_logger
except ImportError:
    debug_logger = None


def _debug_flush(status_code: int, message: str = "") -> None:
    """Flush buffered debug logs on error (no-op unless DEBUG_MODE captures errors)."""
    if debug_logger:
        try:
            debug_logger.flush_on_error(status_code, message)
        except Exception as exc:  # never let debug logging break the response
            logger.debug(f"debug flush failed: {exc}")


def _debug_discard() -> None:
    """Discard buffered debug logs after a successful request."""
    if debug_logger:
        try:
            debug_logger.discard_buffers()
        except Exception as exc:
            logger.debug(f"debug discard failed: {exc}")


def _extract_upstream_error(response: httpx.Response) -> dict:
    """
    Extracts a clean OpenAI-shaped error dict from an upstream error response.

    OpenCode Zen returns Anthropic-shaped error bodies
    ({"type": "error", "error": {"type": ..., "message": ...}}); other
    upstream layers may return OpenAI-shaped ones ({"error": {...}}) or
    plain text. All are surfaced as {"type", "message"} so clients show the
    real upstream message instead of a stringified wrapper.
    """
    try:
        error_data = response.json()
    except ValueError:
        return {"type": "api_error", "message": response.text or f"Upstream error (HTTP {response.status_code})"}

    inner = error_data.get("error") if isinstance(error_data, dict) else None
    if isinstance(inner, dict) and inner.get("message"):
        return {"type": inner.get("type", "api_error"), "message": inner["message"]}
    if isinstance(inner, str):
        return {"type": "api_error", "message": inner}
    return {"type": "api_error", "message": json.dumps(error_data, ensure_ascii=False)}


api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True

router = APIRouter()

@router.get("/")
async def root():
    return {"status": "ok", "message": "OpenCode Gateway is running", "version": APP_VERSION}

@router.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat(), "version": APP_VERSION}

@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def get_models(request: Request):
    logger.info("Request to /v1/models")
    url = f"{OPENCODE_BASE_URL}/models"
    http_client = OpenCodeHttpClient(shared_client=request.app.state.http_client)
    
    try:
        response = await http_client.request_with_retry("GET", url)
        if response.status_code == 200:
            data = response.json()
            if "data" in data:
                openai_models = []
                for model_data in data["data"]:
                    openai_models.append(
                        OpenAIModel(
                            id=model_data.get("id"),
                            owned_by=model_data.get("owned_by", "opencode"),
                            description=model_data.get("id")
                        )
                    )
                return ModelList(data=openai_models)
    except Exception as e:
        logger.error(f"Failed to fetch models from OpenCode: {e}")
        
    # Fallback if the upstream /models call fails: advertise the known-valid
    # upstream model IDs (single source of truth) so a client that picks one
    # does not immediately 400.
    openai_models = [
        OpenAIModel(id=m["modelId"], owned_by="opencode", description=m["modelId"])
        for m in FALLBACK_MODELS
    ]
    return ModelList(data=openai_models)

@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")

    if WEB_SEARCH_ENABLED:
        # Opt-in emulation: offer a web_search function tool so the model can
        # request searches. The tool call is returned to the client like any
        # other function call — the request is NEVER diverted away from the LLM.
        if request_data.tools is None:
            request_data.tools = []
        has_ws = any(getattr(tool.function, "name", "") == "web_search" for tool in request_data.tools if getattr(tool, "function", None))
        if not has_ws:
            from opencode_zen.models_openai import Tool, ToolFunction
            web_search_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Search the web for current information.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string", "description": "Search query"}},
                        "required": ["query"]
                    }
                )
            )
            request_data.tools.append(web_search_tool)

    conversation_id = generate_conversation_id()
    try:
        opencode_payload = build_opencode_payload(request_data, conversation_id, "")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": {"type": "invalid_request_error", "message": str(e)}})

    url = f"{OPENCODE_BASE_URL}/chat/completions"
    
    if request_data.stream:
        http_client = OpenCodeHttpClient(shared_client=None)
    else:
        http_client = OpenCodeHttpClient(shared_client=request.app.state.http_client)

    try:
        response = await http_client.request_with_retry("POST", url, opencode_payload, stream=True)

        if response.status_code == 200:
            if request_data.stream:
                async def stream_wrapper():
                    try:
                        # Verbatim byte passthrough: the upstream is already
                        # OpenAI-shaped SSE, and re-yielding raw bytes preserves
                        # the blank-line event framing SSE parsers require.
                        async for raw_chunk in response.aiter_raw():
                            yield raw_chunk
                        _debug_discard()
                    except Exception as e:
                        logger.error(f"Streaming error: {e}")
                        _debug_flush(502, f"Streaming error: {e}")
                        error_payload = {"error": {"type": "api_error", "message": f"Stream interrupted: {str(e)}"}}
                        yield f"data: {json.dumps(error_payload, ensure_ascii=False)}\n\n".encode("utf-8")
                    finally:
                        if http_client._owns_client:
                            await http_client.close()
                return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
            else:
                # Upstream is always streamed; aggregate SSE into a complete
                # chat.completion response for non-streaming clients.
                try:
                    aggregated = await aggregate_openai_sse(response)
                    _debug_discard()
                    return JSONResponse(content=build_openai_completion(aggregated, request_data.model))
                finally:
                    # The upstream is streamed even for non-streaming clients;
                    # aggregation breaks on [DONE] without exhausting the body,
                    # so close the response to release the pooled connection.
                    await response.aclose()
                    if http_client._owns_client:
                        await http_client.close()
        else:
            retry_after = response.headers.get("retry-after")
            await response.aread()
            await response.aclose()
            if http_client._owns_client:
                await http_client.close()

            _debug_flush(response.status_code, "Upstream returned an error status")
            error_headers = {"retry-after": retry_after} if retry_after else None
            return JSONResponse(
                status_code=response.status_code,
                content={"error": _extract_upstream_error(response)},
                headers=error_headers,
            )
    except HTTPException:
        if http_client._owns_client:
            await http_client.close()
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        _debug_flush(502, f"Request failed: {e}")
        if http_client._owns_client:
            await http_client.close()
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "api_error", "message": f"Connection failed: {str(e)}"}}
        )