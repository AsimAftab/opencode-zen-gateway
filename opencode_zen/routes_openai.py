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
    WEB_SEARCH_ENABLED
)
from opencode_zen.models_openai import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
    ChatMessage
)
from opencode_zen.converters_openai import build_opencode_payload
from opencode_zen.http_client import KiroHttpClient
from opencode_zen.utils import generate_conversation_id
from opencode_zen.mcp_tools import handle_native_web_search

try:
    from opencode_zen.debug_logger import debug_logger
except ImportError:
    debug_logger = None

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
    http_client = KiroHttpClient(shared_client=request.app.state.http_client)
    
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
        
    # Fallback if API fails
    openai_models = [
        OpenAIModel(id="claude-3-5-sonnet-20241022", owned_by="anthropic", description="Claude 3.5 Sonnet")
    ]
    return ModelList(data=openai_models)

@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")
    
    from opencode_zen.truncation_state import get_tool_truncation, get_content_truncation
    from opencode_zen.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    
    modified_messages = []
    for msg in request_data.messages:
        if msg.role == "tool" and msg.tool_call_id:
            truncation_info = get_tool_truncation(msg.tool_call_id)
            if truncation_info:
                synthetic = generate_truncation_tool_result(
                    tool_name=truncation_info.tool_name,
                    tool_use_id=msg.tool_call_id,
                    truncation_info=truncation_info.truncation_info
                )
                modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"
                if isinstance(msg, dict):
                    modified_msg = msg.copy()
                    modified_msg["content"] = modified_content
                else:
                    modified_msg = msg.model_copy(update={"content": modified_content})
                modified_messages.append(modified_msg)
                continue
                
        if msg.role == "assistant" and msg.content:
            text_content = msg.content if isinstance(msg.content, str) else ""
            if text_content:
                truncation_info = get_content_truncation(text_content)
                if truncation_info:
                    modified_messages.append(msg)
                    synthetic_user_msg = ChatMessage(
                        role="user",
                        content=generate_truncation_user_message()
                    )
                    modified_messages.append(synthetic_user_msg)
                    continue
                    
        modified_messages.append(msg)
        
    request_data.messages = modified_messages

    if WEB_SEARCH_ENABLED:
        if request_data.tools is None:
            request_data.tools = []
        has_ws = any(getattr(tool.function, "name", "") == "web_search" for tool in request_data.tools if getattr(tool, "function", None))
        if not has_ws:
            from opencode_zen.models_openai import OpenAITool, OpenAIFunction
            web_search_tool = OpenAITool(
                type="function",
                function=OpenAIFunction(
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

    if request_data.tools:
        for tool in request_data.tools:
            if getattr(tool.function, "name", "") == "web_search":
                return await handle_native_web_search(request, request_data, None, api_format="openai")

    conversation_id = generate_conversation_id()
    try:
        opencode_payload = build_opencode_payload(request_data, conversation_id, "")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": {"type": "invalid_request_error", "message": str(e)}})

    url = f"{OPENCODE_BASE_URL}/chat/completions"
    
    if request_data.stream:
        http_client = KiroHttpClient(shared_client=None)
    else:
        http_client = KiroHttpClient(shared_client=request.app.state.http_client)

    try:
        response = await http_client.request_with_retry("POST", url, opencode_payload, stream=True)
        
        if response.status_code == 200:
            if request_data.stream:
                async def stream_wrapper():
                    try:
                        async for line in response.aiter_lines():
                            if line:
                                yield f"{line}\n"
                    finally:
                        if http_client._owns_client:
                            await http_client.close()
                return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
            else:
                try:
                    await response.aread()
                    return Response(content=response.content, media_type="application/json")
                finally:
                    if http_client._owns_client:
                        await http_client.close()
        else:
            await response.aread()
            try:
                error_data = response.json()
            except ValueError:
                error_data = {"error": {"message": response.text}}
                
            if http_client._owns_client:
                await http_client.close()
                
            return JSONResponse(
                status_code=response.status_code,
                content={"error": {"type": "api_error", "message": f"OpenCode API error: {error_data}"}}
            )
    except Exception as e:
        logger.error(f"Request failed: {e}")
        if http_client._owns_client:
            await http_client.close()
        return JSONResponse(
            status_code=502,
            content={"error": {"type": "api_error", "message": f"Connection failed: {str(e)}"}}
        )