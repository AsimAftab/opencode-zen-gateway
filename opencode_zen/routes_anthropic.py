# -*- coding: utf-8 -*-

"""
FastAPI routes for Anthropic Messages API.
"""

import json
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Security, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from opencode_zen.config import PROXY_API_KEY, OPENCODE_BASE_URL, WEB_SEARCH_ENABLED
from opencode_zen.models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
    AnthropicMessage,
)
from opencode_zen.converters_anthropic import anthropic_to_opencode
from opencode_zen.streaming_anthropic import (
    stream_openai_to_anthropic,
    stream_with_first_token_retry_anthropic,
    collect_anthropic_response,
    format_sse_event,
)
from opencode_zen.http_client import OpenCodeHttpClient
from opencode_zen.tokenizer import estimate_request_tokens
from opencode_zen.utils import generate_conversation_id

try:
    from opencode_zen.debug_logger import debug_logger
except ImportError:
    debug_logger = None


def _extract_upstream_error_anthropic(response: httpx.Response) -> dict:
    """
    Extracts an Anthropic-shaped error body from an upstream error response.

    OpenCode Zen already returns Anthropic-shaped errors
    ({"type": "error", "error": {"type": ..., "message": ...}}) — those are
    passed through verbatim so clients (e.g. Claude Code) display the real
    upstream message. OpenAI-shaped or plain-text errors are wrapped.
    """
    try:
        error_data = response.json()
    except ValueError:
        error_data = None

    if isinstance(error_data, dict):
        inner = error_data.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return {
                "type": "error",
                "error": {"type": inner.get("type", "api_error"), "message": inner["message"]}
            }

    message = response.text or f"Upstream error (HTTP {response.status_code})"
    return {"type": "error", "error": {"type": "api_error", "message": message}}

anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
auth_header = APIKeyHeader(name="Authorization", auto_error=False)

async def verify_anthropic_api_key(
    x_api_key: Optional[str] = Security(anthropic_api_key_header),
    authorization: Optional[str] = Security(auth_header)
) -> bool:
    if x_api_key and x_api_key == PROXY_API_KEY:
        return True
    if authorization and authorization == f"Bearer {PROXY_API_KEY}":
        return True
    
    logger.warning("Access attempt with invalid API key (Anthropic endpoint)")
    raise HTTPException(
        status_code=401,
        detail={
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid or missing API key."
            }
        }
    )

router = APIRouter(tags=["Anthropic API"])

@router.post("/v1/messages", dependencies=[Depends(verify_anthropic_api_key)])
async def messages(
    request: Request,
    request_data: AnthropicMessagesRequest,
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version")
):
    logger.info(f"Request to /v1/messages (model={request_data.model}, stream={request_data.stream})")
    
    # 1. Truncation Recovery
    from opencode_zen.truncation_state import get_tool_truncation, get_content_truncation
    from opencode_zen.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    
    modified_messages = []
    for msg in request_data.messages:
        if msg.role == "user" and msg.content and isinstance(msg.content, list):
            modified_content_blocks = []
            has_modifications = False
            for block in msg.content:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    tool_use_id = block.get("tool_use_id")
                    original_content = block.get("content", "")
                elif hasattr(block, "type"):
                    block_type = block.type
                    tool_use_id = getattr(block, "tool_use_id", None)
                    original_content = getattr(block, "content", "")
                else:
                    modified_content_blocks.append(block)
                    continue
                
                if block_type == "tool_result" and tool_use_id:
                    truncation_info = get_tool_truncation(tool_use_id)
                    if truncation_info:
                        synthetic = generate_truncation_tool_result(
                            tool_name=truncation_info.tool_name,
                            tool_use_id=tool_use_id,
                            truncation_info=truncation_info.truncation_info
                        )
                        modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_content}"
                        if isinstance(block, dict):
                            modified_block = block.copy()
                            modified_block["content"] = modified_content
                        else:
                            modified_block = block.model_copy(update={"content": modified_content})
                        modified_content_blocks.append(modified_block)
                        has_modifications = True
                        continue
                modified_content_blocks.append(block)
            
            if has_modifications:
                modified_msg = msg.model_copy(update={"content": modified_content_blocks})
                modified_messages.append(modified_msg)
                continue
        
        if msg.role == "assistant" and msg.content:
            text_content = ""
            if isinstance(msg.content, str):
                text_content = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
            
            if text_content:
                truncation_info = get_content_truncation(text_content)
                if truncation_info:
                    modified_messages.append(msg)
                    synthetic_user_msg = AnthropicMessage(
                        role="user",
                        content=[{"type": "text", "text": generate_truncation_user_message()}]
                    )
                    modified_messages.append(synthetic_user_msg)
                    continue
        
        modified_messages.append(msg)
    
    request_data.messages = modified_messages

    # 2. Tool list hygiene
    # Server-side tools (e.g. Claude Code's web_search_20250305) have no
    # input_schema and cannot be executed by the upstream — strip them instead
    # of failing or diverting the request, so the rest of the request works.
    if request_data.tools:
        executable_tools = []
        for tool in request_data.tools:
            tool_type = getattr(tool, "type", None)
            input_schema = getattr(tool, "input_schema", None)
            if tool_type and input_schema is None:
                logger.warning(
                    f"Stripping unsupported server-side tool '{getattr(tool, 'name', tool_type)}' "
                    f"(type={tool_type}) — upstream has no server-side tool support"
                )
                continue
            executable_tools.append(tool)
        request_data.tools = executable_tools or None

    # 3. Web Search emulation (opt-in via WEB_SEARCH_ENABLED)
    # Offers a web_search function tool; the model's tool call is returned to
    # the client like any other function call — the request is never diverted.
    if WEB_SEARCH_ENABLED:
        if request_data.tools is None:
            request_data.tools = []
        has_ws = any(getattr(tool, "name", "") == "web_search" for tool in request_data.tools)
        if not has_ws:
            from opencode_zen.models_anthropic import AnthropicTool
            web_search_tool = AnthropicTool(
                name="web_search",
                description="Search the web for current information. Use when you need up-to-date data from the internet.",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query"}},
                    "required": ["query"]
                }
            )
            request_data.tools.append(web_search_tool)

    # 4. Build Payload
    conversation_id = generate_conversation_id()
    try:
        # anthropic_to_opencode now returns OpenAI payload
        opencode_payload = anthropic_to_opencode(request_data, conversation_id, "")
    except ValueError as e:
        logger.error(f"Conversion error: {e}")
        return JSONResponse(status_code=400, content={"type": "error", "error": {"type": "invalid_request_error", "message": str(e)}})

    url = f"{OPENCODE_BASE_URL}/chat/completions"
    
    if request_data.stream:
        http_client = OpenCodeHttpClient(shared_client=None)
    else:
        http_client = OpenCodeHttpClient(shared_client=request.app.state.http_client)
        
    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
    tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
    if isinstance(request_data.system, list):
        system_for_tokenizer = [b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system]
    else:
        system_for_tokenizer = request_data.system

    try:
        response = await http_client.request_with_retry("POST", url, opencode_payload, stream=True)
        
        if response.status_code == 200:
            if request_data.stream:
                async def stream_wrapper():
                    try:
                        async def make_retry_request():
                            return await http_client.request_with_retry("POST", url, opencode_payload, stream=True)

                        async for chunk in stream_with_first_token_retry_anthropic(
                            make_request=make_retry_request,
                            model=request_data.model,
                            model_cache=None,
                            auth_manager=None,
                            initial_response=response,
                            request_messages=messages_for_tokenizer,
                            request_tools=tools_for_tokenizer,
                            request_system=system_for_tokenizer,
                        ):
                            yield chunk
                    except Exception as e:
                        # Surface mid-stream failures as an Anthropic error
                        # event instead of silently truncating the stream.
                        logger.error(f"Streaming error: {e}")
                        yield format_sse_event("error", {
                            "type": "error",
                            "error": {"type": "api_error", "message": f"Stream interrupted: {str(e)}"}
                        })
                    finally:
                        if http_client._owns_client:
                            await http_client.close()

                return StreamingResponse(stream_wrapper(), media_type="text/event-stream")
            else:
                try:
                    anthropic_resp = await collect_anthropic_response(
                        response, request_data.model, None, None,
                        request_messages=messages_for_tokenizer, request_tools=tools_for_tokenizer, request_system=system_for_tokenizer
                    )
                    return JSONResponse(content=anthropic_resp)
                finally:
                    if http_client._owns_client:
                        await http_client.close()
        else:
            await response.aread()
            if http_client._owns_client:
                await http_client.close()

            return JSONResponse(
                status_code=response.status_code,
                content=_extract_upstream_error_anthropic(response)
            )

    except HTTPException:
        if http_client._owns_client:
            await http_client.close()
        raise
    except Exception as e:
        logger.error(f"Request failed: {e}")
        if http_client._owns_client:
            await http_client.close()
        return JSONResponse(
            status_code=502,
            content={"type": "error", "error": {"type": "api_error", "message": f"Connection failed: {str(e)}"}}
        )


@router.post("/v1/messages/count_tokens", dependencies=[Depends(verify_anthropic_api_key)])
async def count_tokens_endpoint(request_data: AnthropicCountTokensRequest):
    """
    Anthropic Count Tokens API (/v1/messages/count_tokens).

    Claude Code calls this endpoint for context management. The upstream has
    no token counting API, so tokens are estimated locally with the tiktoken
    approximation from tokenizer.py.
    """
    logger.info(f"Request to /v1/messages/count_tokens (model={request_data.model})")

    messages = [msg.model_dump() for msg in request_data.messages]
    tools = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
    if isinstance(request_data.system, list):
        system = [b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system]
    else:
        system = request_data.system

    stats = estimate_request_tokens(messages=messages, tools=tools, system_prompt=system)
    return JSONResponse(content={"input_tokens": stats["total_tokens"]})
