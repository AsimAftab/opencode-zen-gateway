# -*- coding: utf-8 -*-

"""
Streaming logic for converting OpenAI stream to Anthropic Messages API format.
"""

import json
import uuid
import httpx
from typing import AsyncGenerator, Optional, Any
from loguru import logger
import asyncio
from opencode_zen.tokenizer import estimate_request_tokens

def generate_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"

def format_sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

async def stream_with_first_token_retry_anthropic(
    make_request,
    model: str,
    model_cache: Any,
    auth_manager: Any,
    initial_response: httpx.Response,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    request_system: Optional[Any] = None,
) -> AsyncGenerator[str, None]:
    
    # Simple pass-through for now, actual implementation handles first token timeout
    async for chunk in stream_openai_to_anthropic(
        initial_response,
        model=model,
        request_messages=request_messages,
        request_tools=request_tools,
        request_system=request_system
    ):
        yield chunk

async def stream_openai_to_anthropic(
    response: httpx.Response,
    model: str,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    request_system: Optional[Any] = None,
) -> AsyncGenerator[str, None]:
    message_id = generate_message_id()
    
    input_tokens = 0
    if request_messages or request_tools or request_system:
        stats = estimate_request_tokens(
            messages=request_messages or [],
            tools=request_tools,
            system_prompt=request_system,
            apply_claude_correction=False
        )
        input_tokens = stats["total_tokens"]

    # Yield message_start
    yield format_sse_event("message_start", {
        "type": "message_start",
        "message": {
            "id": message_id,
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0}
        }
    })

    text_block_index = 0
    in_text_block = False
    
    # Tool call tracking
    in_tool_call = False
    tool_call_index = 1
    current_tool_id = ""
    current_tool_name = ""

    async for line in response.aiter_lines():
        if not line or not line.startswith("data: "):
            continue
            
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break
            
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
            
        if not chunk.get("choices"):
            continue
            
        delta = chunk["choices"][0].get("delta", {})
        
        # Text Content
        if "content" in delta and delta["content"] is not None:
            # End tool call block if we were in one (OpenAI shouldn't mix, but just in case)
            in_tool_call = False
            
            if not in_text_block:
                in_text_block = True
                yield format_sse_event("content_block_start", {
                    "type": "content_block_start",
                    "index": text_block_index,
                    "content_block": {"type": "text", "text": ""}
                })
            
            if delta["content"]:
                yield format_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": delta["content"]}
                })
                
        # Tool Calls
        elif "tool_calls" in delta and delta["tool_calls"]:
            for tc in delta["tool_calls"]:
                # If there's an id, it's a new tool call
                if tc.get("id"):
                    if in_text_block:
                        yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
                        in_text_block = False
                        text_block_index += 1
                        
                    current_tool_id = tc["id"]
                    current_tool_name = tc.get("function", {}).get("name", "unknown")
                    
                    yield format_sse_event("content_block_start", {
                        "type": "content_block_start",
                        "index": tool_call_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": current_tool_id,
                            "name": current_tool_name,
                            "input": {}
                        }
                    })
                    in_tool_call = True
                
                # Delta arguments
                if tc.get("function", {}).get("arguments"):
                    yield format_sse_event("content_block_delta", {
                        "type": "content_block_delta",
                        "index": tool_call_index,
                        "delta": {"type": "input_json_delta", "partial_json": tc["function"]["arguments"]}
                    })
        
        # Finish reason
        finish_reason = chunk["choices"][0].get("finish_reason")
        if finish_reason:
            if in_text_block:
                yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": text_block_index})
            elif in_tool_call:
                yield format_sse_event("content_block_stop", {"type": "content_block_stop", "index": tool_call_index})
                
            anthropic_stop_reason = "end_turn"
            if finish_reason == "tool_calls":
                anthropic_stop_reason = "tool_use"
            elif finish_reason == "length":
                anthropic_stop_reason = "max_tokens"
                
            # Usage
            output_tokens = chunk.get("usage", {}).get("completion_tokens", 0)
            
            yield format_sse_event("message_delta", {
                "type": "message_delta",
                "delta": {"stop_reason": anthropic_stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": output_tokens}
            })
            
            yield format_sse_event("message_stop", {"type": "message_stop"})

async def collect_anthropic_response(
    response: httpx.Response,
    model: str,
    model_cache: Any,
    auth_manager: Any,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    request_system: Optional[Any] = None,
) -> dict:
    # Read the full non-streaming JSON from OpenAI format to Anthropic format
    data = response.json()
    message_id = generate_message_id()
    
    input_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = data.get("usage", {}).get("completion_tokens", 0)
    
    choice = data["choices"][0]
    message = choice["message"]
    finish_reason = choice.get("finish_reason")
    
    anthropic_stop_reason = "end_turn"
    if finish_reason == "tool_calls":
        anthropic_stop_reason = "tool_use"
    elif finish_reason == "length":
        anthropic_stop_reason = "max_tokens"
        
    content_blocks = []
    
    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})
        
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id"),
                "name": tc.get("function", {}).get("name"),
                "input": json.loads(tc.get("function", {}).get("arguments", "{}"))
            })
            
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": anthropic_stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    }
