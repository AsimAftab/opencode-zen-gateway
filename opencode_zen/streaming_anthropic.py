# -*- coding: utf-8 -*-

"""
Streaming logic for converting OpenAI stream to Anthropic Messages API format.

The upstream OpenCode Zen API always speaks OpenAI chat-completions SSE.
This module renders that stream as the Anthropic Messages event sequence:
message_start → ping → content_block_start/delta/stop (per block) →
message_delta → message_stop.
"""

import json
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

import httpx
from loguru import logger

from opencode_zen.sse_aggregator import aggregate_openai_sse, extract_reasoning_delta
from opencode_zen.tokenizer import count_tokens, estimate_request_tokens


def generate_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def format_sse_event(event_type: str, data: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def map_finish_reason_to_stop_reason(finish_reason: Optional[str]) -> str:
    """
    Maps an OpenAI finish_reason to an Anthropic stop_reason.

    Args:
        finish_reason: OpenAI finish_reason (may be None if the stream
            ended without one)

    Returns:
        Anthropic stop_reason string
    """
    if finish_reason == "tool_calls":
        return "tool_use"
    if finish_reason == "length":
        return "max_tokens"
    return "end_turn"


def _estimate_input_tokens(
    request_messages: Optional[list],
    request_tools: Optional[list],
    request_system: Optional[Any],
) -> int:
    """Estimates input tokens for the usage block of message_start.

    Applies the Claude correction factor so this estimate matches what
    /v1/messages/count_tokens reports for the identical request — Claude Code
    mixes the two when tracking context, and a mismatch skews auto-compaction.
    """
    if not (request_messages or request_tools or request_system):
        return 0
    stats = estimate_request_tokens(
        messages=request_messages or [],
        tools=request_tools,
        system_prompt=request_system,
        apply_claude_correction=True,
    )
    return stats["total_tokens"]


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
    """
    Converts an upstream OpenAI SSE stream into Anthropic Messages SSE events.

    Robustness guarantees:
    - Content blocks get sequential indices; a new block always closes the
      previous one first (text → tool, tool → tool, thinking → text, ...).
    - Reasoning deltas (reasoning_content / reasoning) are rendered as an
      Anthropic thinking block.
    - message_delta + message_stop are ALWAYS emitted, even if the upstream
      stream ends without a finish_reason — otherwise clients hang.
    - Usage-only chunks (empty choices) are still read for token counts.
    """
    message_id = generate_message_id()

    input_tokens = _estimate_input_tokens(request_messages, request_tools, request_system)

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
    yield format_sse_event("ping", {"type": "ping"})

    # Block state: exactly one block can be open at a time.
    # current_block is None or one of "text", "thinking", "tool_use".
    current_block: Optional[str] = None
    block_index = -1
    accumulated_text = ""
    finish_reason: Optional[str] = None
    output_tokens = 0
    emitted_tool_use = False
    # Maps an upstream OpenAI tool-call index to the Anthropic block it opened,
    # so repeated fragments of the same call append to one block instead of
    # each opening a new one.
    seen_tool_indexes: Dict[int, int] = {}

    def close_block() -> Optional[str]:
        nonlocal current_block
        if current_block is None:
            return None
        event = format_sse_event(
            "content_block_stop", {"type": "content_block_stop", "index": block_index}
        )
        current_block = None
        return event

    def open_block(block_type: str, content_block: Dict[str, Any]) -> str:
        nonlocal current_block, block_index
        current_block = block_type
        block_index += 1
        return format_sse_event("content_block_start", {
            "type": "content_block_start",
            "index": block_index,
            "content_block": content_block
        })

    async for line in response.aiter_lines():
        if not line or not line.startswith("data: "):
            continue

        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            logger.debug(f"Skipping malformed SSE chunk: {data_str[:200]}")
            continue

        # Usage may arrive on any chunk, including usage-only final chunks
        usage = chunk.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens"):
            output_tokens = usage["completion_tokens"]

        if not chunk.get("choices"):
            continue

        choice = chunk["choices"][0]
        delta = choice.get("delta") or {}

        # Reasoning content → Anthropic thinking block
        reasoning_delta = extract_reasoning_delta(delta)
        if reasoning_delta:
            if current_block != "thinking":
                stop_event = close_block()
                if stop_event:
                    yield stop_event
                yield open_block("thinking", {"type": "thinking", "thinking": "", "signature": ""})
            accumulated_text += reasoning_delta
            yield format_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "thinking_delta", "thinking": reasoning_delta}
            })

        # Text content
        if isinstance(delta.get("content"), str) and delta["content"]:
            if current_block != "text":
                stop_event = close_block()
                if stop_event:
                    yield stop_event
                yield open_block("text", {"type": "text", "text": ""})
            accumulated_text += delta["content"]
            yield format_sse_event("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "text_delta", "text": delta["content"]}
            })

        # Tool calls. OpenAI streams identify concurrent/sequential calls by
        # "index"; a new index (not the mere presence of an id, which some
        # upstreams repeat on every fragment) starts a new Anthropic block.
        for tc in delta.get("tool_calls") or []:
            tc_index = tc.get("index", 0)
            function = tc.get("function") or {}

            if tc_index not in seen_tool_indexes:
                stop_event = close_block()
                if stop_event:
                    yield stop_event
                tool_id = tc.get("id") or f"call_{uuid.uuid4().hex[:24]}"
                tool_name = function.get("name", "unknown")
                yield open_block("tool_use", {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": {}
                })
                seen_tool_indexes[tc_index] = block_index
                emitted_tool_use = True

            arguments = function.get("arguments")
            if arguments:
                accumulated_text += arguments
                yield format_sse_event("content_block_delta", {
                    "type": "content_block_delta",
                    "index": block_index,
                    "delta": {"type": "input_json_delta", "partial_json": arguments}
                })

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    # Terminate the message: always close the open block and emit
    # message_delta + message_stop, even if upstream never sent finish_reason.
    stop_event = close_block()
    if stop_event:
        yield stop_event

    if not output_tokens and accumulated_text:
        output_tokens = count_tokens(accumulated_text, apply_claude_correction=True)

    # If the upstream emitted tool calls but died before sending a
    # finish_reason, report tool_use — otherwise the client sees end_turn and
    # never executes the tools (mirrors collect_anthropic_response).
    stop_reason = map_finish_reason_to_stop_reason(finish_reason)
    if finish_reason is None and emitted_tool_use:
        stop_reason = "tool_use"

    yield format_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": None
        },
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
    """
    Collects the upstream OpenAI SSE stream into a complete Anthropic message.

    The upstream is ALWAYS streamed (converters_core hardcodes "stream": true),
    so a non-streaming client response must be aggregated from SSE — the
    upstream body is never a plain JSON completion.

    Args:
        response: httpx streaming response from the upstream
        model: Model ID to echo back to the client
        model_cache: Unused (legacy signature compatibility)
        auth_manager: Unused (legacy signature compatibility)
        request_messages: Original client messages, for input token estimation
        request_tools: Original client tools, for input token estimation
        request_system: Original client system prompt, for input token estimation

    Returns:
        Anthropic Messages API response dict
    """
    aggregated = await aggregate_openai_sse(response)

    input_tokens = aggregated.prompt_tokens or _estimate_input_tokens(
        request_messages, request_tools, request_system
    )
    output_tokens = aggregated.completion_tokens
    if not output_tokens:
        generated = aggregated.reasoning_content + aggregated.content + "".join(
            tc["function"]["arguments"] for tc in aggregated.tool_calls
        )
        if generated:
            output_tokens = count_tokens(generated, apply_claude_correction=True)

    content_blocks = []
    if aggregated.reasoning_content:
        content_blocks.append({
            "type": "thinking",
            "thinking": aggregated.reasoning_content,
            "signature": ""
        })
    if aggregated.content:
        content_blocks.append({"type": "text", "text": aggregated.content})

    for tc in aggregated.tool_calls:
        raw_arguments = tc["function"]["arguments"]
        try:
            tool_input = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            # The upstream silently truncates large tool-call arguments, leaving
            # invalid JSON. Returning input={} would make the client execute the
            # tool with no arguments as if that were intended — worse than an
            # error. Surface it so the caller returns a visible failure instead.
            logger.error(
                f"Tool call {tc['id']} has truncated/invalid JSON arguments "
                f"({len(raw_arguments)} chars); cannot build a valid tool_use block"
            )
            raise ValueError(
                f"Upstream returned truncated or invalid tool-call arguments "
                f"for tool '{tc['function']['name']}'"
            ) from exc
        content_blocks.append({
            "type": "tool_use",
            "id": tc["id"],
            "name": tc["function"]["name"],
            "input": tool_input
        })

    stop_reason = map_finish_reason_to_stop_reason(aggregated.finish_reason)
    if aggregated.finish_reason is None and aggregated.tool_calls:
        stop_reason = "tool_use"

    return {
        "id": generate_message_id(),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
    }
