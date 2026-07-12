# -*- coding: utf-8 -*-

# OpenCode Zen Gateway
# https://github.com/AsimAftab/opencode-zen-gateway
# Copyright (C) 2026 AsimAftab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Aggregation of upstream OpenAI SSE streams into complete responses.

The gateway always requests a streamed response from the upstream
(converters_core hardcodes "stream": true), so when a client asks for a
non-streaming response, the upstream SSE stream must be collected into a
single complete object. This module is the format-agnostic collector used
by BOTH the OpenAI route (/v1/chat/completions with stream=false) and the
Anthropic route (/v1/messages with stream=false).
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from loguru import logger


@dataclass
class AggregatedCompletion:
    """
    Result of aggregating an upstream OpenAI SSE stream.

    Attributes:
        content: Concatenated assistant text content
        reasoning_content: Concatenated reasoning/thinking content (if the
            upstream model emits reasoning_content or reasoning deltas)
        tool_calls: Completed tool calls in nested OpenAI wire shape
            ({"id", "type", "function": {"name", "arguments"}})
        finish_reason: Last finish_reason seen in the stream (None if the
            stream ended without one)
        prompt_tokens: Prompt token count from upstream usage (0 if absent)
        completion_tokens: Completion token count from upstream usage (0 if absent)
    """
    content: str = ""
    reasoning_content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def extract_reasoning_delta(delta: Dict[str, Any]) -> str:
    """
    Extracts reasoning text from an OpenAI stream delta.

    Different upstream providers use different field names for reasoning:
    DeepSeek-style models emit "reasoning_content", others emit "reasoning".

    Args:
        delta: The delta dict from a stream chunk choice

    Returns:
        Reasoning text fragment, or "" if the delta carries none
    """
    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


async def aggregate_openai_sse(response: httpx.Response) -> AggregatedCompletion:
    """
    Consumes an upstream OpenAI SSE stream and aggregates it into one result.

    Handles:
    - Text content deltas
    - Reasoning deltas (reasoning_content / reasoning)
    - Incremental tool calls (accumulated by stream index)
    - Usage chunks (including usage-only chunks with empty choices)
    - Malformed JSON lines (skipped, not fatal)

    Args:
        response: httpx streaming response from the upstream

    Returns:
        AggregatedCompletion with the complete assistant turn
    """
    result = AggregatedCompletion()
    # OpenAI streams identify concurrent tool calls by "index"
    tool_calls_by_index: Dict[int, Dict[str, Any]] = {}

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

        # Usage can arrive on any chunk, including a final usage-only chunk
        # with an empty choices list
        usage = chunk.get("usage")
        if isinstance(usage, dict):
            result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens) or result.prompt_tokens
            result.completion_tokens = usage.get("completion_tokens", result.completion_tokens) or result.completion_tokens

        choices = chunk.get("choices")
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}

        if isinstance(delta.get("content"), str):
            result.content += delta["content"]

        result.reasoning_content += extract_reasoning_delta(delta)

        for tc in delta.get("tool_calls") or []:
            index = tc.get("index", 0)
            entry = tool_calls_by_index.setdefault(index, {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""}
            })
            if tc.get("id"):
                entry["id"] = tc["id"]
            function = tc.get("function") or {}
            if function.get("name"):
                entry["function"]["name"] = function["name"]
            if function.get("arguments"):
                entry["function"]["arguments"] += function["arguments"]

        if choice.get("finish_reason"):
            result.finish_reason = choice["finish_reason"]

    for index in sorted(tool_calls_by_index):
        entry = tool_calls_by_index[index]
        if not entry["id"]:
            entry["id"] = f"call_{uuid.uuid4().hex[:24]}"
        if not entry["function"]["arguments"]:
            entry["function"]["arguments"] = "{}"
        result.tool_calls.append(entry)

    return result


def build_openai_completion(aggregated: AggregatedCompletion, model: str) -> Dict[str, Any]:
    """
    Builds a complete OpenAI chat.completion response from an aggregated stream.

    Args:
        aggregated: The aggregated upstream stream
        model: Model ID to echo back to the client

    Returns:
        chat.completion response dict in OpenAI format
    """
    message: Dict[str, Any] = {
        "role": "assistant",
        "content": aggregated.content or None,
    }
    if aggregated.reasoning_content:
        message["reasoning_content"] = aggregated.reasoning_content
    if aggregated.tool_calls:
        message["tool_calls"] = aggregated.tool_calls

    finish_reason = aggregated.finish_reason
    if not finish_reason:
        finish_reason = "tool_calls" if aggregated.tool_calls else "stop"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": aggregated.prompt_tokens,
            "completion_tokens": aggregated.completion_tokens,
            "total_tokens": aggregated.prompt_tokens + aggregated.completion_tokens,
        },
    }
