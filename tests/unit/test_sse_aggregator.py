# -*- coding: utf-8 -*-

"""
Unit tests for sse_aggregator.py - aggregation of upstream OpenAI SSE streams.

Covers:
- Text content aggregation
- Reasoning content aggregation (reasoning_content and reasoning fields)
- Incremental tool call assembly (single and multiple, by index)
- Usage extraction (inline and usage-only chunks)
- finish_reason handling and fallbacks
- Malformed chunk resilience
- build_openai_completion output shape
"""

import json

import pytest

from opencode_zen.sse_aggregator import (
    AggregatedCompletion,
    aggregate_openai_sse,
    build_openai_completion,
    extract_reasoning_delta,
)


class FakeStreamResponse:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def sse(chunk: dict) -> str:
    """Formats a chunk dict as an SSE data line."""
    return f"data: {json.dumps(chunk)}"


def delta_chunk(delta: dict, finish_reason=None, usage=None) -> str:
    """Builds an OpenAI stream chunk SSE line with the given delta."""
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return sse(chunk)


class TestExtractReasoningDelta:
    """Tests for extract_reasoning_delta helper."""

    def test_reasoning_content_field(self):
        """
        What it does: Extracts text from the reasoning_content delta field.
        Purpose: DeepSeek-style models emit reasoning_content.
        """
        print("Action: extracting from reasoning_content field")
        assert extract_reasoning_delta({"reasoning_content": "thinking..."}) == "thinking..."

    def test_reasoning_field(self):
        """
        What it does: Extracts text from the reasoning delta field.
        Purpose: Some providers use 'reasoning' instead of 'reasoning_content'.
        """
        print("Action: extracting from reasoning field")
        assert extract_reasoning_delta({"reasoning": "hmm"}) == "hmm"

    def test_no_reasoning_returns_empty(self):
        """
        What it does: Returns empty string when the delta has no reasoning.
        Purpose: Plain content deltas must not produce reasoning text.
        """
        print("Action: extracting from delta without reasoning")
        assert extract_reasoning_delta({"content": "hi"}) == ""

    def test_non_string_reasoning_returns_empty(self):
        """
        What it does: Ignores non-string reasoning values.
        Purpose: Defensive handling of malformed upstream data.
        """
        print("Action: extracting from delta with non-string reasoning")
        assert extract_reasoning_delta({"reasoning_content": {"x": 1}}) == ""


class TestAggregateOpenAISSESuccess:
    """Tests for successful stream aggregation."""

    @pytest.mark.asyncio
    async def test_aggregates_text_content(self):
        """
        What it does: Concatenates content deltas into one string.
        Purpose: Core aggregation path for non-streaming clients.
        """
        print("Setup: stream with two content deltas")
        response = FakeStreamResponse([
            delta_chunk({"role": "assistant", "content": "Hello"}),
            delta_chunk({"content": " world"}),
            delta_chunk({}, finish_reason="stop"),
            "data: [DONE]",
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: content={result.content!r}, finish={result.finish_reason}")
        assert result.content == "Hello world"
        assert result.finish_reason == "stop"
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_aggregates_reasoning_content(self):
        """
        What it does: Concatenates reasoning_content deltas separately from content.
        Purpose: Reasoning models must not have thinking mixed into the answer.
        """
        print("Setup: stream with reasoning and content deltas")
        response = FakeStreamResponse([
            delta_chunk({"reasoning_content": "step 1"}),
            delta_chunk({"reasoning_content": ", step 2"}),
            delta_chunk({"content": "Answer"}),
            delta_chunk({}, finish_reason="stop"),
            "data: [DONE]",
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: reasoning={result.reasoning_content!r}, content={result.content!r}")
        assert result.reasoning_content == "step 1, step 2"
        assert result.content == "Answer"

    @pytest.mark.asyncio
    async def test_assembles_single_tool_call(self):
        """
        What it does: Assembles a tool call from id/name chunk plus argument fragments.
        Purpose: Tool call arguments stream incrementally and must be joined.
        """
        print("Setup: stream with an incremental tool call")
        response = FakeStreamResponse([
            delta_chunk({"tool_calls": [{"index": 0, "id": "call_abc", "type": "function",
                                         "function": {"name": "get_weather", "arguments": ""}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
            delta_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: tool_calls={result.tool_calls}")
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["id"] == "call_abc"
        assert result.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.tool_calls[0]["function"]["arguments"] == '{"city":"Paris"}'
        assert result.finish_reason == "tool_calls"

    @pytest.mark.asyncio
    async def test_assembles_multiple_tool_calls_by_index(self):
        """
        What it does: Keeps concurrent tool calls separate using their stream index.
        Purpose: Parallel tool calls must not have arguments merged together.
        """
        print("Setup: stream with two interleaved tool calls")
        response = FakeStreamResponse([
            delta_chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "a", "arguments": "{}"}}]}),
            delta_chunk({"tool_calls": [{"index": 1, "id": "call_2", "function": {"name": "b", "arguments": ""}}]}),
            delta_chunk({"tool_calls": [{"index": 1, "function": {"arguments": '{"x":1}'}}]}),
            delta_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: {len(result.tool_calls)} tool calls")
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0]["function"]["name"] == "a"
        assert result.tool_calls[1]["function"]["name"] == "b"
        assert result.tool_calls[1]["function"]["arguments"] == '{"x":1}'

    @pytest.mark.asyncio
    async def test_reads_usage_from_usage_only_chunk(self):
        """
        What it does: Extracts usage from a final chunk with empty choices.
        Purpose: Many providers send usage in a trailing usage-only chunk.
        """
        print("Setup: stream ending in a usage-only chunk")
        response = FakeStreamResponse([
            delta_chunk({"content": "hi"}, finish_reason="stop"),
            sse({"id": "chatcmpl-test", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}}),
            "data: [DONE]",
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: prompt={result.prompt_tokens}, completion={result.completion_tokens}")
        assert result.prompt_tokens == 10
        assert result.completion_tokens == 3


class TestAggregateOpenAISSEEdgeCases:
    """Tests for malformed and unusual streams."""

    @pytest.mark.asyncio
    async def test_skips_malformed_json_lines(self):
        """
        What it does: Skips undecodable data lines and keeps aggregating.
        Purpose: One corrupt chunk must not lose the whole response.
        """
        print("Setup: stream with a corrupt JSON line in the middle")
        response = FakeStreamResponse([
            delta_chunk({"content": "Hello"}),
            "data: {not-json",
            delta_chunk({"content": " world"}),
            "data: [DONE]",
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: content={result.content!r}")
        assert result.content == "Hello world"

    @pytest.mark.asyncio
    async def test_stream_without_finish_reason(self):
        """
        What it does: Returns finish_reason=None when the stream never sends one.
        Purpose: Aborted upstream streams must still yield the partial content.
        """
        print("Setup: stream cut off before finish_reason")
        response = FakeStreamResponse([
            delta_chunk({"content": "partial"}),
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: content={result.content!r}, finish={result.finish_reason}")
        assert result.content == "partial"
        assert result.finish_reason is None

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """
        What it does: Returns an empty result for a stream with no data lines.
        Purpose: Defensive handling of empty upstream responses.
        """
        print("Setup: empty stream")
        response = FakeStreamResponse(["", "data: [DONE]"])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print("Comparing: everything empty")
        assert result.content == ""
        assert result.tool_calls == []

    @pytest.mark.asyncio
    async def test_tool_call_without_id_gets_generated_id(self):
        """
        What it does: Generates an id for tool calls the upstream never identified.
        Purpose: Clients require every tool call to carry a non-empty id.
        """
        print("Setup: tool call chunks without any id")
        response = FakeStreamResponse([
            delta_chunk({"tool_calls": [{"index": 0, "function": {"name": "t", "arguments": "{}"}}]}),
            "data: [DONE]",
        ])

        print("Action: aggregating stream")
        result = await aggregate_openai_sse(response)

        print(f"Comparing: id={result.tool_calls[0]['id']!r}")
        assert result.tool_calls[0]["id"].startswith("call_")
        assert len(result.tool_calls[0]["id"]) > 5


class TestBuildOpenAICompletion:
    """Tests for building a chat.completion response from aggregated data."""

    def test_text_response_shape(self):
        """
        What it does: Builds a complete chat.completion dict for a text answer.
        Purpose: Non-streaming OpenAI clients get a spec-shaped response.
        """
        print("Setup: aggregated text result")
        aggregated = AggregatedCompletion(
            content="Hello", finish_reason="stop", prompt_tokens=5, completion_tokens=2
        )

        print("Action: building completion")
        completion = build_openai_completion(aggregated, "test-model")

        print(f"Comparing: {completion['choices'][0]}")
        assert completion["object"] == "chat.completion"
        assert completion["model"] == "test-model"
        assert completion["choices"][0]["message"]["content"] == "Hello"
        assert completion["choices"][0]["finish_reason"] == "stop"
        assert completion["usage"]["total_tokens"] == 7
        assert completion["id"].startswith("chatcmpl-")

    def test_tool_call_response_defaults_finish_reason(self):
        """
        What it does: Defaults finish_reason to tool_calls when tools are present but reason is missing.
        Purpose: Clients dispatch on finish_reason; it must be coherent.
        """
        print("Setup: aggregated result with a tool call and no finish_reason")
        aggregated = AggregatedCompletion(
            tool_calls=[{"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
        )

        print("Action: building completion")
        completion = build_openai_completion(aggregated, "m")

        print(f"Comparing: finish_reason={completion['choices'][0]['finish_reason']}")
        assert completion["choices"][0]["finish_reason"] == "tool_calls"
        assert completion["choices"][0]["message"]["tool_calls"] == aggregated.tool_calls
        assert completion["choices"][0]["message"]["content"] is None

    def test_reasoning_content_included(self):
        """
        What it does: Includes reasoning_content on the message when present.
        Purpose: Reasoning-capable clients can render the thinking separately.
        """
        print("Setup: aggregated result with reasoning")
        aggregated = AggregatedCompletion(content="A", reasoning_content="because...", finish_reason="stop")

        print("Action: building completion")
        completion = build_openai_completion(aggregated, "m")

        print("Comparing: reasoning_content present")
        assert completion["choices"][0]["message"]["reasoning_content"] == "because..."

    def test_empty_result_defaults_to_stop(self):
        """
        What it does: Defaults finish_reason to stop for an empty aggregation.
        Purpose: Even degenerate upstream streams produce a valid response.
        """
        print("Setup: empty aggregated result")
        completion = build_openai_completion(AggregatedCompletion(), "m")

        print(f"Comparing: finish_reason={completion['choices'][0]['finish_reason']}")
        assert completion["choices"][0]["finish_reason"] == "stop"
