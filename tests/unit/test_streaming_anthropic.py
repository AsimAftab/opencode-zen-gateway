# -*- coding: utf-8 -*-

"""
Unit tests for streaming_anthropic.py - OpenAI SSE → Anthropic SSE translation.

Covers:
- Event sequence correctness (message_start → ping → blocks → message_delta → message_stop)
- Content block index management across text/thinking/tool blocks
- Multiple tool calls each getting their own block
- Reasoning deltas rendered as thinking blocks
- Guaranteed termination when upstream ends without finish_reason
- stop_reason mapping
- collect_anthropic_response aggregation from SSE
"""

import json

import pytest

from opencode_zen.streaming_anthropic import (
    collect_anthropic_response,
    format_sse_event,
    map_finish_reason_to_stop_reason,
    stream_openai_to_anthropic,
)


class FakeStreamResponse:
    """Minimal stand-in for an httpx streaming response."""

    def __init__(self, lines):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def sse(chunk: dict) -> str:
    return f"data: {json.dumps(chunk)}"


def delta_chunk(delta: dict, finish_reason=None, usage=None) -> str:
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage is not None:
        chunk["usage"] = usage
    return sse(chunk)


async def collect_events(response, model="test-model"):
    """Runs the translator and parses emitted SSE back into (type, data) tuples."""
    events = []
    async for raw in stream_openai_to_anthropic(response, model=model):
        for block in raw.strip().split("\n\n"):
            lines = block.strip().split("\n")
            event_type = lines[0].removeprefix("event: ")
            data = json.loads(lines[1].removeprefix("data: "))
            events.append((event_type, data))
    return events


class TestFormatSseEvent:
    """Tests for SSE event formatting."""

    def test_format_produces_event_and_data_lines(self):
        """
        What it does: Formats an event as 'event: X\\ndata: {...}\\n\\n'.
        Purpose: Anthropic SSE framing requires named events with blank-line separators.
        """
        print("Action: formatting a ping event")
        raw = format_sse_event("ping", {"type": "ping"})
        assert raw == 'event: ping\ndata: {"type": "ping"}\n\n'


class TestMapFinishReason:
    """Tests for finish_reason → stop_reason mapping."""

    def test_mapping_table(self):
        """
        What it does: Maps OpenAI finish_reasons to Anthropic stop_reasons.
        Purpose: Clients dispatch on stop_reason (tool_use triggers tool execution).
        """
        print("Action: mapping all finish reasons")
        assert map_finish_reason_to_stop_reason("tool_calls") == "tool_use"
        assert map_finish_reason_to_stop_reason("length") == "max_tokens"
        assert map_finish_reason_to_stop_reason("stop") == "end_turn"
        assert map_finish_reason_to_stop_reason(None) == "end_turn"


class TestStreamTranslationSuccess:
    """Tests for the happy-path event sequence."""

    @pytest.mark.asyncio
    async def test_text_stream_event_sequence(self):
        """
        What it does: Translates a plain text stream into the full Anthropic sequence.
        Purpose: The exact event order is what Anthropic SDK clients parse.
        """
        print("Setup: simple text stream")
        response = FakeStreamResponse([
            delta_chunk({"role": "assistant", "content": "Hel"}),
            delta_chunk({"content": "lo"}),
            delta_chunk({}, finish_reason="stop"),
            "data: [DONE]",
        ])

        print("Action: translating stream")
        events = await collect_events(response)
        types = [t for t, _ in events]

        print(f"Comparing sequence: {types}")
        assert types == [
            "message_start", "ping",
            "content_block_start", "content_block_delta", "content_block_delta", "content_block_stop",
            "message_delta", "message_stop",
        ]
        deltas = [d["delta"]["text"] for t, d in events if t == "content_block_delta"]
        assert "".join(deltas) == "Hello"
        message_delta = next(d for t, d in events if t == "message_delta")
        assert message_delta["delta"]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """
        What it does: Renders a tool call as a tool_use block with input_json_delta events.
        Purpose: Tool calling is the core Claude Code workflow.
        """
        print("Setup: stream with one tool call")
        response = FakeStreamResponse([
            delta_chunk({"tool_calls": [{"index": 0, "id": "call_1",
                                         "function": {"name": "read_file", "arguments": ""}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"path":"x"}'}}]}),
            delta_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ])

        print("Action: translating stream")
        events = await collect_events(response)

        starts = [d for t, d in events if t == "content_block_start"]
        print(f"Comparing: block starts={starts}")
        assert len(starts) == 1
        assert starts[0]["content_block"]["type"] == "tool_use"
        assert starts[0]["content_block"]["id"] == "call_1"
        assert starts[0]["content_block"]["name"] == "read_file"

        json_deltas = [d["delta"]["partial_json"] for t, d in events
                       if t == "content_block_delta" and d["delta"]["type"] == "input_json_delta"]
        assert "".join(json_deltas) == '{"path":"x"}'

        message_delta = next(d for t, d in events if t == "message_delta")
        assert message_delta["delta"]["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_text_then_tool_block_indices(self):
        """
        What it does: Gives the text block index 0 and the tool block index 1, closing text first.
        Purpose: Overlapping or reused indices corrupt client-side block assembly.
        """
        print("Setup: text followed by a tool call")
        response = FakeStreamResponse([
            delta_chunk({"content": "Let me check."}),
            delta_chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "t", "arguments": "{}"}}]}),
            delta_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ])

        print("Action: translating stream")
        events = await collect_events(response)

        starts = [(d["index"], d["content_block"]["type"]) for t, d in events if t == "content_block_start"]
        stops = [d["index"] for t, d in events if t == "content_block_stop"]
        print(f"Comparing: starts={starts}, stops={stops}")
        assert starts == [(0, "text"), (1, "tool_use")]
        assert stops == [0, 1]

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_get_separate_blocks(self):
        """
        What it does: Opens a new block (and closes the previous) for each new tool call id.
        Purpose: Parallel tool calls previously shared one index, corrupting the stream.
        """
        print("Setup: two sequential tool calls")
        response = FakeStreamResponse([
            delta_chunk({"tool_calls": [{"index": 0, "id": "call_1", "function": {"name": "a", "arguments": "{}"}}]}),
            delta_chunk({"tool_calls": [{"index": 1, "id": "call_2", "function": {"name": "b", "arguments": "{}"}}]}),
            delta_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ])

        print("Action: translating stream")
        events = await collect_events(response)

        starts = [(d["index"], d["content_block"]["id"]) for t, d in events if t == "content_block_start"]
        stops = [d["index"] for t, d in events if t == "content_block_stop"]
        print(f"Comparing: starts={starts}, stops={stops}")
        assert starts == [(0, "call_1"), (1, "call_2")]
        assert stops == [0, 1]

    @pytest.mark.asyncio
    async def test_reasoning_rendered_as_thinking_block(self):
        """
        What it does: Renders reasoning_content deltas as a thinking block before the text block.
        Purpose: Reasoning models (DeepSeek etc.) surface thinking natively to Anthropic clients.
        """
        print("Setup: reasoning followed by answer text")
        response = FakeStreamResponse([
            delta_chunk({"reasoning_content": "pondering"}),
            delta_chunk({"content": "Answer"}),
            delta_chunk({}, finish_reason="stop"),
            "data: [DONE]",
        ])

        print("Action: translating stream")
        events = await collect_events(response)

        starts = [(d["index"], d["content_block"]["type"]) for t, d in events if t == "content_block_start"]
        print(f"Comparing: starts={starts}")
        assert starts == [(0, "thinking"), (1, "text")]

        thinking_deltas = [d["delta"]["thinking"] for t, d in events
                           if t == "content_block_delta" and d["delta"]["type"] == "thinking_delta"]
        assert "".join(thinking_deltas) == "pondering"

    @pytest.mark.asyncio
    async def test_usage_from_usage_only_chunk(self):
        """
        What it does: Reads completion_tokens from a trailing usage-only chunk.
        Purpose: Usage-only chunks have empty choices and were previously skipped entirely.
        """
        print("Setup: stream with trailing usage-only chunk")
        response = FakeStreamResponse([
            delta_chunk({"content": "hi"}, finish_reason="stop"),
            sse({"id": "x", "choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 42}}),
            "data: [DONE]",
        ])

        print("Action: translating stream")
        events = await collect_events(response)

        message_delta = next(d for t, d in events if t == "message_delta")
        print(f"Comparing: usage={message_delta['usage']}")
        assert message_delta["usage"]["output_tokens"] == 42


class TestStreamTranslationEdgeCases:
    """Tests for degenerate upstream streams."""

    @pytest.mark.asyncio
    async def test_stream_without_finish_reason_still_terminates(self):
        """
        What it does: Emits content_block_stop, message_delta and message_stop even when
        the upstream dies without a finish_reason.
        Purpose: Without guaranteed termination the client hangs forever.
        """
        print("Setup: stream cut off mid-content")
        response = FakeStreamResponse([
            delta_chunk({"content": "partial"}),
        ])

        print("Action: translating stream")
        events = await collect_events(response)
        types = [t for t, _ in events]

        print(f"Comparing: {types}")
        assert types[-2:] == ["message_delta", "message_stop"]
        assert "content_block_stop" in types

    @pytest.mark.asyncio
    async def test_empty_stream_still_emits_full_envelope(self):
        """
        What it does: Emits message_start/message_delta/message_stop for an empty stream.
        Purpose: Clients need a complete message envelope even with no content.
        """
        print("Setup: empty stream")
        response = FakeStreamResponse(["data: [DONE]"])

        print("Action: translating stream")
        events = await collect_events(response)
        types = [t for t, _ in events]

        print(f"Comparing: {types}")
        assert types == ["message_start", "ping", "message_delta", "message_stop"]

    @pytest.mark.asyncio
    async def test_malformed_chunks_are_skipped(self):
        """
        What it does: Skips undecodable data lines without dying.
        Purpose: One corrupt chunk must not kill the whole stream.
        """
        print("Setup: stream with corrupt line")
        response = FakeStreamResponse([
            delta_chunk({"content": "ok"}),
            "data: }garbage{",
            delta_chunk({}, finish_reason="stop"),
            "data: [DONE]",
        ])

        print("Action: translating stream")
        events = await collect_events(response)

        deltas = [d["delta"]["text"] for t, d in events if t == "content_block_delta"]
        print(f"Comparing: deltas={deltas}")
        assert "".join(deltas) == "ok"

    @pytest.mark.asyncio
    async def test_input_tokens_estimated_from_request(self):
        """
        What it does: Puts a locally estimated input token count into message_start usage.
        Purpose: The upstream reports no usage until the end; Claude Code reads input_tokens early.
        """
        print("Setup: stream plus request messages for estimation")
        response = FakeStreamResponse(["data: [DONE]"])

        print("Action: translating stream with request context")
        events = []
        async for raw in stream_openai_to_anthropic(
            response, model="m",
            request_messages=[{"role": "user", "content": "Hello there, how are you today?"}],
        ):
            for block in raw.strip().split("\n\n"):
                lines = block.strip().split("\n")
                events.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))

        message_start = next(d for t, d in events if t == "message_start")
        print(f"Comparing: input_tokens={message_start['message']['usage']['input_tokens']}")
        assert message_start["message"]["usage"]["input_tokens"] > 0


class TestCollectAnthropicResponse:
    """Tests for non-streaming aggregation into an Anthropic message."""

    @pytest.mark.asyncio
    async def test_collects_text_response(self):
        """
        What it does: Aggregates an SSE stream into a complete Anthropic message dict.
        Purpose: The upstream is ALWAYS streamed; non-streaming clients need aggregation.
        """
        print("Setup: text SSE stream")
        response = FakeStreamResponse([
            delta_chunk({"content": "Hello"}),
            delta_chunk({"content": " world"}),
            delta_chunk({}, finish_reason="stop", usage={"prompt_tokens": 4, "completion_tokens": 2}),
            "data: [DONE]",
        ])

        print("Action: collecting response")
        result = await collect_anthropic_response(response, "test-model", None, None)

        print(f"Comparing: {result['content']}")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"] == [{"type": "text", "text": "Hello world"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 4
        assert result["usage"]["output_tokens"] == 2
        assert result["id"].startswith("msg_")

    @pytest.mark.asyncio
    async def test_collects_tool_use_response(self):
        """
        What it does: Builds tool_use blocks with parsed JSON input from tool call fragments.
        Purpose: Non-streaming tool calling must produce ready-to-execute input dicts.
        """
        print("Setup: tool call SSE stream")
        response = FakeStreamResponse([
            delta_chunk({"tool_calls": [{"index": 0, "id": "call_9",
                                         "function": {"name": "search", "arguments": ""}}]}),
            delta_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q":"cats"}'}}]}),
            delta_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ])

        print("Action: collecting response")
        result = await collect_anthropic_response(response, "m", None, None)

        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        print(f"Comparing: {tool_blocks}")
        assert tool_blocks == [{"type": "tool_use", "id": "call_9", "name": "search", "input": {"q": "cats"}}]
        assert result["stop_reason"] == "tool_use"

    @pytest.mark.asyncio
    async def test_invalid_tool_arguments_fall_back_to_empty_input(self):
        """
        What it does: Substitutes {} when tool arguments are not valid JSON.
        Purpose: Truncated upstream arguments must not crash the response.
        """
        print("Setup: tool call with truncated JSON arguments")
        response = FakeStreamResponse([
            delta_chunk({"tool_calls": [{"index": 0, "id": "call_1",
                                         "function": {"name": "t", "arguments": '{"broken":'}}]}),
            delta_chunk({}, finish_reason="tool_calls"),
            "data: [DONE]",
        ])

        print("Action: collecting response")
        result = await collect_anthropic_response(response, "m", None, None)

        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        print(f"Comparing: input={tool_block['input']}")
        assert tool_block["input"] == {}

    @pytest.mark.asyncio
    async def test_reasoning_becomes_thinking_block(self):
        """
        What it does: Puts aggregated reasoning into a leading thinking block.
        Purpose: Anthropic clients render thinking blocks natively.
        """
        print("Setup: reasoning + text stream")
        response = FakeStreamResponse([
            delta_chunk({"reasoning_content": "hmm"}),
            delta_chunk({"content": "Answer"}),
            delta_chunk({}, finish_reason="stop"),
            "data: [DONE]",
        ])

        print("Action: collecting response")
        result = await collect_anthropic_response(response, "m", None, None)

        print(f"Comparing: {result['content']}")
        assert result["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": ""}
        assert result["content"][1] == {"type": "text", "text": "Answer"}
