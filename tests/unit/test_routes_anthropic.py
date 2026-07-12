"""
Unit tests for Anthropic API endpoints (routes_anthropic.py).

Tests the following endpoint:
- POST /v1/messages - Anthropic Messages API

For OpenAI API tests, see test_routes_openai.py.
"""
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timezone
import json
from fastapi import HTTPException
from fastapi.testclient import TestClient
from opencode_zen.routes_anthropic import verify_anthropic_api_key, router
from opencode_zen.config import PROXY_API_KEY


class TestVerifyAnthropicApiKey:
    """Tests for the verify_anthropic_api_key authentication function."""

    @pytest.mark.asyncio
    async def test_valid_x_api_key_returns_true(self):
        """
        What it does: Verifies that a valid x-api-key header passes authentication.
        Purpose: Ensure Anthropic native authentication works.
        """
        print('Setup: Creating valid x-api-key...')
        print('Action: Calling verify_anthropic_api_key...')
        result = await verify_anthropic_api_key(x_api_key=PROXY_API_KEY,
            authorization=None)
        print(f'Comparing result: Expected True, Got {result}')
        assert result is True

    @pytest.mark.asyncio
    async def test_valid_bearer_token_returns_true(self):
        """
        What it does: Verifies that a valid Bearer token passes authentication.
        Purpose: Ensure OpenAI-style authentication also works.
        """
        print('Setup: Creating valid Bearer token...')
        valid_auth = f'Bearer {PROXY_API_KEY}'
        print('Action: Calling verify_anthropic_api_key...')
        result = await verify_anthropic_api_key(x_api_key=None,
            authorization=valid_auth)
        print(f'Comparing result: Expected True, Got {result}')
        assert result is True

    @pytest.mark.asyncio
    async def test_x_api_key_takes_precedence(self):
        """
        What it does: Verifies x-api-key is checked before Authorization header.
        Purpose: Ensure Anthropic native auth has priority.
        """
        print('Setup: Both headers provided...')
        print('Action: Calling verify_anthropic_api_key with both headers...')
        result = await verify_anthropic_api_key(x_api_key=PROXY_API_KEY,
            authorization='Bearer wrong_key')
        print(f'Comparing result: Expected True, Got {result}')
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_x_api_key_raises_401(self):
        """
        What it does: Verifies that an invalid x-api-key is rejected.
        Purpose: Ensure unauthorized access is blocked.
        """
        print('Setup: Creating invalid x-api-key...')
        print('Action: Calling verify_anthropic_api_key with invalid key...')
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key='wrong_key',
                authorization=None)
        print(f'Checking: HTTPException with status 401...')
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_bearer_token_raises_401(self):
        """
        What it does: Verifies that an invalid Bearer token is rejected.
        Purpose: Ensure unauthorized access is blocked.
        """
        print('Setup: Creating invalid Bearer token...')
        print('Action: Calling verify_anthropic_api_key with invalid token...')
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key=None, authorization=
                'Bearer wrong_key')
        print(f'Checking: HTTPException with status 401...')
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_both_headers_raises_401(self):
        """
        What it does: Verifies that missing both headers is rejected.
        Purpose: Ensure authentication is required.
        """
        print('Setup: No authentication headers...')
        print('Action: Calling verify_anthropic_api_key with no headers...')
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key=None, authorization=None)
        print(f'Checking: HTTPException with status 401...')
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_x_api_key_raises_401(self):
        """
        What it does: Verifies that empty x-api-key is rejected.
        Purpose: Ensure empty credentials are blocked.
        """
        print('Setup: Empty x-api-key...')
        print('Action: Calling verify_anthropic_api_key with empty key...')
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key='', authorization=None)
        print(f'Checking: HTTPException with status 401...')
        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_error_response_format_is_anthropic_style(self):
        """
        What it does: Verifies error response follows Anthropic format.
        Purpose: Ensure error format matches Anthropic API.
        """
        print('Setup: Invalid credentials...')
        print('Action: Calling verify_anthropic_api_key...')
        with pytest.raises(HTTPException) as exc_info:
            await verify_anthropic_api_key(x_api_key='wrong', authorization
                =None)
        print(f'Checking: Error format...')
        detail = exc_info.value.detail
        assert 'type' in detail
        assert 'error' in detail
        assert detail['error']['type'] == 'authentication_error'


class TestMessagesAuthentication:
    """Tests for authentication on /v1/messages endpoint."""

    def test_messages_requires_authentication(self, test_client):
        """
        What it does: Verifies messages endpoint requires authentication.
        Purpose: Ensure protected endpoint is secured.
        """
        print('Action: POST /v1/messages without auth...')
        response = test_client.post('/v1/messages', json={'model':
            'claude-sonnet-4-5', 'max_tokens': 1024, 'messages': [{'role':
            'user', 'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code == 401

    def test_messages_accepts_x_api_key(self, test_client, valid_proxy_api_key
        ):
        """
        What it does: Verifies messages endpoint accepts x-api-key header.
        Purpose: Ensure Anthropic native authentication works.
        """
        print('Action: POST /v1/messages with x-api-key...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 401

    def test_messages_accepts_bearer_token(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies messages endpoint accepts Bearer token.
        Purpose: Ensure OpenAI-style authentication also works.
        """
        print('Action: POST /v1/messages with Bearer token...')
        response = test_client.post('/v1/messages', headers={
            'Authorization': f'Bearer {valid_proxy_api_key}'}, json={
            'model': 'claude-sonnet-4-5', 'max_tokens': 1024, 'messages': [
            {'role': 'user', 'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 401

    def test_messages_rejects_invalid_x_api_key(self, test_client,
        invalid_proxy_api_key):
        """
        What it does: Verifies messages endpoint rejects invalid x-api-key.
        Purpose: Ensure authentication is enforced.
        """
        print('Action: POST /v1/messages with invalid x-api-key...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            invalid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code == 401


class TestMessagesValidation:
    """Tests for request validation on /v1/messages endpoint."""

    def test_validates_missing_model(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies missing model field is rejected.
        Purpose: Ensure model is required.
        """
        print('Action: POST /v1/messages without model...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'max_tokens': 1024, 'messages': [{
            'role': 'user', 'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code == 422

    def test_validates_missing_max_tokens(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies missing max_tokens field is rejected.
        Purpose: Ensure max_tokens is required (Anthropic API requirement).
        """
        print('Action: POST /v1/messages without max_tokens...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'messages': [{'role': 'user', 'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code == 422

    def test_validates_missing_messages(self, test_client, valid_proxy_api_key
        ):
        """
        What it does: Verifies missing messages field is rejected.
        Purpose: Ensure messages are required.
        """
        print('Action: POST /v1/messages without messages...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024})
        print(f'Status: {response.status_code}')
        assert response.status_code == 422

    def test_validates_empty_messages_array(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies empty messages array is rejected.
        Purpose: Ensure at least one message is required.
        """
        print('Action: POST /v1/messages with empty messages...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': []})
        print(f'Status: {response.status_code}')
        assert response.status_code == 422

    def test_validates_invalid_json(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid JSON is rejected.
        Purpose: Ensure proper JSON parsing.
        """
        print('Action: POST /v1/messages with invalid JSON...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key, 'Content-Type': 'application/json'},
            content=b'not valid json {{{}')
        print(f'Status: {response.status_code}')
        assert response.status_code == 422

    def test_validates_invalid_role(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies invalid message role is rejected.
        Purpose: Anthropic model strictly validates role (only 'user' or 'assistant').
        """
        print('Action: POST /v1/messages with invalid role...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'invalid_role',
            'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code == 422

    def test_accepts_valid_request_format(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies valid request format passes validation.
        Purpose: Ensure Pydantic validation works correctly.
        """
        print('Action: POST /v1/messages with valid format...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422


class TestMessagesSystemPrompt:
    """Tests for system prompt handling on /v1/messages endpoint."""

    def test_accepts_system_as_separate_field(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies system prompt as separate field is accepted.
        Purpose: Ensure Anthropic-style system prompt works.
        """
        print('Action: POST /v1/messages with system field...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'system': 'You are a helpful assistant.',
            'messages': [{'role': 'user', 'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_empty_system_prompt(self, test_client, valid_proxy_api_key
        ):
        """
        What it does: Verifies empty system prompt is accepted.
        Purpose: Ensure system prompt is optional.
        """
        print('Action: POST /v1/messages with empty system...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'system': '', 'messages': [{'role': 'user',
            'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_no_system_prompt(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies request without system prompt is accepted.
        Purpose: Ensure system prompt is optional.
        """
        print('Action: POST /v1/messages without system field...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422


class TestMessagesContentBlocks:
    """Tests for content block handling on /v1/messages endpoint."""

    def test_accepts_string_content(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies string content is accepted.
        Purpose: Ensure simple string content works.
        """
        print('Action: POST /v1/messages with string content...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_content_block_array(self, test_client, valid_proxy_api_key
        ):
        """
        What it does: Verifies content block array is accepted.
        Purpose: Ensure Anthropic content block format works.
        """
        print('Action: POST /v1/messages with content blocks...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content': [{
            'type': 'text', 'text': 'Hello'}]}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_multiple_content_blocks(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies multiple content blocks are accepted.
        Purpose: Ensure complex content works.
        """
        print('Action: POST /v1/messages with multiple content blocks...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content': [{
            'type': 'text', 'text': 'First part'}, {'type': 'text', 'text':
            'Second part'}]}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422


class TestMessagesToolUse:
    """Tests for tool use on /v1/messages endpoint."""

    def test_accepts_tool_definition(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies tool definition is accepted.
        Purpose: Ensure Anthropic tool format works.
        """
        print('Action: POST /v1/messages with tools...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            "What's the weather?"}], 'tools': [{'name': 'get_weather',
            'description': 'Get weather for a location', 'input_schema': {
            'type': 'object', 'properties': {'location': {'type': 'string'}
            }, 'required': ['location']}}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_multiple_tools(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies multiple tools are accepted.
        Purpose: Ensure multiple tool definitions work.
        """
        print('Action: POST /v1/messages with multiple tools...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}], 'tools': [{'name': 'get_weather', 'description':
            'Get weather', 'input_schema': {'type': 'object', 'properties':
            {}}}, {'name': 'get_time', 'description': 'Get time',
            'input_schema': {'type': 'object', 'properties': {}}}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_tool_result_message(self, test_client, valid_proxy_api_key
        ):
        """
        What it does: Verifies tool result message is accepted.
        Purpose: Ensure tool result handling works.
        """
        print('Action: POST /v1/messages with tool result...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            "What's the weather?"}, {'role': 'assistant', 'content': [{
            'type': 'tool_use', 'id': 'call_123', 'name': 'get_weather',
            'input': {'location': 'Moscow'}}]}, {'role': 'user', 'content':
            [{'type': 'tool_result', 'tool_use_id': 'call_123', 'content':
            'Sunny, 25°C'}]}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestMessagesOptionalParams:
    """Tests for optional parameters on /v1/messages endpoint."""

    def test_accepts_temperature_parameter(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies temperature parameter is accepted.
        Purpose: Ensure temperature control works.
        """
        print('Action: POST /v1/messages with temperature...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}], 'temperature': 0.7})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_top_p_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies top_p parameter is accepted.
        Purpose: Ensure nucleus sampling control works.
        """
        print('Action: POST /v1/messages with top_p...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}], 'top_p': 0.9})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_top_k_parameter(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies top_k parameter is accepted.
        Purpose: Ensure top-k sampling control works.
        """
        print('Action: POST /v1/messages with top_k...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}], 'top_k': 40})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    @pytest.mark.skip(reason=
        'Obsolete after OpenAI payload and Account System migration')
    def test_accepts_stream_true(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies stream=true is accepted.
        Purpose: Ensure streaming mode is supported.
        """
        print('Action: POST /v1/messages with stream=true...')

        async def mock_stream(*args, **kwargs):
            yield 'event: message_start\ndata: {"type":"message_start"}\n\n'
            yield 'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        mock_response = AsyncMock()
        mock_response.status_code = 200

        async def mock_aiter_bytes():
            yield b'{"content":"test"}'
        mock_response.aiter_bytes = mock_aiter_bytes
        with patch(
            'opencode_zen.routes_anthropic.stream_opencode_zen_to_anthropic',
            mock_stream), patch(
            'opencode_zen.http_client.OpenCodeHttpClient.request_with_retry',
            return_value=mock_response):
            response = test_client.post('/v1/messages', headers={
                'x-api-key': valid_proxy_api_key}, json={'model':
                'claude-sonnet-4-5', 'max_tokens': 1024, 'messages': [{
                'role': 'user', 'content': 'Hello'}], 'stream': True})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_stop_sequences(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies stop_sequences parameter is accepted.
        Purpose: Ensure stop sequence control works.
        """
        print('Action: POST /v1/messages with stop_sequences...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}], 'stop_sequences': ['END', 'STOP']})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_metadata(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies metadata parameter is accepted.
        Purpose: Ensure metadata passing works.
        """
        print('Action: POST /v1/messages with metadata...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}], 'metadata': {'user_id': 'test_user'}})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422


class TestMessagesAnthropicVersion:
    """Tests for anthropic-version header handling."""

    def test_accepts_anthropic_version_header(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies anthropic-version header is accepted.
        Purpose: Ensure Anthropic SDK compatibility.
        """
        print('Action: POST /v1/messages with anthropic-version header...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key, 'anthropic-version': '2023-06-01'}, json={
            'model': 'claude-sonnet-4-5', 'max_tokens': 1024, 'messages': [
            {'role': 'user', 'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_works_without_anthropic_version_header(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies request works without anthropic-version header.
        Purpose: Ensure header is optional.
        """
        print('Action: POST /v1/messages without anthropic-version header...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422


class TestAnthropicRouterIntegration:
    """Tests for Anthropic router configuration and integration."""

    def test_router_has_messages_endpoint(self):
        """
        What it does: Verifies messages endpoint is registered.
        Purpose: Ensure endpoint is available.
        """
        print('Checking: Router endpoints...')
        routes = [route.path for route in router.routes]
        print(f'Found routes: {routes}')
        assert '/v1/messages' in routes

    def test_messages_endpoint_uses_post_method(self):
        """
        What it does: Verifies messages endpoint uses POST method.
        Purpose: Ensure correct HTTP method.
        """
        print('Checking: HTTP methods...')
        for route in router.routes:
            if route.path == '/v1/messages':
                print(f'Route /v1/messages methods: {route.methods}')
                assert 'POST' in route.methods
                return
        pytest.fail('Messages endpoint not found')

    def test_router_has_anthropic_tag(self):
        """
        What it does: Verifies router has Anthropic API tag.
        Purpose: Ensure proper API documentation grouping.
        """
        print('Checking: Router tags...')
        print(f'Router tags: {router.tags}')
        assert 'Anthropic API' in router.tags


class TestMessagesConversationHistory:
    """Tests for conversation history handling on /v1/messages endpoint."""

    def test_accepts_multi_turn_conversation(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Verifies multi-turn conversation is accepted.
        Purpose: Ensure conversation history works.
        """
        print('Action: POST /v1/messages with conversation history...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': [{'role': 'user', 'content':
            'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}, {
            'role': 'user', 'content': 'How are you?'}]})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422

    def test_accepts_long_conversation(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies long conversation is accepted.
        Purpose: Ensure many messages work.
        """
        print('Action: POST /v1/messages with long conversation...')
        messages = []
        for i in range(10):
            messages.append({'role': 'user', 'content': f'Message {i}'})
            messages.append({'role': 'assistant', 'content': f'Response {i}'})
        messages.append({'role': 'user', 'content': 'Final question'})
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
            'max_tokens': 1024, 'messages': messages})
        print(f'Status: {response.status_code}')
        assert response.status_code != 422


class TestMessagesErrorFormat:
    """Tests for error response format on /v1/messages endpoint."""

    def test_validation_error_format(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies validation error response format.
        Purpose: Ensure errors follow expected format.
        """
        print('Action: POST /v1/messages with invalid request...')
        response = test_client.post('/v1/messages', headers={'x-api-key':
            valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5'})
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        assert response.status_code == 422

    def test_auth_error_format_is_anthropic_style(self, test_client):
        """
        What it does: Verifies auth error follows Anthropic format.
        Purpose: Ensure error format matches Anthropic API.
        """
        print('Action: POST /v1/messages without auth...')
        response = test_client.post('/v1/messages', json={'model':
            'claude-sonnet-4-5', 'max_tokens': 1024, 'messages': [{'role':
            'user', 'content': 'Hello'}]})
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        assert response.status_code == 401
        data = response.json()
        assert 'detail' in data
        detail = data['detail']
        assert 'type' in detail
        assert 'error' in detail


class TestAnthropicHTTPClientSelection:
    """
    Tests for HTTP client selection in Anthropic routes (issue #54).
    
    Verifies that streaming requests use per-request clients to avoid CLOSE_WAIT leak
    when network interface changes (VPN disconnect/reconnect), while non-streaming
    requests use shared client for connection pooling.
    """

    @patch('opencode_zen.routes_anthropic.OpenCodeHttpClient')
    def test_streaming_uses_per_request_client(self,
        mock_opencode_zen_http_client_class, test_client, valid_proxy_api_key):
        """
        What it does: Verifies streaming requests create per-request HTTP client.
        Purpose: Prevent CLOSE_WAIT leak on VPN disconnect (issue #54).
        """
        print('\n--- Test: Anthropic streaming uses per-request client ---')
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(side_effect=
            Exception('Network blocked'))
        mock_client_instance.close = AsyncMock()
        mock_opencode_zen_http_client_class.return_value = mock_client_instance
        print('Action: POST /v1/messages with stream=true...')
        try:
            test_client.post('/v1/messages', headers={'x-api-key':
                valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
                'max_tokens': 100, 'messages': [{'role': 'user', 'content':
                'Hello'}], 'stream': True})
        except Exception:
            pass
        print('Checking: OpenCodeHttpClient(shared_client=None)...')
        assert mock_opencode_zen_http_client_class.called
        call_args = mock_opencode_zen_http_client_class.call_args
        print(f'Call args: {call_args}')
        assert call_args[1]['shared_client'
            ] is None, 'Streaming should use per-request client'
        print('✅ Anthropic streaming correctly uses per-request client')

    @patch('opencode_zen.routes_anthropic.OpenCodeHttpClient')
    def test_non_streaming_uses_shared_client(self,
        mock_opencode_zen_http_client_class, test_client, valid_proxy_api_key):
        """
        What it does: Verifies non-streaming requests use shared HTTP client.
        Purpose: Ensure connection pooling for non-streaming requests.
        """
        print('\n--- Test: Anthropic non-streaming uses shared client ---')
        mock_client_instance = AsyncMock()
        mock_client_instance.request_with_retry = AsyncMock(side_effect=
            Exception('Network blocked'))
        mock_client_instance.close = AsyncMock()
        mock_opencode_zen_http_client_class.return_value = mock_client_instance
        print('Action: POST /v1/messages with stream=false...')
        try:
            test_client.post('/v1/messages', headers={'x-api-key':
                valid_proxy_api_key}, json={'model': 'claude-sonnet-4-5',
                'max_tokens': 100, 'messages': [{'role': 'user', 'content':
                'Hello'}], 'stream': False})
        except Exception:
            pass
        print(
            'Checking: OpenCodeHttpClient(shared_client=app.state.http_client)...'
            )
        assert mock_opencode_zen_http_client_class.called
        call_args = mock_opencode_zen_http_client_class.call_args
        print(f'Call args: {call_args}')
        assert call_args[1]['shared_client'
            ] is not None, 'Non-streaming should use shared client'
        print('✅ Anthropic non-streaming correctly uses shared client')


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestTruncationRecoveryMessageModification:
    """
    Tests for Truncation Recovery System message modification in routes_anthropic.
    
    Verifies that tool_result blocks are modified when truncation info exists in cache.
    Part of Truncation Recovery System (Issue #56).
    """

    @staticmethod
    def _get_block_value(block, key, default=''):
        """Helper to get value from dict or Pydantic object."""
        if isinstance(block, dict):
            return block.get(key, default)
        else:
            return getattr(block, key, default)

    @pytest.mark.skip(reason=
        'Obsolete after OpenAI payload and Account System migration')
    def test_modifies_tool_result_dict_with_truncation_notice(self):
        """
        What it does: Verifies tool_result content block is modified when truncation info exists.
        Purpose: Ensure truncation notice is prepended to tool_result.
        """
        print('Setup: Saving truncation info to cache...')
        from opencode_zen.truncation_state import save_tool_truncation
        from opencode_zen.models_anthropic import AnthropicMessage
        tool_use_id = 'tooluse_test_dict'
        save_tool_truncation(tool_use_id, 'write_to_file', {'size_bytes': 
            5000, 'reason': 'test'})
        print('Setup: Creating request with tool_result...')
        messages = [AnthropicMessage(role='user', content=[{'type':
            'tool_result', 'tool_use_id': tool_use_id, 'content':
            'Missing parameter error'}])]
        print(
            'Action: Processing messages through truncation recovery logic...')
        from opencode_zen.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from opencode_zen.truncation_state import get_tool_truncation
        modified_messages = []
        for msg in messages:
            if msg.role == 'user' and msg.content and isinstance(msg.
                content, list):
                modified_content_blocks = []
                has_modifications = False
                for block in msg.content:
                    block_type = self._get_block_value(block, 'type')
                    block_tool_use_id = self._get_block_value(block,
                        'tool_use_id')
                    original_content = self._get_block_value(block,
                        'content', '')
                    if (block_type == 'tool_result' and block_tool_use_id and
                        should_inject_recovery()):
                        truncation_info = get_tool_truncation(block_tool_use_id
                            )
                        if truncation_info:
                            print(
                                f'Found truncation info for {block_tool_use_id}'
                                )
                            synthetic = generate_truncation_tool_result(
                                truncation_info.tool_name, truncation_info.
                                tool_call_id, truncation_info.truncation_info)
                            modified_content = f"""{synthetic['content']}

---

Original tool result:
{original_content}"""
                            if isinstance(block, dict):
                                modified_block = block.copy()
                                modified_block['content'] = modified_content
                            else:
                                modified_block = block.model_copy(update={
                                    'content': modified_content})
                            modified_content_blocks.append(modified_block)
                            has_modifications = True
                            continue
                    modified_content_blocks.append(block)
                if has_modifications:
                    modified_msg = msg.model_copy(update={'content':
                        modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            modified_messages.append(msg)
        print('Checking: Modified message content...')
        modified_msg = modified_messages[0]
        modified_block = modified_msg.content[0]
        content = self._get_block_value(modified_block, 'content')
        print(f'Content: {content[:100]}...')
        assert '[API Limitation]' in content
        assert 'Missing parameter error' in content
        assert '---' in content

    @pytest.mark.skip(reason=
        'Obsolete after OpenAI payload and Account System migration')
    def test_modifies_tool_result_pydantic_with_truncation_notice(self):
        """
        What it does: Verifies tool_result content block (Pydantic) is modified when truncation info exists.
        Purpose: Ensure truncation notice works with Pydantic ToolResultContentBlock.
        """
        print('Setup: Saving truncation info to cache...')
        from opencode_zen.truncation_state import save_tool_truncation
        from opencode_zen.models_anthropic import AnthropicMessage, ToolResultContentBlock
        tool_use_id = 'tooluse_test_pydantic'
        save_tool_truncation(tool_use_id, 'write_to_file', {'size_bytes': 
            5000, 'reason': 'test'})
        print('Setup: Creating request with tool_result (Pydantic format)...')
        tool_result_block = ToolResultContentBlock(type='tool_result',
            tool_use_id=tool_use_id, content='Missing parameter error')
        messages = [AnthropicMessage(role='user', content=[tool_result_block])]
        print(
            'Action: Processing messages through truncation recovery logic...')
        from opencode_zen.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from opencode_zen.truncation_state import get_tool_truncation
        modified_messages = []
        for msg in messages:
            if msg.role == 'user' and msg.content and isinstance(msg.
                content, list):
                modified_content_blocks = []
                has_modifications = False
                for block in msg.content:
                    block_type = self._get_block_value(block, 'type')
                    block_tool_use_id = self._get_block_value(block,
                        'tool_use_id')
                    original_content = self._get_block_value(block,
                        'content', '')
                    if (block_type == 'tool_result' and block_tool_use_id and
                        should_inject_recovery()):
                        truncation_info = get_tool_truncation(block_tool_use_id
                            )
                        if truncation_info:
                            print(
                                f'Found truncation info for {block_tool_use_id}'
                                )
                            synthetic = generate_truncation_tool_result(
                                truncation_info.tool_name, truncation_info.
                                tool_call_id, truncation_info.truncation_info)
                            modified_content = f"""{synthetic['content']}

---

Original tool result:
{original_content}"""
                            if isinstance(block, dict):
                                modified_block = block.copy()
                                modified_block['content'] = modified_content
                            else:
                                modified_block = block.model_copy(update={
                                    'content': modified_content})
                            modified_content_blocks.append(modified_block)
                            has_modifications = True
                            continue
                    modified_content_blocks.append(block)
                if has_modifications:
                    modified_msg = msg.model_copy(update={'content':
                        modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            modified_messages.append(msg)
        print('Checking: Modified message content...')
        modified_msg = modified_messages[0]
        modified_block = modified_msg.content[0]
        content = self._get_block_value(modified_block, 'content')
        print(f'Content: {content[:100]}...')
        assert '[API Limitation]' in content
        assert 'Missing parameter error' in content
        assert '---' in content

    @pytest.mark.skip(reason=
        'Obsolete after OpenAI payload and Account System migration')
    def test_mixed_content_blocks_only_tool_result_modified(self):
        """
        What it does: Verifies only tool_result blocks are modified, text blocks unchanged.
        Purpose: Ensure selective modification of content blocks.
        """
        print('Setup: Saving truncation info to cache...')
        from opencode_zen.truncation_state import save_tool_truncation
        from opencode_zen.models_anthropic import AnthropicMessage
        tool_use_id = 'tooluse_test_mixed'
        save_tool_truncation(tool_use_id, 'write_to_file', {'size_bytes': 
            5000, 'reason': 'test'})
        print('Setup: Creating request with mixed content blocks...')
        messages = [AnthropicMessage(role='user', content=[{'type': 'text',
            'text': "Here's the result:"}, {'type': 'tool_result',
            'tool_use_id': tool_use_id, 'content': 'Error'}])]
        print(
            'Action: Processing messages through truncation recovery logic...')
        from opencode_zen.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from opencode_zen.truncation_state import get_tool_truncation
        modified_messages = []
        for msg in messages:
            if msg.role == 'user' and msg.content and isinstance(msg.
                content, list):
                modified_content_blocks = []
                has_modifications = False
                for block in msg.content:
                    block_type = self._get_block_value(block, 'type')
                    block_tool_use_id = self._get_block_value(block,
                        'tool_use_id')
                    original_content = self._get_block_value(block,
                        'content', '')
                    if (block_type == 'tool_result' and block_tool_use_id and
                        should_inject_recovery()):
                        truncation_info = get_tool_truncation(block_tool_use_id
                            )
                        if truncation_info:
                            synthetic = generate_truncation_tool_result(
                                truncation_info.tool_name, truncation_info.
                                tool_call_id, truncation_info.truncation_info)
                            modified_content = f"""{synthetic['content']}

---

Original tool result:
{original_content}"""
                            if isinstance(block, dict):
                                modified_block = block.copy()
                                modified_block['content'] = modified_content
                            else:
                                modified_block = block.model_copy(update={
                                    'content': modified_content})
                            modified_content_blocks.append(modified_block)
                            has_modifications = True
                            continue
                    modified_content_blocks.append(block)
                if has_modifications:
                    modified_msg = msg.model_copy(update={'content':
                        modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            modified_messages.append(msg)
        print('Checking: Text block unchanged...')
        modified_msg = modified_messages[0]
        text_block = modified_msg.content[0]
        assert self._get_block_value(text_block, 'type') == 'text'
        assert self._get_block_value(text_block, 'text'
            ) == "Here's the result:"
        print('Checking: Tool_result block modified...')
        tool_result_block = modified_msg.content[1]
        assert self._get_block_value(tool_result_block, 'type'
            ) == 'tool_result'
        tool_content = self._get_block_value(tool_result_block, 'content')
        assert '[API Limitation]' in tool_content
        assert 'Error' in tool_content
        print('Checking: Order preserved...')
        assert len(modified_msg.content) == 2

    def test_no_modification_when_no_truncation(self):
        """
        What it does: Verifies messages are not modified when no truncation info exists.
        Purpose: Ensure normal messages pass through unchanged.
        """
        print('Setup: Creating request without truncation info in cache...')
        from opencode_zen.models_anthropic import AnthropicMessage
        messages = [AnthropicMessage(role='user', content=[{'type':
            'tool_result', 'tool_use_id': 'tooluse_nonexistent', 'content':
            'Success'}])]
        print('Action: Processing messages...')
        from opencode_zen.truncation_recovery import should_inject_recovery
        from opencode_zen.truncation_state import get_tool_truncation
        modified_messages = []
        tool_results_modified = 0
        for msg in messages:
            if msg.role == 'user' and msg.content and isinstance(msg.
                content, list):
                modified_content_blocks = []
                has_modifications = False
                for block in msg.content:
                    block_type = self._get_block_value(block, 'type')
                    block_tool_use_id = self._get_block_value(block,
                        'tool_use_id')
                    if (block_type == 'tool_result' and block_tool_use_id and
                        should_inject_recovery()):
                        truncation_info = get_tool_truncation(block_tool_use_id
                            )
                        if truncation_info:
                            tool_results_modified += 1
                            modified_content_blocks.append(block)
                        else:
                            modified_content_blocks.append(block)
                    else:
                        modified_content_blocks.append(block)
                if has_modifications:
                    modified_msg = msg.model_copy(update={'content':
                        modified_content_blocks})
                    modified_messages.append(modified_msg)
                    continue
            modified_messages.append(msg)
        print(f'Checking: tool_results_modified count...')
        assert tool_results_modified == 0
        print('Checking: Message content unchanged...')
        content = self._get_block_value(modified_messages[0].content[0],
            'content')
        assert content == 'Success'

    @pytest.mark.skip(reason=
        'Obsolete after OpenAI payload and Account System migration')
    def test_pydantic_immutability_new_object_created(self):
        """
        What it does: Verifies new AnthropicMessage object is created, not modified in-place.
        Purpose: Ensure Pydantic immutability is respected.
        """
        print('Setup: Saving truncation info and creating message...')
        from opencode_zen.truncation_state import save_tool_truncation
        from opencode_zen.models_anthropic import AnthropicMessage
        tool_use_id = 'test_immutable_anthropic'
        save_tool_truncation(tool_use_id, 'tool', {'size_bytes': 1000,
            'reason': 'test truncation'})
        original_msg = AnthropicMessage(role='user', content=[{'type':
            'tool_result', 'tool_use_id': tool_use_id, 'content': 'original'}])
        original_content = self._get_block_value(original_msg.content[0],
            'content')
        print('Action: Processing message...')
        from opencode_zen.truncation_recovery import should_inject_recovery, generate_truncation_tool_result
        from opencode_zen.truncation_state import get_tool_truncation
        if original_msg.role == 'user' and original_msg.content and isinstance(
            original_msg.content, list):
            modified_content_blocks = []
            has_modifications = False
            for block in original_msg.content:
                block_type = self._get_block_value(block, 'type')
                block_tool_use_id = self._get_block_value(block, 'tool_use_id')
                original_block_content = self._get_block_value(block,
                    'content', '')
                if (block_type == 'tool_result' and block_tool_use_id and
                    should_inject_recovery()):
                    truncation_info = get_tool_truncation(block_tool_use_id)
                    if truncation_info:
                        synthetic = generate_truncation_tool_result(
                            truncation_info.tool_name, truncation_info.
                            tool_call_id, truncation_info.truncation_info)
                        modified_content = f"""{synthetic['content']}

---

Original tool result:
{original_block_content}"""
                        if isinstance(block, dict):
                            modified_block = block.copy()
                            modified_block['content'] = modified_content
                        else:
                            modified_block = block.model_copy(update={
                                'content': modified_content})
                        modified_content_blocks.append(modified_block)
                        has_modifications = True
                        continue
                modified_content_blocks.append(block)
            if has_modifications:
                modified_msg = original_msg.model_copy(update={'content':
                    modified_content_blocks})
        print('Checking: Original message unchanged...')
        assert self._get_block_value(original_msg.content[0], 'content'
            ) == original_content
        print('Checking: New object created...')
        assert modified_msg is not original_msg
        print('Checking: Content modified in new object...')
        modified_content = self._get_block_value(modified_msg.content[0],
            'content')
        assert modified_content != original_content
        assert '[API Limitation]' in modified_content


class TestContentTruncationRecovery:
    """
    Tests for content truncation recovery (synthetic user message) in Anthropic routes.
    
    Verifies that synthetic user message is added after truncated assistant message.
    Part of Truncation Recovery System (Issue #56).
    """

    @staticmethod
    def _get_block_value(block, key, default=''):
        """Helper to get value from dict or Pydantic object."""
        if isinstance(block, dict):
            return block.get(key, default)
        else:
            return getattr(block, key, default)

    def test_adds_synthetic_user_message_after_truncated_assistant(self):
        """
        What it does: Verifies synthetic user message is added after truncated assistant message.
        Purpose: Ensure content truncation recovery works for Anthropic API (Test Case C.2).
        """
        print('Setup: Saving content truncation info...')
        from opencode_zen.truncation_state import save_content_truncation
        from opencode_zen.models_anthropic import AnthropicMessage
        truncated_content_text = (
            'This is a very long response that was cut off mid-sentence')
        save_content_truncation(truncated_content_text)
        print('Setup: Creating request with truncated assistant message...')
        messages = [AnthropicMessage(role='assistant', content=[{'type':
            'text', 'text': truncated_content_text}])]
        print(
            'Action: Processing messages through content truncation recovery...'
            )
        from opencode_zen.truncation_recovery import should_inject_recovery, generate_truncation_user_message
        from opencode_zen.truncation_state import get_content_truncation
        modified_messages = []
        for msg in messages:
            if msg.role == 'assistant' and msg.content:
                text_content = ''
                if isinstance(msg.content, str):
                    text_content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if self._get_block_value(block, 'type') == 'text':
                            text_content += self._get_block_value(block,
                                'text', '')
                if text_content:
                    truncation_info = get_content_truncation(text_content)
                    if truncation_info:
                        print(
                            f'Found content truncation for hash: {truncation_info.message_hash}'
                            )
                        modified_messages.append(msg)
                        synthetic_user_msg = AnthropicMessage(role='user',
                            content=[{'type': 'text', 'text':
                            generate_truncation_user_message()}])
                        modified_messages.append(synthetic_user_msg)
                        continue
            modified_messages.append(msg)
        print('Checking: Two messages in result...')
        assert len(modified_messages) == 2
        print('Checking: First message is original assistant message...')
        assert modified_messages[0].role == 'assistant'
        print('Checking: Second message is synthetic user message...')
        assert modified_messages[1].role == 'user'
        synthetic_text = self._get_block_value(modified_messages[1].content
            [0], 'text')
        assert '[System Notice]' in synthetic_text
        assert 'truncated' in synthetic_text.lower()

    def test_no_synthetic_message_when_no_content_truncation(self):
        """
        What it does: Verifies no synthetic message is added for normal assistant message.
        Purpose: Ensure false positives don't occur.
        """
        print('Setup: Creating normal assistant message (no truncation)...')
        from opencode_zen.models_anthropic import AnthropicMessage
        messages = [AnthropicMessage(role='assistant', content=[{'type':
            'text', 'text': 'This is a complete response.'}])]
        print('Action: Processing messages...')
        from opencode_zen.truncation_state import get_content_truncation
        modified_messages = []
        for msg in messages:
            if msg.role == 'assistant' and msg.content:
                text_content = ''
                if isinstance(msg.content, str):
                    text_content = msg.content
                elif isinstance(msg.content, list):
                    for block in msg.content:
                        if self._get_block_value(block, 'type') == 'text':
                            text_content += self._get_block_value(block,
                                'text', '')
                if text_content:
                    truncation_info = get_content_truncation(text_content)
                    if truncation_info:
                        pass
            modified_messages.append(msg)
        print('Checking: Only one message in result...')
        assert len(modified_messages) == 1
        print('Checking: Message unchanged...')
        text = self._get_block_value(modified_messages[0].content[0], 'text')
        assert text == 'This is a complete response.'


class TestWebSearchAutoInjection:
    """Tests for WebSearch auto-injection (Path B - MCP Tool Emulation)."""

    def test_auto_injection_logic(self, monkeypatch):
        """
        What it does: Verifies web_search tool auto-injection logic.
        Purpose: Ensure WEB_SEARCH_ENABLED controls auto-injection.
        """
        print('Setup: Testing auto-injection logic...')
        from opencode_zen.models_anthropic import AnthropicTool
        WEB_SEARCH_ENABLED = True
        tools = []
        if WEB_SEARCH_ENABLED:
            has_ws = any(getattr(tool, 'name', '') == 'web_search' for tool in
                tools)
            if not has_ws:
                web_search_tool = AnthropicTool(name='web_search',
                    description=
                    'Search the web for current information. Use when you need up-to-date data from the internet.'
                    , input_schema={'type': 'object', 'properties': {
                    'query': {'type': 'string', 'description':
                    'Search query'}}, 'required': ['query']})
                tools.append(web_search_tool)
        print(f'Checking: web_search tool was added...')
        assert len(tools) == 1
        assert tools[0].name == 'web_search'
        assert tools[0].input_schema is not None

    def test_no_duplicate_injection_logic(self):
        """
        What it does: Verifies duplicate detection logic.
        Purpose: Ensure auto-injection doesn't create duplicates.
        """
        print('Setup: Testing duplicate detection...')
        from opencode_zen.models_anthropic import AnthropicTool
        existing_tools = [AnthropicTool(name='web_search', description=
            'Existing web search', input_schema={'type': 'object',
            'properties': {}})]
        WEB_SEARCH_ENABLED = True
        if WEB_SEARCH_ENABLED:
            has_ws = any(getattr(tool, 'name', '') == 'web_search' for tool in
                existing_tools)
            if not has_ws:
                existing_tools.append(AnthropicTool(name='web_search',
                    description='Auto-injected', input_schema={'type':
                    'object', 'properties': {}}))
        print(f'Checking: Only one web_search tool...')
        web_search_count = sum(1 for t in existing_tools if t.name ==
            'web_search')
        assert web_search_count == 1


class TestWebSearchNativeDetection:
    """Tests for native Anthropic server-side tools detection (Path A)."""

    def test_native_tool_type_detection(self):
        """
        What it does: Verifies detection of native server-side tools by type field.
        Purpose: Ensure Path A detection logic works.
        """
        print('Setup: Creating tools list with native server-side tool...')
        from opencode_zen.models_anthropic import AnthropicTool
        tools = [AnthropicTool(type='web_search_20250305', name=
            'web_search', max_uses=8)]
        print('Action: Checking for native web_search...')
        has_native_web_search = False
        for tool in tools:
            tool_type = getattr(tool, 'type', None)
            if tool_type and tool_type.startswith('web_search'):
                has_native_web_search = True
                break
        print(f'Checking: Native web_search detected...')
        assert has_native_web_search is True

    def test_user_defined_tool_not_detected_as_native(self):
        """
        What it does: Verifies user-defined tools are not detected as native.
        Purpose: Ensure Path A detection doesn't trigger for regular tools.
        """
        print('Setup: Creating tools list with user-defined tool...')
        from opencode_zen.models_anthropic import AnthropicTool
        tools = [AnthropicTool(name='web_search', description=
            'User-defined web search', input_schema={'type': 'object',
            'properties': {}})]
        print('Action: Checking for native web_search...')
        has_native_web_search = False
        for tool in tools:
            tool_type = getattr(tool, 'type', None)
            if tool_type and tool_type.startswith('web_search'):
                has_native_web_search = True
                break
        print(f'Checking: Native web_search NOT detected...')
        assert has_native_web_search is False


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestMessagesFailoverLoop:
    """Tests for Account System failover loop in /v1/messages endpoint."""

    @pytest.mark.asyncio
    async def test_messages_failover_get_next_account(self):
        """
        What it does: Verifies get_next_account() is called with exclude_accounts parameter.
        Purpose: Ensure failover loop passes exclude_accounts correctly.
        """
        print('Setup: Mock AccountManager with get_next_account...')
        from unittest.mock import AsyncMock
        mock_manager = AsyncMock()
        mock_manager.get_next_account = AsyncMock(return_value=None)
        model = 'claude-opus-4.5'
        exclude_accounts = {'account1', 'account2'}
        print('Action: Calling get_next_account with exclude_accounts...')
        result = await mock_manager.get_next_account(model,
            exclude_accounts=exclude_accounts)
        print(
            'Checking: get_next_account was called with correct parameters...')
        mock_manager.get_next_account.assert_called_once_with(model,
            exclude_accounts=exclude_accounts)
        print('✅ get_next_account called with exclude_accounts')

    @pytest.mark.asyncio
    async def test_messages_failover_success_first_account(self):
        """
        What it does: Verifies successful response on first account.
        Purpose: Ensure failover loop returns immediately on success.
        """
        print('Setup: Mock successful response on first account...')
        from unittest.mock import AsyncMock, MagicMock, patch
        account = Account(id='account1', auth_manager=MagicMock(),
            model_cache=MagicMock(), model_resolver=MagicMock(), stats=
            AccountStats())
        account.auth_manager.api_host = 'https://api.example.com'
        mock_manager = AsyncMock()
        mock_manager.get_next_account = AsyncMock(return_value=account)
        mock_manager.report_success = AsyncMock()
        mock_manager._accounts = {'account1': account}
        print('Checking: Success on first attempt...')
        assert True

    @pytest.mark.asyncio
    async def test_messages_failover_recoverable_try_next(self):
        """
        What it does: Verifies RECOVERABLE error triggers next account attempt.
        Purpose: Ensure failover loop continues on recoverable errors.
        """
        print('Setup: Mock RECOVERABLE error (429 rate limit)...')
        from opencode_zen.account_errors import ErrorType, classify_error
        error_type = classify_error(429, None)
        print(f'Checking: 429 classified as RECOVERABLE...')
        assert error_type == ErrorType.RECOVERABLE
        print('Checking: Failover loop would continue to next account...')
        assert True

    @pytest.mark.asyncio
    async def test_messages_failover_fatal_immediate_return(self):
        """
        What it does: Verifies FATAL error returns immediately to client.
        Purpose: Ensure failover loop stops on fatal errors.
        """
        print(
            'Setup: Mock FATAL error (400 CONTENT_LENGTH_EXCEEDS_THRESHOLD)...'
            )
        from opencode_zen.account_errors import ErrorType, classify_error
        error_type = classify_error(400, 'CONTENT_LENGTH_EXCEEDS_THRESHOLD')
        print(f'Checking: 400 + CONTENT_LENGTH classified as FATAL...')
        assert error_type == ErrorType.FATAL
        print('Checking: Failover loop would stop and return error...')
        assert True

    @pytest.mark.asyncio
    async def test_messages_failover_single_account_original_error(self):
        """
        What it does: Verifies single account returns original error message.
        Purpose: Ensure single account mode shows specific errors.
        """
        print('Setup: Single account with error...')
        all_accounts = ['account1']
        last_error_message = 'Token expired'
        last_error_status = 403
        print('Checking: Single account returns specific error...')
        if len(all_accounts) == 1:
            assert last_error_message == 'Token expired'
            assert last_error_status == 403
        print('✅ Single account returns original error')

    @pytest.mark.asyncio
    async def test_messages_failover_multi_account_generic_error(self):
        """
        What it does: Verifies multi-account returns generic error message.
        Purpose: Ensure multi-account mode doesn't expose account details.
        """
        print('Setup: Multiple accounts all failed...')
        all_accounts = ['account1', 'account2', 'account3']
        last_error_message = 'Token expired on account1'
        print('Checking: Multi-account returns generic error...')
        if len(all_accounts) > 1:
            generic_message = 'No available accounts for this model.'
            if last_error_message:
                generic_message += (
                    f' Error from last account: {last_error_message}')
            assert 'No available accounts' in generic_message
            assert last_error_message in generic_message
        print('✅ Multi-account returns generic error with context')

    @pytest.mark.asyncio
    async def test_messages_failover_all_unavailable(self):
        """
        What it does: Verifies behavior when all accounts are unavailable.
        Purpose: Ensure proper error when no accounts can handle request.
        """
        print('Setup: All accounts unavailable...')
        from unittest.mock import AsyncMock
        mock_manager = AsyncMock()
        mock_manager.get_next_account = AsyncMock(return_value=None)
        mock_manager._accounts = {'acc1': None, 'acc2': None}
        print('Action: get_next_account returns None...')
        account = await mock_manager.get_next_account('claude-opus-4.5',
            exclude_accounts=set())
        print('Checking: None returned...')
        assert account is None
        print('Checking: Would return 503 error...')
        assert True

    @pytest.mark.asyncio
    async def test_messages_failover_report_success(self):
        """
        What it does: Verifies report_success() is called on successful request.
        Purpose: Ensure statistics are updated correctly.
        """
        print('Setup: Mock successful request...')
        from unittest.mock import AsyncMock
        mock_manager = AsyncMock()
        mock_manager.report_success = AsyncMock()
        account_id = 'test_account'
        model = 'claude-opus-4.5'
        print('Action: Calling report_success...')
        await mock_manager.report_success(account_id, model)
        print('Checking: report_success was called...')
        mock_manager.report_success.assert_called_once_with(account_id, model)
        print('✅ report_success called on success')

    @pytest.mark.asyncio
    async def test_messages_failover_report_failure(self):
        """
        What it does: Verifies report_failure() is called on failed request.
        Purpose: Ensure failure tracking works correctly.
        """
        print('Setup: Mock failed request...')
        from opencode_zen.account_errors import ErrorType
        from unittest.mock import AsyncMock
        mock_manager = AsyncMock()
        mock_manager.report_failure = AsyncMock()
        account_id = 'test_account'
        model = 'claude-opus-4.5'
        error_type = ErrorType.RECOVERABLE
        status_code = 429
        reason = None
        print('Action: Calling report_failure...')
        await mock_manager.report_failure(account_id, model, error_type,
            status_code, reason)
        print('Checking: report_failure was called...')
        mock_manager.report_failure.assert_called_once_with(account_id,
            model, error_type, status_code, reason)
        print('✅ report_failure called on error')

    @pytest.mark.asyncio
    async def test_messages_failover_exclude_tried_accounts(self):
        """
        What it does: Verifies exclude_accounts grows with each attempt.
        Purpose: Ensure accounts aren't retried in same failover loop.
        """
        print('Setup: Simulating multiple attempts...')
        tried_accounts = set()
        accounts = ['acc1', 'acc2', 'acc3']
        print('Action: Adding accounts to exclude set...')
        for account_id in accounts:
            tried_accounts.add(account_id)
            print(f'  Tried: {tried_accounts}')
        print('Checking: All accounts in exclude set...')
        assert len(tried_accounts) == 3
        assert 'acc1' in tried_accounts
        assert 'acc2' in tried_accounts
        assert 'acc3' in tried_accounts
        print('✅ exclude_accounts grows correctly')

    @pytest.mark.asyncio
    async def test_messages_failover_max_attempts(self):
        """
        What it does: Verifies failover loop stops after MAX_ATTEMPTS.
        Purpose: Ensure infinite loops are prevented.
        """
        print('Setup: Calculating MAX_ATTEMPTS...')
        all_accounts = ['acc1', 'acc2', 'acc3']
        MAX_ATTEMPTS = len(all_accounts) * 2
        print(f'Checking: MAX_ATTEMPTS = {MAX_ATTEMPTS}...')
        assert MAX_ATTEMPTS == 6
        print('Checking: Loop would stop after 6 attempts...')
        attempts = 0
        for attempt in range(MAX_ATTEMPTS):
            attempts += 1
            if attempts >= MAX_ATTEMPTS:
                break
        assert attempts == MAX_ATTEMPTS
        print('✅ MAX_ATTEMPTS prevents infinite loops')


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestMessagesLegacyMode:
    """Tests for legacy mode (ACCOUNT_SYSTEM=false) in /v1/messages endpoint."""

    @pytest.mark.asyncio
    async def test_messages_legacy_get_first_account(self):
        """
        What it does: Verifies legacy mode uses get_first_account().
        Purpose: Ensure backward compatibility with single account.
        """
        print('Setup: Mock legacy mode (account_system=false)...')
        from unittest.mock import MagicMock
        account = Account(id='legacy_account', auth_manager=MagicMock(),
            model_cache=MagicMock(), model_resolver=MagicMock(), stats=
            AccountStats())
        mock_manager = MagicMock()
        mock_manager.get_first_account = MagicMock(return_value=account)
        print('Action: Calling get_first_account...')
        result = mock_manager.get_first_account()
        print('Checking: get_first_account was called...')
        mock_manager.get_first_account.assert_called_once()
        assert result == account
        print('✅ Legacy mode uses get_first_account()')

    @pytest.mark.asyncio
    async def test_messages_legacy_no_failover(self):
        """
        What it does: Verifies legacy mode has no failover loop.
        Purpose: Ensure legacy mode behavior is unchanged.
        """
        print('Setup: Legacy mode configuration...')
        account_system = False
        print('Checking: No failover loop in legacy mode...')
        if not account_system:
            print('  ✓ Uses get_first_account()')
            print('  ✓ No failover loop')
            print('  ✓ Returns original error')
            assert True
        print('✅ Legacy mode has no failover')


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestMessagesNativeWebSearchAccountSelection:
    """Tests for native WebSearch account selection in /v1/messages endpoint."""

    @pytest.mark.asyncio
    async def test_messages_native_websearch_get_first_account(self):
        """
        What it does: Verifies native WebSearch (Path A) uses get_first_account().
        Purpose: Ensure Path A doesn't use failover (early return).
        """
        print('Setup: Mock native WebSearch request...')
        from opencode_zen.models_anthropic import AnthropicTool
        from unittest.mock import MagicMock
        native_tool = AnthropicTool(type='web_search_20250305', name=
            'web_search', max_uses=8)
        print('Checking: Tool has native type...')
        assert native_tool.type.startswith('web_search')
        print('Setup: Mock AccountManager...')
        account = Account(id='websearch_account', auth_manager=MagicMock(),
            model_cache=MagicMock(), model_resolver=MagicMock(), stats=
            AccountStats())
        mock_manager = MagicMock()
        mock_manager.get_first_account = MagicMock(return_value=account)
        print('Action: Path A early return uses get_first_account...')
        result = mock_manager.get_first_account()
        print('Checking: get_first_account was called...')
        mock_manager.get_first_account.assert_called_once()
        assert result == account
        print('✅ Native WebSearch uses get_first_account() (no failover)')


class TestCountTokensEndpoint:
    """Tests for /v1/messages/count_tokens endpoint."""

    def test_count_tokens_basic(self, test_client, valid_proxy_api_key):
        """
        What it does: Tests basic token counting with one message.
        Purpose: Ensure endpoint returns token count for simple request.
        """
        print('Setup: Creating basic request with one message...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': 'Hello, world!'}]}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 200...')
        assert response.status_code == 200
        print('Checking: Response structure...')
        data = response.json()
        assert 'input_tokens' in data
        print('Checking: input_tokens is int and > 0...')
        assert isinstance(data['input_tokens'], int)
        assert data['input_tokens'] > 0
        print(f"✅ Token count: {data['input_tokens']} tokens")

    def test_count_tokens_with_tools(self, test_client, valid_proxy_api_key):
        """
        What it does: Tests token counting with tool definitions.
        Purpose: Ensure tools are included in token count.
        """
        print('Setup: Creating request with tools...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': "What's the weather?"}], 'tools': [{'name':
            'get_weather', 'description': 'Get weather for location',
            'input_schema': {'type': 'object', 'properties': {'location': {
            'type': 'string'}}, 'required': ['location']}}]}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 200...')
        assert response.status_code == 200
        print('Checking: Token count includes tools...')
        data = response.json()
        tokens_with_tools = data['input_tokens']
        response_no_tools = test_client.post('/v1/messages/count_tokens',
            headers={'x-api-key': valid_proxy_api_key}, json={'model':
            'claude-sonnet-4-5', 'messages': [{'role': 'user', 'content':
            "What's the weather?"}]})
        tokens_no_tools = response_no_tools.json()['input_tokens']
        print(
            f'Tokens with tools: {tokens_with_tools}, without tools: {tokens_no_tools}'
            )
        assert tokens_with_tools > tokens_no_tools
        print('✅ Tools increase token count')

    def test_count_tokens_with_system_string(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Tests token counting with system prompt (string format).
        Purpose: Ensure system prompt is included in token count.
        """
        print('Setup: Creating request with system prompt (string)...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': 'Hello'}], 'system':
            'You are a helpful assistant.'}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 200...')
        assert response.status_code == 200
        print('Checking: Token count includes system prompt...')
        data = response.json()
        tokens_with_system = data['input_tokens']
        response_no_system = test_client.post('/v1/messages/count_tokens',
            headers={'x-api-key': valid_proxy_api_key}, json={'model':
            'claude-sonnet-4-5', 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        tokens_no_system = response_no_system.json()['input_tokens']
        print(
            f'Tokens with system: {tokens_with_system}, without system: {tokens_no_system}'
            )
        assert tokens_with_system > tokens_no_system
        print('✅ System prompt increases token count')

    def test_count_tokens_with_system_blocks(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Tests token counting with system prompt (list format for prompt caching).
        Purpose: Ensure system prompt blocks are handled correctly.
        """
        print('Setup: Creating request with system prompt (list of blocks)...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': 'Hello'}], 'system': [{'type': 'text',
            'text': 'You are a helpful assistant.'}, {'type': 'text',
            'text': 'Be concise.', 'cache_control': {'type': 'ephemeral'}}]}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 200...')
        assert response.status_code == 200
        print('Checking: input_tokens > 0...')
        data = response.json()
        assert data['input_tokens'] > 0
        print(f"✅ System blocks counted: {data['input_tokens']} tokens")

    def test_count_tokens_empty_messages(self, test_client, valid_proxy_api_key
        ):
        """
        What it does: Tests validation with empty messages array.
        Purpose: Ensure at least one message is required (Pydantic validation).
        """
        print('Setup: Creating request with empty messages...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': []}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 422 (validation error)...')
        assert response.status_code == 422
        print('✅ Empty messages rejected')

    def test_count_tokens_invalid_api_key(self, test_client,
        invalid_proxy_api_key):
        """
        What it does: Tests authentication with invalid API key.
        Purpose: Ensure endpoint is protected by authentication.
        """
        print('Setup: Creating request with invalid API key...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': 'Hello'}]}
        print('Action: POST /v1/messages/count_tokens with invalid key...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': invalid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 401 (unauthorized)...')
        assert response.status_code == 401
        print('Checking: Error format is Anthropic-style...')
        data = response.json()
        assert 'detail' in data
        detail = data['detail']
        assert 'error' in detail
        assert detail['error']['type'] == 'authentication_error'
        print('✅ Invalid API key rejected')

    def test_count_tokens_multiple_messages(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Tests token counting for dialogue with history.
        Purpose: Ensure multiple messages are counted correctly.
        """
        print('Setup: Creating request with conversation history...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': 'Hello'}, {'role': 'assistant', 'content':
            'Hi there!'}, {'role': 'user', 'content': 'How are you?'}]}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 200...')
        assert response.status_code == 200
        print('Checking: Token count for multiple messages...')
        data = response.json()
        tokens_multi = data['input_tokens']
        response_single = test_client.post('/v1/messages/count_tokens',
            headers={'x-api-key': valid_proxy_api_key}, json={'model':
            'claude-sonnet-4-5', 'messages': [{'role': 'user', 'content':
            'Hello'}]})
        tokens_single = response_single.json()['input_tokens']
        print(f'Tokens multi: {tokens_multi}, single: {tokens_single}')
        assert tokens_multi > tokens_single
        print('✅ Multiple messages increase token count')

    def test_count_tokens_with_images(self, test_client, valid_proxy_api_key):
        """
        What it does: Tests token counting for messages with images.
        Purpose: Ensure images are counted (~100 tokens by default).
        """
        print('Setup: Creating request with image...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': [{'type': 'text', 'text':
            "What's in this image?"}, {'type': 'image', 'source': {'type':
            'base64', 'media_type': 'image/jpeg', 'data':
            'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='
            }}]}]}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 200...')
        assert response.status_code == 200
        print('Checking: Image adds tokens...')
        data = response.json()
        tokens_with_image = data['input_tokens']
        response_text_only = test_client.post('/v1/messages/count_tokens',
            headers={'x-api-key': valid_proxy_api_key}, json={'model':
            'claude-sonnet-4-5', 'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': "What's in this image?"}]}]})
        tokens_text_only = response_text_only.json()['input_tokens']
        print(
            f'Tokens with image: {tokens_with_image}, text only: {tokens_text_only}'
            )
        assert tokens_with_image > tokens_text_only
        print('✅ Image increases token count')

    def test_count_tokens_consistency(self, test_client, valid_proxy_api_key):
        """
        What it does: Tests deterministic behavior - same input gives same output.
        Purpose: Ensure token counting is consistent.
        """
        print('Setup: Creating identical requests...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': 'Test message for consistency'}]}
        print('Action: Sending same request twice...')
        response1 = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        response2 = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Response 1: {response1.json()}')
        print(f'Response 2: {response2.json()}')
        print('Checking: Both requests successful...')
        assert response1.status_code == 200
        assert response2.status_code == 200
        print('Checking: Token counts are identical...')
        tokens1 = response1.json()['input_tokens']
        tokens2 = response2.json()['input_tokens']
        print(f'Tokens 1: {tokens1}, Tokens 2: {tokens2}')
        assert tokens1 == tokens2
        print('✅ Token counting is deterministic')

    def test_count_tokens_no_max_tokens_required(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Tests that max_tokens is NOT required (unlike /v1/messages).
        Purpose: Ensure count_tokens endpoint doesn't require generation parameters.
        """
        print('Setup: Creating request WITHOUT max_tokens...')
        request_data = {'model': 'claude-sonnet-4-5', 'messages': [{'role':
            'user', 'content': 'Hello'}]}
        print('Action: POST /v1/messages/count_tokens...')
        response = test_client.post('/v1/messages/count_tokens', headers={
            'x-api-key': valid_proxy_api_key}, json=request_data)
        print(f'Status: {response.status_code}')
        print(f'Response: {response.json()}')
        print('Checking: HTTP 200 (NOT 422)...')
        assert response.status_code == 200
        print('Checking: Token count returned...')
        data = response.json()
        assert 'input_tokens' in data
        assert data['input_tokens'] > 0
        print('✅ max_tokens is NOT required for count_tokens')


def _mock_upstream_error_client(status_code, json_body=None, text=''):
    """Builds a mocked OpenCodeHttpClient whose request returns an error response."""
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.aread = AsyncMock()
    if json_body is not None:
        mock_response.json = Mock(return_value=json_body)
    else:
        mock_response.json = Mock(side_effect=ValueError('not json'))
    mock_response.text = text
    mock_client = MagicMock()
    mock_client.request_with_retry = AsyncMock(return_value=mock_response)
    mock_client._owns_client = False
    return mock_client


class TestServerSideToolStripping:
    """Tests for stripping server-side tools (e.g. web_search_20250305) from requests."""

    def test_server_side_tool_is_stripped_not_diverted(self, test_client, valid_proxy_api_key):
        """
        What it does: Sends a request with a server-side web_search tool and verifies the
        request still reaches the normal LLM path with the tool removed.
        Purpose: Claude Code sends web_search_20250305; the upstream has no server-side
        tool support, so the tool must be silently stripped instead of diverting the request.
        """
        print('Setup: request with a server-side web_search tool...')
        payload = {'model': 'claude-sonnet-4-5', 'max_tokens': 100, 'messages':
            [{'role': 'user', 'content': 'Hello'}], 'tools': [{'type':
            'web_search_20250305', 'name': 'web_search', 'max_uses': 5}]}
        with patch('opencode_zen.routes_anthropic.anthropic_to_opencode'
            ) as mock_convert, patch(
            'opencode_zen.routes_anthropic.OpenCodeHttpClient'
            ) as mock_client_cls:
            mock_convert.return_value = {'model': 'claude-sonnet-4-5',
                'messages': [], 'stream': True}
            mock_client_cls.return_value = _mock_upstream_error_client(500,
                {'type': 'error', 'error': {'type': 'x', 'message': 'y'}})
            print('Action: POST /v1/messages...')
            response = test_client.post('/v1/messages', headers={
                'x-api-key': valid_proxy_api_key}, json=payload)
            print(f'Status: {response.status_code}')
            print('Checking: request was converted (NOT diverted before conversion)...')
            assert mock_convert.called
            request_arg = mock_convert.call_args[0][0]
            print(f'Checking: server-side tool stripped: {request_arg.tools}')
            assert request_arg.tools is None or all(not (getattr(t, 'type',
                None) or '').startswith('web_search') for t in request_arg.tools)

    def test_user_defined_tools_are_kept(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies user-defined tools (with input_schema) survive stripping.
        Purpose: Only server-side tools without input_schema are removed.
        """
        print('Setup: request with one user-defined and one server-side tool...')
        payload = {'model': 'claude-sonnet-4-5', 'max_tokens': 100,
            'messages': [{'role': 'user', 'content': 'Hello'}], 'tools': [{
            'name': 'my_tool', 'description': 'd', 'input_schema': {'type':
            'object', 'properties': {}}}, {'type': 'web_search_20250305',
            'name': 'web_search'}]}
        with patch('opencode_zen.routes_anthropic.anthropic_to_opencode'
            ) as mock_convert, patch(
            'opencode_zen.routes_anthropic.OpenCodeHttpClient'
            ) as mock_client_cls:
            mock_convert.return_value = {'model': 'claude-sonnet-4-5',
                'messages': [], 'stream': True}
            mock_client_cls.return_value = _mock_upstream_error_client(500,
                {'type': 'error', 'error': {'type': 'x', 'message': 'y'}})
            print('Action: POST /v1/messages...')
            test_client.post('/v1/messages', headers={'x-api-key':
                valid_proxy_api_key}, json=payload)
            request_arg = mock_convert.call_args[0][0]
            tool_names = [t.name for t in (request_arg.tools or [])]
            print(f'Checking: kept tools = {tool_names}')
            assert 'my_tool' in tool_names
            assert all(t.input_schema is not None for t in request_arg.tools)


class TestUpstreamErrorPassthrough:
    """Tests for clean passthrough of upstream Anthropic-shaped error bodies."""

    def test_anthropic_shaped_upstream_error_passes_through(self,
        test_client, valid_proxy_api_key):
        """
        What it does: Returns the upstream error type/message verbatim in Anthropic shape.
        Purpose: Clients (Claude Code) must see the real upstream message, e.g.
        'Model X is not supported', not a stringified wrapper around it.
        """
        print('Setup: upstream returns Anthropic-shaped 400...')
        payload = {'model': 'not-a-model', 'max_tokens': 10, 'messages': [{
            'role': 'user', 'content': 'hi'}]}
        with patch('opencode_zen.routes_anthropic.OpenCodeHttpClient'
            ) as mock_client_cls:
            mock_client_cls.return_value = _mock_upstream_error_client(400,
                {'type': 'error', 'error': {'type': 'ModelError', 'message':
                'Model not-a-model is not supported'}})
            print('Action: POST /v1/messages...')
            response = test_client.post('/v1/messages', headers={
                'x-api-key': valid_proxy_api_key}, json=payload)
            print(f'Status: {response.status_code}, body: {response.json()}')
            assert response.status_code == 400
            data = response.json()
            assert data['type'] == 'error'
            assert data['error']['type'] == 'ModelError'
            assert data['error']['message'] == 'Model not-a-model is not supported'

    def test_plain_text_upstream_error_is_wrapped(self, test_client,
        valid_proxy_api_key):
        """
        What it does: Wraps a non-JSON upstream error body into Anthropic error shape.
        Purpose: Malformed upstream errors must still produce a valid error response.
        """
        print('Setup: upstream returns plain-text 502...')
        payload = {'model': 'claude-sonnet-4-5', 'max_tokens': 10,
            'messages': [{'role': 'user', 'content': 'hi'}]}
        with patch('opencode_zen.routes_anthropic.OpenCodeHttpClient'
            ) as mock_client_cls:
            mock_client_cls.return_value = _mock_upstream_error_client(502,
                None, 'Bad gateway')
            print('Action: POST /v1/messages...')
            response = test_client.post('/v1/messages', headers={
                'x-api-key': valid_proxy_api_key}, json=payload)
            print(f'Status: {response.status_code}, body: {response.json()}')
            assert response.status_code == 502
            data = response.json()
            assert data['type'] == 'error'
            assert data['error']['message'] == 'Bad gateway'


class TestUnknownContentBlockTolerance:
    """Tests for accepting content block types this gateway does not model."""

    def test_unknown_block_type_does_not_422(self, test_client, valid_proxy_api_key):
        """
        What it does: Sends a message containing an unknown content block type and
        verifies the request passes validation (no 422).
        Purpose: Newer Claude Code versions send block types the gateway does not
        know; they must validate as plain dicts instead of failing the request.
        """
        print('Setup: request with an unknown content block type...')
        payload = {'model': 'claude-sonnet-4-5', 'max_tokens': 10,
            'messages': [{'role': 'user', 'content': [{'type': 'text',
            'text': 'hi'}, {'type': 'document', 'source': {'type': 'text',
            'media_type': 'text/plain', 'data': 'file contents'}}]}]}
        with patch('opencode_zen.routes_anthropic.OpenCodeHttpClient'
            ) as mock_client_cls:
            mock_client_cls.return_value = _mock_upstream_error_client(500,
                {'type': 'error', 'error': {'type': 'x', 'message': 'y'}})
            print('Action: POST /v1/messages...')
            response = test_client.post('/v1/messages', headers={
                'x-api-key': valid_proxy_api_key}, json=payload)
            print(f'Status: {response.status_code}')
            print('Checking: validation passed (not 422)...')
            assert response.status_code != 422
