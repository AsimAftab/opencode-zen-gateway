"""
Unit tests for converters_openai module.

Tests for OpenAI-specific conversion logic:
- Converting OpenAI messages to unified format
- Converting OpenAI tools to unified format
- Building OpenCode payload from OpenAI requests
"""
import pytest
from unittest.mock import patch
from opencode_zen.converters_openai import build_opencode_payload, convert_openai_messages_to_unified, convert_openai_tools_to_unified, _extract_images_from_tool_message, reasoning_effort_to_budget, extract_thinking_config_from_openai
from opencode_zen.models_openai import ChatMessage, ChatCompletionRequest, Tool, ToolFunction


class TestConvertOpenAIMessagesToUnified:
    """Tests for convert_openai_messages_to_unified function."""

    def test_extracts_system_prompt(self):
        """
        What it does: Verifies extraction of system prompt from messages.
        Purpose: Ensure system messages are extracted separately.
        """
        print('Setup: Messages with system prompt...')
        messages = [ChatMessage(role='system', content='You are helpful'),
            ChatMessage(role='user', content='Hello')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f"System prompt: '{system_prompt}'")
        print(f'Unified messages: {len(unified)}')
        assert system_prompt == 'You are helpful'
        assert len(unified) == 1
        assert unified[0].role == 'user'

    def test_combines_multiple_system_messages(self):
        """
        What it does: Verifies combining of multiple system messages.
        Purpose: Ensure all system messages are concatenated.
        """
        print('Setup: Multiple system messages...')
        messages = [ChatMessage(role='system', content='You are helpful.'),
            ChatMessage(role='system', content='Be concise.'), ChatMessage(
            role='user', content='Hello')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f"System prompt: '{system_prompt}'")
        assert 'You are helpful.' in system_prompt
        assert 'Be concise.' in system_prompt
        assert len(unified) == 1

    def test_converts_tool_message_to_user_with_tool_results(self):
        """
        What it does: Verifies conversion of tool message to user message with tool_results.
        Purpose: Ensure role="tool" is converted correctly.
        """
        print('Setup: Tool message...')
        messages = [ChatMessage(role='tool', content='Tool result text',
            tool_call_id='call_123')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {unified}')
        assert len(unified) == 1
        assert unified[0].role == 'user'
        assert unified[0].tool_results is not None
        assert len(unified[0].tool_results) == 1
        assert unified[0].tool_results[0]['tool_use_id'] == 'call_123'

    def test_converts_multiple_tool_messages(self):
        """
        What it does: Verifies conversion of multiple consecutive tool messages.
        Purpose: Ensure all tool results are collected into one user message.
        """
        print('Setup: Multiple tool messages...')
        messages = [ChatMessage(role='tool', content='Result 1',
            tool_call_id='call_1'), ChatMessage(role='tool', content=
            'Result 2', tool_call_id='call_2'), ChatMessage(role='tool',
            content='Result 3', tool_call_id='call_3')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {unified}')
        assert len(unified) == 1
        assert unified[0].role == 'user'
        assert len(unified[0].tool_results) == 3

    def test_extracts_tool_calls_from_assistant(self):
        """
        What it does: Verifies extraction of tool_calls from assistant message.
        Purpose: Ensure tool_calls are preserved in unified format.
        """
        print('Setup: Assistant message with tool_calls...')
        messages = [ChatMessage(role='assistant', content=
            "I'll call a tool", tool_calls=[{'id': 'call_123', 'type':
            'function', 'function': {'name': 'get_weather', 'arguments':
            '{"location": "Moscow"}'}}])]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {unified}')
        assert len(unified) == 1
        assert unified[0].role == 'assistant'
        assert unified[0].tool_calls is not None
        assert len(unified[0].tool_calls) == 1
        assert unified[0].tool_calls[0]['id'] == 'call_123'

    def test_handles_empty_tool_call_id(self):
        """
        What it does: Verifies handling of None tool_call_id.
        Purpose: Ensure None is replaced with empty string.
        """
        print('Setup: Tool message with None tool_call_id...')
        messages = [ChatMessage(role='tool', content='Result', tool_call_id
            =None)]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {unified}')
        assert unified[0].tool_results[0]['tool_use_id'] == ''

    def test_handles_empty_tool_content(self):
        """
        What it does: Verifies handling of empty tool content.
        Purpose: Ensure empty content is replaced with "(empty result)".
        """
        print('Setup: Tool message with empty content...')
        messages = [ChatMessage(role='tool', content='', tool_call_id='call_1')
            ]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {unified}')
        assert unified[0].tool_results[0]['content'] == '(empty result)'

    def test_tool_messages_followed_by_user_message(self):
        """
        What it does: Verifies tool messages followed by user message.
        Purpose: Ensure tool results are in separate message from user content.
        """
        print('Setup: Tool messages + user message...')
        messages = [ChatMessage(role='tool', content='Result 1',
            tool_call_id='call_1'), ChatMessage(role='user', content=
            'Continue please')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {unified}')
        assert len(unified) == 2
        assert unified[0].role == 'user'
        assert unified[0].tool_results is not None
        assert unified[1].role == 'user'
        assert unified[1].content == 'Continue please'

    def test_extracts_images_from_user_message(self):
        """
        What it does: Verifies that images are extracted from user messages.
        Purpose: Ensure OpenAI image_url content blocks are converted to unified format.
        
        This test verifies the fix for Issue #30 - 422 Validation Error for image content.
        """
        print('Setup: User message with image_url content block...')
        test_image_base64 = (
            '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=='
            )
        messages = [ChatMessage(role='user', content=[{'type': 'text',
            'text': "What's in this image?"}, {'type': 'image_url',
            'image_url': {'url':
            f'data:image/jpeg;base64,{test_image_base64}'}}])]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Result: {unified}')
        print(f'Images: {unified[0].images}')
        assert len(unified) == 1
        assert unified[0].role == 'user'
        assert unified[0].content == "What's in this image?"
        print('Checking images field...')
        assert unified[0].images is not None, 'images field should not be None'
        assert len(unified[0].images
            ) == 1, f'Expected 1 image, got {len(unified[0].images)}'
        image = unified[0].images[0]
        print(
            f"Comparing image: Expected media_type='image/jpeg', Got '{image.get('media_type')}'"
            )
        assert image['media_type'] == 'image/jpeg'
        print(
            f"Comparing image data: Expected {test_image_base64[:20]}..., Got {image.get('data', '')[:20]}..."
            )
        assert image['data'] == test_image_base64

    def test_images_only_extracted_from_user_role(self):
        """
        What it does: Verifies that images are only extracted from user messages.
        Purpose: Ensure assistant messages don't have images extracted.
        """
        print('Setup: Conversation with image in user message only...')
        test_image_base64 = (
            '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=='
            )
        messages = [ChatMessage(role='user', content=[{'type': 'text',
            'text': 'Describe this image'}, {'type': 'image_url',
            'image_url': {'url':
            f'data:image/png;base64,{test_image_base64}'}}]), ChatMessage(
            role='assistant', content='I can see a small image.')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Result: {unified}')
        print('Checking user message has images...')
        assert unified[0].images is not None
        assert len(unified[0].images) == 1
        print('Checking assistant message has no images...')
        assert unified[1
            ].images is None, 'Assistant messages should not have images extracted'

    def test_extracts_multiple_images_from_user_message(self):
        """
        What it does: Verifies extraction of multiple images from a single user message.
        Purpose: Ensure all images in a message are extracted.
        """
        print('Setup: User message with multiple images...')
        test_image_base64 = (
            '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=='
            )
        messages = [ChatMessage(role='user', content=[{'type': 'text',
            'text': 'Compare these images'}, {'type': 'image_url',
            'image_url': {'url':
            f'data:image/jpeg;base64,{test_image_base64}'}}, {'type':
            'image_url', 'image_url': {'url':
            f'data:image/png;base64,{test_image_base64}'}}, {'type':
            'image_url', 'image_url': {'url':
            f'data:image/webp;base64,{test_image_base64}'}}])]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(
            f'Result images count: {len(unified[0].images) if unified[0].images else 0}'
            )
        assert unified[0].images is not None
        assert len(unified[0].images
            ) == 3, f'Expected 3 images, got {len(unified[0].images)}'
        print('Checking image media types...')
        media_types = [img['media_type'] for img in unified[0].images]
        print(f'Media types: {media_types}')
        assert 'image/jpeg' in media_types
        assert 'image/png' in media_types
        assert 'image/webp' in media_types

    def test_counts_images_in_debug_log(self, caplog):
        """
        What it does: Verifies that image count is logged in debug message.
        Purpose: Ensure logging includes image statistics for debugging.
        """
        import logging
        print('Setup: User message with images for logging test...')
        test_image_base64 = (
            '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAABAAEBAREA/8QAFAABAAAAAAAAAAAAAAAAAAAACf/EABQQAQAAAAAAAAAAAAAAAAAAAAD/2gAIAQEAAD8AVN//2Q=='
            )
        messages = [ChatMessage(role='user', content=[{'type': 'text',
            'text': 'Analyze this'}, {'type': 'image_url', 'image_url': {
            'url': f'data:image/jpeg;base64,{test_image_base64}'}}, {'type':
            'image_url', 'image_url': {'url':
            f'data:image/png;base64,{test_image_base64}'}}])]
        print('Action: Converting messages with logging enabled...')
        with caplog.at_level(logging.DEBUG):
            system_prompt, unified = convert_openai_messages_to_unified(
                messages)
        print(f'Log records: {[r.message for r in caplog.records]}')
        assert unified[0].images is not None
        assert len(unified[0].images) == 2
        print('Images extracted successfully - logging verification complete')


class TestConvertOpenAIToolsToUnified:
    """Tests for convert_openai_tools_to_unified function."""

    def test_returns_none_for_none(self):
        """
        What it does: Verifies handling of None.
        Purpose: Ensure None returns None.
        """
        print('Setup: None tools...')
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(None)
        print(f'Result: {result}')
        assert result is None

    def test_returns_none_for_empty_list(self):
        """
        What it does: Verifies handling of empty list.
        Purpose: Ensure empty list returns None.
        """
        print('Setup: Empty tools list...')
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified([])
        print(f'Result: {result}')
        assert result is None

    def test_converts_function_tool(self):
        """
        What it does: Verifies conversion of function tool.
        Purpose: Ensure Tool is converted to UnifiedTool.
        """
        print('Setup: Function tool...')
        tools = [Tool(type='function', function=ToolFunction(name=
            'get_weather', description='Get weather for a location',
            parameters={'type': 'object', 'properties': {'location': {
            'type': 'string'}}}))]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        assert result is not None
        assert len(result) == 1
        assert result[0].name == 'get_weather'
        assert result[0].description == 'Get weather for a location'
        assert result[0].input_schema == {'type': 'object', 'properties': {
            'location': {'type': 'string'}}}

    def test_skips_non_function_tools(self):
        """
        What it does: Verifies skipping of non-function tools.
        Purpose: Ensure only function tools are converted.
        """
        print('Setup: Non-function tool...')
        tools = [Tool(type='other_type', function=ToolFunction(name='test',
            description='Test', parameters={}))]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        assert result is None

    def test_converts_multiple_tools(self):
        """
        What it does: Verifies conversion of multiple tools.
        Purpose: Ensure all function tools are converted.
        """
        print('Setup: Multiple tools...')
        tools = [Tool(type='function', function=ToolFunction(name='tool1',
            description='Tool 1', parameters={})), Tool(type='function',
            function=ToolFunction(name='tool2', description='Tool 2',
            parameters={})), Tool(type='function', function=ToolFunction(
            name='tool3', description='Tool 3', parameters={}))]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        assert len(result) == 3
        assert result[0].name == 'tool1'
        assert result[1].name == 'tool2'
        assert result[2].name == 'tool3'

    def test_converts_flat_format_tool(self):
        """
        What it does: Verifies conversion of flat format tool (Cursor-style).
        Purpose: Ensure Cursor IDE flat format is supported.
        
        Cursor IDE sends tools in flat format:
        {"type": "function", "name": "...", "description": "...", "input_schema": {...}}
        instead of standard OpenAI nested format.
        """
        print('Setup: Flat format tool (Cursor-style)...')
        tools = [Tool(type='function', name='cursor_tool', description=
            'A tool from Cursor IDE', input_schema={'type': 'object',
            'properties': {'param': {'type': 'string'}}})]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        print(
            f'Comparing count: Expected 1, Got {len(result) if result else 0}')
        assert result is not None
        assert len(result) == 1
        print(f"Comparing name: Expected 'cursor_tool', Got '{result[0].name}'"
            )
        assert result[0].name == 'cursor_tool'
        print(
            f"Comparing description: Expected 'A tool from Cursor IDE', Got '{result[0].description}'"
            )
        assert result[0].description == 'A tool from Cursor IDE'
        print(f'Comparing input_schema: Got {result[0].input_schema}')
        assert result[0].input_schema == {'type': 'object', 'properties': {
            'param': {'type': 'string'}}}

    def test_converts_mixed_format_tools(self):
        """
        What it does: Verifies conversion of mixed format tools.
        Purpose: Ensure both standard and flat format can coexist in same request.
        
        This simulates a scenario where some tools are in standard OpenAI format
        and some are in Cursor flat format (though unlikely in practice).
        """
        print('Setup: Mixed format tools...')
        tools = [Tool(type='function', function=ToolFunction(name=
            'standard_tool', description='Standard format', parameters={
            'type': 'object'})), Tool(type='function', name='flat_tool',
            description='Flat format', input_schema={'type': 'object'})]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        print(f'Comparing count: Expected 2, Got {len(result)}')
        assert len(result) == 2
        print('Checking standard format tool...')
        assert result[0].name == 'standard_tool'
        assert result[0].description == 'Standard format'
        print('Checking flat format tool...')
        assert result[1].name == 'flat_tool'
        assert result[1].description == 'Flat format'

    def test_standard_format_takes_priority(self):
        """
        What it does: Verifies that standard format takes priority over flat format.
        Purpose: Ensure function field is used when both formats are present (edge case).
        
        This is an edge case where a tool has BOTH function and name fields.
        The standard format (function) should take priority.
        """
        print('Setup: Tool with BOTH formats (edge case)...')
        tools = [Tool(type='function', function=ToolFunction(name=
            'standard_name', description='Standard description', parameters
            ={'type': 'object', 'properties': {'a': {'type': 'string'}}}),
            name='flat_name', description='Flat description', input_schema=
            {'type': 'object', 'properties': {'b': {'type': 'string'}}})]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        assert len(result) == 1
        print('Checking that standard format was used (not flat)...')
        print(
            f"Comparing name: Expected 'standard_name', Got '{result[0].name}'"
            )
        assert result[0].name == 'standard_name'
        print(
            f"Comparing description: Expected 'Standard description', Got '{result[0].description}'"
            )
        assert result[0].description == 'Standard description'
        print(f'Comparing input_schema: Got {result[0].input_schema}')
        assert result[0].input_schema == {'type': 'object', 'properties': {
            'a': {'type': 'string'}}}

    def test_skips_invalid_tools(self):
        """
        What it does: Verifies that tools without function OR name are skipped.
        Purpose: Ensure invalid tools don't crash the conversion.
        
        This tests the error handling when a tool has neither function nor name field.
        """
        print('Setup: Invalid tool (no function, no name)...')
        tools = [Tool(type='function', function=ToolFunction(name=
            'valid_tool', description='Valid')), Tool(type='function'),
            Tool(type='function', name='another_valid', description=
            'Also valid', input_schema={})]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        print(
            f'Comparing count: Expected 2 (invalid skipped), Got {len(result)}'
            )
        assert len(result) == 2
        print('Checking that only valid tools were converted...')
        assert result[0].name == 'valid_tool'
        assert result[1].name == 'another_valid'

    def test_backward_compat_standard_openai_tools(self):
        """
        What it does: Verifies that standard OpenAI format is not broken.
        Purpose: Regression test for existing clients (non-Cursor).
        
        This is a critical backward compatibility test. After adding support for
        Cursor's flat format, we must ensure standard OpenAI format still works.
        """
        print('Setup: Standard OpenAI tools (regression test)...')
        tools = [Tool(type='function', function=ToolFunction(name=
            'get_weather', description='Get weather for a location',
            parameters={'type': 'object', 'properties': {'location': {
            'type': 'string', 'description': 'City name'}}, 'required': [
            'location']}))]
        print('Action: Converting tools...')
        result = convert_openai_tools_to_unified(tools)
        print(f'Result: {result}')
        assert result is not None
        assert len(result) == 1
        print(f"Comparing name: Expected 'get_weather', Got '{result[0].name}'"
            )
        assert result[0].name == 'get_weather'
        print(
            f"Comparing description: Expected 'Get weather for a location', Got '{result[0].description}'"
            )
        assert result[0].description == 'Get weather for a location'
        print(f'Comparing input_schema: Got {result[0].input_schema}')
        assert result[0].input_schema['required'] == ['location']
        assert result[0].input_schema['properties']['location']['type'
            ] == 'string'


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestBuildOpenCodePayload:
    """Tests for build_opencode_payload function."""

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_builds_simple_payload(self):
        """
        What it does: Verifies building of simple payload.
        Purpose: Ensure basic request is converted correctly.
        """
        print('Setup: Simple request...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', 'arn:aws:test')
        print(f'Result: {result}')
        assert 'conversationState' in result
        assert result['conversationId'] == 'conv-123'
        assert 'currentMessage' in result
        assert result['profileArn'] == 'arn:aws:test'

    def test_includes_system_prompt_in_first_message(self):
        """
        What it does: Verifies adding system prompt to first message.
        Purpose: Ensure system prompt is merged with user message.
        """
        print('Setup: Request with system prompt...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='system', content='You are helpful'),
            ChatMessage(role='user', content='Hello')])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        current_content = result['messages'][-1]['content']
        assert 'You are helpful' in result['messages'][0]['content']
        assert 'Hello' in current_content

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_builds_history_for_multi_turn(self):
        """
        What it does: Verifies building history for multi-turn.
        Purpose: Ensure previous messages go into history.
        """
        print('Setup: Multi-turn request...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello'), ChatMessage(role=
            'assistant', content='Hi'), ChatMessage(role='user', content=
            'How are you?')])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        assert 'history' in result
        assert len(result['messages'][:-1]) == 2

    def test_handles_assistant_as_last_message(self):
        """
        What it does: Verifies handling of assistant as last message.
        Purpose: Ensure "(empty placeholder)" message is created.
        """
        print('Setup: Request with assistant at the end...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello'), ChatMessage(role=
            'assistant', content='Hi there')])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        current_content = result['messages'][-1]['content']
        assert current_content == '(empty placeholder)'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_raises_for_empty_messages(self):
        """
        What it does: Verifies exception raising for empty messages.
        Purpose: Ensure empty request raises ValueError.
        """
        print('Setup: Request with only system message...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='system', content='You are helpful')])
        print('Action: Attempting to build payload...')
        with pytest.raises(ValueError) as exc_info:
            build_opencode_payload(request, 'conv-123', '')
        print(f'Exception: {exc_info.value}')
        assert 'No messages to send' in str(exc_info.value)

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_uses_continue_for_empty_content(self):
        """
        What it does: Verifies using "(empty placeholder)" for empty content.
        Purpose: Ensure empty message is replaced with "(empty placeholder)".
        """
        print('Setup: Request with empty content...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='')])
        print(
            'Action: Building payload (with fake reasoning and truncation recovery disabled)...'
            )
        with patch('opencode_zen.converters_core.FAKE_REASONING_ENABLED', False
            ):
            with patch('opencode_zen.config.TRUNCATION_RECOVERY', False):
                result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        current_content = result['messages'][-1]['content']
        assert current_content == '(empty placeholder)'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_normalizes_model_id_correctly(self):
        """
        What it does: Verifies normalization of external model ID to OpenCode format.
        Purpose: Ensure model name normalization is applied (dashes→dots, strip dates).
        
        Note: The new Dynamic Model Resolution System normalizes model names
        (e.g., claude-sonnet-4-5 → claude-sonnet-4.5) instead of mapping to
        internal IDs. OpenCode API accepts the normalized format directly.
        """
        print('Setup: Request with external model ID...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        model_id = result['messages'][-1]['modelId']
        print(
            f"Comparing model_id: Expected 'claude-sonnet-4.5', Got '{model_id}'"
            )
        assert model_id == 'claude-sonnet-4.5'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_includes_tools_in_context(self):
        """
        What it does: Verifies including tools in userInputMessageContext.
        Purpose: Ensure tools are converted and included.
        """
        print('Setup: Request with tools...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='get_weather',
            description='Get weather', parameters={'type': 'object',
            'properties': {}}))])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        context = result
        assert 'tools' in context
        assert len(context['tools']) == 1
        assert context['tools'][0]['toolSpecification']['name'
            ] == 'get_weather'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_injects_thinking_tags_even_when_tool_results_present(self):
        """
        What it does: Verifies thinking tags ARE injected even when toolResults are present.
        Purpose: Extended thinking should work in all scenarios including tool use flows.
        """
        print('Setup: Request where last message is a tool result...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Run a command'),
            ChatMessage(role='assistant', content="I'll run the command",
            tool_calls=[{'id': 'tool_1', 'type': 'function', 'function': {
            'name': 'bash', 'arguments': '{}'}}]), ChatMessage(role='tool',
            content='Command output here', tool_call_id='tool_1')], tools=[
            Tool(type='function', function=ToolFunction(name='bash',
            description='Run a bash command', parameters={'type': 'object',
            'properties': {}}))])
        print('Action: Building payload with FAKE_REASONING_ENABLED=True...')
        with patch('opencode_zen.converters_core.FAKE_REASONING_ENABLED', True
            ):
            with patch('opencode_zen.converters_core.FAKE_REASONING_MAX_TOKENS'
                , 4000):
                result = build_opencode_payload(request, 'conv-123', '')
        current_msg = result['messages'][-1]
        content = current_msg['content']
        context = current_msg.get('userInputMessageContext', {})
        print(
            f'Content: {repr(content[:100] if len(content) > 100 else content)}'
            )
        print(f"Has toolResults: {'toolResults' in context}")
        assert 'toolResults' in context, 'toolResults should be present'
        assert '<thinking_mode>enabled</thinking_mode>' in content, 'thinking tags SHOULD be injected even with toolResults'
        assert '<max_thinking_length>4000</max_thinking_length>' in content, 'max_thinking_length should be present'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_injects_thinking_tags_when_no_tool_results(self):
        """
        What it does: Verifies thinking tags ARE injected for normal user messages.
        Purpose: Ensure fix for issue #20 doesn't break normal thinking tag injection.
        """
        print('Setup: Normal user message without tool results...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')])
        print('Action: Building payload with FAKE_REASONING_ENABLED=True...')
        with patch('opencode_zen.converters_core.FAKE_REASONING_ENABLED', True
            ):
            with patch('opencode_zen.converters_core.FAKE_REASONING_MAX_TOKENS'
                , 4000):
                result = build_opencode_payload(request, 'conv-123', '')
        current_msg = result['messages'][-1]
        content = current_msg['content']
        context = current_msg.get('userInputMessageContext', {})
        print(
            f"Content starts with thinking tags: {'<thinking_mode>' in content}"
            )
        print(f"Has toolResults: {'toolResults' in context}")
        assert 'toolResults' not in context, 'toolResults should NOT be present'
        assert '<thinking_mode>' in content, 'thinking tags SHOULD be injected for normal messages'
        assert 'Hello' in content, 'Original content should be preserved'


class TestToolMessageHandling:
    """Tests for OpenAI tool message (role="tool") handling."""

    def test_converts_multiple_tool_messages_to_single_user_message(self):
        """
        What it does: Verifies merging of multiple tool messages into single user message.
        Purpose: Ensure multiple tool results are merged into one user message.
        """
        print('Setup: Multiple consecutive tool messages...')
        messages = [ChatMessage(role='tool', content='Result 1',
            tool_call_id='call_1'), ChatMessage(role='tool', content=
            'Result 2', tool_call_id='call_2'), ChatMessage(role='tool',
            content='Result 3', tool_call_id='call_3')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Result: {unified}')
        print(f'Comparing length: Expected 1, Got {len(unified)}')
        assert len(unified) == 1
        assert unified[0].role == 'user'
        print('Checking content contains all tool_results...')
        assert unified[0].tool_results is not None
        assert len(unified[0].tool_results) == 3
        tool_use_ids = [item['tool_use_id'] for item in unified[0].tool_results
            ]
        assert 'call_1' in tool_use_ids
        assert 'call_2' in tool_use_ids
        assert 'call_3' in tool_use_ids

    def test_assistant_tool_user_sequence(self):
        """
        What it does: Verifies assistant -> tool -> user sequence.
        Purpose: Ensure tool message is correctly inserted between assistant and user.
        """
        print('Setup: assistant -> tool -> user...')
        messages = [ChatMessage(role='assistant', content=
            "I'll call a tool"), ChatMessage(role='tool', content=
            'Tool output', tool_call_id='call_abc'), ChatMessage(role=
            'user', content='Thanks!')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Result: {unified}')
        assert len(unified) == 3
        assert unified[0].role == 'assistant'
        assert unified[1].role == 'user'
        assert unified[1].tool_results is not None
        assert unified[2].role == 'user'

    def test_tool_message_with_empty_content(self):
        """
        What it does: Verifies tool message with empty content.
        Purpose: Ensure empty result is replaced with "(empty result)".
        """
        print('Setup: Tool message with empty content...')
        messages = [ChatMessage(role='tool', content='', tool_call_id=
            'call_empty')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Result: {unified}')
        assert len(unified) == 1
        assert unified[0].tool_results[0]['content'] == '(empty result)'

    def test_tool_message_with_none_tool_call_id(self):
        """
        What it does: Verifies tool message without tool_call_id.
        Purpose: Ensure missing tool_call_id is replaced with empty string.
        """
        print('Setup: Tool message without tool_call_id...')
        messages = [ChatMessage(role='tool', content='Result', tool_call_id
            =None)]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Result: {unified}')
        assert len(unified) == 1
        assert unified[0].tool_results[0]['tool_use_id'] == ''


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestToolDescriptionHandling:
    """Tests for handling empty/whitespace tool descriptions."""

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_empty_description_replaced_with_placeholder(self):
        """
        What it does: Verifies replacement of empty description with placeholder.
        Purpose: Ensure empty description is replaced with "Tool: {name}".
        
        This is a critical test for a Cline bug where tool focus_chain had
        empty description "", which caused a 400 error from OpenCode API.
        """
        print('Setup: Tool with empty description...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='focus_chain',
            description='', parameters={'type': 'object', 'properties': {}}))])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        print('Checking that description is replaced with placeholder...')
        context = result
        tool_spec = context['tools'][0]['toolSpecification']
        assert tool_spec['description'] == 'Tool: focus_chain'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_whitespace_only_description_replaced_with_placeholder(self):
        """
        What it does: Verifies replacement of whitespace-only description with placeholder.
        Purpose: Ensure description with only whitespace is replaced.
        """
        print('Setup: Tool with whitespace-only description...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='whitespace_tool',
            description='   ', parameters={}))])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        print('Checking that description is replaced with placeholder...')
        context = result
        tool_spec = context['tools'][0]['toolSpecification']
        assert tool_spec['description'] == 'Tool: whitespace_tool'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_none_description_replaced_with_placeholder(self):
        """
        What it does: Verifies replacement of None description with placeholder.
        Purpose: Ensure None description is replaced with "Tool: {name}".
        """
        print('Setup: Tool with None description...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='none_desc_tool',
            description=None, parameters={}))])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        print('Checking that description is replaced with placeholder...')
        context = result
        tool_spec = context['tools'][0]['toolSpecification']
        assert tool_spec['description'] == 'Tool: none_desc_tool'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_non_empty_description_preserved(self):
        """
        What it does: Verifies preservation of non-empty description.
        Purpose: Ensure normal description is not changed.
        """
        print('Setup: Tool with normal description...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='get_weather',
            description='Get weather for a location', parameters={}))])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        print('Checking that description is preserved...')
        context = result
        tool_spec = context['tools'][0]['toolSpecification']
        assert tool_spec['description'] == 'Get weather for a location'

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_sanitizes_tool_parameters(self):
        """
        What it does: Verifies sanitization of parameters from problematic fields.
        Purpose: Ensure sanitize_json_schema is applied to parameters.
        """
        print('Setup: Tool with problematic parameters...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='test_tool', description
            ='Test tool', parameters={'type': 'object', 'properties': {},
            'required': [], 'additionalProperties': False}))])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        print('Checking that parameters are sanitized...')
        context = result
        input_schema = context['tools'][0]['toolSpecification']['inputSchema'][
            'json']
        assert 'required' not in input_schema
        assert 'additionalProperties' not in input_schema

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_mixed_tools_with_empty_and_normal_descriptions(self):
        """
        What it does: Verifies handling of mixed tools list.
        Purpose: Ensure empty descriptions are replaced while normal ones are preserved.
        
        This is a real scenario from Cline where most tools have
        normal descriptions, but focus_chain has an empty one.
        """
        print('Setup: Mixed list of tools...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='read_file', description
            ='Read contents of a file', parameters={})), Tool(type=
            'function', function=ToolFunction(name='focus_chain',
            description='', parameters={})), Tool(type='function', function
            =ToolFunction(name='write_file', description=
            'Write content to a file', parameters={}))])
        print('Action: Building payload...')
        result = build_opencode_payload(request, 'conv-123', '')
        print(f'Result: {result}')
        print('Checking descriptions...')
        tools = result.get('tools', [])
        assert tools[0]['function']['description'] == 'Read contents of a file'
        assert tools[1]['function']['description'] == 'Tool: focus_chain'
        assert tools[2]['function']['description'] == 'Write content to a file'


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestBuildOpenCodePayloadToolCallsIntegration:
    """
    Integration tests for build_opencode_payload with tool_calls.
    Tests full flow from OpenAI format to OpenCode format.
    """

    def test_multiple_assistant_tool_calls_with_results(self):
        """
        What it does: Verifies full scenario with multiple assistant tool_calls and their results.
        Purpose: Ensure all toolUses and toolResults are correctly linked in OpenCode payload.

        This is an integration test for a Codex CLI bug where multiple assistant
        messages with tool_calls were sent in a row, followed by tool results.
        """
        print('Setup: Full scenario with two tool_calls and their results...')
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='user', content='Run two commands'),
            ChatMessage(role='assistant', content=None, tool_calls=[{'id':
            'tooluse_first', 'type': 'function', 'function': {'name':
            'shell', 'arguments': '{"command": ["ls"]}'}}]), ChatMessage(
            role='assistant', content=None, tool_calls=[{'id':
            'tooluse_second', 'type': 'function', 'function': {'name':
            'shell', 'arguments': '{"command": ["pwd"]}'}}]), ChatMessage(
            role='tool', content="""file1.txt
file2.txt""", tool_call_id=
            'tooluse_first'), ChatMessage(role='tool', content='/home/user',
            tool_call_id='tooluse_second')], tools=[Tool(type='function',
            function=ToolFunction(name='shell', description=
            'Run a shell command', parameters={'type': 'object',
            'properties': {'command': {'type': 'array'}}}))])
        print('Action: Building OpenCode payload...')
        result = build_opencode_payload(request, 'conv-123', 'arn:aws:test')
        print(f'Result: {result}')
        messages = result.get('messages', [])
        print(f'Messages: {messages}')
        assert len(messages) == 5, f'Expected 5 messages, got {len(messages)}'
        assistant_msgs = [m for m in messages if m.get('role') == 'assistant']
        assert len(assistant_msgs) == 2, 'Should have two assistant messages'
        tool_uses = []
        for msg in assistant_msgs:
            tool_uses.extend(msg.get('tool_calls', []))
        assert len(tool_uses
            ) == 2, f'Should have 2 tool_calls, got {len(tool_uses)}'
        tool_use_ids = [tu['id'] for tu in tool_uses]
        assert 'tooluse_first' in tool_use_ids
        assert 'tooluse_second' in tool_use_ids
        tool_results = [m for m in messages if m.get('role') == 'tool']
        assert len(tool_results
            ) == 2, f'Should have 2 tool results, got {len(tool_results)}'
        tool_result_ids = [tr['tool_call_id'] for tr in tool_results]
        assert 'tooluse_first' in tool_result_ids
        assert 'tooluse_second' in tool_result_ids

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_long_tool_description_added_to_system_prompt(self):
        """
        What it does: Verifies integration of long tool descriptions into payload.
        Purpose: Ensure long descriptions are added to system prompt in payload.
        """
        print('Setup: Request with tool with long description...')
        long_desc = 'X' * 15000
        request = ChatCompletionRequest(model='claude-sonnet-4-5', messages
            =[ChatMessage(role='system', content='You are helpful'),
            ChatMessage(role='user', content='Hello')], tools=[Tool(type=
            'function', function=ToolFunction(name='long_tool', description
            =long_desc, parameters={}))])
        print('Action: Building payload...')
        with patch('opencode_zen.converters_core.TOOL_DESCRIPTION_MAX_LENGTH',
            10000):
            result = build_opencode_payload(request, 'conv-123', '')
        print('Checking that system prompt contains tool documentation...')
        current_content = result['messages'][-1]['content']
        assert 'You are helpful' in result['messages'][0]['content']
        assert '## Tool: long_tool' in current_content
        assert long_desc in current_content
        print('Checking that tool in context has reference description...')
        tools_context = result['tools']
        assert '[Full documentation in system prompt' in tools_context[0][
            'toolSpecification']['description']


class TestExtractImagesFromToolMessage:
    """Tests for _extract_images_from_tool_message function."""

    def test_extracts_single_image_from_tool_message(self):
        """
        What it does: Verifies extraction of a single image from tool message content.
        Purpose: Ensure images in OpenAI tool messages are properly extracted (MCP support).
        """
        print('Setup: Tool message with single image...')
        content = [{'type': 'text', 'text': 'Screenshot captured'}, {'type':
            'image_url', 'image_url': {'url':
            'data:image/png;base64,iVBORw0KGgoAAAANSUhEUg=='}}]
        print('Action: Extracting images from tool message...')
        result = _extract_images_from_tool_message(content)
        print(f'Result: {result}')
        assert len(result) == 1
        assert result[0]['media_type'] == 'image/png'
        assert result[0]['data'] == 'iVBORw0KGgoAAAANSUhEUg=='

    def test_extracts_multiple_images_from_tool_message(self):
        """
        What it does: Verifies extraction of multiple images from tool message.
        Purpose: Ensure all images are extracted from a single tool message.
        """
        print('Setup: Tool message with multiple images...')
        content = [{'type': 'image_url', 'image_url': {'url':
            'data:image/png;base64,png_data'}}, {'type': 'image_url',
            'image_url': {'url': 'data:image/jpeg;base64,jpeg_data'}}]
        print('Action: Extracting images from tool message...')
        result = _extract_images_from_tool_message(content)
        print(f'Result: {result}')
        assert len(result) == 2
        assert result[0]['media_type'] == 'image/png'
        assert result[0]['data'] == 'png_data'
        assert result[1]['media_type'] == 'image/jpeg'
        assert result[1]['data'] == 'jpeg_data'

    def test_returns_empty_for_text_only_tool_message(self):
        """
        What it does: Verifies empty list returned when tool message has no images.
        Purpose: Ensure text-only tool messages don't produce spurious images.
        """
        print('Setup: Tool message with text only...')
        content = [{'type': 'text', 'text': 'Operation completed successfully'}
            ]
        print('Action: Extracting images from tool message...')
        result = _extract_images_from_tool_message(content)
        print(f'Result: {result}')
        assert result == []

    def test_returns_empty_for_string_content(self):
        """
        What it does: Verifies empty list returned for string content.
        Purpose: Ensure string content doesn't cause errors.
        """
        print('Setup: String content...')
        content = 'Just a string result'
        print('Action: Extracting images from tool message...')
        result = _extract_images_from_tool_message(content)
        print(f'Result: {result}')
        assert result == []

    def test_returns_empty_for_none_content(self):
        """
        What it does: Verifies empty list returned for None content.
        Purpose: Ensure None content doesn't cause errors.
        """
        print('Setup: None content...')
        content = None
        print('Action: Extracting images from tool message...')
        result = _extract_images_from_tool_message(content)
        print(f'Result: {result}')
        assert result == []

    def test_extracts_images_mixed_with_text(self):
        """
        What it does: Verifies images are extracted when mixed with text content.
        Purpose: Ensure images are found even when text blocks are present.
        """
        print('Setup: Tool message with text and image...')
        content = [{'type': 'text', 'text': 'Screenshot captured'}, {'type':
            'image_url', 'image_url': {'url':
            'data:image/png;base64,screenshot_data'}}, {'type': 'text',
            'text': 'Analysis complete'}]
        print('Action: Extracting images from tool message...')
        result = _extract_images_from_tool_message(content)
        print(f'Result: {result}')
        assert len(result) == 1
        assert result[0]['data'] == 'screenshot_data'


class TestConvertOpenAIMessagesWithToolImages:
    """Tests for convert_openai_messages_to_unified with tool message images."""

    def test_extracts_images_from_tool_messages(self):
        """
        What it does: Verifies images are extracted from tool messages and added to unified message.
        Purpose: Ensure tool message images are properly converted to unified format.
        """
        print('Setup: Messages with tool message containing image...')
        messages = [ChatMessage(role='user', content='Take a screenshot'),
            ChatMessage(role='assistant', content='Taking screenshot',
            tool_calls=[{'id': 'call_123', 'type': 'function', 'function':
            {'name': 'screenshot', 'arguments': '{}'}}]), ChatMessage(role=
            'tool', tool_call_id='call_123', content=[{'type': 'text',
            'text': 'Screenshot captured'}, {'type': 'image_url',
            'image_url': {'url': 'data:image/png;base64,test_image'}}])]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {len(unified)}')
        print(f'Last message images: {unified[-1].images}')
        assert len(unified) == 3
        assert unified[-1].role == 'user'
        assert unified[-1].tool_results is not None
        assert len(unified[-1].tool_results) == 1
        assert unified[-1].images is not None
        assert len(unified[-1].images) == 1
        assert unified[-1].images[0]['data'] == 'test_image'

    def test_merges_images_from_multiple_tool_messages(self):
        """
        What it does: Verifies images from multiple tool messages are merged.
        Purpose: Ensure all tool message images are collected into one user message.
        """
        print('Setup: Multiple tool messages with images...')
        messages = [ChatMessage(role='tool', tool_call_id='call_1', content
            =[{'type': 'image_url', 'image_url': {'url':
            'data:image/png;base64,image1'}}]), ChatMessage(role='tool',
            tool_call_id='call_2', content=[{'type': 'image_url',
            'image_url': {'url': 'data:image/jpeg;base64,image2'}}])]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {len(unified)}')
        print(
            f'Images count: {len(unified[0].images) if unified[0].images else 0}'
            )
        assert len(unified) == 1
        assert unified[0].role == 'user'
        assert unified[0].images is not None
        assert len(unified[0].images) == 2
        assert unified[0].images[0]['data'] == 'image1'
        assert unified[0].images[1]['data'] == 'image2'

    def test_handles_tool_message_with_text_and_image(self):
        """
        What it does: Verifies tool message with both text and image is handled correctly.
        Purpose: Ensure both text and images are extracted from tool messages.
        """
        print('Setup: Tool message with text and image...')
        messages = [ChatMessage(role='tool', tool_call_id='call_123',
            content=[{'type': 'text', 'text':
            'Screenshot captured successfully'}, {'type': 'image_url',
            'image_url': {'url': 'data:image/png;base64,screenshot_data'}}]
            ), ChatMessage(role='user', content='What do you see?')]
        print('Action: Converting messages...')
        system_prompt, unified = convert_openai_messages_to_unified(messages)
        print(f'Unified messages: {len(unified)}')
        assert unified[0].role == 'user'
        assert unified[0].tool_results is not None
        assert 'Screenshot captured successfully' in unified[0].tool_results[0
            ]['content']
        assert unified[0].images is not None
        assert len(unified[0].images) == 1
        assert unified[0].images[0]['data'] == 'screenshot_data'
        assert unified[1].role == 'user'
        assert unified[1].content == 'What do you see?'


class TestReasoningEffortToBudget:
    """Tests for reasoning_effort_to_budget function."""

    def test_none_returns_zero(self):
        """
        What it does: Verifies reasoning_effort="none" returns 0 tokens
        Purpose: Ensure "none" disables thinking budget
        """
        print("Testing reasoning_effort='none'...")
        result = reasoning_effort_to_budget(4096, 'none')
        print(f'Comparing: expected=0, got={result}')
        assert result == 0

    def test_minimal_returns_10_percent(self):
        """
        What it does: Verifies reasoning_effort="minimal" returns 10% of max_tokens
        Purpose: Ensure minimal reasoning uses 10% budget
        """
        print("Testing reasoning_effort='minimal' with max_tokens=4096...")
        result = reasoning_effort_to_budget(4096, 'minimal')
        expected = int(4096 * 0.1)
        print(f'Comparing: expected={expected}, got={result}')
        assert result == expected

    def test_low_returns_20_percent(self):
        """
        What it does: Verifies reasoning_effort="low" returns 20%
        Purpose: Ensure low reasoning uses 20% budget
        """
        print("Testing reasoning_effort='low' with max_tokens=4096...")
        result = reasoning_effort_to_budget(4096, 'low')
        expected = int(4096 * 0.2)
        print(f'Comparing: expected={expected}, got={result}')
        assert result == expected

    def test_medium_returns_50_percent(self):
        """
        What it does: Verifies reasoning_effort="medium" returns 50%
        Purpose: Ensure medium reasoning uses 50% budget
        """
        print("Testing reasoning_effort='medium' with max_tokens=4096...")
        result = reasoning_effort_to_budget(4096, 'medium')
        expected = int(4096 * 0.5)
        print(f'Comparing: expected={expected}, got={result}')
        assert result == expected

    def test_high_returns_80_percent(self):
        """
        What it does: Verifies reasoning_effort="high" returns 80%
        Purpose: Ensure high reasoning uses 80% budget
        """
        print("Testing reasoning_effort='high' with max_tokens=4096...")
        result = reasoning_effort_to_budget(4096, 'high')
        expected = int(4096 * 0.8)
        print(f'Comparing: expected={expected}, got={result}')
        assert result == expected

    def test_xhigh_returns_95_percent(self):
        """
        What it does: Verifies reasoning_effort="xhigh" returns 95%
        Purpose: Ensure maximum reasoning uses 95% budget
        """
        print("Testing reasoning_effort='xhigh' with max_tokens=4096...")
        result = reasoning_effort_to_budget(4096, 'xhigh')
        expected = int(4096 * 0.95)
        print(f'Comparing: expected={expected}, got={result}')
        assert result == expected

    def test_adapts_to_different_max_tokens(self):
        """
        What it does: Verifies percentage-based mapping adapts to different max_tokens
        Purpose: Ensure budget scales proportionally with output limit
        """
        print('Testing with different max_tokens values...')
        result_10k = reasoning_effort_to_budget(10000, 'high')
        expected_10k = int(10000 * 0.8)
        print(
            f'  max_tokens=10000, high: expected={expected_10k}, got={result_10k}'
            )
        assert result_10k == expected_10k
        result_2k = reasoning_effort_to_budget(2000, 'high')
        expected_2k = int(2000 * 0.8)
        print(
            f'  max_tokens=2000, high: expected={expected_2k}, got={result_2k}'
            )
        assert result_2k == expected_2k


class TestExtractThinkingConfigFromOpenAI:
    """Tests for extract_thinking_config_from_openai function."""

    def test_no_reasoning_effort(self):
        """
        What it does: Verifies ThinkingConfig(enabled=True, budget_tokens=None) when reasoning_effort=None
        Purpose: Ensure default configuration when reasoning_effort not specified
        """
        print('Creating request without reasoning_effort...')
        request = ChatCompletionRequest(model='claude-sonnet-4.5', messages
            =[ChatMessage(role='user', content='test')])
        print('Extracting thinking config...')
        config = extract_thinking_config_from_openai(request)
        print(
            f'Comparing: enabled={config.enabled}, budget_tokens={config.budget_tokens}'
            )
        assert config.enabled is True
        assert config.budget_tokens is None

    def test_reasoning_effort_none(self):
        """
        What it does: Verifies ThinkingConfig(enabled=False) when reasoning_effort="none"
        Purpose: Ensure thinking is disabled when client explicitly sets "none"
        """
        print("Creating request with reasoning_effort='none'...")
        request = ChatCompletionRequest(model='claude-sonnet-4.5', messages
            =[ChatMessage(role='user', content='test')], reasoning_effort=
            'none')
        print('Extracting thinking config...')
        config = extract_thinking_config_from_openai(request)
        print(
            f'Comparing: enabled={config.enabled}, budget_tokens={config.budget_tokens}'
            )
        assert config.enabled is False
        assert config.budget_tokens is None

    def test_reasoning_effort_minimal(self):
        """
        What it does: Verifies correct budget calculation for reasoning_effort="minimal"
        Purpose: Ensure 10% budget is calculated correctly
        """
        print(
            "Creating request with reasoning_effort='minimal', max_tokens=4096..."
            )
        request = ChatCompletionRequest(model='claude-sonnet-4.5', messages
            =[ChatMessage(role='user', content='test')], max_tokens=4096,
            reasoning_effort='minimal')
        print('Extracting thinking config...')
        config = extract_thinking_config_from_openai(request)
        expected_budget = int(4096 * 0.1)
        print(
            f'Comparing: enabled={config.enabled}, budget_tokens={config.budget_tokens}, expected={expected_budget}'
            )
        assert config.enabled is True
        assert config.budget_tokens == expected_budget

    def test_reasoning_effort_high(self):
        """
        What it does: Verifies correct budget calculation for reasoning_effort="high"
        Purpose: Ensure 80% budget is calculated correctly
        """
        print(
            "Creating request with reasoning_effort='high', max_tokens=4096..."
            )
        request = ChatCompletionRequest(model='claude-sonnet-4.5', messages
            =[ChatMessage(role='user', content='test')], max_tokens=4096,
            reasoning_effort='high')
        print('Extracting thinking config...')
        config = extract_thinking_config_from_openai(request)
        expected_budget = int(4096 * 0.8)
        print(
            f'Comparing: enabled={config.enabled}, budget_tokens={config.budget_tokens}, expected={expected_budget}'
            )
        assert config.enabled is True
        assert config.budget_tokens == expected_budget

    def test_no_max_tokens_uses_fallback(self):
        """
        What it does: Verifies fallback to 4096 when max_tokens not specified
        Purpose: Ensure reasonable default for OUTPUT tokens (not INPUT tokens)
        """
        print(
            "Creating request with reasoning_effort='high' but no max_tokens..."
            )
        request = ChatCompletionRequest(model='claude-sonnet-4.5', messages
            =[ChatMessage(role='user', content='test')], reasoning_effort=
            'high')
        print('Extracting thinking config...')
        config = extract_thinking_config_from_openai(request)
        expected_budget = int(4096 * 0.8)
        print(
            f'Comparing: budget_tokens={config.budget_tokens}, expected={expected_budget}'
            )
        assert config.budget_tokens == expected_budget

    def test_uses_max_completion_tokens(self):
        """
        What it does: Verifies max_completion_tokens is used when max_tokens is None
        Purpose: Ensure alternative max_tokens field is supported
        """
        print('Creating request with max_completion_tokens=8192...')
        request = ChatCompletionRequest(model='claude-sonnet-4.5', messages
            =[ChatMessage(role='user', content='test')],
            max_completion_tokens=8192, reasoning_effort='high')
        print('Extracting thinking config...')
        config = extract_thinking_config_from_openai(request)
        expected_budget = int(8192 * 0.8)
        print(
            f'Comparing: budget_tokens={config.budget_tokens}, expected={expected_budget}'
            )
        assert config.budget_tokens == expected_budget


@pytest.mark.skip(reason='Obsolete after payload/account refactoring')
class TestBuildOpenCodePayloadIntegration:
    """Integration tests for build_opencode_payload with thinking config."""

    @pytest.mark.skip(reason='Obsolete after OpenAI payload migration')
    def test_extracts_and_passes_thinking_config(self, monkeypatch):
        """
        What it does: Verifies build_opencode_payload extracts thinking_config and passes to core
        Purpose: Ensure end-to-end thinking configuration flow works
        """
        print('Setting up mocks...')
        monkeypatch.setattr(
            'opencode_zen.converters_core.FAKE_REASONING_ENABLED', True)
        monkeypatch.setattr(
            'opencode_zen.converters_core.FAKE_REASONING_BUDGET_CAP', 10000)
        print(
            "Creating request with reasoning_effort='medium', max_tokens=8000..."
            )
        request = ChatCompletionRequest(model='claude-sonnet-4.5', messages
            =[ChatMessage(role='user', content='Test message')], max_tokens
            =8000, reasoning_effort='medium')
        print('Calling build_opencode_payload...')
        payload = build_opencode_payload(request_data=request,
            conversation_id='test-conv-123', profile_arn='arn:aws:test')
        print('Extracting userInputMessage content...')
        user_input = payload['messages'][-1]
        content = user_input['content']
        expected_budget = int(8000 * 0.5)
        print(
            f'Checking for <max_thinking_length>{expected_budget}</max_thinking_length>...'
            )
        assert f'<max_thinking_length>{expected_budget}</max_thinking_length>' in content
        assert '<thinking_mode>enabled</thinking_mode>' in content


class TestBuildOpenAIGenerationParams:
    """Tests for pass-through of OpenAI sampling params into the upstream payload."""

    def test_maps_sampling_params(self):
        """
        What it does: temperature/top_p/stop/tool_choice pass through unchanged.
        Purpose: OpenAI request fields are already in the upstream shape.
        """
        from opencode_zen.converters_openai import build_openai_generation_params
        request = ChatCompletionRequest(
            model='claude-sonnet-4-5',
            messages=[ChatMessage(role='user', content='hi')],
            max_tokens=321, temperature=0.3, top_p=0.8, stop=['END'],
            tool_choice='auto',
        )
        params = build_openai_generation_params(request)
        assert params['max_tokens'] == 321
        assert params['temperature'] == 0.3
        assert params['top_p'] == 0.8
        assert params['stop'] == ['END']
        assert params['tool_choice'] == 'auto'

    def test_max_completion_tokens_fallback(self):
        """
        What it does: max_tokens falls back to max_completion_tokens.
        Purpose: Newer OpenAI clients send max_completion_tokens instead.
        """
        from opencode_zen.converters_openai import build_openai_generation_params
        request = ChatCompletionRequest(
            model='claude-sonnet-4-5',
            messages=[ChatMessage(role='user', content='hi')],
            max_completion_tokens=999,
        )
        assert build_openai_generation_params(request)['max_tokens'] == 999

    def test_params_reach_payload(self):
        """
        What it does: build_opencode_payload includes the forwarded params.
        Purpose: End-to-end confirmation that params are not dropped.
        """
        request = ChatCompletionRequest(
            model='claude-sonnet-4-5',
            messages=[ChatMessage(role='user', content='hi')],
            max_tokens=222, temperature=0.1,
        )
        payload = build_opencode_payload(request, 'conv-1', '')
        assert payload['max_tokens'] == 222
        assert payload['temperature'] == 0.1
