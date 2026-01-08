"""
Pytest configuration and shared fixtures for arcllm tests.
"""

import json
from typing import Any

import pytest

# =============================================================================
# OpenAI Response Fixtures
# =============================================================================


@pytest.fixture
def openai_completion_response() -> dict[str, Any]:
    """Standard OpenAI chat completion response."""
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        "system_fingerprint": "fp_123abc",
    }


@pytest.fixture
def openai_tool_call_response() -> dict[str, Any]:
    """OpenAI response with tool calls."""
    return {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 50, "completion_tokens": 30, "total_tokens": 80},
    }


@pytest.fixture
def openai_stream_chunks() -> list[str]:
    """OpenAI streaming response chunks (SSE data)."""
    return [
        'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
        "data: [DONE]\n\n",
    ]


@pytest.fixture
def openai_stream_with_usage() -> list[str]:
    """OpenAI streaming response with usage (stream_options.include_usage=True)."""
    return [
        'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl-789","object":"chat.completion.chunk","created":1677652288,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
        "data: [DONE]\n\n",
    ]


@pytest.fixture
def openai_embedding_response() -> dict[str, Any]:
    """OpenAI embedding response."""
    return {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "index": 0,
                "embedding": [0.0023064255, -0.009327292, 0.015797347],
            },
            {
                "object": "embedding",
                "index": 1,
                "embedding": [0.0012345678, -0.007654321, 0.012345678],
            },
        ],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


# =============================================================================
# Anthropic Response Fixtures
# =============================================================================


@pytest.fixture
def anthropic_completion_response() -> dict[str, Any]:
    """Anthropic messages response."""
    return {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello! How can I assist you today?"}],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 15},
    }


@pytest.fixture
def anthropic_tool_call_response() -> dict[str, Any]:
    """Anthropic response with tool use."""
    return {
        "id": "msg_456",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "I'll check the weather for you."},
            {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "get_weather",
                "input": {"location": "San Francisco", "unit": "celsius"},
            },
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 50, "output_tokens": 40},
    }


@pytest.fixture
def anthropic_stream_events() -> list[str]:
    """Anthropic streaming events."""
    return [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_789","type":"message","role":"assistant","model":"claude-3-5-sonnet-20241022","content":[],"stop_reason":null,"usage":{"input_tokens":10}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]


# =============================================================================
# Gemini Response Fixtures
# =============================================================================


@pytest.fixture
def gemini_completion_response() -> dict[str, Any]:
    """Gemini generateContent response."""
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello! How can I help you?"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 10, "totalTokenCount": 18},
    }


# =============================================================================
# Test Messages
# =============================================================================


@pytest.fixture
def simple_messages() -> list[dict[str, Any]]:
    """Simple user message."""
    return [{"role": "user", "content": "Hello!"}]


@pytest.fixture
def messages_with_system() -> list[dict[str, Any]]:
    """Messages with system prompt."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]


@pytest.fixture
def messages_with_tools() -> list[dict[str, Any]]:
    """Messages for tool calling."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in San Francisco?"},
    ]


@pytest.fixture
def tool_definition() -> list[dict[str, Any]]:
    """Tool definition for weather function."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]


# =============================================================================
# Helper Functions
# =============================================================================


def response_to_bytes(response: dict[str, Any]) -> bytes:
    """Convert dict response to bytes."""
    return json.dumps(response).encode("utf-8")
