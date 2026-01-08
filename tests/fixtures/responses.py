"""
Shared test fixtures for provider response data.

This module contains realistic mock responses from various providers
used in unit tests. All fixtures are based on actual API responses
with sensitive data redacted.
"""

from __future__ import annotations

import json

# =============================================================================
# OpenAI Fixtures
# =============================================================================

OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-test123",
    "object": "chat.completion",
    "created": 1704067200,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I help you today?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}

OPENAI_CHAT_RESPONSE_WITH_TOOLS = {
    "id": "chatcmpl-tool123",
    "object": "chat.completion",
    "created": 1704067200,
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
    "usage": {
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75,
    },
}

OPENAI_STREAMING_CHUNKS = [
    'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-stream1","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
    "data: [DONE]\n\n",
]

OPENAI_STREAMING_WITH_USAGE = [
    'data: {"id":"chatcmpl-stream2","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-stream2","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
    'data: {"id":"chatcmpl-stream2","object":"chat.completion.chunk","created":1704067200,"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}\n\n',
    "data: [DONE]\n\n",
]

OPENAI_EMBEDDING_RESPONSE = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.1, 0.2, 0.3] + [0.0] * 1533,  # 1536 dims
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 5, "total_tokens": 5},
}

# =============================================================================
# Anthropic Fixtures
# =============================================================================

ANTHROPIC_CHAT_RESPONSE = {
    "id": "msg_test123",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello! How can I assist you today?"}],
    "model": "claude-3-5-haiku-20241022",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 9},
}

ANTHROPIC_CHAT_RESPONSE_WITH_TOOLS = {
    "id": "msg_tool123",
    "type": "message",
    "role": "assistant",
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "get_weather",
            "input": {"location": "San Francisco", "unit": "celsius"},
        }
    ],
    "model": "claude-3-5-haiku-20241022",
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 50, "output_tokens": 30},
}

ANTHROPIC_STREAMING_EVENTS = [
    'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_stream1","type":"message","role":"assistant","content":[],"model":"claude-3-5-haiku-20241022","stop_reason":null,"usage":{"input_tokens":10,"output_tokens":0}}}\n\n',
    'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
    'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}\n\n',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
    'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}\n\n',
    'event: message_stop\ndata: {"type":"message_stop"}\n\n',
]

# =============================================================================
# Gemini Fixtures
# =============================================================================

GEMINI_CHAT_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [{"text": "Hello! How may I help you?"}],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 10,
        "candidatesTokenCount": 7,
        "totalTokenCount": 17,
    },
}

GEMINI_CHAT_RESPONSE_WITH_TOOLS = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "functionCall": {
                            "name": "get_weather",
                            "args": {"location": "San Francisco", "unit": "celsius"},
                        }
                    }
                ],
                "role": "model",
            },
            "finishReason": "STOP",
            "index": 0,
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 50,
        "candidatesTokenCount": 20,
        "totalTokenCount": 70,
    },
}

GEMINI_EMBEDDING_RESPONSE = {
    "embeddings": [{"values": [0.1, 0.2, 0.3] + [0.0] * 765}]  # 768 dims
}

# =============================================================================
# Mistral Fixtures
# =============================================================================

MISTRAL_CHAT_RESPONSE = {
    "id": "chat-test123",
    "object": "chat.completion",
    "created": 1704067200,
    "model": "mistral-small-latest",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Bonjour! Comment puis-je vous aider?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 8,
        "total_tokens": 18,
    },
}

# =============================================================================
# Cohere Fixtures
# =============================================================================

COHERE_CHAT_RESPONSE = {
    "response_id": "test-response-123",
    "text": "Hello! How can I assist you today?",
    "generation_id": "gen-123",
    "chat_history": [],
    "finish_reason": "COMPLETE",
    "meta": {
        "api_version": {"version": "1"},
        "billed_units": {"input_tokens": 10, "output_tokens": 8},
        "tokens": {"input_tokens": 10, "output_tokens": 8},
    },
}

COHERE_EMBEDDING_RESPONSE = {
    "id": "embed-123",
    "embeddings": [[0.1, 0.2, 0.3] + [0.0] * 1021],  # 1024 dims
    "texts": ["test text"],
    "meta": {"api_version": {"version": "1"}, "billed_units": {"input_tokens": 2}},
}

# =============================================================================
# Groq Fixtures
# =============================================================================

GROQ_CHAT_RESPONSE = {
    "id": "chatcmpl-groq123",
    "object": "chat.completion",
    "created": 1704067200,
    "model": "llama-3.1-8b-instant",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! I'm here to help.",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16,
        "queue_time": 0.001,
        "prompt_time": 0.002,
        "completion_time": 0.003,
    },
    "x_groq": {"id": "req_123"},
}

# =============================================================================
# Together AI Fixtures
# =============================================================================

TOGETHER_CHAT_RESPONSE = {
    "id": "chat-together123",
    "object": "chat.completion",
    "created": 1704067200,
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hi there! How can I help you?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 7,
        "total_tokens": 17,
    },
}

# =============================================================================
# DeepSeek Fixtures
# =============================================================================

DEEPSEEK_CHAT_RESPONSE = {
    "id": "chatcmpl-deepseek123",
    "object": "chat.completion",
    "created": 1704067200,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! How can I assist you?",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 6,
        "total_tokens": 16,
    },
}

# =============================================================================
# Error Response Fixtures
# =============================================================================

OPENAI_ERROR_401 = {
    "error": {
        "message": "Incorrect API key provided",
        "type": "invalid_request_error",
        "param": None,
        "code": "invalid_api_key",
    }
}

OPENAI_ERROR_429 = {
    "error": {
        "message": "Rate limit exceeded",
        "type": "rate_limit_error",
        "param": None,
        "code": "rate_limit_exceeded",
    }
}

ANTHROPIC_ERROR_401 = {
    "type": "error",
    "error": {"type": "authentication_error", "message": "Invalid API key"},
}

ANTHROPIC_ERROR_429 = {
    "type": "error",
    "error": {"type": "rate_limit_error", "message": "Rate limit exceeded"},
}


# =============================================================================
# Helper Functions
# =============================================================================


def get_openai_response_bytes() -> bytes:
    """Get OpenAI response as bytes for HTTP mocking."""
    return json.dumps(OPENAI_CHAT_RESPONSE).encode()


def get_anthropic_response_bytes() -> bytes:
    """Get Anthropic response as bytes for HTTP mocking."""
    return json.dumps(ANTHROPIC_CHAT_RESPONSE).encode()


def get_gemini_response_bytes() -> bytes:
    """Get Gemini response as bytes for HTTP mocking."""
    return json.dumps(GEMINI_CHAT_RESPONSE).encode()


def make_streaming_bytes(chunks: list[str]) -> bytes:
    """Convert streaming chunks to bytes."""
    return "".join(chunks).encode()
