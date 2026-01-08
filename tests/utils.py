"""
Test utilities for arcllm tests.

Provides mock HTTP clients, assertion helpers, and test decorators.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class MockHTTPResponse:
    """Mock HTTP response for testing."""

    status_code: int
    body: bytes
    headers: dict[str, str]
    request_id: str | None = None

    def iter_bytes(self, chunk_size: int = 8192) -> Iterator[bytes]:
        """Iterate over response body in chunks."""
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]


class MockHTTPClient:
    """Mock HTTP client for testing provider adapters."""

    def __init__(
        self,
        responses: list[MockHTTPResponse] | None = None,
        default_response: MockHTTPResponse | None = None,
    ):
        self.responses = responses or []
        self.default_response = default_response
        self.requests: list[dict[str, Any]] = []
        self._response_index = 0

    def request(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        timeout: float | None = None,
        stream: bool = False,
    ) -> MockHTTPResponse:
        """Record request and return mock response."""
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers or {},
                "body": body,
                "timeout": timeout,
                "stream": stream,
            }
        )

        if self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
            return response

        if self.default_response:
            return self.default_response

        raise ValueError("No mock response configured")

    def get_last_request(self) -> dict[str, Any] | None:
        """Get the most recent request."""
        return self.requests[-1] if self.requests else None

    def get_last_request_body(self) -> dict[str, Any] | None:
        """Get the parsed JSON body of the last request."""
        last = self.get_last_request()
        if last and last.get("body"):
            return json.loads(last["body"])
        return None


def make_mock_response(
    data: dict[str, Any],
    status_code: int = 200,
    headers: dict[str, str] | None = None,
) -> MockHTTPResponse:
    """Create a mock HTTP response from a dict."""
    return MockHTTPResponse(
        status_code=status_code,
        body=json.dumps(data).encode(),
        headers=headers or {"content-type": "application/json"},
    )


def make_streaming_response(
    chunks: list[str],
    status_code: int = 200,
) -> MockHTTPResponse:
    """Create a mock streaming HTTP response."""
    return MockHTTPResponse(
        status_code=status_code,
        body="".join(chunks).encode(),
        headers={"content-type": "text/event-stream"},
    )


# =============================================================================
# Assertion Helpers
# =============================================================================


def assert_model_response_valid(response: Any) -> None:
    """Assert that a ModelResponse has the expected structure."""
    from arcllm.types import ModelResponse

    assert isinstance(response, ModelResponse)
    assert response.id is not None
    assert response.model is not None
    assert response.choices is not None
    assert len(response.choices) > 0

    choice = response.choices[0]
    assert choice.message is not None
    assert choice.message.role == "assistant"


def assert_tool_calls_valid(response: Any) -> None:
    """Assert that tool calls in response are properly formatted."""
    from arcllm.types import ModelResponse, ToolCall

    assert isinstance(response, ModelResponse)
    choice = response.choices[0]
    assert choice.message.tool_calls is not None
    assert len(choice.message.tool_calls) > 0

    for tc in choice.message.tool_calls:
        assert isinstance(tc, ToolCall)
        assert tc.id is not None
        assert tc.type == "function"
        assert tc.function is not None
        assert tc.function.name is not None
        # Arguments should be a JSON string
        assert isinstance(tc.function.arguments, str)
        # Should be parseable
        json.loads(tc.function.arguments)


def assert_usage_valid(response: Any) -> None:
    """Assert that usage information is present and valid."""
    from arcllm.types import ModelResponse

    assert isinstance(response, ModelResponse)
    assert response.usage is not None
    assert response.usage.prompt_tokens >= 0
    assert response.usage.completion_tokens >= 0
    assert response.usage.total_tokens >= 0
    # Total should be sum (when both are positive)
    if response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0:
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )


def assert_embedding_response_valid(response: Any) -> None:
    """Assert that an EmbeddingResponse has the expected structure."""
    from arcllm.types import EmbeddingResponse

    assert isinstance(response, EmbeddingResponse)
    assert response.model is not None
    assert response.data is not None
    assert len(response.data) > 0
    assert response.data[0].embedding is not None
    assert len(response.data[0].embedding) > 0
    assert all(isinstance(v, float) for v in response.data[0].embedding)


def assert_stream_chunks_valid(chunks: list[Any]) -> None:
    """Assert that streaming chunks are valid."""
    from arcllm.types import StreamChunk

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, StreamChunk)
        assert chunk.id is not None
        assert chunk.choices is not None


# =============================================================================
# Test Decorators
# =============================================================================


def requires_env(*env_vars: str):
    """Skip test if required environment variables are not set."""

    def decorator(func):
        missing = [v for v in env_vars if not os.environ.get(v)]
        if missing:
            return pytest.mark.skip(reason=f"Missing required env vars: {', '.join(missing)}")(func)
        return func

    return decorator


def integration_test(func):
    """Mark a test as an integration test."""
    return pytest.mark.integration(func)


def slow_test(func):
    """Mark a test as slow."""
    return pytest.mark.slow(func)


# =============================================================================
# Request Sanitization (for CI logs)
# =============================================================================


def redact_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive data from a dict for safe logging."""
    sensitive_keys = {
        "api_key",
        "api-key",
        "authorization",
        "x-api-key",
        "bearer",
        "token",
        "secret",
        "password",
    }

    def redact_value(key: str, value: Any) -> Any:
        key_lower = key.lower()
        if any(s in key_lower for s in sensitive_keys):
            if isinstance(value, str) and len(value) > 8:
                return value[:4] + "..." + value[-4:]
            return "[REDACTED]"
        if isinstance(value, dict):
            return {k: redact_value(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [
                redact_value(str(i), v) if isinstance(v, dict) else v for i, v in enumerate(value)
            ]
        return value

    return {k: redact_value(k, v) for k, v in data.items()}


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    """Redact sensitive headers for safe logging."""
    return {k: redact_secrets({k: v})[k] for k, v in headers.items()}


# =============================================================================
# Common Test Messages
# =============================================================================

SIMPLE_MESSAGES = [{"role": "user", "content": "Say hello in one word."}]

SYSTEM_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant. Be concise."},
    {"role": "user", "content": "What is 2+2?"},
]

TOOL_MESSAGES = [{"role": "user", "content": "What's the weather in San Francisco?"}]

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature",
                },
            },
            "required": ["location"],
        },
    },
}

MATH_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate",
                }
            },
            "required": ["expression"],
        },
    },
}

JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    },
}

JSON_MODE = {"type": "json_object"}
