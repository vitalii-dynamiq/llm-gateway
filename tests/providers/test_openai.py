"""
Tests for OpenAI provider adapter.
"""

import json
import os
from unittest.mock import patch

import pytest

from fastlitellm.exceptions import (
    AuthenticationError,
    RateLimitError,
    ResponseParseError,
)
from fastlitellm.providers.base import ProviderConfig
from fastlitellm.providers.openai_adapter import OpenAIAdapter


class TestOpenAIAdapter:
    """Tests for OpenAIAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test config."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            config = ProviderConfig()
            return OpenAIAdapter(config)

    @pytest.fixture
    def adapter_with_key(self):
        """Create adapter with explicit key."""
        config = ProviderConfig(api_key="sk-test-key-123")
        return OpenAIAdapter(config)


class TestBuildRequest:
    """Tests for building requests."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-test-key")
        return OpenAIAdapter(config)

    def test_build_simple_request(self, adapter):
        """Test building a simple chat request."""
        messages = [{"role": "user", "content": "Hello"}]
        request = adapter.build_request(
            model="gpt-4o-mini",
            messages=messages,
            stream=False
        )

        assert request.method == "POST"
        assert "chat/completions" in request.url
        assert request.headers["Authorization"] == "Bearer sk-test-key"
        assert request.headers["Content-Type"] == "application/json"

        body = json.loads(request.body.decode("utf-8"))
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == messages
        assert body["stream"] is False

    def test_build_request_with_temperature(self, adapter):
        """Test building request with temperature."""
        request = adapter.build_request(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.7
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["temperature"] == 0.7

    def test_build_request_with_max_tokens(self, adapter):
        """Test building request with max_tokens."""
        request = adapter.build_request(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=100
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["max_tokens"] == 100

    def test_build_request_with_tools(self, adapter, tool_definition):
        """Test building request with tools."""
        request = adapter.build_request(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=tool_definition
        )

        body = json.loads(request.body.decode("utf-8"))
        assert "tools" in body
        assert len(body["tools"]) == 1
        assert body["tools"][0]["function"]["name"] == "get_weather"

    def test_build_request_with_response_format(self, adapter):
        """Test building request with response_format."""
        request = adapter.build_request(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Give me JSON"}],
            response_format={"type": "json_object"}
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["response_format"]["type"] == "json_object"

    def test_build_streaming_request(self, adapter):
        """Test building streaming request."""
        request = adapter.build_request(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            stream_options={"include_usage": True}
        )

        body = json.loads(request.body.decode("utf-8"))
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True

    def test_build_request_o1_model(self, adapter):
        """Test building request for o1 model (uses max_completion_tokens)."""
        request = adapter.build_request(
            model="o1-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1000
        )

        body = json.loads(request.body.decode("utf-8"))
        # o1 models should use max_completion_tokens
        assert body.get("max_completion_tokens") == 1000
        assert "max_tokens" not in body


class TestParseResponse:
    """Tests for parsing responses."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-test-key")
        return OpenAIAdapter(config)

    def test_parse_simple_response(self, adapter, openai_completion_response):
        """Test parsing a simple completion response."""
        data = json.dumps(openai_completion_response).encode("utf-8")
        response = adapter.parse_response(data, "gpt-4o-mini")

        assert response.id == "chatcmpl-123"
        assert response.model == "gpt-4o-mini"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello! How can I help you today?"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 9
        assert response.usage.completion_tokens == 12
        assert response.usage.total_tokens == 21

    def test_parse_tool_call_response(self, adapter, openai_tool_call_response):
        """Test parsing response with tool calls."""
        data = json.dumps(openai_tool_call_response).encode("utf-8")
        response = adapter.parse_response(data, "gpt-4o-mini")

        assert response.choices[0].message.content is None
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1

        tc = response.choices[0].message.tool_calls[0]
        assert tc.id == "call_abc123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"location": "San Francisco", "unit": "celsius"}'

    def test_parse_response_invalid_json(self, adapter):
        """Test parsing invalid JSON raises error."""
        with pytest.raises(ResponseParseError):
            adapter.parse_response(b"not json", "gpt-4o-mini")

    def test_model_extra_contains_usage(self, adapter, openai_completion_response):
        """Test model_extra contains usage."""
        data = json.dumps(openai_completion_response).encode("utf-8")
        response = adapter.parse_response(data, "gpt-4o-mini")

        assert "usage" in response.model_extra
        assert response.model_extra["usage"]["total_tokens"] == 21


class TestParseStreamEvent:
    """Tests for parsing streaming events."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-test-key")
        return OpenAIAdapter(config)

    def test_parse_content_chunk(self, adapter):
        """Test parsing content chunk."""
        event_data = '{"id":"1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}'
        chunk = adapter.parse_stream_event(event_data, "gpt-4o-mini")

        assert chunk is not None
        assert chunk.choices[0].delta.content == "Hello"
        assert chunk.choices[0].finish_reason is None

    def test_parse_role_chunk(self, adapter):
        """Test parsing role chunk (first chunk)."""
        event_data = '{"id":"1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}'
        chunk = adapter.parse_stream_event(event_data, "gpt-4o-mini")

        assert chunk is not None
        assert chunk.choices[0].delta.role == "assistant"

    def test_parse_finish_chunk(self, adapter):
        """Test parsing finish chunk."""
        event_data = '{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}'
        chunk = adapter.parse_stream_event(event_data, "gpt-4o-mini")

        assert chunk is not None
        assert chunk.choices[0].finish_reason == "stop"

    def test_parse_done_event(self, adapter):
        """Test [DONE] event returns None."""
        chunk = adapter.parse_stream_event("[DONE]", "gpt-4o-mini")
        assert chunk is None

    def test_parse_empty_event(self, adapter):
        """Test empty event returns None."""
        chunk = adapter.parse_stream_event("", "gpt-4o-mini")
        assert chunk is None

    def test_parse_tool_call_chunk(self, adapter):
        """Test parsing tool call in streaming."""
        event_data = '''{"id":"1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"get_weather","arguments":"{\\"loc"}}]},"finish_reason":null}]}'''
        chunk = adapter.parse_stream_event(event_data, "gpt-4o-mini")

        assert chunk is not None
        assert chunk.choices[0].delta.tool_calls is not None
        assert chunk.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"

    def test_parse_usage_in_stream(self, adapter):
        """Test parsing usage in final stream chunk."""
        event_data = '{"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}'
        chunk = adapter.parse_stream_event(event_data, "gpt-4o-mini")

        assert chunk is not None
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 10
        assert chunk.usage.total_tokens == 15


class TestParseError:
    """Tests for parsing errors."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-test-key")
        return OpenAIAdapter(config)

    def test_parse_401_error(self, adapter):
        """Test parsing 401 authentication error."""
        error_data = json.dumps({
            "error": {
                "message": "Invalid API Key",
                "type": "invalid_request_error"
            }
        }).encode("utf-8")

        error = adapter.parse_error(401, error_data, "req-123")
        assert isinstance(error, AuthenticationError)
        assert "Invalid API Key" in str(error)

    def test_parse_429_error(self, adapter):
        """Test parsing 429 rate limit error."""
        error_data = json.dumps({
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error"
            }
        }).encode("utf-8")

        error = adapter.parse_error(429, error_data, "req-123")
        assert isinstance(error, RateLimitError)

    def test_parse_non_json_error(self, adapter):
        """Test parsing non-JSON error response."""
        error = adapter.parse_error(500, b"Internal Server Error", "req-123")
        assert "Internal Server Error" in str(error)


class TestEmbeddings:
    """Tests for embedding requests."""

    @pytest.fixture
    def adapter(self):
        """Create adapter with test key."""
        config = ProviderConfig(api_key="sk-test-key")
        return OpenAIAdapter(config)

    def test_build_embedding_request(self, adapter):
        """Test building embedding request."""
        request = adapter.build_embedding_request(
            model="text-embedding-3-small",
            input=["Hello", "World"]
        )

        assert request.method == "POST"
        assert "embeddings" in request.url

        body = json.loads(request.body.decode("utf-8"))
        assert body["model"] == "text-embedding-3-small"
        assert body["input"] == ["Hello", "World"]

    def test_parse_embedding_response(self, adapter, openai_embedding_response):
        """Test parsing embedding response."""
        data = json.dumps(openai_embedding_response).encode("utf-8")
        response = adapter.parse_embedding_response(data, "text-embedding-3-small")

        assert response.model == "text-embedding-3-small"
        assert len(response.data) == 2
        assert len(response.data[0].embedding) == 3
        assert response.usage.prompt_tokens == 8
