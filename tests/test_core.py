"""
Tests for fastlitellm.core module.

Tests the main API functions (completion, acompletion, embedding, aembedding)
using mocked HTTP responses.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from fastlitellm.core import (
    _build_provider_config,
    _get_adapter,
    completion,
    embedding,
    stream_chunk_builder,
)
from fastlitellm.http.client import HTTPResponse
from fastlitellm.types import (
    ChunkChoice,
    ChunkDelta,
    StreamChunk,
    Usage,
)


class TestBuildProviderConfig:
    """Tests for _build_provider_config."""

    def test_default_config(self):
        """Test default configuration."""
        config = _build_provider_config()
        assert config.api_key is None
        assert config.api_base is None
        assert config.timeout == 60.0
        assert config.max_retries == 3

    def test_custom_config(self):
        """Test custom configuration."""
        config = _build_provider_config(
            api_key="test-key",
            api_base="https://custom.api.com",
            timeout=30.0,
            max_retries=5,
        )
        assert config.api_key == "test-key"
        assert config.api_base == "https://custom.api.com"
        assert config.timeout == 30.0
        assert config.max_retries == 5

    def test_base_url_alias(self):
        """Test base_url as alias for api_base."""
        config = _build_provider_config(base_url="https://custom.api.com")
        assert config.api_base == "https://custom.api.com"

    def test_azure_config(self):
        """Test Azure-specific configuration."""
        config = _build_provider_config(
            azure_deployment="my-deployment",
            azure_ad_token="my-ad-token",
        )
        assert config.azure_deployment == "my-deployment"
        assert config.azure_ad_token == "my-ad-token"

    def test_aws_config(self):
        """Test AWS Bedrock configuration."""
        config = _build_provider_config(
            aws_region="us-east-1",
            aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
            aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
        )
        assert config.aws_region == "us-east-1"
        assert config.aws_access_key_id == "AKIAIOSFODNN7EXAMPLE"

    def test_vertex_config(self):
        """Test Vertex AI configuration."""
        config = _build_provider_config(
            vertex_project="my-project",
            vertex_location="us-central1",
        )
        assert config.vertex_project == "my-project"
        assert config.vertex_location == "us-central1"


class TestGetAdapter:
    """Tests for _get_adapter."""

    def test_get_openai_adapter(self):
        """Test getting OpenAI adapter."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter, model_id = _get_adapter("openai/gpt-4o-mini")
            assert adapter.provider_name == "openai"
            assert model_id == "gpt-4o-mini"

    def test_get_adapter_with_inferred_provider(self):
        """Test provider inference from model name."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            adapter, model_id = _get_adapter("gpt-4o-mini")
            assert adapter.provider_name == "openai"
            assert model_id == "gpt-4o-mini"

    def test_get_adapter_with_explicit_provider(self):
        """Test explicit provider override."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            adapter, _model_id = _get_adapter("some-model", provider="anthropic")
            assert adapter.provider_name == "anthropic"


class TestCompletion:
    """Tests for completion function."""

    @pytest.fixture
    def mock_response(self) -> dict[str, Any]:
        """Standard mock completion response."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
            },
        }

    def test_simple_completion(self, mock_response):
        """Test basic completion call."""
        mock_http_response = HTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_response).encode("utf-8"),
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("fastlitellm.core._get_http_client") as mock_client,
        ):
            mock_client.return_value.request.return_value = mock_http_response

            response = completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello!"}],
            )

            assert response.id == "chatcmpl-123"
            assert response.choices[0].message.content == "Hello! How can I help you?"
            assert response.usage is not None
            assert response.usage.total_tokens == 21

    def test_completion_with_temperature(self, mock_response):
        """Test completion with temperature parameter."""
        mock_http_response = HTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_response).encode("utf-8"),
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("fastlitellm.core._get_http_client") as mock_client,
        ):
            mock_client.return_value.request.return_value = mock_http_response

            completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hello!"}],
                temperature=0.7,
            )

            # Verify request was made with temperature
            call_args = mock_client.return_value.request.call_args
            body = json.loads(call_args.kwargs["body"])
            assert body["temperature"] == 0.7

    def test_completion_with_tools(self, mock_response):
        """Test completion with tool definitions."""
        mock_http_response = HTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_response).encode("utf-8"),
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("fastlitellm.core._get_http_client") as mock_client,
        ):
            mock_client.return_value.request.return_value = mock_http_response

            completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Weather?"}],
                tools=tools,
            )

            call_args = mock_client.return_value.request.call_args
            body = json.loads(call_args.kwargs["body"])
            assert "tools" in body


class TestStreamChunkBuilder:
    """Tests for stream_chunk_builder function."""

    def test_basic_chunk_building(self):
        """Test building response from basic chunks."""
        chunks = [
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(role="assistant"),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(content="Hello"),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(content="!"),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(),
                        finish_reason="stop",
                    )
                ],
            ),
        ]

        response = stream_chunk_builder(chunks)

        assert response.id == "chunk-1"
        assert response.model == "gpt-4o-mini"
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content == "Hello!"
        assert response.choices[0].finish_reason == "stop"

    def test_chunk_building_with_usage(self):
        """Test building response with usage in final chunk."""
        chunks = [
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(content="Hi"),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(),
                        finish_reason="stop",
                    )
                ],
                usage=Usage(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            ),
        ]

        response = stream_chunk_builder(chunks)

        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    def test_chunk_building_with_tool_calls(self):
        """Test building response with tool call deltas."""
        chunks = [
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(role="assistant"),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {"name": "get_weather"},
                                }
                            ]
                        ),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"loc'},
                                }
                            ]
                        ),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(
                            tool_calls=[
                                {
                                    "index": 0,
                                    "function": {"arguments": 'ation": "NYC"}'},
                                }
                            ]
                        ),
                    )
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(
                        index=0,
                        delta=ChunkDelta(),
                        finish_reason="tool_calls",
                    )
                ],
            ),
        ]

        response = stream_chunk_builder(chunks)

        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        tc = response.choices[0].message.tool_calls[0]
        assert tc.id == "call_123"
        assert tc.function.name == "get_weather"
        assert tc.function.arguments == '{"location": "NYC"}'

    def test_empty_chunks_raises_error(self):
        """Test that empty chunks list raises ValueError."""
        with pytest.raises(ValueError, match="No chunks provided"):
            stream_chunk_builder([])

    def test_multiple_choices(self):
        """Test building response with multiple choice indices."""
        chunks = [
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(index=0, delta=ChunkDelta(content="A")),
                    ChunkChoice(index=1, delta=ChunkDelta(content="B")),
                ],
            ),
            StreamChunk(
                id="chunk-1",
                model="gpt-4o-mini",
                created=1234567890,
                choices=[
                    ChunkChoice(index=0, delta=ChunkDelta(), finish_reason="stop"),
                    ChunkChoice(index=1, delta=ChunkDelta(), finish_reason="stop"),
                ],
            ),
        ]

        response = stream_chunk_builder(chunks)

        assert len(response.choices) == 2
        assert response.choices[0].message.content == "A"
        assert response.choices[1].message.content == "B"


class TestEmbedding:
    """Tests for embedding function."""

    @pytest.fixture
    def mock_embedding_response(self) -> dict[str, Any]:
        """Standard mock embedding response."""
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3],
                },
                {
                    "object": "embedding",
                    "index": 1,
                    "embedding": [0.4, 0.5, 0.6],
                },
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }

    def test_simple_embedding(self, mock_embedding_response):
        """Test basic embedding call."""
        mock_http_response = HTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_embedding_response).encode("utf-8"),
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("fastlitellm.core._get_http_client") as mock_client,
        ):
            mock_client.return_value.request.return_value = mock_http_response

            response = embedding(
                model="text-embedding-3-small",
                input=["Hello", "World"],
            )

            assert response.model == "text-embedding-3-small"
            assert len(response.data) == 2
            assert response.data[0].embedding == [0.1, 0.2, 0.3]
            assert response.usage.prompt_tokens == 8

    def test_single_string_input(self, mock_embedding_response):
        """Test embedding with single string input."""
        mock_http_response = HTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_embedding_response).encode("utf-8"),
        )

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("fastlitellm.core._get_http_client") as mock_client,
        ):
            mock_client.return_value.request.return_value = mock_http_response

            # Single string should be converted to list
            embedding(
                model="text-embedding-3-small",
                input="Hello",
            )

            call_args = mock_client.return_value.request.call_args
            body = json.loads(call_args.kwargs["body"])
            assert body["input"] == ["Hello"]
