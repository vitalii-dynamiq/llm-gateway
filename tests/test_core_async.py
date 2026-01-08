"""
Tests for async functionality in arcllm.core module.

Tests acompletion and aembedding with mocked async HTTP.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from arcllm.core import acompletion, aembedding
from arcllm.http.async_client import AsyncHTTPResponse


class TestACompletion:
    """Tests for acompletion function."""

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
                        "content": "Hello!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 2,
                "total_tokens": 11,
            },
        }

    @pytest.mark.asyncio
    async def test_simple_async_completion(self, mock_response):
        """Test basic async completion call."""
        mock_http_response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_response).encode("utf-8"),
        )

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_http_response

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("arcllm.core._get_async_http_client", return_value=mock_client),
        ):
            response = await acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi!"}],
            )

            assert response.id == "chatcmpl-123"
            assert response.choices[0].message.content == "Hello!"
            assert response.usage is not None

    @pytest.mark.asyncio
    async def test_async_completion_with_params(self, mock_response):
        """Test async completion with additional parameters."""
        mock_http_response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_response).encode("utf-8"),
        )

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_http_response

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("arcllm.core._get_async_http_client", return_value=mock_client),
        ):
            await acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi!"}],
                temperature=0.7,
                max_tokens=100,
            )

            # Verify request was made
            mock_client.request.assert_called_once()


class TestAEmbedding:
    """Tests for aembedding function."""

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
                }
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }

    @pytest.mark.asyncio
    async def test_simple_async_embedding(self, mock_embedding_response):
        """Test basic async embedding call."""
        mock_http_response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_embedding_response).encode("utf-8"),
        )

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_http_response

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("arcllm.core._get_async_http_client", return_value=mock_client),
        ):
            response = await aembedding(
                model="text-embedding-3-small",
                input="Hello world",
            )

            assert response.model == "text-embedding-3-small"
            assert len(response.data) == 1

    @pytest.mark.asyncio
    async def test_async_embedding_list_input(self, mock_embedding_response):
        """Test async embedding with list input."""
        mock_embedding_response["data"].append(
            {
                "object": "embedding",
                "index": 1,
                "embedding": [0.4, 0.5, 0.6],
            }
        )

        mock_http_response = AsyncHTTPResponse(
            status_code=200,
            headers={},
            body=json.dumps(mock_embedding_response).encode("utf-8"),
        )

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_http_response

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("arcllm.core._get_async_http_client", return_value=mock_client),
        ):
            response = await aembedding(
                model="text-embedding-3-small",
                input=["Hello", "World"],
            )

            assert len(response.data) == 2


class TestAsyncStreaming:
    """Tests for async streaming."""

    @pytest.mark.asyncio
    async def test_async_streaming_basic(self):
        """Test basic async streaming."""

        async def mock_stream():
            chunks = [
                b'data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}\n\n',
                b'data: {"id":"1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n',
                b'data: {"id":"1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                b"data: [DONE]\n\n",
            ]
            for chunk in chunks:
                yield chunk

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_stream()

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("arcllm.core._get_async_http_client", return_value=mock_client),
        ):
            result = await acompletion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

            chunks = []
            async for chunk in result:
                chunks.append(chunk)

            assert len(chunks) == 3  # Excludes [DONE]
