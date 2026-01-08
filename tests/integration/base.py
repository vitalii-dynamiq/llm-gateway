"""
Base class and utilities for integration tests.

Provides common test patterns, retry logic, and assertion helpers
for all provider integration tests.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import pytest


class IntegrationTestBase:
    """Base class for provider integration tests."""

    # Override in subclasses
    PROVIDER: str = ""
    ENV_VAR: str = ""
    PRIMARY_MODEL: str = ""
    EMBEDDING_MODEL: str | None = None
    SUPPORTS_TOOLS: bool = True
    SUPPORTS_STRUCTURED_OUTPUT: bool = True
    SUPPORTS_STREAMING: bool = True
    SUPPORTS_EMBEDDINGS: bool = False

    # Retry configuration
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 2.0
    TIMEOUT: float = 60.0

    @classmethod
    def setup_class(cls) -> None:
        """Check if credentials are available."""
        if not os.environ.get(cls.ENV_VAR):
            pytest.skip(f"Missing {cls.ENV_VAR} environment variable")

    def retry_on_rate_limit(self, func, *args, **kwargs) -> Any:
        """Execute function with retry on rate limit errors."""
        from fastlitellm.exceptions import RateLimitError

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitError as e:
                last_error = e
                wait_time = self.RETRY_DELAY * (2**attempt)
                print(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                time.sleep(wait_time)
        raise last_error

    async def retry_on_rate_limit_async(self, func, *args, **kwargs) -> Any:
        """Execute async function with retry on rate limit errors."""
        import asyncio

        from fastlitellm.exceptions import RateLimitError

        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                last_error = e
                wait_time = self.RETRY_DELAY * (2**attempt)
                print(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
        raise last_error

    # =========================================================================
    # Basic Completion Tests
    # =========================================================================

    def test_simple_completion(self) -> None:
        """Test basic chat completion."""
        from fastlitellm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message is not None
        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.asyncio
    async def test_async_completion(self) -> None:
        """Test async chat completion."""
        from fastlitellm import acompletion

        response = await self.retry_on_rate_limit_async(
            acompletion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'hi' and nothing else."}],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices is not None
        assert len(response.choices) > 0

    # =========================================================================
    # Streaming Tests
    # =========================================================================

    def test_streaming_completion(self) -> None:
        """Test streaming chat completion."""
        if not self.SUPPORTS_STREAMING:
            pytest.skip(f"{self.PROVIDER} does not support streaming")

        from fastlitellm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            max_tokens=50,
            stream=True,
        )

        chunks = []
        content_parts = []

        for chunk in response:
            chunks.append(chunk)
            if (
                chunk.choices
                and chunk.choices[0].delta
                and chunk.choices[0].delta.content
            ):
                content_parts.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0, "Expected at least one chunk"
        assert len(content_parts) > 0, "Expected some content deltas"

        full_content = "".join(content_parts)
        assert len(full_content) > 0, "Expected non-empty assembled content"

    def test_streaming_with_chunk_builder(self) -> None:
        """Test streaming with stream_chunk_builder."""
        if not self.SUPPORTS_STREAMING:
            pytest.skip(f"{self.PROVIDER} does not support streaming")

        from fastlitellm import completion, stream_chunk_builder

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say 'test'."}],
            max_tokens=10,
            stream=True,
        )

        chunks = list(response)
        final_response = stream_chunk_builder(chunks)

        assert final_response is not None
        assert final_response.choices is not None
        assert len(final_response.choices) > 0
        assert final_response.choices[0].message.content is not None

    # =========================================================================
    # Tool Calling Tests
    # =========================================================================

    def test_tool_calling(self) -> None:
        """Test tool calling capability."""
        if not self.SUPPORTS_TOOLS:
            pytest.skip(f"{self.PROVIDER} does not support tool calling")

        from fastlitellm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"}
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
        )

        assert response is not None
        # Model should either respond with tool call or text
        choice = response.choices[0]
        # Check if tool calls present or regular content
        has_tool_calls = (
            choice.message.tool_calls is not None
            and len(choice.message.tool_calls) > 0
        )
        has_content = choice.message.content is not None

        assert has_tool_calls or has_content, "Expected tool calls or content"

        if has_tool_calls:
            tool_call = choice.message.tool_calls[0]
            assert tool_call.function is not None
            assert tool_call.function.name == "get_weather"
            # Arguments should be valid JSON
            args = json.loads(tool_call.function.arguments)
            assert "location" in args

    # =========================================================================
    # Structured Output Tests
    # =========================================================================

    def test_structured_output_json_mode(self) -> None:
        """Test JSON mode structured output."""
        if not self.SUPPORTS_STRUCTURED_OUTPUT:
            pytest.skip(f"{self.PROVIDER} does not support structured output")

        from fastlitellm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with fields 'name' (string) and 'age' (number). Example: {\"name\": \"Alice\", \"age\": 30}",
                }
            ],
            response_format={"type": "json_object"},
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None

        # Should be valid JSON
        try:
            data = json.loads(content)
            assert isinstance(data, dict)
        except json.JSONDecodeError:
            pytest.fail(f"Response is not valid JSON: {content}")

    # =========================================================================
    # Usage Tests
    # =========================================================================

    def test_usage_reporting(self) -> None:
        """Test that usage information is reported."""
        from fastlitellm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5,
        )

        assert response is not None
        assert response.usage is not None
        assert response.usage.prompt_tokens >= 0
        assert response.usage.completion_tokens >= 0
        assert response.usage.total_tokens >= 0

    # =========================================================================
    # Embedding Tests
    # =========================================================================

    def test_embeddings(self) -> None:
        """Test embedding generation."""
        if not self.SUPPORTS_EMBEDDINGS or not self.EMBEDDING_MODEL:
            pytest.skip(f"{self.PROVIDER} does not support embeddings")

        from fastlitellm import embedding

        response = self.retry_on_rate_limit(
            embedding,
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=["Hello, world!"],
        )

        assert response is not None
        assert response.data is not None
        assert len(response.data) > 0
        assert response.data[0].embedding is not None
        assert len(response.data[0].embedding) > 0
        assert all(isinstance(v, float) for v in response.data[0].embedding)

    # =========================================================================
    # Error Handling Tests
    # =========================================================================

    def test_invalid_model_error(self) -> None:
        """Test that invalid model raises appropriate error."""
        from fastlitellm import completion
        from fastlitellm.exceptions import FastLiteLLMError

        with pytest.raises(FastLiteLLMError):
            completion(
                model=f"{self.PROVIDER}/nonexistent-model-xyz123",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
            )
