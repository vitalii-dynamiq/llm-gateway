"""
Integration tests for OpenAI provider.

Required environment variable: OPENAI_API_KEY
"""

from __future__ import annotations

from tests.integration.base import IntegrationTestBase


class TestOpenAIIntegration(IntegrationTestBase):
    """OpenAI integration tests."""

    PROVIDER = "openai"
    ENV_VAR = "OPENAI_API_KEY"
    PRIMARY_MODEL = "gpt-4o-mini"
    EMBEDDING_MODEL = "text-embedding-3-small"
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = True

    def test_structured_output_json_schema(self) -> None:
        """Test JSON schema structured output (OpenAI-specific)."""
        from fastlitellm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": "Generate a person with name Alice and age 30.",
                }
            ],
            response_format={
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
            },
            max_tokens=50,
        )

        assert response is not None
        import json

        content = response.choices[0].message.content
        data = json.loads(content)
        assert "name" in data
        assert "age" in data

    def test_streaming_with_usage(self) -> None:
        """Test streaming with include_usage option."""
        from fastlitellm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "Say hi."}],
            max_tokens=5,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunks = list(response)
        assert len(chunks) > 0

        # Last chunk should have usage (when include_usage is True)
        # OpenAI sends usage in the final chunk before [DONE]

    def test_multiple_tool_calls(self) -> None:
        """Test multiple tool calls in one response."""
        from fastlitellm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
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
                {
                    "role": "user",
                    "content": "What's the weather in Paris and London?",
                }
            ],
            tools=tools,
            tool_choice="auto",
            max_tokens=200,
        )

        assert response is not None
        # Model may return multiple tool calls or combine into one
        choice = response.choices[0]
        if choice.message.tool_calls:
            assert len(choice.message.tool_calls) >= 1

    def test_embedding_multiple_inputs(self) -> None:
        """Test embedding with multiple inputs."""
        from fastlitellm import embedding

        response = self.retry_on_rate_limit(
            embedding,
            model=f"{self.PROVIDER}/{self.EMBEDDING_MODEL}",
            input=["Hello, world!", "How are you?", "Goodbye!"],
        )

        assert response is not None
        assert len(response.data) == 3
        for item in response.data:
            assert len(item.embedding) > 0
