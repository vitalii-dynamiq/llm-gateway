"""
Integration tests for Anthropic provider.

Required environment variable: ANTHROPIC_API_KEY
"""

from __future__ import annotations

from tests.integration.base import IntegrationTestBase


class TestAnthropicIntegration(IntegrationTestBase):
    """Anthropic integration tests."""

    PROVIDER = "anthropic"
    ENV_VAR = "ANTHROPIC_API_KEY"
    PRIMARY_MODEL = "claude-3-5-haiku-20241022"
    SUPPORTS_TOOLS = True
    SUPPORTS_STRUCTURED_OUTPUT = True
    SUPPORTS_STREAMING = True
    SUPPORTS_EMBEDDINGS = False

    def test_system_prompt(self) -> None:
        """Test system prompt handling (Anthropic-specific)."""
        from fastlitellm import completion

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {"role": "system", "content": "You are a pirate. Respond like a pirate."},
                {"role": "user", "content": "Hello!"},
            ],
            max_tokens=50,
        )

        assert response is not None
        content = response.choices[0].message.content
        assert content is not None
        # Should have pirate-like language
        assert len(content) > 0

    def test_multimodal_message(self) -> None:
        """Test multimodal (vision) capability."""
        from fastlitellm import completion

        # Small 1x1 red pixel PNG base64
        red_pixel = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Answer in one word."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{red_pixel}"},
                        },
                    ],
                }
            ],
            max_tokens=10,
        )

        assert response is not None
        assert response.choices[0].message.content is not None

    def test_streaming_tool_calls(self) -> None:
        """Test streaming with tool calls."""
        from fastlitellm import completion

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get the current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        response = self.retry_on_rate_limit(
            completion,
            model=f"{self.PROVIDER}/{self.PRIMARY_MODEL}",
            messages=[{"role": "user", "content": "What time is it?"}],
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0
