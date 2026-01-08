"""
Groq adapter for arcllm.

Groq provides an OpenAI-compatible API for fast inference.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arcllm.exceptions import UnsupportedModelError
from arcllm.providers.base import (
    ProviderConfig,
    RequestData,
    register_provider,
)
from arcllm.providers.openai_adapter import OpenAIAdapter

if TYPE_CHECKING:
    from arcllm.types import EmbeddingResponse

__all__ = ["GroqAdapter"]


class GroqAdapter(OpenAIAdapter):
    """
    Adapter for Groq API.

    Groq uses OpenAI-compatible format.
    """

    provider_name = "groq"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.groq.com/openai/v1"

    def _get_headers(self) -> dict[str, str]:
        """Get request headers."""
        api_key = self._get_api_key("GROQ_API_KEY")
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        if self.config.extra_headers:
            headers.update(self.config.extra_headers)
        return headers

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Groq does not support embeddings."""
        raise UnsupportedModelError(
            "Groq does not provide an embeddings API",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Groq does not support embeddings."""
        raise UnsupportedModelError(
            "Groq does not provide an embeddings API",
            provider=self.provider_name,
        )


# Register on import
register_provider("groq", GroqAdapter)
