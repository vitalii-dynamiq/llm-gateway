"""
Base adapter protocol and provider registry.

All provider adapters implement the Adapter protocol which defines:
- build_request: Convert OpenAI-like params to provider HTTP request
- parse_response: Convert provider response to ModelResponse
- parse_stream_event: Convert streaming event to StreamChunk
- parse_error: Convert provider error to FastLiteLLMError
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fastlitellm.exceptions import (
    FastLiteLLMError,
    UnsupportedModelError,
    UnsupportedParameterError,
)

if TYPE_CHECKING:
    from fastlitellm.types import (
        EmbeddingResponse,
        ModelResponse,
        StreamChunk,
    )

__all__ = [
    "SUPPORTED_PROVIDERS",
    "Adapter",
    "ProviderConfig",
    "RequestData",
    "get_provider",
    "parse_model_string",
    "register_provider",
]


# Provider registry
_PROVIDERS: dict[str, type[Adapter]] = {}

# List of supported provider prefixes
SUPPORTED_PROVIDERS = [
    "openai",
    "azure",
    "anthropic",
    "gemini",
    "vertex_ai",
    "bedrock",
    "mistral",
    "cohere",
    "groq",
    "together_ai",
    "fireworks_ai",
    "deepseek",
    "perplexity",
    "databricks",
    "ollama",
]


@dataclass(slots=True)
class ProviderConfig:
    """Configuration for a provider adapter."""

    api_key: str | None = None
    api_base: str | None = None
    api_version: str | None = None
    organization: str | None = None
    project: str | None = None
    timeout: float = 60.0
    max_retries: int = 3
    # Azure-specific
    azure_deployment: str | None = None
    azure_ad_token: str | None = None
    # AWS Bedrock-specific
    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    # Vertex AI-specific
    vertex_project: str | None = None
    vertex_location: str | None = None
    # Custom headers
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RequestData:
    """HTTP request data produced by adapter."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None = None
    timeout: float = 60.0


# Common parameters supported by most providers
COMMON_PARAMS = {
    "model",
    "messages",
    "stream",
    "temperature",
    "top_p",
    "max_tokens",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "tools",
    "tool_choice",
    "response_format",
    "n",
    "logprobs",
    "top_logprobs",
    "user",
}


def parse_model_string(model: str) -> tuple[str, str]:
    """
    Parse a model string into (provider, model_id).

    Supports formats:
    - "provider/model" -> (provider, model)
    - "model" -> (inferred_provider, model)

    Provider inference:
    - gpt-* -> openai
    - claude-* -> anthropic
    - gemini-* -> gemini
    - mistral-* -> mistral
    - command-* -> cohere
    - llama-* -> together_ai (or groq)
    - deepseek-* -> deepseek
    """
    if "/" in model:
        parts = model.split("/", 1)
        provider = parts[0].lower().replace("-", "_")
        model_id = parts[1]
        return provider, model_id

    # Infer provider from model name
    model_lower = model.lower()
    if model_lower.startswith("gpt-") or model_lower.startswith("o1"):
        return "openai", model
    elif model_lower.startswith("claude-"):
        return "anthropic", model
    elif model_lower.startswith("gemini-"):
        return "gemini", model
    elif model_lower.startswith("mistral-") or model_lower.startswith("codestral"):
        return "mistral", model
    elif model_lower.startswith("command-"):
        return "cohere", model
    elif model_lower.startswith("deepseek-"):
        return "deepseek", model
    elif model_lower.startswith("llama-") or model_lower.startswith("llama3"):
        return "groq", model  # Default to Groq for Llama
    elif model_lower.startswith("pplx-"):
        return "perplexity", model

    # Default to OpenAI if can't infer
    return "openai", model


@runtime_checkable
class Adapter(Protocol):
    """
    Protocol defining the interface for provider adapters.

    Each provider must implement these methods to handle request/response
    transformation and error mapping.
    """

    # Provider identifier
    provider_name: str

    def __init__(self, config: ProviderConfig) -> None:
        """Initialize adapter with configuration."""
        ...

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """
        Build HTTP request from completion parameters.

        Args:
            model: Model identifier (without provider prefix)
            messages: OpenAI-format messages list
            stream: Whether to request streaming response
            drop_params: If True, silently drop unsupported params
            **kwargs: Additional parameters

        Returns:
            RequestData with method, url, headers, body

        Raises:
            UnsupportedParameterError: If drop_params=False and unsupported params present
        """
        ...

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """
        Parse HTTP response body into ModelResponse.

        Args:
            data: Raw response bytes
            model: Model identifier for response

        Returns:
            ModelResponse with choices, usage, etc.

        Raises:
            ResponseParseError: If response cannot be parsed
        """
        ...

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """
        Parse a single SSE event data into StreamChunk.

        Args:
            data: The 'data' field from SSE event
            model: Model identifier

        Returns:
            StreamChunk or None if event should be skipped (e.g., [DONE])
        """
        ...

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> FastLiteLLMError:
        """
        Parse error response into appropriate exception.

        Args:
            status_code: HTTP status code
            data: Response body bytes
            request_id: Request ID if available

        Returns:
            Appropriate FastLiteLLMError subclass
        """
        ...

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """
        Build HTTP request for embedding.

        Args:
            model: Model identifier
            input: List of texts to embed
            **kwargs: Additional parameters

        Returns:
            RequestData with method, url, headers, body
        """
        ...

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """
        Parse embedding response.

        Args:
            data: Raw response bytes
            model: Model identifier

        Returns:
            EmbeddingResponse with embeddings and usage
        """
        ...


class BaseAdapter(ABC):
    """
    Base class for provider adapters with common functionality.

    Subclasses must implement the abstract methods for provider-specific logic.
    """

    provider_name: str = "base"

    # Parameters supported by this provider (override in subclass)
    supported_params: set[str] = COMMON_PARAMS

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    def _check_params(
        self,
        drop_params: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Check and filter parameters based on provider support.

        Args:
            drop_params: If True, silently drop unsupported params
            **kwargs: All parameters passed to the request

        Returns:
            Filtered kwargs dict

        Raises:
            UnsupportedParameterError: If drop_params=False and unsupported params
        """
        # Always include these core params
        core_params = {"model", "messages", "stream"}
        unsupported = set(kwargs.keys()) - self.supported_params - core_params

        if unsupported:
            if drop_params:
                # Remove unsupported params
                return {k: v for k, v in kwargs.items() if k in self.supported_params}
            else:
                raise UnsupportedParameterError(
                    f"Unsupported parameters for {self.provider_name}: {sorted(unsupported)}",
                    provider=self.provider_name,
                    unsupported_params=list(unsupported),
                )
        return kwargs

    def _get_api_key(self, env_var: str, param_key: str = "api_key") -> str:
        """Get API key from config or environment."""
        key = self.config.api_key or os.environ.get(env_var)
        if not key:
            raise FastLiteLLMError(
                f"API key not provided. Set {env_var} or pass api_key parameter.",
                provider=self.provider_name,
            )
        return key

    @abstractmethod
    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build HTTP request - must be implemented by subclass."""
        ...

    @abstractmethod
    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse response - must be implemented by subclass."""
        ...

    @abstractmethod
    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse stream event - must be implemented by subclass."""
        ...

    @abstractmethod
    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> FastLiteLLMError:
        """Parse error - must be implemented by subclass."""
        ...

    def build_embedding_request(
        self,
        *,
        model: str,
        input: list[str],
        **kwargs: Any,
    ) -> RequestData:
        """Build embedding request - override in subclass if supported."""
        raise UnsupportedModelError(
            f"Embeddings not supported by {self.provider_name}",
            provider=self.provider_name,
        )

    def parse_embedding_response(self, data: bytes, model: str) -> EmbeddingResponse:
        """Parse embedding response - override in subclass if supported."""
        raise UnsupportedModelError(
            f"Embeddings not supported by {self.provider_name}",
            provider=self.provider_name,
        )


def register_provider(name: str, adapter_class: type[Adapter]) -> None:
    """Register a provider adapter class."""
    _PROVIDERS[name.lower()] = adapter_class


def get_provider(name: str, config: ProviderConfig | None = None) -> Adapter:
    """
    Get an adapter instance for the given provider.

    Args:
        name: Provider name (e.g., "openai", "anthropic")
        config: Optional provider configuration

    Returns:
        Adapter instance

    Raises:
        UnsupportedModelError: If provider is not supported
    """
    name = name.lower()
    if name not in _PROVIDERS:
        raise UnsupportedModelError(
            f"Provider '{name}' is not supported. Supported providers: {list(_PROVIDERS.keys())}",
            provider=name,
        )

    adapter_class = _PROVIDERS[name]
    return adapter_class(config or ProviderConfig())


# Import provider modules to trigger registration
# This is done at the bottom to avoid circular imports
def _register_all_providers() -> None:
    """Register all provider adapters."""
    # Import each provider module - they self-register on import
    try:
        from fastlitellm.providers import openai_adapter

        register_provider("openai", openai_adapter.OpenAIAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import azure_adapter

        register_provider("azure", azure_adapter.AzureOpenAIAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import anthropic_adapter

        register_provider("anthropic", anthropic_adapter.AnthropicAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import gemini_adapter

        register_provider("gemini", gemini_adapter.GeminiAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import vertex_adapter

        register_provider("vertex_ai", vertex_adapter.VertexAIAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import bedrock_adapter

        register_provider("bedrock", bedrock_adapter.BedrockAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import mistral_adapter

        register_provider("mistral", mistral_adapter.MistralAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import cohere_adapter

        register_provider("cohere", cohere_adapter.CohereAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import groq_adapter

        register_provider("groq", groq_adapter.GroqAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import together_adapter

        register_provider("together_ai", together_adapter.TogetherAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import fireworks_adapter

        register_provider("fireworks_ai", fireworks_adapter.FireworksAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import deepseek_adapter

        register_provider("deepseek", deepseek_adapter.DeepSeekAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import perplexity_adapter

        register_provider("perplexity", perplexity_adapter.PerplexityAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import databricks_adapter

        register_provider("databricks", databricks_adapter.DatabricksAdapter)
    except ImportError:
        pass

    try:
        from fastlitellm.providers import ollama_adapter

        register_provider("ollama", ollama_adapter.OllamaAdapter)
    except ImportError:
        pass


# Don't auto-register on import - lazy load instead
