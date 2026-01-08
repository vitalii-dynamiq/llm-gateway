"""
fastlitellm - A fast, lightweight, API-compatible alternative to LiteLLM SDK.

This package provides a minimal, high-performance interface for calling
multiple LLM providers with a unified API compatible with LiteLLM.

Basic Usage:
    import fastlitellm

    # Simple completion
    response = fastlitellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

    # Async completion
    response = await fastlitellm.acompletion(
        model="anthropic/claude-3-5-sonnet-latest",
        messages=[{"role": "user", "content": "Hello!"}]
    )

    # Streaming
    stream = fastlitellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=True
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

    # Embeddings
    embeddings = fastlitellm.embedding(
        model="text-embedding-3-small",
        input=["Hello world", "Goodbye world"]
    )

Supported Providers:
    - OpenAI (openai/)
    - Azure OpenAI (azure/)
    - Anthropic (anthropic/)
    - Google Gemini (gemini/)
    - Google Vertex AI (vertex_ai/)
    - AWS Bedrock (bedrock/)
    - Mistral (mistral/)
    - Cohere (cohere/)
    - Groq (groq/)
    - Together AI (together_ai/)
    - Fireworks AI (fireworks_ai/)
    - DeepSeek (deepseek/)
    - Perplexity (perplexity/)
    - Databricks (databricks/)
    - Ollama (ollama/)

See README.md for complete documentation.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "AuthenticationError",
    "Choice",
    "ChunkChoice",
    "ChunkDelta",
    "ConnectionError",
    "ContentFilterError",
    "EmbeddingData",
    "EmbeddingResponse",
    "EmbeddingUsage",
    # Exceptions
    "FastLiteLLMError",
    "FunctionCall",
    "InvalidRequestError",
    "Message",
    # Types
    "ModelResponse",
    "ProviderAPIError",
    "RateLimitError",
    "ResponseParseError",
    "StreamChunk",
    "StreamingResponse",
    "TimeoutError",
    "ToolCall",
    "UnsupportedModelError",
    "UnsupportedParameterError",
    "Usage",
    # Version
    "__version__",
    "acompletion",
    "aembedding",
    # Core API
    "completion",
    "completion_cost",
    # Pricing
    "cost_per_token",
    "embedding",
    # Capabilities
    "get_max_tokens",
    "get_model_pricing",
    "stream_chunk_builder",
    "supports_pdf_input",
    "supports_structured_output",
    "supports_tools",
    "supports_vision",
]

# Core API functions
# Capabilities
from fastlitellm.capabilities import (
    get_max_tokens,
    supports_pdf_input,
    supports_structured_output,
    supports_tools,
    supports_vision,
)
from fastlitellm.core import (
    acompletion,
    aembedding,
    completion,
    embedding,
    stream_chunk_builder,
)

# Exceptions
from fastlitellm.exceptions import (
    AuthenticationError,
    ConnectionError,
    ContentFilterError,
    FastLiteLLMError,
    InvalidRequestError,
    ProviderAPIError,
    RateLimitError,
    ResponseParseError,
    TimeoutError,
    UnsupportedModelError,
    UnsupportedParameterError,
)

# Pricing
from fastlitellm.pricing import (
    completion_cost,
    cost_per_token,
    get_model_pricing,
)

# Register all providers on first import
from fastlitellm.providers.base import register_all_providers

# Types
from fastlitellm.types import (
    Choice,
    ChunkChoice,
    ChunkDelta,
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    FunctionCall,
    Message,
    ModelResponse,
    StreamChunk,
    StreamingResponse,
    ToolCall,
    Usage,
)

register_all_providers()
