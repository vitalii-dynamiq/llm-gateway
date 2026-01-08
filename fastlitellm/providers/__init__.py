"""
Provider adapters for fastlitellm.

Each provider implements the Adapter protocol to handle:
- Request building (converting OpenAI-like params to provider format)
- Response parsing (converting provider response to unified ModelResponse)
- Streaming event parsing
"""

from fastlitellm.providers.base import (
    SUPPORTED_PROVIDERS,
    Adapter,
    ProviderConfig,
    RequestData,
    get_provider,
    register_provider,
)

__all__ = [
    "SUPPORTED_PROVIDERS",
    "Adapter",
    "ProviderConfig",
    "RequestData",
    "get_provider",
    "register_provider",
]
