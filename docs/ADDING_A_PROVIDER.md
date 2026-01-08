# Adding a New Provider to arcllm

This guide explains how to add support for a new LLM provider to arcllm.

## Overview

Each provider is implemented as an adapter that:
1. Converts arcllm's unified input format to the provider's API format
2. Makes HTTP requests to the provider's API
3. Converts the provider's response back to arcllm's unified format

## Step-by-Step Guide

### 1. Create the Adapter File

Create a new file in `arcllm/providers/`:

```python
# arcllm/providers/newprovider_adapter.py

from __future__ import annotations

import json
import os
import time
from typing import Any

from arcllm.types import (
    ModelResponse,
    Choice,
    Message,
    ToolCall,
    FunctionCall,
    Usage,
    StreamChunk,
    ChunkChoice,
    ChunkDelta,
    EmbeddingResponse,
)
from arcllm.exceptions import (
    ArcLLMError,
    AuthenticationError,
    RateLimitError,
    ProviderAPIError,
    ResponseParseError,
)
from arcllm.providers.base import (
    BaseAdapter,
    ProviderConfig,
    RequestData,
    COMMON_PARAMS,
    register_provider,
)

__all__ = ["NewProviderAdapter"]


class NewProviderAdapter(BaseAdapter):
    """Adapter for NewProvider API."""

    provider_name = "newprovider"
    
    # Define which parameters this provider supports
    supported_params = COMMON_PARAMS | {
        "provider_specific_param",
    }

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.newprovider.com/v1"

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        api_key = self._get_api_key("NEWPROVIDER_API_KEY")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def build_request(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        stream: bool = False,
        drop_params: bool = False,
        **kwargs: Any,
    ) -> RequestData:
        """Build HTTP request from input parameters."""
        # Check/filter parameters
        kwargs = self._check_params(drop_params, **kwargs)
        
        # Convert messages to provider format if needed
        provider_messages = self._convert_messages(messages)
        
        # Build request body
        body = {
            "model": model,
            "messages": provider_messages,
            "stream": stream,
        }
        
        # Add optional parameters
        if "temperature" in kwargs:
            body["temperature"] = kwargs["temperature"]
        # ... add other parameters
        
        url = f"{self._api_base}/chat/completions"
        body_bytes = json.dumps(body).encode("utf-8")
        
        return RequestData(
            method="POST",
            url=url,
            headers=self._get_headers(),
            body=body_bytes,
            timeout=self.config.timeout,
        )

    def parse_response(self, data: bytes, model: str) -> ModelResponse:
        """Parse provider response into ModelResponse."""
        try:
            resp = json.loads(data.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ResponseParseError(f"Invalid JSON: {e}", raw_data=data)
        
        # Convert to ModelResponse
        # ... implementation depends on provider format
        
        return ModelResponse(
            id=resp.get("id", ""),
            model=model,
            choices=[...],
            usage=usage,
        )

    def parse_stream_event(self, data: str, model: str) -> StreamChunk | None:
        """Parse a streaming event."""
        if not data or data == "[DONE]":
            return None
        
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            return None
        
        # Convert to StreamChunk
        # ... implementation depends on provider format
        
        return StreamChunk(...)

    def parse_error(
        self,
        status_code: int,
        data: bytes,
        request_id: str | None = None,
    ) -> ArcLLMError:
        """Parse error response into appropriate exception."""
        try:
            error_data = json.loads(data.decode("utf-8"))
            message = error_data.get("error", {}).get("message", "Unknown error")
        except:
            message = data.decode("utf-8", errors="replace")
        
        if status_code == 401:
            return AuthenticationError(message, provider=self.provider_name)
        elif status_code == 429:
            return RateLimitError(message, provider=self.provider_name)
        else:
            return ProviderAPIError(message, provider=self.provider_name, status_code=status_code)


# Register the provider
register_provider("newprovider", NewProviderAdapter)
```

### 2. Register the Provider

Add the import to `arcllm/providers/base.py` in the `_register_all_providers()` function:

```python
try:
    from arcllm.providers import newprovider_adapter
    register_provider("newprovider", newprovider_adapter.NewProviderAdapter)
except ImportError:
    pass
```

### 3. Add Pricing Data

Add pricing information to `arcllm/pricing/tables.py`:

```python
NEWPROVIDER_PRICING: dict[str, ModelPricing] = {
    "model-name-1": ModelPricing(input_cost=1.0, output_cost=2.0),
    "model-name-2": ModelPricing(input_cost=0.5, output_cost=1.0),
}

# Add to ALL_PRICING dict
ALL_PRICING["newprovider"] = NEWPROVIDER_PRICING
```

### 4. Add Capability Data

Add model capabilities to `arcllm/capabilities/tables.py`:

```python
NEWPROVIDER_CAPABILITIES: dict[str, ModelCapabilities] = {
    "model-name-1": ModelCapabilities(
        max_tokens=4096,
        context_window=128000,
        supports_vision=True,
        supports_tools=True,
        supports_structured_output=True,
    ),
}

# Add to ALL_CAPABILITIES dict
ALL_CAPABILITIES["newprovider"] = NEWPROVIDER_CAPABILITIES
```

### 5. Write Tests

Create test file `tests/providers/test_newprovider.py`:

```python
import pytest
import json
from arcllm.providers.newprovider_adapter import NewProviderAdapter
from arcllm.providers.base import ProviderConfig


class TestNewProviderAdapter:
    @pytest.fixture
    def adapter(self):
        config = ProviderConfig(api_key="test-key")
        return NewProviderAdapter(config)

    def test_build_request(self, adapter):
        request = adapter.build_request(
            model="model-name",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert request.method == "POST"
        assert "chat/completions" in request.url
        
        body = json.loads(request.body.decode())
        assert body["model"] == "model-name"
        assert body["messages"][0]["content"] == "Hello"

    def test_parse_response(self, adapter):
        # Create fixture for provider's response format
        response_data = {...}
        
        result = adapter.parse_response(
            json.dumps(response_data).encode(),
            "model-name"
        )
        
        assert result.choices[0].message.content == "expected content"
```

### 6. Add to SUPPORTED_PROVIDERS

Update `SUPPORTED_PROVIDERS` list in `arcllm/providers/base.py`:

```python
SUPPORTED_PROVIDERS = [
    ...
    "newprovider",
]
```

## Checklist

- [ ] Created adapter file in `arcllm/providers/`
- [ ] Implemented `build_request()` method
- [ ] Implemented `parse_response()` method
- [ ] Implemented `parse_stream_event()` method
- [ ] Implemented `parse_error()` method
- [ ] Implemented `build_embedding_request()` if provider supports embeddings
- [ ] Implemented `parse_embedding_response()` if provider supports embeddings
- [ ] Registered provider in `_register_all_providers()`
- [ ] Added to `SUPPORTED_PROVIDERS` list
- [ ] Added pricing data to pricing/tables.py
- [ ] Added capability data to capabilities/tables.py
- [ ] Written unit tests with request/response fixtures
- [ ] Tested with real API (manual verification)
- [ ] Updated documentation if needed

## Tips

### OpenAI-Compatible Providers

Many providers offer OpenAI-compatible APIs. For these, you can inherit from `OpenAIAdapter`:

```python
from arcllm.providers.openai_adapter import OpenAIAdapter

class NewProviderAdapter(OpenAIAdapter):
    provider_name = "newprovider"

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_base = config.api_base or "https://api.newprovider.com/v1"

    def _get_headers(self) -> dict[str, str]:
        api_key = self._get_api_key("NEWPROVIDER_API_KEY")
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
```

### Message Format Conversion

If the provider uses a different message format, implement a conversion method:

```python
def _convert_messages(
    self, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert OpenAI-style messages to provider format."""
    converted = []
    for msg in messages:
        # Handle system, user, assistant, tool messages
        # ...
    return converted
```

### Tool Call Conversion

If tools/function calling format differs:

```python
def _convert_tools(
    self, tools: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Convert OpenAI tool format to provider format."""
    # ...
```

### Error Handling

Map provider-specific error codes to arcllm exceptions:

```python
def parse_error(self, status_code: int, data: bytes, request_id: str | None = None):
    # Parse provider-specific error format
    # Map to appropriate exception class
    pass
```

## Testing Your Provider

1. **Unit tests**: Test request building and response parsing with fixtures
2. **Integration tests**: Test against real API (if possible, use test/sandbox endpoints)
3. **Streaming tests**: Test SSE parsing with realistic streaming data
4. **Error tests**: Test error handling for various HTTP status codes
