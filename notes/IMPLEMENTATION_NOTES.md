# fastlitellm Implementation Notes

## Overview

This document provides implementation notes for the fastlitellm library - a fast, lightweight, API-compatible alternative to LiteLLM SDK.

## Architecture Decisions

### 1. Zero Runtime Dependencies

The library uses only Python stdlib to minimize footprint and security surface. Key stdlib modules used:
- `http.client` - Sync HTTP
- `asyncio` - Async HTTP
- `ssl` - TLS support
- `json` - JSON parsing
- `dataclasses` - Type definitions
- `gzip`, `zlib` - Compression

### 2. Type System

All response types use `@dataclass(slots=True)` for:
- Memory efficiency (no `__dict__`)
- Faster attribute access
- Clear contracts

### 3. Provider Adapters

Each provider implements the `Adapter` protocol:
- `build_request()` - Converts unified input to provider format
- `parse_response()` - Converts provider response to unified output
- `parse_stream_event()` - Handles streaming events
- `parse_error()` - Maps provider errors to fastlitellm exceptions

Many providers (Groq, Together AI, Fireworks, DeepSeek, Perplexity, Databricks) use OpenAI-compatible APIs, so they inherit from `OpenAIAdapter`.

### 4. HTTP Clients

#### Sync Client (`http/client.py`)
- Connection pooling per host
- Retry with exponential backoff and jitter
- Streaming support via generator
- Gzip/deflate decompression

#### Async Client (`http/async_client.py`)
- True async I/O (no blocking)
- Uses `asyncio.open_connection`
- Chunked transfer encoding support
- Same retry logic as sync

### 5. SSE Parser (`http/sse.py`)

Handles Server-Sent Events:
- Partial chunk handling (UTF-8 boundaries)
- Multi-line data fields
- Comment filtering
- [DONE] marker detection

## Supported Providers

1. **OpenAI** - Full support (completion, streaming, tools, embeddings)
2. **Azure OpenAI** - Inherits from OpenAI, different auth/endpoints
3. **Anthropic** - Custom message format, different streaming protocol
4. **Google Gemini** - AI Studio API
5. **Google Vertex AI** - Same format as Gemini, different auth
6. **AWS Bedrock** - AWS SigV4 auth, multiple model families
7. **Mistral** - OpenAI-compatible
8. **Cohere** - Custom format, different streaming protocol
9. **Groq** - OpenAI-compatible
10. **Together AI** - OpenAI-compatible
11. **Fireworks AI** - OpenAI-compatible
12. **DeepSeek** - OpenAI-compatible
13. **Perplexity** - OpenAI-compatible
14. **Databricks** - OpenAI-compatible, custom auth
15. **Ollama** - OpenAI-compatible, local

## Key Features

### Tool Calling

All providers normalize tool calls to OpenAI format:
```python
tool_call.id = "call_xxx"
tool_call.type = "function"
tool_call.function.name = "name"
tool_call.function.arguments = '{"json": "string"}'  # JSON STRING
```

### Structured Output

Supports:
- `response_format={"type": "json_object"}` - JSON mode
- `response_format={"type": "json_schema", "json_schema": {...}}` - JSON schema

### Usage Tracking

- Provider-reported usage only (no tiktoken)
- Streaming usage via `stream_options={"include_usage": True}`
- Usage accessible via `response.usage` and `response.model_extra["usage"]`

### Cost Calculation

- Curated pricing tables for all providers
- `cost_per_token(model, prompt_tokens, completion_tokens)`
- `completion_cost(response)` for total cost

### Capability Queries

- `get_max_tokens(model)` - Max output tokens
- `supports_vision(model)` - Image input support
- `supports_pdf_input(model)` - PDF support
- `supports_tools(model)` - Tool calling support
- `supports_structured_output(model)` - JSON mode support

## Testing

- 159 tests covering:
  - Type serialization
  - Exception hierarchy
  - SSE parsing
  - Provider request/response parsing
  - Pricing calculations
  - Capability queries

## Performance Considerations

1. **Minimal imports** - Providers lazy-loaded
2. **Connection reuse** - HTTP connection pooling
3. **Efficient parsing** - Direct JSON parsing, no intermediate objects
4. **`__slots__`** - Memory efficiency for response types
5. **Generator streaming** - No buffering entire response

## Future Improvements

1. Add more provider-specific tests with fixtures
2. Add integration tests (requires API keys)
3. Add caching for pricing/capability lookups
4. Add request logging hooks
5. Add proxy support for HTTP clients
6. Improve error messages with more context
