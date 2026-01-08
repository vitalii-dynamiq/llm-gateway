# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-08

### Added

- Initial release of fastlitellm
- Core API functions:
  - `completion()` - Synchronous chat completion
  - `acompletion()` - Asynchronous chat completion
  - `embedding()` - Create embeddings
  - `aembedding()` - Async embeddings
  - `stream_chunk_builder()` - Build response from stream chunks
- Unified response types:
  - `ModelResponse` - Chat completion response
  - `StreamChunk` - Streaming response chunk
  - `EmbeddingResponse` - Embedding response
- Provider support (15 providers):
  - OpenAI
  - Azure OpenAI
  - Anthropic Claude
  - Google Gemini (AI Studio)
  - Google Vertex AI
  - AWS Bedrock
  - Mistral AI
  - Cohere
  - Groq
  - Together AI
  - Fireworks AI
  - DeepSeek
  - Perplexity
  - Databricks
  - Ollama
- Pricing functions:
  - `cost_per_token()` - Calculate cost from token counts
  - `completion_cost()` - Calculate cost from response
  - Curated pricing tables for all providers
- Capability functions:
  - `get_max_tokens()` - Get model's max output tokens
  - `supports_vision()` - Check vision support
  - `supports_pdf_input()` - Check PDF support
  - `supports_tools()` - Check tool calling support
  - `supports_structured_output()` - Check JSON mode support
- Exception hierarchy:
  - `FastLiteLLMError` (base)
  - `AuthenticationError`
  - `RateLimitError`
  - `TimeoutError`
  - `ProviderAPIError`
  - `UnsupportedModelError`
  - `UnsupportedParameterError`
  - `ResponseParseError`
- HTTP client with:
  - Connection pooling
  - Retry with exponential backoff and jitter
  - Timeout handling
  - Streaming support
  - SSE parsing
- Full async support using stdlib asyncio
- Zero runtime dependencies (stdlib only)
- Comprehensive test suite
- Documentation:
  - AGENTS.md for AI coding agents
  - docs/ADDING_A_PROVIDER.md guide
  - Example scripts

### Notes

- Token counting is NOT implemented - uses provider-reported usage only
- Designed for Python 3.13+
