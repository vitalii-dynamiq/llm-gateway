# Provider Capability Matrix

This document lists the capabilities supported by each provider adapter in fastlitellm.

## Feature Support by Provider

| Provider | Streaming | Tools | Structured Output | Embeddings | Usage Tracking |
|----------|-----------|-------|-------------------|------------|----------------|
| OpenAI | ✅ | ✅ | ✅ JSON mode + schema | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ✅ JSON mode | ❌ | ✅ |
| Google Gemini | ✅ | ✅ | ✅ JSON mode + schema | ✅ | ✅ |
| Mistral | ✅ | ✅ | ✅ JSON mode | ✅ | ✅ |
| Cohere | ✅ | ✅ | ❌ | ✅ | ✅ |
| Groq | ✅ | ✅ | ✅ JSON mode | ❌ | ✅ |
| Together AI | ✅ | ✅ | ✅ JSON mode | ❌ | ✅ |
| Fireworks AI | ✅ | ✅ | ✅ JSON mode | ❌ | ✅ |
| DeepSeek | ✅ | ✅ | ✅ JSON mode | ❌ | ✅ |
| Perplexity | ✅ | ❌ | ❌ | ❌ | ✅ |
| Azure OpenAI | ✅ | ✅ | ✅ JSON mode + schema | ✅ | ✅ |
| AWS Bedrock | ✅ | ✅ | ❌ | ✅ | ✅ |
| Vertex AI | ✅ | ✅ | ✅ JSON mode + schema | ✅ | ✅ |
| Databricks | ✅ | ✅ | ❌ | ✅ | ✅ |
| Ollama | ✅ | ✅ | ✅ JSON mode | ✅ | ❌ |

## Streaming Usage (include_usage)

Streaming usage reporting (via `stream_options={"include_usage": True}`) is supported by:

| Provider | Support | Notes |
|----------|---------|-------|
| OpenAI | ✅ | Usage in final chunk |
| Anthropic | ✅ | Usage in message_delta event |
| Gemini | ⚠️ | Usage available via usageMetadata |
| Mistral | ✅ | Usage in final chunk |
| Cohere | ⚠️ | Partial support |
| Groq | ✅ | Usage in final chunk |
| Together | ⚠️ | Model-dependent |
| Fireworks | ⚠️ | Model-dependent |
| DeepSeek | ✅ | Usage in final chunk |
| Perplexity | ❌ | Not supported |

## Tool Calling Format

All providers that support tool calling return tool calls in OpenAI-compatible format:

```python
response.choices[0].message.tool_calls[0].function.name      # str
response.choices[0].message.tool_calls[0].function.arguments # JSON string
```

**Important**: `function.arguments` is always a JSON string, not a parsed dict.
Use `tool_call.function.parse_arguments()` to get a dict.

## Structured Output

### JSON Mode

Enable with `response_format={"type": "json_object"}`:

```python
response = completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"}
)
```

### JSON Schema (OpenAI, Gemini, Vertex)

Enable with full schema specification:

```python
response = completion(
    model="gpt-4o-mini",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "response",
            "schema": {"type": "object", "properties": {...}}
        }
    }
)
```

## Model-Specific Notes

### OpenAI
- `o1` and `o1-mini` models use `max_completion_tokens` instead of `max_tokens`
- `o1-preview` does not support tools or system messages

### Anthropic
- System messages are extracted to a separate `system` parameter
- Max tokens is required (defaults to 4096)
- PDF input supported for Claude 3.5 Sonnet

### Gemini
- Context windows up to 2M tokens (gemini-1.5-pro)
- Native PDF and video support

### Groq
- Very fast inference (up to 300 tok/s)
- Limited max_tokens on some models

### Perplexity
- Online models include web search results
- No tool calling support
