# Anthropic Provider

## Overview

arcllm supports Claude models via the Anthropic Messages API.

## API Documentation

- **Official Docs**: https://docs.anthropic.com/en/api
- **Models**: https://docs.anthropic.com/en/docs/about-claude/models

## Configuration

```python
import arcllm

# Via environment variable (recommended)
# export ANTHROPIC_API_KEY="sk-ant-..."

response = arcllm.completion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Or with prefix inference
response = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Supported Models

### Claude 3.5
- `claude-3-5-sonnet-20241022`, `claude-3-5-sonnet-latest`
- `claude-3-5-haiku-20241022`, `claude-3-5-haiku-latest`

### Claude 3
- `claude-3-opus-20240229`, `claude-3-opus-latest`
- `claude-3-sonnet-20240229`
- `claude-3-haiku-20240307`

### Legacy
- `claude-2.1`, `claude-2.0`
- `claude-instant-1.2`

## Feature Support

| Feature | Supported | Notes |
|---------|-----------|-------|
| Streaming | ✅ | Full support |
| Tool Calling | ✅ | Converted to Anthropic format |
| Structured Output | ✅ | JSON mode via system prompt |
| Vision | ✅ | Claude 3+ models |
| PDF Input | ✅ | Claude 3.5 Sonnet |
| Embeddings | ❌ | Not available |
| Usage in Stream | ✅ | Via message_delta event |

## Message Conversion

arcllm automatically converts OpenAI-style messages to Anthropic format:

- System messages are extracted to the `system` parameter
- Tool results are wrapped in user messages
- Multimodal content is converted to Anthropic's content block format

## Examples

### Basic Completion
```python
response = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    max_tokens=100
)
```

### Tool Calling
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}}
        }
    }
}]
response = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    tools=tools
)
```

### Streaming
```python
stream = arcllm.completion(
    model="claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Write a haiku"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Notes

- `max_tokens` is required (arcllm defaults to 4096 if not specified)
- Tool call IDs use Anthropic's format (`toolu_xxx`) but are compatible

## Integration Test Model

- **Model**: `claude-3-5-haiku-20241022`
- **Why**: Fast, affordable, supports tools and streaming
