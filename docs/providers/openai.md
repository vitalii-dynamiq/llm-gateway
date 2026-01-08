# OpenAI Provider

## Overview

fastlitellm supports all OpenAI chat completion and embedding models.

## API Documentation

- **Official Docs**: https://platform.openai.com/docs/api-reference
- **Models**: https://platform.openai.com/docs/models

## Configuration

```python
import fastlitellm

# Via environment variable (recommended)
# export OPENAI_API_KEY="sk-..."

response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Or explicit API key
response = fastlitellm.completion(
    model="openai/gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
    api_key="sk-..."
)
```

## Supported Models

### Chat Completion
- `gpt-4o`, `gpt-4o-2024-11-20`, `gpt-4o-2024-08-06`
- `gpt-4o-mini`, `gpt-4o-mini-2024-07-18`
- `gpt-4-turbo`, `gpt-4-turbo-preview`
- `gpt-4`, `gpt-4-32k`
- `gpt-3.5-turbo`
- `o1`, `o1-2024-12-17`
- `o1-mini`, `o1-preview`

### Embeddings
- `text-embedding-3-small`
- `text-embedding-3-large`
- `text-embedding-ada-002`

## Feature Support

| Feature | Supported | Notes |
|---------|-----------|-------|
| Streaming | ✅ | Full support |
| Tool Calling | ✅ | Parallel tool calls supported |
| Structured Output | ✅ | JSON mode + JSON schema |
| Vision | ✅ | GPT-4o, GPT-4 Turbo |
| Embeddings | ✅ | Multiple models |
| Usage in Stream | ✅ | `stream_options={"include_usage": True}` |

## Examples

### Streaming
```python
stream = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Count to 5"}],
    stream=True,
    stream_options={"include_usage": True}
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Tool Calling
```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string"}
        }}
    }
}]
response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Weather in NYC?"}],
    tools=tools
)
```

### JSON Schema Output
```python
response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Generate user profile"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "user_profile",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }
)
```

## Integration Test Model

- **Model**: `gpt-4o-mini`
- **Why**: Fast, cheap, supports all features
