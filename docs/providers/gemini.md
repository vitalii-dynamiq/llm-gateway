# Google Gemini Provider

## Overview

arcllm supports Google's Gemini models via the Generative AI API.

## API Documentation

- **Official Docs**: https://ai.google.dev/api/rest
- **Models**: https://ai.google.dev/gemini-api/docs/models/gemini

## Configuration

```python
import arcllm

# Via environment variable (recommended)
# export GEMINI_API_KEY="..."
# or export GOOGLE_API_KEY="..."

response = arcllm.completion(
    model="gemini/gemini-1.5-flash",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Supported Models

### Gemini 2.0
- `gemini-2.0-flash-exp` (free preview)
- `gemini-2.0-flash-thinking-exp` (free preview)

### Gemini 1.5
- `gemini-1.5-pro`, `gemini-1.5-pro-latest`
- `gemini-1.5-flash`, `gemini-1.5-flash-latest`
- `gemini-1.5-flash-8b`

### Gemini 1.0
- `gemini-1.0-pro`, `gemini-pro`

### Embeddings
- `text-embedding-004`
- `embedding-001`

## Feature Support

| Feature | Supported | Notes |
|---------|-----------|-------|
| Streaming | ✅ | Full support |
| Tool Calling | ✅ | Converted to Gemini format |
| Structured Output | ✅ | JSON mode + schema |
| Vision | ✅ | Images, video, PDF |
| PDF Input | ✅ | Native support |
| Embeddings | ✅ | Free tier available |
| Usage in Stream | ✅ | Via usageMetadata |

## Context Windows

Gemini models support extremely long context:

| Model | Context Window |
|-------|----------------|
| gemini-1.5-pro | 2,097,152 tokens |
| gemini-1.5-flash | 1,048,576 tokens |
| gemini-2.0-flash-exp | 1,048,576 tokens |
| gemini-1.0-pro | 32,768 tokens |

## Examples

### Basic Completion
```python
response = arcllm.completion(
    model="gemini-1.5-flash",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### JSON Schema Output
```python
response = arcllm.completion(
    model="gemini-1.5-flash",
    messages=[{"role": "user", "content": "Generate a product"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "product",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"}
                }
            }
        }
    }
)
```

### Streaming
```python
stream = arcllm.completion(
    model="gemini-1.5-flash",
    messages=[{"role": "user", "content": "Count to 10"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Embeddings
```python
response = arcllm.embedding(
    model="gemini/text-embedding-004",
    input=["Hello world"]
)
```

## Integration Test Model

- **Model**: `gemini-1.5-flash`
- **Why**: Fast, cheap, 1M context, full feature support
