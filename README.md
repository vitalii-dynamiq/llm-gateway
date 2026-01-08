# fastlitellm

A fast, lightweight, API-compatible alternative to LiteLLM SDK.

## Features

- **Zero Dependencies**: Uses only Python stdlib (no runtime dependencies)
- **LiteLLM Compatible**: Drop-in replacement for common LiteLLM usage patterns
- **15 Providers**: OpenAI, Anthropic, Gemini, Azure, Bedrock, Mistral, and more
- **Full Async Support**: True async I/O with asyncio
- **Streaming**: Full streaming support with usage tracking
- **Tool Calling**: Unified tool/function calling interface
- **Type Safe**: Fully typed with modern Python type hints
- **Fast**: Minimal overhead, connection pooling, efficient parsing

## Installation

```bash
pip install fastlitellm
```

## Quick Start

```python
import fastlitellm

# Simple completion
response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)

# Streaming
stream = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")

# Async
response = await fastlitellm.acompletion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Supported Providers

| Provider | Prefix | Environment Variable |
|----------|--------|---------------------|
| OpenAI | `openai/` | `OPENAI_API_KEY` |
| Azure OpenAI | `azure/` | `AZURE_OPENAI_API_KEY` |
| Anthropic | `anthropic/` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini/` | `GEMINI_API_KEY` |
| Google Vertex AI | `vertex_ai/` | `GOOGLE_ACCESS_TOKEN` |
| AWS Bedrock | `bedrock/` | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` |
| Mistral | `mistral/` | `MISTRAL_API_KEY` |
| Cohere | `cohere/` | `COHERE_API_KEY` |
| Groq | `groq/` | `GROQ_API_KEY` |
| Together AI | `together_ai/` | `TOGETHER_API_KEY` |
| Fireworks AI | `fireworks_ai/` | `FIREWORKS_API_KEY` |
| DeepSeek | `deepseek/` | `DEEPSEEK_API_KEY` |
| Perplexity | `perplexity/` | `PERPLEXITY_API_KEY` |
| Databricks | `databricks/` | `DATABRICKS_TOKEN` |
| Ollama | `ollama/` | (local, no key needed) |

## API Reference

### completion()

```python
response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=100,
    stream=False,
    tools=None,
    tool_choice=None,
    response_format=None,
    drop_params=False,  # Silently drop unsupported params
)
```

### acompletion()

Async version of `completion()`:

```python
response = await fastlitellm.acompletion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### embedding()

```python
response = fastlitellm.embedding(
    model="text-embedding-3-small",
    input=["Hello world", "Goodbye world"]
)
print(response.data[0].embedding)  # List[float]
```

### stream_chunk_builder()

Build a complete response from streaming chunks:

```python
stream = fastlitellm.completion(model="gpt-4o-mini", messages=messages, stream=True)
chunks = list(stream)
response = fastlitellm.stream_chunk_builder(chunks)
```

### cost_per_token()

```python
prompt_cost, completion_cost = fastlitellm.cost_per_token(
    model="gpt-4o-mini",
    prompt_tokens=1000,
    completion_tokens=500
)
```

### Capability Functions

```python
max_tokens = fastlitellm.get_max_tokens("gpt-4o")  # 16384
has_vision = fastlitellm.supports_vision("gpt-4o")  # True
has_pdf = fastlitellm.supports_pdf_input("claude-3-5-sonnet-latest")  # True
has_tools = fastlitellm.supports_tools("gpt-4o")  # True
```

## Response Format

### ModelResponse

```python
response.id                          # Response ID
response.model                       # Model used
response.choices[0].message.content  # Text content
response.choices[0].message.tool_calls  # Tool calls (if any)
response.choices[0].finish_reason    # "stop", "length", "tool_calls"
response.usage.prompt_tokens         # Input tokens
response.usage.completion_tokens     # Output tokens
response.usage.total_tokens          # Total tokens
response.model_extra["usage"]        # Usage as dict (for compatibility)
```

### StreamChunk

```python
chunk.choices[0].delta.content      # Incremental content
chunk.choices[0].delta.tool_calls   # Tool call deltas
chunk.choices[0].finish_reason      # Set on final chunk
chunk.usage                         # Set if stream_options.include_usage=True
```

## Tool Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
    tools=tools,
    tool_choice="auto"
)

if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        print(f"Call {tc.function.name} with {tc.function.arguments}")
```

## Structured Output

### JSON Mode

```python
response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List 3 colors as JSON"}],
    response_format={"type": "json_object"}
)
data = json.loads(response.choices[0].message.content)
```

### JSON Schema

```python
response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Generate a user profile"}],
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

## Error Handling

```python
from fastlitellm import (
    FastLiteLLMError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
    ProviderAPIError,
    UnsupportedModelError,
)

try:
    response = fastlitellm.completion(...)
except AuthenticationError as e:
    print(f"Auth failed: {e.message}")
except RateLimitError as e:
    print(f"Rate limited, retry after: {e.retry_after}")
except TimeoutError as e:
    print(f"Timed out: {e.timeout_seconds}s")
except ProviderAPIError as e:
    print(f"API error {e.status_code}: {e.message}")
except FastLiteLLMError as e:
    print(f"Error: {e}")
```

## Configuration

### Per-Request Configuration

```python
response = fastlitellm.completion(
    model="gpt-4o-mini",
    messages=messages,
    api_key="sk-xxx",  # Override API key
    api_base="https://custom.endpoint.com/v1",  # Custom endpoint
    timeout=120.0,  # Request timeout
    max_retries=5,  # Retry count
)
```

### Azure Configuration

```python
response = fastlitellm.completion(
    model="azure/my-deployment",
    messages=messages,
    api_base="https://myresource.openai.azure.com",
    api_key="xxx",  # or set AZURE_OPENAI_API_KEY
    api_version="2024-10-21",
)
```

## Migration from LiteLLM

fastlitellm is designed for easy migration:

```python
# Before
import litellm
response = litellm.completion(model="gpt-4o-mini", messages=messages)

# After
import fastlitellm
response = fastlitellm.completion(model="gpt-4o-mini", messages=messages)

# Or use aliasing
import fastlitellm as litellm
response = litellm.completion(model="gpt-4o-mini", messages=messages)
```

## Design Decisions

1. **No Token Counting**: We use provider-reported usage only. No tiktoken dependency.
2. **Stdlib Only**: Zero runtime dependencies for minimal footprint and security.
3. **Python 3.13+**: Uses modern typing features for cleaner code.
4. **Explicit Over Magic**: Clear error messages, no silent parameter dropping by default.

## Contributing

See [AGENTS.md](AGENTS.md) for development guidelines.

## License

MIT License
