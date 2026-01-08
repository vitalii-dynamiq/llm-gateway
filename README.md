<p align="center">
  <img src="https://raw.githubusercontent.com/arcllm/arcllm/main/docs/assets/logo.svg" alt="ArcLLM" width="400">
</p>

<h3 align="center">The arc connecting you to every LLM</h3>

<p align="center">
  <strong>Zero dependencies. Maximum performance. One unified API.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/arcllm/"><img src="https://img.shields.io/pypi/v/arcllm?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/arcllm/"><img src="https://img.shields.io/pypi/pyversions/arcllm" alt="Python"></a>
  <a href="https://github.com/arcllm/arcllm/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License"></a>
  <a href="https://github.com/arcllm/arcllm/actions"><img src="https://img.shields.io/github/actions/workflow/status/arcllm/arcllm/ci.yml?branch=main" alt="CI"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-providers">Providers</a> •
  <a href="#features">Features</a> •
  <a href="#documentation">Docs</a>
</p>

---

## Why ArcLLM?

| Feature | ArcLLM | Others |
|---------|--------|--------|
| **Dependencies** | 0 (stdlib only) | 10-50+ packages |
| **Install size** | ~100KB | 50-200MB |
| **Cold start** | ~10ms | 500ms-2s |
| **API** | OpenAI-compatible | Varies |

ArcLLM is built for developers who want **speed**, **simplicity**, and **reliability** when working with LLMs.

## Installation

```bash
pip install arcllm
```

That's it. No dependency hell. No version conflicts. Just works.

## Quick Start

```python
import arcllm

# Simple completion
response = arcllm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

### Streaming

```python
stream = arcllm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a haiku about coding"}],
    stream=True
)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### Async

```python
response = await arcllm.acompletion(
    model="anthropic/claude-3-5-sonnet-latest",
    messages=[{"role": "user", "content": "Explain quantum computing"}]
)
```

### Different Providers

```python
# OpenAI
response = arcllm.completion(model="gpt-4o", messages=messages)

# Anthropic
response = arcllm.completion(model="anthropic/claude-3-5-sonnet-latest", messages=messages)

# Google Gemini
response = arcllm.completion(model="gemini/gemini-1.5-pro", messages=messages)

# Groq (ultra-fast inference)
response = arcllm.completion(model="groq/llama-3.3-70b-versatile", messages=messages)

# Local with Ollama
response = arcllm.completion(model="ollama/llama3.2", messages=messages)
```

## Supported Providers

| Provider | Prefix | Models | Environment Variable |
|----------|--------|--------|---------------------|
| **OpenAI** | `openai/` | GPT-4o, GPT-4, o1, o3 | `OPENAI_API_KEY` |
| **Anthropic** | `anthropic/` | Claude 3.5, Claude 3 | `ANTHROPIC_API_KEY` |
| **Google Gemini** | `gemini/` | Gemini 1.5, Gemini 2.0 | `GEMINI_API_KEY` |
| **Azure OpenAI** | `azure/` | GPT-4o, GPT-4 | `AZURE_OPENAI_API_KEY` |
| **AWS Bedrock** | `bedrock/` | Claude, Llama, Titan | AWS credentials |
| **Google Vertex** | `vertex_ai/` | Gemini, PaLM | `GOOGLE_ACCESS_TOKEN` |
| **Mistral** | `mistral/` | Mistral Large, Codestral | `MISTRAL_API_KEY` |
| **Groq** | `groq/` | Llama 3.3, Mixtral | `GROQ_API_KEY` |
| **Together AI** | `together_ai/` | Llama, Mixtral, Qwen | `TOGETHER_API_KEY` |
| **Fireworks** | `fireworks_ai/` | Llama, Mixtral | `FIREWORKS_API_KEY` |
| **DeepSeek** | `deepseek/` | DeepSeek V3, Coder | `DEEPSEEK_API_KEY` |
| **Perplexity** | `perplexity/` | Sonar, Online | `PERPLEXITY_API_KEY` |
| **Cohere** | `cohere/` | Command R+ | `COHERE_API_KEY` |
| **Databricks** | `databricks/` | DBRX, Llama | `DATABRICKS_TOKEN` |
| **Ollama** | `ollama/` | Any local model | (local) |

## Features

### 🛠️ Tool Calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
        }
    }
}]

response = arcllm.completion(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools
)

if response.choices[0].message.tool_calls:
    for tool_call in response.choices[0].message.tool_calls:
        print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
```

### 📋 Structured Output

```python
response = arcllm.completion(
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
                    "age": {"type": "integer"},
                    "interests": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["name", "age"]
            }
        }
    }
)
```

### 🖼️ Vision

```python
response = arcllm.completion(
    model="gpt-4o",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }]
)
```

### 📊 Embeddings

```python
response = arcllm.embedding(
    model="text-embedding-3-small",
    input=["Hello world", "Goodbye world"]
)
print(f"Dimensions: {len(response.data[0].embedding)}")
```

### 💰 Cost Tracking

```python
response = arcllm.completion(model="gpt-4o", messages=messages)

# Calculate cost
cost = arcllm.completion_cost(response)
print(f"Cost: ${cost:.6f}")

# Or get per-token pricing
input_cost, output_cost = arcllm.cost_per_token(
    model="gpt-4o",
    prompt_tokens=1000,
    completion_tokens=500
)
```

### 🔍 Model Capabilities

```python
# Check what models can do
arcllm.supports_vision("gpt-4o")           # True
arcllm.supports_vision("gpt-3.5-turbo")    # False

arcllm.supports_tools("claude-3-5-sonnet-latest")  # True
arcllm.supports_pdf_input("gemini-1.5-pro")        # True

arcllm.get_max_tokens("gpt-4o")  # 16384
```

## Error Handling

```python
from arcllm import (
    ArcLLMError,
    AuthenticationError,
    RateLimitError,
    TimeoutError,
)

try:
    response = arcllm.completion(model="gpt-4o", messages=messages)
except AuthenticationError:
    print("Check your API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after {e.retry_after}s")
except TimeoutError:
    print("Request timed out")
except ArcLLMError as e:
    print(f"Error: {e.message}")
```

## Configuration

```python
# Per-request configuration
response = arcllm.completion(
    model="gpt-4o",
    messages=messages,
    api_key="sk-...",           # Override API key
    api_base="https://...",     # Custom endpoint
    timeout=120.0,              # Request timeout
    max_retries=5,              # Retry count
)

# Azure OpenAI
response = arcllm.completion(
    model="azure/my-deployment",
    messages=messages,
    api_base="https://myresource.openai.azure.com",
    api_version="2024-10-21",
)
```

## Migration from LiteLLM

ArcLLM is designed as a drop-in replacement:

```python
# Before
import litellm
response = litellm.completion(model="gpt-4o", messages=messages)

# After
import arcllm
response = arcllm.completion(model="gpt-4o", messages=messages)

# Or alias it
import arcllm as litellm
response = litellm.completion(model="gpt-4o", messages=messages)
```

## Documentation

- [Adding a Provider](docs/ADDING_A_PROVIDER.md)
- [Provider Capabilities](docs/providers/CAPABILITIES.md)
- [Performance Guide](docs/PERF.md)
- [Contributing](CONTRIBUTING.md)

## Why "Arc"?

An **arc** is the shortest path between two points. ArcLLM is the shortest path between your code and any LLM provider—minimal, direct, efficient.

## License

Apache 2.0 - see [LICENSE](LICENSE)

---

<p align="center">
  <sub>Built with ❤️ for developers who value simplicity</sub>
</p>
