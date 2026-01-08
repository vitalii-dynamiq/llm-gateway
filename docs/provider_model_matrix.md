# Provider Model Matrix

This document lists the supported models, capabilities, and required credentials for each provider in fastlitellm.

**Last Updated:** 2025-01-08

## Quick Reference: Required Environment Variables

| Provider | Environment Variable(s) | Documentation |
|----------|------------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | [OpenAI API Keys](https://platform.openai.com/api-keys) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` | [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/) |
| Anthropic | `ANTHROPIC_API_KEY` | [Anthropic Console](https://console.anthropic.com/) |
| Google Gemini | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | [Google AI Studio](https://aistudio.google.com/apikey) |
| Google Vertex AI | `GOOGLE_ACCESS_TOKEN`, `VERTEX_PROJECT`, `VERTEX_LOCATION` | [Vertex AI](https://cloud.google.com/vertex-ai/docs) |
| AWS Bedrock | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION` | [AWS Bedrock](https://docs.aws.amazon.com/bedrock/) |
| Mistral | `MISTRAL_API_KEY` | [Mistral AI](https://console.mistral.ai/) |
| Cohere | `COHERE_API_KEY` | [Cohere Dashboard](https://dashboard.cohere.com/) |
| Groq | `GROQ_API_KEY` | [Groq Console](https://console.groq.com/) |
| Together AI | `TOGETHER_API_KEY` | [Together AI](https://api.together.xyz/) |
| Fireworks AI | `FIREWORKS_API_KEY` | [Fireworks AI](https://fireworks.ai/) |
| DeepSeek | `DEEPSEEK_API_KEY` | [DeepSeek Platform](https://platform.deepseek.com/) |
| Perplexity | `PERPLEXITY_API_KEY` | [Perplexity API](https://docs.perplexity.ai/) |
| Databricks | `DATABRICKS_TOKEN`, `DATABRICKS_HOST` | [Databricks](https://docs.databricks.com/) |
| Ollama | None (local) | [Ollama](https://ollama.ai/) |

---

## OpenAI

**Documentation:** https://platform.openai.com/docs/models

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `gpt-4o-mini` | ✅ | ✅ | ✅ | ✅ |
| `gpt-4o` | ✅ | ✅ | ✅ | ✅ |
| `gpt-4-turbo` | ✅ | ✅ | ✅ | ✅ |
| `gpt-3.5-turbo` | ✅ | ✅ | ✅ | ✅ |
| `o1-mini` | ✅ | ❌ | ❌ | ✅ |
| `o1` | ✅ | ✅ | ✅ | ✅ |

### Embedding Models

| Model ID | Dimensions | Max Input |
|----------|------------|-----------|
| `text-embedding-3-small` | 1536 | 8191 tokens |
| `text-embedding-3-large` | 3072 | 8191 tokens |
| `text-embedding-ada-002` | 1536 | 8191 tokens |

### Test Configuration
```yaml
primary_model: gpt-4o-mini  # Fast, cheap, full-featured
embedding_model: text-embedding-3-small
```

---

## Anthropic

**Documentation:** https://docs.anthropic.com/en/docs/about-claude/models

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `claude-3-5-sonnet-20241022` | ✅ | ✅ | ✅ | ✅ |
| `claude-3-5-haiku-20241022` | ✅ | ✅ | ✅ | ✅ |
| `claude-3-opus-20240229` | ✅ | ✅ | ✅ | ✅ |
| `claude-3-sonnet-20240229` | ✅ | ✅ | ✅ | ✅ |
| `claude-3-haiku-20240307` | ✅ | ✅ | ✅ | ✅ |

### Embedding Models
Anthropic does not provide embedding models.

### Test Configuration
```yaml
primary_model: claude-3-5-haiku-20241022  # Fast, cost-effective
secondary_model: claude-3-5-sonnet-20241022  # Full-featured
```

---

## Google Gemini (AI Studio)

**Documentation:** https://ai.google.dev/gemini-api/docs/models/gemini

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `gemini-1.5-flash` | ✅ | ✅ | ✅ | ✅ |
| `gemini-1.5-flash-8b` | ✅ | ✅ | ✅ | ✅ |
| `gemini-1.5-pro` | ✅ | ✅ | ✅ | ✅ |
| `gemini-2.0-flash-exp` | ✅ | ✅ | ✅ | ✅ |

### Embedding Models

| Model ID | Dimensions |
|----------|------------|
| `text-embedding-004` | 768 |

### Test Configuration
```yaml
primary_model: gemini-1.5-flash  # Fast, free tier available
embedding_model: text-embedding-004
```

---

## Google Vertex AI

**Documentation:** https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/overview

Uses same models as Gemini but requires GCP authentication.

### Test Configuration
```yaml
primary_model: gemini-1.5-flash
requires:
  - GOOGLE_ACCESS_TOKEN
  - VERTEX_PROJECT
  - VERTEX_LOCATION (default: us-central1)
```

---

## AWS Bedrock

**Documentation:** https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `anthropic.claude-3-5-sonnet-20241022-v2:0` | ✅ | ✅ | ✅ | ✅ |
| `anthropic.claude-3-haiku-20240307-v1:0` | ✅ | ✅ | ✅ | ✅ |
| `meta.llama3-70b-instruct-v1:0` | ✅ | ❌ | ❌ | ✅ |
| `amazon.titan-text-premier-v1:0` | ✅ | ❌ | ❌ | ✅ |

### Embedding Models

| Model ID | Dimensions |
|----------|------------|
| `amazon.titan-embed-text-v2:0` | 1024 |
| `cohere.embed-english-v3` | 1024 |

### Test Configuration
```yaml
primary_model: anthropic.claude-3-haiku-20240307-v1:0
embedding_model: amazon.titan-embed-text-v2:0
requires:
  - AWS_ACCESS_KEY_ID
  - AWS_SECRET_ACCESS_KEY
  - AWS_REGION (default: us-east-1)
```

---

## Mistral

**Documentation:** https://docs.mistral.ai/getting-started/models/

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `mistral-small-latest` | ✅ | ✅ | ✅ | ✅ |
| `mistral-large-latest` | ✅ | ✅ | ✅ | ✅ |
| `mistral-nemo-latest` | ✅ | ✅ | ✅ | ✅ |
| `codestral-latest` | ✅ | ✅ | ❌ | ✅ |
| `ministral-8b-latest` | ✅ | ✅ | ❌ | ✅ |

### Embedding Models

| Model ID | Dimensions |
|----------|------------|
| `mistral-embed` | 1024 |

### Test Configuration
```yaml
primary_model: mistral-small-latest
embedding_model: mistral-embed
```

---

## Cohere

**Documentation:** https://docs.cohere.com/docs/models

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `command-r` | ✅ | ✅ | ✅ | ✅ |
| `command-r-plus` | ✅ | ✅ | ✅ | ✅ |
| `command-light` | ✅ | ❌ | ❌ | ✅ |

### Embedding Models

| Model ID | Dimensions |
|----------|------------|
| `embed-english-v3.0` | 1024 |
| `embed-multilingual-v3.0` | 1024 |
| `embed-english-light-v3.0` | 384 |

### Test Configuration
```yaml
primary_model: command-r
embedding_model: embed-english-v3.0
```

---

## Groq

**Documentation:** https://console.groq.com/docs/models

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `llama-3.3-70b-versatile` | ✅ | ✅ | ✅ | ✅ |
| `llama-3.1-8b-instant` | ✅ | ✅ | ✅ | ✅ |
| `mixtral-8x7b-32768` | ✅ | ✅ | ❌ | ✅ |
| `gemma2-9b-it` | ✅ | ✅ | ✅ | ✅ |

### Embedding Models
Groq does not provide embedding models.

### Test Configuration
```yaml
primary_model: llama-3.1-8b-instant  # Fast, free tier
secondary_model: llama-3.3-70b-versatile
```

---

## Together AI

**Documentation:** https://docs.together.ai/docs/chat-models

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | ✅ | ✅ | ✅ | ✅ |
| `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | ✅ | ✅ | ✅ | ✅ |
| `Qwen/Qwen2.5-7B-Instruct-Turbo` | ✅ | ✅ | ✅ | ✅ |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | ✅ | ✅ | ❌ | ✅ |

### Embedding Models

| Model ID | Dimensions |
|----------|------------|
| `togethercomputer/m2-bert-80M-8k-retrieval` | 768 |

### Test Configuration
```yaml
primary_model: meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
```

---

## Fireworks AI

**Documentation:** https://docs.fireworks.ai/models

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `accounts/fireworks/models/llama-v3p1-8b-instruct` | ✅ | ✅ | ✅ | ✅ |
| `accounts/fireworks/models/llama-v3p1-70b-instruct` | ✅ | ✅ | ✅ | ✅ |
| `accounts/fireworks/models/mixtral-8x7b-instruct` | ✅ | ✅ | ❌ | ✅ |

### Embedding Models

| Model ID | Dimensions |
|----------|------------|
| `nomic-ai/nomic-embed-text-v1.5` | 768 |

### Test Configuration
```yaml
primary_model: accounts/fireworks/models/llama-v3p1-8b-instruct
```

---

## DeepSeek

**Documentation:** https://platform.deepseek.com/api-docs

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `deepseek-chat` | ✅ | ✅ | ✅ | ✅ |
| `deepseek-coder` | ✅ | ✅ | ✅ | ✅ |
| `deepseek-reasoner` | ✅ | ❌ | ❌ | ✅ |

### Embedding Models
DeepSeek does not provide embedding models via their standard API.

### Test Configuration
```yaml
primary_model: deepseek-chat
```

---

## Perplexity

**Documentation:** https://docs.perplexity.ai/guides/model-cards

### Chat Models (Integration Test Models)

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `sonar` | ✅ | ❌ | ❌ | ✅ |
| `sonar-pro` | ✅ | ❌ | ❌ | ✅ |
| `sonar-reasoning` | ✅ | ❌ | ❌ | ✅ |

### Embedding Models
Perplexity does not provide embedding models.

### Test Configuration
```yaml
primary_model: sonar  # Cost-effective, online search
```

---

## Databricks

**Documentation:** https://docs.databricks.com/en/machine-learning/foundation-models/index.html

### Chat Models (Integration Test Models)
Models depend on your Databricks deployment. Common options:

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `databricks-meta-llama-3-1-70b-instruct` | ✅ | ✅ | ✅ | ✅ |
| `databricks-dbrx-instruct` | ✅ | ❌ | ❌ | ✅ |

### Test Configuration
```yaml
primary_model: databricks-meta-llama-3-1-70b-instruct
requires:
  - DATABRICKS_TOKEN
  - DATABRICKS_HOST
```

---

## Ollama (Local)

**Documentation:** https://ollama.ai/library

### Chat Models (Integration Test Models)
Models must be pulled locally first: `ollama pull <model>`

| Model ID | Streaming | Tool Calling | Structured Output | Usage Reporting |
|----------|-----------|--------------|-------------------|-----------------|
| `llama3.2` | ✅ | ✅ | ✅ | ✅ |
| `mistral` | ✅ | ✅ | ✅ | ✅ |
| `phi3` | ✅ | ✅ | ✅ | ✅ |
| `qwen2.5` | ✅ | ✅ | ✅ | ✅ |

### Embedding Models

| Model ID | Dimensions |
|----------|------------|
| `nomic-embed-text` | 768 |
| `mxbai-embed-large` | 1024 |

### Test Configuration
```yaml
primary_model: llama3.2:1b  # Small, fast for testing
embedding_model: nomic-embed-text
requires:
  - OLLAMA_HOST (default: http://localhost:11434)
```

---

## Integration Test Model Selection

For CI integration tests, we use the smallest/cheapest models that still support all features:

| Provider | Primary Test Model | Features Tested |
|----------|-------------------|-----------------|
| OpenAI | `gpt-4o-mini` | All |
| Anthropic | `claude-3-5-haiku-20241022` | All |
| Gemini | `gemini-1.5-flash` | All |
| Mistral | `mistral-small-latest` | All |
| Cohere | `command-r` | All |
| Groq | `llama-3.1-8b-instant` | All |
| Together AI | `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | All |
| Fireworks AI | `accounts/fireworks/models/llama-v3p1-8b-instruct` | All |
| DeepSeek | `deepseek-chat` | All except tool calling edge cases |
| Perplexity | `sonar` | Basic chat only |

---

## Capability Legend

- ✅ **Fully Supported**: Feature works reliably
- ⚠️ **Partial Support**: Feature works with limitations
- ❌ **Not Supported**: Feature not available for this model

## Notes

1. **Streaming**: All providers support SSE streaming for chat completions.
2. **Tool Calling**: Format varies by provider; fastlitellm normalizes to OpenAI format.
3. **Structured Output**: JSON mode support varies; some providers require specific prompting.
4. **Usage Reporting**: All providers report token usage; streaming usage requires `include_usage` option where supported.
5. **Rate Limits**: CI tests include retry logic to handle rate limiting. See [CI_RELIABILITY.md](CI_RELIABILITY.md) for details.
