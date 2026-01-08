# Provider Documentation

This directory contains documentation for each supported LLM provider, including:

- API documentation links
- Supported models
- Feature support (streaming, tools, structured output, embeddings)
- Integration test models

## Provider List

| Provider | Status | Docs | Integration Tests |
|----------|--------|------|-------------------|
| [OpenAI](./openai.md) | ✅ Stable | [Link](https://platform.openai.com/docs/api-reference) | `gpt-4o-mini` |
| [Anthropic](./anthropic.md) | ✅ Stable | [Link](https://docs.anthropic.com/en/api) | `claude-3-5-haiku-20241022` |
| [Google Gemini](./gemini.md) | ✅ Stable | [Link](https://ai.google.dev/api/rest) | `gemini-1.5-flash` |
| [Mistral](./mistral.md) | ✅ Stable | [Link](https://docs.mistral.ai/api/) | `mistral-small-latest` |
| [Cohere](./cohere.md) | ✅ Stable | [Link](https://docs.cohere.com/reference/chat) | `command-r` |
| [Groq](./groq.md) | ✅ Stable | [Link](https://console.groq.com/docs/api-reference) | `llama-3.3-70b-versatile` |
| [Together AI](./together.md) | ✅ Stable | [Link](https://docs.together.ai/reference/chat) | `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| [Fireworks AI](./fireworks.md) | ✅ Stable | [Link](https://docs.fireworks.ai/api-reference) | `accounts/fireworks/models/llama-v3p3-70b-instruct` |
| [DeepSeek](./deepseek.md) | ✅ Stable | [Link](https://platform.deepseek.com/api-docs) | `deepseek-chat` |
| [Perplexity](./perplexity.md) | ✅ Stable | [Link](https://docs.perplexity.ai/api-reference) | `sonar` |
| [Azure OpenAI](./azure.md) | ✅ Stable | [Link](https://learn.microsoft.com/en-us/azure/ai-services/openai/reference) | deployment-specific |
| [AWS Bedrock](./bedrock.md) | ⚠️ Beta | [Link](https://docs.aws.amazon.com/bedrock/latest/APIReference/) | - |
| [Google Vertex AI](./vertex.md) | ⚠️ Beta | [Link](https://cloud.google.com/vertex-ai/docs/reference/rest) | - |
| [Databricks](./databricks.md) | ⚠️ Beta | [Link](https://docs.databricks.com/en/machine-learning/foundation-models/api-reference.html) | - |
| [Ollama](./ollama.md) | ✅ Stable | [Link](https://github.com/ollama/ollama/blob/main/docs/api.md) | local only |

## Capability Matrix

See [CAPABILITIES.md](./CAPABILITIES.md) for a detailed feature matrix.
