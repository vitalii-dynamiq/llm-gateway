"""
Pricing tables for supported models.

Prices are in USD per 1 million tokens.
Last updated: 2025-01-08

To update prices:
1. Check provider pricing pages
2. Update the relevant dict in this file
3. Update PRICING_VERSION
4. Run tests to ensure format is valid
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastlitellm.exceptions import FastLiteLLMError

if TYPE_CHECKING:
    from fastlitellm.types import ModelResponse

__all__ = [
    "PRICING_VERSION",
    "ModelPricing",
    "completion_cost",
    "cost_per_token",
    "get_model_pricing",
]


# Version for tracking pricing table updates
PRICING_VERSION = "2025.01.08"


@dataclass(slots=True, frozen=True)
class ModelPricing:
    """Pricing information for a model."""

    input_cost_per_million: float
    output_cost_per_million: float
    # Optional: cached input pricing (for prompt caching)
    cached_input_cost_per_million: float | None = None


# =============================================================================
# OpenAI Pricing (USD per 1M tokens)
# https://openai.com/pricing
# =============================================================================

OPENAI_PRICING: dict[str, ModelPricing] = {
    # GPT-4o series
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-11-20": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-08-06": ModelPricing(2.50, 10.00),
    "gpt-4o-2024-05-13": ModelPricing(5.00, 15.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4o-mini-2024-07-18": ModelPricing(0.15, 0.60),
    "chatgpt-4o-latest": ModelPricing(5.00, 15.00),
    # GPT-4o Audio
    "gpt-4o-audio-preview": ModelPricing(2.50, 10.00),
    "gpt-4o-audio-preview-2024-12-17": ModelPricing(2.50, 10.00),
    "gpt-4o-audio-preview-2024-10-01": ModelPricing(2.50, 10.00),
    # o1 series (reasoning)
    "o1": ModelPricing(15.00, 60.00),
    "o1-2024-12-17": ModelPricing(15.00, 60.00),
    "o1-preview": ModelPricing(15.00, 60.00),
    "o1-preview-2024-09-12": ModelPricing(15.00, 60.00),
    "o1-mini": ModelPricing(3.00, 12.00),
    "o1-mini-2024-09-12": ModelPricing(3.00, 12.00),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-2024-04-09": ModelPricing(10.00, 30.00),
    "gpt-4-turbo-preview": ModelPricing(10.00, 30.00),
    "gpt-4-0125-preview": ModelPricing(10.00, 30.00),
    "gpt-4-1106-preview": ModelPricing(10.00, 30.00),
    "gpt-4-vision-preview": ModelPricing(10.00, 30.00),
    "gpt-4-1106-vision-preview": ModelPricing(10.00, 30.00),
    # GPT-4
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-4-0613": ModelPricing(30.00, 60.00),
    "gpt-4-0314": ModelPricing(30.00, 60.00),
    "gpt-4-32k": ModelPricing(60.00, 120.00),
    "gpt-4-32k-0613": ModelPricing(60.00, 120.00),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-0125": ModelPricing(0.50, 1.50),
    "gpt-3.5-turbo-1106": ModelPricing(1.00, 2.00),
    "gpt-3.5-turbo-instruct": ModelPricing(1.50, 2.00),
    "gpt-3.5-turbo-16k": ModelPricing(3.00, 4.00),
    # Embeddings
    "text-embedding-3-small": ModelPricing(0.02, 0.0),
    "text-embedding-3-large": ModelPricing(0.13, 0.0),
    "text-embedding-ada-002": ModelPricing(0.10, 0.0),
}


# =============================================================================
# Anthropic Pricing (USD per 1M tokens)
# https://www.anthropic.com/pricing
# =============================================================================

ANTHROPIC_PRICING: dict[str, ModelPricing] = {
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-sonnet-latest": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-sonnet-20240620": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, 0.08),
    "claude-3-5-haiku-latest": ModelPricing(0.80, 4.00, 0.08),
    # Claude 3 series
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, 1.50),
    "claude-3-opus-latest": ModelPricing(15.00, 75.00, 1.50),
    "claude-3-sonnet-20240229": ModelPricing(3.00, 15.00, 0.30),
    "claude-3-haiku-20240307": ModelPricing(0.25, 1.25, 0.03),
    # Legacy Claude 2
    "claude-2.1": ModelPricing(8.00, 24.00),
    "claude-2.0": ModelPricing(8.00, 24.00),
    "claude-instant-1.2": ModelPricing(0.80, 2.40),
}


# =============================================================================
# Google Gemini Pricing (USD per 1M tokens)
# https://ai.google.dev/pricing
# =============================================================================

GEMINI_PRICING: dict[str, ModelPricing] = {
    # Gemini 2.0
    "gemini-2.0-flash-exp": ModelPricing(0.0, 0.0),  # Free preview
    "gemini-2.0-flash-thinking-exp": ModelPricing(0.0, 0.0),  # Free preview
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),  # <=128k
    "gemini-1.5-pro-latest": ModelPricing(1.25, 5.00),
    "gemini-1.5-pro-001": ModelPricing(1.25, 5.00),
    "gemini-1.5-pro-002": ModelPricing(1.25, 5.00),
    # Gemini 1.5 Flash
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),  # <=128k
    "gemini-1.5-flash-latest": ModelPricing(0.075, 0.30),
    "gemini-1.5-flash-001": ModelPricing(0.075, 0.30),
    "gemini-1.5-flash-002": ModelPricing(0.075, 0.30),
    "gemini-1.5-flash-8b": ModelPricing(0.0375, 0.15),  # <=128k
    "gemini-1.5-flash-8b-001": ModelPricing(0.0375, 0.15),
    # Gemini 1.0 Pro
    "gemini-1.0-pro": ModelPricing(0.50, 1.50),
    "gemini-1.0-pro-latest": ModelPricing(0.50, 1.50),
    "gemini-1.0-pro-001": ModelPricing(0.50, 1.50),
    "gemini-pro": ModelPricing(0.50, 1.50),  # Alias
    # Embeddings
    "text-embedding-004": ModelPricing(0.00, 0.0),  # Free tier
    "embedding-001": ModelPricing(0.00, 0.0),
}


# =============================================================================
# Mistral Pricing (USD per 1M tokens)
# https://mistral.ai/technology/#pricing
# =============================================================================

MISTRAL_PRICING: dict[str, ModelPricing] = {
    # Premier models
    "mistral-large-latest": ModelPricing(2.00, 6.00),
    "mistral-large-2411": ModelPricing(2.00, 6.00),
    "mistral-large-2407": ModelPricing(2.00, 6.00),
    "pixtral-large-latest": ModelPricing(2.00, 6.00),
    "pixtral-large-2411": ModelPricing(2.00, 6.00),
    # Free models
    "mistral-small-latest": ModelPricing(0.20, 0.60),
    "mistral-small-2409": ModelPricing(0.20, 0.60),
    "pixtral-12b-2409": ModelPricing(0.15, 0.15),
    "mistral-nemo-latest": ModelPricing(0.15, 0.15),
    "mistral-nemo-2407": ModelPricing(0.15, 0.15),
    # Codestral
    "codestral-latest": ModelPricing(0.20, 0.60),
    "codestral-2405": ModelPricing(0.20, 0.60),
    # Ministral
    "ministral-3b-latest": ModelPricing(0.04, 0.04),
    "ministral-3b-2410": ModelPricing(0.04, 0.04),
    "ministral-8b-latest": ModelPricing(0.10, 0.10),
    "ministral-8b-2410": ModelPricing(0.10, 0.10),
    # Legacy
    "mistral-medium-latest": ModelPricing(2.70, 8.10),
    "mistral-tiny": ModelPricing(0.25, 0.25),
    "open-mistral-7b": ModelPricing(0.25, 0.25),
    "open-mixtral-8x7b": ModelPricing(0.70, 0.70),
    "open-mixtral-8x22b": ModelPricing(2.00, 6.00),
    # Embeddings
    "mistral-embed": ModelPricing(0.10, 0.0),
}


# =============================================================================
# Cohere Pricing (USD per 1M tokens)
# https://cohere.com/pricing
# =============================================================================

COHERE_PRICING: dict[str, ModelPricing] = {
    # Command R+
    "command-r-plus": ModelPricing(2.50, 10.00),
    "command-r-plus-08-2024": ModelPricing(2.50, 10.00),
    "command-r-plus-04-2024": ModelPricing(3.00, 15.00),
    # Command R
    "command-r": ModelPricing(0.15, 0.60),
    "command-r-08-2024": ModelPricing(0.15, 0.60),
    "command-r-03-2024": ModelPricing(0.50, 1.50),
    # Command
    "command": ModelPricing(1.00, 2.00),
    "command-light": ModelPricing(0.30, 0.60),
    "command-nightly": ModelPricing(1.00, 2.00),
    "command-light-nightly": ModelPricing(0.30, 0.60),
    # Embeddings
    "embed-english-v3.0": ModelPricing(0.10, 0.0),
    "embed-multilingual-v3.0": ModelPricing(0.10, 0.0),
    "embed-english-light-v3.0": ModelPricing(0.10, 0.0),
    "embed-multilingual-light-v3.0": ModelPricing(0.10, 0.0),
    "embed-english-v2.0": ModelPricing(0.10, 0.0),
    "embed-multilingual-v2.0": ModelPricing(0.10, 0.0),
}


# =============================================================================
# Groq Pricing (USD per 1M tokens)
# https://groq.com/pricing/
# =============================================================================

GROQ_PRICING: dict[str, ModelPricing] = {
    # Llama 3.3
    "llama-3.3-70b-versatile": ModelPricing(0.59, 0.79),
    "llama-3.3-70b-specdec": ModelPricing(0.59, 0.99),
    # Llama 3.2
    "llama-3.2-90b-vision-preview": ModelPricing(0.90, 0.90),
    "llama-3.2-11b-vision-preview": ModelPricing(0.18, 0.18),
    "llama-3.2-3b-preview": ModelPricing(0.06, 0.06),
    "llama-3.2-1b-preview": ModelPricing(0.04, 0.04),
    # Llama 3.1
    "llama-3.1-405b-reasoning": ModelPricing(0.00, 0.00),  # Preview
    "llama-3.1-70b-versatile": ModelPricing(0.59, 0.79),
    "llama-3.1-8b-instant": ModelPricing(0.05, 0.08),
    # Llama 3
    "llama3-70b-8192": ModelPricing(0.59, 0.79),
    "llama3-8b-8192": ModelPricing(0.05, 0.08),
    # Mixtral
    "mixtral-8x7b-32768": ModelPricing(0.24, 0.24),
    # Gemma
    "gemma2-9b-it": ModelPricing(0.20, 0.20),
    "gemma-7b-it": ModelPricing(0.07, 0.07),
}


# =============================================================================
# Together AI Pricing (USD per 1M tokens)
# https://www.together.ai/pricing
# =============================================================================

TOGETHER_PRICING: dict[str, ModelPricing] = {
    # Llama 3.3
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    # Llama 3.2
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": ModelPricing(1.20, 1.20),
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": ModelPricing(0.18, 0.18),
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ModelPricing(0.06, 0.06),
    # Llama 3.1
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ModelPricing(3.50, 3.50),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelPricing(0.88, 0.88),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelPricing(0.18, 0.18),
    # Qwen 2.5
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelPricing(1.20, 1.20),
    "Qwen/Qwen2.5-7B-Instruct-Turbo": ModelPricing(0.30, 0.30),
    # Mixtral
    "mistralai/Mixtral-8x22B-Instruct-v0.1": ModelPricing(1.20, 1.20),
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelPricing(0.60, 0.60),
    # DeepSeek
    "deepseek-ai/DeepSeek-V3": ModelPricing(0.90, 0.90),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelPricing(0.90, 0.90),
}


# =============================================================================
# Fireworks AI Pricing (USD per 1M tokens)
# https://fireworks.ai/pricing
# =============================================================================

FIREWORKS_PRICING: dict[str, ModelPricing] = {
    # Llama 3.3
    "accounts/fireworks/models/llama-v3p3-70b-instruct": ModelPricing(0.90, 0.90),
    # Llama 3.2
    "accounts/fireworks/models/llama-v3p2-90b-vision-instruct": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/llama-v3p2-11b-vision-instruct": ModelPricing(0.20, 0.20),
    "accounts/fireworks/models/llama-v3p2-3b-instruct": ModelPricing(0.10, 0.10),
    "accounts/fireworks/models/llama-v3p2-1b-instruct": ModelPricing(0.10, 0.10),
    # Llama 3.1
    "accounts/fireworks/models/llama-v3p1-405b-instruct": ModelPricing(3.00, 3.00),
    "accounts/fireworks/models/llama-v3p1-70b-instruct": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/llama-v3p1-8b-instruct": ModelPricing(0.20, 0.20),
    # Mixtral
    "accounts/fireworks/models/mixtral-8x22b-instruct": ModelPricing(0.90, 0.90),
    "accounts/fireworks/models/mixtral-8x7b-instruct": ModelPricing(0.50, 0.50),
    # Qwen
    "accounts/fireworks/models/qwen2p5-72b-instruct": ModelPricing(0.90, 0.90),
    # DeepSeek
    "accounts/fireworks/models/deepseek-v3": ModelPricing(0.90, 0.90),
}


# =============================================================================
# DeepSeek Pricing (USD per 1M tokens)
# https://platform.deepseek.com/api-docs/pricing
# =============================================================================

DEEPSEEK_PRICING: dict[str, ModelPricing] = {
    "deepseek-chat": ModelPricing(0.14, 0.28, 0.014),  # V3
    "deepseek-reasoner": ModelPricing(0.55, 2.19),  # R1
    "deepseek-coder": ModelPricing(0.14, 0.28),
}


# =============================================================================
# Perplexity Pricing (USD per 1M tokens)
# https://docs.perplexity.ai/guides/pricing
# =============================================================================

PERPLEXITY_PRICING: dict[str, ModelPricing] = {
    # Sonar Pro (online models)
    "sonar-pro": ModelPricing(3.00, 15.00),
    "sonar": ModelPricing(1.00, 1.00),
    # Sonar Reasoning
    "sonar-reasoning-pro": ModelPricing(2.00, 8.00),
    "sonar-reasoning": ModelPricing(1.00, 5.00),
    # Legacy
    "llama-3.1-sonar-small-128k-online": ModelPricing(0.20, 0.20),
    "llama-3.1-sonar-large-128k-online": ModelPricing(1.00, 1.00),
    "llama-3.1-sonar-huge-128k-online": ModelPricing(5.00, 5.00),
}


# =============================================================================
# Combined pricing lookup
# =============================================================================

ALL_PRICING: dict[str, dict[str, ModelPricing]] = {
    "openai": OPENAI_PRICING,
    "anthropic": ANTHROPIC_PRICING,
    "gemini": GEMINI_PRICING,
    "vertex_ai": GEMINI_PRICING,  # Same models, same pricing
    "mistral": MISTRAL_PRICING,
    "cohere": COHERE_PRICING,
    "groq": GROQ_PRICING,
    "together_ai": TOGETHER_PRICING,
    "fireworks_ai": FIREWORKS_PRICING,
    "deepseek": DEEPSEEK_PRICING,
    "perplexity": PERPLEXITY_PRICING,
}


class UnknownModelPricingError(FastLiteLLMError):
    """Raised when pricing is not available for a model."""

    pass


def _normalize_model_name(model: str) -> tuple[str | None, str]:
    """
    Normalize model name and extract provider if specified.

    Returns:
        Tuple of (provider or None, normalized model name)
    """
    provider = None
    model_name = model

    # Check for provider prefix
    if "/" in model:
        parts = model.split("/", 1)
        if parts[0].lower() in ALL_PRICING:
            provider = parts[0].lower()
            model_name = parts[1]
        elif parts[0].lower().replace("-", "_") in ALL_PRICING:
            provider = parts[0].lower().replace("-", "_")
            model_name = parts[1]

    return provider, model_name


def get_model_pricing(model: str) -> ModelPricing:
    """
    Get pricing information for a model.

    Args:
        model: Model identifier (with or without provider prefix)

    Returns:
        ModelPricing with input and output costs per million tokens

    Raises:
        UnknownModelPricingError: If pricing is not available
    """
    provider, model_name = _normalize_model_name(model)

    # If provider specified, look only in that provider's pricing
    if provider:
        pricing_table = ALL_PRICING.get(provider, {})
        if model_name in pricing_table:
            return pricing_table[model_name]
        raise UnknownModelPricingError(
            f"No pricing available for model '{model_name}' from provider '{provider}'",
            model=model,
            provider=provider,
        )

    # Search all providers
    for _prov, pricing_table in ALL_PRICING.items():
        if model_name in pricing_table:
            return pricing_table[model_name]

    raise UnknownModelPricingError(
        f"No pricing available for model '{model}'",
        model=model,
    )


def cost_per_token(
    model: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> tuple[float, float]:
    """
    Calculate cost for given token counts.

    Args:
        model: Model identifier
        prompt_tokens: Number of prompt/input tokens
        completion_tokens: Number of completion/output tokens

    Returns:
        Tuple of (prompt_cost, completion_cost) in USD

    Raises:
        UnknownModelPricingError: If pricing is not available
    """
    pricing = get_model_pricing(model)

    prompt_cost = (prompt_tokens / 1_000_000) * pricing.input_cost_per_million
    completion_cost = (completion_tokens / 1_000_000) * pricing.output_cost_per_million

    return (prompt_cost, completion_cost)


def completion_cost(
    response: ModelResponse,
    model: str | None = None,
) -> float:
    """
    Calculate total cost for a completion response.

    Args:
        response: ModelResponse from completion call
        model: Optional model override (uses response.model if not provided)

    Returns:
        Total cost in USD

    Raises:
        UnknownModelPricingError: If pricing is not available
    """
    model_name = model or response.model
    if not model_name:
        raise ValueError("Model name required but not provided")

    usage = response.usage
    if usage is None:
        return 0.0

    prompt_cost, completion_cost = cost_per_token(
        model_name,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )

    return prompt_cost + completion_cost
