"""
Model capability tables for supported models.

Last updated: 2025-01-08

Capabilities tracked:
- max_tokens: Maximum output tokens
- context_window: Maximum input context length
- supports_vision: Can process image inputs
- supports_pdf_input: Can process PDF documents directly
- supports_tools: Supports tool/function calling
- supports_structured_output: Supports JSON mode or JSON schema output

To update capabilities:
1. Check provider documentation
2. Update the relevant dict in this file
3. Update CAPABILITIES_VERSION
4. Run tests to ensure format is valid
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CAPABILITIES_VERSION",
    "ModelCapabilities",
    "get_max_tokens",
    "get_model_capabilities",
    "supports_pdf_input",
    "supports_structured_output",
    "supports_tools",
    "supports_vision",
]


# Version for tracking capability table updates
CAPABILITIES_VERSION = "2025.01.08"


@dataclass(slots=True, frozen=True)
class ModelCapabilities:
    """Capability information for a model."""

    max_tokens: int | None
    context_window: int | None
    supports_vision: bool = False
    supports_pdf_input: bool = False
    supports_tools: bool = False
    supports_structured_output: bool = False


# =============================================================================
# OpenAI Capabilities
# =============================================================================

OPENAI_CAPABILITIES: dict[str, ModelCapabilities] = {
    # GPT-4o series
    "gpt-4o": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-2024-11-20": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-2024-08-06": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-2024-05-13": ModelCapabilities(4096, 128000, True, False, True, True),
    "gpt-4o-mini": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-mini-2024-07-18": ModelCapabilities(16384, 128000, True, False, True, True),
    "chatgpt-4o-latest": ModelCapabilities(16384, 128000, True, False, True, True),
    # GPT-4o Audio
    "gpt-4o-audio-preview": ModelCapabilities(16384, 128000, True, False, True, True),
    "gpt-4o-audio-preview-2024-12-17": ModelCapabilities(16384, 128000, True, False, True, True),
    # o1 series (reasoning)
    "o1": ModelCapabilities(100000, 200000, True, False, True, True),
    "o1-2024-12-17": ModelCapabilities(100000, 200000, True, False, True, True),
    "o1-preview": ModelCapabilities(32768, 128000, False, False, False, False),
    "o1-preview-2024-09-12": ModelCapabilities(32768, 128000, False, False, False, False),
    "o1-mini": ModelCapabilities(65536, 128000, False, False, False, False),
    "o1-mini-2024-09-12": ModelCapabilities(65536, 128000, False, False, False, False),
    # GPT-4 Turbo
    "gpt-4-turbo": ModelCapabilities(4096, 128000, True, False, True, True),
    "gpt-4-turbo-2024-04-09": ModelCapabilities(4096, 128000, True, False, True, True),
    "gpt-4-turbo-preview": ModelCapabilities(4096, 128000, False, False, True, True),
    "gpt-4-0125-preview": ModelCapabilities(4096, 128000, False, False, True, True),
    "gpt-4-1106-preview": ModelCapabilities(4096, 128000, False, False, True, True),
    "gpt-4-vision-preview": ModelCapabilities(4096, 128000, True, False, True, False),
    # GPT-4
    "gpt-4": ModelCapabilities(8192, 8192, False, False, True, False),
    "gpt-4-0613": ModelCapabilities(8192, 8192, False, False, True, False),
    "gpt-4-32k": ModelCapabilities(8192, 32768, False, False, True, False),
    "gpt-4-32k-0613": ModelCapabilities(8192, 32768, False, False, True, False),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": ModelCapabilities(4096, 16385, False, False, True, True),
    "gpt-3.5-turbo-0125": ModelCapabilities(4096, 16385, False, False, True, True),
    "gpt-3.5-turbo-1106": ModelCapabilities(4096, 16385, False, False, True, True),
    "gpt-3.5-turbo-16k": ModelCapabilities(4096, 16385, False, False, True, False),
}


# =============================================================================
# Anthropic Capabilities
# =============================================================================

ANTHROPIC_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022": ModelCapabilities(8192, 200000, True, True, True, True),
    "claude-3-5-sonnet-latest": ModelCapabilities(8192, 200000, True, True, True, True),
    "claude-3-5-sonnet-20240620": ModelCapabilities(8192, 200000, True, False, True, True),
    "claude-3-5-haiku-20241022": ModelCapabilities(8192, 200000, True, False, True, True),
    "claude-3-5-haiku-latest": ModelCapabilities(8192, 200000, True, False, True, True),
    # Claude 3 series
    "claude-3-opus-20240229": ModelCapabilities(4096, 200000, True, False, True, True),
    "claude-3-opus-latest": ModelCapabilities(4096, 200000, True, False, True, True),
    "claude-3-sonnet-20240229": ModelCapabilities(4096, 200000, True, False, True, True),
    "claude-3-haiku-20240307": ModelCapabilities(4096, 200000, True, False, True, True),
    # Legacy Claude 2
    "claude-2.1": ModelCapabilities(4096, 200000, False, False, False, False),
    "claude-2.0": ModelCapabilities(4096, 100000, False, False, False, False),
    "claude-instant-1.2": ModelCapabilities(4096, 100000, False, False, False, False),
}


# =============================================================================
# Google Gemini Capabilities
# =============================================================================

GEMINI_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Gemini 2.0
    "gemini-2.0-flash-exp": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-2.0-flash-thinking-exp": ModelCapabilities(8192, 32767, False, False, False, False),
    # Gemini 1.5 Pro
    "gemini-1.5-pro": ModelCapabilities(8192, 2097152, True, True, True, True),
    "gemini-1.5-pro-latest": ModelCapabilities(8192, 2097152, True, True, True, True),
    "gemini-1.5-pro-001": ModelCapabilities(8192, 2097152, True, True, True, True),
    "gemini-1.5-pro-002": ModelCapabilities(8192, 2097152, True, True, True, True),
    # Gemini 1.5 Flash
    "gemini-1.5-flash": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-latest": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-001": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-002": ModelCapabilities(8192, 1048576, True, True, True, True),
    "gemini-1.5-flash-8b": ModelCapabilities(8192, 1048576, True, True, True, True),
    # Gemini 1.0 Pro
    "gemini-1.0-pro": ModelCapabilities(8192, 32768, False, False, True, False),
    "gemini-pro": ModelCapabilities(8192, 32768, False, False, True, False),
}


# =============================================================================
# Mistral Capabilities
# =============================================================================

MISTRAL_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Large models
    "mistral-large-latest": ModelCapabilities(131072, 131072, False, False, True, True),
    "mistral-large-2411": ModelCapabilities(131072, 131072, False, False, True, True),
    "mistral-large-2407": ModelCapabilities(131072, 131072, False, False, True, True),
    # Pixtral (vision)
    "pixtral-large-latest": ModelCapabilities(131072, 131072, True, False, True, True),
    "pixtral-large-2411": ModelCapabilities(131072, 131072, True, False, True, True),
    "pixtral-12b-2409": ModelCapabilities(131072, 131072, True, False, True, False),
    # Small models
    "mistral-small-latest": ModelCapabilities(32768, 32768, False, False, True, True),
    "mistral-small-2409": ModelCapabilities(32768, 32768, False, False, True, True),
    "mistral-nemo-latest": ModelCapabilities(131072, 131072, False, False, True, True),
    "mistral-nemo-2407": ModelCapabilities(131072, 131072, False, False, True, True),
    # Codestral
    "codestral-latest": ModelCapabilities(32768, 32768, False, False, True, False),
    "codestral-2405": ModelCapabilities(32768, 32768, False, False, True, False),
    # Ministral
    "ministral-3b-latest": ModelCapabilities(131072, 131072, False, False, True, False),
    "ministral-8b-latest": ModelCapabilities(131072, 131072, False, False, True, False),
}


# =============================================================================
# Cohere Capabilities
# =============================================================================

COHERE_CAPABILITIES: dict[str, ModelCapabilities] = {
    "command-r-plus": ModelCapabilities(4096, 128000, False, False, True, False),
    "command-r-plus-08-2024": ModelCapabilities(4096, 128000, False, False, True, False),
    "command-r": ModelCapabilities(4096, 128000, False, False, True, False),
    "command-r-08-2024": ModelCapabilities(4096, 128000, False, False, True, False),
    "command": ModelCapabilities(4096, 4096, False, False, False, False),
    "command-light": ModelCapabilities(4096, 4096, False, False, False, False),
}


# =============================================================================
# Groq Capabilities
# =============================================================================

GROQ_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Llama 3.3
    "llama-3.3-70b-versatile": ModelCapabilities(32768, 128000, False, False, True, True),
    # Llama 3.2 Vision
    "llama-3.2-90b-vision-preview": ModelCapabilities(8192, 128000, True, False, True, False),
    "llama-3.2-11b-vision-preview": ModelCapabilities(8192, 128000, True, False, True, False),
    # Llama 3.2
    "llama-3.2-3b-preview": ModelCapabilities(8192, 128000, False, False, True, True),
    "llama-3.2-1b-preview": ModelCapabilities(8192, 128000, False, False, True, True),
    # Llama 3.1
    "llama-3.1-70b-versatile": ModelCapabilities(32768, 128000, False, False, True, True),
    "llama-3.1-8b-instant": ModelCapabilities(8192, 128000, False, False, True, True),
    # Llama 3
    "llama3-70b-8192": ModelCapabilities(8192, 8192, False, False, True, False),
    "llama3-8b-8192": ModelCapabilities(8192, 8192, False, False, True, False),
    # Mixtral
    "mixtral-8x7b-32768": ModelCapabilities(32768, 32768, False, False, True, False),
    # Gemma
    "gemma2-9b-it": ModelCapabilities(8192, 8192, False, False, True, True),
    "gemma-7b-it": ModelCapabilities(8192, 8192, False, False, False, False),
}


# =============================================================================
# Together AI Capabilities
# =============================================================================

TOGETHER_CAPABILITIES: dict[str, ModelCapabilities] = {
    # Llama 3.3
    "meta-llama/Llama-3.3-70B-Instruct-Turbo": ModelCapabilities(
        8192, 131072, False, False, True, True
    ),
    # Llama 3.2 Vision
    "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo": ModelCapabilities(
        4096, 131072, True, False, True, False
    ),
    "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo": ModelCapabilities(
        4096, 131072, True, False, True, False
    ),
    # Llama 3.2
    "meta-llama/Llama-3.2-3B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    # Llama 3.1
    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo": ModelCapabilities(
        4096, 131072, False, False, True, True
    ),
    # Qwen 2.5
    "Qwen/Qwen2.5-72B-Instruct-Turbo": ModelCapabilities(4096, 131072, False, False, True, True),
    # DeepSeek
    "deepseek-ai/DeepSeek-V3": ModelCapabilities(8192, 131072, False, False, True, True),
}


# =============================================================================
# DeepSeek Capabilities
# =============================================================================

DEEPSEEK_CAPABILITIES: dict[str, ModelCapabilities] = {
    "deepseek-chat": ModelCapabilities(8192, 65536, False, False, True, True),
    "deepseek-reasoner": ModelCapabilities(8192, 65536, False, False, False, False),
    "deepseek-coder": ModelCapabilities(8192, 65536, False, False, True, True),
}


# =============================================================================
# Perplexity Capabilities
# =============================================================================

PERPLEXITY_CAPABILITIES: dict[str, ModelCapabilities] = {
    "sonar-pro": ModelCapabilities(8192, 200000, False, False, False, False),
    "sonar": ModelCapabilities(8192, 128000, False, False, False, False),
    "sonar-reasoning-pro": ModelCapabilities(8192, 128000, False, False, False, False),
    "sonar-reasoning": ModelCapabilities(8192, 128000, False, False, False, False),
}


# =============================================================================
# Combined capability lookup
# =============================================================================

ALL_CAPABILITIES: dict[str, dict[str, ModelCapabilities]] = {
    "openai": OPENAI_CAPABILITIES,
    "anthropic": ANTHROPIC_CAPABILITIES,
    "gemini": GEMINI_CAPABILITIES,
    "vertex_ai": GEMINI_CAPABILITIES,
    "mistral": MISTRAL_CAPABILITIES,
    "cohere": COHERE_CAPABILITIES,
    "groq": GROQ_CAPABILITIES,
    "together_ai": TOGETHER_CAPABILITIES,
    "deepseek": DEEPSEEK_CAPABILITIES,
    "perplexity": PERPLEXITY_CAPABILITIES,
}

# Default capabilities for unknown models
DEFAULT_CAPABILITIES = ModelCapabilities(
    max_tokens=4096,
    context_window=8192,
    supports_vision=False,
    supports_pdf_input=False,
    supports_tools=False,
    supports_structured_output=False,
)


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
        if parts[0].lower() in ALL_CAPABILITIES:
            provider = parts[0].lower()
            model_name = parts[1]
        elif parts[0].lower().replace("-", "_") in ALL_CAPABILITIES:
            provider = parts[0].lower().replace("-", "_")
            model_name = parts[1]

    return provider, model_name


def get_model_capabilities(model: str) -> ModelCapabilities:
    """
    Get capability information for a model.

    Args:
        model: Model identifier (with or without provider prefix)

    Returns:
        ModelCapabilities for the model, or DEFAULT_CAPABILITIES if unknown
    """
    provider, model_name = _normalize_model_name(model)

    # If provider specified, look only in that provider's capabilities
    if provider:
        cap_table = ALL_CAPABILITIES.get(provider, {})
        if model_name in cap_table:
            return cap_table[model_name]
        return DEFAULT_CAPABILITIES

    # Search all providers
    for cap_table in ALL_CAPABILITIES.values():
        if model_name in cap_table:
            return cap_table[model_name]

    return DEFAULT_CAPABILITIES


def get_max_tokens(model: str) -> int | None:
    """
    Get maximum output tokens for a model.

    Args:
        model: Model identifier

    Returns:
        Maximum tokens or None if unknown
    """
    caps = get_model_capabilities(model)
    return caps.max_tokens


def supports_vision(model: str) -> bool:
    """
    Check if model supports vision/image inputs.

    Args:
        model: Model identifier

    Returns:
        True if model supports vision
    """
    caps = get_model_capabilities(model)
    return caps.supports_vision


def supports_pdf_input(model: str) -> bool:
    """
    Check if model supports direct PDF document input.

    Args:
        model: Model identifier

    Returns:
        True if model supports PDF input
    """
    caps = get_model_capabilities(model)
    return caps.supports_pdf_input


def supports_tools(model: str) -> bool:
    """
    Check if model supports tool/function calling.

    Args:
        model: Model identifier

    Returns:
        True if model supports tools
    """
    caps = get_model_capabilities(model)
    return caps.supports_tools


def supports_structured_output(model: str) -> bool:
    """
    Check if model supports structured output (JSON mode/schema).

    Args:
        model: Model identifier

    Returns:
        True if model supports structured output
    """
    caps = get_model_capabilities(model)
    return caps.supports_structured_output
