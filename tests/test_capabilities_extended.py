"""
Extended tests for capabilities module to improve coverage.
"""

from __future__ import annotations

from arcllm.capabilities.tables import (
    ALL_CAPABILITIES,
    ANTHROPIC_CAPABILITIES,
    GEMINI_CAPABILITIES,
    GROQ_CAPABILITIES,
    OPENAI_CAPABILITIES,
    ModelCapabilities,
    _normalize_model_name,
)


class TestCapabilitiesNormalization:
    """Tests for model name normalization in capabilities."""

    def test_normalize_simple_model(self):
        """Test normalizing model without provider prefix."""
        provider, model = _normalize_model_name("gpt-4o-mini")
        assert provider is None
        assert model == "gpt-4o-mini"

    def test_normalize_with_provider_prefix(self):
        """Test normalizing model with provider prefix."""
        provider, model = _normalize_model_name("openai/gpt-4o-mini")
        assert provider == "openai"
        assert model == "gpt-4o-mini"

    def test_normalize_hyphenated_provider(self):
        """Test normalizing with hyphenated provider name."""
        provider, model = _normalize_model_name("together-ai/model")
        assert provider == "together_ai"
        assert model == "model"


class TestCapabilitiesTables:
    """Tests for capabilities tables content."""

    def test_openai_gpt4o_capabilities(self):
        """Test GPT-4o has correct capabilities."""
        caps = OPENAI_CAPABILITIES["gpt-4o"]
        assert caps.max_tokens == 16384
        assert caps.context_window == 128000
        assert caps.supports_vision is True
        assert caps.supports_tools is True
        assert caps.supports_structured_output is True

    def test_anthropic_claude_capabilities(self):
        """Test Claude has correct capabilities."""
        caps = ANTHROPIC_CAPABILITIES["claude-3-5-sonnet-20241022"]
        assert caps.supports_vision is True
        assert caps.supports_pdf_input is True
        assert caps.supports_tools is True

    def test_gemini_long_context(self):
        """Test Gemini has long context window."""
        caps = GEMINI_CAPABILITIES["gemini-1.5-pro"]
        assert caps.context_window == 2097152  # 2M tokens

    def test_groq_llama_capabilities(self):
        """Test Groq Llama has correct capabilities."""
        caps = GROQ_CAPABILITIES["llama-3.3-70b-versatile"]
        assert caps.supports_tools is True
        assert caps.supports_structured_output is True

    def test_all_capabilities_structure(self):
        """Test ALL_CAPABILITIES has proper structure."""
        for _provider, caps in ALL_CAPABILITIES.items():
            assert isinstance(caps, dict)
            for _model, model_caps in caps.items():
                assert isinstance(model_caps, ModelCapabilities)


class TestModelCapabilitiesDataclass:
    """Tests for ModelCapabilities dataclass."""

    def test_capabilities_with_all_fields(self):
        """Test creating capabilities with all fields."""
        caps = ModelCapabilities(
            max_tokens=4096,
            context_window=8192,
            supports_vision=True,
            supports_pdf_input=True,
            supports_tools=True,
            supports_structured_output=True,
        )
        assert caps.max_tokens == 4096
        assert caps.context_window == 8192
        assert caps.supports_vision is True
        assert caps.supports_pdf_input is True
        assert caps.supports_tools is True
        assert caps.supports_structured_output is True

    def test_capabilities_defaults(self):
        """Test default values for capabilities."""
        caps = ModelCapabilities(max_tokens=None, context_window=None)
        assert caps.supports_vision is False
        assert caps.supports_pdf_input is False
        assert caps.supports_tools is False
        assert caps.supports_structured_output is False
