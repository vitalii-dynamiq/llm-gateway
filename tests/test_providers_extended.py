"""
Extended tests for provider adapters.

Tests to improve coverage for smaller provider adapters that
extend OpenAI-compatible patterns.
"""

from __future__ import annotations

import json

import pytest

from arcllm.providers.base import ProviderConfig


class TestDeepSeekAdapter:
    """Tests for DeepSeekAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create DeepSeek adapter with test key."""
        from arcllm.providers.deepseek_adapter import DeepSeekAdapter

        config = ProviderConfig(api_key="test-key")
        return DeepSeekAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "deepseek"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "deepseek" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"
        body = json.loads(request.body.decode())
        assert body["model"] == "deepseek-chat"


class TestFireworksAdapter:
    """Tests for FireworksAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Fireworks adapter with test key."""
        from arcllm.providers.fireworks_adapter import FireworksAdapter

        config = ProviderConfig(api_key="test-key")
        return FireworksAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "fireworks_ai"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "fireworks" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="accounts/fireworks/models/llama-v3p3-70b-instruct",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"


class TestTogetherAdapter:
    """Tests for TogetherAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Together adapter with test key."""
        from arcllm.providers.together_adapter import TogetherAdapter

        config = ProviderConfig(api_key="test-key")
        return TogetherAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "together_ai"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "together" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"


class TestGroqAdapter:
    """Tests for GroqAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Groq adapter with test key."""
        from arcllm.providers.groq_adapter import GroqAdapter

        config = ProviderConfig(api_key="test-key")
        return GroqAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "groq"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "groq" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"
        body = json.loads(request.body.decode())
        assert body["model"] == "llama-3.3-70b-versatile"


class TestPerplexityAdapter:
    """Tests for PerplexityAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create Perplexity adapter with test key."""
        from arcllm.providers.perplexity_adapter import PerplexityAdapter

        config = ProviderConfig(api_key="test-key")
        return PerplexityAdapter(config)

    def test_provider_name(self, adapter):
        """Test provider name is set correctly."""
        assert adapter.provider_name == "perplexity"

    def test_api_base(self, adapter):
        """Test default API base."""
        assert "perplexity" in adapter._api_base.lower()

    def test_build_request(self, adapter):
        """Test building request."""
        request = adapter.build_request(
            model="sonar-pro",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert request.method == "POST"


class TestCapabilitiesExtended:
    """Extended tests for capabilities module."""

    def test_get_model_capabilities_all_providers(self):
        """Test capabilities for models from all providers."""
        from arcllm.capabilities import get_model_capabilities

        models = [
            "gpt-4o",
            "claude-3-5-sonnet-20241022",
            "gemini-1.5-pro",
            "mistral-large-latest",
            "llama-3.3-70b-versatile",
            "deepseek-chat",
            "sonar-pro",
        ]
        for model in models:
            caps = get_model_capabilities(model)
            assert caps.max_tokens is not None or caps.context_window is not None

    def test_default_capabilities(self):
        """Test default capabilities for unknown models."""
        from arcllm.capabilities.tables import DEFAULT_CAPABILITIES

        assert DEFAULT_CAPABILITIES.max_tokens == 4096
        assert DEFAULT_CAPABILITIES.context_window == 8192
        assert DEFAULT_CAPABILITIES.supports_vision is False

    def test_capabilities_version(self):
        """Test capabilities version is defined."""
        from arcllm.capabilities import CAPABILITIES_VERSION

        assert CAPABILITIES_VERSION is not None
        assert len(CAPABILITIES_VERSION) > 0


class TestBaseProviderRegister:
    """Tests for provider registration in base module."""

    def test_register_all_providers_idempotent(self):
        """Test that register_all_providers can be called multiple times."""
        from arcllm.providers.base import _PROVIDERS, register_all_providers

        initial_count = len(_PROVIDERS)
        register_all_providers()
        # Should not change (providers already registered)
        assert len(_PROVIDERS) >= initial_count
