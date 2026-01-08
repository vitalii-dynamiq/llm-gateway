"""
Model capability tables for fastlitellm.

Provides:
- get_max_tokens(model) -> int | None
- supports_vision(model) -> bool
- supports_pdf_input(model) -> bool
- supports_tools(model) -> bool
- supports_structured_output(model) -> bool
"""

from fastlitellm.capabilities.tables import (
    CAPABILITIES_VERSION,
    ModelCapabilities,
    get_max_tokens,
    get_model_capabilities,
    supports_pdf_input,
    supports_structured_output,
    supports_tools,
    supports_vision,
)

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
