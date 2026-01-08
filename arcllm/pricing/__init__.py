"""
Pricing tables and cost calculation for arcllm.

Provides:
- cost_per_token(model, prompt_tokens, completion_tokens)
- completion_cost(response, model)
- get_model_pricing(model)
"""

from arcllm.pricing.tables import (
    PRICING_VERSION,
    ModelPricing,
    completion_cost,
    cost_per_token,
    get_model_pricing,
)

__all__ = [
    "PRICING_VERSION",
    "ModelPricing",
    "completion_cost",
    "cost_per_token",
    "get_model_pricing",
]
