"""Activation collection and management."""

from wisent.core.activations.prompt_construction_strategy import PromptConstructionStrategy
from wisent.core.activations.core.atoms import (
    ActivationAggregationStrategy,
    LayerActivations,
)

__all__ = [
    "ActivationCollector",
    "PromptConstructionStrategy",
    "ActivationAggregationStrategy",
    "LayerActivations",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "ActivationCollector":
        from wisent.core.activations.activations_collector import ActivationCollector
        return ActivationCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
