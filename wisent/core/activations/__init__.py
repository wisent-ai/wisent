"""Activation collection and management."""

from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.activations.extraction_strategy import (
    ExtractionStrategy,
    build_extraction_texts,
    extract_activation,
    add_extraction_strategy_args,
)
from wisent.core.activations.classifier_inference_strategy import (
    ClassifierInferenceStrategy,
    extract_inference_activation,
    get_inference_score,
    get_recommended_inference_strategy,
    add_classifier_inference_strategy_args,
)

__all__ = [
    "ActivationCollector",
    "Activations",
    "ExtractionStrategy",
    "build_extraction_texts",
    "extract_activation",
    "add_extraction_strategy_args",
    "ClassifierInferenceStrategy",
    "extract_inference_activation",
    "get_inference_score",
    "get_recommended_inference_strategy",
    "add_classifier_inference_strategy_args",
    "LayerActivations",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "ActivationCollector":
        from wisent.core.activations.activations_collector import ActivationCollector
        return ActivationCollector
    if name == "Activations":
        from wisent.core.activations.activations import Activations
        return Activations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
