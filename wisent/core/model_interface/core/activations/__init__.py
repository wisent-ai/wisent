import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if not d.startswith((".", "_")))
    if _root != _base:
        __path__.append(_root)

"""Activation collection and management."""

from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.activations.strategies import (
    ExtractionStrategy,
    ExtractionComponent,
    tokenizer_has_chat_template,
    build_extraction_texts,
    extract_activation,
    add_extraction_strategy_args,
    add_extraction_component_args,
    ClassifierInferenceStrategy,
    extract_inference_activation,
    get_inference_score,
    get_recommended_inference_strategy,
    add_classifier_inference_strategy_args,
)
from wisent.core.activations.core.optimal_extraction import (
    OptimalExtractionResult,
    compute_signal_trajectory,
    extract_at_optimal_position,
    extract_at_max_diff_norm,
    extract_batch_optimal,
    compare_extraction_strategies,
)

__all__ = [
    "ActivationCollector",
    "Activations",
    "ExtractionStrategy",
    "ExtractionComponent",
    "tokenizer_has_chat_template",
    "build_extraction_texts",
    "extract_activation",
    "add_extraction_strategy_args",
    "add_extraction_component_args",
    "ClassifierInferenceStrategy",
    "extract_inference_activation",
    "get_inference_score",
    "get_recommended_inference_strategy",
    "add_classifier_inference_strategy_args",
    "LayerActivations",
    # Optimal extraction (two-pass approach for maximum signal)
    "OptimalExtractionResult",
    "compute_signal_trajectory",
    "extract_at_optimal_position",
    "extract_at_max_diff_norm",  # Direction-free alternative
    "extract_batch_optimal",
    "compare_extraction_strategies",
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
