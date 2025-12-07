"""
Utility functions for parser arguments.

Shared helper functions used across multiple command parsers.
"""

from typing import List, Optional

from wisent.core.errors import ModelNotProvidedError, InvalidValueError


def parse_layers_from_arg(layer_arg: str, model=None) -> List[int]:
    """
    Parse layer argument into list of integers.

    Args:
        layer_arg: String like "15", "14-16", "14,15,16", or "-1" (for auto-optimization)
        model: Model object (needed for determining available layers)

    Returns:
        List of layer indices
    """
    # Handle special cases
    if layer_arg == "-1":
        # Signal for auto-optimization - return single layer list
        return [-1]

    # Use existing parse_layer_range logic
    layers = parse_layer_range(layer_arg, model)
    if layers is None:
        # "all" case - auto-detect model layers
        if model is not None:
            from wisent.core.hyperparameter_optimizer import detect_model_layers

            total_layers = detect_model_layers(model)
            return list(range(total_layers))
        # If no model provided, we cannot determine layers - this should not happen
        raise ModelNotProvidedError()

    return layers


def parse_layer_range(layer_range_str: str, model=None) -> Optional[List[int]]:
    """
    Parse layer range string into list of integers.

    Args:
        layer_range_str: String like "8-24", "10,15,20", or "all"
        model: Model object (needed for "all" option)

    Returns:
        List of layer indices, or None if "all" (will be auto-detected later)
    """
    if layer_range_str.lower() == "all":
        # Return None to signal auto-detection
        return None
    if "-" in layer_range_str:
        # Range format: "8-24"
        start, end = map(int, layer_range_str.split("-"))
        return list(range(start, end + 1))
    if "," in layer_range_str:
        # Comma-separated format: "10,15,20"
        return [int(x.strip()) for x in layer_range_str.split(",")]
    # Single layer
    return [int(layer_range_str)]


def aggregate_token_scores(token_scores: List[float], method: str) -> float:
    """
    Aggregate token scores using the specified method.

    Args:
        token_scores: List of token scores (probabilities)
        method: Aggregation method ("average", "final", "first", "max", "min")

    Returns:
        Aggregated score
    """
    if not token_scores:
        return 0.5

    # Convert any tensor values to floats and filter out None values
    clean_scores = []
    for i, score in enumerate(token_scores):
        if score is None:
            raise InvalidValueError(param_name=f"token_score[{i}]", actual=None, expected="float value")
        if hasattr(score, "item"):  # Handle tensors
            raise InvalidValueError(param_name=f"token_score[{i}]", actual=str(type(score)), expected="float, got tensor")
        if not isinstance(score, (int, float)):
            raise InvalidValueError(param_name=f"token_score[{i}]", actual=type(score).__name__, expected="float")
        clean_scores.append(float(score))

    if not clean_scores:
        return 0.5

    if method == "average":
        return sum(clean_scores) / len(clean_scores)
    if method == "final":
        return clean_scores[-1]
    if method == "first":
        return clean_scores[0]
    if method == "max":
        return max(clean_scores)
    if method == "min":
        return min(clean_scores)
    # Default to average if unknown method
    return sum(clean_scores) / len(clean_scores)
