"""
Steering method recommendation based on Zwiad metrics.

Delegates to the configurable recommendation system which scores all 9
registered steering methods. Weights are either learned (from a saved
config) or default (zeros for untuned methods).
"""

from typing import Dict, Any, Optional

from wisent.core.constants import MIN_CONCEPT_PAIRS
from .recommendation.config import (
    RecommendationConfig, Thresholds as SteeringThresholds)
from .recommendation.configurable import compute_configurable_recommendation


_LEARNED_CONFIG_PATH = "~/.wisent/learned_recommendation_config.json"


def compute_steering_recommendation(
    metrics: Dict[str, Any],
    thresholds=None,
) -> Dict[str, Any]:
    """Compute steering method recommendation from Zwiad metrics.

    Uses learned config if available, otherwise default config.
    The thresholds parameter is accepted for backward compatibility
    but ignored (use RecommendationConfig instead).
    """
    from pathlib import Path
    learned = Path(_LEARNED_CONFIG_PATH).expanduser()
    if learned.exists():
        cfg = RecommendationConfig.load(learned)
    else:
        cfg = RecommendationConfig.default()
    return compute_configurable_recommendation(metrics, cfg)


def compute_per_layer_recommendation(
    per_layer_metrics: Dict[int, Dict[str, Any]],
    thresholds=None,
) -> Dict[str, Any]:
    """Compute recommendations for each layer and identify best layer."""
    if not per_layer_metrics:
        return {"error": "no layer metrics provided"}
    per_layer = {}
    layer_scores = {}
    for layer, metrics in per_layer_metrics.items():
        rec = compute_steering_recommendation(metrics)
        per_layer[layer] = rec
        icd = rec["raw_signals"].get("icd")
        stability = rec["raw_signals"].get("stability")
        alignment = rec["raw_signals"].get("alignment")
        score = 0.0
        if icd is not None:
            score += icd * 2.0
        if stability is not None:
            score += stability
        if alignment is not None:
            score += alignment
        layer_scores[layer] = score
    best_layer = max(layer_scores, key=layer_scores.get)
    return {
        "per_layer": per_layer, "layer_scores": layer_scores,
        "best_layer": best_layer, "best_layer_recommendation": per_layer[best_layer],
    }


def get_method_description(method: str) -> str:
    """Get description of a steering method."""
    from wisent.core.steering_methods.registry import SteeringMethodRegistry
    name = method.lower().replace(" ", "_")
    if SteeringMethodRegistry.validate_method(name):
        return SteeringMethodRegistry.get(name).description
    return "Unknown method"


def get_method_requirements(method: str) -> Dict[str, Any]:
    """Get requirements/assumptions for a steering method."""
    from wisent.core.steering_methods.registry import SteeringMethodRegistry
    name = method.lower().replace(" ", "_")
    if not SteeringMethodRegistry.validate_method(name):
        return {}
    defn = SteeringMethodRegistry.get(name)
    return {
        "min_pairs": defn.optimization_config.get("min_pairs", MIN_CONCEPT_PAIRS),
        "default_strength": defn.default_strength,
        "strength_range": defn.strength_range,
        "parameters": [p.name for p in defn.parameters],
    }
