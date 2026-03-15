"""Score NURT (Current) predicted effectiveness.

NURT projects activations into a learned concept subspace and applies
flow matching for on-manifold transport between positive and negative
representations. Profile affinity: C_SOLID, D_MULTICONCEPT.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_nurt"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def _moderate_score(value: float, center: float, width: float) -> float:
    """Score how close value is to a moderate target."""
    if width <= SCORE_RANGE_MIN:
        return SCORE_RANGE_MIN
    return max(SCORE_RANGE_MIN, SCORE_RANGE_MAX - abs(value - center) / width)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of NURT from ZWIAD metrics."""
    coherence = metrics.get("concept_coherence", SCORER_NEUTRAL_DEFAULT)
    curvature = metrics.get("manifold_curvature", SCORER_NEUTRAL_DEFAULT)
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    center = _w("curvature_center")
    width = _w("curvature_width")
    moderate_curv = _moderate_score(curvature, center, width)
    extreme_threshold = _w("extreme_curvature_threshold")
    extreme_penalty = max(curvature - extreme_threshold, SCORE_RANGE_MIN)
    score = (
        _w("w_coherence") * coherence
        + _w("w_curvature_moderate") * moderate_curv
        + _w("w_signal") * signal
        - _w("w_extreme_curvature_penalty") * extreme_penalty
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
