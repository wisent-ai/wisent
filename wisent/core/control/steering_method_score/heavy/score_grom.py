"""Score GROM (Thunder) predicted effectiveness.

GROM jointly optimizes steering directions, per-token gating networks,
and intensity scaling on curved manifolds. Works best with high
manifold curvature and nonlinear geometry.
Profile affinity: F_NONLINEAR, H_DEGENERATE.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_grom"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of GROM from ZWIAD metrics."""
    curvature = metrics.get("manifold_curvature", SCORER_NEUTRAL_DEFAULT)
    knn = metrics.get("knn_accuracy", SCORER_NEUTRAL_DEFAULT)
    linear = metrics.get("linear_probe_accuracy", SCORER_NEUTRAL_DEFAULT)
    nonlinear_gap = max(knn - linear, SCORE_RANGE_MIN)
    threshold = _w("linearity_threshold")
    linearity_penalty = max(linear - threshold, SCORE_RANGE_MIN)
    score = (
        _w("w_curvature") * curvature
        + _w("w_knn") * knn
        + _w("w_nonlinear_gap") * nonlinear_gap
        - _w("w_linearity_penalty") * linearity_penalty
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
