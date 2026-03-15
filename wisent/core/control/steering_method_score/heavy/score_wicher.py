"""Score WICHER (Whirlwind) predicted effectiveness.

WICHER uses iterative quasi-Newton (Broyden) updates in a low-dimensional
concept subspace. Works best with moderate geometry complexity where
adaptive iteration improves over single-shot methods.
Profile affinity: E_MARGINAL, F_NONLINEAR.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_wicher"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of WICHER from ZWIAD metrics."""
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    knn = metrics.get("knn_accuracy", SCORER_NEUTRAL_DEFAULT)
    linear = metrics.get("linear_probe_accuracy", SCORER_NEUTRAL_DEFAULT)
    curvature = metrics.get("manifold_curvature", SCORER_NEUTRAL_DEFAULT)
    complexity = (knn + curvature) / (SCORE_RANGE_MAX + SCORE_RANGE_MAX)
    threshold = _w("simplicity_threshold")
    simplicity = max(linear - threshold, SCORE_RANGE_MIN)
    score = (
        _w("w_signal") * signal
        + _w("w_complexity") * complexity
        + _w("w_knn") * knn
        - _w("w_simplicity_penalty") * simplicity
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
