"""Score MLP (Neural Classifier Gradient) predicted effectiveness.

MLP trains a small neural network to find nonlinear decision boundaries.
Works best when there is a gap between KNN and linear probe accuracy,
indicating nonlinear structure that linear methods miss.
Profile affinity: E_MARGINAL, F_NONLINEAR.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_mlp"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of MLP from ZWIAD metrics."""
    knn = metrics.get("knn_accuracy", SCORER_NEUTRAL_DEFAULT)
    linear = metrics.get("linear_probe_accuracy", SCORER_NEUTRAL_DEFAULT)
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    nonlinear_gap = max(knn - linear, SCORE_RANGE_MIN)
    ceiling = _w("linearity_ceiling")
    ceiling_scale = _w("ceiling_scale")
    linearity_excess = max(linear - ceiling, SCORE_RANGE_MIN) * ceiling_scale
    score = (
        _w("w_nonlinear_gap") * nonlinear_gap
        + _w("w_signal") * signal
        + _w("w_knn") * knn
        - _w("w_linearity_penalty") * linearity_excess
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
