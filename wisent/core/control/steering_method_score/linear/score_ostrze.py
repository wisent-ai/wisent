"""Score OSTRZE (Blade) predicted effectiveness.

OSTRZE finds steering directions via regularized logistic regression.
Works best with clear orthogonal decision boundaries and strong
class separation. Profile affinity: A_PRISTINE, B_STRONG.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
    SCORER_CLOUD_SEP_NORMALIZER,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_ostrze"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of OSTRZE from ZWIAD metrics."""
    linear = metrics.get("linear_probe_accuracy", SCORER_NEUTRAL_DEFAULT)
    cloud_sep_raw = metrics.get("cloud_separation_ratio", SCORER_NEUTRAL_DEFAULT)
    cloud_sep = min(cloud_sep_raw / SCORER_CLOUD_SEP_NORMALIZER, SCORE_RANGE_MAX)
    fisher = metrics.get("fisher_fisher_gini", SCORER_NEUTRAL_DEFAULT)
    curvature = metrics.get("manifold_curvature", SCORER_NEUTRAL_DEFAULT)
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    score = (
        _w("w_linear_probe") * linear
        + _w("w_cloud_separation") * cloud_sep
        + _w("w_fisher_gini") * fisher
        + _w("w_signal") * signal
        - _w("w_curvature_penalty") * curvature
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
