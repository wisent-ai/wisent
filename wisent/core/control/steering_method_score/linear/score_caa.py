"""Score CAA (Contrastive Activation Addition) predicted effectiveness.

CAA uses difference-in-means to find steering directions. Works best
when activation geometry is linearly separable with strong signal.
Profile affinity: A_PRISTINE, B_STRONG, C_SOLID, G_WEAK.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
    SCORER_CLOUD_SEP_NORMALIZER,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_caa"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of CAA from ZWIAD metrics."""
    linear = metrics.get("linear_probe_accuracy", SCORER_NEUTRAL_DEFAULT)
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    cloud_sep_raw = metrics.get("cloud_separation_ratio", SCORER_NEUTRAL_DEFAULT)
    cloud_sep = min(cloud_sep_raw / SCORER_CLOUD_SEP_NORMALIZER, SCORE_RANGE_MAX)
    curvature = metrics.get("manifold_curvature", SCORER_NEUTRAL_DEFAULT)
    fragmented = metrics.get("is_fragmented", SCORE_RANGE_MIN)
    score = (
        _w("w_linear_probe") * linear
        + _w("w_signal") * signal
        + _w("w_cloud_separation") * cloud_sep
        - _w("w_curvature_penalty") * curvature
        - _w("w_fragmentation_penalty") * fragmented
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
