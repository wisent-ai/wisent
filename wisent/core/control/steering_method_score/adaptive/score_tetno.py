"""Score TETNO (Heartbeat) predicted effectiveness.

TETNO uses gated steering with adaptive intensity based on model
uncertainty. Works best with moderate signal and few concentrated
directions. Profile affinity: E_MARGINAL.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_tetno"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def _moderate_score(value: float, center: float, width: float) -> float:
    """Score how close value is to a moderate target."""
    if width <= SCORE_RANGE_MIN:
        return SCORE_RANGE_MIN
    return max(SCORE_RANGE_MIN, SCORE_RANGE_MAX - abs(value - center) / width)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of TETNO from ZWIAD metrics."""
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    icd_var = metrics.get("icd_top1_variance", SCORER_NEUTRAL_DEFAULT)
    coherence = metrics.get("concept_coherence", SCORER_NEUTRAL_DEFAULT)
    linear = metrics.get("linear_probe_accuracy", SCORER_NEUTRAL_DEFAULT)
    center = _w("moderate_center")
    width = _w("moderate_width")
    moderate_var = _moderate_score(icd_var, center, width)
    low_coherence = SCORE_RANGE_MAX - coherence
    ceiling = _w("linearity_ceiling")
    linearity_excess = max(linear - ceiling, SCORE_RANGE_MIN)
    score = (
        _w("w_signal") * signal
        + _w("w_moderate_variance") * moderate_var
        + _w("w_low_coherence") * low_coherence
        - _w("w_linearity_penalty") * linearity_excess
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
