"""Score PRZELOM (Breakthrough) predicted effectiveness.

PRZELOM inverts the entropic optimal transport cost using Q-projections
for adaptive transport in attention space. Similar to SZLAK but favors
spectral sharpness over concentration.
Profile affinity: D_MULTICONCEPT, E_MARGINAL.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_przelom"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of PRZELOM from ZWIAD metrics."""
    editability = metrics.get("composite_editability", SCORER_NEUTRAL_DEFAULT)
    sharpness = metrics.get("spectral_sharpness", SCORER_NEUTRAL_DEFAULT)
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    threshold = _w("degenerate_signal_threshold")
    degenerate = max(threshold - signal, SCORE_RANGE_MIN)
    score = (
        _w("w_editability") * editability
        + _w("w_spectral_sharpness") * sharpness
        + _w("w_signal") * signal
        - _w("w_degenerate_penalty") * degenerate
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
