"""Score SZLAK (Trail) predicted effectiveness.

SZLAK computes optimal transport plans between positive and negative
activation distributions in attention space using entropic OT and
Sinkhorn. Works best when activation space is editable with
concentrated spectral structure.
Profile affinity: D_MULTICONCEPT, E_MARGINAL.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_szlak"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of SZLAK from ZWIAD metrics."""
    editability = metrics.get("composite_editability", SCORER_NEUTRAL_DEFAULT)
    spectral = metrics.get("spectral_concentration", SCORER_NEUTRAL_DEFAULT)
    signal = metrics.get("signal_strength", SCORER_NEUTRAL_DEFAULT)
    threshold = _w("degenerate_signal_threshold")
    degenerate = max(threshold - signal, SCORE_RANGE_MIN)
    score = (
        _w("w_editability") * editability
        + _w("w_spectral_concentration") * spectral
        + _w("w_signal") * signal
        - _w("w_degenerate_penalty") * degenerate
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
