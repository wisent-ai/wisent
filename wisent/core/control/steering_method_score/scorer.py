"""Shared scoring logic for steering method prediction from ZWIAD metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

from wisent.core.utils.config_tools.constants import (
    EXIT_CODE_ERROR,
    SCORE_RANGE_MAX,
    SCORE_RANGE_MIN,
)


def metric_product(metrics: Dict[str, float], required: List[str]) -> float:
    """Product of required metrics. Returns SCORE_RANGE_MIN if any missing."""
    score = SCORE_RANGE_MAX
    for key in required:
        value = metrics.get(key)
        if value is None:
            return SCORE_RANGE_MIN
        score *= float(value)
    return score


def rank_methods(metrics: Dict[str, float]) -> List[Tuple[str, float]]:
    """Score all methods, return sorted (method_name, score) descending."""
    from wisent.core.control.steering_method_score.methods.linear_scorers import (
        get_score_caa, get_score_ostrze,
    )
    from wisent.core.control.steering_method_score.methods.classifier_scorers import (
        get_score_mlp, get_score_tetno, get_score_grom,
    )
    from wisent.core.control.steering_method_score.methods.subspace_scorers import (
        get_score_tecza, get_score_nurt, get_score_wicher,
    )
    from wisent.core.control.steering_method_score.methods.transport_scorers import (
        get_score_szlak, get_score_przelom,
    )

    scorers = {
        "caa": get_score_caa,
        "ostrze": get_score_ostrze,
        "mlp": get_score_mlp,
        "tecza": get_score_tecza,
        "tetno": get_score_tetno,
        "nurt": get_score_nurt,
        "grom": get_score_grom,
        "wicher": get_score_wicher,
        "szlak": get_score_szlak,
        "przelom": get_score_przelom,
    }

    scored = [(name, fn(metrics)) for name, fn in scorers.items()]
    scored.sort(key=lambda pair: pair[-EXIT_CODE_ERROR], reverse=True)
    return scored
