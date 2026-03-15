"""Predict steering method effectiveness from ZWIAD metrics.

Scores each steering method's predicted effectiveness using ZWIAD
geometry metrics. Used by select_intervention() and profile_benchmark()
to rank methods before expensive steering runs.
"""
from wisent.core.control.steering_method_score.scorer import rank_methods
from wisent.core.control.steering_method_score.scorer import flatten_zwiad_results

from wisent.core.control.steering_method_score.linear import (
    score_caa,
    score_ostrze,
    score_mlp,
)
from wisent.core.control.steering_method_score.adaptive import (
    score_tecza,
    score_tetno,
    score_nurt,
)
from wisent.core.control.steering_method_score.heavy import (
    score_grom,
    score_wicher,
    score_szlak,
    score_przelom,
)

ALL_SCORERS = {
    "caa": score_caa,
    "ostrze": score_ostrze,
    "mlp": score_mlp,
    "tecza": score_tecza,
    "tetno": score_tetno,
    "nurt": score_nurt,
    "grom": score_grom,
    "wicher": score_wicher,
    "szlak": score_szlak,
    "przelom": score_przelom,
}
