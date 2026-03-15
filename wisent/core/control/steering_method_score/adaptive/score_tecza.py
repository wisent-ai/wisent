"""Score TECZA (Rainbow) predicted effectiveness.

TECZA discovers multiple independent concept directions per layer.
Works best when activation space contains multiple distinct sub-concepts
with high concept coherence. Profile affinity: D_MULTICONCEPT.
"""
from typing import Dict

from wisent.core.utils.config_tools.constants import (
    SCORE_RANGE_MIN, SCORE_RANGE_MAX, SCORER_NEUTRAL_DEFAULT,
)
from wisent.core.control.steering_methods.configs.optimal import get_optimal

_METHOD_KEY = "scorer_tecza"


def _w(param: str) -> float:
    return get_optimal(param, method=_METHOD_KEY)


def get_score(metrics: Dict[str, float]) -> float:
    """Predict effectiveness of TECZA from ZWIAD metrics."""
    coherence = metrics.get("concept_coherence", SCORER_NEUTRAL_DEFAULT)
    icd_var = metrics.get("icd_top1_variance", SCORER_NEUTRAL_DEFAULT)
    n_concepts = metrics.get("n_concepts", SCORE_RANGE_MAX)
    concept_threshold = _w("concept_threshold")
    multi_concept_bonus = SCORE_RANGE_MAX if n_concepts > concept_threshold else SCORE_RANGE_MIN
    single_concept = SCORE_RANGE_MAX if n_concepts <= SCORE_RANGE_MAX else SCORE_RANGE_MIN
    score = (
        _w("w_coherence") * coherence
        + _w("w_icd_variance") * icd_var
        + _w("w_n_concepts") * multi_concept_bonus
        - _w("w_single_concept_penalty") * single_concept
    )
    return max(SCORE_RANGE_MIN, min(SCORE_RANGE_MAX, score))
