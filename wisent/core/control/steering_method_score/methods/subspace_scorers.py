"""Scorers for subspace-based steering methods: TECZA, NURT, WICHER.

TECZA discovers multiple steering directions: needs signal + dispersed
concept structure. The complement of concept_coherence measures variance
NOT in top SVD direction.

NURT uses flow matching in SVD subspace: needs signal + coherent concept
(SVD captures it well). Opposite of TECZA.

WICHER uses Broyden iteration in SVD subspace: needs signal + dominant
ICD component (icd_top1_variance) for stable subspace.
"""
from typing import Dict

from wisent.core.control.steering_method_score.scorer import metric_product
from wisent.core.utils.config_tools.constants import SCORE_RANGE_MAX, SCORE_RANGE_MIN

_REQUIRED_NURT = ["signal_strength", "concept_coherence"]
_REQUIRED_WICHER = ["signal_strength", "icd_top1_variance"]


def get_score_tecza(metrics: Dict[str, float]) -> float:
    """Product of signal_strength and complement of concept_coherence."""
    signal = metrics.get("signal_strength")
    coherence = metrics.get("concept_coherence")
    if signal is None or coherence is None:
        return SCORE_RANGE_MIN
    return signal * (SCORE_RANGE_MAX - coherence)


def get_score_nurt(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, concept_coherence."""
    return metric_product(metrics, _REQUIRED_NURT)


def get_score_wicher(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, icd_top1_variance."""
    return metric_product(metrics, _REQUIRED_WICHER)
