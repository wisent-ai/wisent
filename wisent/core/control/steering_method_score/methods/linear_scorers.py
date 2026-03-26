"""Scorers for linear steering methods: CAA and OSTRZE.

CAA uses diff-of-means: needs signal, linear separability, AND coherent
direction (if directions vary, the mean cancels out).

OSTRZE uses logistic regression: needs signal + linear separability.
Works despite direction diversity because the classifier finds a hyperplane.
"""
from typing import Dict

from wisent.core.control.steering_method_score.scorer import metric_product

_REQUIRED_CAA = [
    "signal_strength", "linear_probe_accuracy", "concept_coherence",
]
_REQUIRED_OSTRZE = ["signal_strength", "linear_probe_accuracy"]


def get_score_caa(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, linear_probe_accuracy, concept_coherence."""
    return metric_product(metrics, _REQUIRED_CAA)


def get_score_ostrze(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, linear_probe_accuracy."""
    return metric_product(metrics, _REQUIRED_OSTRZE)
