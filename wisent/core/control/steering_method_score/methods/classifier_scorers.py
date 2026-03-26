"""Scorers for classifier-based steering methods: MLP, TETNO, GROM.

MLP uses a neural classifier: needs signal + nonlinear classification
capability (knn_accuracy).

TETNO uses gated dynamic steering: needs signal + concentrated Fisher
information (fisher_fisher_gini) for gate detection.

GROM uses manifold optimization: needs signal + nonlinear classifiability
+ actual curved geometry (manifold_curvature) to optimize over.
"""
from typing import Dict

from wisent.core.control.steering_method_score.scorer import metric_product

_REQUIRED_MLP = ["signal_strength", "knn_accuracy"]
_REQUIRED_TETNO = ["signal_strength", "fisher_fisher_gini"]
_REQUIRED_GROM = ["signal_strength", "knn_accuracy", "manifold_curvature"]


def get_score_mlp(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, knn_accuracy."""
    return metric_product(metrics, _REQUIRED_MLP)


def get_score_tetno(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, fisher_fisher_gini."""
    return metric_product(metrics, _REQUIRED_TETNO)


def get_score_grom(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, knn_accuracy, manifold_curvature."""
    return metric_product(metrics, _REQUIRED_GROM)
