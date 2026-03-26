"""Scorers for transport-based steering methods: SZLAK, PRZELOM.

SZLAK uses entropic optimal transport: needs signal + editable activation
space (composite_editability) because transport requires that steering
modifications survive through the softmax bottleneck.

PRZELOM uses Q-projection transport: same prerequisites as SZLAK.
"""
from typing import Dict

from wisent.core.control.steering_method_score.scorer import metric_product

_REQUIRED_SZLAK = ["signal_strength", "composite_editability"]
_REQUIRED_PRZELOM = ["signal_strength", "composite_editability"]


def get_score_szlak(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, composite_editability."""
    return metric_product(metrics, _REQUIRED_SZLAK)


def get_score_przelom(metrics: Dict[str, float]) -> float:
    """Product of signal_strength, composite_editability."""
    return metric_product(metrics, _REQUIRED_PRZELOM)
