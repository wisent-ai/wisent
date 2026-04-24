"""Personalization evaluation sub-modules.

Text-quality heuristics (evaluate_quality, _is_gibberish, _is_incoherent)
live in wisent.core.reading.evaluators.core.text_quality — they are not
specific to personalization and are shared across all generation evaluators.
"""

from .alignment import evaluate_alignment, estimate_alignment
from .difference import evaluate_difference

__all__ = [
    "evaluate_alignment",
    "estimate_alignment",
    "evaluate_difference",
]
