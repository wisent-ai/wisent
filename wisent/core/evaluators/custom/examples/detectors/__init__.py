"""AI detection evaluators."""

from .gptzero import (
    GPTZeroEvaluator,
    create_gptzero_evaluator,
)
from .roberta_detector import (
    RobertaDetectorEvaluator,
    create_roberta_detector_evaluator,
)
from .desklib_detector import (
    DesklibDetectorEvaluator,
    create_desklib_detector_evaluator,
)

__all__ = [
    "GPTZeroEvaluator",
    "create_gptzero_evaluator",
    "RobertaDetectorEvaluator",
    "create_roberta_detector_evaluator",
    "DesklibDetectorEvaluator",
    "create_desklib_detector_evaluator",
]
