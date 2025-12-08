"""
Example custom evaluators for various use cases.

These examples demonstrate how to create custom evaluators for:
- AI detection (GPTZero, RoBERTa detector, etc.)
- Writing quality
- Style matching
- Custom API integrations
"""

from wisent.core.evaluators.custom.examples.gptzero import (
    GPTZeroEvaluator,
    create_gptzero_evaluator,
)
from wisent.core.evaluators.custom.examples.humanization import (
    HumanizationEvaluator,
    create_humanization_evaluator,
)
from wisent.core.evaluators.custom.examples.roberta_detector import (
    RobertaDetectorEvaluator,
    create_roberta_detector_evaluator,
)
from wisent.core.evaluators.custom.examples.desklib_detector import (
    DesklibDetectorEvaluator,
    create_desklib_detector_evaluator,
)

__all__ = [
    "GPTZeroEvaluator",
    "create_gptzero_evaluator",
    "HumanizationEvaluator",
    "create_humanization_evaluator",
    "RobertaDetectorEvaluator",
    "create_roberta_detector_evaluator",
    "DesklibDetectorEvaluator",
    "create_desklib_detector_evaluator",
]
