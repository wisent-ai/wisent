"""
Example custom evaluators for various use cases.

These examples demonstrate how to create custom evaluators for:
- AI detection (GPTZero, Originality.ai, etc.)
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

__all__ = [
    "GPTZeroEvaluator",
    "create_gptzero_evaluator",
    "HumanizationEvaluator",
    "create_humanization_evaluator",
]
