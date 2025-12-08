"""
Custom evaluator module for user-defined evaluation functions.

Allows users to pass custom evaluator functions (like API calls to GPTZero,
Claude, or any external service) for optimization objectives.
"""

from wisent.core.evaluators.custom.custom_evaluator import (
    CustomEvaluator,
    CallableEvaluator,
    APIEvaluator,
    create_custom_evaluator,
)

__all__ = [
    "CustomEvaluator",
    "CallableEvaluator", 
    "APIEvaluator",
    "create_custom_evaluator",
]
