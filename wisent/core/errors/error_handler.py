"""Comprehensive error handling for Wisent.

This module provides informative error classes and utilities for proper error handling
throughout the codebase. NO FALLBACKS - errors should be raised immediately with
detailed information about what went wrong and how to fix it.
"""

import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


class WisentError(Exception):
    """Base exception for all Wisent errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            details_str = "\n".join(f"  - {k}: {v}" for k, v in self.details.items())
            return f"{self.message}\nDetails:\n{details_str}"
        return self.message


class EvaluationError(WisentError):
    """Raised when evaluation fails."""
    pass


class MissingParameterError(EvaluationError):
    """Raised when required parameters are missing for evaluation."""

    def __init__(self, missing_params: list, evaluator_name: str, task_name: Optional[str] = None):
        message = f"Evaluator '{evaluator_name}' requires missing parameters: {', '.join(missing_params)}"
        details = {
            "evaluator": evaluator_name,
            "missing_parameters": missing_params,
            "task": task_name or "unknown"
        }
        super().__init__(message, details)


class InvalidChoicesError(EvaluationError):
    """Raised when choices are invalid or missing for multiple choice evaluation."""

    def __init__(self, reason: str, task_name: str, choices: Optional[list] = None):
        message = f"Invalid choices for task '{task_name}': {reason}"
        details = {
            "task": task_name,
            "reason": reason,
            "choices_provided": choices
        }
        super().__init__(message, details)


class ModelNotProvidedError(EvaluationError):
    """Raised when model is required but not provided."""

    def __init__(self, evaluator_name: str, task_name: str):
        message = (
            f"Evaluator '{evaluator_name}' requires a model for log likelihood computation, "
            f"but none was provided for task '{task_name}'. "
            f"Pass model=<WisentModel> in kwargs to evaluate()."
        )
        details = {
            "evaluator": evaluator_name,
            "task": task_name,
            "solution": "Pass model parameter in kwargs"
        }
        super().__init__(message, details)


def require_all_parameters(params: Dict[str, Any], context: str, task_name: Optional[str] = None):
    """Raise error if any required parameters are None or missing.

    Args:
        params: Dict of parameter_name -> value
        context: Context where parameters are required
        task_name: Optional task name for better error messages

    Raises:
        MissingParameterError: If any parameters are None
    """
    missing = [name for name, value in params.items() if value is None]
    if missing:
        raise MissingParameterError(
            missing_params=missing,
            evaluator_name=context,
            task_name=task_name
        )


def validate_choices(choices: Optional[list], task_name: str, min_choices: int = 2):
    """Validate that choices are provided and valid.

    Args:
        choices: List of answer choices
        task_name: Name of the task
        min_choices: Minimum number of choices required

    Raises:
        InvalidChoicesError: If choices are invalid
    """
    if choices is None:
        raise InvalidChoicesError(
            reason="No choices provided",
            task_name=task_name,
            choices=None
        )

    if not isinstance(choices, list):
        raise InvalidChoicesError(
            reason=f"Choices must be a list, got {type(choices).__name__}",
            task_name=task_name,
            choices=choices
        )

    if len(choices) < min_choices:
        raise InvalidChoicesError(
            reason=f"Need at least {min_choices} choices, got {len(choices)}",
            task_name=task_name,
            choices=choices
        )

    if any(not isinstance(c, str) or not c.strip() for c in choices):
        raise InvalidChoicesError(
            reason="All choices must be non-empty strings",
            task_name=task_name,
            choices=choices
        )
