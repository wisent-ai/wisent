"""Configuration, validation, and agent error classes."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from wisent.core.errors._error_codes_enum import ErrorCode
from wisent.core.errors._error_codes_base import WisentError

# =============================================================================

class MissingParameterError(WisentError):
    def __init__(self, params: List[str], context: Optional[str] = None):
        details = {"params": params}
        if context:
            details["context"] = context
        super().__init__(ErrorCode.MISSING_PARAMETER, details, params=", ".join(params))


class InvalidChoicesError(WisentError):
    def __init__(self, task_name: str, reason: str, choices: Optional[List] = None):
        details = {"task": task_name, "reason": reason}
        if choices:
            details["choices"] = choices
        super().__init__(ErrorCode.INVALID_CHOICES, details, task_name=task_name, reason=reason)


class ModelNotProvidedError(WisentError):
    def __init__(self, context: Optional[str] = None):
        details = {"context": context} if context else {}
        super().__init__(ErrorCode.MODEL_NOT_PROVIDED, details)


class InvalidRangeError(WisentError):
    def __init__(self, param: str, value: Any, min_val: Any, max_val: Any):
        super().__init__(ErrorCode.INVALID_RANGE, {"param": param, "value": value}, min_val=min_val, max_val=max_val, value=value)


class InvalidValueError(WisentError):
    def __init__(self, param: str, reason: str, value: Optional[Any] = None):
        details = {"param": param, "reason": reason}
        if value is not None:
            details["value"] = value
        super().__init__(ErrorCode.INVALID_VALUE, details, param=param, reason=reason)


class DuplicateNameError(WisentError):
    def __init__(self, entity_type: str, name: str):
        super().__init__(ErrorCode.DUPLICATE_NAME, {"entity_type": entity_type, "name": name}, entity_type=entity_type, name=name)


class UnknownTypeError(WisentError):
    def __init__(self, entity_type: str, value: str, valid_values: Optional[List[str]] = None):
        details = {"entity_type": entity_type, "value": value}
        if valid_values:
            details["valid_values"] = valid_values
        super().__init__(ErrorCode.UNKNOWN_TYPE, details, entity_type=entity_type, value=value)


class InsufficientDataError(WisentError):
    def __init__(self, reason: str, required: Optional[int] = None, actual: Optional[int] = None):
        details = {"reason": reason}
        if required is not None:
            details["required"] = required
        if actual is not None:
            details["actual"] = actual
        super().__init__(ErrorCode.INSUFFICIENT_DATA, details, reason=reason)


class LayerRangeError(WisentError):
    def __init__(self, reason: str):
        super().__init__(ErrorCode.LAYER_RANGE_ERROR, {"reason": reason}, reason=reason)


# =============================================================================
# Agent/Diagnostic Errors
# =============================================================================

class ClassifierLoadError(WisentError):
    def __init__(self, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.CLASSIFIER_LOAD_FAILED, cause=cause)


class ClassifierCreationError(WisentError):
    def __init__(self, issue_type: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.CLASSIFIER_CREATION_FAILED, {"issue_type": issue_type}, cause, issue_type=issue_type)


class NoQualityScoresError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_QUALITY_SCORES)


class NoConfidenceScoresError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_CONFIDENCE_SCORES)


class ClassifierConfigRequiredError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.CLASSIFIER_CONFIG_REQUIRED)


class DeviceBenchmarkError(WisentError):
    def __init__(self, task_name: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.DEVICE_BENCHMARK_FAILED, {"task": task_name}, cause, task_name=task_name)


class BudgetCalculationError(WisentError):
    def __init__(self, task_type: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.BUDGET_CALCULATION_FAILED, {"task_type": task_type}, cause, task_type=task_type)


class ResourceEstimationError(WisentError):
    def __init__(self, resource_type: str, task_name: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.RESOURCE_ESTIMATION_FAILED, {"resource_type": resource_type, "task": task_name}, cause, resource_type=resource_type, task_name=task_name)


class NoBenchmarkDataError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_BENCHMARK_DATA)


class TrainingDataGenerationError(WisentError):
    def __init__(self, issues: List[str], cause: Optional[Exception] = None):
        super().__init__(ErrorCode.TRAINING_DATA_GENERATION_FAILED, {"issues": issues}, cause, issues=", ".join(issues))


class NoSuitableClassifierError(WisentError):
    def __init__(self, issue_type: str):
        super().__init__(ErrorCode.NO_SUITABLE_CLASSIFIER, {"issue_type": issue_type}, issue_type=issue_type)


class ImprovementMethodUnknownError(WisentError):
    def __init__(self, method: str, available_methods: Optional[List[str]] = None):
        details = {"method": method}
        if available_methods:
            details["available"] = available_methods
        super().__init__(ErrorCode.IMPROVEMENT_METHOD_UNKNOWN, details, method=method)


# =============================================================================
# Steering/Training Errors
# =============================================================================
