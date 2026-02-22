"""Evaluation and model error classes."""
from __future__ import annotations

from typing import Optional

from wisent.core.errors._error_codes_enum import ErrorCode
from wisent.core.errors._error_codes_base import WisentError

# Evaluation Errors
# =============================================================================

class EvaluationError(WisentError):
    def __init__(self, task_name: str, response: Optional[str] = None, expected: Optional[str] = None, cause: Optional[Exception] = None):
        details = {"task": task_name}
        if response:
            details["response"] = response[:100] + "..." if len(response) > 100 else response
        if expected:
            details["expected"] = expected
        super().__init__(ErrorCode.EVALUATION_FAILED, details, cause, task_name=task_name)


class ExactMatchError(WisentError):
    def __init__(self, index: int, prediction: str, ground_truth: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.EXACT_MATCH_FAILED, {"prediction": prediction[:100], "ground_truth": ground_truth}, cause, index=index)


class BigCodeEvaluationError(WisentError):
    def __init__(self, task_name: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.BIGCODE_EVALUATION_FAILED, {"task": task_name}, cause, task_name=task_name)


class TaskNameRequiredError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.TASK_NAME_REQUIRED)


class NumericalExtractionError(WisentError):
    def __init__(self, response: str):
        super().__init__(ErrorCode.NUMERICAL_EXTRACTION_FAILED, {"response": response[:200]})


class TextExtractionError(WisentError):
    def __init__(self, response: str):
        super().__init__(ErrorCode.TEXT_EXTRACTION_FAILED, {"response": response[:200]})


class ExtractorNotFoundError(WisentError):
    def __init__(self, task_name: str):
        super().__init__(ErrorCode.EXTRACTOR_NOT_FOUND, {"task": task_name}, task_name=task_name)


class ExtractorReturnedNoneError(WisentError):
    def __init__(self, task_name: str):
        super().__init__(ErrorCode.EXTRACTOR_RETURNED_NONE, {"task": task_name}, task_name=task_name)


class BigCodeTaskRequiresFlagError(WisentError):
    def __init__(self, task_name: str):
        super().__init__(ErrorCode.BIGCODE_TASK_REQUIRES_FLAG, {"task": task_name}, task_name=task_name)


# =============================================================================
# Model/Chat Errors
# =============================================================================

class ChatTemplateNotAvailableError(WisentError):
    def __init__(self, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.CHAT_TEMPLATE_NOT_AVAILABLE, cause=cause)


class ChatTemplateError(WisentError):
    def __init__(self, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.CHAT_TEMPLATE_FAILED, cause=cause)


class EmptyGenerationError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.EMPTY_GENERATION)


class TokenizerMissingMethodError(WisentError):
    def __init__(self, method_name: str = "apply_chat_template"):
        super().__init__(ErrorCode.TOKENIZER_MISSING_METHOD, {"method": method_name}, method_name=method_name)


class ModelArchitectureUnknownError(WisentError):
    def __init__(self, model_type: Optional[str] = None):
        details = {"model_type": model_type} if model_type else {}
        super().__init__(ErrorCode.MODEL_ARCHITECTURE_UNKNOWN, details)


class DecoderLayersNotFoundError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.DECODER_LAYERS_NOT_FOUND)


class HiddenSizeNotFoundError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.HIDDEN_SIZE_NOT_FOUND)


class NoHiddenStatesError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_HIDDEN_STATES)


class LayerNotFoundError(WisentError):
    def __init__(self, layer_name: str):
        super().__init__(ErrorCode.LAYER_NOT_FOUND, {"layer": layer_name}, layer_name=layer_name)


class ModelConfigAccessError(WisentError):
    def __init__(self, model_name: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.MODEL_CONFIG_ACCESS_FAILED, {"model": model_name}, cause, model_name=model_name)


# =============================================================================
# Configuration/Validation Errors
