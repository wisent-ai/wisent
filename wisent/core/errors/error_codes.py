"""Centralized error codes and messages for Wisent.

This module defines all error codes, messages, and custom exceptions used throughout
the codebase. NO FALLBACKS - errors should be raised immediately with clear codes.
"""

from enum import Enum
from typing import Optional, Any, Dict


class ErrorCode(str, Enum):
    """Centralized error codes for all Wisent errors."""
    
    # Data Loading Errors (1xxx)
    TASK_LOAD_FAILED = "E1001"
    TASK_NOT_FOUND = "E1002"
    FALLBACK_NOT_PERMITTED = "E1003"
    NO_DOCS_AVAILABLE = "E1004"
    DATASET_LOAD_FAILED = "E1005"
    VERSION_VALIDATION_FAILED = "E1006"
    VERSION_INFO_FAILED = "E1007"
    BENCHMARK_LOAD_FAILED = "E1008"
    
    # Evaluation Errors (2xxx)
    EVALUATION_FAILED = "E2001"
    EXACT_MATCH_FAILED = "E2002"
    BIGCODE_EVALUATION_FAILED = "E2003"
    TASK_NAME_REQUIRED = "E2004"
    NUMERICAL_EXTRACTION_FAILED = "E2005"
    TEXT_EXTRACTION_FAILED = "E2006"
    EXTRACTOR_NOT_FOUND = "E2007"
    
    # Model/Chat Errors (3xxx)
    CHAT_TEMPLATE_NOT_AVAILABLE = "E3001"
    CHAT_TEMPLATE_FAILED = "E3002"
    EMPTY_GENERATION = "E3003"
    TOKENIZER_MISSING_METHOD = "E3004"
    
    # Configuration Errors (4xxx)
    MISSING_PARAMETER = "E4001"
    INVALID_CHOICES = "E4002"
    MODEL_NOT_PROVIDED = "E4003"


ERROR_MESSAGES: Dict[ErrorCode, str] = {
    # Data Loading Errors
    ErrorCode.TASK_LOAD_FAILED: "Failed to load task '{task_name}'. All loading approaches exhausted.",
    ErrorCode.TASK_NOT_FOUND: "Task '{task_name}' not found in lm-evaluation-harness.",
    ErrorCode.FALLBACK_NOT_PERMITTED: "Fallback loading is not permitted for task '{task_name}'. Task must be loaded through proper channels.",
    ErrorCode.NO_DOCS_AVAILABLE: "No labeled documents available for task '{task_name}'. Task does not expose docs from validation/test/train/fewshot.",
    ErrorCode.DATASET_LOAD_FAILED: "Fallback dataset loading is not permitted for task '{task_name}'. Task must provide docs through standard lm-evaluation-harness methods.",
    ErrorCode.VERSION_VALIDATION_FAILED: "Failed to validate release version '{version}'. Could not load available versions from data loader.",
    ErrorCode.VERSION_INFO_FAILED: "Failed to get version info for '{version}'. Data loader error.",
    ErrorCode.BENCHMARK_LOAD_FAILED: "Failed to load benchmark '{benchmark_name}'.",
    
    # Evaluation Errors
    ErrorCode.EVALUATION_FAILED: "LMEvalHarnessGroundTruth evaluation failed for task '{task_name}'.",
    ErrorCode.EXACT_MATCH_FAILED: "LMEvalHarnessGroundTruth evaluation failed for prediction {index}.",
    ErrorCode.BIGCODE_EVALUATION_FAILED: "BigCode evaluation failed for coding task '{task_name}'. Ensure task documents are provided and properly formatted.",
    ErrorCode.TASK_NAME_REQUIRED: "task_name is required for benchmark evaluation. Provide the task name to enable proper LMEvalHarnessGroundTruth evaluation.",
    ErrorCode.NUMERICAL_EXTRACTION_FAILED: "Could not extract numerical answer from response. Response must contain an explicit answer marker.",
    ErrorCode.TEXT_EXTRACTION_FAILED: "Could not extract text answer from response. Response must contain an explicit answer marker.",
    ErrorCode.EXTRACTOR_NOT_FOUND: "No extractor registered for task '{task_name}'. Register an extractor in _EXTRACTOR_REGISTRY or use a task name that matches known patterns.",
    
    # Model/Chat Errors
    ErrorCode.CHAT_TEMPLATE_NOT_AVAILABLE: "Chat template is required but not available. Use a model/tokenizer that supports chat templates.",
    ErrorCode.CHAT_TEMPLATE_FAILED: "Failed to apply chat template. Ensure the tokenizer has a valid chat template configured.",
    ErrorCode.EMPTY_GENERATION: "Empty generation detected. Model produced no output tokens. Check model configuration, input formatting, and generation parameters.",
    ErrorCode.TOKENIZER_MISSING_METHOD: "Tokenizer does not have apply_chat_template method. Use a tokenizer that supports chat templates.",
    
    # Configuration Errors
    ErrorCode.MISSING_PARAMETER: "Required parameters missing: {params}.",
    ErrorCode.INVALID_CHOICES: "Invalid choices for task '{task_name}': {reason}.",
    ErrorCode.MODEL_NOT_PROVIDED: "Model required for evaluation but not provided.",
}


class WisentError(Exception):
    """Base exception for all Wisent errors with error code support."""

    def __init__(
        self,
        code: ErrorCode,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        **format_args
    ):
        self.code = code
        self.details = details or {}
        self.cause = cause
        
        # Format the message with provided arguments
        message_template = ERROR_MESSAGES.get(code, f"Unknown error: {code}")
        try:
            self.message = message_template.format(**format_args)
        except KeyError:
            self.message = message_template
        
        # Include cause in message if provided
        if cause:
            self.message = f"{self.message} Cause: {cause}"
        
        super().__init__(self.message)

    def __str__(self):
        result = f"[{self.code.value}] {self.message}"
        if self.details:
            details_str = "\n".join(f"  - {k}: {v}" for k, v in self.details.items())
            result = f"{result}\nDetails:\n{details_str}"
        return result


# Data Loading Errors
class TaskLoadError(WisentError):
    """Raised when task loading fails."""
    
    def __init__(self, task_name: str, cause: Optional[Exception] = None):
        super().__init__(
            code=ErrorCode.TASK_LOAD_FAILED,
            details={"task": task_name},
            cause=cause,
            task_name=task_name
        )


class FallbackNotPermittedError(WisentError):
    """Raised when fallback loading is attempted but not permitted."""
    
    def __init__(self, task_name: str):
        super().__init__(
            code=ErrorCode.FALLBACK_NOT_PERMITTED,
            details={"task": task_name},
            task_name=task_name
        )


class NoDocsAvailableError(WisentError):
    """Raised when no documents are available for a task."""
    
    def __init__(self, task_name: str):
        super().__init__(
            code=ErrorCode.NO_DOCS_AVAILABLE,
            details={"task": task_name},
            task_name=task_name
        )


class DatasetLoadError(WisentError):
    """Raised when dataset fallback loading is attempted."""
    
    def __init__(self, task_name: str):
        super().__init__(
            code=ErrorCode.DATASET_LOAD_FAILED,
            details={"task": task_name},
            task_name=task_name
        )


class VersionValidationError(WisentError):
    """Raised when version validation fails."""
    
    def __init__(self, version: str, cause: Optional[Exception] = None):
        super().__init__(
            code=ErrorCode.VERSION_VALIDATION_FAILED,
            details={"version": version},
            cause=cause,
            version=version
        )


class VersionInfoError(WisentError):
    """Raised when version info retrieval fails."""
    
    def __init__(self, version: str, cause: Optional[Exception] = None):
        super().__init__(
            code=ErrorCode.VERSION_INFO_FAILED,
            details={"version": version},
            cause=cause,
            version=version
        )


class BenchmarkLoadError(WisentError):
    """Raised when benchmark loading fails."""
    
    def __init__(self, benchmark_name: str, cause: Optional[Exception] = None):
        super().__init__(
            code=ErrorCode.BENCHMARK_LOAD_FAILED,
            details={"benchmark": benchmark_name},
            cause=cause,
            benchmark_name=benchmark_name
        )


# Evaluation Errors
class EvaluationError(WisentError):
    """Raised when evaluation fails."""
    
    def __init__(
        self,
        task_name: str,
        response: Optional[str] = None,
        expected: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        details = {"task": task_name}
        if response:
            details["response"] = response[:100] + "..." if len(response) > 100 else response
        if expected:
            details["expected"] = expected
        
        super().__init__(
            code=ErrorCode.EVALUATION_FAILED,
            details=details,
            cause=cause,
            task_name=task_name
        )


class ExactMatchError(WisentError):
    """Raised when exact match evaluation fails for a specific prediction."""
    
    def __init__(
        self,
        index: int,
        prediction: str,
        ground_truth: str,
        cause: Optional[Exception] = None
    ):
        super().__init__(
            code=ErrorCode.EXACT_MATCH_FAILED,
            details={
                "prediction": prediction[:100] + "..." if len(prediction) > 100 else prediction,
                "ground_truth": ground_truth
            },
            cause=cause,
            index=index
        )


class BigCodeEvaluationError(WisentError):
    """Raised when BigCode evaluation fails."""
    
    def __init__(self, task_name: str, cause: Optional[Exception] = None):
        super().__init__(
            code=ErrorCode.BIGCODE_EVALUATION_FAILED,
            details={"task": task_name},
            cause=cause,
            task_name=task_name
        )


class TaskNameRequiredError(WisentError):
    """Raised when task_name is required but not provided."""
    
    def __init__(self):
        super().__init__(code=ErrorCode.TASK_NAME_REQUIRED)


class NumericalExtractionError(WisentError):
    """Raised when numerical answer extraction fails."""
    
    def __init__(self, response: str):
        super().__init__(
            code=ErrorCode.NUMERICAL_EXTRACTION_FAILED,
            details={"response": response[:200] + "..." if len(response) > 200 else response}
        )


class TextExtractionError(WisentError):
    """Raised when text answer extraction fails."""
    
    def __init__(self, response: str):
        super().__init__(
            code=ErrorCode.TEXT_EXTRACTION_FAILED,
            details={"response": response[:200] + "..." if len(response) > 200 else response}
        )


class ExtractorNotFoundError(WisentError):
    """Raised when no extractor is found for a task."""
    
    def __init__(self, task_name: str):
        super().__init__(
            code=ErrorCode.EXTRACTOR_NOT_FOUND,
            details={
                "task": task_name,
                "known_patterns": ["math", "code", "hle", "gpqa", "science"]
            },
            task_name=task_name
        )


# Model/Chat Errors
class ChatTemplateNotAvailableError(WisentError):
    """Raised when chat template is required but not available."""
    
    def __init__(self, cause: Optional[Exception] = None):
        super().__init__(
            code=ErrorCode.CHAT_TEMPLATE_NOT_AVAILABLE,
            cause=cause
        )


class ChatTemplateError(WisentError):
    """Raised when chat template application fails."""
    
    def __init__(self, cause: Optional[Exception] = None):
        super().__init__(
            code=ErrorCode.CHAT_TEMPLATE_FAILED,
            cause=cause
        )


class EmptyGenerationError(WisentError):
    """Raised when model produces empty generation."""
    
    def __init__(self):
        super().__init__(code=ErrorCode.EMPTY_GENERATION)


class TokenizerMissingMethodError(WisentError):
    """Raised when tokenizer is missing required method."""
    
    def __init__(self, method_name: str = "apply_chat_template"):
        super().__init__(
            code=ErrorCode.TOKENIZER_MISSING_METHOD,
            details={"missing_method": method_name}
        )
