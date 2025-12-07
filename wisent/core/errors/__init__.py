"""Wisent error handling module."""

from .error_codes import (
    ErrorCode,
    ERROR_MESSAGES,
    WisentError,
    # Data Loading Errors
    TaskLoadError,
    FallbackNotPermittedError,
    NoDocsAvailableError,
    DatasetLoadError,
    VersionValidationError,
    VersionInfoError,
    BenchmarkLoadError,
    # Evaluation Errors
    EvaluationError,
    ExactMatchError,
    BigCodeEvaluationError,
    TaskNameRequiredError,
    NumericalExtractionError,
    TextExtractionError,
    ExtractorNotFoundError,
    # Model/Chat Errors
    ChatTemplateNotAvailableError,
    ChatTemplateError,
    EmptyGenerationError,
    TokenizerMissingMethodError,
)

__all__ = [
    "ErrorCode",
    "ERROR_MESSAGES",
    "WisentError",
    # Data Loading Errors
    "TaskLoadError",
    "FallbackNotPermittedError",
    "NoDocsAvailableError",
    "DatasetLoadError",
    "VersionValidationError",
    "VersionInfoError",
    "BenchmarkLoadError",
    # Evaluation Errors
    "EvaluationError",
    "ExactMatchError",
    "BigCodeEvaluationError",
    "TaskNameRequiredError",
    "NumericalExtractionError",
    "TextExtractionError",
    "ExtractorNotFoundError",
    # Model/Chat Errors
    "ChatTemplateNotAvailableError",
    "ChatTemplateError",
    "EmptyGenerationError",
    "TokenizerMissingMethodError",
]