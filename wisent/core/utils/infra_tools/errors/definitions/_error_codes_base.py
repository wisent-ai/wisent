"""Base WisentError and data loading error classes."""
from __future__ import annotations

from typing import Optional, Any, Dict, List
from wisent.core.utils.infra_tools.errors._error_codes_enum import ErrorCode, ERROR_MESSAGES

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
        
        message_template = ERROR_MESSAGES.get(code, f"Unknown error: {code}")
        try:
            self.message = message_template.format(**format_args)
        except KeyError:
            self.message = message_template
        
        if cause:
            self.message = f"{self.message} Cause: {cause}"
        
        super().__init__(self.message)

    def __str__(self):
        result = f"[{self.code.value}] {self.message}"
        if self.details:
            details_str = "\n".join(f"  - {k}: {v}" for k, v in self.details.items())
            result = f"{result}\nDetails:\n{details_str}"
        return result


# =============================================================================
# Data Loading Errors
# =============================================================================

class TaskLoadError(WisentError):
    def __init__(self, task_name: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.TASK_LOAD_FAILED, {"task": task_name}, cause, task_name=task_name)


class TaskNotFoundError(WisentError):
    def __init__(self, task_name: str, available_tasks: Optional[List[str]] = None):
        details = {"task": task_name}
        if available_tasks:
            details["available"] = available_tasks
        super().__init__(ErrorCode.TASK_NOT_FOUND, details, task_name=task_name)


class FallbackNotPermittedError(WisentError):
    def __init__(self, task_name: str):
        super().__init__(ErrorCode.FALLBACK_NOT_PERMITTED, {"task": task_name}, task_name=task_name)


class NoDocsAvailableError(WisentError):
    def __init__(self, task_name: str):
        super().__init__(ErrorCode.NO_DOCS_AVAILABLE, {"task": task_name}, task_name=task_name)


class DatasetLoadError(WisentError):
    def __init__(self, task_name: str):
        super().__init__(ErrorCode.DATASET_LOAD_FAILED, {"task": task_name}, task_name=task_name)


class VersionValidationError(WisentError):
    def __init__(self, version: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.VERSION_VALIDATION_FAILED, {"version": version}, cause, version=version)


class VersionInfoError(WisentError):
    def __init__(self, version: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.VERSION_INFO_FAILED, {"version": version}, cause, version=version)


class BenchmarkLoadError(WisentError):
    def __init__(self, benchmark_name: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.BENCHMARK_LOAD_FAILED, {"benchmark": benchmark_name}, cause, benchmark_name=benchmark_name)


class FileLoadError(WisentError):
    def __init__(self, file_path: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.FILE_LOAD_FAILED, {"file": file_path}, cause, file_path=file_path)


class InvalidJSONError(WisentError):
    def __init__(self, file_path: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.INVALID_JSON, {"file": file_path}, cause, file_path=file_path)


class InvalidDataFormatError(WisentError):
    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.INVALID_DATA_FORMAT, details or {}, reason=reason)


class GroupExpansionError(WisentError):
    def __init__(self, task_name: str):
        super().__init__(ErrorCode.GROUP_EXPANSION_FAILED, {"task": task_name}, task_name=task_name)


# =============================================================================
