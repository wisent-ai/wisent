"""Centralized error codes and messages for Wisent.

This module defines all error codes, messages, and custom exceptions used throughout
the codebase. NO FALLBACKS - errors should be raised immediately with clear codes.
"""

from enum import Enum
from typing import Optional, Any, Dict, List


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
    FILE_LOAD_FAILED = "E1009"
    INVALID_JSON = "E1010"
    INVALID_DATA_FORMAT = "E1011"
    GROUP_EXPANSION_FAILED = "E1012"
    
    # Evaluation Errors (2xxx)
    EVALUATION_FAILED = "E2001"
    EXACT_MATCH_FAILED = "E2002"
    BIGCODE_EVALUATION_FAILED = "E2003"
    TASK_NAME_REQUIRED = "E2004"
    NUMERICAL_EXTRACTION_FAILED = "E2005"
    TEXT_EXTRACTION_FAILED = "E2006"
    EXTRACTOR_NOT_FOUND = "E2007"
    EXTRACTOR_RETURNED_NONE = "E2008"
    BIGCODE_TASK_REQUIRES_FLAG = "E2009"
    
    # Model/Chat Errors (3xxx)
    CHAT_TEMPLATE_NOT_AVAILABLE = "E3001"
    CHAT_TEMPLATE_FAILED = "E3002"
    EMPTY_GENERATION = "E3003"
    TOKENIZER_MISSING_METHOD = "E3004"
    MODEL_ARCHITECTURE_UNKNOWN = "E3005"
    DECODER_LAYERS_NOT_FOUND = "E3006"
    HIDDEN_SIZE_NOT_FOUND = "E3007"
    NO_HIDDEN_STATES = "E3008"
    LAYER_NOT_FOUND = "E3009"
    MODEL_CONFIG_ACCESS_FAILED = "E3010"
    
    # Configuration/Validation Errors (4xxx)
    MISSING_PARAMETER = "E4001"
    INVALID_CHOICES = "E4002"
    MODEL_NOT_PROVIDED = "E4003"
    INVALID_RANGE = "E4004"
    INVALID_VALUE = "E4005"
    DUPLICATE_NAME = "E4006"
    UNKNOWN_TYPE = "E4007"
    INSUFFICIENT_DATA = "E4008"
    LAYER_RANGE_ERROR = "E4009"
    
    # Agent/Diagnostic Errors (5xxx)
    CLASSIFIER_LOAD_FAILED = "E5001"
    CLASSIFIER_CREATION_FAILED = "E5002"
    NO_QUALITY_SCORES = "E5003"
    NO_CONFIDENCE_SCORES = "E5004"
    CLASSIFIER_CONFIG_REQUIRED = "E5005"
    DEVICE_BENCHMARK_FAILED = "E5006"
    BUDGET_CALCULATION_FAILED = "E5007"
    RESOURCE_ESTIMATION_FAILED = "E5008"
    NO_BENCHMARK_DATA = "E5009"
    TRAINING_DATA_GENERATION_FAILED = "E5010"
    NO_SUITABLE_CLASSIFIER = "E5011"
    IMPROVEMENT_METHOD_UNKNOWN = "E5012"
    
    # Steering/Training Errors (6xxx)
    STEERING_METHOD_UNKNOWN = "E6001"
    NO_TRAINING_RESULT = "E6002"
    CONTROL_VECTOR_DIAGNOSTICS_FAILED = "E6003"
    NO_TRAINED_VECTORS = "E6004"
    STEERING_TRAINER_NOT_FOUND = "E6005"
    INVALID_LAYER_ID = "E6006"
    NO_CANDIDATE_LAYERS = "E6007"
    OPTIMIZATION_FAILED = "E6008"
    VECTOR_QUALITY_TOO_LOW = "E6009"
    
    # Contrastive Pairs Errors (7xxx)
    PAIR_GENERATION_FAILED = "E7001"
    INVALID_PAIR_STRUCTURE = "E7002"
    NO_ACTIVATION_DATA = "E7003"
    PAIR_DIAGNOSTICS_FAILED = "E7004"
    PROMPT_MUST_BE_STRING = "E7005"
    RESPONSE_MUST_BE_STRING = "E7006"
    
    # Docker/Runtime Errors (8xxx)
    DOCKER_RUNTIME_ERROR = "E8001"
    CONTAINER_NOT_RUNNING = "E8002"
    EXECUTION_FAILED = "E8003"
    
    # Calibration/Timing Errors (9xxx)
    CALIBRATION_FAILED = "E9001"
    CALIBRATION_DATA_MISSING = "E9002"
    CALIBRATION_DATA_INVALID = "E9003"
    CALIBRATION_REQUIRED = "E9004"
    
    # Serialization Errors (10xxx)
    DECODE_FAILED = "E10001"
    INVALID_JSON_STRUCTURE = "E10002"
    MISSING_REQUIRED_FIELD = "E10003"
    
    # Multimodal Errors (11xxx)
    NO_WAVEFORM_DATA = "E11001"
    NO_PIXEL_DATA = "E11002"
    NO_FRAME_DATA = "E11003"
    NO_STATE_DATA = "E11004"
    NO_ACTION_DATA = "E11005"
    EMPTY_TRAJECTORY = "E11006"
    MULTIMODAL_CONTENT_REQUIRED = "E11007"


ERROR_MESSAGES: Dict[ErrorCode, str] = {
    # Data Loading Errors
    ErrorCode.TASK_LOAD_FAILED: "Failed to load task '{task_name}'. All loading approaches exhausted.",
    ErrorCode.TASK_NOT_FOUND: "Task '{task_name}' not found.",
    ErrorCode.FALLBACK_NOT_PERMITTED: "Fallback loading is not permitted for task '{task_name}'. Task must be loaded through proper channels.",
    ErrorCode.NO_DOCS_AVAILABLE: "No labeled documents available for task '{task_name}'. Task does not expose docs from validation/test/train/fewshot.",
    ErrorCode.DATASET_LOAD_FAILED: "Fallback dataset loading is not permitted for task '{task_name}'. Task must provide docs through standard lm-evaluation-harness methods.",
    ErrorCode.VERSION_VALIDATION_FAILED: "Failed to validate release version '{version}'. Could not load available versions from data loader.",
    ErrorCode.VERSION_INFO_FAILED: "Failed to get version info for '{version}'. Data loader error.",
    ErrorCode.BENCHMARK_LOAD_FAILED: "Failed to load benchmark '{benchmark_name}'.",
    ErrorCode.FILE_LOAD_FAILED: "Failed to load file '{file_path}'.",
    ErrorCode.INVALID_JSON: "Invalid JSON in file '{file_path}'.",
    ErrorCode.INVALID_DATA_FORMAT: "Invalid data format: {reason}.",
    ErrorCode.GROUP_EXPANSION_FAILED: "No group expansion found for task '{task_name}'.",
    
    # Evaluation Errors
    ErrorCode.EVALUATION_FAILED: "Evaluation failed for task '{task_name}'.",
    ErrorCode.EXACT_MATCH_FAILED: "Evaluation failed for prediction {index}.",
    ErrorCode.BIGCODE_EVALUATION_FAILED: "BigCode evaluation failed for coding task '{task_name}'. Ensure task documents are provided and properly formatted.",
    ErrorCode.TASK_NAME_REQUIRED: "task_name is required for benchmark evaluation.",
    ErrorCode.NUMERICAL_EXTRACTION_FAILED: "Could not extract numerical answer from response. Response must contain an explicit answer marker.",
    ErrorCode.TEXT_EXTRACTION_FAILED: "Could not extract text answer from response. Response must contain an explicit answer marker.",
    ErrorCode.EXTRACTOR_NOT_FOUND: "No extractor registered for task '{task_name}'.",
    ErrorCode.EXTRACTOR_RETURNED_NONE: "Extractor returned None for task '{task_name}'.",
    ErrorCode.BIGCODE_TASK_REQUIRES_FLAG: "Task '{task_name}' is a BigCode task. Use --bigcode flag or BigCodeTaskLoader.",
    
    # Model/Chat Errors
    ErrorCode.CHAT_TEMPLATE_NOT_AVAILABLE: "Chat template is required but not available. Use a model/tokenizer that supports chat templates.",
    ErrorCode.CHAT_TEMPLATE_FAILED: "Failed to apply chat template. Ensure the tokenizer has a valid chat template configured.",
    ErrorCode.EMPTY_GENERATION: "Empty generation detected. Model produced no output tokens.",
    ErrorCode.TOKENIZER_MISSING_METHOD: "Tokenizer does not have '{method_name}' method.",
    ErrorCode.MODEL_ARCHITECTURE_UNKNOWN: "Unknown model architecture. Cannot proceed with operation.",
    ErrorCode.DECODER_LAYERS_NOT_FOUND: "Could not resolve decoder layers for steering hooks.",
    ErrorCode.HIDDEN_SIZE_NOT_FOUND: "Could not infer hidden size from model config.",
    ErrorCode.NO_HIDDEN_STATES: "No hidden_states returned. Model may not support it.",
    ErrorCode.LAYER_NOT_FOUND: "Could not find layer '{layer_name}' in model.",
    ErrorCode.MODEL_CONFIG_ACCESS_FAILED: "Cannot access model configuration for '{model_name}'.",
    
    # Configuration/Validation Errors
    ErrorCode.MISSING_PARAMETER: "Required parameters missing: {params}.",
    ErrorCode.INVALID_CHOICES: "Invalid choices for task '{task_name}': {reason}.",
    ErrorCode.MODEL_NOT_PROVIDED: "Model required for evaluation but not provided.",
    ErrorCode.INVALID_RANGE: "Value must be in range [{min_val}, {max_val}], got {value}.",
    ErrorCode.INVALID_VALUE: "Invalid value for '{param}': {reason}.",
    ErrorCode.DUPLICATE_NAME: "Duplicate {entity_type} name: '{name}'.",
    ErrorCode.UNKNOWN_TYPE: "Unknown {entity_type}: '{value}'.",
    ErrorCode.INSUFFICIENT_DATA: "Insufficient data: {reason}.",
    ErrorCode.LAYER_RANGE_ERROR: "Cannot determine layer range: {reason}.",
    
    # Agent/Diagnostic Errors
    ErrorCode.CLASSIFIER_LOAD_FAILED: "Failed to load classifiers. System cannot operate without them.",
    ErrorCode.CLASSIFIER_CREATION_FAILED: "Failed to create classifier for '{issue_type}'.",
    ErrorCode.NO_QUALITY_SCORES: "No quality scores available. All classifiers failed.",
    ErrorCode.NO_CONFIDENCE_SCORES: "No confidence scores available. All classifiers failed.",
    ErrorCode.CLASSIFIER_CONFIG_REQUIRED: "classifier_configs is required. No fallback mode available.",
    ErrorCode.DEVICE_BENCHMARK_FAILED: "Device benchmark failed for '{task_name}'. Run device benchmark first.",
    ErrorCode.BUDGET_CALCULATION_FAILED: "Budget calculation failed for task '{task_type}'. Run device benchmark first.",
    ErrorCode.RESOURCE_ESTIMATION_FAILED: "{resource_type} estimation failed for task '{task_name}'.",
    ErrorCode.NO_BENCHMARK_DATA: "No benchmark data available for device. Run benchmark first.",
    ErrorCode.TRAINING_DATA_GENERATION_FAILED: "Could not generate training data for issues: {issues}.",
    ErrorCode.NO_SUITABLE_CLASSIFIER: "No suitable classifier found for issue type: '{issue_type}'.",
    ErrorCode.IMPROVEMENT_METHOD_UNKNOWN: "Unknown improvement method: '{method}'.",
    
    # Steering/Training Errors
    ErrorCode.STEERING_METHOD_UNKNOWN: "Unknown steering method: '{method}'.",
    ErrorCode.NO_TRAINING_RESULT: "No result to save. Run the trainer first.",
    ErrorCode.CONTROL_VECTOR_DIAGNOSTICS_FAILED: "Control vector diagnostics found critical issues.",
    ErrorCode.NO_TRAINED_VECTORS: "No trained vectors to save.",
    ErrorCode.STEERING_TRAINER_NOT_FOUND: "No trainer registered for method: '{method}'.",
    ErrorCode.INVALID_LAYER_ID: "layer_id {layer_id} exceeds model layers ({num_layers}).",
    ErrorCode.NO_CANDIDATE_LAYERS: "No valid candidate layers to optimize.",
    ErrorCode.OPTIMIZATION_FAILED: "Optimization failed: {reason}.",
    ErrorCode.VECTOR_QUALITY_TOO_LOW: "Steering vector quality too low ({quality}). {reason} Use --accept-low-quality-vector to override.",
    
    # Contrastive Pairs Errors
    ErrorCode.PAIR_GENERATION_FAILED: "Failed to generate contrastive pairs: {reason}.",
    ErrorCode.INVALID_PAIR_STRUCTURE: "Invalid pair structure: {reason}.",
    ErrorCode.NO_ACTIVATION_DATA: "No activation data found in contrastive pairs.",
    ErrorCode.PAIR_DIAGNOSTICS_FAILED: "Contrastive pair diagnostics found critical issues.",
    ErrorCode.PROMPT_MUST_BE_STRING: "'prompt' must be a non-empty string.",
    ErrorCode.RESPONSE_MUST_BE_STRING: "'{field}' must be a non-empty string.",
    
    # Docker/Runtime Errors
    ErrorCode.DOCKER_RUNTIME_ERROR: "Docker runtime error: {reason}.",
    ErrorCode.CONTAINER_NOT_RUNNING: "Container is not running.",
    ErrorCode.EXECUTION_FAILED: "Execution failed: {reason}.",
    
    # Calibration/Timing Errors
    ErrorCode.CALIBRATION_FAILED: "Calibration failed: {reason}.",
    ErrorCode.CALIBRATION_DATA_MISSING: "No calibration data available. Run calibration first.",
    ErrorCode.CALIBRATION_DATA_INVALID: "Calibration file contains invalid data.",
    ErrorCode.CALIBRATION_REQUIRED: "Calibration cannot be skipped. Accurate timing requires calibration.",
    
    # Serialization Errors
    ErrorCode.DECODE_FAILED: "Failed to decode {data_type}: {reason}.",
    ErrorCode.INVALID_JSON_STRUCTURE: "Invalid JSON structure: {reason}.",
    ErrorCode.MISSING_REQUIRED_FIELD: "Missing required field: '{field}'.",
    
    # Multimodal Errors
    ErrorCode.NO_WAVEFORM_DATA: "No waveform data. Load from file_path first.",
    ErrorCode.NO_PIXEL_DATA: "No pixel data. Load from file_path first.",
    ErrorCode.NO_FRAME_DATA: "No frame data. Load from file_path first.",
    ErrorCode.NO_STATE_DATA: "No numerical state data.",
    ErrorCode.NO_ACTION_DATA: "No action data.",
    ErrorCode.EMPTY_TRAJECTORY: "Empty trajectory.",
    ErrorCode.MULTIMODAL_CONTENT_REQUIRED: "MultimodalContent requires at least one content item.",
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

class SteeringMethodUnknownError(WisentError):
    def __init__(self, method: str, available_methods: Optional[List[str]] = None):
        details = {"method": method}
        if available_methods:
            details["available"] = available_methods
        super().__init__(ErrorCode.STEERING_METHOD_UNKNOWN, details, method=method)


class NoTrainingResultError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_TRAINING_RESULT)


class ControlVectorDiagnosticsError(WisentError):
    def __init__(self, details: Optional[Dict[str, Any]] = None):
        super().__init__(ErrorCode.CONTROL_VECTOR_DIAGNOSTICS_FAILED, details or {})


class NoTrainedVectorsError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_TRAINED_VECTORS)


class SteeringTrainerNotFoundError(WisentError):
    def __init__(self, method: str):
        super().__init__(ErrorCode.STEERING_TRAINER_NOT_FOUND, {"method": method}, method=method)


class InvalidLayerIdError(WisentError):
    def __init__(self, layer_id: int, num_layers: int):
        super().__init__(ErrorCode.INVALID_LAYER_ID, {"layer_id": layer_id, "num_layers": num_layers}, layer_id=layer_id, num_layers=num_layers)


class NoCandidateLayersError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_CANDIDATE_LAYERS)


class OptimizationError(WisentError):
    def __init__(self, reason: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.OPTIMIZATION_FAILED, {"reason": reason}, cause, reason=reason)


class VectorQualityTooLowError(WisentError):
    def __init__(self, quality: str, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            ErrorCode.VECTOR_QUALITY_TOO_LOW, 
            details or {}, 
            quality=quality, 
            reason=reason
        )


# =============================================================================
# Contrastive Pairs Errors
# =============================================================================

class PairGenerationError(WisentError):
    def __init__(self, reason: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.PAIR_GENERATION_FAILED, {"reason": reason}, cause, reason=reason)


class InvalidPairStructureError(WisentError):
    def __init__(self, reason: str):
        super().__init__(ErrorCode.INVALID_PAIR_STRUCTURE, {"reason": reason}, reason=reason)


class NoActivationDataError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_ACTIVATION_DATA)


class PairDiagnosticsError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.PAIR_DIAGNOSTICS_FAILED)


class PromptMustBeStringError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.PROMPT_MUST_BE_STRING)


class ResponseMustBeStringError(WisentError):
    def __init__(self, field: str = "model_response"):
        super().__init__(ErrorCode.RESPONSE_MUST_BE_STRING, {"field": field}, field=field)


# =============================================================================
# Docker/Runtime Errors
# =============================================================================

class DockerRuntimeError(WisentError):
    def __init__(self, reason: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.DOCKER_RUNTIME_ERROR, {"reason": reason}, cause, reason=reason)


class ContainerNotRunningError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.CONTAINER_NOT_RUNNING)


class ExecutionError(WisentError):
    def __init__(self, reason: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.EXECUTION_FAILED, {"reason": reason}, cause, reason=reason)


# =============================================================================
# Calibration/Timing Errors
# =============================================================================

class CalibrationError(WisentError):
    def __init__(self, reason: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.CALIBRATION_FAILED, {"reason": reason}, cause, reason=reason)


class CalibrationDataMissingError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.CALIBRATION_DATA_MISSING)


class CalibrationDataInvalidError(WisentError):
    def __init__(self, file_path: Optional[str] = None):
        details = {"file": file_path} if file_path else {}
        super().__init__(ErrorCode.CALIBRATION_DATA_INVALID, details)


class CalibrationRequiredError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.CALIBRATION_REQUIRED)


# =============================================================================
# Serialization Errors
# =============================================================================

class DecodeError(WisentError):
    def __init__(self, data_type: str, reason: str, cause: Optional[Exception] = None):
        super().__init__(ErrorCode.DECODE_FAILED, {"data_type": data_type, "reason": reason}, cause, data_type=data_type, reason=reason)


class InvalidJSONStructureError(WisentError):
    def __init__(self, reason: str):
        super().__init__(ErrorCode.INVALID_JSON_STRUCTURE, {"reason": reason}, reason=reason)


class MissingRequiredFieldError(WisentError):
    def __init__(self, field: str):
        super().__init__(ErrorCode.MISSING_REQUIRED_FIELD, {"field": field}, field=field)


# =============================================================================
# Multimodal Errors
# =============================================================================

class NoWaveformDataError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_WAVEFORM_DATA)


class NoPixelDataError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_PIXEL_DATA)


class NoFrameDataError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_FRAME_DATA)


class NoStateDataError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_STATE_DATA)


class NoActionDataError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.NO_ACTION_DATA)


class EmptyTrajectoryError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.EMPTY_TRAJECTORY)


class MultimodalContentRequiredError(WisentError):
    def __init__(self):
        super().__init__(ErrorCode.MULTIMODAL_CONTENT_REQUIRED)
