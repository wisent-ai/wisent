"""Centralized error codes and messages for Wisent.

This module defines all error codes, messages, and custom exceptions used throughout
the codebase. NO FALLBACKS - errors should be raised immediately with clear codes.
"""

from __future__ import annotations

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

