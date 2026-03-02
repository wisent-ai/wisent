"""Steering, runtime, calibration, serialization, and multimodal errors."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from wisent.core.errors._error_codes_enum import ErrorCode
from wisent.core.errors._error_codes_base import WisentError


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
