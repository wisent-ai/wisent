"""Centralized error codes and messages for Wisent.

Re-exports from split modules for backward compatibility.
"""
from wisent.core.errors._error_codes_enum import ErrorCode, ERROR_MESSAGES
from wisent.core.errors._error_codes_base import (
    WisentError,
    TaskLoadError, TaskNotFoundError, FallbackNotPermittedError,
    NoDocsAvailableError, DatasetLoadError, VersionValidationError,
    VersionInfoError, BenchmarkLoadError, FileLoadError,
    InvalidJSONError, InvalidDataFormatError, GroupExpansionError,
)
from wisent.core.errors._error_codes_eval import (
    EvaluationError, ExactMatchError, BigCodeEvaluationError,
    TaskNameRequiredError, NumericalExtractionError, TextExtractionError,
    ExtractorNotFoundError, ExtractorReturnedNoneError,
    BigCodeTaskRequiresFlagError,
    ChatTemplateNotAvailableError, ChatTemplateError,
    EmptyGenerationError, TokenizerMissingMethodError,
    ModelArchitectureUnknownError, DecoderLayersNotFoundError,
    HiddenSizeNotFoundError, NoHiddenStatesError,
    LayerNotFoundError, ModelConfigAccessError,
)
from wisent.core.errors._error_codes_config import (
    MissingParameterError, InvalidChoicesError, ModelNotProvidedError,
    InvalidRangeError, InvalidValueError, DuplicateNameError,
    UnknownTypeError, InsufficientDataError, LayerRangeError,
    ClassifierLoadError, ClassifierCreationError,
    NoQualityScoresError, NoConfidenceScoresError,
    ClassifierConfigRequiredError, DeviceBenchmarkError,
    BudgetCalculationError, ResourceEstimationError,
    NoBenchmarkDataError, TrainingDataGenerationError,
    NoSuitableClassifierError, ImprovementMethodUnknownError,
)
from wisent.core.errors._error_codes_other import (
    SteeringMethodUnknownError, NoTrainingResultError,
    ControlVectorDiagnosticsError, NoTrainedVectorsError,
    SteeringTrainerNotFoundError, InvalidLayerIdError,
    NoCandidateLayersError, OptimizationError, VectorQualityTooLowError,
    PairGenerationError, InvalidPairStructureError,
    NoActivationDataError, PairDiagnosticsError,
    PromptMustBeStringError, ResponseMustBeStringError,
    DockerRuntimeError, ContainerNotRunningError, ExecutionError,
    CalibrationError, CalibrationDataMissingError,
    CalibrationDataInvalidError, CalibrationRequiredError,
    DecodeError, InvalidJSONStructureError, MissingRequiredFieldError,
    NoWaveformDataError, NoPixelDataError, NoFrameDataError,
    NoStateDataError, NoActionDataError, EmptyTrajectoryError,
    MultimodalContentRequiredError,
)
