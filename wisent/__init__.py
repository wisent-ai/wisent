import os
os.environ["NUMBA_NUM_THREADS"] = "1"

__version__ = "0.7.1332"

from wisent.core.diversity_processors import (
    OpenerPenaltyProcessor,
    TriePenaltyProcessor,
    PhraseLedger,
    build_diversity_processors,
)

# Multi-modal support
from wisent.core.wisent import Wisent, TraitConfig

from wisent.core.modalities import (
    Modality,
    ModalityContent,
    TextContent,
    AudioContent,
    VideoContent,
    ImageContent,
    RobotState,
    RobotAction,
    RobotTrajectory,
    MultimodalContent,
)

from wisent.core.adapters import (
    BaseAdapter,
    TextAdapter,
    AudioAdapter,
    VideoAdapter,
    RoboticsAdapter,
    MultimodalAdapter,
)

__all__ = [
    # Version
    "__version__",
    # Diversity processors
    "OpenerPenaltyProcessor",
    "TriePenaltyProcessor",
    "PhraseLedger",
    "build_diversity_processors",
    # Main interface
    "Wisent",
    "TraitConfig",
    # Modalities
    "Modality",
    "ModalityContent",
    "TextContent",
    "AudioContent",
    "VideoContent",
    "ImageContent",
    "RobotState",
    "RobotAction",
    "RobotTrajectory",
    "MultimodalContent",
    # Adapters
    "BaseAdapter",
    "TextAdapter",
    "AudioAdapter",
    "VideoAdapter",
    "RoboticsAdapter",
    "MultimodalAdapter",
]
