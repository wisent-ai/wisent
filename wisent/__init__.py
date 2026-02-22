import os
os.environ["NUMBA_NUM_THREADS"] = "1"

# Extend __path__ so sub-packages moved into support/ remain importable
_base = os.path.dirname(__file__)
for _root, _dirs, _files in os.walk(_base):
    if _root != _base:
        __path__.append(_root)

__version__ = "0.7.1539"

from wisent.core.tasks.base.diversity_processors import (
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
