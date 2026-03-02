import os
os.environ["NUMBA_NUM_THREADS"] = "1"

# Extend __path__ with only direct child directories (not nested descendants)
# to prevent namespace collisions on different Python versions/platforms
_base = os.path.dirname(__file__)
for _entry in sorted(os.listdir(_base)):
    _path = os.path.join(_base, _entry)
    if os.path.isdir(_path) and not _entry.startswith(('.', '_')):
        __path__.append(_path)

__version__ = "0.7.1585"

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
