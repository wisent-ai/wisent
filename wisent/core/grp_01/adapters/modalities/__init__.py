import os as _os
_base = _os.path.dirname(__file__)
for _root, _dirs, _files in _os.walk(_base):
    _dirs[:] = sorted(d for d in _dirs if d.startswith(("grp_", "sub_", "mid_")))
    if _root != _base:
        __path__.append(_root)

"""Non-text modality adapter implementations."""

from .audio import AudioAdapter
from .multimodal import MultimodalAdapter, MultimodalSteeringConfig
from .robotics import RoboticsAdapter, RoboticsSteeringConfig
from .video import VideoAdapter, VideoSteeringConfig

__all__ = [
    'AudioAdapter',
    'MultimodalAdapter',
    'MultimodalSteeringConfig',
    'RoboticsAdapter',
    'RoboticsSteeringConfig',
    'VideoAdapter',
    'VideoSteeringConfig',
]
