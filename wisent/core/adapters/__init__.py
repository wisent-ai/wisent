"""
Multi-modal adapters for Wisent contrastive steering.

Adapters provide a unified interface for different modalities (text, audio, video, robotics)
while keeping the core steering logic modality-agnostic.
"""
from wisent.core.adapters.base import BaseAdapter, AdapterError
from wisent.core.adapters.text import TextAdapter
from wisent.core.adapters.modalities import (
    AudioAdapter,
    VideoAdapter,
    RoboticsAdapter,
    MultimodalAdapter,
)

__all__ = [
    "BaseAdapter",
    "AdapterError",
    "TextAdapter",
    "AudioAdapter",
    "VideoAdapter",
    "RoboticsAdapter",
    "MultimodalAdapter",
]
