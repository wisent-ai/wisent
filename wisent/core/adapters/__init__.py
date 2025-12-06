"""
Multi-modal adapters for Wisent contrastive steering.

Adapters provide a unified interface for different modalities (text, audio, video, robotics)
while keeping the core steering logic modality-agnostic.
"""
from wisent.core.adapters.base import BaseAdapter, AdapterError
from wisent.core.adapters.text import TextAdapter
from wisent.core.adapters.audio import AudioAdapter
from wisent.core.adapters.video import VideoAdapter
from wisent.core.adapters.robotics import RoboticsAdapter
from wisent.core.adapters.multimodal import MultimodalAdapter

__all__ = [
    "BaseAdapter",
    "AdapterError",
    "TextAdapter",
    "AudioAdapter",
    "VideoAdapter",
    "RoboticsAdapter",
    "MultimodalAdapter",
]
