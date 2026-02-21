"""
Multi-modal content types for Wisent.

This module defines the content types that can be used as inputs/outputs
across different modalities (text, audio, video, robotics).

All types are re-exported here to preserve the original import path:
  from wisent.core.modalities import TextContent, AudioContent, ...
"""
from wisent.core.modalities.text_content import (
    Modality,
    ModalityContent,
    TextContent,
)
from wisent.core.modalities.media_content import (
    AudioContent,
    ImageContent,
    VideoContent,
)
from wisent.core.modalities.robot_content import (
    RobotState,
    RobotAction,
    RobotTrajectory,
    MultimodalContent,
    ContentType,
    wrap_content,
)

__all__ = [
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
    "ContentType",
    "wrap_content",
]
