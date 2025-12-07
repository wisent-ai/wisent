"""
Multi-modal content types for Wisent.

This module defines the content types that can be used as inputs/outputs
across different modalities (text, audio, video, robotics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, List, Optional, Any, Dict
from enum import Enum, auto
from pathlib import Path

import torch
import numpy as np

from wisent.core.errors import (
    NoWaveformDataError,
    NoPixelDataError,
    NoFrameDataError,
    NoStateDataError,
    NoActionDataError,
    EmptyTrajectoryError,
    MultimodalContentRequiredError,
    InvalidValueError,
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
]


class Modality(Enum):
    """Supported modalities for contrastive steering."""
    TEXT = auto()
    AUDIO = auto()
    VIDEO = auto()
    IMAGE = auto()
    ROBOT_STATE = auto()
    ROBOT_ACTION = auto()
    MULTIMODAL = auto()


@dataclass(frozen=True, slots=True)
class ModalityContent:
    """Base class for all modality content types."""
    modality: Modality = field(init=False)

    def to_tensor(self) -> torch.Tensor:
        """Convert content to tensor representation."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModalityContent":
        """Deserialize from dictionary."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class TextContent(ModalityContent):
    """Text content - backward compatible with current string-based system."""
    text: str
    modality: Modality = field(default=Modality.TEXT, init=False)

    def __str__(self) -> str:
        return self.text

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError("TextContent requires tokenization - use adapter.encode()")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "text", "text": self.text}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextContent":
        return cls(text=data["text"])


@dataclass(frozen=True, slots=True)
class AudioContent(ModalityContent):
    """
    Audio content for speech/audio steering.

    Attributes:
        waveform: Raw audio waveform as tensor [channels, samples] or numpy array
        sample_rate: Sample rate in Hz (e.g., 16000, 44100)
        file_path: Optional path to audio file
    """
    waveform: torch.Tensor | np.ndarray | None = None
    sample_rate: int = 16000
    file_path: Path | str | None = None
    modality: Modality = field(default=Modality.AUDIO, init=False)

    def __post_init__(self):
        if self.waveform is None and self.file_path is None:
            raise InvalidValueError(param="AudioContent", reason="requires either waveform or file_path")

    def to_tensor(self) -> torch.Tensor:
        if self.waveform is not None:
            if isinstance(self.waveform, np.ndarray):
                return torch.from_numpy(self.waveform)
            return self.waveform
        raise NoWaveformDataError()

    def to_dict(self) -> Dict[str, Any]:
        data = {"type": "audio", "sample_rate": self.sample_rate}
        if self.file_path is not None:
            data["file_path"] = str(self.file_path)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioContent":
        return cls(
            sample_rate=data.get("sample_rate", 16000),
            file_path=data.get("file_path"),
        )

    @classmethod
    def from_file(cls, file_path: str | Path, sample_rate: int = 16000) -> "AudioContent":
        """Load audio from file (requires torchaudio or librosa)."""
        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(file_path))
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)
            return cls(waveform=waveform, sample_rate=sample_rate, file_path=file_path)
        except ImportError:
            return cls(file_path=file_path, sample_rate=sample_rate)


@dataclass(frozen=True, slots=True)
class ImageContent(ModalityContent):
    """
    Image content for vision tasks.

    Attributes:
        pixels: Image tensor [C, H, W] or numpy array [H, W, C]
        file_path: Optional path to image file
    """
    pixels: torch.Tensor | np.ndarray | None = None
    file_path: Path | str | None = None
    modality: Modality = field(default=Modality.IMAGE, init=False)

    def __post_init__(self):
        if self.pixels is None and self.file_path is None:
            raise InvalidValueError(param="ImageContent", reason="requires either pixels or file_path")

    def to_tensor(self) -> torch.Tensor:
        if self.pixels is not None:
            if isinstance(self.pixels, np.ndarray):
                # Convert HWC to CHW
                if self.pixels.ndim == 3 and self.pixels.shape[-1] in (1, 3, 4):
                    return torch.from_numpy(self.pixels).permute(2, 0, 1)
                return torch.from_numpy(self.pixels)
            return self.pixels
        raise NoPixelDataError()

    def to_dict(self) -> Dict[str, Any]:
        data = {"type": "image"}
        if self.file_path is not None:
            data["file_path"] = str(self.file_path)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageContent":
        return cls(file_path=data.get("file_path"))

    @classmethod
    def from_file(cls, file_path: str | Path) -> "ImageContent":
        """Load image from file (requires PIL or torchvision)."""
        try:
            from PIL import Image
            import torchvision.transforms as T
            img = Image.open(file_path).convert("RGB")
            transform = T.ToTensor()
            pixels = transform(img)
            return cls(pixels=pixels, file_path=file_path)
        except ImportError:
            return cls(file_path=file_path)


@dataclass(frozen=True, slots=True)
class VideoContent(ModalityContent):
    """
    Video content for video understanding/generation tasks.

    Attributes:
        frames: Video tensor [T, C, H, W] or list of frame tensors
        fps: Frames per second
        file_path: Optional path to video file
    """
    frames: torch.Tensor | List[torch.Tensor] | np.ndarray | None = None
    fps: float = 30.0
    file_path: Path | str | None = None
    modality: Modality = field(default=Modality.VIDEO, init=False)

    def __post_init__(self):
        if self.frames is None and self.file_path is None:
            raise InvalidValueError(param="VideoContent", reason="requires either frames or file_path")

    @property
    def num_frames(self) -> int:
        if self.frames is None:
            return 0
        if isinstance(self.frames, list):
            return len(self.frames)
        return self.frames.shape[0]

    def to_tensor(self) -> torch.Tensor:
        if self.frames is not None:
            if isinstance(self.frames, list):
                return torch.stack(self.frames)
            if isinstance(self.frames, np.ndarray):
                return torch.from_numpy(self.frames)
            return self.frames
        raise NoFrameDataError()

    def to_dict(self) -> Dict[str, Any]:
        data = {"type": "video", "fps": self.fps}
        if self.file_path is not None:
            data["file_path"] = str(self.file_path)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoContent":
        return cls(
            fps=data.get("fps", 30.0),
            file_path=data.get("file_path"),
        )

    @classmethod
    def from_file(cls, file_path: str | Path, max_frames: int | None = None) -> "VideoContent":
        """Load video from file (requires decord or torchvision)."""
        try:
            from decord import VideoReader, cpu
            vr = VideoReader(str(file_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            n_frames = len(vr)
            if max_frames and n_frames > max_frames:
                indices = np.linspace(0, n_frames - 1, max_frames, dtype=int)
            else:
                indices = np.arange(n_frames)
            frames = vr.get_batch(indices).asnumpy()  # [T, H, W, C]
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
            return cls(frames=frames, fps=fps, file_path=file_path)
        except ImportError:
            return cls(file_path=file_path)


@dataclass(frozen=True, slots=True)
class RobotState(ModalityContent):
    """
    Robot state observation.

    Attributes:
        joint_positions: Joint angles/positions [n_joints]
        joint_velocities: Joint velocities [n_joints]
        end_effector_pose: End effector position + orientation [7] (xyz + quaternion)
        gripper_state: Gripper openness [0, 1]
        image_obs: Optional camera observation
        proprioception: Additional proprioceptive data
    """
    joint_positions: torch.Tensor | np.ndarray | None = None
    joint_velocities: torch.Tensor | np.ndarray | None = None
    end_effector_pose: torch.Tensor | np.ndarray | None = None
    gripper_state: float | None = None
    image_obs: ImageContent | None = None
    proprioception: Dict[str, Any] | None = None
    modality: Modality = field(default=Modality.ROBOT_STATE, init=False)

    def to_tensor(self) -> torch.Tensor:
        """Concatenate all numerical state into a single vector."""
        parts = []
        if self.joint_positions is not None:
            jp = self.joint_positions
            if isinstance(jp, np.ndarray):
                jp = torch.from_numpy(jp)
            parts.append(jp.flatten())
        if self.joint_velocities is not None:
            jv = self.joint_velocities
            if isinstance(jv, np.ndarray):
                jv = torch.from_numpy(jv)
            parts.append(jv.flatten())
        if self.end_effector_pose is not None:
            ee = self.end_effector_pose
            if isinstance(ee, np.ndarray):
                ee = torch.from_numpy(ee)
            parts.append(ee.flatten())
        if self.gripper_state is not None:
            parts.append(torch.tensor([self.gripper_state]))
        if not parts:
            raise NoStateDataError()
        return torch.cat(parts)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"type": "robot_state"}
        if self.joint_positions is not None:
            jp = self.joint_positions
            data["joint_positions"] = jp.tolist() if hasattr(jp, "tolist") else list(jp)
        if self.joint_velocities is not None:
            jv = self.joint_velocities
            data["joint_velocities"] = jv.tolist() if hasattr(jv, "tolist") else list(jv)
        if self.end_effector_pose is not None:
            ee = self.end_effector_pose
            data["end_effector_pose"] = ee.tolist() if hasattr(ee, "tolist") else list(ee)
        if self.gripper_state is not None:
            data["gripper_state"] = self.gripper_state
        if self.image_obs is not None:
            data["image_obs"] = self.image_obs.to_dict()
        if self.proprioception is not None:
            data["proprioception"] = self.proprioception
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotState":
        image_obs = None
        if "image_obs" in data and data["image_obs"] is not None:
            image_obs = ImageContent.from_dict(data["image_obs"])
        return cls(
            joint_positions=np.array(data["joint_positions"]) if "joint_positions" in data else None,
            joint_velocities=np.array(data["joint_velocities"]) if "joint_velocities" in data else None,
            end_effector_pose=np.array(data["end_effector_pose"]) if "end_effector_pose" in data else None,
            gripper_state=data.get("gripper_state"),
            image_obs=image_obs,
            proprioception=data.get("proprioception"),
        )


@dataclass(frozen=True, slots=True)
class RobotAction(ModalityContent):
    """
    Robot action command.

    Attributes:
        joint_velocities: Target joint velocities [n_joints]
        end_effector_delta: Delta position/orientation for end effector [6 or 7]
        gripper_action: Gripper command (0=close, 1=open)
        raw_action: Raw action vector if using learned policy
    """
    joint_velocities: torch.Tensor | np.ndarray | None = None
    end_effector_delta: torch.Tensor | np.ndarray | None = None
    gripper_action: float | None = None
    raw_action: torch.Tensor | np.ndarray | None = None
    modality: Modality = field(default=Modality.ROBOT_ACTION, init=False)

    def to_tensor(self) -> torch.Tensor:
        if self.raw_action is not None:
            if isinstance(self.raw_action, np.ndarray):
                return torch.from_numpy(self.raw_action)
            return self.raw_action
        parts = []
        if self.joint_velocities is not None:
            jv = self.joint_velocities
            if isinstance(jv, np.ndarray):
                jv = torch.from_numpy(jv)
            parts.append(jv.flatten())
        if self.end_effector_delta is not None:
            ee = self.end_effector_delta
            if isinstance(ee, np.ndarray):
                ee = torch.from_numpy(ee)
            parts.append(ee.flatten())
        if self.gripper_action is not None:
            parts.append(torch.tensor([self.gripper_action]))
        if not parts:
            raise NoActionDataError()
        return torch.cat(parts)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"type": "robot_action"}
        if self.joint_velocities is not None:
            jv = self.joint_velocities
            data["joint_velocities"] = jv.tolist() if hasattr(jv, "tolist") else list(jv)
        if self.end_effector_delta is not None:
            ee = self.end_effector_delta
            data["end_effector_delta"] = ee.tolist() if hasattr(ee, "tolist") else list(ee)
        if self.gripper_action is not None:
            data["gripper_action"] = self.gripper_action
        if self.raw_action is not None:
            ra = self.raw_action
            data["raw_action"] = ra.tolist() if hasattr(ra, "tolist") else list(ra)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotAction":
        return cls(
            joint_velocities=np.array(data["joint_velocities"]) if "joint_velocities" in data else None,
            end_effector_delta=np.array(data["end_effector_delta"]) if "end_effector_delta" in data else None,
            gripper_action=data.get("gripper_action"),
            raw_action=np.array(data["raw_action"]) if "raw_action" in data else None,
        )


@dataclass(frozen=True, slots=True)
class RobotTrajectory(ModalityContent):
    """
    A sequence of robot states and actions.

    Attributes:
        states: List of RobotState observations
        actions: List of RobotAction commands (len = len(states) - 1 typically)
        rewards: Optional reward signal at each step
        metadata: Additional trajectory metadata (task description, etc.)
    """
    states: tuple[RobotState, ...] = field(default_factory=tuple)
    actions: tuple[RobotAction, ...] = field(default_factory=tuple)
    rewards: tuple[float, ...] | None = None
    metadata: Dict[str, Any] | None = None
    modality: Modality = field(default=Modality.ROBOT_ACTION, init=False)

    @property
    def length(self) -> int:
        return len(self.states)

    def to_tensor(self) -> torch.Tensor:
        """Stack all state tensors into [T, state_dim]."""
        if not self.states:
            raise EmptyTrajectoryError()
        return torch.stack([s.to_tensor() for s in self.states])

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "type": "robot_trajectory",
            "states": [s.to_dict() for s in self.states],
            "actions": [a.to_dict() for a in self.actions],
        }
        if self.rewards is not None:
            data["rewards"] = list(self.rewards)
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobotTrajectory":
        return cls(
            states=tuple(RobotState.from_dict(s) for s in data.get("states", [])),
            actions=tuple(RobotAction.from_dict(a) for a in data.get("actions", [])),
            rewards=tuple(data["rewards"]) if "rewards" in data else None,
            metadata=data.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class MultimodalContent(ModalityContent):
    """
    Combined content from multiple modalities.

    Example: Image + text prompt for VLM, audio + video for multimodal understanding
    """
    contents: tuple[ModalityContent, ...] = field(default_factory=tuple)
    modality: Modality = field(default=Modality.MULTIMODAL, init=False)

    def __post_init__(self):
        if not self.contents:
            raise MultimodalContentRequiredError()

    def get_by_modality(self, modality: Modality) -> List[ModalityContent]:
        """Get all content items of a specific modality."""
        return [c for c in self.contents if c.modality == modality]

    def get_text(self) -> TextContent | None:
        """Get text content if present."""
        texts = self.get_by_modality(Modality.TEXT)
        return texts[0] if texts else None

    def get_image(self) -> ImageContent | None:
        """Get image content if present."""
        images = self.get_by_modality(Modality.IMAGE)
        return images[0] if images else None

    def get_audio(self) -> AudioContent | None:
        """Get audio content if present."""
        audios = self.get_by_modality(Modality.AUDIO)
        return audios[0] if audios else None

    def get_video(self) -> VideoContent | None:
        """Get video content if present."""
        videos = self.get_by_modality(Modality.VIDEO)
        return videos[0] if videos else None

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError("MultimodalContent requires adapter-specific encoding")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "multimodal",
            "contents": [c.to_dict() for c in self.contents],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalContent":
        contents = []
        for item in data.get("contents", []):
            item_type = item.get("type", "text")
            if item_type == "text":
                contents.append(TextContent.from_dict(item))
            elif item_type == "image":
                contents.append(ImageContent.from_dict(item))
            elif item_type == "audio":
                contents.append(AudioContent.from_dict(item))
            elif item_type == "video":
                contents.append(VideoContent.from_dict(item))
            elif item_type == "robot_state":
                contents.append(RobotState.from_dict(item))
            elif item_type == "robot_action":
                contents.append(RobotAction.from_dict(item))
            elif item_type == "robot_trajectory":
                contents.append(RobotTrajectory.from_dict(item))
        return cls(contents=tuple(contents))


# Type alias for any content type
ContentType = Union[
    str,  # Backward compatible with plain strings
    TextContent,
    AudioContent,
    VideoContent,
    ImageContent,
    RobotState,
    RobotAction,
    RobotTrajectory,
    MultimodalContent,
]


def wrap_content(content: ContentType) -> ModalityContent:
    """Convert raw content to ModalityContent, wrapping strings as TextContent."""
    if isinstance(content, str):
        return TextContent(text=content)
    if isinstance(content, ModalityContent):
        return content
    raise TypeError(f"Cannot wrap content of type {type(content)}")
