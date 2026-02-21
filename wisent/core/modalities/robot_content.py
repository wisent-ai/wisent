"""Robot and multimodal content types extracted from modalities __init__."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union, List, Any, Dict
import torch
import numpy as np
from wisent.core.modalities.text_content import Modality, ModalityContent, TextContent
from wisent.core.modalities.media_content import AudioContent, ImageContent, VideoContent
from wisent.core.errors import (
    NoStateDataError, NoActionDataError, EmptyTrajectoryError, MultimodalContentRequiredError,
)

@dataclass(frozen=True, slots=True)
class RobotState(ModalityContent):
    """Robot state observation with joint positions, velocities, end effector pose, gripper, image obs."""
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
    """Robot action command with joint velocities, end effector delta, gripper, raw action."""
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
    """A sequence of robot states and actions with optional rewards and metadata."""
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
    """Combined content from multiple modalities (image+text, audio+video, etc.)."""
    contents: tuple[ModalityContent, ...] = field(default_factory=tuple)
    modality: Modality = field(default=Modality.MULTIMODAL, init=False)

    def __post_init__(self):
        if not self.contents:
            raise MultimodalContentRequiredError()

    def get_by_modality(self, modality: Modality) -> List[ModalityContent]:
        """Get all content items of a specific modality."""
        return [c for c in self.contents if c.modality == modality]

    def get_text(self) -> TextContent | None:
        texts = self.get_by_modality(Modality.TEXT)
        return texts[0] if texts else None

    def get_image(self) -> ImageContent | None:
        images = self.get_by_modality(Modality.IMAGE)
        return images[0] if images else None

    def get_audio(self) -> AudioContent | None:
        audios = self.get_by_modality(Modality.AUDIO)
        return audios[0] if audios else None

    def get_video(self) -> VideoContent | None:
        videos = self.get_by_modality(Modality.VIDEO)
        return videos[0] if videos else None

    def to_tensor(self) -> torch.Tensor:
        raise NotImplementedError("MultimodalContent requires adapter-specific encoding")

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "multimodal", "contents": [c.to_dict() for c in self.contents]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalContent":
        contents = []
        type_map = {
            "text": TextContent, "image": ImageContent, "audio": AudioContent,
            "video": VideoContent, "robot_state": RobotState, "robot_action": RobotAction,
            "robot_trajectory": RobotTrajectory,
        }
        for item in data.get("contents", []):
            item_type = item.get("type", "text")
            content_cls = type_map.get(item_type)
            if content_cls is not None:
                contents.append(content_cls.from_dict(item))
        return cls(contents=tuple(contents))

# Type alias for any content type
ContentType = Union[
    str, TextContent, AudioContent, VideoContent, ImageContent,
    RobotState, RobotAction, RobotTrajectory, MultimodalContent,
]

def wrap_content(content: ContentType) -> ModalityContent:
    """Convert raw content to ModalityContent, wrapping strings as TextContent."""
    if isinstance(content, str):
        return TextContent(text=content)
    if isinstance(content, ModalityContent):
        return content
    raise TypeError(f"Cannot wrap content of type {type(content)}")
