"""
Base modality types and text content for Wisent.

Contains the Modality enum, ModalityContent base class, and TextContent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict
from enum import Enum, auto

import torch


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
