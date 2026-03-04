"""
Media content types: Audio, Image, and Video.

Extracted from wisent.core.primitives.models.modalities.__init__ to keep files under 300 lines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import torch
import numpy as np

from wisent.core.primitives.models.modalities.text_content import Modality, ModalityContent
from wisent.core.utils.infra_tools.errors import (
    NoWaveformDataError,
    NoPixelDataError,
    NoFrameDataError,
    InvalidValueError,
)


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
    sample_rate: Optional[int] = None
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
            sample_rate=data.get("sample_rate"),
            file_path=data.get("file_path"),
        )

    @classmethod
    def from_file(cls, file_path: str | Path, sample_rate: int = None) -> "AudioContent":
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
    fps: Optional[float] = None
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
            fps=data.get("fps"),
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
