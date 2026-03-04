"""
Core video adapter: config, initialization, detection, model loading, encoding.

Extracted from video.py to keep files under 300 lines.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging

import torch
import torch.nn as nn
import numpy as np

from wisent.core.primitives.model_interface.adapters.base import (
    BaseAdapter,
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.primitives.models.modalities import (
    Modality,
    VideoContent,
    TextContent,
)
from wisent.core.utils.infra_tools.errors import UnknownTypeError
from wisent.core.primitives.model_interface.core.activations.core.atoms import LayerActivations

logger = logging.getLogger(__name__)


@dataclass
class VideoSteeringConfig(SteeringConfig):
    """
    Extended steering config for video with temporal handling.

    Attributes:
        frame_mode: How to handle per-frame steering
            - "all": Apply to all frames equally
            - "keyframes": Apply only to key frames (every N frames)
            - "temporal_decay": Decay steering strength over time
            - "temporal_ramp": Increase steering strength over time
        keyframe_interval: Interval for keyframe mode
        temporal_decay_rate: Decay rate for temporal_decay mode
    """
    keyframe_interval: int = field(kw_only=True)
    frame_mode: Optional[str] = None
    temporal_decay_rate: float = 0.9


class VideoAdapterCore(BaseAdapter[VideoContent, Union[torch.Tensor, str]]):
    """
    Core adapter for video model steering.

    Contains initialization, model type detection, model loading,
    processor, frame sampling, layer resolution, and encode/decode methods.
    """

    name = "video"
    modality = Modality.VIDEO

    MODEL_TYPE_VIDEOMAE = "videomae"
    MODEL_TYPE_TIMESFORMER = "timesformer"
    MODEL_TYPE_VIVIT = "vivit"
    MODEL_TYPE_XCLIP = "xclip"
    MODEL_TYPE_GENERIC = "generic"

    def __init__(
        self,
        model_name: str | None = None,
        model: nn.Module | None = None,
        processor: Any | None = None,
        device: str | None = None,
        model_type: str | None = None,
        *,
        num_frames: int,
        frame_sample_rate: int,
        **kwargs: Any,
    ):
        super().__init__(model=model, model_name=model_name, device=device, **kwargs)
        self._processor = processor
        self._model_type = model_type
        self.num_frames = num_frames
        self.frame_sample_rate = frame_sample_rate
        self._encoder_layers: List[nn.Module] | None = None
        self._hidden_size: int | None = None

    def _detect_model_type(self) -> str:
        """Detect the type of video model."""
        if self._model_type:
            return self._model_type
        name_lower = (self.model_name or "").lower()
        if "videomae" in name_lower:
            return self.MODEL_TYPE_VIDEOMAE
        elif "timesformer" in name_lower:
            return self.MODEL_TYPE_TIMESFORMER
        elif "vivit" in name_lower:
            return self.MODEL_TYPE_VIVIT
        elif "xclip" in name_lower or "x-clip" in name_lower:
            return self.MODEL_TYPE_XCLIP
        model_class = self.model.__class__.__name__.lower()
        if "videomae" in model_class:
            return self.MODEL_TYPE_VIDEOMAE
        elif "timesformer" in model_class:
            return self.MODEL_TYPE_TIMESFORMER
        elif "vivit" in model_class:
            return self.MODEL_TYPE_VIVIT
        return self.MODEL_TYPE_GENERIC

    def _load_model(self) -> nn.Module:
        """Load the video model."""
        if self.model_name is None:
            raise AdapterError("model_name is required when model is not provided")
        model_type = self._detect_model_type() if self._model_type is None else self._model_type
        try:
            if model_type == self.MODEL_TYPE_VIDEOMAE:
                from transformers import VideoMAEModel
                model = VideoMAEModel.from_pretrained(self.model_name, **self._kwargs)
            elif model_type == self.MODEL_TYPE_TIMESFORMER:
                from transformers import TimesformerModel
                model = TimesformerModel.from_pretrained(self.model_name, **self._kwargs)
            elif model_type == self.MODEL_TYPE_VIVIT:
                from transformers import VivitModel
                model = VivitModel.from_pretrained(self.model_name, **self._kwargs)
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True, **self._kwargs,
                )
            if self.device:
                model = model.to(self.device)
            return model
        except ImportError as e:
            raise AdapterError(f"Required library not installed: {e}")
        except Exception as e:
            raise AdapterError(f"Failed to load model {self.model_name}: {e}")

    @property
    def processor(self) -> Any:
        """Get the video processor/feature extractor."""
        if self._processor is None:
            if self.model_name is None:
                raise AdapterError("model_name required for processor loading")
            model_type = self._detect_model_type()
            try:
                if model_type == self.MODEL_TYPE_VIDEOMAE:
                    from transformers import VideoMAEImageProcessor
                    self._processor = VideoMAEImageProcessor.from_pretrained(self.model_name)
                elif model_type == self.MODEL_TYPE_TIMESFORMER:
                    from transformers import AutoImageProcessor
                    self._processor = AutoImageProcessor.from_pretrained(self.model_name)
                else:
                    from transformers import AutoProcessor
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_name, trust_remote_code=True,
                    )
            except Exception as e:
                raise AdapterError(f"Failed to load processor: {e}")
        return self._processor

    def _sample_frames(self, video: VideoContent) -> torch.Tensor:
        """Sample frames from video for model input."""
        frames = video.to_tensor()
        total_frames = frames.shape[0]
        if total_frames <= self.num_frames:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        return frames[indices]

    def _resolve_encoder_layers(self) -> List[nn.Module]:
        """Find encoder/transformer layers in the model."""
        if self._encoder_layers is not None:
            return self._encoder_layers
        m = self.model
        candidates = [
            "encoder.layer", "encoder.layers",
            "videomae.encoder.layer", "timesformer.encoder.layer",
            "vivit.encoder.layer", "transformer.layers", "blocks",
        ]
        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList):
                    self._encoder_layers = list(obj)
                    self._encoder_path = path
                    return self._encoder_layers
            except AttributeError:
                continue
        self._encoder_layers = []
        return self._encoder_layers

    @property
    def hidden_size(self) -> int:
        """Get the model's hidden dimension."""
        if self._hidden_size is not None:
            return self._hidden_size
        config = self.model.config
        self._hidden_size = (
            getattr(config, "hidden_size", None)
            or getattr(config, "embed_dim", None)
            or getattr(config, "d_model", None)
        )
        if self._hidden_size is None:
            for p in self.model.parameters():
                if p.ndim >= 2:
                    self._hidden_size = int(p.shape[-1])
                    break
        if self._hidden_size is None:
            raise AdapterError("Could not determine hidden size")
        return self._hidden_size

    def encode(self, content: VideoContent) -> torch.Tensor:
        """Encode video to latent representation."""
        frames = self._sample_frames(content)
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
        frames_list = []
        for i in range(frames.shape[1]):
            frame = frames[0, i]
            if frame.max() <= 1.0:
                frame = (frame * 255).byte()
            frame_np = frame.permute(1, 2, 0).numpy()
            frames_list.append(frame_np)
        inputs = self.processor(frames_list, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs.hidden_states[-1]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent representation (pool over patches)."""
        return latent.mean(dim=1)
