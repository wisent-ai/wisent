"""
Video adapter for video understanding and generation steering.

Supports models like VideoMAE, TimeSformer, and video generation models.
Enables contrastive steering for:
- Video content safety
- Action/behavior steering in generated video
- Style and motion control
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import numpy as np

from wisent.core.adapters.base import (
    BaseAdapter,
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.modalities import (
    Modality,
    VideoContent,
    TextContent,
)
from wisent.core.errors import UnknownTypeError
from wisent.core.activations.core.atoms import LayerActivations

__all__ = ["VideoAdapter"]

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
    frame_mode: str = "all"
    keyframe_interval: int = 4
    temporal_decay_rate: float = 0.9


class VideoAdapter(BaseAdapter[VideoContent, Union[torch.Tensor, str]]):
    """
    Adapter for video model steering.

    Supports various video models:
    - VideoMAE (video encoding): Steer video representations
    - TimeSformer (video classification): Steer classification behavior
    - Video generation models: Steer synthesis style/content

    Example:
        >>> adapter = VideoAdapter(model_name="MCG-NJU/videomae-base")
        >>> video = VideoContent.from_file("action.mp4")
        >>> activations = adapter.extract_activations(video)
        >>> # Steer toward safe actions
        >>> output = adapter.generate(video, steering_vectors=safe_vectors)
    """

    name = "video"
    modality = Modality.VIDEO

    # Supported model types
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
        num_frames: int = 16,
        frame_sample_rate: int = 4,
        **kwargs: Any,
    ):
        """
        Initialize the video adapter.

        Args:
            model_name: HuggingFace model identifier
            model: Pre-loaded model
            processor: Pre-loaded processor/feature extractor
            device: Target device
            model_type: Force model type detection
            num_frames: Number of frames to sample from video
            frame_sample_rate: Sample every N frames
            **kwargs: Additional model loading arguments
        """
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
                model = VideoMAEModel.from_pretrained(
                    self.model_name,
                    **self._kwargs,
                )
            elif model_type == self.MODEL_TYPE_TIMESFORMER:
                from transformers import TimesformerModel
                model = TimesformerModel.from_pretrained(
                    self.model_name,
                    **self._kwargs,
                )
            elif model_type == self.MODEL_TYPE_VIVIT:
                from transformers import VivitModel
                model = VivitModel.from_pretrained(
                    self.model_name,
                    **self._kwargs,
                )
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    **self._kwargs,
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
                        self.model_name,
                        trust_remote_code=True,
                    )
            except Exception as e:
                raise AdapterError(f"Failed to load processor: {e}")

        return self._processor

    def _sample_frames(self, video: VideoContent) -> torch.Tensor:
        """
        Sample frames from video for model input.

        Args:
            video: Video content

        Returns:
            Sampled frames tensor [num_frames, C, H, W]
        """
        frames = video.to_tensor()  # [T, C, H, W]
        total_frames = frames.shape[0]

        if total_frames <= self.num_frames:
            # Repeat frames if video is too short
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Sample uniformly
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)

        return frames[indices]

    def _resolve_encoder_layers(self) -> List[nn.Module]:
        """Find encoder/transformer layers in the model."""
        if self._encoder_layers is not None:
            return self._encoder_layers

        m = self.model

        candidates = [
            "encoder.layer",
            "encoder.layers",
            "videomae.encoder.layer",
            "timesformer.encoder.layer",
            "vivit.encoder.layer",
            "transformer.layers",
            "blocks",
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
        """
        Encode video to latent representation.

        Args:
            content: Video content

        Returns:
            Video embeddings [batch, num_patches, hidden_dim]
        """
        frames = self._sample_frames(content)

        # Process frames
        if frames.dim() == 4:  # [T, C, H, W]
            frames = frames.unsqueeze(0)  # [1, T, C, H, W]

        # Convert to list of PIL images or numpy arrays for processor
        frames_list = []
        for i in range(frames.shape[1]):
            frame = frames[0, i]  # [C, H, W]
            if frame.max() <= 1.0:
                frame = (frame * 255).byte()
            frame_np = frame.permute(1, 2, 0).numpy()  # [H, W, C]
            frames_list.append(frame_np)

        inputs = self.processor(frames_list, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            return outputs.hidden_states[-1]

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation.

        For encoder-only models, returns the latent itself.
        For generation models, would produce video frames.

        Args:
            latent: Encoded representation

        Returns:
            Decoded output (classification logits or video frames)
        """
        # Most video encoders don't have a decoder
        # Return the pooled representation
        return latent.mean(dim=1)  # Pool over patches

    def get_intervention_points(self) -> List[InterventionPoint]:
        """Get available intervention points in the video model."""
        points = []

        encoder_layers = self._resolve_encoder_layers()
        for i, _ in enumerate(encoder_layers):
            recommended = (len(encoder_layers) // 3) <= i <= (2 * len(encoder_layers) // 3)
            points.append(
                InterventionPoint(
                    name=f"encoder.{i}",
                    module_path=f"{self._encoder_path}.{i}",
                    description=f"Encoder layer {i}",
                    recommended=recommended,
                )
            )

        return points

    def extract_activations(
        self,
        content: VideoContent,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """
        Extract activations from video model layers.

        Args:
            content: Video input
            layers: Layer names to extract (None = all)

        Returns:
            LayerActivations with per-layer tensors
        """
        all_points = {ip.name: ip for ip in self.get_intervention_points()}
        target_layers = layers if layers else list(all_points.keys())

        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                activations[layer_name] = output.detach().cpu()
            return hook

        try:
            for layer_name in target_layers:
                if layer_name not in all_points:
                    continue
                ip = all_points[layer_name]
                module = self._get_module_by_path(ip.module_path)
                if module is not None:
                    handle = module.register_forward_hook(make_hook(layer_name))
                    hooks.append(handle)

            _ = self.encode(content)

        finally:
            for handle in hooks:
                handle.remove()

        return LayerActivations(activations)

    def _create_temporal_steering_hook(
        self,
        vector: torch.Tensor,
        config: VideoSteeringConfig,
        num_frames: int,
    ):
        """
        Create a hook with temporal-aware steering.

        Args:
            vector: Steering vector
            config: Video steering configuration
            num_frames: Number of frames in the video

        Returns:
            Hook function
        """
        def hook(module: nn.Module, input: tuple, output: torch.Tensor) -> torch.Tensor:
            v = vector.to(output.device, output.dtype)

            if config.normalize:
                v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)

            # Handle temporal modes
            if config.frame_mode == "keyframes":
                # Create mask for keyframes
                mask = torch.zeros(num_frames, device=output.device)
                mask[::config.keyframe_interval] = 1.0
                v = v * mask.view(-1, 1, 1) if v.dim() >= 3 else v

            elif config.frame_mode == "temporal_decay":
                # Decay strength over time
                decay = torch.tensor(
                    [config.temporal_decay_rate ** i for i in range(num_frames)],
                    device=output.device,
                )
                v = v * decay.view(-1, 1, 1) if v.dim() >= 3 else v * decay.mean()

            elif config.frame_mode == "temporal_ramp":
                # Increase strength over time
                ramp = torch.linspace(0.1, 1.0, num_frames, device=output.device)
                v = v * ramp.view(-1, 1, 1) if v.dim() >= 3 else v * ramp.mean()

            v = v * config.scale

            while v.dim() < output.dim():
                v = v.unsqueeze(0)

            return output + v

        return hook

    def forward_with_steering(
        self,
        content: VideoContent,
        steering_vectors: LayerActivations,
        config: SteeringConfig | VideoSteeringConfig | None = None,
    ) -> torch.Tensor:
        """
        Process video with steering applied.

        Args:
            content: Video input
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Returns:
            Steered video embeddings
        """
        config = config or VideoSteeringConfig()

        with self._steering_hooks(steering_vectors, config):
            return self.encode(content)

    def _generate_unsteered(
        self,
        content: VideoContent,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Generate output without steering."""
        return self.encode(content)

    def extract_frame_activations(
        self,
        content: VideoContent,
        layers: List[str] | None = None,
    ) -> Dict[int, LayerActivations]:
        """
        Extract activations per frame (for detailed temporal analysis).

        Args:
            content: Video input
            layers: Layer names to extract

        Returns:
            Dict mapping frame index to LayerActivations
        """
        frames = self._sample_frames(content)
        frame_activations = {}

        for i in range(frames.shape[0]):
            # Create single-frame video content
            single_frame = VideoContent(
                frames=frames[i:i+1],
                fps=content.fps,
            )
            frame_activations[i] = self.extract_activations(single_frame, layers)

        return frame_activations

    def compute_temporal_steering_vector(
        self,
        positive_video: VideoContent,
        negative_video: VideoContent,
        layer: str,
        aggregation: str = "mean",
    ) -> torch.Tensor:
        """
        Compute a steering vector from positive/negative video pair.

        Args:
            positive_video: Desired video behavior
            negative_video: Undesired video behavior
            layer: Layer to extract from
            aggregation: How to aggregate across frames ("mean", "last", "first")

        Returns:
            Steering vector
        """
        pos_acts = self.extract_activations(positive_video, [layer])
        neg_acts = self.extract_activations(negative_video, [layer])

        pos_tensor = pos_acts[layer]
        neg_tensor = neg_acts[layer]

        # Aggregate across sequence dimension
        if aggregation == "mean":
            pos_agg = pos_tensor.mean(dim=1)
            neg_agg = neg_tensor.mean(dim=1)
        elif aggregation == "last":
            pos_agg = pos_tensor[:, -1]
            neg_agg = neg_tensor[:, -1]
        elif aggregation == "first":
            pos_agg = pos_tensor[:, 0]
            neg_agg = neg_tensor[:, 0]
        else:
            raise UnknownTypeError(entity_type="aggregation", value=aggregation, valid_values=["mean", "max", "last", "first"])

        return pos_agg.mean(dim=0) - neg_agg.mean(dim=0)
