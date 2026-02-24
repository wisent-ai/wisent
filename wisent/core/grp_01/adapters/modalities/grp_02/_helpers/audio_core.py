"""
Core audio adapter: initialization, detection, model loading, encoding.

Extracted from audio.py to keep files under 300 lines.
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
    AudioContent,
    TextContent,
)
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.utils import preferred_dtype
from wisent.core.constants import AUDIO_SAMPLE_RATE

logger = logging.getLogger(__name__)


class AudioAdapterCore(BaseAdapter[AudioContent, Union[str, torch.Tensor]]):
    """
    Core adapter for audio model steering.

    Contains initialization, model type detection, model loading,
    processor, layer resolution, and encode methods.
    """

    name = "audio"
    modality = Modality.AUDIO

    MODEL_TYPE_WHISPER = "whisper"
    MODEL_TYPE_WAV2VEC = "wav2vec2"
    MODEL_TYPE_HUBERT = "hubert"
    MODEL_TYPE_ENCODEC = "encodec"
    MODEL_TYPE_GENERIC = "generic"

    def __init__(
        self,
        model_name: str | None = None,
        model: nn.Module | None = None,
        processor: Any | None = None,
        device: str | None = None,
        model_type: str | None = None,
        sample_rate: int = AUDIO_SAMPLE_RATE,
        **kwargs: Any,
    ):
        super().__init__(model=model, model_name=model_name, device=device, **kwargs)
        self._processor = processor
        self._model_type = model_type
        self.sample_rate = sample_rate
        self._encoder_layers: List[nn.Module] | None = None
        self._decoder_layers: List[nn.Module] | None = None
        self._hidden_size: int | None = None

    def _detect_model_type(self) -> str:
        """Detect the type of audio model."""
        if self._model_type:
            return self._model_type
        name_lower = (self.model_name or "").lower()
        if "whisper" in name_lower:
            return self.MODEL_TYPE_WHISPER
        elif "wav2vec" in name_lower:
            return self.MODEL_TYPE_WAV2VEC
        elif "hubert" in name_lower:
            return self.MODEL_TYPE_HUBERT
        elif "encodec" in name_lower:
            return self.MODEL_TYPE_ENCODEC
        model_class = self.model.__class__.__name__.lower()
        if "whisper" in model_class:
            return self.MODEL_TYPE_WHISPER
        elif "wav2vec" in model_class:
            return self.MODEL_TYPE_WAV2VEC
        elif "hubert" in model_class:
            return self.MODEL_TYPE_HUBERT
        return self.MODEL_TYPE_GENERIC

    def _load_model(self) -> nn.Module:
        """Load the audio model."""
        if self.model_name is None:
            raise AdapterError("model_name is required when model is not provided")
        model_type = self._detect_model_type() if self._model_type is None else self._model_type
        try:
            if model_type == self.MODEL_TYPE_WHISPER:
                from transformers import WhisperForConditionalGeneration
                model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=preferred_dtype(self.device),
                    **self._kwargs,
                )
            elif model_type in (self.MODEL_TYPE_WAV2VEC, self.MODEL_TYPE_HUBERT):
                from transformers import Wav2Vec2Model, HubertModel
                model_cls = HubertModel if model_type == self.MODEL_TYPE_HUBERT else Wav2Vec2Model
                model = model_cls.from_pretrained(self.model_name, **self._kwargs)
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
        """Get the audio processor/feature extractor."""
        if self._processor is None:
            if self.model_name is None:
                raise AdapterError("model_name required for processor loading")
            model_type = self._detect_model_type()
            try:
                if model_type == self.MODEL_TYPE_WHISPER:
                    from transformers import WhisperProcessor
                    self._processor = WhisperProcessor.from_pretrained(self.model_name)
                elif model_type in (self.MODEL_TYPE_WAV2VEC, self.MODEL_TYPE_HUBERT):
                    from transformers import Wav2Vec2Processor
                    self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                else:
                    from transformers import AutoProcessor
                    self._processor = AutoProcessor.from_pretrained(
                        self.model_name, trust_remote_code=True,
                    )
            except Exception as e:
                raise AdapterError(f"Failed to load processor: {e}")
        return self._processor

    def _resolve_encoder_layers(self) -> List[nn.Module]:
        """Find encoder layers in the model."""
        if self._encoder_layers is not None:
            return self._encoder_layers
        m = self.model
        candidates = [
            "encoder.layers", "model.encoder.layers",
            "encoder.transformer.layers",
            "wav2vec2.encoder.layers", "hubert.encoder.layers",
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

    def _resolve_decoder_layers(self) -> List[nn.Module]:
        """Find decoder layers (for seq2seq models like Whisper)."""
        if self._decoder_layers is not None:
            return self._decoder_layers
        m = self.model
        candidates = ["decoder.layers", "model.decoder.layers"]
        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList):
                    self._decoder_layers = list(obj)
                    self._decoder_path = path
                    return self._decoder_layers
            except AttributeError:
                continue
        self._decoder_layers = []
        return self._decoder_layers

    @property
    def hidden_size(self) -> int:
        """Get the model's hidden dimension."""
        if self._hidden_size is not None:
            return self._hidden_size
        config = self.model.config
        self._hidden_size = (
            getattr(config, "hidden_size", None)
            or getattr(config, "d_model", None)
            or getattr(config, "encoder_hidden_size", None)
        )
        if self._hidden_size is None:
            for p in self.model.parameters():
                if p.ndim >= 2:
                    self._hidden_size = int(p.shape[-1])
                    break
        if self._hidden_size is None:
            raise AdapterError("Could not determine hidden size")
        return self._hidden_size

    def encode(self, content: AudioContent) -> torch.Tensor:
        """Encode audio to latent representation."""
        waveform = content.to_tensor()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if content.sample_rate != self.sample_rate:
            try:
                import torchaudio
                resampler = torchaudio.transforms.Resample(
                    content.sample_rate, self.sample_rate
                )
                waveform = resampler(waveform)
            except ImportError:
                logger.warning(
                    f"torchaudio not available for resampling. "
                    f"Input sample rate {content.sample_rate} != target {self.sample_rate}"
                )
        model_type = self._detect_model_type()
        if model_type == self.MODEL_TYPE_WHISPER:
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate, return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                encoder_outputs = self.model.model.encoder(**inputs)
                return encoder_outputs.last_hidden_state
        elif model_type in (self.MODEL_TYPE_WAV2VEC, self.MODEL_TYPE_HUBERT):
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate, return_tensors="pt", padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.last_hidden_state
        else:
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate, return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state
                return outputs.hidden_states[-1]
