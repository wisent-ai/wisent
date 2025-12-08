"""
Audio adapter for speech and audio model steering.

Supports models like Whisper, Wav2Vec2, and audio generation models.
Enables contrastive steering for:
- Speech emotion/tone control
- Audio content moderation
- Voice style steering
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
from wisent.core.utils.device import preferred_dtype

__all__ = ["AudioAdapter"]

logger = logging.getLogger(__name__)


class AudioAdapter(BaseAdapter[AudioContent, Union[str, torch.Tensor]]):
    """
    Adapter for audio model steering.

    Supports various audio models:
    - Whisper (speech-to-text): Steer transcription behavior
    - Wav2Vec2 (audio encoding): Steer audio representations
    - Audio generation models: Steer synthesis style/tone

    Example:
        >>> adapter = AudioAdapter(model_name="openai/whisper-large-v3")
        >>> audio = AudioContent.from_file("speech.wav")
        >>> activations = adapter.extract_activations(audio)
        >>> # Steer toward calm speech patterns
        >>> output = adapter.generate(audio, steering_vectors=calm_vectors)
    """

    name = "audio"
    modality = Modality.AUDIO

    # Supported model types
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
        sample_rate: int = 16000,
        **kwargs: Any,
    ):
        """
        Initialize the audio adapter.

        Args:
            model_name: HuggingFace model identifier (e.g., "openai/whisper-large-v3")
            model: Pre-loaded model
            processor: Pre-loaded processor/feature extractor
            device: Target device
            model_type: Force model type detection ("whisper", "wav2vec2", etc.)
            sample_rate: Target sample rate for audio processing
            **kwargs: Additional model loading arguments
        """
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

        # Try to detect from model class
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
                # Use centralized preferred_dtype for consistency
                model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=preferred_dtype(self.device),
                    **self._kwargs,
                )
            elif model_type in (self.MODEL_TYPE_WAV2VEC, self.MODEL_TYPE_HUBERT):
                from transformers import Wav2Vec2Model, HubertModel
                model_cls = HubertModel if model_type == self.MODEL_TYPE_HUBERT else Wav2Vec2Model
                model = model_cls.from_pretrained(
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
                        self.model_name,
                        trust_remote_code=True,
                    )
            except Exception as e:
                raise AdapterError(f"Failed to load processor: {e}")

        return self._processor

    def _resolve_encoder_layers(self) -> List[nn.Module]:
        """Find encoder layers in the model."""
        if self._encoder_layers is not None:
            return self._encoder_layers

        m = self.model

        # Common paths for encoder layers
        candidates = [
            "encoder.layers",
            "model.encoder.layers",
            "encoder.transformer.layers",
            "wav2vec2.encoder.layers",
            "hubert.encoder.layers",
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

        # Fallback: no encoder layers found
        self._encoder_layers = []
        return self._encoder_layers

    def _resolve_decoder_layers(self) -> List[nn.Module]:
        """Find decoder layers (for seq2seq models like Whisper)."""
        if self._decoder_layers is not None:
            return self._decoder_layers

        m = self.model

        candidates = [
            "decoder.layers",
            "model.decoder.layers",
        ]

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
        """
        Encode audio to latent representation.

        Args:
            content: Audio content with waveform or file path

        Returns:
            Audio embeddings [batch, time_steps, hidden_dim]
        """
        # Get waveform
        waveform = content.to_tensor()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add batch dimension

        # Ensure correct sample rate
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

        # Process through feature extractor
        model_type = self._detect_model_type()

        if model_type == self.MODEL_TYPE_WHISPER:
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                encoder_outputs = self.model.model.encoder(**inputs)
                return encoder_outputs.last_hidden_state

        elif model_type in (self.MODEL_TYPE_WAV2VEC, self.MODEL_TYPE_HUBERT):
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                return outputs.last_hidden_state

        else:
            # Generic approach
            inputs = self.processor(
                waveform.squeeze(0).numpy(),
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                if hasattr(outputs, "last_hidden_state"):
                    return outputs.last_hidden_state
                return outputs.hidden_states[-1]

    def decode(self, latent: torch.Tensor) -> str:
        """
        Decode latent representation to text (for ASR models).

        Args:
            latent: Encoder output tensor [batch, time_steps, hidden_dim]

        Returns:
            Transcribed text
        """
        model_type = self._detect_model_type()

        if model_type == self.MODEL_TYPE_WHISPER:
            # Generate from encoder output
            with torch.no_grad():
                # Ensure proper shape [batch, seq, hidden]
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                
                generated_ids = self.model.generate(
                    encoder_outputs={"last_hidden_state": latent},
                    max_new_tokens=448,
                )
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        elif model_type in (self.MODEL_TYPE_WAV2VEC, self.MODEL_TYPE_HUBERT):
            # Wav2Vec2/HuBERT CTC decoding
            with torch.no_grad():
                # Ensure proper shape
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                
                # For CTC models, we need the logits from the LM head
                # If we have raw hidden states, pass through lm_head if available
                if hasattr(self.model, "lm_head"):
                    logits = self.model.lm_head(latent)
                else:
                    # Latent is already logits
                    logits = latent
                
                # Greedy CTC decoding
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode with processor
                transcription = self.processor.batch_decode(predicted_ids)[0]
                
                # Clean up CTC output (remove duplicates and blanks)
                # The processor should handle this, but clean up any remaining artifacts
                transcription = self._clean_ctc_output(transcription)
                
            return transcription

        elif model_type == self.MODEL_TYPE_ENCODEC:
            # EnCodec produces audio codes, not text
            # Return a representation of the audio codes
            with torch.no_grad():
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                # EnCodec latents are typically audio codes
                # Convert to string representation
                codes = latent.argmax(dim=-1) if latent.shape[-1] > 1 else latent.squeeze(-1)
                return f"[AudioCodes: shape={codes.shape}, min={codes.min().item()}, max={codes.max().item()}]"

        else:
            # Generic fallback: try common decoding patterns
            with torch.no_grad():
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                
                # Try to find and use lm_head
                if hasattr(self.model, "lm_head"):
                    logits = self.model.lm_head(latent)
                    predicted_ids = torch.argmax(logits, dim=-1)
                    return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                # Try generate if available
                if hasattr(self.model, "generate"):
                    generated_ids = self.model.generate(
                        encoder_outputs={"last_hidden_state": latent},
                        max_new_tokens=256,
                    )
                    return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                # Last resort: return tensor info
                return f"[Latent: shape={latent.shape}, dtype={latent.dtype}]"

    def _clean_ctc_output(self, text: str) -> str:
        """
        Clean CTC decoder output by removing artifacts.
        
        Args:
            text: Raw CTC decoded text
            
        Returns:
            Cleaned transcription
        """
        import re
        
        # Remove [PAD], [UNK], <pad>, <unk> tokens
        text = re.sub(r'\[PAD\]|\[UNK\]|<pad>|<unk>|\|', ' ', text, flags=re.IGNORECASE)
        
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text

    def get_intervention_points(self) -> List[InterventionPoint]:
        """Get available intervention points in the audio model."""
        points = []

        # Encoder layers
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

        # Decoder layers (if present)
        decoder_layers = self._resolve_decoder_layers()
        for i, _ in enumerate(decoder_layers):
            recommended = (len(decoder_layers) // 3) <= i <= (2 * len(decoder_layers) // 3)
            points.append(
                InterventionPoint(
                    name=f"decoder.{i}",
                    module_path=f"{self._decoder_path}.{i}",
                    description=f"Decoder layer {i}",
                    recommended=recommended,
                )
            )

        return points

    def extract_activations(
        self,
        content: AudioContent,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """
        Extract activations from audio model layers.

        Args:
            content: Audio input
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

            # Forward pass
            _ = self.encode(content)

        finally:
            for handle in hooks:
                handle.remove()

        return LayerActivations(activations)

    def forward_with_steering(
        self,
        content: AudioContent,
        steering_vectors: LayerActivations,
        config: SteeringConfig | None = None,
    ) -> str:
        """
        Process audio with steering applied.

        Args:
            content: Audio input
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Returns:
            Steered output (transcription for ASR, embeddings for encoders)
        """
        config = config or SteeringConfig()

        with self._steering_hooks(steering_vectors, config):
            embedding = self.encode(content)
            return self.decode(embedding)

    def _generate_unsteered(
        self,
        content: AudioContent,
        **kwargs: Any,
    ) -> str:
        """Generate output without steering."""
        embedding = self.encode(content)
        return self.decode(embedding)

    def transcribe(
        self,
        content: AudioContent,
        steering_vectors: LayerActivations | None = None,
        config: SteeringConfig | None = None,
        language: str | None = None,
        task: str = "transcribe",
    ) -> str:
        """
        Transcribe audio to text (convenience method for ASR).

        Args:
            content: Audio input
            steering_vectors: Optional steering vectors
            config: Steering configuration
            language: Target language code
            task: "transcribe" or "translate"

        Returns:
            Transcribed text
        """
        model_type = self._detect_model_type()
        if model_type != self.MODEL_TYPE_WHISPER:
            raise AdapterError("transcribe() only supported for Whisper models")

        waveform = content.to_tensor()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        inputs = self.processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        generate_kwargs = {}
        if language:
            generate_kwargs["language"] = language
        generate_kwargs["task"] = task

        if steering_vectors is not None:
            config = config or SteeringConfig()
            with self._steering_hooks(steering_vectors, config):
                with torch.no_grad():
                    generated_ids = self.model.generate(**inputs, **generate_kwargs)
        else:
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **generate_kwargs)

        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
