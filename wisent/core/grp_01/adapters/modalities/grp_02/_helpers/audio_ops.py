"""
Audio adapter operations: decode, transcribe, steering, activations.

Extracted from audio.py to keep files under 300 lines.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Union

import torch
import torch.nn as nn

from wisent.core.adapters.base import (
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.modalities import AudioContent
from wisent.core.activations.core.atoms import LayerActivations


class AudioOpsMixin:
    """Mixin with decode, transcribe, steering, and activation methods."""

    def decode(self, latent: torch.Tensor) -> str:
        """Decode latent representation to text (for ASR models)."""
        model_type = self._detect_model_type()
        if model_type == self.MODEL_TYPE_WHISPER:
            with torch.no_grad():
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                generated_ids = self.model.generate(
                    encoder_outputs={"last_hidden_state": latent},
                    max_new_tokens=448,
                )
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        elif model_type in (self.MODEL_TYPE_WAV2VEC, self.MODEL_TYPE_HUBERT):
            with torch.no_grad():
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                if hasattr(self.model, "lm_head"):
                    logits = self.model.lm_head(latent)
                else:
                    logits = latent
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = self.processor.batch_decode(predicted_ids)[0]
                transcription = self._clean_ctc_output(transcription)
            return transcription
        elif model_type == self.MODEL_TYPE_ENCODEC:
            with torch.no_grad():
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                codes = latent.argmax(dim=-1) if latent.shape[-1] > 1 else latent.squeeze(-1)
                return f"[AudioCodes: shape={codes.shape}, min={codes.min().item()}, max={codes.max().item()}]"
        else:
            with torch.no_grad():
                if latent.dim() == 2:
                    latent = latent.unsqueeze(0)
                if hasattr(self.model, "lm_head"):
                    logits = self.model.lm_head(latent)
                    predicted_ids = torch.argmax(logits, dim=-1)
                    return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                if hasattr(self.model, "generate"):
                    generated_ids = self.model.generate(
                        encoder_outputs={"last_hidden_state": latent},
                        max_new_tokens=256,
                    )
                    return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return f"[Latent: shape={latent.shape}, dtype={latent.dtype}]"

    def _clean_ctc_output(self, text: str) -> str:
        """Clean CTC decoder output by removing artifacts."""
        text = re.sub(r'\[PAD\]|\[UNK\]|<pad>|<unk>|\|', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def get_intervention_points(self) -> List[InterventionPoint]:
        """Get available intervention points in the audio model."""
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
        """Extract activations from audio model layers."""
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

    def forward_with_steering(
        self,
        content: AudioContent,
        steering_vectors: LayerActivations,
        config: SteeringConfig | None = None,
    ) -> str:
        """Process audio with steering applied."""
        config = config or SteeringConfig()
        with self._steering_hooks(steering_vectors, config):
            embedding = self.encode(content)
            return self.decode(embedding)

    def _generate_unsteered(self, content: AudioContent, **kwargs: Any) -> str:
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
        """Transcribe audio to text (convenience method for ASR)."""
        model_type = self._detect_model_type()
        if model_type != self.MODEL_TYPE_WHISPER:
            raise AdapterError("transcribe() only supported for Whisper models")
        waveform = content.to_tensor()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        inputs = self.processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=self.sample_rate, return_tensors="pt",
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
