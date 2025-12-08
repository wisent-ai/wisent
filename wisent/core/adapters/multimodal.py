"""
Multimodal adapter for vision-language and other combined modality models.

Supports models like:
- LLaVA, Qwen-VL (vision-language)
- ImageBind (unified embeddings)
- Video-LLMs
- Any model that processes multiple modalities
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn

from wisent.core.adapters.base import (
    BaseAdapter,
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.modalities import (
    Modality,
    ModalityContent,
    MultimodalContent,
    TextContent,
    ImageContent,
    AudioContent,
    VideoContent,
)
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.utils.device import preferred_dtype

__all__ = ["MultimodalAdapter", "MultimodalSteeringConfig"]

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSteeringConfig(SteeringConfig):
    """
    Extended steering config for multimodal models.

    Attributes:
        steer_modalities: Which modality encoders to steer ("all", "text", "vision", "audio")
        cross_modal_steering: Whether to apply steering at cross-attention layers
        fusion_steering: Whether to steer at modality fusion layers
    """
    steer_modalities: str | List[str] = "all"
    cross_modal_steering: bool = True
    fusion_steering: bool = True


class MultimodalAdapter(BaseAdapter[MultimodalContent, Union[str, torch.Tensor]]):
    """
    Adapter for multimodal model steering.

    Supports vision-language models (VLMs), audio-text models, and other
    multimodal architectures. Enables steering at:
    - Individual modality encoders
    - Cross-attention layers
    - Fusion/projection layers
    - Decoder/LLM layers

    Example:
        >>> adapter = MultimodalAdapter(model_name="llava-hf/llava-1.5-7b-hf")
        >>> content = MultimodalContent(contents=(
        ...     ImageContent.from_file("image.jpg"),
        ...     TextContent("What is in this image?")
        ... ))
        >>> output = adapter.generate(content, steering_vectors=safe_vectors)
    """

    name = "multimodal"
    modality = Modality.MULTIMODAL

    # Supported model types
    MODEL_TYPE_LLAVA = "llava"
    MODEL_TYPE_QWEN_VL = "qwen_vl"
    MODEL_TYPE_IDEFICS = "idefics"
    MODEL_TYPE_IMAGEBIND = "imagebind"
    MODEL_TYPE_GENERIC = "generic"

    def __init__(
        self,
        model_name: str | None = None,
        model: nn.Module | None = None,
        processor: Any | None = None,
        device: str | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the multimodal adapter.

        Args:
            model_name: HuggingFace model identifier
            model: Pre-loaded model
            processor: Pre-loaded processor
            device: Target device
            model_type: Force model type detection
            **kwargs: Additional model loading arguments
        """
        super().__init__(model=model, model_name=model_name, device=device, **kwargs)
        self._processor = processor
        self._model_type = model_type
        self._vision_encoder_layers: List[nn.Module] | None = None
        self._language_layers: List[nn.Module] | None = None
        self._cross_attention_layers: List[nn.Module] | None = None
        self._hidden_size: int | None = None

    def _detect_model_type(self) -> str:
        """Detect the type of multimodal model."""
        if self._model_type:
            return self._model_type

        name_lower = (self.model_name or "").lower()

        if "llava" in name_lower:
            return self.MODEL_TYPE_LLAVA
        elif "qwen" in name_lower and ("vl" in name_lower or "visual" in name_lower):
            return self.MODEL_TYPE_QWEN_VL
        elif "idefics" in name_lower:
            return self.MODEL_TYPE_IDEFICS
        elif "imagebind" in name_lower:
            return self.MODEL_TYPE_IMAGEBIND

        if self._model is not None:
            model_class = self._model.__class__.__name__.lower()
            if "llava" in model_class:
                return self.MODEL_TYPE_LLAVA
            elif "qwen" in model_class:
                return self.MODEL_TYPE_QWEN_VL
            elif "idefics" in model_class:
                return self.MODEL_TYPE_IDEFICS

        return self.MODEL_TYPE_GENERIC

    def _load_model(self) -> nn.Module:
        """Load the multimodal model."""
        if self.model_name is None:
            raise AdapterError("model_name is required when model is not provided")

        model_type = self._detect_model_type() if self._model_type is None else self._model_type

        try:
            # Use centralized preferred_dtype for consistency across codebase
            dtype = preferred_dtype(self.device)
            device_map = "auto" if self.device in ("cuda", "auto") else None
            
            if model_type == self.MODEL_TYPE_LLAVA:
                from transformers import LlavaForConditionalGeneration
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    **self._kwargs,
                )
            elif model_type == self.MODEL_TYPE_QWEN_VL:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map=device_map,
                    **self._kwargs,
                )
            elif model_type == self.MODEL_TYPE_IDEFICS:
                from transformers import IdeficsForVisionText2Text
                model = IdeficsForVisionText2Text.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map=device_map,
                    **self._kwargs,
                )
            else:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    **self._kwargs,
                )
                if self.device and self.device not in ("auto",):
                    model = model.to(self.device)

            return model

        except ImportError as e:
            raise AdapterError(f"Required library not installed: {e}")
        except Exception as e:
            raise AdapterError(f"Failed to load model {self.model_name}: {e}")

    @property
    def processor(self) -> Any:
        """Get the multimodal processor."""
        if self._processor is None:
            if self.model_name is None:
                raise AdapterError("model_name required for processor loading")

            try:
                from transformers import AutoProcessor
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )
            except Exception as e:
                raise AdapterError(f"Failed to load processor: {e}")

        return self._processor

    def _resolve_vision_encoder_layers(self) -> List[nn.Module]:
        """Find vision encoder layers."""
        if self._vision_encoder_layers is not None:
            return self._vision_encoder_layers

        m = self.model
        candidates = [
            "vision_tower.vision_model.encoder.layers",
            "vision_model.encoder.layers",
            "visual.transformer.resblocks",
            "vision_encoder.layers",
            "image_encoder.layers",
        ]

        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)):
                    self._vision_encoder_layers = list(obj)
                    self._vision_path = path
                    return self._vision_encoder_layers
            except AttributeError:
                continue

        self._vision_encoder_layers = []
        return self._vision_encoder_layers

    def _resolve_language_layers(self) -> List[nn.Module]:
        """Find language model layers."""
        if self._language_layers is not None:
            return self._language_layers

        m = self.model
        candidates = [
            "language_model.model.layers",
            "model.layers",
            "transformer.h",
            "llm.model.layers",
            "text_model.encoder.layers",
        ]

        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                if isinstance(obj, (nn.ModuleList, list)):
                    self._language_layers = list(obj)
                    self._language_path = path
                    return self._language_layers
            except AttributeError:
                continue

        self._language_layers = []
        return self._language_layers

    @property
    def hidden_size(self) -> int:
        """Get the model's hidden dimension."""
        if self._hidden_size is not None:
            return self._hidden_size

        config = self.model.config

        # Try various config attributes
        for attr in ["hidden_size", "d_model", "text_config.hidden_size"]:
            parts = attr.split(".")
            obj = config
            try:
                for part in parts:
                    obj = getattr(obj, part)
                if isinstance(obj, int):
                    self._hidden_size = obj
                    return self._hidden_size
            except AttributeError:
                continue

        # Infer from parameters
        for p in self.model.parameters():
            if p.ndim >= 2:
                self._hidden_size = int(p.shape[-1])
                return self._hidden_size

        raise AdapterError("Could not determine hidden size")

    def _prepare_inputs(self, content: MultimodalContent) -> Dict[str, Any]:
        """
        Prepare inputs for the multimodal model.

        Args:
            content: Multimodal content with various modality items

        Returns:
            Dict of model inputs
        """
        text = content.get_text()
        image = content.get_image()
        video = content.get_video()

        # Build prompt/inputs based on available content
        text_input = text.text if text else ""

        if image is not None:
            # Load image if from file
            if image.pixels is None and image.file_path:
                from PIL import Image
                pil_image = Image.open(image.file_path).convert("RGB")
            else:
                pixels = image.to_tensor()
                if pixels.max() <= 1.0:
                    pixels = (pixels * 255).byte()
                pil_image = pixels.permute(1, 2, 0).numpy()
                from PIL import Image
                pil_image = Image.fromarray(pil_image)

            inputs = self.processor(
                text=text_input,
                images=pil_image,
                return_tensors="pt",
            )
        elif video is not None:
            frames = video.to_tensor()
            # Convert frames to list of PIL images
            from PIL import Image
            pil_frames = []
            for i in range(frames.shape[0]):
                frame = frames[i]
                if frame.max() <= 1.0:
                    frame = (frame * 255).byte()
                frame_np = frame.permute(1, 2, 0).numpy()
                pil_frames.append(Image.fromarray(frame_np))

            inputs = self.processor(
                text=text_input,
                images=pil_frames,
                return_tensors="pt",
            )
        else:
            inputs = self.processor(
                text=text_input,
                return_tensors="pt",
            )

        # Move to device
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        return inputs

    def encode(self, content: MultimodalContent) -> torch.Tensor:
        """
        Encode multimodal content to latent representation.

        Args:
            content: Multimodal content

        Returns:
            Combined embedding tensor
        """
        inputs = self._prepare_inputs(content)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states"):
                return outputs.hidden_states[-1]
            else:
                raise AdapterError("Model output format not recognized")

    def decode(self, latent: torch.Tensor) -> str:
        """
        Decode latent representation to text output.

        For multimodal models, this generates text from the hidden state
        by passing through the language model head and sampling.

        Args:
            latent: Hidden state tensor [batch, seq_len, hidden_dim]

        Returns:
            Generated text
        """
        model_type = self._detect_model_type()
        
        with torch.no_grad():
            # Ensure proper shape
            if latent.dim() == 2:
                latent = latent.unsqueeze(0)
            
            # Move to model device
            latent = latent.to(self.model.device)
            
            # Try different decoding strategies based on model type
            if model_type == self.MODEL_TYPE_LLAVA:
                return self._decode_llava(latent)
            elif model_type == self.MODEL_TYPE_QWEN_VL:
                return self._decode_qwen_vl(latent)
            elif model_type == self.MODEL_TYPE_IDEFICS:
                return self._decode_idefics(latent)
            else:
                return self._decode_generic(latent)
    
    def _decode_llava(self, latent: torch.Tensor) -> str:
        """Decode for LLaVA models."""
        # LLaVA uses a standard LM head
        if hasattr(self.model, "language_model"):
            lm = self.model.language_model
            if hasattr(lm, "lm_head"):
                logits = lm.lm_head(latent)
                return self._greedy_decode(logits)
        
        return self._decode_generic(latent)
    
    def _decode_qwen_vl(self, latent: torch.Tensor) -> str:
        """Decode for Qwen-VL models."""
        # Qwen-VL uses standard transformer architecture
        if hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(latent)
            return self._greedy_decode(logits)
        
        return self._decode_generic(latent)
    
    def _decode_idefics(self, latent: torch.Tensor) -> str:
        """Decode for IDEFICS models."""
        # IDEFICS uses embed_out for LM head
        if hasattr(self.model, "embed_out"):
            logits = self.model.embed_out(latent)
            return self._greedy_decode(logits)
        elif hasattr(self.model, "lm_head"):
            logits = self.model.lm_head(latent)
            return self._greedy_decode(logits)
        
        return self._decode_generic(latent)
    
    def _decode_generic(self, latent: torch.Tensor) -> str:
        """Generic decoding fallback."""
        # Try common LM head names
        lm_head = None
        for attr in ["lm_head", "embed_out", "output_projection", "head"]:
            if hasattr(self.model, attr):
                lm_head = getattr(self.model, attr)
                break
            # Check in language_model submodule
            if hasattr(self.model, "language_model"):
                lm = self.model.language_model
                if hasattr(lm, attr):
                    lm_head = getattr(lm, attr)
                    break
        
        if lm_head is not None:
            logits = lm_head(latent)
            return self._greedy_decode(logits)
        
        # Last resort: sample from latent directly (not ideal)
        # Return info about the latent instead
        return f"[Latent decoded: shape={latent.shape}, mean={latent.mean().item():.4f}]"
    
    def _greedy_decode(self, logits: torch.Tensor, max_length: int = 256) -> str:
        """
        Perform greedy decoding from logits.
        
        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            max_length: Maximum generation length
            
        Returns:
            Decoded text string
        """
        # Get token predictions
        if logits.dim() == 3:
            # Take last position for next token prediction
            next_token_logits = logits[:, -1, :]
        else:
            next_token_logits = logits
        
        # Greedy selection
        predicted_ids = torch.argmax(next_token_logits, dim=-1)
        
        # Handle batch dimension
        if predicted_ids.dim() == 0:
            predicted_ids = predicted_ids.unsqueeze(0)
        if predicted_ids.dim() == 1:
            predicted_ids = predicted_ids.unsqueeze(0)
        
        # Decode tokens
        try:
            text = self.processor.decode(predicted_ids[0], skip_special_tokens=True)
        except Exception:
            # Fallback to tokenizer if processor doesn't have decode
            if hasattr(self.processor, "tokenizer"):
                text = self.processor.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
            else:
                text = f"[Token IDs: {predicted_ids[0].tolist()[:10]}...]"
        
        return text

    def get_intervention_points(self) -> List[InterventionPoint]:
        """Get available intervention points across all modality encoders."""
        points = []

        # Vision encoder layers
        vision_layers = self._resolve_vision_encoder_layers()
        for i, _ in enumerate(vision_layers):
            recommended = i >= len(vision_layers) // 2
            points.append(
                InterventionPoint(
                    name=f"vision.{i}",
                    module_path=f"{self._vision_path}.{i}",
                    description=f"Vision encoder layer {i}",
                    recommended=recommended,
                )
            )

        # Language model layers
        language_layers = self._resolve_language_layers()
        for i, _ in enumerate(language_layers):
            recommended = (len(language_layers) // 3) <= i <= (2 * len(language_layers) // 3)
            points.append(
                InterventionPoint(
                    name=f"language.{i}",
                    module_path=f"{self._language_path}.{i}",
                    description=f"Language model layer {i}",
                    recommended=recommended,
                )
            )

        # Look for projection/fusion layers
        m = self.model
        projection_candidates = [
            ("multi_modal_projector", "Multimodal projector"),
            ("mm_projector", "MM projector"),
            ("vision_projection", "Vision projection"),
        ]
        for path, desc in projection_candidates:
            try:
                module = getattr(m, path, None)
                if module is not None:
                    points.append(
                        InterventionPoint(
                            name="projection",
                            module_path=path,
                            description=desc,
                            recommended=True,
                        )
                    )
                    break
            except AttributeError:
                continue

        return points

    def extract_activations(
        self,
        content: MultimodalContent,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """
        Extract activations from multimodal model layers.

        Args:
            content: Multimodal input
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

            inputs = self._prepare_inputs(content)
            with torch.no_grad():
                self.model(**inputs)

        finally:
            for handle in hooks:
                handle.remove()

        return LayerActivations(activations)

    def forward_with_steering(
        self,
        content: MultimodalContent,
        steering_vectors: LayerActivations,
        config: SteeringConfig | MultimodalSteeringConfig | None = None,
    ) -> str:
        """
        Generate output with steering applied.

        Args:
            content: Multimodal input
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Returns:
            Generated text output
        """
        config = config or MultimodalSteeringConfig()

        # Filter steering vectors based on config
        if isinstance(config, MultimodalSteeringConfig):
            if config.steer_modalities != "all":
                modalities = (
                    [config.steer_modalities]
                    if isinstance(config.steer_modalities, str)
                    else config.steer_modalities
                )
                filtered = {}
                for name, vec in steering_vectors.items():
                    modality = name.split(".")[0]
                    if modality in modalities or name == "projection":
                        filtered[name] = vec
                steering_vectors = LayerActivations(filtered)

        inputs = self._prepare_inputs(content)

        with self._steering_hooks(steering_vectors, config):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                )

        # Decode output
        generated = self.processor.decode(outputs[0], skip_special_tokens=True)
        return generated

    def _generate_unsteered(
        self,
        content: MultimodalContent,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """Generate output without steering."""
        inputs = self._prepare_inputs(content)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                **kwargs,
            )

        return self.processor.decode(outputs[0], skip_special_tokens=True)

    def generate(
        self,
        content: MultimodalContent,
        steering_vectors: LayerActivations | None = None,
        config: SteeringConfig | MultimodalSteeringConfig | None = None,
        **generation_kwargs: Any,
    ) -> str:
        """
        Generate text output from multimodal input.

        Args:
            content: Multimodal content (image + text, video + text, etc.)
            steering_vectors: Optional steering vectors
            config: Steering configuration
            **generation_kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if steering_vectors is not None:
            return self.forward_with_steering(content, steering_vectors, config)
        return self._generate_unsteered(content, **generation_kwargs)

    def compute_cross_modal_steering_vector(
        self,
        positive_content: MultimodalContent,
        negative_content: MultimodalContent,
        layer: str,
    ) -> torch.Tensor:
        """
        Compute steering vector from multimodal content pairs.

        Args:
            positive_content: Desired multimodal behavior
            negative_content: Undesired multimodal behavior
            layer: Layer to extract from

        Returns:
            Steering vector
        """
        pos_acts = self.extract_activations(positive_content, [layer])
        neg_acts = self.extract_activations(negative_content, [layer])

        pos_tensor = pos_acts[layer]
        neg_tensor = neg_acts[layer]

        # Mean pool and subtract
        pos_pooled = pos_tensor.mean(dim=1) if pos_tensor.dim() > 2 else pos_tensor.mean(dim=0)
        neg_pooled = neg_tensor.mean(dim=1) if neg_tensor.dim() > 2 else neg_tensor.mean(dim=0)

        return pos_pooled.squeeze() - neg_pooled.squeeze()
