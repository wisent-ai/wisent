"""
Text adapter for language model steering.

This adapter wraps the existing WisentModel functionality, providing
backward compatibility while conforming to the unified adapter interface.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from wisent.core.adapters.base import (
    BaseAdapter,
    AdapterError,
    InterventionPoint,
    SteeringConfig,
)
from wisent.core.modalities import (
    Modality,
    TextContent,
    wrap_content,
)
from wisent.core.activations.core.atoms import LayerActivations, RawActivationMap
from wisent.core.utils.device import preferred_dtype

__all__ = ["TextAdapter"]


class TextAdapter(BaseAdapter[TextContent, str]):
    """
    Adapter for text/language model steering.

    Wraps HuggingFace causal language models and provides steering
    at decoder layer outputs. This is the default adapter and maintains
    backward compatibility with the original Wisent API.

    Example:
        >>> adapter = TextAdapter(model_name="meta-llama/Llama-3-8B-Instruct")
        >>> text = TextContent("What is 2+2?")
        >>> activations = adapter.extract_activations(text, layers=["layer.12"])
        >>> output = adapter.generate(text, steering_vectors=my_vectors)
    """

    name = "text"
    modality = Modality.TEXT

    def __init__(
        self,
        model_name: str | None = None,
        model: PreTrainedModel | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the text adapter.

        Args:
            model_name: HuggingFace model identifier or local path
            model: Pre-loaded model (optional)
            tokenizer: Pre-loaded tokenizer (optional)
            device: Target device
            torch_dtype: Model dtype (default: auto-detect)
            **kwargs: Additional arguments passed to model loading
        """
        super().__init__(model=model, model_name=model_name, device=device, **kwargs)
        self._tokenizer = tokenizer
        self._torch_dtype = torch_dtype
        self._decoder_layers: List[nn.Module] | None = None
        self._hidden_size: int | None = None

    def _load_model(self) -> PreTrainedModel:
        """Load the language model."""
        if self.model_name is None:
            raise AdapterError("model_name is required when model is not provided")

        # Determine dtype and device settings
        load_kwargs = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
            "attn_implementation": "eager",
        }

        # Use centralized preferred_dtype for consistency across codebase
        if self._torch_dtype:
            load_kwargs["torch_dtype"] = self._torch_dtype
        else:
            load_kwargs["torch_dtype"] = preferred_dtype(self.device)
        
        if self.device == "mps":
            load_kwargs["device_map"] = "mps"
        elif self.device in ("cuda", "auto"):
            load_kwargs["device_map"] = "auto"

        # Add any extra kwargs
        load_kwargs.update(self._kwargs)

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        # Move to device if device_map wasn't used
        if load_kwargs.get("device_map") is None and self.device:
            model.to(self.device)

        return model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            if self.model_name is None:
                raise AdapterError("model_name required for tokenizer loading")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def _resolve_decoder_layers(self) -> List[nn.Module]:
        """Find the decoder layers in the model."""
        if self._decoder_layers is not None:
            return self._decoder_layers

        m = self.model
        candidates = [
            "layers",
            "model.layers",
            "model.decoder.layers",
            "transformer.h",
            "base_model.model.layers",
            "blocks",
            "model.blocks",
        ]

        for path in candidates:
            obj = m
            try:
                for attr in path.split("."):
                    if attr:
                        obj = getattr(obj, attr)
                if isinstance(obj, nn.ModuleList) or (
                    isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], nn.Module)
                ):
                    self._decoder_layers = list(obj)
                    self._layer_path = path
                    return self._decoder_layers
            except AttributeError:
                continue

        raise AdapterError("Could not find decoder layers in model")

    @property
    def hidden_size(self) -> int:
        """Get the model's hidden dimension."""
        if self._hidden_size is not None:
            return self._hidden_size

        config = self.model.config
        self._hidden_size = getattr(config, "hidden_size", None) or getattr(config, "n_embd", None)

        if self._hidden_size is None:
            # Infer from parameters
            for p in self.model.parameters():
                if p.ndim >= 2:
                    self._hidden_size = int(p.shape[-1])
                    break

        if self._hidden_size is None:
            raise AdapterError("Could not determine hidden size")

        return self._hidden_size

    @property
    def num_layers(self) -> int:
        """Get the number of decoder layers."""
        return len(self._resolve_decoder_layers())

    def encode(self, content: TextContent | str) -> torch.Tensor:
        """
        Encode text to token embeddings.

        Args:
            content: Text content or string

        Returns:
            Token embeddings [batch, seq_len, hidden_dim]
        """
        if isinstance(content, str):
            content = TextContent(text=content)

        # Tokenize
        inputs = self.tokenizer(
            content.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Get embeddings from the model
        with torch.no_grad():
            if hasattr(self.model, "get_input_embeddings"):
                embed_layer = self.model.get_input_embeddings()
                embeddings = embed_layer(inputs["input_ids"])
            else:
                outputs = self.model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[0]

        return embeddings

    def decode(self, latent: torch.Tensor) -> str:
        """
        Decode latent representation back to text.

        For LLMs, this typically means generating from the latent state.

        Args:
            latent: Latent tensor

        Returns:
            Decoded text string
        """
        raise NotImplementedError(
            "Direct latent decoding not supported for LLMs. Use generate() instead."
        )

    def get_intervention_points(self) -> List[InterventionPoint]:
        """
        Get available intervention points (decoder layers).

        Returns:
            List of intervention points for each decoder layer
        """
        layers = self._resolve_decoder_layers()
        points = []

        for i, layer in enumerate(layers):
            # Determine if this is a recommended layer (middle-to-late layers)
            recommended = (len(layers) // 3) <= i <= (2 * len(layers) // 3)

            points.append(
                InterventionPoint(
                    name=f"layer.{i}",
                    module_path=f"{self._layer_path}.{i}",
                    description=f"Decoder layer {i}",
                    recommended=recommended,
                )
            )

        return points

    def extract_activations(
        self,
        content: TextContent | str,
        layers: List[str] | None = None,
    ) -> LayerActivations:
        """
        Extract activations from specified layers.

        Args:
            content: Input text
            layers: Layer names to extract (None = all)

        Returns:
            LayerActivations mapping layer names to tensors
        """
        if isinstance(content, str):
            content = TextContent(text=content)

        # Determine which layers to extract
        all_points = {ip.name: ip for ip in self.get_intervention_points()}
        if layers is None:
            target_layers = list(all_points.keys())
        else:
            target_layers = layers

        # Set up activation storage
        activations: Dict[str, torch.Tensor] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module, input, output):
                # Handle tuple outputs (common in transformers)
                if isinstance(output, tuple):
                    output = output[0]
                # Store activation (detached, on CPU)
                activations[layer_name] = output.detach().cpu()
            return hook

        try:
            # Register hooks
            for layer_name in target_layers:
                if layer_name not in all_points:
                    continue
                ip = all_points[layer_name]
                module = self._get_module_by_path(ip.module_path)
                if module is not None:
                    handle = module.register_forward_hook(make_hook(layer_name))
                    hooks.append(handle)

            # Forward pass
            inputs = self.tokenizer(
                content.text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                self.model(**inputs)

        finally:
            # Clean up hooks
            for handle in hooks:
                handle.remove()

        return LayerActivations(activations)

    def forward_with_steering(
        self,
        content: TextContent | str,
        steering_vectors: LayerActivations,
        config: SteeringConfig | None = None,
    ) -> str:
        """
        Generate text with steering applied.

        Args:
            content: Input text
            steering_vectors: Steering vectors to apply
            config: Steering configuration

        Returns:
            Generated text with steering
        """
        if isinstance(content, str):
            content = TextContent(text=content)

        config = config or SteeringConfig()

        # Tokenize input
        inputs = self.tokenizer(
            content.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Apply steering during generation
        with self._steering_hooks(steering_vectors, config):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode output
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove input prompt from output if present
        if generated.startswith(content.text):
            generated = generated[len(content.text):].strip()

        return generated

    def _generate_unsteered(
        self,
        content: TextContent | str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> str:
        """Generate text without steering."""
        if isinstance(content, str):
            content = TextContent(text=content)

        inputs = self.tokenizer(
            content.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs,
            )

        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if generated.startswith(content.text):
            generated = generated[len(content.text):].strip()

        return generated

    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply chat template to messages.

        Args:
            messages: List of {"role": ..., "content": ...} dicts
            add_generation_prompt: Whether to add assistant prompt

        Returns:
            Formatted prompt string
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
