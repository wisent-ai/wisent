"""Extracted TextAdapter methods: generation and chat template formatting."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import torch

from wisent.core.adapters.base import SteeringConfig
from wisent.core.activations.core.atoms import LayerActivations
from wisent.core.modalities import TextContent
from wisent.core.constants import DEFAULT_MAX_NEW_TOKENS_ADAPTER, DEFAULT_INFERENCE_TEMPERATURE


class TextAdapterGenerationMixin:
    """Mixin providing generation and chat template methods for TextAdapter."""

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
                max_new_tokens=DEFAULT_MAX_NEW_TOKENS_ADAPTER,
                do_sample=True,
                temperature=DEFAULT_INFERENCE_TEMPERATURE,
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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_ADAPTER,
        temperature: float = DEFAULT_INFERENCE_TEMPERATURE,
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
