"""
ComponentHookManager for extracting activations from transformer submodules.

Manages PyTorch forward hooks and forward pre-hooks to capture outputs
from specific transformer components (attention, MLP, per-head, etc.).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional
import torch
import torch.nn as nn

from wisent.core.utils.config_tools.constants import ARCHITECTURE_MODULE_LIMIT
from wisent.core.primitives.model_interface.core.activations.strategies.extraction_strategy import ExtractionComponent
from wisent.core.reading.modules.utilities.signal_analysis.structure.transformer_analysis import (
    TransformerComponent,
    get_component_hook_points,
)

# Map ExtractionComponent to TransformerComponent for hook point resolution
_COMPONENT_MAP = {
    ExtractionComponent.ATTN_OUTPUT: TransformerComponent.ATTENTION,
    ExtractionComponent.MLP_OUTPUT: TransformerComponent.MLP,
    ExtractionComponent.PER_HEAD: TransformerComponent.PER_HEAD,
    ExtractionComponent.MLP_INTERMEDIATE: TransformerComponent.MLP_INTERMEDIATE,
    ExtractionComponent.POST_ATTN_RESIDUAL: TransformerComponent.POST_ATTN_RESIDUAL,
    ExtractionComponent.PRE_ATTN_LAYERNORM: TransformerComponent.PRE_ATTN_LAYERNORM,
    ExtractionComponent.Q_PROJ: TransformerComponent.Q_PROJ,
    ExtractionComponent.K_PROJ: TransformerComponent.K_PROJ,
    ExtractionComponent.V_PROJ: TransformerComponent.V_PROJ,
    ExtractionComponent.MLP_GATE_ACTIVATION: TransformerComponent.MLP_GATE,
    ExtractionComponent.ATTENTION_SCORES: TransformerComponent.ATTENTION_SCORES,
    ExtractionComponent.EMBEDDING_OUTPUT: TransformerComponent.EMBEDDING,
    ExtractionComponent.FINAL_LAYERNORM: TransformerComponent.FINAL_LAYERNORM,
    ExtractionComponent.LOGITS: TransformerComponent.LOGITS,
}

# Components that use pre-hooks (capture input) vs forward hooks (capture output)
_PRE_HOOK_COMPONENTS = {
    ExtractionComponent.PER_HEAD,
    ExtractionComponent.MLP_INTERMEDIATE,
    ExtractionComponent.POST_ATTN_RESIDUAL,
}

# Global components (not per-layer, hooked on a single module)
_GLOBAL_COMPONENTS = {
    ExtractionComponent.EMBEDDING_OUTPUT,
    ExtractionComponent.FINAL_LAYERNORM,
    ExtractionComponent.LOGITS,
}


def _get_submodule(model: nn.Module, target: str) -> nn.Module:
    """Resolve a dotted path to a submodule."""
    parts = target.split(".")
    current = model
    for part in parts:
        current = getattr(current, part)
    return current


def _detect_model_type(model: nn.Module) -> str:
    """Detect model architecture family from config or module names."""
    config = getattr(model, "config", None)
    if config is not None:
        model_type = getattr(config, "model_type", "")
        if model_type:
            return model_type

    # Fallback: check for common module patterns
    module_names = [name for name, _ in model.named_modules()]
    joined = " ".join(module_names[:ARCHITECTURE_MODULE_LIMIT])
    if "self_attn" in joined:
        return "llama"  # Llama-like architecture
    if "attn" in joined and "transformer.h" in joined:
        return "gpt2"
    return "llama"  # Default to Llama-like


class ComponentHookManager:
    """
    Manages forward hooks for extracting from transformer components.

    Usage:
        manager = ComponentHookManager(model, component, layers=[8, 12])
        with manager.hooks_active():
            output = model(**inputs, output_hidden_states=True)
            captured = manager.get_captured()
    """

    def __init__(
        self,
        model: nn.Module,
        component: ExtractionComponent,
        layers: List[int],
    ):
        self.model = model
        self.component = component
        self.layers = layers
        self._handles: List[torch.utils.hooks.RemovableHook] = []
        self._captured: Dict[int, torch.Tensor] = {}
        self._model_type = _detect_model_type(model)

    def _make_forward_hook(self, layer_idx: int):
        """Create a forward hook that captures output[0]."""
        def hook(module, input, output):
            if isinstance(output, tuple):
                self._captured[layer_idx] = output[0].detach()
            else:
                self._captured[layer_idx] = output.detach()
        return hook

    def _make_pre_hook(self, layer_idx: int):
        """Create a forward pre-hook that captures the input."""
        def hook(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                self._captured[layer_idx] = input[0].detach()
            else:
                self._captured[layer_idx] = input.detach()
        return hook

    def _register_hooks(self) -> None:
        """Register hooks on the appropriate submodules."""
        tc = _COMPONENT_MAP.get(self.component)
        if tc is None:
            raise ValueError(
                f"Component {self.component.value} does not support hooks. "
                f"Use RESIDUAL_STREAM with output_hidden_states instead."
            )

        use_pre_hook = self.component in _PRE_HOOK_COMPONENTS

        for layer_idx in self.layers:
            hook_points = get_component_hook_points(
                self._model_type, layer_idx, tc
            )
            for point_name in hook_points:
                try:
                    submodule = _get_submodule(self.model, point_name)
                except AttributeError:
                    raise AttributeError(
                        f"Cannot find submodule '{point_name}' in model. "
                        f"Model type detected as '{self._model_type}'."
                    )

                if use_pre_hook:
                    handle = submodule.register_forward_pre_hook(
                        self._make_pre_hook(layer_idx)
                    )
                else:
                    handle = submodule.register_forward_hook(
                        self._make_forward_hook(layer_idx)
                    )
                self._handles.append(handle)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @contextmanager
    def hooks_active(self):
        """Context manager that registers hooks, yields, then removes them."""
        self._captured.clear()
        self._register_hooks()
        try:
            yield self
        finally:
            self._remove_hooks()

    def get_captured(self) -> Dict[int, torch.Tensor]:
        """
        Get captured activations after a forward pass.

        Returns:
            Dict mapping layer index -> tensor.
            For forward hooks: [batch, seq_len, hidden_dim]
            For pre-hooks: [batch, seq_len, dim] where dim depends on component.
        """
        return dict(self._captured)

    def clear(self) -> None:
        """Clear captured activations."""
        self._captured.clear()
