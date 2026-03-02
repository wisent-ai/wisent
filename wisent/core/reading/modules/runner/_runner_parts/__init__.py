"""Runner parts package for geometry_runner split.

Contains TransformerComponent enum and ComponentActivationExtractor class
(consolidated from _extractors.py to comply with 5-file-per-folder limit).
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List
import torch

from wisent.core.reading.modules import get_component_hook_points


class TransformerComponent(Enum):
    """Which part of transformer to extract activations from."""
    RESIDUAL = "residual"
    RESIDUAL_PRE = "residual_pre"
    RESIDUAL_MID = "residual_mid"
    MLP_OUTPUT = "mlp_output"
    ATTN_OUTPUT = "attn_output"
    ATTN_HEAD = "attn_head"


class ComponentActivationExtractor:
    """
    Extract activations from specific transformer components using hooks.

    Example:
        ```python
        extractor = ComponentActivationExtractor(model, tokenizer)
        mlp_acts = extractor.extract(
            texts=["Hello world", "Goodbye world"],
            layer=16,
            component=TransformerComponent.MLP_OUTPUT,
        )
        ```
    """

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._hook_outputs = {}
        self._hooks = []

    def _get_module_by_name(self, name: str):
        """Get a module by its full name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _register_hook(self, module_name: str):
        """Register a forward hook on a module."""
        try:
            module = self._get_module_by_name(module_name)

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    self._hook_outputs[module_name] = output[0].detach()
                else:
                    self._hook_outputs[module_name] = output.detach()

            handle = module.register_forward_hook(hook_fn)
            self._hooks.append(handle)
            return True
        except Exception as e:
            print(f"Warning: Could not register hook for {module_name}: {e}")
            return False

    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._hook_outputs = {}

    def extract(
        self,
        texts: List[str],
        layer: int,
        component: TransformerComponent,
        token_position: int = -1,
    ) -> torch.Tensor:
        """
        Extract activations from a specific component.

        Args:
            texts: List of input texts
            layer: Layer index
            component: Which component to extract from
            token_position: Which token position to extract (-1 = last)

        Returns:
            Tensor of shape [len(texts), hidden_dim]
        """
        model_type = type(self.model).__name__
        hook_points = get_component_hook_points(model_type, layer, component)
        if not hook_points:
            raise ValueError(f"No hook points found for {component} in {model_type}")
        self._clear_hooks()
        for hook_point in hook_points:
            self._register_hook(hook_point)
        activations = []
        try:
            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.tokenizer.model_max_length,
                    ).to(self.device)
                    _ = self.model(**inputs)
                    for hook_point in hook_points:
                        if hook_point in self._hook_outputs:
                            output = self._hook_outputs[hook_point]
                            act = output[0, token_position, :]
                            activations.append(act.cpu())
                            break
                    self._hook_outputs = {}
        finally:
            self._clear_hooks()
        if not activations:
            raise RuntimeError(f"Failed to extract activations for {component}")
        return torch.stack(activations)

    def extract_all_components(
        self,
        texts: List[str],
        layer: int,
        token_position: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations from all components at a layer.

        Returns:
            Dict mapping component name to activations tensor
        """
        results = {}
        for component in [
            TransformerComponent.RESIDUAL,
            TransformerComponent.MLP_OUTPUT,
            TransformerComponent.ATTN_OUTPUT,
        ]:
            try:
                acts = self.extract(texts, layer, component, token_position)
                results[component.value] = acts
            except Exception as e:
                print(f"Warning: Failed to extract {component.value}: {e}")
        return results
