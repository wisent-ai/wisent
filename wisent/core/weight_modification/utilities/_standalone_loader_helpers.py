"""Extracted standalone loader: TETNOHooks and load_model function."""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from wisent.core.utils.infra_tools.errors import MissingParameterError


class TETNOHooks:
    """Runtime hooks for TETNO conditional steering."""

    def __init__(self, model, tetno_data: Dict[str, Any],
                 base_strength: float, *, gate_scale_factor: float):
        self.model = model
        self.tetno_data = tetno_data
        self.base_strength = base_strength
        self._gate_scale_factor = gate_scale_factor

        self._hooks = []
        self._current_gate = None

        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers

        # Load steering data
        self.behavior_vectors = {
            k: v.to(model.device) for k, v in tetno_data["behavior_vectors"].items()
        }
        self.condition_vector = tetno_data["condition_vector"].to(model.device)
        self.layer_scales = tetno_data["layer_scales"]
        self.optimal_threshold = tetno_data["optimal_threshold"]

        # Map layer names to indices
        self._layer_name_to_idx = {}
        for layer_name in self.behavior_vectors.keys():
            try:
                idx = int(str(layer_name).split("_")[-1])
                self._layer_name_to_idx[layer_name] = idx
            except (ValueError, IndexError):
                pass

        # Sensor layer
        layer_indices = list(self._layer_name_to_idx.values())
        self._sensor_layer_idx = (
            layer_indices[len(layer_indices) // 2] if layer_indices else None
        )
        if self._sensor_layer_idx is None:
            raise MissingParameterError(params=["sensor_layer"], context="TETNO hooks: no layer indices found in behavior vectors")

    def install(self) -> None:
        """Install forward hooks."""
        self.remove()

        if self._sensor_layer_idx < len(self._layers):
            hook = self._layers[self._sensor_layer_idx].register_forward_hook(
                self._sensor_hook)
            self._hooks.append(hook)

        for layer_name in self.behavior_vectors.keys():
            layer_idx = self._layer_name_to_idx.get(layer_name)
            if layer_idx is not None and layer_idx < len(self._layers):
                hook = self._layers[layer_idx].register_forward_hook(
                    lambda m, i, o, ln=layer_name: self._steering_hook(
                        m, i, o, ln)
                )
                self._hooks.append(hook)

    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._current_gate = None

    def _sensor_hook(self, module, input, output):
        """Compute gate from condition vector similarity."""
        hidden = output[0] if isinstance(output, tuple) else output
        sensor_hidden = hidden[:, -1, :] if hidden.dim() == 3 else hidden

        h_norm = F.normalize(sensor_hidden, p=2, dim=-1)
        c_norm = F.normalize(self.condition_vector, p=2, dim=-1)
        similarity = (h_norm * c_norm).sum(dim=-1)
        self._current_gate = torch.sigmoid(
            (similarity - self.optimal_threshold) / self._gate_scale_factor)

        return output

    def _steering_hook(self, module, input, output, layer_name):
        """Apply conditional steering."""
        if self._current_gate is None:
            return output

        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None

        behavior = self.behavior_vectors.get(layer_name)
        if behavior is None:
            return output

        behavior = behavior.to(hidden.device)
        gate = self._current_gate.to(hidden.device)
        if layer_name not in self.layer_scales:
            raise KeyError(f"No layer_scale for '{layer_name}' in TETNO data")
        scale = self.layer_scales[layer_name]

        if hidden.dim() == 3:
            gate = gate.view(-1, 1, 1)
            behavior = behavior.view(1, 1, -1)
        else:
            gate = gate.view(-1, 1)
            behavior = behavior.view(1, -1)

        hidden = hidden + gate * self.base_strength * scale * behavior

        return (hidden,) + rest if rest else hidden

    def get_current_gate(self) -> Optional[float]:
        if self._current_gate is not None:
            return self._current_gate.mean().item()
        return None


def load_model(
    model_path: str,
    device_map: Optional[str] = None,
    torch_dtype=None,
    install_hooks: bool = True,
    GROMHooksClass=None,
    TETNOHooksClass=TETNOHooks,
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Load a GROM or TETNO steered model.

    Args:
        model_path: Path to model directory or HuggingFace repo
        device_map: Device placement ("auto", "cuda", "cpu", "mps")
        torch_dtype: Model dtype (e.g., torch.float16)
        install_hooks: Whether to install dynamic steering hooks
        GROMHooksClass: GROMHooks class (injected to avoid circular imports)
        TETNOHooksClass: TETNOHooks class (defaults to local TETNOHooks)

    Returns:
        Tuple of (model, tokenizer, hooks)
        - hooks is None if no dynamic steering or install_hooks=False
    """
    model_path = (Path(model_path)
                  if not str(model_path).startswith(("http", "hf://"))
                  else model_path)

    # Check for GROM or TETNO config
    is_local = isinstance(model_path, Path) and model_path.exists()

    grom_config = None
    tetno_config = None

    if is_local:
        grom_config_path = model_path / "grom_config.json"
        tetno_config_path = model_path / "tetno_config.json"

        if grom_config_path.exists():
            with open(grom_config_path) as f:
                grom_config = json.load(f)
        elif tetno_config_path.exists():
            with open(tetno_config_path) as f:
                tetno_config = json.load(f)
    else:
        # HuggingFace Hub
        from huggingface_hub import hf_hub_download
        try:
            config_file = hf_hub_download(
                repo_id=str(model_path), filename="grom_config.json")
            with open(config_file) as f:
                grom_config = json.load(f)
        except Exception:
            try:
                config_file = hf_hub_download(
                    repo_id=str(model_path), filename="tetno_config.json")
                with open(config_file) as f:
                    tetno_config = json.load(f)
            except Exception:
                pass

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), device_map=device_map,
        torch_dtype=torch_dtype, trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    hooks = None

    if install_hooks:
        if grom_config and grom_config.get("mode") in ("dynamic", "hybrid"):
            if is_local:
                grom_data = torch.load(
                    model_path / "grom_steering.pt", map_location="cpu")
            else:
                from huggingface_hub import hf_hub_download
                data_file = hf_hub_download(
                    repo_id=str(model_path), filename="grom_steering.pt")
                grom_data = torch.load(data_file, map_location="cpu")

            if GROMHooksClass is not None:
                hooks = GROMHooksClass(model, grom_data)
                hooks.install()

        elif tetno_config and tetno_config.get("mode") in ("dynamic", "hybrid"):
            if is_local:
                tetno_data = torch.load(
                    model_path / "tetno_steering.pt", map_location="cpu")
            else:
                from huggingface_hub import hf_hub_download
                data_file = hf_hub_download(
                    repo_id=str(model_path), filename="tetno_steering.pt")
                tetno_data = torch.load(data_file, map_location="cpu")

            gate_scale = tetno_config.get("gate_scale_factor")
            if gate_scale is None:
                raise MissingParameterError(params=["gate_scale_factor"], context="tetno_config.json must include gate_scale_factor")
            base_strength = tetno_config.get("base_strength")
            if base_strength is None:
                raise MissingParameterError(params=["base_strength"], context="tetno_config.json must include base_strength")
            hooks = TETNOHooksClass(model, tetno_data, base_strength=base_strength, gate_scale_factor=gate_scale)
            hooks.install()

    return model, tokenizer, hooks
