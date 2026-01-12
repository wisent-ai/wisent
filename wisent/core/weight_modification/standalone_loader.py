"""
Standalone loader for TITAN and PULSE models.

This file is saved alongside exported models so users can load them
without installing the full wisent package.

Usage:
    from standalone_loader import load_model
    model, tokenizer, hooks = load_model("./my_titan_model")
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer


class GatingNetwork(nn.Module):
    """Learned gating network matching TITAN architecture."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        logit = self.net(x)
        return torch.sigmoid(logit / temperature)


class IntensityNetwork(nn.Module):
    """Predicts per-layer steering intensity matching TITAN architecture."""
    def __init__(self, input_dim: int, num_layers: int, hidden_dim: int = 64, max_alpha: float = 3.0):
        super().__init__()
        self.max_alpha = max_alpha
        self.num_layers = num_layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_layers),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        raw = self.net(x)
        return self.max_alpha * torch.sigmoid(raw)


class TITANHooks:
    """Runtime hooks for TITAN dynamic steering."""
    
    def __init__(self, model, titan_data: Dict[str, Any], base_strength: float = 1.0):
        self.model = model
        self.titan_data = titan_data
        self.base_strength = base_strength
        
        self._hooks = []
        self._current_gate = None
        self._current_intensities = None
        
        # Get model layers
        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers
        
        # Setup layer mapping
        self.layer_order = titan_data["layer_order"]
        self._layer_name_to_idx = {}
        for layer_name in self.layer_order:
            try:
                idx = int(str(layer_name).split("_")[-1])
                self._layer_name_to_idx[layer_name] = idx
            except (ValueError, IndexError):
                pass
        
        # Find sensor layer
        sensor_layer = titan_data.get("sensor_layer")
        if sensor_layer is None:
            sensor_layer = self._layer_name_to_idx.get(
                self.layer_order[len(self.layer_order)//2], 15
            )
        self._sensor_layer_idx = sensor_layer
        
        # Load directions
        self.directions = {k: v.to(model.device) for k, v in titan_data["directions"].items()}
        self.direction_weights = {k: v.to(model.device) for k, v in titan_data["direction_weights"].items()}
        
        # Load networks if present
        self.gate_network = None
        self.intensity_network = None
        
        if "gate_network_state" in titan_data:
            config = titan_data["gate_network_config"]
            self.gate_network = GatingNetwork(config["input_dim"], config.get("hidden_dim", 128))
            self.gate_network.load_state_dict(titan_data["gate_network_state"])
            self.gate_network = self.gate_network.to(model.device).eval()
        
        if "intensity_network_state" in titan_data:
            config = titan_data["intensity_network_config"]
            self.intensity_network = IntensityNetwork(
                config["input_dim"], config["num_layers"], config.get("hidden_dim", 64),
                max_alpha=titan_data.get("max_alpha", 3.0)
            )
            self.intensity_network.load_state_dict(titan_data["intensity_network_state"])
            self.intensity_network = self.intensity_network.to(model.device).eval()
        
        self.gate_temperature = titan_data.get("gate_temperature", 0.5)
        self.max_alpha = titan_data.get("max_alpha", 3.0)
    
    def _get_effective_direction(self, layer_name: str) -> torch.Tensor:
        """Get weighted combination of directions for a layer."""
        dirs = self.directions[layer_name]
        weights = self.direction_weights[layer_name]
        weights_norm = F.softmax(weights, dim=0)
        return (dirs * weights_norm.unsqueeze(1)).sum(dim=0)
    
    def install(self) -> None:
        """Install forward hooks."""
        self.remove()
        
        if self._sensor_layer_idx < len(self._layers):
            hook = self._layers[self._sensor_layer_idx].register_forward_hook(self._sensor_hook)
            self._hooks.append(hook)
        
        for layer_name in self.layer_order:
            layer_idx = self._layer_name_to_idx.get(layer_name)
            if layer_idx is not None and layer_idx < len(self._layers):
                hook = self._layers[layer_idx].register_forward_hook(
                    lambda m, i, o, ln=layer_name: self._steering_hook(m, i, o, ln)
                )
                self._hooks.append(hook)
    
    def remove(self) -> None:
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        self._current_gate = None
        self._current_intensities = None
    
    def _sensor_hook(self, module, input, output):
        """Capture sensor activation and compute gate/intensity."""
        hidden = output[0] if isinstance(output, tuple) else output
        sensor_hidden = hidden[:, -1, :] if hidden.dim() == 3 else hidden
        
        if self.gate_network is not None:
            with torch.no_grad():
                self._current_gate = self.gate_network(sensor_hidden.float())
        else:
            self._current_gate = torch.ones(sensor_hidden.shape[0], 1, device=sensor_hidden.device)
        
        if self.intensity_network is not None:
            with torch.no_grad():
                intensities = self.intensity_network(sensor_hidden.float())
                self._current_intensities = {
                    layer: intensities[:, i:i+1] for i, layer in enumerate(self.layer_order)
                }
        else:
            self._current_intensities = {
                layer: torch.ones(sensor_hidden.shape[0], 1, device=sensor_hidden.device) 
                for layer in self.layer_order
            }
        
        return output
    
    def _steering_hook(self, module, input, output, layer_name):
        """Apply dynamic steering."""
        if self._current_gate is None:
            return output
        
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        
        direction = self._get_effective_direction(layer_name).to(hidden.device)
        gate = self._current_gate.to(hidden.device)
        intensity = self._current_intensities.get(layer_name, torch.ones_like(gate)).to(hidden.device)
        
        if hidden.dim() == 3:
            gate = gate.view(-1, 1, 1)
            intensity = intensity.view(-1, 1, 1)
            direction = direction.view(1, 1, -1)
        else:
            gate = gate.view(-1, 1)
            intensity = intensity.view(-1, 1)
            direction = direction.view(1, -1)
        
        hidden = hidden + gate * intensity * self.base_strength * direction
        
        return (hidden,) + rest if rest else hidden
    
    def get_current_gate(self) -> Optional[float]:
        if self._current_gate is not None:
            return self._current_gate.mean().item()
        return None
    
    def get_current_intensities(self) -> Optional[Dict[str, float]]:
        if self._current_intensities is not None:
            return {k: v.mean().item() for k, v in self._current_intensities.items()}
        return None


class PULSEHooks:
    """Runtime hooks for PULSE conditional steering."""
    
    def __init__(self, model, pulse_data: Dict[str, Any], base_strength: float = 1.0):
        self.model = model
        self.pulse_data = pulse_data
        self.base_strength = base_strength
        
        self._hooks = []
        self._current_gate = None
        
        if hasattr(model, "model"):
            self._layers = model.model.layers
        elif hasattr(model, "transformer"):
            self._layers = model.transformer.h
        else:
            self._layers = model.layers
        
        # Load steering data
        self.behavior_vectors = {k: v.to(model.device) for k, v in pulse_data["behavior_vectors"].items()}
        self.condition_vector = pulse_data["condition_vector"].to(model.device)
        self.layer_scales = pulse_data["layer_scales"]
        self.optimal_threshold = pulse_data["optimal_threshold"]
        
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
        self._sensor_layer_idx = layer_indices[len(layer_indices)//2] if layer_indices else 15
    
    def install(self) -> None:
        """Install forward hooks."""
        self.remove()
        
        if self._sensor_layer_idx < len(self._layers):
            hook = self._layers[self._sensor_layer_idx].register_forward_hook(self._sensor_hook)
            self._hooks.append(hook)
        
        for layer_name in self.behavior_vectors.keys():
            layer_idx = self._layer_name_to_idx.get(layer_name)
            if layer_idx is not None and layer_idx < len(self._layers):
                hook = self._layers[layer_idx].register_forward_hook(
                    lambda m, i, o, ln=layer_name: self._steering_hook(m, i, o, ln)
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
        self._current_gate = torch.sigmoid((similarity - self.optimal_threshold) / 0.1)
        
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
        scale = self.layer_scales.get(layer_name, 1.0)
        
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
    device_map: str = "auto",
    torch_dtype = None,
    install_hooks: bool = True,
) -> Tuple[Any, Any, Optional[Any]]:
    """
    Load a TITAN or PULSE steered model.
    
    Args:
        model_path: Path to model directory or HuggingFace repo
        device_map: Device placement ("auto", "cuda", "cpu", "mps")
        torch_dtype: Model dtype (e.g., torch.float16)
        install_hooks: Whether to install dynamic steering hooks
    
    Returns:
        Tuple of (model, tokenizer, hooks)
        - hooks is None if no dynamic steering or install_hooks=False
    
    Example:
        model, tokenizer, hooks = load_model("./my_titan_model")
        output = model.generate(...)
        print(f"Gate: {hooks.get_current_gate()}")
        hooks.remove()  # When done
    """
    model_path = Path(model_path) if not str(model_path).startswith(("http", "hf://")) else model_path
    
    # Check for TITAN or PULSE config
    is_local = isinstance(model_path, Path) and model_path.exists()
    
    titan_config = None
    pulse_config = None
    
    if is_local:
        titan_config_path = model_path / "titan_config.json"
        pulse_config_path = model_path / "pulse_config.json"
        
        if titan_config_path.exists():
            with open(titan_config_path) as f:
                titan_config = json.load(f)
        elif pulse_config_path.exists():
            with open(pulse_config_path) as f:
                pulse_config = json.load(f)
    else:
        # HuggingFace Hub
        from huggingface_hub import hf_hub_download
        try:
            config_file = hf_hub_download(repo_id=str(model_path), filename="titan_config.json")
            with open(config_file) as f:
                titan_config = json.load(f)
        except:
            try:
                config_file = hf_hub_download(repo_id=str(model_path), filename="pulse_config.json")
                with open(config_file) as f:
                    pulse_config = json.load(f)
            except:
                pass
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    
    hooks = None
    
    if install_hooks:
        if titan_config and titan_config.get("mode") in ("dynamic", "hybrid"):
            # Load TITAN data
            if is_local:
                titan_data = torch.load(model_path / "titan_steering.pt", map_location="cpu")
            else:
                from huggingface_hub import hf_hub_download
                data_file = hf_hub_download(repo_id=str(model_path), filename="titan_steering.pt")
                titan_data = torch.load(data_file, map_location="cpu")
            
            hooks = TITANHooks(model, titan_data)
            hooks.install()
        
        elif pulse_config and pulse_config.get("mode") in ("dynamic", "hybrid"):
            # Load PULSE data
            if is_local:
                pulse_data = torch.load(model_path / "pulse_steering.pt", map_location="cpu")
            else:
                from huggingface_hub import hf_hub_download
                data_file = hf_hub_download(repo_id=str(model_path), filename="pulse_steering.pt")
                pulse_data = torch.load(data_file, map_location="cpu")
            
            hooks = PULSEHooks(model, pulse_data, base_strength=0.5)
            hooks.install()
    
    return model, tokenizer, hooks
