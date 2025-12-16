"""
Shared utilities for comparison experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel


def load_steering_vector(path: str | Path, default_method: str = "unknown") -> dict:
    """
    Load a steering vector from file.

    Args:
        path: Path to steering vector file (.json or .pt)
        default_method: Default method name if not found in file

    Returns:
        Dictionary with steering vectors and metadata
    """
    path = Path(path)

    if path.suffix == ".pt":
        data = torch.load(path, map_location="cpu", weights_only=False)
        layer_idx = str(data.get("layer_index", data.get("layer", 1)))
        return {
            "steering_vectors": {layer_idx: data["steering_vector"].tolist()},
            "layers": [layer_idx],
            "model": data.get("model", "unknown"),
            "method": data.get("method", default_method),
            "trait_label": data.get("trait_label", "unknown"),
        }
    else:
        with open(path) as f:
            return json.load(f)


def apply_steering_to_model(
    model: "WisentModel",
    steering_data: dict,
    scale: float = 1.0,
) -> None:
    """
    Apply loaded steering vectors to a WisentModel.

    Args:
        model: WisentModel instance
        steering_data: Dictionary from load_steering_vector()
        scale: Scaling factor for steering strength
    """
    raw_map = {}
    for layer_str, vec_list in steering_data["steering_vectors"].items():
        raw_map[layer_str] = torch.tensor(vec_list, dtype=torch.float32)

    model.set_steering_from_raw(raw_map, scale=scale, normalize=False)
    model.apply_steering()


def remove_steering(model: "WisentModel") -> None:
    """Remove steering from a WisentModel."""
    model.detach()
    model.clear_steering()


def convert_to_lm_eval_format(
    steering_data: dict,
    output_path: str | Path,
    scale: float = 1.0,
) -> Path:
    """
    Convert our steering vector format to lm-eval's steered model format.

    lm-eval expects:
    {
        "layers.N": {
            "steering_vector": tensor of shape (1, hidden_dim),
            "steering_coefficient": float,
            "action": "add"
        }
    }
    """
    output_path = Path(output_path)

    lm_eval_config = {}
    for layer_str, vec_list in steering_data["steering_vectors"].items():
        vec = torch.tensor(vec_list, dtype=torch.float32)
        # lm-eval expects shape (1, hidden_dim)
        if vec.dim() == 1:
            vec = vec.unsqueeze(0)

        layer_key = f"layers.{layer_str}"
        lm_eval_config[layer_key] = {
            "steering_vector": vec,
            "steering_coefficient": scale,
            "action": "add",
        }

    torch.save(lm_eval_config, output_path)
    return output_path
