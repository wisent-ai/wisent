"""
Our steering method wrapper for comparison experiments.

Uses the existing wisent infrastructure to create steering vectors.
Runs steering vector generation in subprocess to guarantee memory cleanup.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel


def generate_steering_vector(
    task: str,
    model_name: str,
    output_path: str | Path,
    trait_label: str = "correctness",
    num_pairs: int = 50,
    method: str = "caa",
    layers: str | None = None,
    normalize: bool = True,
    device: str = "cuda:0",
    keep_intermediate: bool = False,
) -> Path:
    """
    Generate a steering vector using wisent CLI in subprocess.

    Runs in subprocess to guarantee GPU memory is freed when done.
    """
    output_path = Path(output_path)

    cmd = [
        "wisent", "generate-vector-from-task",
        "--task", task,
        "--trait-label", trait_label,
        "--model", model_name,
        "--num-pairs", str(num_pairs),
        "--method", method,
        "--output", str(output_path),
        "--device", device,
        "--accept-low-quality-vector",
    ]

    if layers:
        cmd.extend(["--layers", layers])

    if normalize:
        cmd.append("--normalize")

    if keep_intermediate:
        cmd.append("--keep-intermediate")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate steering vector (exit code {result.returncode})")

    return output_path


def load_steering_vector(path: str | Path) -> dict:
    """
    Load a steering vector from file.

    Args:
        path: Path to steering vector file (.json or .pt)

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
            "method": data.get("method", "unknown"),
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


