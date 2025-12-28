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

from wisent.comparison.utils import apply_steering_to_model, remove_steering, convert_to_lm_eval_format

if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel

__all__ = ["generate_steering_vector", "apply_steering_to_model", "remove_steering", "convert_to_lm_eval_format"]


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
    extraction_strategy: str = "mc_balanced",
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
        "--extraction-strategy", extraction_strategy,
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




