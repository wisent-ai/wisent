"""Helper methods for WisentSteeringTrainer: save_result and layer resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import torch

from wisent.core.primitives.model_interface.core.activations.core.atoms import RawActivationMap
from wisent.core.weight_modification.trainers.core.atoms import TrainingResult
from wisent.core.utils.config_tools.constants import JSON_INDENT
from wisent.core.utils.infra_tools.errors import NoTrainingResultError


def save_result_impl(result, last_result, output_dir):
    """
    Persist vectors, metadata, and the pair set (with activations) to disk.

    Files written:
        - metadata.json
        - steering_vectors.pt
        - pairs_with_activations.pt
        - steering_vectors_summary.json

    returns:
        Path to the created directory.
    """
    result = result or last_result
    if result is None:
        raise NoTrainingResultError()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Vectors
    raw_map: RawActivationMap = result.steered_vectors.to_dict()
    cpu_map = {k: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for k, v in raw_map.items()}
    torch.save(cpu_map, out / "steering_vectors.pt")

    # Summary (json-serializable)
    vec_summary = {
        k: None if v is None else {
            "shape": tuple(v.shape),
            "dtype": str(v.dtype),
        }
        for k, v in cpu_map.items()
    }
    (out / "steering_vectors_summary.json").write_text(json.dumps(vec_summary, indent=JSON_INDENT))

    # Metadata
    (out / "metadata.json").write_text(json.dumps(result.metadata, indent=JSON_INDENT))

    # Full pair set with activations (Python pickle via torch.save)
    torch.save(result.pair_set_with_activations, out / "pairs_with_activations.pt")

    return out


def resolve_layers(model, spec):
    """
    Convert a user-facing spec into canonical layer names.
    If None, return None (meaning: use all layers in the collector/model).
    """
    if spec is None:
        return None

    if isinstance(spec, (list, tuple)):
        names: list[str] = []
        for item in spec:
            if isinstance(item, int):
                names.append(str(item))
            else:
                names.extend(parse_layer_token(item))
        return sorted(set(names), key=lambda s: (len(s), s))

    if isinstance(spec, int):
        return [str(spec)]

    names: list[str] = []
    for token in str(spec).replace(" ", "").split(","):
        names.extend(parse_layer_token(token))
    return sorted(set(names), key=lambda s: (len(s), s))


def parse_layer_token(token: str) -> list[str]:
    """Parse a token like "5", "10-20", "10..20" into a list of names."""
    if not token:
        return []
    if "-" in token or ".." in token:
        a, b = token.replace("..", "-").split("-")
        a_i, b_i = int(a), int(b)
        lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
        return [str(i) for i in range(lo, hi + 1)]
    else:
        return [str(int(token))]
