"""Loader for contrastive pair sets."""
from __future__ import annotations

import json
import pathlib

import torch
import numpy as np

from wisent.core.contrastive_pairs.io.serialization import (
    _validate_top_level,
    _validate_pair_obj,
    _decode_activations,
    _maybe_decode_response,
)

def load_contrastive_pair_set(
    filepath: str | Path,
    return_backend: str = "torch",
    validate: bool = True,
) -> ContrastivePairSet:
    """Load a ContrastivePairSet from a JSON file and decode activations.

    Args:
        filepath: path to the JSON file.
        return_backend: 'torch' (default), 'numpy', or 'list'. If torch is not
            installed, will automatically fall back to 'numpy'.
        validate: If True (default), runs diagnostics validation on the loaded pairs.
            Set to False to skip validation (useful for welfare pairs with intentionally
            similar positive/negative responses).

    Returns:
        ContrastivePairSet

        Format of loaded data:
        {
            "name": "name of the set",
            "task_type": "task type string",
            "pairs": [
                {
                    "prompt": "The input prompt",
                    "positive_response": {
                        "model_response": "The positive response",
                        "activations": VectorPayload or None,
                        "label": "positive"
                    },
                    "negative_response": {
                        "model_response": "The negative response",
                        "activations": VectorPayload or None,
                        "label": "negative"
                    },
                    "label": "overall label" or None,
                    "trait_description": "description of the trait" or None
                },
                ...
            ]
        }

    """
    filepath = Path(filepath)
    with filepath.open("r", encoding="utf-8") as f:
        data = json.load(f)

    _validate_top_level(data)

    decoded_pairs: list[dict[str, ]] = []
    for pair in data["pairs"]:
        _validate_pair_obj(pair)
        p = dict(pair)
        p["positive_response"] = _maybe_decode_response(p.get("positive_response", {}), return_backend)
        p["negative_response"] = _maybe_decode_response(p.get("negative_response", {}), return_backend)
        decoded_pairs.append(p)
    
    list_of_pairs = [ContrastivePair.from_dict(p) for p in decoded_pairs]

    cps = ContrastivePairSet(name=str(data["name"]), pairs=list_of_pairs, task_type=data.get("task_type"))

    if validate:
        cps.validate()

