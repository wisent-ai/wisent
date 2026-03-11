"""
CLI factory for creating ZapisSteeringObject from enriched pairs.

Bridges the argument parser to the ZAPIS training pipeline:
extract args -> compute per-layer mean-of-differences vectors
-> wrap in steering object with c_keys/c_values coefficients.
"""

from __future__ import annotations

import torch

from wisent.core.control.steering_methods.steering_object import SteeringObjectMetadata
from .zapis_steering_object import ZapisSteeringObject
from wisent.core.utils.config_tools.constants import (
    AXIS_ROWS,
    INDEX_LAST,
    ZAPIS_OFFSET_TOKEN_DEFAULT,
)


def _require_arg(args, attr_name):
    val = getattr(args, attr_name, None)
    if val is None:
        raise ValueError(
            f"Parameter '{attr_name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return val


def _create_zapis_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> ZapisSteeringObject:
    """Create ZAPIS object with per-layer steering vectors."""

    c_keys = _require_arg(args, "zapis_c_keys")
    c_values = _require_arg(args, "zapis_c_values")
    offset_token = getattr(args, "zapis_offset_token", ZAPIS_OFFSET_TOKEN_DEFAULT)

    vectors = {}

    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        if not pos_list or not neg_list:
            continue

        pos = torch.stack(
            [t.detach().float().reshape(INDEX_LAST) for t in pos_list],
            dim=AXIS_ROWS,
        )
        neg = torch.stack(
            [t.detach().float().reshape(INDEX_LAST) for t in neg_list],
            dim=AXIS_ROWS,
        )

        direction = pos.mean(dim=AXIS_ROWS) - neg.mean(dim=AXIS_ROWS)
        layer_int = int(layer_str)
        vectors[layer_int] = direction.detach()

        dir_norm = direction.norm().item()
        print(
            f"   Layer {layer_str}: "
            f"n_pos={pos.shape[AXIS_ROWS]}, n_neg={neg.shape[AXIS_ROWS]}, "
            f"dir_norm={dir_norm:.4f}"
        )

    return ZapisSteeringObject(
        metadata=metadata,
        vectors=vectors,
        c_keys=c_keys,
        c_values=c_values,
        offset_token=offset_token,
    )
