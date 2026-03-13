"""
CLI factory for creating NurtSteeringObject from enriched pairs.

Bridges the argument parser to the Nurt training pipeline:
extract args -> train per-layer flow networks -> wrap in steering object.
"""

from __future__ import annotations

import torch

from wisent.core.control.steering_methods.steering_object import SteeringObjectMetadata
from wisent.core.control.steering_methods.methods.nurt import (
    NurtMethod,
    NurtSteeringObject,
)
from wisent.core.utils.config_tools.constants import (
    SS_NURT_FLOW_HIDDEN_DIM_MIN,
    NURT_CONCEPT_DIM_MIN,
)


def _require_arg(args, attr_name):
    val = getattr(args, attr_name, None)
    if val is None:
        raise ValueError(
            f"Parameter '{attr_name}' is required. "
            f"Run 'wisent optimize-steering auto' first, or pass it explicitly."
        )
    return val


def _create_nurt_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> NurtSteeringObject:
    """Create Concept Flow steering object with per-layer flow networks."""

    num_dims = _require_arg(args, "nurt_num_dims")
    max_concept_dim = _require_arg(args, "nurt_max_concept_dim")
    variance_threshold = _require_arg(args, "nurt_variance_threshold")
    training_epochs = _require_arg(args, "nurt_training_epochs")
    lr = _require_arg(args, "nurt_lr")
    lr_min = _require_arg(args, "nurt_lr_min")
    weight_decay = _require_arg(args, "nurt_weight_decay")
    max_grad_norm = _require_arg(args, "nurt_max_grad_norm")
    num_integration_steps = _require_arg(args, "nurt_num_integration_steps")
    t_max = _require_arg(args, "nurt_t_max")
    flow_hidden_dim_raw = _require_arg(args, "nurt_hidden_dim")
    flow_hidden_dim = (
        flow_hidden_dim_raw if flow_hidden_dim_raw > SS_NURT_FLOW_HIDDEN_DIM_MIN
        else None
    )

    method = NurtMethod(
        num_dims=num_dims,
        max_concept_dim=max_concept_dim,
        variance_threshold=variance_threshold,
        training_epochs=training_epochs,
        lr=lr,
        lr_min=lr_min,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        num_integration_steps=num_integration_steps,
        t_max=t_max,
        flow_hidden_dim=flow_hidden_dim,
    )

    # Prepare per-layer data
    from wisent.core.control.steering_methods.methods.nurt.subspace import (
        discover_concept_subspace,
        project_to_subspace,
    )
    from wisent.core.control.steering_methods.methods.nurt.flow_network import (
        FlowVelocityNetwork,
    )

    flow_networks = {}
    concept_bases = {}
    mean_neg_dict = {}
    mean_pos_dict = {}
    layer_variance = {}

    for layer_str in available_layers:
        pos_list = layer_activations[layer_str]["positive"]
        neg_list = layer_activations[layer_str]["negative"]
        if not pos_list or not neg_list:
            continue

        pos = torch.stack([t.detach().float().reshape(-1) for t in pos_list], dim=0)
        neg = torch.stack([t.detach().float().reshape(-1) for t in neg_list], dim=0)

        # Discover subspace
        Vh, S, k = discover_concept_subspace(
            pos, neg, variance_threshold=variance_threshold,
            nurt_num_dims=num_dims, nurt_max_concept_dim=max_concept_dim,
            min_concept_dim=NURT_CONCEPT_DIM_MIN,
        )
        # Project
        z_pos = project_to_subspace(pos, Vh)
        z_neg = project_to_subspace(neg, Vh)

        # Train flow network
        network = method._train_flow_network(z_pos, z_neg, k)

        layer_int = int(layer_str)
        flow_networks[layer_int] = network
        concept_bases[layer_int] = Vh.detach()
        mean_neg_dict[layer_int] = z_neg.mean(dim=0).detach()
        mean_pos_dict[layer_int] = z_pos.mean(dim=0).detach()

        var_exp = ((S[:k] ** 2).sum() / (S ** 2).sum()).item() if S.sum() > 0 else 0.0
        layer_variance[layer_int] = var_exp
        print(f"   Layer {layer_str}: k={k}, var_explained={var_exp:.3f}")

    return NurtSteeringObject(
        metadata=metadata,
        flow_networks=flow_networks,
        concept_bases=concept_bases,
        mean_neg=mean_neg_dict,
        mean_pos=mean_pos_dict,
        num_integration_steps=num_integration_steps,
        t_max=t_max,
        layer_variance=layer_variance,
    )
