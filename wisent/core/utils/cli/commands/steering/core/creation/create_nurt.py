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
    DEFAULT_VARIANCE_THRESHOLD,
    MLP_LEARNING_RATE,
    NURT_NUM_DIMS,
    NURT_NUM_INTEGRATION_STEPS,
    NURT_T_MAX,
    NURT_TRAINING_EPOCHS,
)


def _create_nurt_steering_object(
    metadata: SteeringObjectMetadata,
    layer_activations: dict,
    available_layers: list,
    args,
) -> NurtSteeringObject:
    """Create Concept Flow steering object with per-layer flow networks."""

    num_dims = getattr(args, "nurt_num_dims", NURT_NUM_DIMS)
    variance_threshold = getattr(args, "nurt_variance_threshold", DEFAULT_VARIANCE_THRESHOLD)
    training_epochs = getattr(args, "nurt_training_epochs", NURT_TRAINING_EPOCHS)
    lr = getattr(args, "nurt_lr", MLP_LEARNING_RATE)
    num_integration_steps = getattr(args, "nurt_num_integration_steps", NURT_NUM_INTEGRATION_STEPS)
    t_max = getattr(args, "nurt_t_max", NURT_T_MAX)
    flow_hidden_dim_raw = getattr(args, "nurt_hidden_dim", NURT_NUM_DIMS)
    flow_hidden_dim = flow_hidden_dim_raw if flow_hidden_dim_raw > 0 else None

    method = NurtMethod(
        num_dims=num_dims,
        variance_threshold=variance_threshold,
        training_epochs=training_epochs,
        lr=lr,
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
            pos, neg, num_dims=num_dims, variance_threshold=variance_threshold,
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
