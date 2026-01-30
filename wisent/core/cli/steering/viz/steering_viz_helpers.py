"""Helper functions for steering visualization - direction discovery and method setup."""

import numpy as np
import torch
from typing import Tuple, Any

# Re-export behavioral collection functions from separate module
from wisent.core.cli.steering_behavioral import (
    extract_response,
    collect_behavioral_labels,
    collect_behavioral_labels_all_layers,
)

# Re-export from steering_viz_utils for backwards compatibility
from wisent.core.geometry.steering_viz_utils import (
    create_steering_object_from_pairs,
    extract_activations_from_responses,
    load_reference_activations,
    train_classifier_and_predict,
    save_viz_summary,
)


def select_steering_direction(
    pos_ref: torch.Tensor,
    neg_ref: torch.Tensor,
    direction_method: str,
    behavioral_activations: np.ndarray = None,
    behavioral_labels: np.ndarray = None,
) -> Tuple[torch.Tensor, str, float]:
    """
    Select steering direction based on the specified method.

    For behavioral method, pass activations and labels from actual generation.
    Returns (steering_vector, description, accuracy).
    Accuracy is the behavioral classifier accuracy (0.5 for non-behavioral methods).
    """
    from wisent.core.geometry.steering_discovery import generate_candidate_directions

    pos_ref_np = pos_ref.cpu().numpy()
    neg_ref_np = neg_ref.cpu().numpy()

    acc = 0.5  # Default accuracy (random baseline) for non-behavioral methods

    if direction_method == "mean_diff":
        steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
        desc = "naive mean(pos) - mean(neg)"
    elif direction_method == "pca_0":
        candidates = generate_candidate_directions(pos_ref_np, neg_ref_np, n_random=0, n_pca=5)
        pca_candidates = [(name, d) for name, d in candidates if name.startswith("pca_") and not name.endswith("_neg")]
        if pca_candidates:
            steering_vector = torch.from_numpy(pca_candidates[0][1]).float()
            desc = "PCA component 0 of difference vectors"
        else:
            steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
            desc = "fallback to mean_diff (no PCA components)"
    elif direction_method == "search":
        candidates = generate_candidate_directions(pos_ref_np, neg_ref_np, n_random=10, n_pca=5)
        best_score = -float('inf')
        best_name = None
        best_direction = None
        for name, direction in candidates:
            pos_proj = np.dot(pos_ref_np, direction)
            neg_proj = np.dot(neg_ref_np, direction)
            score = pos_proj.mean() - neg_proj.mean()
            if score > best_score:
                best_score = score
                best_name = name
                best_direction = direction
        steering_vector = torch.from_numpy(best_direction).float()
        desc = f"best candidate: {best_name} (score: {best_score:.4f})"
    elif direction_method == "behavioral":
        if behavioral_activations is not None and behavioral_labels is not None:
            # Train logistic regression on actual behavioral labels
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            clf.fit(behavioral_activations, behavioral_labels)
            direction = clf.coef_[0]
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            steering_vector = torch.from_numpy(direction).float()
            acc = clf.score(behavioral_activations, behavioral_labels)
            desc = f"behavioral direction from actual outputs (acc={acc:.2f})"
        else:
            steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
            desc = "behavioral (no labels provided, using mean_diff)"
    else:
        steering_vector = (pos_ref.mean(dim=0) - neg_ref.mean(dim=0))
        desc = f"unknown method '{direction_method}', using mean_diff"

    return steering_vector, desc, acc


def create_steering_method(
    steering_method_name: str,
    strength: float,
    pos_ref_np: np.ndarray,
    neg_ref_np: np.ndarray,
) -> Any:
    """
    Create and fit a steering method.

    Returns fitted SteeringMethod instance, or None for default linear.
    """
    from wisent.core.geometry.advanced_steering import (
        LinearSteering, ClampingSteering, ProjectionSteering,
        ReplacementSteering, ContrastSteering, MLPSteering, AdaptiveSteering,
    )

    if steering_method_name == "linear":
        method = LinearSteering(strength=strength)
    elif steering_method_name == "clamping":
        method = ClampingSteering(margin=0.1)
    elif steering_method_name == "projection":
        method = ProjectionSteering(n_components=5, strength=1.0)
    elif steering_method_name == "replacement":
        method = ReplacementSteering(blend=0.5)
    elif steering_method_name == "contrast":
        method = ContrastSteering(strength=strength)
    elif steering_method_name == "mlp":
        method = MLPSteering(hidden_dim=256, epochs=100)
    elif steering_method_name == "adaptive":
        method = AdaptiveSteering(max_strength=strength * 2)
    else:
        return None

    method.fit(pos_ref_np, neg_ref_np)
    return method


def load_all_layer_activations(
    model_name: str,
    task_name: str,
    num_layers: int,
    train_ids: set,
    prompt_format: str = "chat",
    extraction_strategy: str = "last_token",
    limit: int = 200,
    database_url: str = None,
    layers: list = None,
) -> Tuple[dict, dict]:
    """
    Load activations for all (or specified) layers from database.

    Returns:
        (pos_by_layer, neg_by_layer) dicts mapping layer number -> numpy activations
    """
    from wisent.core.geometry.repscan_with_concepts import load_activations_from_database

    if layers is None:
        layers = list(range(num_layers))

    pos_by_layer, neg_by_layer = {}, {}
    for layer in layers:
        try:
            pos, neg = load_activations_from_database(
                model_name=model_name, task_name=task_name, layer=layer,
                prompt_format=prompt_format, extraction_strategy=extraction_strategy,
                limit=limit, database_url=database_url, pair_ids=train_ids
            )
            pos_by_layer[layer] = pos.cpu().numpy()
            neg_by_layer[layer] = neg.cpu().numpy()
        except Exception as e:
            print(f"  Warning: Could not load layer {layer}: {e}")
    return pos_by_layer, neg_by_layer


def parse_layer_config(config_str: str) -> dict:
    """Parse 'layer:value,layer:value' format into dict."""
    if not config_str:
        return {}
    result = {}
    for item in config_str.split(','):
        if ':' in item:
            layer, value = item.split(':', 1)
            result[int(layer.strip())] = value.strip()
    return result


def create_all_layer_steering(
    pos_by_layer: dict,
    neg_by_layer: dict,
    default_method: str,
    default_strength: float,
    direction_method: str = "mean_diff",
    layer_strengths: dict = None,
    layer_methods: dict = None,
    behavioral_acts_by_layer: dict = None,
    behavioral_labels: np.ndarray = None,
) -> Tuple[dict, dict, dict]:
    """
    Create steering vectors and methods for all layers with per-layer customization.
    For behavioral direction, pass behavioral_acts_by_layer and behavioral_labels.

    When using behavioral direction, strength is scaled by classifier accuracy:
    - Layers with higher accuracy get higher strength (they predict behavior better)
    - Strength = default_strength * (acc - 0.5) / (max_acc - 0.5) for acc > 0.5
    - Layers with acc <= 0.5 (worse than random) get strength 0

    Returns:
        (steering_vectors_dict, methods_dict, scales_dict)
    """
    steering_vectors = {}
    methods = {}
    scales = {}
    layer_strengths = layer_strengths or {}
    layer_methods = layer_methods or {}
    behavioral_acts_by_layer = behavioral_acts_by_layer or {}

    # First pass: compute directions and accuracies for all layers
    layer_data = {}
    for layer in pos_by_layer.keys():
        pos_acts = pos_by_layer[layer]
        neg_acts = neg_by_layer.get(layer)
        if neg_acts is None:
            continue

        beh_acts = behavioral_acts_by_layer.get(layer)
        steering_vec, desc, acc = select_steering_direction(
            torch.from_numpy(pos_acts).float(),
            torch.from_numpy(neg_acts).float(),
            direction_method,
            behavioral_activations=beh_acts,
            behavioral_labels=behavioral_labels
        )
        layer_data[layer] = {
            "vector": steering_vec,
            "desc": desc,
            "acc": acc,
            "pos_acts": pos_acts,
            "neg_acts": neg_acts,
        }

    # For behavioral method, scale strength by accuracy
    if direction_method == "behavioral" and layer_data:
        max_acc = max(d["acc"] for d in layer_data.values())
        print(f"    Max behavioral accuracy: {max_acc:.3f}")
    else:
        max_acc = 1.0

    # Second pass: create steering with accuracy-scaled strength
    for layer, data in layer_data.items():
        layer_name = f"layer.{layer}"

        # Get base strength (from explicit config or default)
        base_strength = layer_strengths.get(layer, default_strength)
        if isinstance(base_strength, str):
            base_strength = float(base_strength)

        # Use provided strength directly (no hardcoded scaling - tuning happens externally)
        strength = base_strength

        scales[layer_name] = strength
        steering_vectors[layer_name] = data["vector"]

        # Get per-layer method type
        method_name = layer_methods.get(layer, default_method)
        method = create_steering_method(method_name, strength, data["pos_acts"], data["neg_acts"])
        if method is not None:
            methods[layer_name] = method

        acc_str = f", acc={data['acc']:.2f}" if direction_method == "behavioral" else ""
        print(f"    Layer {layer}: {method_name} @ strength {strength:.3f}{acc_str} ({data['desc'][:25]}...)")

    return steering_vectors, methods, scales
