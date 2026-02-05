"""
Layer and position metrics for activation geometry.

Metrics for cross-layer consistency and token position dependence.
"""

import torch
import numpy as np
from typing import Dict, Any, Tuple


def compute_cross_layer_consistency(
    activations_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, Any]:
    """
    Compute consistency of steering direction across layers.

    Describes:
    - Do different layers have similar steering directions?
    - Which layers agree/disagree?
    - Is there a consistent "concept" across layers?
    """
    layers = sorted(activations_by_layer.keys())

    if len(layers) < 2:
        return {"error": "need at least 2 layers"}

    # Compute steering direction for each layer
    directions = {}
    for layer in layers:
        pos, neg = activations_by_layer[layer]
        pos = pos.float().cpu().numpy()
        neg = neg.float().cpu().numpy()
        n = min(len(pos), len(neg))

        mean_diff = (pos[:n] - neg[:n]).mean(axis=0)
        norm = np.linalg.norm(mean_diff)

        if norm > 1e-8:
            directions[layer] = mean_diff / norm
        else:
            directions[layer] = None

    # Compute pairwise cosine similarities between layers
    layer_pairs = []
    similarities = []

    for i, l1 in enumerate(layers):
        for l2 in layers[i+1:]:
            if directions[l1] is not None and directions[l2] is not None:
                # Project to common dimension if needed
                d1, d2 = directions[l1], directions[l2]
                if len(d1) == len(d2):
                    sim = float(np.dot(d1, d2))
                    layer_pairs.append((l1, l2))
                    similarities.append(sim)

    similarities = np.array(similarities)

    # Find most/least consistent layer pairs
    if len(similarities) > 0:
        most_similar_idx = np.argmax(similarities)
        least_similar_idx = np.argmin(similarities)

        # Per-layer consistency (average similarity with other layers)
        per_layer_consistency = {}
        for layer in layers:
            if directions[layer] is not None:
                layer_sims = []
                for other in layers:
                    if other != layer and directions[other] is not None:
                        d1, d2 = directions[layer], directions[other]
                        if len(d1) == len(d2):
                            layer_sims.append(np.dot(d1, d2))
                if layer_sims:
                    per_layer_consistency[layer] = float(np.mean(layer_sims))
    else:
        most_similar_idx = None
        least_similar_idx = None
        per_layer_consistency = {}

    return {
        "n_layers": len(layers),
        "layers": layers,

        # Overall consistency
        "mean_cross_layer_similarity": float(similarities.mean()) if len(similarities) > 0 else None,
        "std_cross_layer_similarity": float(similarities.std()) if len(similarities) > 0 else None,
        "min_cross_layer_similarity": float(similarities.min()) if len(similarities) > 0 else None,
        "max_cross_layer_similarity": float(similarities.max()) if len(similarities) > 0 else None,

        # Most/least similar pairs
        "most_similar_pair": layer_pairs[most_similar_idx] if most_similar_idx is not None else None,
        "most_similar_value": float(similarities[most_similar_idx]) if most_similar_idx is not None else None,
        "least_similar_pair": layer_pairs[least_similar_idx] if least_similar_idx is not None else None,
        "least_similar_value": float(similarities[least_similar_idx]) if least_similar_idx is not None else None,

        # Per-layer
        "per_layer_consistency": per_layer_consistency,

        # All pairwise similarities
        "pairwise_similarities": {f"{p[0]}-{p[1]}": float(s) for p, s in zip(layer_pairs, similarities)},
    }


def compute_token_position_metrics(
    pos_activations_by_position: Dict[int, torch.Tensor],
    neg_activations_by_position: Dict[int, torch.Tensor],
) -> Dict[str, Any]:
    """
    Compute token position dependence metrics.

    Describes:
    - Is the signal at all positions or just certain ones?
    - How does signal strength vary by position?

    Args:
        pos_activations_by_position: Dict mapping position -> activations
        neg_activations_by_position: Dict mapping position -> activations
    """
    positions = sorted(set(pos_activations_by_position.keys()) & set(neg_activations_by_position.keys()))

    if len(positions) < 2:
        return {"error": "need at least 2 positions"}

    # Compute signal strength at each position
    from sklearn.linear_model import LogisticRegression

    position_metrics = {}
    directions = {}

    for pos_idx in positions:
        pos = pos_activations_by_position[pos_idx].float().cpu().numpy()
        neg = neg_activations_by_position[pos_idx].float().cpu().numpy()
        n = min(len(pos), len(neg))

        if n < 5:
            continue

        # Linear probe accuracy
        X = np.vstack([pos[:n], neg[:n]])
        y = np.array([1] * n + [0] * n)

        try:
            clf = LogisticRegression( random_state=42)
            clf.fit(X, y)
            accuracy = clf.score(X, y)
        except:
            accuracy = 0.5

        # Steering direction
        mean_diff = (pos[:n] - neg[:n]).mean(axis=0)
        norm = np.linalg.norm(mean_diff)

        position_metrics[pos_idx] = {
            "linear_accuracy": float(accuracy),
            "steering_norm": float(norm),
        }

        if norm > 1e-8:
            directions[pos_idx] = mean_diff / norm

    # Cross-position consistency
    position_list = list(directions.keys())
    if len(position_list) >= 2:
        cross_pos_sims = []
        for i, p1 in enumerate(position_list):
            for p2 in position_list[i+1:]:
                if len(directions[p1]) == len(directions[p2]):
                    cross_pos_sims.append(np.dot(directions[p1], directions[p2]))
        cross_pos_sims = np.array(cross_pos_sims)
    else:
        cross_pos_sims = np.array([])

    # Find best position
    if position_metrics:
        best_position = max(position_metrics.keys(), key=lambda p: position_metrics[p]["linear_accuracy"])
    else:
        best_position = None

    return {
        "n_positions": len(positions),
        "positions": positions,
        "per_position_metrics": position_metrics,

        # Cross-position consistency
        "cross_position_similarity_mean": float(cross_pos_sims.mean()) if len(cross_pos_sims) > 0 else None,
        "cross_position_similarity_std": float(cross_pos_sims.std()) if len(cross_pos_sims) > 0 else None,

        # Best position
        "best_position": best_position,
        "best_position_accuracy": position_metrics[best_position]["linear_accuracy"] if best_position else None,
    }
