"""Steering direction discovery - find directions that actually improve behavior."""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class DiscoveryResult:
    """Result from a steering direction discovery method."""
    method: str
    direction: np.ndarray
    layer: int
    score: float  # Truthfulness improvement
    details: Dict


def discover_behavioral_direction(
    base_activations: np.ndarray,
    base_labels: List[str],  # "TRUTHFUL" or "UNTRUTHFUL"
    steered_activations: np.ndarray = None,
    steered_labels: List[str] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    Find direction that separates actual truthful from untruthful outputs.

    Instead of using pos/neg reference labels, uses actual generation results.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    # Convert labels to binary
    y = np.array([1 if l == "TRUTHFUL" else 0 for l in base_labels])

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(base_activations)

    # Train logistic regression to find separating direction
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)

    # The coefficient vector is the separating direction
    direction = clf.coef_[0]
    direction = direction / (np.linalg.norm(direction) + 1e-8)

    # Transform back to original scale
    direction_original = direction / (scaler.scale_ + 1e-8)
    direction_original = direction_original / (np.linalg.norm(direction_original) + 1e-8)

    # Compute separation quality
    probs = clf.predict_proba(X_scaled)[:, 1]
    truthful_mean = probs[y == 1].mean() if (y == 1).any() else 0
    untruthful_mean = probs[y == 0].mean() if (y == 0).any() else 0

    details = {
        "accuracy": clf.score(X_scaled, y),
        "truthful_mean_prob": truthful_mean,
        "untruthful_mean_prob": untruthful_mean,
        "separation": truthful_mean - untruthful_mean,
        "n_truthful": int((y == 1).sum()),
        "n_untruthful": int((y == 0).sum()),
    }

    return direction_original, details


def search_directions(
    candidate_directions: List[np.ndarray],
    evaluate_fn: Callable[[np.ndarray], Tuple[int, int]],  # Returns (base_truthful, steered_truthful)
    base_truthful_count: int,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Search through candidate directions to find ones that improve truthfulness.

    Returns list of (direction, base_count, steered_count) sorted by improvement.
    """
    results = []
    for direction in candidate_directions:
        base_count, steered_count = evaluate_fn(direction)
        improvement = steered_count - base_count
        results.append((direction, base_count, steered_count, improvement))

    # Sort by improvement (descending)
    results.sort(key=lambda x: x[3], reverse=True)
    return [(r[0], r[1], r[2]) for r in results]


def generate_candidate_directions(
    pos_activations: np.ndarray,
    neg_activations: np.ndarray,
    n_random: int = 20,
    n_pca: int = 10,
) -> List[Tuple[str, np.ndarray]]:
    """Generate candidate steering directions to search."""
    from sklearn.decomposition import PCA

    candidates = []

    # 1. Mean difference (baseline)
    mean_diff = pos_activations.mean(axis=0) - neg_activations.mean(axis=0)
    mean_diff = mean_diff / (np.linalg.norm(mean_diff) + 1e-8)
    candidates.append(("mean_diff", mean_diff))

    # 2. Negative mean difference
    candidates.append(("neg_mean_diff", -mean_diff))

    # 3. PCA components of the difference vectors
    n_pairs = min(len(pos_activations), len(neg_activations))
    diffs = pos_activations[:n_pairs] - neg_activations[:n_pairs]
    n_pca_actual = min(n_pca, n_pairs - 1, diffs.shape[1])
    if n_pca_actual > 0:
        pca = PCA(n_components=n_pca_actual, random_state=42)
        pca.fit(diffs)
        for i, comp in enumerate(pca.components_):
            comp_norm = comp / (np.linalg.norm(comp) + 1e-8)
            candidates.append((f"pca_{i}", comp_norm))
            candidates.append((f"pca_{i}_neg", -comp_norm))

    # 4. Random directions in the subspace spanned by activations
    all_acts = np.vstack([pos_activations, neg_activations])
    pca_all = PCA(n_components=min(50, len(all_acts) - 1), random_state=42)
    pca_all.fit(all_acts)

    rng = np.random.RandomState(42)
    for i in range(n_random):
        # Random combination of top PCA components
        weights = rng.randn(min(20, len(pca_all.components_)))
        direction = np.zeros(all_acts.shape[1])
        for j, w in enumerate(weights):
            direction += w * pca_all.components_[j]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        candidates.append((f"random_{i}", direction))

    # 5. Per-class centroids to overall centroid
    pos_centroid = pos_activations.mean(axis=0)
    neg_centroid = neg_activations.mean(axis=0)
    overall_centroid = all_acts.mean(axis=0)

    pos_to_overall = overall_centroid - pos_centroid
    pos_to_overall = pos_to_overall / (np.linalg.norm(pos_to_overall) + 1e-8)
    candidates.append(("pos_to_center", pos_to_overall))

    neg_to_overall = overall_centroid - neg_centroid
    neg_to_overall = neg_to_overall / (np.linalg.norm(neg_to_overall) + 1e-8)
    candidates.append(("neg_to_center", neg_to_overall))

    return candidates


def search_layers(
    adapter,
    prompts: List[str],
    pos_ref_by_layer: Dict[int, np.ndarray],
    neg_ref_by_layer: Dict[int, np.ndarray],
    evaluate_fn: Callable[[str], str],  # Returns "TRUTHFUL" or "UNTRUTHFUL"
    layers: List[int],
    strength: float = 1.0,
) -> List[Tuple[int, int, int, float]]:
    """
    Search across layers to find which layer steering actually improves behavior.

    Returns list of (layer, base_truthful, steered_truthful, improvement_pct) sorted by improvement.
    """
    from wisent.core.activations.core.atoms import LayerActivations
    from wisent.core.adapters.base import SteeringConfig

    results = []

    for layer in layers:
        if layer not in pos_ref_by_layer or layer not in neg_ref_by_layer:
            continue

        pos_ref = pos_ref_by_layer[layer]
        neg_ref = neg_ref_by_layer[layer]

        # Compute steering vector for this layer
        steering_vec = torch.from_numpy(pos_ref.mean(axis=0) - neg_ref.mean(axis=0)).float()
        layer_name = f"layer.{layer}"
        steering_vectors = LayerActivations({layer_name: strength * steering_vec})

        base_truthful = 0
        steered_truthful = 0

        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            formatted = adapter.apply_chat_template(messages, add_generation_prompt=True)

            # Base response
            base_resp = adapter._generate_unsteered(formatted, max_new_tokens=100, temperature=0.1)
            base_resp = _extract_response(base_resp)
            base_eval = evaluate_fn(base_resp)
            if base_eval == "TRUTHFUL":
                base_truthful += 1

            # Steered response
            steered_resp = adapter.forward_with_steering(
                formatted, steering_vectors=steering_vectors, config=SteeringConfig(scale=1.0)
            )
            steered_resp = _extract_response(steered_resp)
            steered_eval = evaluate_fn(steered_resp)
            if steered_eval == "TRUTHFUL":
                steered_truthful += 1

        improvement = (steered_truthful - base_truthful) / len(prompts) * 100
        results.append((layer, base_truthful, steered_truthful, improvement))

    results.sort(key=lambda x: x[3], reverse=True)
    return results


def extract_generation_activations(
    adapter,
    prompts: List[str],
    responses: List[str],
    labels: List[str],
    layer: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract activations at the moment of generating responses.

    Returns (truthful_activations, untruthful_activations) from actual generations.
    """
    layer_name = f"layer.{layer}"
    truthful_acts = []
    untruthful_acts = []

    for prompt, response, label in zip(prompts, responses, labels):
        # Get activation at the point where generation starts
        messages = [{"role": "user", "content": prompt}]
        formatted = adapter.apply_chat_template(messages, add_generation_prompt=True)

        acts = adapter.extract_activations(formatted, layers=[layer_name])
        act = acts.get(layer_name)
        if act is not None:
            last_token_act = act[0, -1, :].cpu().numpy()
            if label == "TRUTHFUL":
                truthful_acts.append(last_token_act)
            else:
                untruthful_acts.append(last_token_act)

    return np.array(truthful_acts), np.array(untruthful_acts)


def compute_generation_direction(
    truthful_acts: np.ndarray,
    untruthful_acts: np.ndarray,
) -> Tuple[np.ndarray, Dict]:
    """Compute steering direction from generation-time activations."""
    if len(truthful_acts) == 0 or len(untruthful_acts) == 0:
        return None, {"error": "Not enough samples"}

    direction = truthful_acts.mean(axis=0) - untruthful_acts.mean(axis=0)
    norm = np.linalg.norm(direction)
    direction = direction / (norm + 1e-8)

    details = {
        "n_truthful": len(truthful_acts),
        "n_untruthful": len(untruthful_acts),
        "direction_norm": float(norm),
    }

    return direction, details


def _extract_response(raw_response: str) -> str:
    """Extract assistant response from raw generation."""
    if "assistant\n\n" in raw_response:
        return raw_response.split("assistant\n\n", 1)[-1].strip()
    elif "assistant\n" in raw_response:
        return raw_response.split("assistant\n", 1)[-1].strip()
    return raw_response


def compare_directions(
    original_direction: np.ndarray,
    discovered_directions: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compare discovered directions to original mean-diff direction."""
    similarities = {}
    for name, direction in discovered_directions.items():
        cos_sim = np.dot(original_direction, direction)
        similarities[name] = float(cos_sim)
    return similarities
