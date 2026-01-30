"""
Multi-direction accuracy analysis for steering vectors.

Tests how many separation directions are needed for good classification.
"""

import numpy as np
from typing import Dict, Any, List


def compute_multi_direction_accuracy(
    pos_activations,
    neg_activations,
    k_values: List[int] = [1, 2, 3, 5, 10],
    n_folds: int = 5,
    n_bootstrap: int = 50,
) -> Dict[str, Any]:
    """
    Test how many separation directions are needed for good classification.

    Uses bootstrap + SVD to find multiple separation directions:
    1. Bootstrap N subsets of pairs
    2. Compute diff-mean for each subset (each is a separation direction)
    3. SVD on matrix of diff-means -> principal separation directions
    4. Test linear probe accuracy using top-k directions

    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k_values: List of k values to test (number of directions)
        n_folds: Number of CV folds
        n_bootstrap: Number of bootstrap samples for direction discovery

    Returns:
        Dict with:
            - accuracy_by_k: {k: accuracy} for each k
            - min_k_for_good: minimum k where accuracy >= 0.6
            - saturation_k: k where accuracy stops improving significantly
            - gain_from_multi: accuracy(best_k) - accuracy(k=1)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score

        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        n_pairs = min(n_pos, n_neg)

        if n_pairs < 5:
            return {
                "accuracy_by_k": {k: 0.5 for k in k_values},
                "min_k_for_good": -1,
                "saturation_k": 1,
                "gain_from_multi": 0.0,
            }

        pos_np = pos_activations.float().cpu().numpy()
        neg_np = neg_activations.float().cpu().numpy()

        rng = np.random.RandomState(42)
        subset_size = max(n_pairs // 2, 3)

        diff_means = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_pairs, size=subset_size, replace=False)
            pos_subset = pos_np[indices]
            neg_subset = neg_np[indices]

            diff_mean = pos_subset.mean(axis=0) - neg_subset.mean(axis=0)
            norm = np.linalg.norm(diff_mean)
            if norm > 1e-8:
                diff_means.append(diff_mean / norm)

        if len(diff_means) < 2:
            return {
                "accuracy_by_k": {k: 0.5 for k in k_values},
                "min_k_for_good": -1,
                "saturation_k": 1,
                "gain_from_multi": 0.0,
            }

        diff_matrix = np.stack(diff_means, axis=0)
        U, S, Vh = np.linalg.svd(diff_matrix, full_matrices=False)

        max_k = min(max(k_values), len(S), Vh.shape[0])
        if max_k < 1:
            return {
                "accuracy_by_k": {k: 0.5 for k in k_values},
                "min_k_for_good": -1,
                "saturation_k": 1,
                "gain_from_multi": 0.0,
            }

        X_full = np.vstack([pos_np, neg_np])
        y = np.array([1] * n_pos + [0] * n_neg)

        n_folds_actual = min(n_folds, min(n_pos, n_neg))
        if n_folds_actual < 2:
            n_folds_actual = 2

        accuracy_by_k = {}

        for k in k_values:
            if k > max_k:
                accuracy_by_k[k] = accuracy_by_k.get(max_k, 0.5)
                continue

            top_k_directions = Vh[:k]
            X_projected = X_full @ top_k_directions.T

            clf = LogisticRegression(max_iter=1000, solver='lbfgs')
            try:
                scores = cross_val_score(clf, X_projected, y, cv=n_folds_actual, scoring='accuracy')
                accuracy_by_k[k] = float(scores.mean())
            except Exception:
                accuracy_by_k[k] = 0.5

        min_k_for_good = -1
        for k in sorted(k_values):
            if accuracy_by_k.get(k, 0) >= 0.6:
                min_k_for_good = k
                break

        sorted_ks = sorted([k for k in k_values if k <= max_k])
        saturation_k = sorted_ks[0] if sorted_ks else 1

        for i in range(1, len(sorted_ks)):
            k_prev = sorted_ks[i-1]
            k_curr = sorted_ks[i]
            improvement = accuracy_by_k.get(k_curr, 0) - accuracy_by_k.get(k_prev, 0)
            if improvement < 0.02:
                saturation_k = k_prev
                break
            saturation_k = k_curr

        acc_k1 = accuracy_by_k.get(1, 0.5)
        best_acc = max(accuracy_by_k.values()) if accuracy_by_k else 0.5
        gain_from_multi = best_acc - acc_k1

        return {
            "accuracy_by_k": accuracy_by_k,
            "min_k_for_good": min_k_for_good,
            "saturation_k": saturation_k,
            "gain_from_multi": gain_from_multi,
        }
    except Exception:
        return {
            "accuracy_by_k": {k: 0.5 for k in k_values},
            "min_k_for_good": -1,
            "saturation_k": 1,
            "gain_from_multi": 0.0,
        }
