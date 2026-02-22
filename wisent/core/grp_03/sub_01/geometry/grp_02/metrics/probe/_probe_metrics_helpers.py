"""Extracted from probe_metrics.py - compute_knn_pacmap_accuracy."""

import numpy as np
import torch


def compute_knn_pacmap_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
    n_components: int = 10,
    n_folds: int = 5,
) -> float:
    """Compute k-NN accuracy on PaCMAP-reduced features.

    PaCMAP preserves both local AND global structure better than UMAP.

    Args:
        pos_activations: Positive class activation tensors
        neg_activations: Negative class activation tensors
        k: Number of nearest neighbors
        n_components: Number of PaCMAP components
        n_folds: Number of cross-validation folds

    Returns:
        Mean cross-validated accuracy, or 0.5 on failure
    """
    try:
        import pacmap
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        n_pos = len(pos_activations)
        n_neg = len(neg_activations)

        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5

        X = torch.cat(
            [pos_activations, neg_activations], dim=0
        ).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)

        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5

        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=min(10, len(X) // 4),
            MN_ratio=0.5,
            FP_ratio=2.0,
            random_state=42,
        )
        X_pacmap = reducer.fit_transform(X)

        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(
            clf, X_pacmap, y, cv=n_folds, scoring='accuracy'
        )
        return float(scores.mean())
    except ImportError:
        return 0.5
    except Exception:
        return 0.5
