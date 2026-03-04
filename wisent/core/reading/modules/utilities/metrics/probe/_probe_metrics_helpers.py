"""Extracted from probe_metrics.py - compute_knn_pacmap_accuracy."""

import numpy as np
import torch
from wisent.core import constants as _C


def compute_knn_pacmap_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    knn_default_k: int,
    *,
    pacmap_n_components: int,
    pacmap_neighbors_max: int,
    pacmap_neighbors_divisor: int,
    pacmap_mn_ratio: float,
    pacmap_fp_ratio: float,
    adaptive_cv_min_folds: int,
    knn_min_class_offset: int,
    cv_folds: int,
) -> float:
    """Compute k-NN accuracy on PaCMAP-reduced features."""
    try:
        import pacmap
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score

        n_pos = len(pos_activations)
        n_neg = len(neg_activations)

        if n_pos < knn_default_k + knn_min_class_offset or n_neg < knn_default_k + knn_min_class_offset:
            return _C.CHANCE_LEVEL_ACCURACY

        X = torch.cat(
            [pos_activations, neg_activations], dim=_C.BINARY_CLASS_NEGATIVE
        ).float().cpu().numpy()
        y = np.array([_C.BINARY_CLASS_POSITIVE] * n_pos + [_C.BINARY_CLASS_NEGATIVE] * n_neg)

        n_folds = min(cv_folds, min(n_pos, n_neg))
        if n_folds < adaptive_cv_min_folds:
            return _C.CHANCE_LEVEL_ACCURACY

        reducer = pacmap.PaCMAP(
            n_components=pacmap_n_components,
            n_neighbors=min(pacmap_neighbors_max, len(X) // pacmap_neighbors_divisor),
            MN_ratio=pacmap_mn_ratio,
            FP_ratio=pacmap_fp_ratio,
            random_state=_C.DEFAULT_RANDOM_SEED,
        )
        X_pacmap = reducer.fit_transform(X)

        clf = KNeighborsClassifier(n_neighbors=knn_default_k)
        scores = cross_val_score(
            clf, X_pacmap, y, cv=n_folds, scoring='accuracy'
        )
        return float(scores.mean())
    except ImportError:
        return _C.CHANCE_LEVEL_ACCURACY
    except Exception:
        return _C.CHANCE_LEVEL_ACCURACY
