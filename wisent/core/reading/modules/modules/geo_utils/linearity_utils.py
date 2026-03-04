"""Linearity validation helpers for the Linear Representation Hypothesis."""
from typing import Tuple, Dict, List
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from wisent.core.utils.config_tools.constants import (ZERO_THRESHOLD, DEFAULT_RANDOM_SEED, CONFIDENCE_LEVEL, N_COMPONENTS_2D)

def _prepare_data(
    pos: torch.Tensor, neg: torch.Tensor,
    linearity_max_pairs: int = None, linearity_pca_components: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tensors to numpy, subsample if large, apply PCA, create labels."""
    for _n, _v in [("linearity_max_pairs", linearity_max_pairs), ("linearity_pca_components", linearity_pca_components)]:
        if _v is None: raise ValueError(f"{_n} is required")
    from sklearn.decomposition import PCA
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg
    import sys; print(f"  [TRACE] _prepare_data: {len(pos_np)} pairs, {pos_np.shape[1]} dims", file=sys.stderr, flush=True)
    if len(pos_np) > linearity_max_pairs:
        idx = np.random.RandomState(DEFAULT_RANDOM_SEED).choice(len(pos_np), linearity_max_pairs, replace=False)
        idx.sort()
        pos_np, neg_np = pos_np[idx], neg_np[idx]
    X = np.vstack([pos_np, neg_np])
    y = np.array([1] * len(pos_np) + [0] * len(neg_np))
    n_samples, n_features = X.shape
    pca_dims = min(n_samples - 1, n_features, linearity_pca_components)
    if pca_dims < n_features and pca_dims >= 2:
        X = PCA(n_components=pca_dims, random_state=DEFAULT_RANDOM_SEED).fit_transform(X)
    return X, y

def compute_probe_accuracies(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_splits: int = None,
    random_state: int = DEFAULT_RANDOM_SEED,
    linearity_hidden_dim_large: int = None,
    linearity_max_pairs: int = None,
    linearity_pca_components: int = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute linear and nonlinear probe accuracies with cross-validation."""
    for _n, _v in [("linearity_hidden_dim_large", linearity_hidden_dim_large), ("n_splits", n_splits)]:
        if _v is None: raise ValueError(f"{_n} is required")
    X, y = _prepare_data(pos, neg, linearity_max_pairs=linearity_max_pairs, linearity_pca_components=linearity_pca_components)
    n_per_class = min(len(pos), len(neg))
    n_splits = max(2, min(n_splits, n_per_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    linear_model = LogisticRegression(solver="lbfgs", random_state=random_state)
    nonlinear_model = MLPClassifier(
        hidden_layer_sizes=(linearity_hidden_dim_large,), random_state=random_state
    )

    linear_scores = cross_val_score(linear_model, X, y, cv=cv, scoring="accuracy")
    nonlinear_scores = cross_val_score(nonlinear_model, X, y, cv=cv, scoring="accuracy")

    return (
        float(np.mean(linear_scores)),
        float(np.mean(nonlinear_scores)),
        linear_scores,
        nonlinear_scores,
    )

def analyze_residuals(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_clusters: int = None,
    random_state: int = DEFAULT_RANDOM_SEED,
    linearity_nonlinear_error: float = None,
    linearity_n_init: int = None,
    linearity_max_pairs: int = None,
    linearity_pca_components: int = None,
) -> Dict[str, float]:
    """Analyze linear probe residuals for systematic patterns.

    If linear model errors cluster (high silhouette), this indicates
    the linear assumption fails in predictable regions of the space.
    """
    for _n, _v in [("n_clusters", n_clusters), ("linearity_nonlinear_error", linearity_nonlinear_error), ("linearity_n_init", linearity_n_init)]:
        if _v is None: raise ValueError(f"{_n} is required")
    X, y = _prepare_data(pos, neg, linearity_max_pairs=linearity_max_pairs, linearity_pca_components=linearity_pca_components)

    model = LogisticRegression(solver="lbfgs", random_state=random_state)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    errors = np.abs(y - probs)

    error_mask = errors > linearity_nonlinear_error
    if error_mask.sum() < n_clusters * 2:
        return {
            "residual_silhouette": 0.0,
            "n_high_error": int(error_mask.sum()),
            "clusters_found": False,
        }

    high_error_X = X[error_mask]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=linearity_n_init)
    cluster_labels = kmeans.fit_predict(high_error_X)

    if len(np.unique(cluster_labels)) > 1:
        sil = silhouette_score(high_error_X, cluster_labels)
    else:
        sil = 0.0

    return {
        "residual_silhouette": float(sil),
        "n_high_error": int(error_mask.sum()),
        "clusters_found": True,
    }

def bootstrap_gap_ci(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_bootstrap: int = None,
    ci_level: float = CONFIDENCE_LEVEL,
    random_state: int = DEFAULT_RANDOM_SEED,
    linearity_hidden_dim_small: int = None,
    linearity_mlp_max_iter: int = None,
    linearity_min_gaps: int = None,
    linearity_max_pairs: int = None,
    linearity_pca_components: int = None,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval on linear-nonlinear gap."""
    for _n, _v in [("n_bootstrap", n_bootstrap), ("linearity_hidden_dim_small", linearity_hidden_dim_small), ("linearity_mlp_max_iter", linearity_mlp_max_iter), ("linearity_min_gaps", linearity_min_gaps)]:
        if _v is None: raise ValueError(f"{_n} is required")
    rng = np.random.RandomState(random_state)
    X, y = _prepare_data(pos, neg, linearity_max_pairs=linearity_max_pairs, linearity_pca_components=linearity_pca_components)
    n = len(X)

    gaps = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        linear = LogisticRegression(solver="lbfgs", random_state=DEFAULT_RANDOM_SEED)
        nonlinear = MLPClassifier(hidden_layer_sizes=(linearity_hidden_dim_small,), max_iter=linearity_mlp_max_iter, random_state=DEFAULT_RANDOM_SEED)

        try:
            linear.fit(X_boot, y_boot)
            nonlinear.fit(X_boot, y_boot)

            linear_acc = linear.score(X_boot, y_boot)
            nonlinear_acc = nonlinear.score(X_boot, y_boot)
            gaps.append(nonlinear_acc - linear_acc)
        except Exception:
            continue

    if len(gaps) < linearity_min_gaps:
        return 0.0, 0.0, 0.0

    gaps = np.array(gaps)
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(gaps, 100 * alpha / 2))
    ci_upper = float(np.percentile(gaps, 100 * (1 - alpha / 2)))

    return float(np.mean(gaps)), ci_lower, ci_upper

def test_cross_context_linearity(
    contexts: List[Tuple[torch.Tensor, torch.Tensor]],
    random_state: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, float]:
    """Test if linear directions transfer across contexts (Lampinen-style).

    Each context is a (pos, neg) tuple from different conversational contexts.
    Trains on one context, tests on others to see if directions generalize.
    """
    if len(contexts) < 2:
        return {
            "transfer_accuracy": 0.0,
            "n_contexts": len(contexts),
            "sufficient_contexts": False,
        }

    transfer_scores = []

    for i, (pos_train, neg_train) in enumerate(contexts):
        X_train, y_train = _prepare_data(pos_train, neg_train)

        model = LogisticRegression(solver="lbfgs", random_state=random_state)
        model.fit(X_train, y_train)

        for j, (pos_test, neg_test) in enumerate(contexts):
            if i == j:
                continue
            X_test, y_test = _prepare_data(pos_test, neg_test)
            acc = model.score(X_test, y_test)
            transfer_scores.append(acc)

    return {
        "transfer_accuracy": float(np.mean(transfer_scores)),
        "transfer_std": float(np.std(transfer_scores)),
        "n_contexts": len(contexts),
        "sufficient_contexts": True,
    }

def ramsey_polynomial_test(
    pos: torch.Tensor, neg: torch.Tensor,
    degree: int = None, n_features: int = None, random_state: int = DEFAULT_RANDOM_SEED,
    linearity_reg_c: float = None, linearity_max_pairs: int = None, linearity_pca_components: int = None,
    *, cv_folds: int,
) -> Dict[str, float]:
    """Ramsey-style test: do polynomial features significantly improve fit?

    If polynomial features don't help, the relationship is likely linear.
    Uses PCA to reduce dimensionality before polynomial expansion.
    """
    for _n, _v in [("degree", degree), ("n_features", n_features), ("linearity_reg_c", linearity_reg_c)]:
        if _v is None: raise ValueError(f"{_n} is required")
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.decomposition import PCA
    X, y = _prepare_data(pos, neg, linearity_max_pairs=linearity_max_pairs, linearity_pca_components=linearity_pca_components)

    n_components = min(n_features, X.shape[1], len(X) // 2)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    n_per_class = min(len(pos), len(neg)); _ns = max(N_COMPONENTS_2D, min(cv_folds, n_per_class))
    cv = StratifiedKFold(n_splits=_ns, shuffle=True, random_state=random_state)
    linear = LogisticRegression(solver="lbfgs", random_state=random_state)
    linear_scores = cross_val_score(linear, X_reduced, y, cv=cv, scoring="accuracy")
    linear_acc = float(np.mean(linear_scores))

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_reduced)

    poly_model = LogisticRegression(solver="lbfgs", random_state=random_state, C=linearity_reg_c)
    poly_scores = cross_val_score(poly_model, X_poly, y, cv=cv, scoring="accuracy")
    poly_acc = float(np.mean(poly_scores))

    improvement = poly_acc - linear_acc

    return {
        "linear_accuracy": linear_acc,
        "polynomial_accuracy": poly_acc,
        "improvement": improvement,
        "polynomial_degree": degree,
        "n_pca_components": n_components,
    }

def regression_probe(
    pos: torch.Tensor, neg: torch.Tensor, target: np.ndarray,
    n_splits: int = None, random_state: int = DEFAULT_RANDOM_SEED,
    linearity_ridge_alpha: float = None, linearity_bootstrap_iter: int = None,
    linearity_z_score_threshold: float = None,
) -> Dict[str, float]:
    """Regression probe for predicting continuous outcomes."""
    for _n, _v in [("n_splits", n_splits), ("linearity_ridge_alpha", linearity_ridge_alpha), ("linearity_bootstrap_iter", linearity_bootstrap_iter), ("linearity_z_score_threshold", linearity_z_score_threshold)]:
        if _v is None: raise ValueError(f"{_n} is required")
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold
    from scipy import stats

    diff = pos.cpu().numpy() - neg.cpu().numpy() if isinstance(pos, torch.Tensor) else pos - neg

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = Ridge(alpha=linearity_ridge_alpha)

    r2_scores = cross_val_score(model, diff, target, cv=cv, scoring="r2")
    r2_mean = float(np.mean(r2_scores))

    model.fit(diff, target)
    preds = model.predict(diff)
    corr, p_value = stats.pearsonr(preds, target)

    rng = np.random.RandomState(random_state)
    null_r2s = []
    for _ in range(linearity_bootstrap_iter):
        target_perm = rng.permutation(target)
        null_scores = cross_val_score(model, diff, target_perm, cv=cv, scoring="r2")
        null_r2s.append(np.mean(null_scores))

    null_mean = float(np.mean(null_r2s))
    null_std = float(np.std(null_r2s))
    z_score = (r2_mean - null_mean) / (null_std + ZERO_THRESHOLD)

    return {
        "r2_mean": r2_mean,
        "r2_std": float(np.std(r2_scores)),
        "correlation": float(corr),
        "correlation_p_value": float(p_value),
        "null_r2_mean": null_mean,
        "z_score_vs_null": z_score,
        "significant": z_score > linearity_z_score_threshold,
    }
