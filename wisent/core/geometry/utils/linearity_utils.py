"""Helper functions for rigorous linearity validation.

These utilities implement econometric-style diagnostics for testing
the Linear Representation Hypothesis (LRH).
"""
from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _prepare_data(
    pos: torch.Tensor, neg: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert tensors to numpy and create labels."""
    pos_np = pos.cpu().numpy() if isinstance(pos, torch.Tensor) else pos
    neg_np = neg.cpu().numpy() if isinstance(neg, torch.Tensor) else neg
    X = np.vstack([pos_np, neg_np])
    y = np.array([1] * len(pos_np) + [0] * len(neg_np))
    return X, y


def compute_probe_accuracies(
    pos: torch.Tensor,
    neg: torch.Tensor,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute linear and nonlinear probe accuracies with cross-validation.

    Returns:
        Tuple of (linear_acc, nonlinear_acc, linear_scores, nonlinear_scores)
        where scores are per-fold accuracies for statistical testing.
    """
    X, y = _prepare_data(pos, neg)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    linear_model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    nonlinear_model = MLPClassifier(
        hidden_layer_sizes=(64,), max_iter=500, random_state=random_state
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
    n_clusters: int = 3,
    random_state: int = 42,
) -> Dict[str, float]:
    """Analyze linear probe residuals for systematic patterns.

    If linear model errors cluster (high silhouette), this indicates
    the linear assumption fails in predictable regions of the space.

    Returns:
        Dictionary with residual_silhouette and cluster info.
    """
    X, y = _prepare_data(pos, neg)

    model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    model.fit(X, y)

    probs = model.predict_proba(X)[:, 1]
    errors = np.abs(y - probs)

    error_mask = errors > 0.3
    if error_mask.sum() < n_clusters * 2:
        return {
            "residual_silhouette": 0.0,
            "n_high_error": int(error_mask.sum()),
            "clusters_found": False,
        }

    high_error_X = X[error_mask]

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
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
    n_bootstrap: int = 100,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval on linear-nonlinear gap.

    Returns:
        Tuple of (gap_mean, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    X, y = _prepare_data(pos, neg)
    n = len(X)

    gaps = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        X_boot, y_boot = X[idx], y[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        linear = LogisticRegression(max_iter=500, solver="lbfgs", random_state=42)
        nonlinear = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=42)

        try:
            linear.fit(X_boot, y_boot)
            nonlinear.fit(X_boot, y_boot)

            linear_acc = linear.score(X_boot, y_boot)
            nonlinear_acc = nonlinear.score(X_boot, y_boot)
            gaps.append(nonlinear_acc - linear_acc)
        except Exception:
            continue

    if len(gaps) < 10:
        return 0.0, 0.0, 0.0

    gaps = np.array(gaps)
    alpha = 1 - ci_level
    ci_lower = float(np.percentile(gaps, 100 * alpha / 2))
    ci_upper = float(np.percentile(gaps, 100 * (1 - alpha / 2)))

    return float(np.mean(gaps)), ci_lower, ci_upper


def test_cross_context_linearity(
    contexts: List[Tuple[torch.Tensor, torch.Tensor]],
    random_state: int = 42,
) -> Dict[str, float]:
    """Test if linear directions transfer across contexts (Lampinen-style).

    Each context is a (pos, neg) tuple from different conversational contexts.
    Trains on one context, tests on others to see if directions generalize.

    Returns:
        Dictionary with transfer_accuracy and generalization metrics.
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

        model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
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
    pos: torch.Tensor,
    neg: torch.Tensor,
    degree: int = 2,
    n_features: int = 50,
    random_state: int = 42,
) -> Dict[str, float]:
    """Ramsey-style test: do polynomial features significantly improve fit?

    If polynomial features don't help, the relationship is likely linear.
    Uses PCA to reduce dimensionality before polynomial expansion.

    Returns:
        Dictionary with linear_acc, poly_acc, and improvement metrics.
    """
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.decomposition import PCA

    X, y = _prepare_data(pos, neg)

    n_components = min(n_features, X.shape[1], len(X) // 2)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_reduced = pca.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    linear = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    linear_scores = cross_val_score(linear, X_reduced, y, cv=cv, scoring="accuracy")
    linear_acc = float(np.mean(linear_scores))

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_reduced)

    poly_model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state, C=0.1)
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
    pos: torch.Tensor,
    neg: torch.Tensor,
    target: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, float]:
    """Regression probe for predicting continuous outcomes.

    Stronger than classification for predicting actual behavior magnitudes.
    Uses difference vectors to predict continuous target (e.g., toxicity score).

    Args:
        pos: Positive activations [n_samples, dim]
        neg: Negative activations [n_samples, dim]
        target: Continuous target values [n_samples]

    Returns:
        Dictionary with R², correlation, and comparison to null.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score, KFold
    from scipy import stats

    diff = pos.cpu().numpy() - neg.cpu().numpy() if isinstance(pos, torch.Tensor) else pos - neg

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = Ridge(alpha=1.0)

    r2_scores = cross_val_score(model, diff, target, cv=cv, scoring="r2")
    r2_mean = float(np.mean(r2_scores))

    model.fit(diff, target)
    preds = model.predict(diff)
    corr, p_value = stats.pearsonr(preds, target)

    rng = np.random.RandomState(random_state)
    null_r2s = []
    for _ in range(50):
        target_perm = rng.permutation(target)
        null_scores = cross_val_score(model, diff, target_perm, cv=cv, scoring="r2")
        null_r2s.append(np.mean(null_scores))

    null_mean = float(np.mean(null_r2s))
    null_std = float(np.std(null_r2s))
    z_score = (r2_mean - null_mean) / (null_std + 1e-10)

    return {
        "r2_mean": r2_mean,
        "r2_std": float(np.std(r2_scores)),
        "correlation": float(corr),
        "correlation_p_value": float(p_value),
        "null_r2_mean": null_mean,
        "z_score_vs_null": z_score,
        "significant": z_score > 2.0,
    }
