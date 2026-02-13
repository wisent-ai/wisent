"""Null distribution tests for RepScan signal validation."""

from typing import Dict, List, Tuple
import torch
import numpy as np

from ...metrics.probe.signal_metrics import (
    _adaptive_k, _adaptive_cv, _adaptive_pca_components,
    _adaptive_mlp_hidden, _adaptive_n_permutations,
    _knn_accuracy, _knn_pca_accuracy, _mlp_accuracy,
)


def _compute_null_distribution(
    X: np.ndarray,
    y: np.ndarray,
    metric_fn,
    n_permutations: int,
    random_state: int = 42,
) -> np.ndarray:
    """Compute null distribution by permuting labels."""
    rng = np.random.RandomState(random_state)
    null_scores = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y)
        null_scores.append(metric_fn(X, y_perm))
    return np.array(null_scores)


def compute_signal_vs_null(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    metric_keys: List[str],
    n_permutations: int = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute signal metrics relative to permutation null distribution.

    For each metric, returns:
    - real_score: the actual metric value
    - null_mean: mean of null distribution
    - null_std: std of null distribution
    - z_score: (real - null_mean) / null_std
    - p_value: fraction of null >= real
    - is_significant: p_value < 0.05
    """
    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations

    X = np.vstack([pos_np, neg_np])
    y = np.array([1] * len(pos_np) + [0] * len(neg_np))

    n_samples, n_features = X.shape
    k = _adaptive_k(n_samples)
    cv = _adaptive_cv(n_samples)
    pca_components = _adaptive_pca_components(n_samples, n_features)
    mlp_hidden = _adaptive_mlp_hidden(n_features)

    if n_permutations is None:
        n_permutations = _adaptive_n_permutations(n_samples)

    results = {}

    metric_fns = {}
    if "knn_accuracy" in metric_keys:
        metric_fns["knn_accuracy"] = lambda X, y: _knn_accuracy(X, y, k=k, cv=cv)
    if "knn_pca_accuracy" in metric_keys:
        metric_fns["knn_pca_accuracy"] = lambda X, y: _knn_pca_accuracy(X, y, n_components=pca_components, k=k, cv=cv)
    if "mlp_probe_accuracy" in metric_keys:
        metric_fns["mlp_probe_accuracy"] = lambda X, y: _mlp_accuracy(X, y, hidden=mlp_hidden, cv=cv)

    for metric_name, metric_fn in metric_fns.items():
        real_score = metric_fn(X, y)
        null_dist = _compute_null_distribution(X, y, metric_fn, n_permutations)
        null_mean = float(null_dist.mean())
        null_std = float(null_dist.std())

        if null_std > 1e-10:
            z_score = (real_score - null_mean) / null_std
        else:
            z_score = 0.0 if abs(real_score - null_mean) < 1e-10 else float('inf')

        p_value = float((null_dist >= real_score).mean())

        results[metric_name] = {
            "real_score": real_score,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z_score,
            "p_value": p_value,
            "is_significant": p_value < 0.05,
        }

    return results


def compute_signal_vs_nonsense(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    model,
    tokenizer,
    metric_keys: List[str],
    layer: int = None,
    device: str = "cuda",
    n_nonsense_pairs: int = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare signal metrics against random token (nonsense) baseline.

    This tests: "Is this signal different from random noise?"
    Generates activations from random tokens and compares metrics.
    """
    from ...data.nonsense.nonsense_baseline import generate_nonsense_activations

    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations

    n_samples = len(pos_np)
    if n_nonsense_pairs is None:
        n_nonsense_pairs = max(30, min(n_samples, 100))

    nonsense_pos, nonsense_neg = generate_nonsense_activations(
        model, tokenizer, n_pairs=n_nonsense_pairs, layer=layer, device=device
    )
    nonsense_pos_np = nonsense_pos.cpu().numpy()
    nonsense_neg_np = nonsense_neg.cpu().numpy()

    X_real = np.vstack([pos_np, neg_np])
    y_real = np.array([1] * len(pos_np) + [0] * len(neg_np))

    X_nonsense = np.vstack([nonsense_pos_np, nonsense_neg_np])
    y_nonsense = np.array([1] * len(nonsense_pos_np) + [0] * len(nonsense_neg_np))

    n_features = X_real.shape[1]
    k = _adaptive_k(n_samples)
    cv = _adaptive_cv(n_samples)
    pca_components = _adaptive_pca_components(n_samples, n_features)
    mlp_hidden = _adaptive_mlp_hidden(n_features)

    k_nonsense = _adaptive_k(len(nonsense_pos_np) * 2)
    cv_nonsense = _adaptive_cv(len(nonsense_pos_np) * 2)

    results = {}

    if "knn_accuracy" in metric_keys:
        real_score = _knn_accuracy(X_real, y_real, k=k, cv=cv)
        nonsense_score = _knn_accuracy(X_nonsense, y_nonsense, k=k_nonsense, cv=cv_nonsense)
        z_score = (real_score - nonsense_score) / 0.1
        results["knn_accuracy"] = {
            "real_score": real_score, "nonsense_score": nonsense_score,
            "z_score": z_score, "is_significant": z_score > 2.0,
        }

    if "knn_pca_accuracy" in metric_keys:
        pca_nonsense = _adaptive_pca_components(len(nonsense_pos_np) * 2, n_features)
        real_score = _knn_pca_accuracy(X_real, y_real, n_components=pca_components, k=k, cv=cv)
        nonsense_score = _knn_pca_accuracy(X_nonsense, y_nonsense, n_components=pca_nonsense, k=k_nonsense, cv=cv_nonsense)
        z_score = (real_score - nonsense_score) / 0.1
        results["knn_pca_accuracy"] = {
            "real_score": real_score, "nonsense_score": nonsense_score,
            "z_score": z_score, "is_significant": z_score > 2.0,
        }

    if "mlp_probe_accuracy" in metric_keys:
        real_score = _mlp_accuracy(X_real, y_real, hidden=mlp_hidden, cv=cv)
        nonsense_score = _mlp_accuracy(X_nonsense, y_nonsense, hidden=mlp_hidden, cv=cv_nonsense)
        z_score = (real_score - nonsense_score) / 0.1
        results["mlp_probe_accuracy"] = {
            "real_score": real_score, "nonsense_score": nonsense_score,
            "z_score": z_score, "is_significant": z_score > 2.0,
        }

    return results


def compute_aggregate_signal(
    signal_results: Dict[str, Dict[str, float]],
    correction: str = "bonferroni",
) -> Tuple[float, float, bool]:
    """
    Aggregate signal across multiple metrics with multiple testing correction.

    Args:
        signal_results: Dict of metric results
        correction: "none", "bonferroni", or "max_null"

    Returns: (max_z_score, corrected_min_p_value, any_significant_after_correction)
    """
    if not signal_results:
        return 0.0, 1.0, False

    n_tests = len(signal_results)
    z_scores = [m["z_score"] for m in signal_results.values()]
    p_values = [m.get("p_value", 0.0) for m in signal_results.values()]

    max_z = max(z_scores)
    min_p = min(p_values) if p_values and all(p > 0 for p in p_values) else 0.0

    if correction == "bonferroni":
        corrected_p = min(1.0, min_p * n_tests)
        threshold = 0.05 / n_tests
    elif correction == "max_null":
        corrected_p = _max_null_correction(min_p, n_tests)
        threshold = 0.05
    else:
        corrected_p = min_p
        threshold = 0.05

    any_sig = corrected_p < threshold

    return max_z, corrected_p, any_sig


def _max_null_correction(min_p: float, n_tests: int) -> float:
    """
    Max-null correction for multiple testing.

    Under the null, the minimum p-value from n tests follows Beta(1, n).
    Returns the corrected p-value accounting for multiple testing.
    """
    if min_p <= 0:
        return 0.0
    corrected = 1.0 - (1.0 - min_p) ** n_tests
    return min(1.0, corrected)


def compute_signal_with_bounds(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    metric_keys: List[str],
    n_permutations: int = None,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Compute signal metrics with estimation error bounds.

    Instead of claiming "no guarantees", we bound estimation error:
    - Accuracy estimate ± margin of error
    - Effect size (Cohen's d) with confidence interval
    - Comparison vs null is bounded, not "feasible/infeasible"
    """
    results = compute_signal_vs_null(pos_activations, neg_activations, metric_keys, n_permutations)

    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations
    n_samples = len(pos_np) + len(neg_np)

    z_crit = 1.96 if confidence_level == 0.95 else 2.576

    for metric_name, metric_result in results.items():
        acc = metric_result["real_score"]
        se = np.sqrt(acc * (1 - acc) / n_samples)
        margin = z_crit * se

        null_mean = metric_result["null_mean"]
        null_std = metric_result["null_std"]
        effect_size = (acc - null_mean) / (null_std + 1e-10)

        metric_result.update({
            "accuracy_lower": max(0.0, acc - margin),
            "accuracy_upper": min(1.0, acc + margin),
            "margin_of_error": margin,
            "effect_size_cohens_d": effect_size,
            "estimation_reliable": margin < 0.1,
        })

    return results
