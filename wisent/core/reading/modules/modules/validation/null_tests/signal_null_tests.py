"""Null distribution tests for Zwiad signal validation."""

from typing import Dict, List, Tuple
import torch
import numpy as np

from wisent.core.utils.config_tools.constants import (
    ZERO_THRESHOLD, DEFAULT_RANDOM_SEED, CONFIDENCE_LEVEL, SIGNIFICANCE_ALPHA,
    NULL_TEST_Z_SCORE_SIGNIFICANT, Z_CRITICAL_95, Z_CRITICAL_99, COMBO_OFFSET,
    N_COMPONENTS_2D,
)

from wisent.core.reading.modules.utilities.metrics.probe.signal_metrics import (
    _adaptive_k, _adaptive_cv, _adaptive_pca_components,
    _adaptive_mlp_hidden, _adaptive_n_permutations,
    _knn_accuracy, _knn_pca_accuracy, _mlp_accuracy,
)


def _compute_null_distribution(
    X: np.ndarray,
    y: np.ndarray,
    metric_fn,
    n_permutations: int,
    random_state: int = DEFAULT_RANDOM_SEED,
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
    mlp_early_stopping_min_samples: int = None,
    mlp_probe_max_iter: int = None,
    *,
    pca_max_components_null: int,
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

    # Subsample pairs for permutation testing when dataset is large
    n_pairs = len(pos_np)
    if n_pairs > 1000:
        idx = np.random.RandomState(DEFAULT_RANDOM_SEED).choice(n_pairs, 1000, replace=False)
        idx.sort()
        pos_np = pos_np[idx]
        neg_np = neg_np[idx]

    X = np.vstack([pos_np, neg_np])
    y = np.array([1] * len(pos_np) + [0] * len(neg_np))

    n_samples, n_features = X.shape
    # PCA reduce for high-dimensional concatenated data
    pca_max = min(n_samples - COMBO_OFFSET, n_features, pca_max_components_null)
    if pca_max < n_features and pca_max >= N_COMPONENTS_2D:
        from sklearn.decomposition import PCA
        X = PCA(n_components=pca_max, random_state=DEFAULT_RANDOM_SEED).fit_transform(X)
        n_features = pca_max
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
        metric_fns["mlp_probe_accuracy"] = lambda X, y: _mlp_accuracy(
            X, y, hidden=mlp_hidden, cv=cv,
            mlp_early_stopping_min_samples=mlp_early_stopping_min_samples,
            mlp_probe_max_iter=mlp_probe_max_iter)

    for metric_name, metric_fn in metric_fns.items():
        real_score = metric_fn(X, y)
        null_dist = _compute_null_distribution(X, y, metric_fn, n_permutations)
        null_mean = float(null_dist.mean())
        null_std = float(null_dist.std())

        if null_std > ZERO_THRESHOLD:
            z_score = (real_score - null_mean) / null_std
        else:
            z_score = 0.0 if abs(real_score - null_mean) < ZERO_THRESHOLD else float('inf')

        p_value = float((null_dist >= real_score).mean())

        results[metric_name] = {
            "real_score": real_score,
            "null_mean": null_mean,
            "null_std": null_std,
            "z_score": z_score,
            "p_value": p_value,
            "is_significant": p_value < SIGNIFICANCE_ALPHA,
        }

    return results


def compute_signal_vs_nonsense(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    model, tokenizer,
    metric_keys: List[str],
    device: str,
    layer: int = None,
    mlp_early_stopping_min_samples: int = None,
    nonsense_pairs_min: int = None, nonsense_pairs_max: int = None,
    mlp_probe_max_iter: int = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare signal metrics against random token (nonsense) baseline.

    This tests: "Is this signal different from random noise?"
    Generates activations from random tokens and compares metrics.
    """
    from wisent.core.reading.modules.utilities.data.sources.nonsense.nonsense_baseline import generate_nonsense_activations

    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations

    n_samples = len(pos_np)
    n_nonsense_pairs = max(nonsense_pairs_min, min(n_samples, nonsense_pairs_max))


    nonsense_pos, nonsense_neg = generate_nonsense_activations(
        model, tokenizer, device=device, n_pairs=n_nonsense_pairs, layer=layer,
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
            "z_score": z_score, "is_significant": z_score > NULL_TEST_Z_SCORE_SIGNIFICANT,
        }

    if "knn_pca_accuracy" in metric_keys:
        pca_nonsense = _adaptive_pca_components(len(nonsense_pos_np) * 2, n_features)
        real_score = _knn_pca_accuracy(X_real, y_real, n_components=pca_components, k=k, cv=cv)
        nonsense_score = _knn_pca_accuracy(X_nonsense, y_nonsense, n_components=pca_nonsense, k=k_nonsense, cv=cv_nonsense)
        z_score = (real_score - nonsense_score) / 0.1
        results["knn_pca_accuracy"] = {
            "real_score": real_score, "nonsense_score": nonsense_score,
            "z_score": z_score, "is_significant": z_score > NULL_TEST_Z_SCORE_SIGNIFICANT,
        }

    if "mlp_probe_accuracy" in metric_keys:
        real_score = _mlp_accuracy(X_real, y_real, hidden=mlp_hidden, cv=cv,
            mlp_early_stopping_min_samples=mlp_early_stopping_min_samples, mlp_probe_max_iter=mlp_probe_max_iter)
        nonsense_score = _mlp_accuracy(X_nonsense, y_nonsense, hidden=mlp_hidden, cv=cv_nonsense,
            mlp_early_stopping_min_samples=mlp_early_stopping_min_samples, mlp_probe_max_iter=mlp_probe_max_iter)
        z_score = (real_score - nonsense_score) / 0.1
        results["mlp_probe_accuracy"] = {
            "real_score": real_score, "nonsense_score": nonsense_score,
            "z_score": z_score, "is_significant": z_score > NULL_TEST_Z_SCORE_SIGNIFICANT,
        }

    return results


def compute_aggregate_signal(
    signal_results: Dict[str, Dict[str, float]],
    correction: str,
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
        threshold = SIGNIFICANCE_ALPHA / n_tests
    elif correction == "max_null":
        corrected_p = _max_null_correction(min_p, n_tests)
        threshold = SIGNIFICANCE_ALPHA
    else:
        corrected_p = min_p
        threshold = SIGNIFICANCE_ALPHA

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
    confidence_level: float = CONFIDENCE_LEVEL,
    mlp_early_stopping_min_samples: int = None,
    mlp_probe_max_iter: int = None,
    *,
    pca_max_components_null: int,
    significance_margin: float,
) -> Dict[str, Dict[str, float]]:
    """Compute signal metrics with estimation error bounds."""
    results = compute_signal_vs_null(
        pos_activations, neg_activations, metric_keys, n_permutations,
        mlp_early_stopping_min_samples=mlp_early_stopping_min_samples,
        mlp_probe_max_iter=mlp_probe_max_iter,
        pca_max_components_null=pca_max_components_null)

    pos_np = pos_activations.cpu().numpy() if isinstance(pos_activations, torch.Tensor) else pos_activations
    neg_np = neg_activations.cpu().numpy() if isinstance(neg_activations, torch.Tensor) else neg_activations
    n_samples = len(pos_np) + len(neg_np)

    z_crit = Z_CRITICAL_95 if confidence_level == 0.95 else Z_CRITICAL_99

    for metric_name, metric_result in results.items():
        acc = metric_result["real_score"]
        se = np.sqrt(acc * (1 - acc) / n_samples)
        margin = z_crit * se

        null_mean = metric_result["null_mean"]
        null_std = metric_result["null_std"]
        effect_size = (acc - null_mean) / (null_std + ZERO_THRESHOLD)

        metric_result.update({
            "accuracy_lower": max(0.0, acc - margin),
            "accuracy_upper": min(1.0, acc + margin),
            "margin_of_error": margin,
            "effect_size_cohens_d": effect_size,
            "estimation_reliable": margin < significance_margin,
        })

    return results
