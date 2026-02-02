"""Sample-to-dimension ratio checks and effective sample size estimation."""

import numpy as np
from typing import Dict, Any


def compute_sample_dimension_ratio(
    n_samples: int,
    ambient_dim: int,
    effective_dim: float,
) -> Dict[str, Any]:
    """
    Check if sample size is adequate for dimensionality.

    Rules of thumb:
    - n/d > 10: Safe for most methods
    - n/d > 5: Acceptable with regularization
    - n/d > 2: Marginal, high variance expected
    - n/d < 2: Dangerous, results unreliable
    """
    ratio = n_samples / ambient_dim if ambient_dim > 0 else float('inf')
    eff_ratio = n_samples / effective_dim if effective_dim > 0 else float('inf')

    if eff_ratio >= 20:
        adequate, warning, severity = True, None, "none"
    elif eff_ratio >= 10:
        adequate, warning, severity = True, None, "none"
    elif eff_ratio >= 5:
        adequate = True
        warning = f"Marginal sample size: n/d_eff={eff_ratio:.1f}. Consider regularization."
        severity = "mild"
    elif eff_ratio >= 2:
        adequate = False
        warning = f"Low sample size: n/d_eff={eff_ratio:.1f}. High variance expected."
        severity = "moderate"
    else:
        adequate = False
        warning = f"Severely underpowered: n/d_eff={eff_ratio:.1f}. Results unreliable."
        severity = "severe"

    return {
        "n_samples": n_samples,
        "ambient_dim": ambient_dim,
        "effective_dim": effective_dim,
        "sample_dim_ratio": ratio,
        "effective_sample_dim_ratio": eff_ratio,
        "ratio_adequate": adequate,
        "ratio_warning": warning,
        "severity": severity,
    }


def compute_effective_sample_size(X: np.ndarray, method: str = "eigenvalue") -> Dict[str, Any]:
    """Compute effective sample size accounting for correlation."""
    n, d = X.shape
    X_centered = X - X.mean(axis=0)
    stds = X_centered.std(axis=0)
    stds[stds < 1e-10] = 1.0
    X_standardized = X_centered / stds

    corr_matrix = np.corrcoef(X_standardized.T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    mask = ~np.eye(d, dtype=bool)
    avg_abs_corr = np.abs(corr_matrix[mask]).mean() if d > 1 else 0.0

    try:
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        eigenvalues = np.sort(eigenvalues)[::-1]

        eff_dim = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum() if eigenvalues.sum() > 1e-10 else 1.0

        if len(eigenvalues) > 1 and eigenvalues[0] > 1e-10:
            log_eigs = np.log(eigenvalues[eigenvalues > 1e-10] + 1e-10)
            decay_rate = -np.polyfit(np.arange(len(log_eigs)), log_eigs, 1)[0] if len(log_eigs) > 1 else 0.0
        else:
            decay_rate = 0.0

        cond = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 1e-10 else float('inf')
    except Exception:
        eff_dim, decay_rate, cond, eigenvalues = d, 0.0, float('inf'), np.ones(d)

    n_eff = n * min(d, eff_dim) / d if method == "eigenvalue" and d > 0 else n * (1 - avg_abs_corr ** 2)
    n_eff = max(1.0, min(n_eff, n))

    return {
        "effective_n": float(n_eff),
        "effective_dim": float(eff_dim),
        "avg_abs_correlation": float(avg_abs_corr),
        "eigenvalue_decay_rate": float(decay_rate),
        "condition_number": float(min(cond, 1e10)),
        "top_5_eigenvalues": eigenvalues[:5].tolist() if len(eigenvalues) >= 5 else eigenvalues.tolist(),
        "n_eff_ratio": float(n_eff / n) if n > 0 else 1.0,
    }


def recommend_sample_size(
    effective_dim: float,
    target_power: float = 0.80,
    target_effect_size: str = "medium",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Recommend number of contrastive pairs given effective dimensionality.

    Args:
        effective_dim: Effective dimension of the activation space (measure with
            participation_ratio, effective_rank, or compute_effective_dimensions)
        target_power: Desired statistical power (default 0.80)
        target_effect_size: "small" (d=0.2), "medium" (d=0.5), or "large" (d=0.8)
        alpha: Significance level (default 0.05)

    Returns:
        Dict with recommended sample sizes for different effect sizes.

    Example:
        >>> from wisent.core.geometry.analysis.intrinsic_dim import participation_ratio
        >>> eff_dim = participation_ratio(diff_vectors)
        >>> rec = recommend_sample_size(eff_dim)
        >>> print(f"Need {rec['recommended_pairs']} pairs for medium effects")
    """
    from scipy import stats

    effect_sizes = {"small": 0.2, "medium": 0.5, "large": 0.8}
    d = effect_sizes.get(target_effect_size, 0.5)

    def required_n(eff_d: float, effect: float, power: float, a: float) -> int:
        """Compute required n for target power at given effect size."""
        for test_n in range(8, 20000, 2):
            df = max(1, min(test_n - 2, test_n - eff_d - 1))
            t_crit = stats.t.ppf(1 - a / 2, df)
            n_per_group = test_n / 2
            test_power = 1 - stats.t.cdf(t_crit, df, loc=effect * np.sqrt(n_per_group / 2))
            if test_power >= power:
                return test_n
        return 20000

    recommended = required_n(effective_dim, d, target_power, alpha)
    n_for_small = required_n(effective_dim, effect_sizes["small"], target_power, alpha)
    n_for_medium = required_n(effective_dim, effect_sizes["medium"], target_power, alpha)
    n_for_large = required_n(effective_dim, effect_sizes["large"], target_power, alpha)

    n_d_ratio = recommended / effective_dim if effective_dim > 0 else float('inf')

    return {
        "effective_dim": effective_dim,
        "target_power": target_power,
        "target_effect_size": target_effect_size,
        "alpha": alpha,
        "recommended_pairs": recommended,
        "n_for_small_effect": n_for_small,
        "n_for_medium_effect": n_for_medium,
        "n_for_large_effect": n_for_large,
        "n_to_d_eff_ratio": n_d_ratio,
    }


def recommend_sample_size_from_data(
    pos: np.ndarray,
    neg: np.ndarray,
    target_power: float = 0.80,
    target_effect_size: str = "medium",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Recommend sample size using empirically measured effective dimension.

    Uses participation ratio to estimate effective dimensionality from
    a pilot sample of contrastive activations.

    Args:
        pos: Pilot positive activations [n_pilot, hidden_size]
        neg: Pilot negative activations [n_pilot, hidden_size]
        target_power: Desired statistical power
        target_effect_size: "small", "medium", or "large"
        alpha: Significance level

    Returns:
        Dict with recommendations based on measured effective dimension.

    Example:
        >>> # After collecting 50 pilot pairs
        >>> rec = recommend_sample_size_from_data(pos_activations, neg_activations)
        >>> print(f"Need {rec['recommended_pairs']} pairs total")
    """
    diff = pos - neg
    hidden_size = diff.shape[1]
    n_pilot = len(diff)

    # Compute participation ratio as effective dimension estimate
    diff_centered = diff - diff.mean(axis=0)
    cov = np.cov(diff_centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)

    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    measured_effective_dim = (sum_eig ** 2) / sum_eig_sq if sum_eig_sq > 1e-10 else 1.0

    # Get recommendation using measured effective dim
    rec = recommend_sample_size(
        effective_dim=measured_effective_dim,
        target_power=target_power,
        target_effect_size=target_effect_size,
        alpha=alpha,
    )

    rec["ambient_dim"] = hidden_size
    rec["effective_dim_ratio"] = measured_effective_dim / hidden_size
    rec["pilot_samples"] = n_pilot

    return rec


def compare_extraction_strategies(
    activations_by_strategy: Dict[str, tuple],
    target_power: float = 0.80,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare extraction strategies by their effective dimension and sample requirements.

    Lower effective dimension means the signal is more concentrated, requiring
    fewer samples for the same statistical power.

    Args:
        activations_by_strategy: Dict mapping strategy name to (pos, neg) activation tuples
        target_power: Desired statistical power

    Returns:
        Dict mapping strategy name to analysis results, sorted by effective dimension.

    Example:
        >>> strategies = {
        ...     "chat_last": (pos_last, neg_last),
        ...     "chat_mean": (pos_mean, neg_mean),
        ... }
        >>> results = compare_extraction_strategies(strategies)
        >>> for name, r in results.items():
        ...     print(f"{name}: eff_dim={r['effective_dim']:.1f}, need {r['n_for_medium_effect']} pairs")
    """
    results = {}

    for strategy_name, (pos, neg) in activations_by_strategy.items():
        rec = recommend_sample_size_from_data(pos, neg, target_power=target_power)
        results[strategy_name] = rec

    # Sort by effective dimension (lower is better)
    results = dict(sorted(results.items(), key=lambda x: x[1]["effective_dim"]))

    return results
