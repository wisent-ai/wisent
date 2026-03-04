"""
Signal analysis metrics for activation representations.

Includes signal-to-noise ratio, null distribution comparison,
bootstrap estimates, and saturation analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from wisent.core.utils.config_tools.constants import (
    NORM_EPS,
    DEFAULT_RANDOM_SEED, STAT_ALPHA,
    CI_PERCENTILE_LOW, CI_PERCENTILE_HIGH, N_COMPONENTS_2D,
)


def compute_signal_to_noise(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute signal-to-noise ratio of the separation.
    Signal = distance between class means, Noise = average within-class std.
    """
    try:
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        pos_mean = pos.mean(axis=0)
        neg_mean = neg.mean(axis=0)
        signal = np.linalg.norm(pos_mean - neg_mean)
        pos_std = np.std(pos, axis=0).mean()
        neg_std = np.std(neg, axis=0).mean()
        noise = (pos_std + neg_std) / 2
        if noise < NORM_EPS:
            raise ValueError("signal_to_noise failed: noise below NORM_EPS threshold")
        return float(signal / noise)
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"signal_to_noise computation failed: {e}") from e


def compute_null_distribution(
    activations: torch.Tensor,
    n_permutations: int = None,
    auto_min_pairs: int = None,
    auto_n_folds: int = None,
    *,
    signal_null_mean: float,
    signal_null_std: float,
    null_percentile_high: float,
) -> Dict[str, float]:
    """
    Compute null distribution by random permutation.
    Randomly split activations into two groups and compute
    metrics, to establish baseline for random split.
    """
    if n_permutations is None:
        raise ValueError("n_permutations is required")
    if auto_min_pairs is None:
        raise ValueError("auto_min_pairs is required")
    if auto_n_folds is None:
        raise ValueError("auto_n_folds is required")
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        n = len(activations)
        if n < auto_min_pairs:
            return {"null_mean": signal_null_mean, "null_std": signal_null_std}
        null_accuracies = []
        rng = np.random.RandomState(DEFAULT_RANDOM_SEED)
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            half = n // N_COMPONENTS_2D
            pos = activations[perm[:half]]
            neg = activations[perm[half:N_COMPONENTS_2D*half]]
            acc = compute_linear_probe_accuracy(pos, neg, n_folds=auto_n_folds)
            null_accuracies.append(acc)
        return {
            "null_mean": float(np.mean(null_accuracies)),
            "null_std": float(np.std(null_accuracies)),
            "null_high": float(np.percentile(null_accuracies, null_percentile_high)),
        }
    except Exception:
        return {"null_mean": signal_null_mean, "null_std": signal_null_std}


def compare_to_null(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    null_distribution: Dict[str, float],
    auto_n_folds: int = None,
    *,
    signal_null_std: float,
    blend_default: float,
) -> Dict[str, Any]:
    """Compare observed metrics to null distribution."""
    if auto_n_folds is None:
        raise ValueError("auto_n_folds is required")
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        observed = compute_linear_probe_accuracy(pos_activations, neg_activations, auto_n_folds)
        null_mean = null_distribution["null_mean"]
        null_std = null_distribution["null_std"]
        if null_std < NORM_EPS:
            null_std = signal_null_std
        z_score = (observed - null_mean) / null_std
        erf_val = np.erf(z_score / np.sqrt(N_COMPONENTS_2D))
        p_value = blend_default - blend_default * erf_val
        return {
            "observed": observed,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "is_significant": p_value < STAT_ALPHA,
        }
    except Exception as e:
        raise ValueError(f"significance_of_separation failed: {e}") from e


def validate_concept(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    *,
    min_accuracy: float,
    min_z_score: float = None,
    auto_n_folds: int = None,
    signal_icd_threshold: int,
) -> Dict[str, Any]:
    """Validate that activations contain a real, extractable concept."""
    if min_z_score is None:
        raise ValueError("min_z_score is required")
    if auto_n_folds is None:
        raise ValueError("auto_n_folds is required")
    from .probe_metrics import compute_linear_probe_accuracy, compute_signal_strength
    from .icd import compute_icd
    try:
        linear_acc = compute_linear_probe_accuracy(pos_activations, neg_activations, auto_n_folds)
        signal = compute_signal_strength(pos_activations, neg_activations, auto_n_folds)
        icd_result = compute_icd(pos_activations, neg_activations)
        is_valid = (
            linear_acc >= min_accuracy and
            signal >= min_accuracy and
            icd_result["icd"] < signal_icd_threshold
        )
        return {
            "is_valid": is_valid,
            "linear_accuracy": linear_acc,
            "signal_strength": signal,
            "icd": icd_result["icd"],
            "reasons": [] if is_valid else [
                f"linear_acc={linear_acc:.2f} < {min_accuracy}" if linear_acc < min_accuracy else None,
                f"signal={signal:.2f} < {min_accuracy}" if signal < min_accuracy else None,
                f"icd={icd_result['icd']:.1f} >= {signal_icd_threshold}" if icd_result["icd"] >= signal_icd_threshold else None,
            ],
        }
    except Exception as e:
        return {"is_valid": False, "error": str(e)}


def compute_bootstrap_signal_estimate(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = None,
    subset_fraction: float = None,
    auto_min_pairs: int = None,
    auto_n_folds: int = None,
    *,
    vq_min_pairs: int,
    bootstrap_ci_low_default: float,
    bootstrap_ci_high_default: float,
    signal_null_std: float,
    blend_default: float,
) -> Dict[str, float]:
    """Estimate signal metrics with bootstrap confidence intervals."""
    if auto_min_pairs is None:
        raise ValueError("auto_min_pairs is required")
    if auto_n_folds is None:
        raise ValueError("auto_n_folds is required")
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        subset_size = max(int(n_pairs * subset_fraction), vq_min_pairs)
        if n_pairs < auto_min_pairs:
            return {
                "mean": blend_default, "std": signal_null_std,
                "ci_low": bootstrap_ci_low_default, "ci_high": bootstrap_ci_high_default,
            }
        rng = np.random.RandomState(DEFAULT_RANDOM_SEED)
        accuracies = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_pairs, size=subset_size, replace=True)
            pos_sub = pos_activations[indices]
            neg_sub = neg_activations[indices]
            acc = compute_linear_probe_accuracy(pos_sub, neg_sub, n_folds=auto_n_folds)
            accuracies.append(acc)
        return {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "ci_low": float(np.percentile(accuracies, CI_PERCENTILE_LOW)),
            "ci_high": float(np.percentile(accuracies, CI_PERCENTILE_HIGH)),
        }
    except Exception:
        return {
            "mean": blend_default, "std": signal_null_std,
            "ci_low": bootstrap_ci_low_default, "ci_high": bootstrap_ci_high_default,
        }


def compute_saturation_check(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    sample_sizes: tuple = None,
    *,
    cv_folds: int,
    signal_saturation_delta: float,
) -> Dict[str, Any]:
    """
    Check if metrics have saturated (more data won't help).
    Train on increasing amounts of data to see if accuracy plateaus.
    """
    if sample_sizes is None:
        raise ValueError("sample_sizes is required")
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        sample_sizes = [s for s in sample_sizes if s <= n_pairs]
        if len(sample_sizes) < 2:
            return {
                "is_saturated": False,
                "accuracies": {},
                "recommendation": "Need more data",
            }
        accuracies = {}
        for size in sample_sizes:
            acc = compute_linear_probe_accuracy(
                pos_activations[:size], neg_activations[:size],
                n_folds=min(cv_folds, size // N_COMPONENTS_2D),
            )
            accuracies[size] = acc
        # Check if last two are similar (saturated)
        sorted_sizes = sorted(accuracies.keys())
        if len(sorted_sizes) >= 2:
            last = accuracies[sorted_sizes[-1]]
            second_last = accuracies[sorted_sizes[-2]]
            is_saturated = abs(last - second_last) < signal_saturation_delta
        else:
            is_saturated = False
        return {
            "is_saturated": is_saturated,
            "accuracies": accuracies,
            "final_accuracy": accuracies[max(sorted_sizes)] if sorted_sizes else None,
        }
    except Exception:
        return {"is_saturated": False, "accuracies": {}, "error": "Failed to compute"}


def find_optimal_pair_count(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    *, target_accuracy: float,
    max_pairs: int = None,
    cv_folds: int,
    signal_saturation_delta: float,
    sample_sizes: tuple,
) -> Dict[str, Any]:
    """Find minimum number of pairs needed to reach target accuracy."""
    saturation = compute_saturation_check(
        pos_activations, neg_activations, sample_sizes=sample_sizes,
        cv_folds=cv_folds, signal_saturation_delta=signal_saturation_delta,
    )
    accuracies = saturation["accuracies"]
    if not accuracies:
        return {"optimal_pairs": -1, "reason": "No data"}
    for size in sorted(accuracies.keys()):
        if accuracies[size] >= target_accuracy:
            return {
                "optimal_pairs": size,
                "achieved_accuracy": accuracies[size],
                "is_achievable": True,
            }
    max_acc = max(accuracies.values())
    return {
        "optimal_pairs": -1,
        "max_accuracy": max_acc,
        "is_achievable": False,
        "reason": f"Max accuracy {max_acc:.2f} < target {target_accuracy:.2f}",
    }
