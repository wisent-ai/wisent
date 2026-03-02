"""
Signal analysis metrics for activation representations.

Includes signal-to-noise ratio, null distribution comparison,
bootstrap estimates, and saturation analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from wisent.core.utils.config_tools.constants import (
    NORM_EPS, DEFAULT_SCORE, BLEND_DEFAULT, DEFAULT_SCALE,
    SIGNAL_NULL_MEAN, SIGNAL_NULL_STD, SIGNAL_SATURATION_DELTA,
    SIGNAL_POWER_SAMPLE_SIZES, SIGNAL_ICD_THRESHOLD, AUTO_MIN_PAIRS,
    AUTO_N_FOLDS, DEFAULT_RANDOM_SEED, LINEARITY_N_BOOTSTRAP,
    STABILITY_N_BOOTSTRAP, STABILITY_SUBSAMPLE_RATIO, STAT_ALPHA,
    DETECTION_THRESHOLD, LINEARITY_Z_SCORE_THRESHOLD, SIMILARITY_THRESHOLD,
    BOOTSTRAP_CI_LOW_DEFAULT, BOOTSTRAP_CI_HIGH_DEFAULT, CV_FOLDS,
    VQ_MIN_PAIRS, PERCENTILE_HIGH, CI_PERCENTILE_LOW, CI_PERCENTILE_HIGH,
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
            return DEFAULT_SCORE
        return float(signal / noise)
    except Exception:
        return DEFAULT_SCORE


def compute_null_distribution(
    activations: torch.Tensor,
    n_permutations: int = LINEARITY_N_BOOTSTRAP,
) -> Dict[str, float]:
    """
    Compute null distribution by random permutation.
    Randomly split activations into two groups and compute
    metrics, to establish baseline for random split.
    """
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        n = len(activations)
        if n < AUTO_MIN_PAIRS:
            return {"null_mean": SIGNAL_NULL_MEAN, "null_std": SIGNAL_NULL_STD}
        null_accuracies = []
        rng = np.random.RandomState(DEFAULT_RANDOM_SEED)
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            half = n // 2
            pos = activations[perm[:half]]
            neg = activations[perm[half:2*half]]
            acc = compute_linear_probe_accuracy(pos, neg, n_folds=AUTO_N_FOLDS)
            null_accuracies.append(acc)
        return {
            "null_mean": float(np.mean(null_accuracies)),
            "null_std": float(np.std(null_accuracies)),
            "null_95th": float(np.percentile(null_accuracies, PERCENTILE_HIGH)),
        }
    except Exception:
        return {"null_mean": SIGNAL_NULL_MEAN, "null_std": SIGNAL_NULL_STD}


def compare_to_null(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    null_distribution: Dict[str, float],
) -> Dict[str, Any]:
    """Compare observed metrics to null distribution."""
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        observed = compute_linear_probe_accuracy(pos_activations, neg_activations)
        null_mean = null_distribution.get("null_mean", SIGNAL_NULL_MEAN)
        null_std = null_distribution.get("null_std", SIGNAL_NULL_STD)
        if null_std < NORM_EPS:
            null_std = SIGNAL_NULL_STD
        z_score = (observed - null_mean) / null_std
        p_value = 1 - BLEND_DEFAULT * (1 + np.erf(z_score / np.sqrt(2)))
        return {
            "observed": observed,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "is_significant": p_value < STAT_ALPHA,
        }
    except Exception:
        return {
            "observed": BLEND_DEFAULT,
            "z_score": DEFAULT_SCORE,
            "p_value": DEFAULT_SCALE,
            "is_significant": False,
        }


def validate_concept(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    min_accuracy: float = DETECTION_THRESHOLD,
    min_z_score: float = LINEARITY_Z_SCORE_THRESHOLD,
) -> Dict[str, Any]:
    """Validate that activations contain a real, extractable concept."""
    from .probe_metrics import compute_linear_probe_accuracy, compute_signal_strength
    from .icd import compute_icd
    try:
        linear_acc = compute_linear_probe_accuracy(pos_activations, neg_activations)
        signal = compute_signal_strength(pos_activations, neg_activations)
        icd_result = compute_icd(pos_activations, neg_activations)
        is_valid = (
            linear_acc >= min_accuracy and
            signal >= min_accuracy and
            icd_result["icd"] < SIGNAL_ICD_THRESHOLD
        )
        return {
            "is_valid": is_valid,
            "linear_accuracy": linear_acc,
            "signal_strength": signal,
            "icd": icd_result["icd"],
            "reasons": [] if is_valid else [
                f"linear_acc={linear_acc:.2f} < {min_accuracy}" if linear_acc < min_accuracy else None,
                f"signal={signal:.2f} < {min_accuracy}" if signal < min_accuracy else None,
                f"icd={icd_result['icd']:.1f} >= {SIGNAL_ICD_THRESHOLD}" if icd_result["icd"] >= SIGNAL_ICD_THRESHOLD else None,
            ],
        }
    except Exception as e:
        return {"is_valid": False, "error": str(e)}


def compute_bootstrap_signal_estimate(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = STABILITY_N_BOOTSTRAP,
    subset_fraction: float = STABILITY_SUBSAMPLE_RATIO,
) -> Dict[str, float]:
    """Estimate signal metrics with bootstrap confidence intervals."""
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        subset_size = max(int(n_pairs * subset_fraction), VQ_MIN_PAIRS)
        if n_pairs < AUTO_MIN_PAIRS:
            return {
                "mean": BLEND_DEFAULT, "std": SIGNAL_NULL_STD,
                "ci_low": BOOTSTRAP_CI_LOW_DEFAULT, "ci_high": BOOTSTRAP_CI_HIGH_DEFAULT,
            }
        rng = np.random.RandomState(DEFAULT_RANDOM_SEED)
        accuracies = []
        for _ in range(n_bootstrap):
            indices = rng.choice(n_pairs, size=subset_size, replace=True)
            pos_sub = pos_activations[indices]
            neg_sub = neg_activations[indices]
            acc = compute_linear_probe_accuracy(pos_sub, neg_sub, n_folds=AUTO_N_FOLDS)
            accuracies.append(acc)
        return {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "ci_low": float(np.percentile(accuracies, CI_PERCENTILE_LOW)),
            "ci_high": float(np.percentile(accuracies, CI_PERCENTILE_HIGH)),
        }
    except Exception:
        return {
            "mean": BLEND_DEFAULT, "std": SIGNAL_NULL_STD,
            "ci_low": BOOTSTRAP_CI_LOW_DEFAULT, "ci_high": BOOTSTRAP_CI_HIGH_DEFAULT,
        }


def compute_saturation_check(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    sample_sizes: List[int] = None,
) -> Dict[str, Any]:
    """
    Check if metrics have saturated (more data won't help).
    Train on increasing amounts of data to see if accuracy plateaus.
    """
    from .probe_metrics import compute_linear_probe_accuracy
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        if sample_sizes is None:
            sample_sizes = list(SIGNAL_POWER_SAMPLE_SIZES)
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
                n_folds=min(CV_FOLDS, size // 2),
            )
            accuracies[size] = acc
        # Check if last two are similar (saturated)
        sorted_sizes = sorted(accuracies.keys())
        if len(sorted_sizes) >= 2:
            last = accuracies[sorted_sizes[-1]]
            second_last = accuracies[sorted_sizes[-2]]
            is_saturated = abs(last - second_last) < SIGNAL_SATURATION_DELTA
        else:
            is_saturated = False
        return {
            "is_saturated": is_saturated,
            "accuracies": accuracies,
            "final_accuracy": accuracies[sorted_sizes[-1]] if sorted_sizes else BLEND_DEFAULT,
        }
    except Exception:
        return {"is_saturated": False, "accuracies": {}, "error": "Failed to compute"}


def find_optimal_pair_count(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    target_accuracy: float = SIMILARITY_THRESHOLD,
    max_pairs: int = None,
) -> Dict[str, Any]:
    """Find minimum number of pairs needed to reach target accuracy."""
    saturation = compute_saturation_check(pos_activations, neg_activations)
    accuracies = saturation.get("accuracies", {})
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
