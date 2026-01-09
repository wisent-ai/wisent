"""
Signal analysis metrics for activation representations.

Includes signal-to-noise ratio, null distribution comparison,
bootstrap estimates, and saturation analysis.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional


def compute_signal_to_noise(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute signal-to-noise ratio of the separation.
    
    Signal = distance between class means
    Noise = average within-class std
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
        
        if noise < 1e-8:
            return 0.0
        
        return float(signal / noise)
    except Exception:
        return 0.0


def compute_null_distribution(
    activations: torch.Tensor,
    n_permutations: int = 100,
) -> Dict[str, float]:
    """
    Compute null distribution by random permutation.
    
    Randomly split activations into two groups and compute
    metrics, to establish baseline for random split.
    """
    from .probe_metrics import compute_linear_probe_accuracy
    
    try:
        n = len(activations)
        if n < 10:
            return {"null_mean": 0.5, "null_std": 0.1}
        
        null_accuracies = []
        rng = np.random.RandomState(42)
        
        for _ in range(n_permutations):
            perm = rng.permutation(n)
            half = n // 2
            pos = activations[perm[:half]]
            neg = activations[perm[half:2*half]]
            
            acc = compute_linear_probe_accuracy(pos, neg, n_folds=3)
            null_accuracies.append(acc)
        
        return {
            "null_mean": float(np.mean(null_accuracies)),
            "null_std": float(np.std(null_accuracies)),
            "null_95th": float(np.percentile(null_accuracies, 95)),
        }
    except Exception:
        return {"null_mean": 0.5, "null_std": 0.1}


def compare_to_null(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    null_distribution: Dict[str, float],
) -> Dict[str, Any]:
    """
    Compare observed metrics to null distribution.
    """
    from .probe_metrics import compute_linear_probe_accuracy
    
    try:
        observed = compute_linear_probe_accuracy(pos_activations, neg_activations)
        
        null_mean = null_distribution.get("null_mean", 0.5)
        null_std = null_distribution.get("null_std", 0.1)
        
        if null_std < 1e-8:
            null_std = 0.1
        
        z_score = (observed - null_mean) / null_std
        p_value = 1 - 0.5 * (1 + np.erf(z_score / np.sqrt(2)))
        
        return {
            "observed": observed,
            "z_score": float(z_score),
            "p_value": float(p_value),
            "is_significant": p_value < 0.05,
        }
    except Exception:
        return {
            "observed": 0.5,
            "z_score": 0.0,
            "p_value": 1.0,
            "is_significant": False,
        }


def validate_concept(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    min_accuracy: float = 0.6,
    min_z_score: float = 2.0,
) -> Dict[str, Any]:
    """
    Validate that activations contain a real, extractable concept.
    """
    from .probe_metrics import compute_linear_probe_accuracy, compute_signal_strength
    from .icd import compute_icd
    
    try:
        linear_acc = compute_linear_probe_accuracy(pos_activations, neg_activations)
        signal = compute_signal_strength(pos_activations, neg_activations)
        icd_result = compute_icd(pos_activations, neg_activations)
        
        is_valid = (
            linear_acc >= min_accuracy and
            signal >= min_accuracy and
            icd_result["icd"] < 20
        )
        
        return {
            "is_valid": is_valid,
            "linear_accuracy": linear_acc,
            "signal_strength": signal,
            "icd": icd_result["icd"],
            "reasons": [] if is_valid else [
                f"linear_acc={linear_acc:.2f} < {min_accuracy}" if linear_acc < min_accuracy else None,
                f"signal={signal:.2f} < {min_accuracy}" if signal < min_accuracy else None,
                f"icd={icd_result['icd']:.1f} >= 20" if icd_result["icd"] >= 20 else None,
            ],
        }
    except Exception as e:
        return {
            "is_valid": False,
            "error": str(e),
        }


def compute_bootstrap_signal_estimate(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_bootstrap: int = 50,
    subset_fraction: float = 0.8,
) -> Dict[str, float]:
    """
    Estimate signal metrics with bootstrap confidence intervals.
    """
    from .probe_metrics import compute_linear_probe_accuracy
    
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        subset_size = max(int(n_pairs * subset_fraction), 5)
        
        if n_pairs < 10:
            return {
                "mean": 0.5,
                "std": 0.1,
                "ci_low": 0.4,
                "ci_high": 0.6,
            }
        
        rng = np.random.RandomState(42)
        accuracies = []
        
        for _ in range(n_bootstrap):
            indices = rng.choice(n_pairs, size=subset_size, replace=True)
            pos_sub = pos_activations[indices]
            neg_sub = neg_activations[indices]
            
            acc = compute_linear_probe_accuracy(pos_sub, neg_sub, n_folds=3)
            accuracies.append(acc)
        
        return {
            "mean": float(np.mean(accuracies)),
            "std": float(np.std(accuracies)),
            "ci_low": float(np.percentile(accuracies, 2.5)),
            "ci_high": float(np.percentile(accuracies, 97.5)),
        }
    except Exception:
        return {
            "mean": 0.5,
            "std": 0.1,
            "ci_low": 0.4,
            "ci_high": 0.6,
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
            sample_sizes = [10, 20, 50, 100, 200, 500]
        
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
                pos_activations[:size],
                neg_activations[:size],
                n_folds=min(5, size // 2),
            )
            accuracies[size] = acc
        
        # Check if last two are similar (saturated)
        sorted_sizes = sorted(accuracies.keys())
        if len(sorted_sizes) >= 2:
            last = accuracies[sorted_sizes[-1]]
            second_last = accuracies[sorted_sizes[-2]]
            is_saturated = abs(last - second_last) < 0.02
        else:
            is_saturated = False
        
        return {
            "is_saturated": is_saturated,
            "accuracies": accuracies,
            "final_accuracy": accuracies[sorted_sizes[-1]] if sorted_sizes else 0.5,
        }
    except Exception:
        return {
            "is_saturated": False,
            "accuracies": {},
            "error": "Failed to compute",
        }


def find_optimal_pair_count(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    target_accuracy: float = 0.8,
    max_pairs: int = None,
) -> Dict[str, Any]:
    """
    Find minimum number of pairs needed to reach target accuracy.
    """
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
