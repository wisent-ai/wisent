"""
Steerability metrics for predicting steering effectiveness.

These metrics predict whether steering (CAA, etc.) will work
for a given set of activations.
"""

import torch
import numpy as np
from typing import Dict, Any


def _empty_steerability_metrics() -> Dict[str, float]:
    """Return empty steerability metrics."""
    return {
        "diff_mean_alignment": 0.0,
        "caa_probe_alignment": 0.0,
        "pct_positive_alignment": 0.5,
        "steering_vector_norm_ratio": 0.0,
        "cluster_direction_angle": 90.0,
        "per_cluster_alignment_k2": 0.0,
        "spherical_silhouette_k2": 0.0,
        "effective_steering_dims": 1,
        "steerability_score": 0.0,
    }


def compute_steerability_metrics(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics that predict whether steering will work.
    
    Key insight from experiments:
    - TruthfulQA: diff_mean_alignment=0.22, steering works (+12% accuracy)
    - HellaSwag: diff_mean_alignment=0.05, steering fails (0% improvement)
    """
    try:
        n_pairs = min(len(pos_activations), len(neg_activations))
        
        if n_pairs < 5:
            return _empty_steerability_metrics()
        
        diff_vectors = (pos_activations[:n_pairs] - neg_activations[:n_pairs]).float().cpu().numpy()
        
        diff_mean = diff_vectors.mean(axis=0)
        diff_mean_norm = np.linalg.norm(diff_mean)
        
        if diff_mean_norm < 1e-8:
            return _empty_steerability_metrics()
        
        diff_mean_normalized = diff_mean / diff_mean_norm
        
        diff_norms = np.linalg.norm(diff_vectors, axis=1, keepdims=True)
        valid_mask = diff_norms.squeeze() > 1e-8
        
        if valid_mask.sum() < 5:
            return _empty_steerability_metrics()
        
        diff_normalized = diff_vectors[valid_mask] / diff_norms[valid_mask]
        
        alignments = diff_normalized @ diff_mean_normalized
        diff_mean_alignment = float(alignments.mean())
        pct_positive_alignment = float((alignments > 0).mean())
        
        avg_diff_norm = float(diff_norms[valid_mask].mean())
        steering_vector_norm_ratio = diff_mean_norm / (avg_diff_norm + 1e-8)
        
        from sklearn.cluster import KMeans
        
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        cluster_labels = km.fit_predict(diff_normalized)
        
        c0_mask = cluster_labels == 0
        c1_mask = cluster_labels == 1
        
        if c0_mask.sum() >= 2 and c1_mask.sum() >= 2:
            dir_c0 = diff_vectors[valid_mask][c0_mask].mean(axis=0)
            dir_c1 = diff_vectors[valid_mask][c1_mask].mean(axis=0)
            
            dir_c0_norm = dir_c0 / (np.linalg.norm(dir_c0) + 1e-8)
            dir_c1_norm = dir_c1 / (np.linalg.norm(dir_c1) + 1e-8)
            
            cos_angle = np.clip(np.dot(dir_c0_norm, dir_c1_norm), -1, 1)
            cluster_direction_angle = float(np.degrees(np.arccos(np.abs(cos_angle))))
            
            align_c0 = (diff_normalized[c0_mask] @ dir_c0_norm).mean()
            align_c1 = (diff_normalized[c1_mask] @ dir_c1_norm).mean()
            per_cluster_alignment = float((align_c0 * c0_mask.sum() + align_c1 * c1_mask.sum()) / valid_mask.sum())
        else:
            cluster_direction_angle = 0.0
            per_cluster_alignment = diff_mean_alignment
        
        def spherical_silhouette(X_norm, labels):
            n = len(X_norm)
            k = len(set(labels))
            if k < 2:
                return 0.0
            
            silhouettes = []
            for i in range(min(n, 200)):
                same_cluster = labels == labels[i]
                if same_cluster.sum() > 1:
                    a = 1 - (X_norm[i] @ X_norm[same_cluster].T).mean()
                else:
                    a = 0
                
                b = float('inf')
                for c in range(k):
                    if c != labels[i]:
                        other = labels == c
                        if other.sum() > 0:
                            b_c = 1 - (X_norm[i] @ X_norm[other].T).mean()
                            b = min(b, b_c)
                
                if b != float('inf'):
                    silhouettes.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
            
            return float(np.mean(silhouettes)) if silhouettes else 0.0
        
        spherical_sil = spherical_silhouette(diff_normalized, cluster_labels)
        
        try:
            cov = np.cov(diff_normalized.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)
            
            total_var = eigenvalues.sum()
            if total_var > 0:
                cumsum = np.cumsum(eigenvalues) / total_var
                effective_dims = int(np.searchsorted(cumsum, 0.9) + 1)
            else:
                effective_dims = 1
        except:
            effective_dims = 1
        
        try:
            from sklearn.linear_model import LogisticRegression
            pos_np = pos_activations[:n_pairs].float().cpu().numpy()
            neg_np = neg_activations[:n_pairs].float().cpu().numpy()
            X = np.vstack([pos_np, neg_np])
            y = np.array([1] * n_pairs + [0] * n_pairs)
            probe = LogisticRegression(max_iter=500, random_state=42)
            probe.fit(X, y)
            probe_dir = probe.coef_[0]
            probe_dir_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
            caa_probe_alignment = float(np.dot(diff_mean_normalized, probe_dir_norm))
        except:
            caa_probe_alignment = diff_mean_alignment
        
        steerability_score = (
            0.5 * max(0, caa_probe_alignment) +
            0.2 * pct_positive_alignment +
            0.15 * min(1.0, steering_vector_norm_ratio) +
            0.15 * (1 - cluster_direction_angle / 180)
        )
        steerability_score = float(np.clip(steerability_score, 0, 1))
        
        return {
            "diff_mean_alignment": diff_mean_alignment,
            "caa_probe_alignment": caa_probe_alignment,
            "pct_positive_alignment": pct_positive_alignment,
            "steering_vector_norm_ratio": float(steering_vector_norm_ratio),
            "cluster_direction_angle": cluster_direction_angle,
            "per_cluster_alignment_k2": per_cluster_alignment,
            "spherical_silhouette_k2": spherical_sil,
            "effective_steering_dims": effective_dims,
            "steerability_score": steerability_score,
        }
    except Exception:
        return _empty_steerability_metrics()


def compute_linearity_score(
    linear_probe_accuracy: float,
    best_nonlinear_accuracy: float,
    direction_stability: float,
    diff_intrinsic_dim: float,
    pairwise_consistency: float,
    ambient_dim: int = 4096,
) -> Dict[str, Any]:
    """
    Compute overall linearity score combining multiple signals.
    
    A representation is "linear" if:
    1. Linear probe accuracy is close to nonlinear accuracy
    2. Direction is stable across subsets
    3. Intrinsic dimension of diffs is low
    4. Pairwise diff consistency is high
    """
    try:
        linearity_gap = best_nonlinear_accuracy - linear_probe_accuracy
        linearity_from_gap = max(0, 1 - linearity_gap * 5)
        
        relative_dim = diff_intrinsic_dim / ambient_dim
        linearity_from_dim = max(0, 1 - relative_dim * 10)
        
        linearity_score = (
            0.3 * linearity_from_gap +
            0.25 * direction_stability +
            0.25 * linearity_from_dim +
            0.2 * pairwise_consistency
        )
        linearity_score = float(np.clip(linearity_score, 0, 1))
        
        is_linear = linearity_score > 0.6
        
        if is_linear:
            recommendation = "CAA"
        elif linearity_score > 0.4:
            recommendation = "PRISM"
        else:
            recommendation = "TITAN"
        
        return {
            "linearity_score": linearity_score,
            "is_linear": is_linear,
            "recommendation": recommendation,
            "linearity_from_gap": linearity_from_gap,
            "linearity_from_dim": linearity_from_dim,
        }
    except Exception:
        return {
            "linearity_score": 0.5,
            "is_linear": False,
            "recommendation": "PRISM",
            "linearity_from_gap": 0.5,
            "linearity_from_dim": 0.5,
        }


def compute_recommendation(
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate steering method recommendation from metrics.
    """
    try:
        linear_probe = metrics.get("linear_probe_accuracy", 0.5)
        signal_strength = metrics.get("signal_strength", 0.5)
        steerability = metrics.get("steerability_score", 0.5)
        icd = metrics.get("icd", 10)
        
        if linear_probe > 0.7 and steerability > 0.5 and icd < 5:
            method = "CAA"
            confidence = min(linear_probe, steerability)
        elif signal_strength > 0.7 and icd < 10:
            method = "PRISM"
            confidence = signal_strength * 0.8
        else:
            method = "TITAN"
            confidence = 0.5
        
        return {
            "recommended_method": method,
            "confidence": float(confidence),
            "reasoning": f"linear={linear_probe:.2f}, signal={signal_strength:.2f}, icd={icd:.1f}",
        }
    except Exception:
        return {
            "recommended_method": "CAA",
            "confidence": 0.5,
            "reasoning": "default",
        }


def compute_adaptive_recommendation(
    metrics: Dict[str, Any],
    n_pairs: int,
) -> Dict[str, Any]:
    """
    Generate adaptive recommendation based on sample size.
    """
    base_rec = compute_recommendation(metrics)
    
    if n_pairs < 20:
        base_rec["warning"] = "Low sample size may affect reliability"
        base_rec["confidence"] *= 0.7
    
    return base_rec


def compute_robust_recommendation(
    metrics: Dict[str, Any],
    bootstrap_metrics: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Generate robust recommendation using bootstrap confidence intervals.
    """
    base_rec = compute_recommendation(metrics)
    
    if bootstrap_metrics:
        std = bootstrap_metrics.get("signal_std", 0.1)
        if std > 0.15:
            base_rec["warning"] = "High variance in metrics"
            base_rec["confidence"] *= 0.8
    
    return base_rec
