"""
Steerability metrics for predicting steering effectiveness.

These metrics predict whether steering (CAA, etc.) will work
for a given set of activations.
"""

import torch
import numpy as np
from typing import Dict, Any


def _empty_steerability_metrics() -> Dict[str, Any]:
    """Return empty steerability metrics when computation fails."""
    return {
        "diff_mean_alignment": None,
        "caa_probe_alignment": None,
        "pct_positive_alignment": None,
        "steering_vector_norm_ratio": None,
        "cluster_direction_angle": None,
        "per_cluster_alignment_k2": None,
        "spherical_silhouette_k2": None,
        "effective_steering_dims": None,
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
            probe = LogisticRegression( random_state=42)
            probe.fit(X, y)
            probe_dir = probe.coef_[0]
            probe_dir_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
            caa_probe_alignment = float(np.dot(diff_mean_normalized, probe_dir_norm))
        except:
            caa_probe_alignment = diff_mean_alignment
        
        return {
            "diff_mean_alignment": diff_mean_alignment,
            "caa_probe_alignment": caa_probe_alignment,
            "pct_positive_alignment": pct_positive_alignment,
            "steering_vector_norm_ratio": float(steering_vector_norm_ratio),
            "cluster_direction_angle": cluster_direction_angle,
            "per_cluster_alignment_k2": per_cluster_alignment,
            "spherical_silhouette_k2": spherical_sil,
            "effective_steering_dims": effective_dims,
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
    Return raw metrics relevant to linearity assessment.

    Does NOT compute a combined score or make recommendations.
    """
    try:
        linearity_gap = best_nonlinear_accuracy - linear_probe_accuracy
        relative_dim = diff_intrinsic_dim / ambient_dim if ambient_dim > 0 else None

        return {
            "linear_probe_accuracy": linear_probe_accuracy,
            "best_nonlinear_accuracy": best_nonlinear_accuracy,
            "linearity_gap": linearity_gap,
            "direction_stability": direction_stability,
            "diff_intrinsic_dim": diff_intrinsic_dim,
            "relative_dim": relative_dim,
            "pairwise_consistency": pairwise_consistency,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_recommendation(
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return raw metrics relevant for steering method selection.

    Does NOT make recommendations - just extracts and returns the metrics
    that would be relevant for choosing between steering methods.

    The 6 steering methods are:
    - CAA: Simple mean difference
    - Hyperplane: SVM-based
    - MLP: Neural probe
    - PRISM: Multi-directional
    - PULSE: Conditional gating
    - TITAN: Full adaptive
    """
    try:
        # Extract metrics
        linear_probe = metrics.get("linear_probe_accuracy", None)
        mlp_probe = metrics.get("mlp_probe_accuracy", None)
        signal_strength = metrics.get("signal_strength", None)
        steerability = metrics.get("steer_caa_probe_alignment", metrics.get("caa_probe_alignment", None))
        icd = metrics.get("icd_icd", metrics.get("icd", None))
        direction_stability = metrics.get("direction_stability_score", metrics.get("direction_stability", None))
        n_concepts = metrics.get("n_concepts", None)
        concept_coherence = metrics.get("concept_coherence", None)
        consistency_mean = metrics.get("consistency_consistency_score", metrics.get("consistency_mean", None))

        # Compute derived values (not thresholds, just computations)
        nonlinearity_gap = None
        if mlp_probe is not None and linear_probe is not None:
            nonlinearity_gap = mlp_probe - linear_probe

        return {
            "linear_probe": linear_probe,
            "mlp_probe": mlp_probe,
            "nonlinearity_gap": nonlinearity_gap,
            "signal_strength": signal_strength,
            "steerability": steerability,
            "icd": icd,
            "direction_stability": direction_stability,
            "n_concepts": n_concepts,
            "concept_coherence": concept_coherence,
            "consistency": consistency_mean,
        }
    except Exception as e:
        return {"error": str(e)}


def compute_adaptive_recommendation(
    metrics: Dict[str, Any],
    n_pairs: int,
) -> Dict[str, Any]:
    """
    Return raw metrics with sample size info.
    """
    base_rec = compute_recommendation(metrics)
    base_rec["n_pairs"] = n_pairs
    return base_rec


def compute_robust_recommendation(
    metrics: Dict[str, Any],
    bootstrap_metrics: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Return raw metrics with bootstrap info if available.
    """
    base_rec = compute_recommendation(metrics)
    if bootstrap_metrics:
        base_rec["bootstrap_std"] = bootstrap_metrics.get("signal_std", None)
    return base_rec


def compute_final_steering_prescription(
    per_layer_metrics: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract raw metrics per layer for steering decisions.

    Does NOT make recommendations or pick a "best" layer.
    Just returns the raw metrics for each layer so the user can decide.

    Args:
        per_layer_metrics: Dict mapping layer index to RepScan metrics for that layer

    Returns:
        Dict with per-layer raw metrics
    """
    if not per_layer_metrics:
        return {"per_layer": {}}

    per_layer = {}
    for layer, metrics in per_layer_metrics.items():
        per_layer[layer] = compute_recommendation(metrics)

    return {"per_layer": per_layer}
