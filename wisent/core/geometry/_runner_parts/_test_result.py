"""GeometryTestResult dataclass for geometry search results."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class GeometryTestResult:
    """Result of a single geometry test."""
    benchmark: str
    strategy: str
    layers: List[int]
    # Step 1: Is there any signal? (MLP CV accuracy)
    signal_strength: float  # MLP CV accuracy, ~0.5 = no signal, >0.6 = signal exists
    has_signal: bool  # signal_strength > 0.6
    # Step 2: Is signal linear? (Linear probe CV accuracy)
    linear_probe_accuracy: float  # Linear CV accuracy, high = linear, low = nonlinear
    is_linear: bool  # linear_probe_accuracy > 0.6 AND close to signal_strength
    # Nonlinear signal metrics
    knn_accuracy_k5: float
    knn_accuracy_k10: float
    knn_accuracy_k20: float
    knn_pca_accuracy: float
    knn_umap_accuracy: float
    knn_pacmap_accuracy: float
    mlp_probe_accuracy: float
    best_nonlinear_accuracy: float
    mmd_rbf: float
    local_dim_pos: float
    local_dim_neg: float
    local_dim_ratio: float
    fisher_max: float
    fisher_gini: float
    fisher_top10_ratio: float
    num_dims_fisher_above_1: int
    density_ratio: float
    # Step 3: Geometry details (only meaningful if has_signal=True)
    best_structure: str
    best_score: float
    # All structure scores
    linear_score: float
    cone_score: float
    orthogonal_score: float
    manifold_score: float
    sparse_score: float
    cluster_score: float
    bimodal_score: float
    # Linear details
    cohens_d: float
    variance_explained: float
    within_class_consistency: float
    # Cone details
    raw_mean_cosine_similarity: float
    positive_correlation_fraction: float
    # Orthogonal details
    near_zero_fraction: float
    # Manifold details
    pca_top2_variance: float
    local_nonlinearity: float
    # Sparse details
    gini_coefficient: float
    active_fraction: float
    top_10_contribution: float
    # Cluster details
    best_silhouette: float
    best_k: int
    # Multi-direction analysis
    multi_dir_accuracy_k1: float
    multi_dir_accuracy_k2: float
    multi_dir_accuracy_k3: float
    multi_dir_accuracy_k5: float
    multi_dir_accuracy_k10: float
    multi_dir_min_k_for_good: int
    multi_dir_saturation_k: int
    multi_dir_gain: float
    # Steerability metrics
    diff_mean_alignment: float
    pct_positive_alignment: float
    steering_vector_norm_ratio: float
    cluster_direction_angle: float
    per_cluster_alignment_k2: float
    spherical_silhouette_k2: float
    effective_steering_dims: int
    steerability_score: float
    # Recommendation
    recommended_method: str
    # ICD metrics
    icd: float = 0.0
    icd_top1_variance: float = 0.0
    icd_top5_variance: float = 0.0
    # Nonsense baseline comparison
    nonsense_baseline_accuracy: float = 0.5
    signal_vs_baseline_ratio: float = 1.0
    signal_above_baseline: float = 0.0
    has_real_signal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "strategy": self.strategy,
            "layers": self.layers,
            "signal_strength": self.signal_strength,
            "has_signal": self.has_signal,
            "linear_probe_accuracy": self.linear_probe_accuracy,
            "is_linear": self.is_linear,
            "nonlinear_metrics": {
                "knn_accuracy_k5": self.knn_accuracy_k5,
                "knn_accuracy_k10": self.knn_accuracy_k10,
                "knn_accuracy_k20": self.knn_accuracy_k20,
                "knn_pca_accuracy": self.knn_pca_accuracy,
                "knn_umap_accuracy": self.knn_umap_accuracy,
                "knn_pacmap_accuracy": self.knn_pacmap_accuracy,
                "mlp_probe_accuracy": self.mlp_probe_accuracy,
                "best_nonlinear_accuracy": self.best_nonlinear_accuracy,
                "mmd_rbf": self.mmd_rbf,
                "local_dim_pos": self.local_dim_pos,
                "local_dim_neg": self.local_dim_neg,
                "local_dim_ratio": self.local_dim_ratio,
                "fisher_max": self.fisher_max,
                "fisher_gini": self.fisher_gini,
                "fisher_top10_ratio": self.fisher_top10_ratio,
                "num_dims_fisher_above_1": self.num_dims_fisher_above_1,
                "density_ratio": self.density_ratio,
            },
            "best_structure": self.best_structure,
            "best_score": self.best_score,
            "structure_scores": {
                "linear": self.linear_score,
                "cone": self.cone_score,
                "orthogonal": self.orthogonal_score,
                "manifold": self.manifold_score,
                "sparse": self.sparse_score,
                "cluster": self.cluster_score,
                "bimodal": self.bimodal_score,
            },
            "linear_details": {
                "cohens_d": self.cohens_d,
                "variance_explained": self.variance_explained,
                "within_class_consistency": self.within_class_consistency,
            },
            "cone_details": {
                "raw_mean_cosine_similarity": self.raw_mean_cosine_similarity,
                "positive_correlation_fraction": self.positive_correlation_fraction,
            },
            "orthogonal_details": {
                "near_zero_fraction": self.near_zero_fraction,
            },
            "manifold_details": {
                "pca_top2_variance": self.pca_top2_variance,
                "local_nonlinearity": self.local_nonlinearity,
            },
            "sparse_details": {
                "gini_coefficient": self.gini_coefficient,
                "active_fraction": self.active_fraction,
                "top_10_contribution": self.top_10_contribution,
            },
            "cluster_details": {
                "best_silhouette": self.best_silhouette,
                "best_k": self.best_k,
            },
            "multi_direction_analysis": {
                "accuracy_k1": self.multi_dir_accuracy_k1,
                "accuracy_k2": self.multi_dir_accuracy_k2,
                "accuracy_k3": self.multi_dir_accuracy_k3,
                "accuracy_k5": self.multi_dir_accuracy_k5,
                "accuracy_k10": self.multi_dir_accuracy_k10,
                "min_k_for_good": self.multi_dir_min_k_for_good,
                "saturation_k": self.multi_dir_saturation_k,
                "gain_from_multi": self.multi_dir_gain,
            },
            "steerability_metrics": {
                "diff_mean_alignment": self.diff_mean_alignment,
                "pct_positive_alignment": self.pct_positive_alignment,
                "steering_vector_norm_ratio": self.steering_vector_norm_ratio,
                "cluster_direction_angle": self.cluster_direction_angle,
                "per_cluster_alignment_k2": self.per_cluster_alignment_k2,
                "spherical_silhouette_k2": self.spherical_silhouette_k2,
                "effective_steering_dims": self.effective_steering_dims,
                "steerability_score": self.steerability_score,
            },
            "icd_metrics": {
                "icd": self.icd,
                "top1_variance": self.icd_top1_variance,
                "top5_variance": self.icd_top5_variance,
            },
            "nonsense_baseline": {
                "nonsense_accuracy": self.nonsense_baseline_accuracy,
                "signal_ratio": self.signal_vs_baseline_ratio,
                "signal_above_baseline": self.signal_above_baseline,
                "has_real_signal": self.has_real_signal,
            },
            "recommended_method": self.recommended_method,
        }
