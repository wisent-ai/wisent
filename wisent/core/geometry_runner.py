"""
Geometry search runner.

Runs geometry tests across the search space using cached activations.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch

import numpy as np

from wisent.core.geometry_search_space import GeometrySearchSpace, GeometrySearchConfig
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.activation_cache import (
    ActivationCache,
    CachedActivations,
    RawActivationCache,
    RawCachedActivations,
    collect_and_cache_activations,
    collect_and_cache_raw_activations,
    get_strategy_text_family,
)
from wisent.core.utils.layer_combinations import get_layer_combinations

# Import functions from modular geometry package
from wisent.core.geometry import (
    compute_signal_strength,
    compute_linear_probe_accuracy,
    compute_mlp_probe_accuracy,
    compute_knn_accuracy,
    compute_knn_pca_accuracy,
    compute_knn_umap_accuracy,
    compute_knn_pacmap_accuracy,
    compute_mmd_rbf,
    compute_density_ratio,
    compute_fisher_per_dimension,
    estimate_local_intrinsic_dim,
    compute_local_intrinsic_dims,
    compute_diff_intrinsic_dim,
    compute_direction_from_pairs,
    compute_direction_stability,
    compute_multi_direction_accuracy,
    compute_pairwise_diff_consistency,
    compute_steerability_metrics,
    compute_linearity_score,
    compute_recommendation,
    compute_adaptive_recommendation,
    compute_robust_recommendation,
    generate_nonsense_activations,
    compute_nonsense_baseline,
    analyze_with_nonsense_baseline,
    detect_multiple_concepts,
    split_by_concepts,
    analyze_concept_independence,
    compute_concept_coherence,
    compute_concept_stability,
    decompose_into_concepts,
    find_mixed_pairs,
    get_pure_concept_pairs,
    recommend_per_concept_steering,
    compute_signal_to_noise,
    compute_null_distribution,
    compare_to_null,
    validate_concept,
    compute_bootstrap_signal_estimate,
    compute_saturation_check,
    find_optimal_pair_count,
    compute_icd,
    run_full_repscan,
    run_full_repscan_with_layer_search,
    run_full_repscan_with_steering_eval,
    evaluate_steering_effectiveness,
    evaluate_activation_regions,
    compute_geometry_metrics,
    TransformerComponent,
    analyze_transformer_components,
    get_component_hook_points,
    compare_components_for_benchmark,
    compare_concept_granularity,
)


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
    
    # NEW: Nonlinear signal metrics
    knn_accuracy_k5: float  # k-NN CV accuracy with k=5
    knn_accuracy_k10: float  # k-NN CV accuracy with k=10
    knn_accuracy_k20: float  # k-NN CV accuracy with k=20
    knn_pca_accuracy: float  # k-NN on PCA-50 features (addresses curse of dimensionality)
    knn_umap_accuracy: float  # k-NN on UMAP-10 features (preserves nonlinear structure)
    knn_pacmap_accuracy: float  # k-NN on PaCMAP-10 features (preserves local+global structure)
    mlp_probe_accuracy: float  # MLP probe accuracy (nonlinear baseline)
    best_nonlinear_accuracy: float  # max(knn_k10, knn_pca, knn_umap, knn_pacmap, mlp) - best nonlinear signal
    mmd_rbf: float  # Maximum Mean Discrepancy with RBF kernel
    local_dim_pos: float  # Local intrinsic dimension of positive class
    local_dim_neg: float  # Local intrinsic dimension of negative class
    local_dim_ratio: float  # Ratio of local dimensions
    fisher_max: float  # Max Fisher ratio across all dimensions
    fisher_gini: float  # Gini coefficient of Fisher ratios (concentration)
    fisher_top10_ratio: float  # Fraction of total Fisher in top 10 dims
    num_dims_fisher_above_1: int  # Number of dimensions with Fisher > 1
    density_ratio: float  # Ratio of avg intra-class distances
    
    # Step 3: Geometry details (only meaningful if has_signal=True)
    # Best structure detected
    best_structure: str  # 'linear', 'cone', 'cluster', 'manifold', 'sparse', 'bimodal', 'orthogonal'
    best_score: float
    
    # All structure scores
    linear_score: float
    cone_score: float
    orthogonal_score: float
    manifold_score: float
    sparse_score: float
    cluster_score: float
    bimodal_score: float
    
    # Detailed metrics per structure
    # Linear
    cohens_d: float  # separation quality
    variance_explained: float  # by primary direction
    within_class_consistency: float
    
    # Cone
    raw_mean_cosine_similarity: float  # between diff vectors
    positive_correlation_fraction: float  # fraction in same half-space
    
    # Orthogonal
    near_zero_fraction: float  # fraction of near-zero correlations
    
    # Manifold
    pca_top2_variance: float  # variance by top 2 PCs
    local_nonlinearity: float  # curvature measure
    
    # Sparse
    gini_coefficient: float  # inequality of activations
    active_fraction: float  # fraction of active neurons
    top_10_contribution: float  # contribution of top 10 neurons
    
    # Cluster
    best_silhouette: float  # clustering quality
    best_k: int  # optimal number of clusters
    
    # Multi-direction analysis: how many directions needed?
    multi_dir_accuracy_k1: float  # accuracy with 1 direction (same as diff-mean)
    multi_dir_accuracy_k2: float  # accuracy with 2 directions
    multi_dir_accuracy_k3: float  # accuracy with 3 directions
    multi_dir_accuracy_k5: float  # accuracy with 5 directions
    multi_dir_accuracy_k10: float  # accuracy with 10 directions
    multi_dir_min_k_for_good: int  # minimum k where accuracy >= 0.6 (-1 if never)
    multi_dir_saturation_k: int  # k where accuracy stops improving
    multi_dir_gain: float  # gain from using multiple directions vs 1
    
    # Steerability metrics: predict whether CAA steering will work
    # Key insight: TQA has diff_mean_alignment=0.22 (steering works +12%)
    #              HS has diff_mean_alignment=0.05 (steering fails 0%)
    diff_mean_alignment: float  # mean cosine between individual diffs and diff-mean
    pct_positive_alignment: float  # % of pairs where alignment > 0
    steering_vector_norm_ratio: float  # norm of diff-mean / avg individual diff norm
    cluster_direction_angle: float  # angle between k=2 cluster steering directions
    per_cluster_alignment_k2: float  # alignment within k=2 clusters
    spherical_silhouette_k2: float  # silhouette using cosine distance
    effective_steering_dims: int  # how many dimensions explain 90% of diff variance
    steerability_score: float  # overall 0-1 score predicting steering success
    
    # Recommendation
    recommended_method: str
    
    # ICD (Intrinsic Concept Dimensionality) metrics
    icd: float = 0.0  # Effective rank of difference vectors
    icd_top1_variance: float = 0.0  # Variance explained by top direction
    icd_top5_variance: float = 0.0  # Variance explained by top 5 directions
    
    # Nonsense baseline comparison (if computed)
    nonsense_baseline_accuracy: float = 0.5  # Accuracy on random token pairs
    signal_vs_baseline_ratio: float = 1.0  # real_acc / nonsense_acc
    signal_above_baseline: float = 0.0  # real_acc - nonsense_acc
    has_real_signal: bool = False  # True if signal significantly above baseline
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "strategy": self.strategy,
            "layers": self.layers,
            # Step 1: Signal detection
            "signal_strength": self.signal_strength,
            "has_signal": self.has_signal,
            # Step 2: Linearity check
            "linear_probe_accuracy": self.linear_probe_accuracy,
            "is_linear": self.is_linear,
            # NEW: Nonlinear signal metrics
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
            # Step 3: Geometry (only meaningful if has_signal)
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



@dataclass
class GeometrySearchResults:
    """Results from a full geometry search."""
    model_name: str
    config: GeometrySearchConfig
    results: List[GeometryTestResult] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    extraction_time_seconds: float = 0.0
    test_time_seconds: float = 0.0
    
    # Counts
    benchmarks_tested: int = 0
    strategies_tested: int = 0
    layer_combos_tested: int = 0
    
    def add_result(self, result: GeometryTestResult) -> None:
        self.results.append(result)
    
    def get_best_by_linear_score(self, n: int = 10) -> List[GeometryTestResult]:
        """Get top N configurations by linear score."""
        return sorted(self.results, key=lambda r: r.linear_score, reverse=True)[:n]
    
    def get_best_by_structure(self, structure: str, n: int = 10) -> List[GeometryTestResult]:
        """Get top N configurations by a specific structure score."""
        score_attr = f"{structure}_score"
        return sorted(
            self.results, 
            key=lambda r: getattr(r, score_attr, 0.0), 
            reverse=True
        )[:n]
    
    def get_structure_distribution(self) -> Dict[str, int]:
        """Count how many configurations have each structure as best."""
        counts: Dict[str, int] = {}
        for r in self.results:
            s = r.best_structure
            counts[s] = counts.get(s, 0) + 1
        return counts
    
    def get_summary_by_benchmark(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics grouped by benchmark."""
        by_bench: Dict[str, List[float]] = {}
        for r in self.results:
            if r.benchmark not in by_bench:
                by_bench[r.benchmark] = []
            by_bench[r.benchmark].append(r.linear_score)
        
        return {
            bench: {
                "mean": sum(scores) / len(scores),
                "max": max(scores),
                "min": min(scores),
                "count": len(scores),
            }
            for bench, scores in by_bench.items()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config": self.config.to_dict(),
            "total_time_seconds": self.total_time_seconds,
            "extraction_time_seconds": self.extraction_time_seconds,
            "test_time_seconds": self.test_time_seconds,
            "benchmarks_tested": self.benchmarks_tested,
            "strategies_tested": self.strategies_tested,
            "layer_combos_tested": self.layer_combos_tested,
            "results": [r.to_dict() for r in self.results],
        }
    
    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "GeometrySearchResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        config = GeometrySearchConfig(
            pairs_per_benchmark=data.get("config", {}).get("pairs_per_benchmark", 50),
            max_layer_combo_size=data.get("config", {}).get("max_layer_combo_size", 3),
            random_seed=data.get("config", {}).get("random_seed", 42),
        )
        
        results = cls(
            model_name=data.get("model_name", "unknown"),
            config=config,
            total_time_seconds=data.get("total_time_seconds", 0.0),
            extraction_time_seconds=data.get("extraction_time_seconds", 0.0),
            test_time_seconds=data.get("test_time_seconds", 0.0),
            benchmarks_tested=data.get("benchmarks_tested", 0),
            strategies_tested=data.get("strategies_tested", 0),
            layer_combos_tested=data.get("layer_combos_tested", 0),
        )
        
        # Load individual test results
        for r_data in data.get("results", []):
            try:
                result = GeometryTestResult(
                    benchmark=r_data.get("benchmark", "unknown"),
                    strategy=r_data.get("strategy", "unknown"),
                    layers=r_data.get("layers", []),
                    n_samples=r_data.get("n_samples", 0),
                    signal_strength=r_data.get("signal_strength", 0.5),
                    linear_score=r_data.get("linear_score", 0.0),
                    linear_probe_accuracy=r_data.get("linear_probe_accuracy", 0.5),
                    best_structure=r_data.get("best_structure", "unknown"),
                    is_linear=r_data.get("is_linear", False),
                    cohens_d=r_data.get("cohens_d", 0.0),
                    knn_accuracy_k5=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k5", 0.5),
                    knn_accuracy_k10=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k10", 0.5),
                    knn_accuracy_k20=r_data.get("nonlinear_signals", {}).get("knn_accuracy_k20", 0.5),
                    knn_pca_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pca_accuracy", 0.5),
                    knn_umap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_umap_accuracy", 0.5),
                    knn_pacmap_accuracy=r_data.get("nonlinear_signals", {}).get("knn_pacmap_accuracy", 0.5),
                    mlp_probe_accuracy=r_data.get("nonlinear_signals", {}).get("mlp_probe_accuracy", 0.5),
                    mmd_rbf=r_data.get("nonlinear_signals", {}).get("mmd_rbf", 0.0),
                    local_dim_pos=r_data.get("nonlinear_signals", {}).get("local_dim_pos", 0.0),
                    local_dim_neg=r_data.get("nonlinear_signals", {}).get("local_dim_neg", 0.0),
                    local_dim_ratio=r_data.get("nonlinear_signals", {}).get("local_dim_ratio", 1.0),
                    fisher_max=r_data.get("nonlinear_signals", {}).get("fisher_max", 0.0),
                    fisher_mean=r_data.get("nonlinear_signals", {}).get("fisher_mean", 0.0),
                    fisher_top10_mean=r_data.get("nonlinear_signals", {}).get("fisher_top10_mean", 0.0),
                    density_ratio=r_data.get("nonlinear_signals", {}).get("density_ratio", 1.0),
                    manifold_score=r_data.get("manifold_score", 0.0),
                    cluster_score=r_data.get("cluster_score", 0.0),
                    sparse_score=r_data.get("sparse_score", 0.0),
                    hybrid_score=r_data.get("hybrid_score", 0.0),
                    all_scores={},
                    pca_top2_variance=r_data.get("manifold_details", {}).get("pca_top2_variance", 0.0),
                    local_nonlinearity=r_data.get("manifold_details", {}).get("local_nonlinearity", 0.0),
                    gini_coefficient=r_data.get("sparse_details", {}).get("gini_coefficient", 0.0),
                    active_fraction=r_data.get("sparse_details", {}).get("active_fraction", 0.0),
                    top_10_contribution=r_data.get("sparse_details", {}).get("top_10_contribution", 0.0),
                    best_silhouette=r_data.get("cluster_details", {}).get("best_silhouette", 0.0),
                    best_k=r_data.get("cluster_details", {}).get("best_k", 2),
                    multi_dir_accuracy_k1=r_data.get("multi_direction_analysis", {}).get("accuracy_k1", 0.5),
                    multi_dir_accuracy_k2=r_data.get("multi_direction_analysis", {}).get("accuracy_k2", 0.5),
                    multi_dir_accuracy_k3=r_data.get("multi_direction_analysis", {}).get("accuracy_k3", 0.5),
                    multi_dir_accuracy_k5=r_data.get("multi_direction_analysis", {}).get("accuracy_k5", 0.5),
                    multi_dir_accuracy_k10=r_data.get("multi_direction_analysis", {}).get("accuracy_k10", 0.5),
                    multi_dir_min_k_for_good=r_data.get("multi_direction_analysis", {}).get("min_k_for_good", -1),
                    multi_dir_saturation_k=r_data.get("multi_direction_analysis", {}).get("saturation_k", 1),
                    multi_dir_gain=r_data.get("multi_direction_analysis", {}).get("gain_from_multi", 0.0),
                    diff_mean_alignment=r_data.get("steerability_metrics", {}).get("diff_mean_alignment", 0.0),
                    pct_positive_alignment=r_data.get("steerability_metrics", {}).get("pct_positive_alignment", 0.5),
                    steering_vector_norm_ratio=r_data.get("steerability_metrics", {}).get("steering_vector_norm_ratio", 0.0),
                    cluster_direction_angle=r_data.get("steerability_metrics", {}).get("cluster_direction_angle", 90.0),
                    per_cluster_alignment_k2=r_data.get("steerability_metrics", {}).get("per_cluster_alignment_k2", 0.0),
                    spherical_silhouette_k2=r_data.get("steerability_metrics", {}).get("spherical_silhouette_k2", 0.0),
                    effective_steering_dims=r_data.get("steerability_metrics", {}).get("effective_steering_dims", 1),
                    steerability_score=r_data.get("steerability_metrics", {}).get("steerability_score", 0.0),
                    icd=r_data.get("icd_metrics", {}).get("icd", 0.0),
                    icd_top1_variance=r_data.get("icd_metrics", {}).get("top1_variance", 0.0),
                    icd_top5_variance=r_data.get("icd_metrics", {}).get("top5_variance", 0.0),
                    nonsense_baseline_accuracy=r_data.get("nonsense_baseline", {}).get("nonsense_accuracy", 0.5),
                    signal_vs_baseline_ratio=r_data.get("nonsense_baseline", {}).get("signal_ratio", 1.0),
                    signal_above_baseline=r_data.get("nonsense_baseline", {}).get("signal_above_baseline", 0.0),
                    has_real_signal=r_data.get("nonsense_baseline", {}).get("has_real_signal", True),
                    recommended_method=r_data.get("recommended_method", "unknown"),
                )
                results.results.append(result)
            except Exception:
                pass
        
        return results



@dataclass
class RepScanResult:
    """Complete result from run_full_repscan()."""
    # Basic info
    benchmark: str
    model_name: str
    layer: int
    
    # Pair optimization
    optimal_n_pairs: int
    pair_search_history: List[Dict[str, Any]]
    saturation_status: str  # "OPTIMAL", "MAX_REACHED", "NO_SIGNAL"
    
    # Signal classification
    signal_exists: bool
    signal_type: str  # "LINEAR", "NONLINEAR", "MULTIMODAL", "NO_SIGNAL"
    
    # Method recommendation
    recommended_method: str  # "CAA", "PRISM", "PULSE", "TITAN", "NO_METHOD"
    confidence: float
    reason: str
    
    # Core metrics
    linear_probe_accuracy: float
    best_nonlinear_accuracy: float
    knn_umap_accuracy: float
    knn_pacmap_accuracy: float
    
    # Geometry metrics (diff vectors)
    icd: float
    icd_top1_variance: float
    diff_mean_alignment: float
    caa_probe_alignment: float  # NEW: cosine(CAA_direction, probe_direction) - key predictor!
    steerability_score: float
    effective_steering_dims: int
    
    # Multi-direction metrics
    multi_dir_gain: float
    spherical_silhouette_k2: float
    cluster_direction_angle: float
    
    # Quality metrics
    cohens_d: float
    bootstrap_std: float
    direction_stability: float  # cosine similarity between bootstraps
    
    # Steering vector (if signal exists)
    steering_direction: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding tensor)."""
        d = {
            "benchmark": self.benchmark,
            "model_name": self.model_name,
            "layer": self.layer,
            "optimal_n_pairs": self.optimal_n_pairs,
            "pair_search_history": self.pair_search_history,
            "saturation_status": self.saturation_status,
            "signal_exists": self.signal_exists,
            "signal_type": self.signal_type,
            "recommended_method": self.recommended_method,
            "confidence": self.confidence,
            "reason": self.reason,
            "linear_probe_accuracy": self.linear_probe_accuracy,
            "best_nonlinear_accuracy": self.best_nonlinear_accuracy,
            "knn_umap_accuracy": self.knn_umap_accuracy,
            "knn_pacmap_accuracy": self.knn_pacmap_accuracy,
            "icd": self.icd,
            "icd_top1_variance": self.icd_top1_variance,
            "diff_mean_alignment": self.diff_mean_alignment,
            "caa_probe_alignment": self.caa_probe_alignment,
            "steerability_score": self.steerability_score,
            "effective_steering_dims": self.effective_steering_dims,
            "multi_dir_gain": self.multi_dir_gain,
            "spherical_silhouette_k2": self.spherical_silhouette_k2,
            "cluster_direction_angle": self.cluster_direction_angle,
            "cohens_d": self.cohens_d,
            "bootstrap_std": self.bootstrap_std,
            "direction_stability": self.direction_stability,
        }
        return d



@dataclass
class RepScanLayerResult:
    """Result from layer search - extends RepScanResult with layer analysis."""
    # Best layer result
    best_result: RepScanResult
    
    # Layer search info
    optimal_layer: int
    optimal_layer_range: List[int]  # Top layers within 5% of best
    layer_search_history: List[Dict[str, Any]]
    
    # Model info
    num_layers: int
    layers_tested: List[int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = self.best_result.to_dict()
        d.update({
            "optimal_layer": self.optimal_layer,
            "optimal_layer_range": self.optimal_layer_range,
            "layer_search_history": self.layer_search_history,
            "num_layers": self.num_layers,
            "layers_tested": self.layers_tested,
        })
        return d



@dataclass
class SteeringEvaluationResult:
    """Result from evaluating steering effectiveness."""
    # Basic info
    steering_strength: float
    
    # Activation shift metrics
    neg_to_pos_shift: float  # How much neg activations moved toward pos
    pos_stability: float  # How much pos activations stayed in place
    separation_after: float  # Linear probe accuracy after steering
    
    # Classification metrics
    neg_classified_as_pos_before: float  # % of neg classified as pos before
    neg_classified_as_pos_after: float  # % of neg classified as pos after
    flip_rate: float  # % of neg that flipped to pos classification
    
    # Geometric metrics
    cosine_shift_neg: float  # Avg cosine similarity of neg shift to steering direction
    magnitude_ratio: float  # |steering_vector| / |avg_activation|
    
    # Quality metrics
    steering_effective: bool  # Did steering work?
    optimal_strength: float  # Best strength found
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steering_strength": self.steering_strength,
            "neg_to_pos_shift": self.neg_to_pos_shift,
            "pos_stability": self.pos_stability,
            "separation_after": self.separation_after,
            "neg_classified_as_pos_before": self.neg_classified_as_pos_before,
            "neg_classified_as_pos_after": self.neg_classified_as_pos_after,
            "flip_rate": self.flip_rate,
            "cosine_shift_neg": self.cosine_shift_neg,
            "magnitude_ratio": self.magnitude_ratio,
            "steering_effective": self.steering_effective,
            "optimal_strength": self.optimal_strength,
        }



@dataclass 
class ComponentAnalysisResult:
    """Result from analyzing different transformer components."""
    # Which component has strongest signal
    best_component: str  # "residual", "mlp", "attn", "head_X"
    best_component_accuracy: float
    
    # Per-component results
    residual_accuracy: float
    mlp_accuracy: float
    attn_accuracy: float
    head_accuracies: Dict[int, float]  # head_idx -> accuracy
    
    # Best attention heads
    top_heads: List[Tuple[int, float]]  # [(head_idx, accuracy), ...]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "best_component": self.best_component,
            "best_component_accuracy": self.best_component_accuracy,
            "residual_accuracy": self.residual_accuracy,
            "mlp_accuracy": self.mlp_accuracy,
            "attn_accuracy": self.attn_accuracy,
            "head_accuracies": self.head_accuracies,
            "top_heads": self.top_heads,
        }



class TransformerComponent(Enum):
    """Which part of transformer to extract activations from."""
    RESIDUAL = "residual"      # Block output (residual stream after MLP)
    RESIDUAL_PRE = "residual_pre"  # Before attention
    RESIDUAL_MID = "residual_mid"  # After attention, before MLP
    MLP_OUTPUT = "mlp_output"  # MLP output only
    ATTN_OUTPUT = "attn_output"  # Attention output only
    ATTN_HEAD = "attn_head"    # Individual attention head



class ComponentActivationExtractor:
    """
    Extract activations from specific transformer components using hooks.
    
    Example:
        ```python
        extractor = ComponentActivationExtractor(model, tokenizer)
        
        # Get MLP output at layer 15
        mlp_acts = extractor.extract(
            texts=["Hello world", "Goodbye world"],
            layer=15,
            component=TransformerComponent.MLP_OUTPUT,
        )
        ```
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._hook_outputs = {}
        self._hooks = []
    
    def _get_module_by_name(self, name: str):
        """Get a module by its full name."""
        parts = name.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _register_hook(self, module_name: str):
        """Register a forward hook on a module."""
        try:
            module = self._get_module_by_name(module_name)
            
            def hook_fn(module, input, output):
                # Handle different output formats
                if isinstance(output, tuple):
                    self._hook_outputs[module_name] = output[0].detach()
                else:
                    self._hook_outputs[module_name] = output.detach()
            
            handle = module.register_forward_hook(hook_fn)
            self._hooks.append(handle)
            return True
        except Exception as e:
            print(f"Warning: Could not register hook for {module_name}: {e}")
            return False
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._hook_outputs = {}
    
    def extract(
        self,
        texts: List[str],
        layer: int,
        component: TransformerComponent,
        token_position: int = -1,  # -1 = last token
    ) -> torch.Tensor:
        """
        Extract activations from a specific component.
        
        Args:
            texts: List of input texts
            layer: Layer index
            component: Which component (RESIDUAL, MLP_OUTPUT, ATTN_OUTPUT, etc.)
            token_position: Which token position to extract (-1 = last)
            
        Returns:
            Tensor of shape [len(texts), hidden_dim]
        """
        # Detect model type
        model_type = type(self.model).__name__
        
        # Get hook points
        hook_points = get_component_hook_points(model_type, layer, component)
        
        if not hook_points:
            raise ValueError(f"No hook points found for {component} in {model_type}")
        
        # Clear previous hooks
        self._clear_hooks()
        
        # Register hooks
        for hook_point in hook_points:
            self._register_hook(hook_point)
        
        activations = []
        
        try:
            with torch.no_grad():
                for text in texts:
                    # Tokenize
                    inputs = self.tokenizer(
                        text, 
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    
                    # Forward pass (hooks will capture activations)
                    _ = self.model(**inputs)
                    
                    # Get hooked output
                    for hook_point in hook_points:
                        if hook_point in self._hook_outputs:
                            output = self._hook_outputs[hook_point]
                            # output shape: [batch, seq_len, hidden_dim]
                            act = output[0, token_position, :]  # [hidden_dim]
                            activations.append(act.cpu())
                            break
                    
                    # Clear for next iteration
                    self._hook_outputs = {}
        
        finally:
            self._clear_hooks()
        
        if not activations:
            raise RuntimeError(f"Failed to extract activations for {component}")
        
        return torch.stack(activations)
    
    def extract_all_components(
        self,
        texts: List[str],
        layer: int,
        token_position: int = -1,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract activations from all components at a layer.
        
        Returns:
            Dict mapping component name to activations tensor
        """
        results = {}
        
        for component in [
            TransformerComponent.RESIDUAL,
            TransformerComponent.MLP_OUTPUT,
            TransformerComponent.ATTN_OUTPUT,
        ]:
            try:
                acts = self.extract(texts, layer, component, token_position)
                results[component.value] = acts
            except Exception as e:
                print(f"Warning: Failed to extract {component.value}: {e}")
        
        return results



@dataclass
class MultiConceptAnalysis:
    """Result from analyzing if multiple concepts exist in a contrastive pair set."""
    # Overall assessment
    num_concepts_detected: int
    is_multi_concept: bool
    confidence: float
    
    # Evidence from different methods
    icd: float  # Intrinsic Concept Dimensionality
    icd_suggests_multi: bool
    
    cluster_count: int  # Number of clusters in diff vectors
    cluster_silhouette: float  # Cluster quality
    clusters_suggest_multi: bool
    
    pca_variance_ratio: List[float]  # Top-k explained variance ratios
    pca_effective_rank: float  # How many PCs needed for 90% variance
    pca_suggests_multi: bool
    
    multi_dir_accuracy: Dict[int, float]  # k -> accuracy with k directions
    multi_dir_gain: float  # Gain from using multiple directions
    directions_suggest_multi: bool
    
    # Per-concept info (if clusters found)
    concept_directions: Optional[List[torch.Tensor]] = None
    concept_sizes: Optional[List[int]] = None
    concept_accuracies: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_concepts_detected": self.num_concepts_detected,
            "is_multi_concept": self.is_multi_concept,
            "confidence": self.confidence,
            "icd": self.icd,
            "icd_suggests_multi": self.icd_suggests_multi,
            "cluster_count": self.cluster_count,
            "cluster_silhouette": self.cluster_silhouette,
            "clusters_suggest_multi": self.clusters_suggest_multi,
            "pca_variance_ratio": self.pca_variance_ratio,
            "pca_effective_rank": self.pca_effective_rank,
            "pca_suggests_multi": self.pca_suggests_multi,
            "multi_dir_accuracy": self.multi_dir_accuracy,
            "multi_dir_gain": self.multi_dir_gain,
            "directions_suggest_multi": self.directions_suggest_multi,
            "concept_sizes": self.concept_sizes,
            "concept_accuracies": self.concept_accuracies,
        }



@dataclass
class ConceptValidityResult:
    """Result from analyzing if a set of pairs represents a valid concept."""
    # Overall
    is_valid_concept: bool
    validity_score: float  # 0-1, higher = more valid
    concept_level: str  # "instance", "category", "domain", "noise"
    
    # Internal coherence - do all pairs point same direction?
    coherence_score: float  # 0-1, cosine similarity of individual diffs to mean
    coherence_std: float
    is_coherent: bool
    
    # Stability - does direction hold with subsamples?
    stability_score: float  # 0-1, avg cosine sim between subsample directions
    stability_std: float
    is_stable: bool
    
    # Signal quality
    signal_strength: float  # Linear probe accuracy
    signal_to_noise: float  # Cohen's d or similar
    has_signal: bool
    
    # Granularity indicators
    icd: float  # Lower = more abstract concept
    specificity: float  # How specific vs general is this concept
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid_concept": self.is_valid_concept,
            "validity_score": self.validity_score,
            "concept_level": self.concept_level,
            "coherence_score": self.coherence_score,
            "coherence_std": self.coherence_std,
            "is_coherent": self.is_coherent,
            "stability_score": self.stability_score,
            "stability_std": self.stability_std,
            "is_stable": self.is_stable,
            "signal_strength": self.signal_strength,
            "signal_to_noise": self.signal_to_noise,
            "has_signal": self.has_signal,
            "icd": self.icd,
            "specificity": self.specificity,
        }



@dataclass
class ConceptDecomposition:
    """Result from decomposing a contrastive pair set into constituent concepts."""
    # How many concepts found
    n_concepts: int
    decomposition_method: str  # "clustering", "nmf", "ica"
    
    # Per-concept info
    concept_directions: List[torch.Tensor]
    concept_sizes: List[int]  # How many pairs belong to each
    concept_coherences: List[float]
    concept_validities: List[bool]
    
    # Relationships between concepts
    concept_relationships: List[Dict[str, Any]]  # List of {concept_i, concept_j, relationship, strength}
    
    # Pair assignments
    pair_to_concept: List[int]  # Which concept each pair belongs to (-1 if unclear)
    pair_concept_scores: np.ndarray  # [n_pairs, n_concepts] soft assignment scores
    
    # Overall quality
    total_explained_variance: float
    reconstruction_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_concepts": self.n_concepts,
            "decomposition_method": self.decomposition_method,
            "concept_sizes": self.concept_sizes,
            "concept_coherences": self.concept_coherences,
            "concept_validities": self.concept_validities,
            "concept_relationships": self.concept_relationships,
            "pair_to_concept": self.pair_to_concept,
            "total_explained_variance": self.total_explained_variance,
            "reconstruction_error": self.reconstruction_error,
        }



class GeometryRunner:
    """
    Runs geometry search across the search space.
    
    Uses activation caching for efficiency:
    1. Extract ALL layers once per (benchmark, strategy)
    2. Test all layer combinations from cache
    3. Compare against nonsense baseline (random tokens)
    """
    
    def __init__(
        self,
        search_space: GeometrySearchSpace,
        model: "WisentModel",
        cache_dir: Optional[str] = None,
    ):
        self.search_space = search_space
        self.model = model
        self.cache_dir = cache_dir or f"/tmp/wisent_geometry_cache_{model.model_name.replace('/', '_')}"
        self.cache = ActivationCache(self.cache_dir)
        # NEW: Raw activation cache (stores full sequences, shared between strategies in same family)
        self.raw_cache = RawActivationCache(self.cache_dir)
        # Cache for nonsense baseline activations per (n_pairs, layer)
        self._nonsense_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def _get_nonsense_cache_path(self, n_pairs: int, layer: int) -> Path:
        """Get disk cache path for nonsense baseline."""
        cache_dir = Path(self.cache_dir) / "nonsense_baseline"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_prefix = self.model.model_name.replace("/", "_")
        return cache_dir / f"{model_prefix}_n{n_pairs}_layer{layer}.pt"
    
    def get_nonsense_baseline(
        self,
        n_pairs: int,
        layer: int,
        device: str = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get or generate nonsense baseline activations.
        
        Caches results both in memory and on disk so we only generate once 
        per (n_pairs, layer) combination. This ensures fair comparison: 
        same number of pairs as the benchmark.
        
        Args:
            n_pairs: Number of nonsense pairs to generate (should match benchmark size)
            layer: Which layer to extract from
            device: Device to use (default: model's device)
            
        Returns:
            Tuple of (nonsense_pos, nonsense_neg) tensors
        """
        cache_key = (n_pairs, layer)
        
        # Check memory cache first
        if cache_key in self._nonsense_cache:
            return self._nonsense_cache[cache_key]
        
        # Check disk cache
        cache_path = self._get_nonsense_cache_path(n_pairs, layer)
        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
                nonsense_pos = cached["positive"]
                nonsense_neg = cached["negative"]
                self._nonsense_cache[cache_key] = (nonsense_pos, nonsense_neg)
                return nonsense_pos, nonsense_neg
            except Exception:
                # Corrupted cache, regenerate
                pass
        
        # Generate new nonsense activations
        device = device or str(self.model.hf_model.device)
        
        nonsense_pos, nonsense_neg = generate_nonsense_activations(
            model=self.model.hf_model,
            tokenizer=self.model.tokenizer,
            n_pairs=n_pairs,
            layer=layer,
            device=device,
        )
        
        # Save to memory cache
        self._nonsense_cache[cache_key] = (nonsense_pos, nonsense_neg)
        
        # Save to disk cache
        try:
            torch.save({
                "positive": nonsense_pos.cpu(),
                "negative": nonsense_neg.cpu(),
                "n_pairs": n_pairs,
                "layer": layer,
                "model": self.model.model_name,
            }, cache_path)
        except Exception:
            pass  # Disk cache is optional
        
        return nonsense_pos, nonsense_neg
    
    def clear_nonsense_cache(self, disk: bool = False) -> None:
        """
        Clear the nonsense baseline cache.
        
        Args:
            disk: If True, also clear disk cache
        """
        self._nonsense_cache.clear()
        
        if disk:
            cache_dir = Path(self.cache_dir) / "nonsense_baseline"
            if cache_dir.exists():
                import shutil
                shutil.rmtree(cache_dir)
    
    def run(
        self,
        benchmarks: Optional[List[str]] = None,
        strategies: Optional[List[ExtractionStrategy]] = None,
        max_layer_combo_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> GeometrySearchResults:
        """
        Run the geometry search.
        
        Args:
            benchmarks: Benchmarks to test (default: all from search space)
            strategies: Strategies to test (default: all from search space)
            max_layer_combo_size: Override max layer combo size
            show_progress: Print progress
            
        Returns:
            GeometrySearchResults with all test results
        """
        benchmarks = benchmarks or self.search_space.benchmarks
        strategies = strategies or self.search_space.strategies
        max_combo = max_layer_combo_size or self.search_space.config.max_layer_combo_size
        
        # Get layer combinations
        num_layers = self.model.num_layers
        layer_combos = get_layer_combinations(num_layers, max_combo)
        
        results = GeometrySearchResults(
            model_name=self.model.model_name,
            config=self.search_space.config,
        )
        
        start_time = time.time()
        extraction_time = 0.0
        test_time = 0.0
        
        total_extractions = len(benchmarks) * len(strategies)
        extraction_count = 0
        
        for benchmark in benchmarks:
            for strategy in strategies:
                extraction_count += 1
                
                if show_progress:
                    print(f"\n[{extraction_count}/{total_extractions}] {benchmark} / {strategy.value}")
                
                # Get or create cached activations
                extract_start = time.time()
                try:
                    cached = self._get_cached_activations(benchmark, strategy, show_progress)
                except Exception as e:
                    if show_progress:
                        print(f"  SKIP: {e}")
                    continue
                extraction_time += time.time() - extract_start
                
                # Test all layer combinations
                test_start = time.time()
                for combo in layer_combos:
                    result = compute_geometry_metrics(cached, combo)
                    results.add_result(result)
                test_time += time.time() - test_start
                
                results.benchmarks_tested = len(set(r.benchmark for r in results.results))
                results.strategies_tested = len(set(r.strategy for r in results.results))
                results.layer_combos_tested = len(results.results)
                
                if show_progress:
                    print(f"  Tested {len(layer_combos)} layer combos")
        
        results.total_time_seconds = time.time() - start_time
        results.extraction_time_seconds = extraction_time
        results.test_time_seconds = test_time
        
        return results
    
    def _get_cached_activations(
        self,
        benchmark: str,
        strategy: ExtractionStrategy,
        show_progress: bool = True,
    ) -> CachedActivations:
        """
        Get cached activations, extracting if necessary.
        
        Uses raw activation cache to share data between strategies in the same
        text family (e.g., chat_last, chat_mean, chat_first all share same forward pass).
        """
        # Check legacy cache first (for backward compatibility)
        if self.cache.has(self.model.model_name, benchmark, strategy):
            if show_progress:
                print(f"  Loading from cache...")
            return self.cache.get(self.model.model_name, benchmark, strategy)
        
        # Check raw cache for this text family
        text_family = get_strategy_text_family(strategy)
        if self.raw_cache.has(self.model.model_name, benchmark, text_family):
            if show_progress:
                print(f"  Loading from raw cache ({text_family} family)...")
            raw_cached = self.raw_cache.get(self.model.model_name, benchmark, text_family)
            # Convert to CachedActivations for requested strategy
            cached = raw_cached.to_cached_activations(strategy, self.model.tokenizer)
            # Save to legacy cache for faster future access
            self.cache.put(cached)
            return cached
        
        # Need to extract - load pairs first
        if show_progress:
            print(f"  Loading pairs...")
        
        pairs = self._load_pairs(benchmark)
        
        if show_progress:
            print(f"  Extracting raw activations for {len(pairs)} pairs ({text_family} family)...")
        
        # Collect RAW activations (full sequences) - shared for all strategies in family
        raw_cached = collect_and_cache_raw_activations(
            model=self.model,
            pairs=pairs,
            benchmark=benchmark,
            strategy=strategy,  # Determines text family
            cache=self.raw_cache,
            show_progress=show_progress,
        )
        
        # Convert to CachedActivations for requested strategy
        cached = raw_cached.to_cached_activations(strategy, self.model.tokenizer)
        # Save to legacy cache for faster future access
        self.cache.put(cached)
        
        return cached
    
    def _load_pairs(self, benchmark: str) -> List:
        """Load contrastive pairs for a benchmark."""
        from lm_eval.tasks import TaskManager
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
        
        tm = TaskManager()
        try:
            task_dict = tm.load_task_or_group([benchmark])
            task = list(task_dict.values())[0]
        except Exception:
            task = None
        
        # pairs_per_benchmark <= 0 means "use all available"
        limit = self.search_space.config.pairs_per_benchmark
        if limit <= 0:
            limit = None  # No limit
        
        pairs = lm_build_contrastive_pairs(
            benchmark, 
            task, 
            limit=limit
        )
        
        # Random sample if we have more pairs than needed (only if limit is set)
        if limit and len(pairs) > limit:
            random.seed(self.search_space.config.random_seed)
            pairs = random.sample(pairs, limit)
        
        return pairs


