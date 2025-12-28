"""
Geometry search runner.

Runs geometry tests across the search space using cached activations.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import torch

import numpy as np

from wisent.core.geometry_search_space import GeometrySearchSpace, GeometrySearchConfig
from wisent.core.activations.extraction_strategy import ExtractionStrategy
from wisent.core.activations.activation_cache import (
    ActivationCache,
    CachedActivations,
    collect_and_cache_activations,
)
from wisent.core.utils.layer_combinations import get_layer_combinations


def compute_signal_strength(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_folds: int = 5,
) -> float:
    """
    Compute signal strength using MLP cross-validation accuracy.
    
    This measures whether there is ANY extractable signal (linear or nonlinear)
    that generalizes to unseen data. Random/nonsense data gives ~0.5.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy (0.5 = no signal, >0.7 = signal exists)
    """
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < 5 or n_neg < 5:
            return 0.5  # Not enough data
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = MLPClassifier(
            hidden_layer_sizes=(16,),
            max_iter=500,
            random_state=42,
        )
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


def compute_knn_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
    n_folds: int = 5,
) -> float:
    """
    Compute k-NN cross-validation accuracy.
    
    Measures local separability without assuming linearity.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy
    """
    try:
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < k + 1 or n_neg < k + 1:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


def compute_mmd_rbf(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute Maximum Mean Discrepancy with RBF kernel.
    
    Measures distribution difference without assuming linearity.
    Higher values indicate more separable distributions.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        MMD value (0 = identical distributions)
    """
    try:
        from sklearn.metrics.pairwise import rbf_kernel
        from scipy.spatial.distance import cdist
        
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        # Use median heuristic for gamma
        all_data = np.vstack([pos, neg])
        dists = cdist(all_data, all_data, 'euclidean')
        gamma = 1.0 / (2 * np.median(dists[dists > 0]) ** 2 + 1e-10)
        
        K_pp = rbf_kernel(pos, pos, gamma=gamma)
        K_nn = rbf_kernel(neg, neg, gamma=gamma)
        K_pn = rbf_kernel(pos, neg, gamma=gamma)
        
        m = len(pos)
        n = len(neg)
        
        mmd = (K_pp.sum() / (m * m) + 
               K_nn.sum() / (n * n) - 
               2 * K_pn.sum() / (m * n))
        
        return float(max(0, mmd))
    except Exception:
        return 0.0


def estimate_local_intrinsic_dim(X: np.ndarray, k: int = 10) -> float:
    """
    Estimate local intrinsic dimensionality using MLE method.
    Based on Levina & Bickel (2004).
    
    Args:
        X: [N, D] data matrix
        k: Number of neighbors for estimation
        
    Returns:
        Estimated intrinsic dimension
    """
    from scipy.spatial.distance import cdist
    
    if len(X) < k + 1:
        return float(X.shape[1])
    
    dists = cdist(X, X, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    
    sorted_dists = np.sort(dists, axis=1)[:, :k]
    
    dims = []
    for i in range(len(X)):
        T_k = sorted_dists[i, k-1]
        if T_k < 1e-10:
            continue
        log_ratios = np.log(sorted_dists[i, :k-1] / T_k + 1e-10)
        if len(log_ratios) > 0 and log_ratios.sum() < 0:
            dim_est = -(k - 1) / log_ratios.sum()
            dims.append(min(dim_est, X.shape[1]))
    
    return float(np.median(dims)) if dims else float(X.shape[1])


def compute_local_intrinsic_dims(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    k: int = 10,
) -> tuple:
    """
    Compute local intrinsic dimension for pos and neg separately.
    
    Different local dimensions suggest different geometric structures.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        k: Number of neighbors
        
    Returns:
        (local_dim_pos, local_dim_neg, ratio)
    """
    try:
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        dim_pos = estimate_local_intrinsic_dim(pos, k)
        dim_neg = estimate_local_intrinsic_dim(neg, k)
        ratio = dim_pos / (dim_neg + 1e-10)
        
        return dim_pos, dim_neg, ratio
    except Exception:
        return 0.0, 0.0, 1.0


def compute_fisher_per_dimension(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> dict:
    """
    Compute Fisher ratio for each dimension and summary stats.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Dict with fisher_max, fisher_gini, fisher_top10_ratio, num_dims_above_1
    """
    try:
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        n_dims = pos.shape[1]
        fishers = np.zeros(n_dims)
        
        for d in range(n_dims):
            pos_d = pos[:, d]
            neg_d = neg[:, d]
            
            mean_pos = pos_d.mean()
            mean_neg = neg_d.mean()
            var_pos = pos_d.var()
            var_neg = neg_d.var()
            
            between_var = (mean_pos - mean_neg) ** 2
            within_var = (var_pos + var_neg) / 2
            
            if within_var > 1e-10:
                fishers[d] = between_var / within_var
        
        # Summary stats
        fisher_max = float(fishers.max())
        
        # Gini coefficient
        values = np.abs(fishers)
        if values.sum() > 1e-10:
            values = np.sort(values)
            n = len(values)
            fisher_gini = (2 * np.sum((np.arange(1, n+1) * values)) / (n * values.sum())) - (n + 1) / n
        else:
            fisher_gini = 0.0
        
        # Top 10 ratio
        sorted_fishers = np.sort(fishers)[::-1]
        top10_sum = sorted_fishers[:10].sum()
        total_sum = fishers.sum() + 1e-10
        fisher_top10_ratio = float(top10_sum / total_sum)
        
        num_dims_above_1 = int((fishers > 1.0).sum())
        
        return {
            "fisher_max": fisher_max,
            "fisher_gini": float(fisher_gini),
            "fisher_top10_ratio": fisher_top10_ratio,
            "num_dims_fisher_above_1": num_dims_above_1,
        }
    except Exception:
        return {
            "fisher_max": 0.0,
            "fisher_gini": 0.0,
            "fisher_top10_ratio": 0.0,
            "num_dims_fisher_above_1": 0,
        }


def compute_density_ratio(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
) -> float:
    """
    Compute ratio of average intra-class distances.
    
    Values far from 1 suggest different local geometries.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        
    Returns:
        Density ratio (pos avg dist / neg avg dist)
    """
    try:
        from scipy.spatial.distance import cdist
        
        pos = pos_activations.float().cpu().numpy()
        neg = neg_activations.float().cpu().numpy()
        
        if len(pos) < 2 or len(neg) < 2:
            return 1.0
        
        pos_dists = cdist(pos, pos, 'euclidean')
        neg_dists = cdist(neg, neg, 'euclidean')
        
        np.fill_diagonal(pos_dists, np.nan)
        np.fill_diagonal(neg_dists, np.nan)
        
        avg_pos = np.nanmean(pos_dists)
        avg_neg = np.nanmean(neg_dists)
        
        if avg_neg < 1e-10:
            return 1.0
        
        return float(avg_pos / avg_neg)
    except Exception:
        return 1.0


def compute_linear_probe_accuracy(
    pos_activations: torch.Tensor,
    neg_activations: torch.Tensor,
    n_folds: int = 5,
) -> float:
    """
    Compute linear probe cross-validation accuracy.
    
    If signal_strength is high but linear_probe is low, the signal is nonlinear.
    If both are high, signal is linear and CAA should work.
    
    Args:
        pos_activations: [N, hidden_dim] positive class activations
        neg_activations: [N, hidden_dim] negative class activations
        n_folds: Number of CV folds
        
    Returns:
        Cross-validation accuracy (0.5 = no linear signal)
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        
        n_pos = len(pos_activations)
        n_neg = len(neg_activations)
        
        if n_pos < 5 or n_neg < 5:
            return 0.5
        
        X = torch.cat([pos_activations, neg_activations], dim=0).float().cpu().numpy()
        y = np.array([1] * n_pos + [0] * n_neg)
        
        n_folds = min(n_folds, min(n_pos, n_neg))
        if n_folds < 2:
            return 0.5
        
        clf = LogisticRegression(max_iter=1000, solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=n_folds, scoring='accuracy')
        return float(scores.mean())
    except Exception:
        return 0.5


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
    
    # Recommendation
    recommended_method: str
    
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


def compute_geometry_metrics(
    cached: CachedActivations,
    layers: List[int],
) -> GeometryTestResult:
    """
    Compute geometry metrics for a layer combination from cached activations.
    
    Uses the comprehensive detect_geometry_structure() to get scores for:
    - linear, cone, cluster, manifold, sparse, bimodal, orthogonal
    
    Args:
        cached: Cached activations with all layers
        layers: Layer indices (0-based) to analyze
        
    Returns:
        GeometryTestResult with all structure scores
    """
    from wisent.core.contrastive_pairs.diagnostics.control_vectors import (
        detect_geometry_structure,
        GeometryAnalysisConfig,
    )
    
    # Stack positive and negative activations for specified layers
    # Convert 0-based indices to 1-based layer names used in cache
    pos_acts_list = []
    neg_acts_list = []
    
    for layer_idx in layers:
        layer_name = str(layer_idx + 1)  # Convert 0-based to 1-based
        try:
            pos = cached.get_positive_activations(layer_name)  # [num_pairs, hidden_size]
            neg = cached.get_negative_activations(layer_name)  # [num_pairs, hidden_size]
            pos_acts_list.append(pos)
            neg_acts_list.append(neg)
        except (KeyError, IndexError):
            continue
    
    if not pos_acts_list:
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            signal_strength=0.5,
            has_signal=False,
            linear_probe_accuracy=0.5,
            is_linear=False,
            # Nonlinear metrics
            knn_accuracy_k5=0.5,
            knn_accuracy_k10=0.5,
            knn_accuracy_k20=0.5,
            mmd_rbf=0.0,
            local_dim_pos=0.0,
            local_dim_neg=0.0,
            local_dim_ratio=1.0,
            fisher_max=0.0,
            fisher_gini=0.0,
            fisher_top10_ratio=0.0,
            num_dims_fisher_above_1=0,
            density_ratio=1.0,
            # Structure scores
            best_structure="error",
            best_score=0.0,
            linear_score=0.0,
            cone_score=0.0,
            orthogonal_score=0.0,
            manifold_score=0.0,
            sparse_score=0.0,
            cluster_score=0.0,
            bimodal_score=0.0,
            cohens_d=0.0,
            variance_explained=0.0,
            within_class_consistency=0.0,
            raw_mean_cosine_similarity=0.0,
            positive_correlation_fraction=0.0,
            near_zero_fraction=0.0,
            pca_top2_variance=0.0,
            local_nonlinearity=0.0,
            gini_coefficient=0.0,
            active_fraction=0.0,
            top_10_contribution=0.0,
            best_silhouette=0.0,
            best_k=0,
            recommended_method="error: no activations",
        )
    
    # Concatenate across layers: [num_pairs, hidden_size * num_layers]
    pos_activations = torch.cat(pos_acts_list, dim=-1)
    neg_activations = torch.cat(neg_acts_list, dim=-1)
    
    # Convert to float32 for geometry analysis (bf16/float16 can cause dtype mismatches)
    pos_activations = pos_activations.float()
    neg_activations = neg_activations.float()
    
    # Run comprehensive geometry detection
    config = GeometryAnalysisConfig(
        num_components=5,
        optimization_steps=50,  # Reduced for speed since we're testing many combos
    )
    
    try:
        result = detect_geometry_structure(pos_activations, neg_activations, config)
        
        # Step 1: Compute signal strength (MLP CV accuracy)
        signal_strength = compute_signal_strength(pos_activations, neg_activations)
        has_signal = signal_strength > 0.6
        
        # Step 2: Compute linear probe accuracy
        linear_probe_accuracy = compute_linear_probe_accuracy(pos_activations, neg_activations)
        # Signal is linear if: has signal AND linear probe is close to MLP (within 0.1)
        is_linear = has_signal and linear_probe_accuracy > 0.6 and (signal_strength - linear_probe_accuracy) < 0.15
        
        # Step 2b: Compute nonlinear signal metrics
        knn_k5 = compute_knn_accuracy(pos_activations, neg_activations, k=5)
        knn_k10 = compute_knn_accuracy(pos_activations, neg_activations, k=10)
        knn_k20 = compute_knn_accuracy(pos_activations, neg_activations, k=20)
        mmd = compute_mmd_rbf(pos_activations, neg_activations)
        local_dim_pos, local_dim_neg, local_dim_ratio = compute_local_intrinsic_dims(pos_activations, neg_activations)
        fisher_stats = compute_fisher_per_dimension(pos_activations, neg_activations)
        density_rat = compute_density_ratio(pos_activations, neg_activations)
        
        # Determine recommendation based on signal analysis
        if not has_signal:
            recommendation = "NO_SIGNAL"
        elif is_linear:
            recommendation = "CAA"  # Linear signal -> use Contrastive Activation Addition
        else:
            recommendation = "NONLINEAR"  # Nonlinear signal -> need different method
        
        # Helper to safely get detail
        def get_detail(struct_name: str, key: str, default=0.0):
            if struct_name in result.all_scores:
                return result.all_scores[struct_name].details.get(key, default)
            return default
        
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            signal_strength=signal_strength,
            has_signal=has_signal,
            linear_probe_accuracy=linear_probe_accuracy,
            is_linear=is_linear,
            # Nonlinear metrics
            knn_accuracy_k5=knn_k5,
            knn_accuracy_k10=knn_k10,
            knn_accuracy_k20=knn_k20,
            mmd_rbf=mmd,
            local_dim_pos=local_dim_pos,
            local_dim_neg=local_dim_neg,
            local_dim_ratio=local_dim_ratio,
            fisher_max=fisher_stats["fisher_max"],
            fisher_gini=fisher_stats["fisher_gini"],
            fisher_top10_ratio=fisher_stats["fisher_top10_ratio"],
            num_dims_fisher_above_1=fisher_stats["num_dims_fisher_above_1"],
            density_ratio=density_rat,
            # Structure scores
            best_structure=result.best_structure.value,
            best_score=result.best_score,
            linear_score=result.all_scores.get("linear", type('', (), {'score': 0.0})()).score,
            cone_score=result.all_scores.get("cone", type('', (), {'score': 0.0})()).score,
            orthogonal_score=result.all_scores.get("orthogonal", type('', (), {'score': 0.0})()).score,
            manifold_score=result.all_scores.get("manifold", type('', (), {'score': 0.0})()).score,
            sparse_score=result.all_scores.get("sparse", type('', (), {'score': 0.0})()).score,
            cluster_score=result.all_scores.get("cluster", type('', (), {'score': 0.0})()).score,
            bimodal_score=result.all_scores.get("bimodal", type('', (), {'score': 0.0})()).score,
            # Linear details
            cohens_d=get_detail("linear", "cohens_d", 0.0),
            variance_explained=get_detail("linear", "variance_explained", 0.0),
            within_class_consistency=get_detail("linear", "within_class_consistency", 0.0),
            # Cone details
            raw_mean_cosine_similarity=get_detail("cone", "raw_mean_cosine_similarity", 0.0),
            positive_correlation_fraction=get_detail("cone", "positive_correlation_fraction", 0.0),
            # Orthogonal details
            near_zero_fraction=get_detail("orthogonal", "near_zero_fraction", 0.0),
            # Manifold details
            pca_top2_variance=get_detail("manifold", "pca_top2_variance", 0.0),
            local_nonlinearity=get_detail("manifold", "local_nonlinearity", 0.0),
            # Sparse details
            gini_coefficient=get_detail("sparse", "gini_coefficient", 0.0),
            active_fraction=get_detail("sparse", "active_fraction", 0.0),
            top_10_contribution=get_detail("sparse", "top_10_contribution", 0.0),
            # Cluster details
            best_silhouette=get_detail("cluster", "best_silhouette", 0.0),
            best_k=int(get_detail("cluster", "best_k", 2)),
            # Recommendation based on signal analysis
            recommended_method=recommendation,
        )
    except Exception as e:
        return GeometryTestResult(
            benchmark=cached.benchmark,
            strategy=cached.strategy.value,
            layers=layers,
            signal_strength=0.5,
            has_signal=False,
            linear_probe_accuracy=0.5,
            is_linear=False,
            # Nonlinear metrics
            knn_accuracy_k5=0.5,
            knn_accuracy_k10=0.5,
            knn_accuracy_k20=0.5,
            mmd_rbf=0.0,
            local_dim_pos=0.0,
            local_dim_neg=0.0,
            local_dim_ratio=1.0,
            fisher_max=0.0,
            fisher_gini=0.0,
            fisher_top10_ratio=0.0,
            num_dims_fisher_above_1=0,
            density_ratio=1.0,
            # Structure scores
            best_structure="error",
            best_score=0.0,
            linear_score=0.0,
            cone_score=0.0,
            orthogonal_score=0.0,
            manifold_score=0.0,
            sparse_score=0.0,
            cluster_score=0.0,
            bimodal_score=0.0,
            cohens_d=0.0,
            variance_explained=0.0,
            within_class_consistency=0.0,
            raw_mean_cosine_similarity=0.0,
            positive_correlation_fraction=0.0,
            near_zero_fraction=0.0,
            pca_top2_variance=0.0,
            local_nonlinearity=0.0,
            gini_coefficient=0.0,
            active_fraction=0.0,
            top_10_contribution=0.0,
            best_silhouette=0.0,
            best_k=0,
            recommended_method=f"error: {str(e)}",
        )


class GeometryRunner:
    """
    Runs geometry search across the search space.
    
    Uses activation caching for efficiency:
    1. Extract ALL layers once per (benchmark, strategy)
    2. Test all layer combinations from cache
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
        """Get cached activations, extracting if necessary."""
        # Check cache
        if self.cache.has(self.model.model_name, benchmark, strategy):
            if show_progress:
                print(f"  Loading from cache...")
            return self.cache.get(self.model.model_name, benchmark, strategy)
        
        # Need to extract - load pairs first
        if show_progress:
            print(f"  Loading pairs...")
        
        pairs = self._load_pairs(benchmark)
        
        if show_progress:
            print(f"  Extracting activations for {len(pairs)} pairs...")
        
        return collect_and_cache_activations(
            model=self.model,
            pairs=pairs,
            benchmark=benchmark,
            strategy=strategy,
            cache=self.cache,
            show_progress=show_progress,
        )
    
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
        
        pairs = lm_build_contrastive_pairs(
            benchmark, 
            task, 
            limit=self.search_space.config.pairs_per_benchmark
        )
        
        # Random sample if we have more pairs than needed
        if len(pairs) > self.search_space.config.pairs_per_benchmark:
            random.seed(self.search_space.config.random_seed)
            pairs = random.sample(pairs, self.search_space.config.pairs_per_benchmark)
        
        return pairs


# Type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from wisent.core.models.wisent_model import WisentModel
