#!/usr/bin/env python3
"""
Nonlinear signal analysis for contrastive pairs.

Evaluates representation quality without assuming linearity:
1. Fisher per-dimension distribution
2. kNN accuracy (local separability)
3. Kernel MMD (distribution difference)
4. Local intrinsic dimension
5. Cross-strategy stability
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics.pairwise import rbf_kernel

from wisent.core.models.wisent_model import WisentModel
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent.core.activations import ExtractionStrategy
from wisent.core.activations.activation_cache import collect_and_cache_activations
from lm_eval.tasks import TaskManager


@dataclass
class NonlinearSignalResult:
    """Result of nonlinear signal analysis for one layer."""
    layer: int
    # Fisher distribution metrics
    fisher_mean: float
    fisher_max: float
    fisher_gini: float  # concentration of signal
    fisher_top10_ratio: float  # fraction of signal in top 10 dims
    num_dims_fisher_above_1: int
    # kNN metrics
    knn_accuracy_k5: float
    knn_accuracy_k10: float
    knn_accuracy_k20: float
    # MMD metrics
    mmd_rbf: float
    mmd_linear: float
    # Local geometry
    local_dim_pos: float  # intrinsic dim of positive class
    local_dim_neg: float  # intrinsic dim of negative class
    local_dim_ratio: float  # ratio of dims
    # Cluster metrics
    silhouette_score: float
    density_ratio: float  # ratio of avg distances


@dataclass 
class BenchmarkNonlinearAnalysis:
    """Full nonlinear analysis for a benchmark."""
    benchmark: str
    model: str
    strategy: str
    num_pairs: int
    per_layer_results: List[NonlinearSignalResult]
    best_layer_knn: int
    best_knn_accuracy: float
    best_layer_mmd: int
    best_mmd: float


def compute_fisher_per_dimension(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Compute Fisher ratio for each dimension independently."""
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
    
    return fishers


def compute_gini(values: np.ndarray) -> float:
    """Compute Gini coefficient - measures concentration."""
    values = np.abs(values)
    if values.sum() < 1e-10:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(values)
    return (2 * np.sum((np.arange(1, n+1) * values)) / (n * values.sum())) - (n + 1) / n


def compute_knn_accuracy(pos: np.ndarray, neg: np.ndarray, k: int = 5) -> float:
    """Compute k-NN cross-validation accuracy."""
    X = np.vstack([pos, neg])
    y = np.array([1] * len(pos) + [0] * len(neg))
    
    if len(X) < k + 1:
        return 0.5
    
    knn = KNeighborsClassifier(n_neighbors=k)
    try:
        scores = cross_val_score(knn, X, y, cv=min(5, len(X) // 2))
        return float(scores.mean())
    except:
        return 0.5


def compute_mmd_rbf(pos: np.ndarray, neg: np.ndarray, gamma: float = None) -> float:
    """Compute Maximum Mean Discrepancy with RBF kernel."""
    if gamma is None:
        # Use median heuristic
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


def compute_mmd_linear(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute MMD with linear kernel (just mean difference)."""
    mean_diff = pos.mean(axis=0) - neg.mean(axis=0)
    return float(np.linalg.norm(mean_diff))


def estimate_local_intrinsic_dim(X: np.ndarray, k: int = 10) -> float:
    """
    Estimate local intrinsic dimensionality using MLE method.
    Based on Levina & Bickel (2004).
    """
    if len(X) < k + 1:
        return float(X.shape[1])
    
    # Compute k-NN distances
    dists = cdist(X, X, 'euclidean')
    np.fill_diagonal(dists, np.inf)
    
    # Sort to get k nearest
    sorted_dists = np.sort(dists, axis=1)[:, :k]
    
    # MLE estimator
    dims = []
    for i in range(len(X)):
        T_k = sorted_dists[i, k-1]
        if T_k < 1e-10:
            continue
        log_ratios = np.log(sorted_dists[i, :k-1] / T_k + 1e-10)
        if len(log_ratios) > 0:
            dim_est = -(k - 1) / log_ratios.sum() if log_ratios.sum() < 0 else X.shape[1]
            dims.append(min(dim_est, X.shape[1]))
    
    return float(np.median(dims)) if dims else float(X.shape[1])


def compute_silhouette(pos: np.ndarray, neg: np.ndarray) -> float:
    """Compute silhouette score for pos/neg clustering."""
    from sklearn.metrics import silhouette_score
    
    X = np.vstack([pos, neg])
    labels = np.array([1] * len(pos) + [0] * len(neg))
    
    if len(np.unique(labels)) < 2:
        return 0.0
    
    try:
        return float(silhouette_score(X, labels))
    except:
        return 0.0


def compute_density_ratio(pos: np.ndarray, neg: np.ndarray) -> float:
    """
    Compute ratio of average intra-class distances.
    Values far from 1 suggest different local geometries.
    """
    if len(pos) < 2 or len(neg) < 2:
        return 1.0
    
    pos_dists = cdist(pos, pos, 'euclidean')
    neg_dists = cdist(neg, neg, 'euclidean')
    
    # Average distance (excluding diagonal)
    np.fill_diagonal(pos_dists, np.nan)
    np.fill_diagonal(neg_dists, np.nan)
    
    avg_pos = np.nanmean(pos_dists)
    avg_neg = np.nanmean(neg_dists)
    
    if avg_neg < 1e-10:
        return 1.0
    
    return float(avg_pos / avg_neg)


def analyze_layer(
    pos: np.ndarray,
    neg: np.ndarray,
    layer: int
) -> NonlinearSignalResult:
    """Analyze signal quality for one layer."""
    
    # Fisher per dimension
    fishers = compute_fisher_per_dimension(pos, neg)
    fisher_mean = float(fishers.mean())
    fisher_max = float(fishers.max())
    fisher_gini = compute_gini(fishers)
    
    # Top 10 dims ratio
    sorted_fishers = np.sort(fishers)[::-1]
    top10_sum = sorted_fishers[:10].sum()
    total_sum = fishers.sum() + 1e-10
    fisher_top10_ratio = float(top10_sum / total_sum)
    
    num_dims_above_1 = int((fishers > 1.0).sum())
    
    # kNN accuracies
    knn_5 = compute_knn_accuracy(pos, neg, k=5)
    knn_10 = compute_knn_accuracy(pos, neg, k=10)
    knn_20 = compute_knn_accuracy(pos, neg, k=20)
    
    # MMD
    mmd_rbf = compute_mmd_rbf(pos, neg)
    mmd_linear = compute_mmd_linear(pos, neg)
    
    # Local intrinsic dimension
    local_dim_pos = estimate_local_intrinsic_dim(pos)
    local_dim_neg = estimate_local_intrinsic_dim(neg)
    local_dim_ratio = local_dim_pos / (local_dim_neg + 1e-10)
    
    # Clustering metrics
    silhouette = compute_silhouette(pos, neg)
    density_ratio = compute_density_ratio(pos, neg)
    
    return NonlinearSignalResult(
        layer=layer,
        fisher_mean=fisher_mean,
        fisher_max=fisher_max,
        fisher_gini=fisher_gini,
        fisher_top10_ratio=fisher_top10_ratio,
        num_dims_fisher_above_1=num_dims_above_1,
        knn_accuracy_k5=knn_5,
        knn_accuracy_k10=knn_10,
        knn_accuracy_k20=knn_20,
        mmd_rbf=mmd_rbf,
        mmd_linear=mmd_linear,
        local_dim_pos=local_dim_pos,
        local_dim_neg=local_dim_neg,
        local_dim_ratio=local_dim_ratio,
        silhouette_score=silhouette,
        density_ratio=density_ratio,
    )


def analyze_benchmark(
    model: WisentModel,
    benchmark: str,
    strategy: ExtractionStrategy,
    limit: int = 100,
    cache_dir: str = "/tmp/wisent_nonlinear_cache",
) -> Optional[BenchmarkNonlinearAnalysis]:
    """Run full nonlinear analysis on a benchmark."""
    
    # Load task from lm-eval
    try:
        tm = TaskManager()
        task_dict = tm.load_task_or_group([benchmark])
        task = list(task_dict.values())[0]
    except:
        task = None
    
    # Load pairs
    pairs = lm_build_contrastive_pairs(benchmark, task, limit=limit)
    if not pairs:
        print(f"No pairs for {benchmark}")
        return None
    
    print(f"Loaded {len(pairs)} pairs for {benchmark}")
    
    # Collect activations
    cached = collect_and_cache_activations(
        model=model,
        pairs=pairs,
        benchmark=benchmark,
        strategy=strategy,
        cache_dir=cache_dir,
        show_progress=True,
    )
    
    # Analyze each layer
    results = []
    n_layers = model.num_layers
    
    for layer in range(n_layers):
        layer_name = str(layer + 1)  # 1-based in cache
        try:
            pos_vectors = cached.get_positive_activations(layer_name)
            neg_vectors = cached.get_negative_activations(layer_name)
        except KeyError:
            continue
        
        pos_np = pos_vectors.float().cpu().numpy()
        neg_np = neg_vectors.float().cpu().numpy()
        
        # Skip if NaN present
        if np.isnan(pos_np).any() or np.isnan(neg_np).any():
            print(f"  Layer {layer}: skipping due to NaN")
            continue
        
        result = analyze_layer(pos_np, neg_np, layer)
        results.append(result)
    
    if not results:
        return None
    
    # Find best layers
    best_knn_idx = np.argmax([r.knn_accuracy_k10 for r in results])
    best_mmd_idx = np.argmax([r.mmd_rbf for r in results])
    
    return BenchmarkNonlinearAnalysis(
        benchmark=benchmark,
        model=model.model_name,
        strategy=strategy.value,
        num_pairs=len(pairs),
        per_layer_results=results,
        best_layer_knn=results[best_knn_idx].layer,
        best_knn_accuracy=results[best_knn_idx].knn_accuracy_k10,
        best_layer_mmd=results[best_mmd_idx].layer,
        best_mmd=results[best_mmd_idx].mmd_rbf,
    )


def print_results(analysis: BenchmarkNonlinearAnalysis):
    """Print analysis results."""
    print(f"\n--- Results for {analysis.benchmark} / {analysis.strategy} ---")
    print(f"Pairs: {analysis.num_pairs}")
    print(f"Best layer (kNN): {analysis.best_layer_knn}, accuracy: {analysis.best_knn_accuracy:.3f}")
    print(f"Best layer (MMD): {analysis.best_layer_mmd}, MMD: {analysis.best_mmd:.4f}")
    
    print(f"\n{'Layer':<6} {'kNN-10':<8} {'MMD-RBF':<10} {'Fisher-max':<12} {'Fisher-Gini':<12} {'Dims>1':<8} {'Silhouette':<12} {'LocalDim-P':<12} {'LocalDim-N':<12}")
    print("-" * 100)
    
    for r in analysis.per_layer_results:
        print(f"{r.layer:<6} {r.knn_accuracy_k10:<8.3f} {r.mmd_rbf:<10.4f} {r.fisher_max:<12.2f} {r.fisher_gini:<12.3f} {r.num_dims_fisher_above_1:<8} {r.silhouette_score:<12.3f} {r.local_dim_pos:<12.1f} {r.local_dim_neg:<12.1f}")


def main():
    parser = argparse.ArgumentParser(description="Nonlinear signal analysis")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--benchmarks", nargs="+", default=["halulens", "truthfulqa_generation"])
    parser.add_argument("--strategies", nargs="+", default=["chat_last", "chat_mean"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--cache-dir", default="/tmp/wisent_nonlinear_cache")
    parser.add_argument("--output-dir", default="/tmp/nonlinear_results")
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model = WisentModel(args.model)
    print(f"Model loaded. Layers: {model.num_layers}")
    
    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for benchmark in args.benchmarks:
        for strategy_name in args.strategies:
            print(f"\n{'='*60}")
            print(f"Analyzing: {benchmark} / {strategy_name}")
            print(f"{'='*60}")
            
            try:
                strategy = ExtractionStrategy(strategy_name)
                analysis = analyze_benchmark(
                    model=model,
                    benchmark=benchmark,
                    strategy=strategy,
                    limit=args.limit,
                    cache_dir=args.cache_dir,
                )
                
                if analysis:
                    print_results(analysis)
                    all_results.append(asdict(analysis))
                    
            except Exception as e:
                print(f"Error analyzing {benchmark}/{strategy_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    output_file = Path(args.output_dir) / f"nonlinear_{args.model.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"{'Benchmark':<25} {'Strategy':<15} {'BestLayer':<10} {'kNN-10':<10} {'MMD':<10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['benchmark']:<25} {r['strategy']:<15} {r['best_layer_knn']:<10} {r['best_knn_accuracy']:<10.3f} {r['best_mmd']:<10.4f}")


if __name__ == "__main__":
    main()
