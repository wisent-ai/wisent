#!/usr/bin/env python3
"""
Effective rank analysis for contrastive pairs.

Computes PCA effective rank and mean cosine similarity to evaluate signal quality.
Can also find linear subsets within benchmarks.
"""

import argparse
import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from wisent.core.models.wisent_model import WisentModel
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent.core.activations import ExtractionStrategy
from wisent.core.activations.activation_cache import collect_and_cache_activations, ActivationCache
from lm_eval.tasks import TaskManager


@dataclass
class EffectiveRankResult:
    """Result of effective rank analysis for one layer."""
    layer: int
    rank_50: int  # dims for 50% variance
    rank_80: int  # dims for 80% variance
    rank_90: int  # dims for 90% variance
    participation_ratio: float  # effective dimensionality
    mean_cosine: float  # mean pairwise cosine similarity
    pc1_variance: float  # variance explained by PC1
    pc1_3_variance: float  # variance explained by PC1-3
    # Fisher ratio metrics - how much each PC carries label signal
    fisher_ratio_pc1: float  # Fisher ratio for PC1
    fisher_ratio_pc2: float  # Fisher ratio for PC2
    fisher_ratio_pc3: float  # Fisher ratio for PC3
    top_fisher_pc: int  # which PC has highest Fisher ratio
    max_fisher_ratio: float  # maximum Fisher ratio across PCs
    cumulative_fisher_top3: float  # sum of Fisher ratios for top 3 PCs


@dataclass
class LinearSubset:
    """A subset of pairs with potentially linear signal."""
    pair_indices: List[int]
    rank_80: int
    mean_cosine: float
    pc1_variance: float


@dataclass
class BenchmarkAnalysis:
    """Full analysis for a benchmark."""
    benchmark: str
    model: str
    strategy: str
    num_pairs: int
    per_layer_results: List[EffectiveRankResult]
    best_layer: int
    best_fisher_ratio: float  # max Fisher ratio across layers
    linear_subsets: Optional[List[LinearSubset]] = None


def compute_fisher_ratio(projections: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Fisher ratio for a 1D projection.
    
    Fisher ratio = (between-class variance) / (within-class variance)
    Higher = better separation by this component.
    """
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    pos_proj = projections[pos_mask]
    neg_proj = projections[neg_mask]
    
    if len(pos_proj) < 2 or len(neg_proj) < 2:
        return 0.0
    
    # Between-class variance
    mean_pos = pos_proj.mean()
    mean_neg = neg_proj.mean()
    between_var = (mean_pos - mean_neg) ** 2
    
    # Within-class variance (pooled)
    var_pos = pos_proj.var()
    var_neg = neg_proj.var()
    within_var = (var_pos + var_neg) / 2
    
    if within_var < 1e-10:
        return 0.0
    
    return float(between_var / within_var)


def compute_effective_rank(
    pos_np: np.ndarray, 
    neg_np: np.ndarray,
) -> EffectiveRankResult:
    """
    Compute effective rank metrics for activation data.
    
    Args:
        pos_np: positive class activations [N_pos, hidden_dim]
        neg_np: negative class activations [N_neg, hidden_dim]
    """
    # Combine data
    all_data = np.concatenate([pos_np, neg_np], axis=0)
    labels = np.array([1] * len(pos_np) + [0] * len(neg_np))
    
    # Difference vectors (for backward compatibility with mean cosine)
    n_pairs = min(len(pos_np), len(neg_np))
    diff_np = pos_np[:n_pairs] - neg_np[:n_pairs]
    
    # Center all data
    all_centered = all_data - all_data.mean(axis=0)
    
    # SVD on all data
    U, S, Vh = np.linalg.svd(all_centered, full_matrices=False)
    
    # Variance explained
    var_explained = (S ** 2) / (S ** 2).sum()
    cumvar = np.cumsum(var_explained)
    
    # Effective rank metrics
    rank_50 = int(np.searchsorted(cumvar, 0.5) + 1)
    rank_80 = int(np.searchsorted(cumvar, 0.8) + 1)
    rank_90 = int(np.searchsorted(cumvar, 0.9) + 1)
    
    # Participation ratio
    participation_ratio = float((S ** 2).sum() ** 2 / (S ** 4).sum())
    
    # Mean cosine similarity on difference vectors
    norms = np.linalg.norm(diff_np, axis=1, keepdims=True)
    diff_norm = diff_np / (norms + 1e-8)
    cos_sim_matrix = diff_norm @ diff_norm.T
    n = cos_sim_matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    mean_cosine = float(cos_sim_matrix[mask].mean())
    
    # Project all data onto top PCs
    projections = all_centered @ Vh.T  # [N, num_components]
    
    # Compute Fisher ratio for each PC
    num_pcs = min(10, projections.shape[1])
    fisher_ratios = []
    for pc_idx in range(num_pcs):
        fr = compute_fisher_ratio(projections[:, pc_idx], labels)
        fisher_ratios.append(fr)
    
    # Find top Fisher PC
    top_fisher_pc = int(np.argmax(fisher_ratios))
    max_fisher_ratio = float(max(fisher_ratios))
    
    # Cumulative Fisher for top 3
    sorted_fisher = sorted(fisher_ratios, reverse=True)
    cumulative_fisher_top3 = float(sum(sorted_fisher[:3]))
    
    return EffectiveRankResult(
        layer=-1,  # Will be set by caller
        rank_50=rank_50,
        rank_80=rank_80,
        rank_90=rank_90,
        participation_ratio=participation_ratio,
        mean_cosine=mean_cosine,
        pc1_variance=float(var_explained[0]),
        pc1_3_variance=float(cumvar[min(2, len(cumvar)-1)]),
        fisher_ratio_pc1=float(fisher_ratios[0]) if len(fisher_ratios) > 0 else 0.0,
        fisher_ratio_pc2=float(fisher_ratios[1]) if len(fisher_ratios) > 1 else 0.0,
        fisher_ratio_pc3=float(fisher_ratios[2]) if len(fisher_ratios) > 2 else 0.0,
        top_fisher_pc=top_fisher_pc,
        max_fisher_ratio=max_fisher_ratio,
        cumulative_fisher_top3=cumulative_fisher_top3,
    )


def find_linear_subsets(
    diff_np: np.ndarray,
    threshold: float = 0.15,
    min_subset_size: int = 5,
) -> List[LinearSubset]:
    """Find subsets of pairs with higher similarity (potential linear signal)."""
    norms = np.linalg.norm(diff_np, axis=1, keepdims=True)
    diff_norm = diff_np / (norms + 1e-8)
    
    cos_sim_full = diff_norm @ diff_norm.T
    np.fill_diagonal(cos_sim_full, 0)
    
    # Greedy clustering by similarity
    mean_sim_per_pair = cos_sim_full.mean(axis=1)
    
    linear_subsets = []
    remaining = set(range(len(diff_np)))
    
    while len(remaining) >= min_subset_size:
        # Start new subset with most "central" remaining pair
        subset = []
        current_remaining = set(remaining)
        
        # Start with highest avg similarity pair
        best_start = max(current_remaining, key=lambda x: mean_sim_per_pair[x])
        subset.append(best_start)
        current_remaining.remove(best_start)
        
        # Greedily add pairs with high similarity to current subset
        while current_remaining:
            best = None
            best_sim = -1
            for candidate in current_remaining:
                avg_sim = np.mean([cos_sim_full[candidate, p] for p in subset])
                if avg_sim > best_sim:
                    best_sim = avg_sim
                    best = candidate
            
            if best_sim < threshold:
                break
            
            subset.append(best)
            current_remaining.remove(best)
        
        if len(subset) >= min_subset_size:
            # Compute metrics for this subset (use diff vectors as pseudo pos/neg)
            subset_diffs = diff_np[subset]
            # Split diffs in half to create pseudo pos/neg for Fisher computation
            half = len(subset_diffs) // 2
            if half >= 2:
                result = compute_effective_rank(subset_diffs[:half], subset_diffs[half:])
            else:
                # Too few samples, skip Fisher
                result = compute_effective_rank(subset_diffs, subset_diffs)
            
            linear_subsets.append(LinearSubset(
                pair_indices=subset,
                rank_80=result.rank_80,
                mean_cosine=result.mean_cosine,
                pc1_variance=result.pc1_variance,
            ))
            
            # Remove these pairs from remaining
            remaining -= set(subset)
        else:
            # No more valid subsets, stop
            break
    
    return linear_subsets


def analyze_benchmark(
    model: WisentModel,
    benchmark: str,
    strategy: ExtractionStrategy,
    cache: ActivationCache,
    limit: Optional[int] = None,
    find_subsets: bool = True,
) -> BenchmarkAnalysis:
    """Analyze effective rank for a benchmark across all layers."""
    
    # Load pairs
    tm = TaskManager()
    try:
        task_dict = tm.load_task_or_group([benchmark])
        task = list(task_dict.values())[0]
    except:
        task = None
    
    pairs = lm_build_contrastive_pairs(benchmark, task, limit=limit)
    print(f"Loaded {len(pairs)} pairs for {benchmark}")
    
    if len(pairs) < 10:
        raise ValueError(f"Not enough pairs for {benchmark}: {len(pairs)}")
    
    # Collect activations
    cached = collect_and_cache_activations(
        model=model,
        pairs=pairs,
        benchmark=benchmark if limit is None else f"{benchmark}_limit{limit}",
        strategy=strategy,
        cache=cache,
        show_progress=True,
    )
    
    # Analyze each layer
    per_layer_results = []
    best_layer = 0
    best_fisher = -1
    
    for layer_idx in range(cached.num_layers):
        layer_name = str(layer_idx + 1)  # 1-based in cache
        
        try:
            pos_vectors = cached.get_positive_activations(layer_name)
            neg_vectors = cached.get_negative_activations(layer_name)
        except KeyError:
            continue
        
        pos_np = pos_vectors.float().cpu().numpy()
        neg_np = neg_vectors.float().cpu().numpy()
        
        result = compute_effective_rank(pos_np, neg_np)
        result.layer = layer_idx
        
        per_layer_results.append(result)
        
        # Use max Fisher ratio as best criterion (not mean cosine)
        if result.max_fisher_ratio > best_fisher:
            best_fisher = result.max_fisher_ratio
            best_layer = layer_idx
    
    # Find linear subsets on best layer
    linear_subsets = None
    if find_subsets:
        layer_name = str(best_layer + 1)
        pos_vectors = cached.get_positive_activations(layer_name)
        neg_vectors = cached.get_negative_activations(layer_name)
        n_pairs = min(len(pos_vectors), len(neg_vectors))
        diff_np = (pos_vectors[:n_pairs] - neg_vectors[:n_pairs]).float().cpu().numpy()
        linear_subsets = find_linear_subsets(diff_np)
    
    return BenchmarkAnalysis(
        benchmark=benchmark,
        model=model.model_name,
        strategy=strategy.value,
        num_pairs=cached.num_pairs,
        per_layer_results=per_layer_results,
        best_layer=best_layer,
        best_fisher_ratio=best_fisher,
        linear_subsets=linear_subsets,
    )


def main():
    parser = argparse.ArgumentParser(description="Effective rank analysis for contrastive pairs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to use")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                       default=["truthfulqa_generation", "livecodebench", "halulens"],
                       help="Benchmarks to analyze")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                       help="Extraction strategies (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of pairs (None = all)")
    parser.add_argument("--cache-dir", type=str, default="/tmp/wisent_effective_rank_cache",
                       help="Cache directory for activations")
    parser.add_argument("--output-dir", type=str, default="./effective_rank_results",
                       help="Output directory for results")
    parser.add_argument("--no-subsets", action="store_true", help="Skip linear subset detection")
    
    args = parser.parse_args()
    
    # All available strategies
    ALL_STRATEGIES = [
        "chat_mean",
        "chat_first", 
        "chat_last",
        "chat_max_norm",
        "chat_weighted",
        "role_play",
        "mc_balanced",
        "completion_last",
        "completion_mean",
        "mc_completion",
    ]
    
    strategies_to_test = args.strategies if args.strategies else ALL_STRATEGIES
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    cache = ActivationCache(args.cache_dir)
    
    print(f"Loading model: {args.model}")
    model = WisentModel(args.model, device="cuda")
    print(f"Model loaded. Layers: {model.num_layers}")
    print(f"Strategies to test: {strategies_to_test}")
    
    results = {}
    
    for benchmark in args.benchmarks:
        results[benchmark] = {}
        
        for strategy_name in strategies_to_test:
            print(f"\n{'='*60}")
            print(f"Analyzing: {benchmark} / {strategy_name}")
            print('='*60)
            
            try:
                strategy = ExtractionStrategy(strategy_name)
                
                analysis = analyze_benchmark(
                    model=model,
                    benchmark=benchmark,
                    strategy=strategy,
                    cache=cache,
                    limit=args.limit,
                    find_subsets=not args.no_subsets,
                )
                
                # Print summary
                print(f"\n--- Results for {benchmark} / {strategy_name} ---")
                print(f"Pairs: {analysis.num_pairs}")
                print(f"Best layer (by Fisher): {analysis.best_layer}")
                print(f"Best Fisher ratio: {analysis.best_fisher_ratio:.3f}")
                
                # Find best layer result to show mean cosine too
                best_result = next((r for r in analysis.per_layer_results if r.layer == analysis.best_layer), None)
                if best_result:
                    print(f"Mean cosine at best layer: {best_result.mean_cosine:.3f}")
                
                print(f"\n{'Layer':<6} {'Rank80':<7} {'MeanCos':<8} {'Fisher1':<8} {'Fisher2':<8} {'MaxFish':<8} {'TopPC':<6} {'PC1%':<6}")
                print("-" * 70)
                for r in analysis.per_layer_results:
                    print(f"{r.layer:<6} {r.rank_80:<7} {r.mean_cosine:<8.3f} {r.fisher_ratio_pc1:<8.2f} {r.fisher_ratio_pc2:<8.2f} {r.max_fisher_ratio:<8.2f} {r.top_fisher_pc:<6} {r.pc1_variance*100:<6.1f}")
                
                if analysis.linear_subsets:
                    print(f"\n--- Linear Subsets ---")
                    for i, subset in enumerate(analysis.linear_subsets):
                        print(f"Subset {i}: n={len(subset.pair_indices)}, rank80={subset.rank_80}, "
                              f"mean_cos={subset.mean_cosine:.3f}, pc1={subset.pc1_variance:.3f}")
                
                # Convert to dict for JSON serialization
                results[benchmark][strategy_name] = {
                    "benchmark": analysis.benchmark,
                    "model": analysis.model,
                    "strategy": analysis.strategy,
                    "num_pairs": analysis.num_pairs,
                    "best_layer": analysis.best_layer,
                    "best_fisher_ratio": analysis.best_fisher_ratio,
                    "per_layer_results": [asdict(r) for r in analysis.per_layer_results],
                    "linear_subsets": [asdict(s) for s in analysis.linear_subsets] if analysis.linear_subsets else None,
                }
                
            except Exception as e:
                print(f"Error analyzing {benchmark}/{strategy_name}: {e}")
                import traceback
                traceback.print_exc()
                results[benchmark][strategy_name] = {"error": str(e)}
    
    # Save results
    output_file = os.path.join(args.output_dir, f"effective_rank_{args.model.replace('/', '_')}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print("SUMMARY: Best Fisher ratio per benchmark/strategy")
    print('='*100)
    print(f"{'Benchmark':<25} {'Strategy':<20} {'BestLayer':<10} {'MaxFisher':<10} {'MeanCos':<10} {'Rank80':<8}")
    print("-" * 100)
    
    for benchmark in results:
        for strategy_name in results[benchmark]:
            data = results[benchmark][strategy_name]
            if "error" in data:
                print(f"{benchmark:<25} {strategy_name:<20} ERROR")
            else:
                best_layer = data["best_layer"]
                best_fisher = data["best_fisher_ratio"]
                # Find rank80 and mean_cosine for best layer
                best_result = next(
                    (r for r in data["per_layer_results"] if r["layer"] == best_layer),
                    None
                )
                best_rank80 = best_result["rank_80"] if best_result else -1
                best_cos = best_result["mean_cosine"] if best_result else 0
                print(f"{benchmark:<25} {strategy_name:<20} {best_layer:<10} {best_fisher:<10.2f} {best_cos:<10.3f} {best_rank80:<8}")


if __name__ == "__main__":
    main()
