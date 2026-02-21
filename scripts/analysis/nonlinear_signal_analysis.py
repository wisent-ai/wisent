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
from pathlib import Path
from typing import Optional
from dataclasses import asdict

from wisent.core.models.wisent_model import WisentModel
from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent.core.activations import ExtractionStrategy
from wisent.core.activations.activation_cache import collect_and_cache_activations
from lm_eval.tasks import TaskManager

from nonlinear_signal_analysis_compute import (
    BenchmarkNonlinearAnalysis,
    analyze_layer,
)


def analyze_benchmark(
    model: WisentModel,
    benchmark: str,
    strategy: ExtractionStrategy,
    limit: int = 100,
    cache_dir: str = "/tmp/wisent_nonlinear_cache",
) -> Optional[BenchmarkNonlinearAnalysis]:
    """Run full nonlinear analysis on a benchmark."""
    try:
        tm = TaskManager()
        task_dict = tm.load_task_or_group([benchmark])
        task = list(task_dict.values())[0]
    except:
        task = None

    pairs = lm_build_contrastive_pairs(benchmark, task, limit=limit)
    if not pairs:
        print(f"No pairs for {benchmark}")
        return None
    print(f"Loaded {len(pairs)} pairs for {benchmark}")

    cached = collect_and_cache_activations(
        model=model, pairs=pairs, benchmark=benchmark,
        strategy=strategy, cache_dir=cache_dir, show_progress=True,
    )

    results = []
    for layer in range(model.num_layers):
        layer_name = str(layer + 1)
        try:
            pos_vectors = cached.get_positive_activations(layer_name)
            neg_vectors = cached.get_negative_activations(layer_name)
        except KeyError:
            continue
        pos_np = pos_vectors.float().cpu().numpy()
        neg_np = neg_vectors.float().cpu().numpy()
        if np.isnan(pos_np).any() or np.isnan(neg_np).any():
            print(f"  Layer {layer}: skipping due to NaN")
            continue
        result = analyze_layer(pos_np, neg_np, layer)
        results.append(result)

    if not results:
        return None

    best_knn_idx = np.argmax([r.knn_accuracy_k10 for r in results])
    best_mmd_idx = np.argmax([r.mmd_rbf for r in results])

    return BenchmarkNonlinearAnalysis(
        benchmark=benchmark, model=model.model_name, strategy=strategy.value,
        num_pairs=len(pairs), per_layer_results=results,
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

    print(f"Loading model: {args.model}")
    model = WisentModel(args.model)
    print(f"Model loaded. Layers: {model.num_layers}")

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
                    model=model, benchmark=benchmark, strategy=strategy,
                    limit=args.limit, cache_dir=args.cache_dir,
                )
                if analysis:
                    print_results(analysis)
                    all_results.append(asdict(analysis))
            except Exception as e:
                print(f"Error analyzing {benchmark}/{strategy_name}: {e}")
                import traceback
                traceback.print_exc()

    output_file = Path(args.output_dir) / f"nonlinear_{args.model.replace('/', '_')}.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    print(f"{'Benchmark':<25} {'Strategy':<15} {'BestLayer':<10} {'kNN-10':<10} {'MMD':<10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['benchmark']:<25} {r['strategy']:<15} {r['best_layer_knn']:<10} {r['best_knn_accuracy']:<10.3f} {r['best_mmd']:<10.4f}")


if __name__ == "__main__":
    main()
