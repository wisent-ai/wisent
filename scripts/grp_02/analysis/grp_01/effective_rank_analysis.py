#!/usr/bin/env python3
"""
Effective rank analysis for contrastive pairs.

Computes PCA effective rank and mean cosine similarity to evaluate signal quality.
Can also find linear subsets within benchmarks.
"""

import argparse
import json
import os
from typing import Optional
from dataclasses import asdict

from wisent.core.primitives.models.wisent_model import WisentModel
from wisent.extractors.lm_eval.lm_task_pairs_generation import lm_build_contrastive_pairs
from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy
from wisent.core.primitives.model_interface.core.activations.activation_cache import collect_and_cache_activations, ActivationCache
from lm_eval.tasks import TaskManager

from effective_rank_analysis_compute import (
    EffectiveRankResult,
    BenchmarkAnalysis,
    compute_effective_rank,
    find_linear_subsets,
)


def analyze_benchmark(
    model: WisentModel,
    benchmark: str,
    strategy: ExtractionStrategy,
    cache: ActivationCache,
    limit: Optional[int] = None,
    find_subsets: bool = True,
) -> BenchmarkAnalysis:
    """Analyze effective rank for a benchmark across all layers."""
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

    cached = collect_and_cache_activations(
        model=model, pairs=pairs,
        benchmark=benchmark if limit is None else f"{benchmark}_limit{limit}",
        strategy=strategy, cache=cache, show_progress=True,
    )

    per_layer_results = []
    best_layer = 0
    best_fisher = -1

    for layer_idx in range(cached.num_layers):
        layer_name = str(layer_idx + 1)
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

        if result.max_fisher_ratio > best_fisher:
            best_fisher = result.max_fisher_ratio
            best_layer = layer_idx

    linear_subsets = None
    if find_subsets:
        layer_name = str(best_layer + 1)
        pos_vectors = cached.get_positive_activations(layer_name)
        neg_vectors = cached.get_negative_activations(layer_name)
        n_pairs = min(len(pos_vectors), len(neg_vectors))
        diff_np = (pos_vectors[:n_pairs] - neg_vectors[:n_pairs]).float().cpu().numpy()
        linear_subsets = find_linear_subsets(diff_np)

    return BenchmarkAnalysis(
        benchmark=benchmark, model=model.model_name,
        strategy=strategy.value, num_pairs=cached.num_pairs,
        per_layer_results=per_layer_results, best_layer=best_layer,
        best_fisher_ratio=best_fisher, linear_subsets=linear_subsets,
    )


def main():
    parser = argparse.ArgumentParser(description="Effective rank analysis for contrastive pairs")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B", help="Model to use")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                       default=["truthfulqa_generation", "livecodebench", "halulens"])
    parser.add_argument("--strategies", type=str, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-dir", type=str, default="/tmp/wisent_effective_rank_cache")
    parser.add_argument("--output-dir", type=str, default="./effective_rank_results")
    parser.add_argument("--no-subsets", action="store_true")
    args = parser.parse_args()

    ALL_STRATEGIES = [
        "chat_mean", "chat_first", "chat_last", "chat_max_norm",
        "chat_weighted", "role_play", "mc_balanced",
        "completion_last", "completion_mean", "mc_completion",
    ]
    strategies_to_test = args.strategies if args.strategies else ALL_STRATEGIES

    os.makedirs(args.output_dir, exist_ok=True)
    cache = ActivationCache(args.cache_dir)

    print(f"Loading model: {args.model}")
    model = WisentModel(args.model, device="cuda")
    print(f"Model loaded. Layers: {model.num_layers}")

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
                    model=model, benchmark=benchmark, strategy=strategy,
                    cache=cache, limit=args.limit, find_subsets=not args.no_subsets,
                )
                print(f"\n--- Results for {benchmark} / {strategy_name} ---")
                print(f"Pairs: {analysis.num_pairs}, Best layer: {analysis.best_layer}")
                print(f"Best Fisher ratio: {analysis.best_fisher_ratio:.3f}")

                print(f"\n{'Layer':<6} {'Rank80':<7} {'MeanCos':<8} {'Fisher1':<8} {'Fisher2':<8} {'MaxFish':<8} {'TopPC':<6} {'PC1%':<6}")
                print("-" * 70)
                for r in analysis.per_layer_results:
                    print(f"{r.layer:<6} {r.rank_80:<7} {r.mean_cosine:<8.3f} {r.fisher_ratio_pc1:<8.2f} {r.fisher_ratio_pc2:<8.2f} {r.max_fisher_ratio:<8.2f} {r.top_fisher_pc:<6} {r.pc1_variance*100:<6.1f}")

                if analysis.linear_subsets:
                    print(f"\n--- Linear Subsets ---")
                    for i, subset in enumerate(analysis.linear_subsets):
                        print(f"Subset {i}: n={len(subset.pair_indices)}, rank80={subset.rank_80}, "
                              f"mean_cos={subset.mean_cosine:.3f}, pc1={subset.pc1_variance:.3f}")

                results[benchmark][strategy_name] = {
                    "benchmark": analysis.benchmark, "model": analysis.model,
                    "strategy": analysis.strategy, "num_pairs": analysis.num_pairs,
                    "best_layer": analysis.best_layer, "best_fisher_ratio": analysis.best_fisher_ratio,
                    "per_layer_results": [asdict(r) for r in analysis.per_layer_results],
                    "linear_subsets": [asdict(s) for s in analysis.linear_subsets] if analysis.linear_subsets else None,
                }
            except Exception as e:
                print(f"Error analyzing {benchmark}/{strategy_name}: {e}")
                import traceback
                traceback.print_exc()
                results[benchmark][strategy_name] = {"error": str(e)}

    output_file = os.path.join(args.output_dir, f"effective_rank_{args.model.replace('/', '_')}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
