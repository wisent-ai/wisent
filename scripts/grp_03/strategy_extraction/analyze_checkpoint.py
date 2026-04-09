#!/usr/bin/env python3
"""Analyze checkpoint results, reporting, aggregation, and verification."""

import json
import os
import argparse
import importlib.util
from collections import defaultdict
from typing import Dict

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from wisent.core.constants import VIZ_DPI

_metrics_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'wisent', 'core', 'activations', 'core', 'diagnostics', 'metrics.py')
_spec = importlib.util.spec_from_file_location("metrics", _metrics_path)
_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_metrics)
compute_pairwise_consistency = _metrics.compute_pairwise_consistency
compute_linear_nonlinear_accuracy = _metrics.compute_linear_nonlinear_accuracy
compute_steering_quality = _metrics.compute_steering_quality


def compute_full_metrics(pos: np.ndarray, neg: np.ndarray) -> Dict:
    """Compute comprehensive diagnostics metrics."""
    pos_t = torch.tensor(pos, dtype=torch.float32)
    neg_t = torch.tensor(neg, dtype=torch.float32)
    directions = pos_t - neg_t
    consistency, consistency_std = compute_pairwise_consistency(directions)
    linear_acc, nonlinear_acc = compute_linear_nonlinear_accuracy(pos_t, neg_t, cv_folds=3)
    steer_acc, effect_size = compute_steering_quality(directions)
    return {"linear_acc": linear_acc, "nonlinear_acc": nonlinear_acc,
        "consistency": consistency, "steer_acc": steer_acc,
        "effect_size": effect_size}


def verify_benchmark_completion(results: Dict, benchmark: str, model_name: str) -> bool:
    """Verify that a benchmark was actually processed and saved correctly."""
    if benchmark not in results.get("benchmarks", {}):
        print(f"    VERIFY FAIL: {benchmark} not in results")
        return False
    bench_data = results["benchmarks"][benchmark]
    if "strategies" not in bench_data:
        print(f"    VERIFY FAIL: {benchmark} has no strategies")
        return False
    if not bench_data["strategies"]:
        print(f"    VERIFY FAIL: {benchmark} strategies dict is empty")
        return False
    has_valid_metrics = False
    for strategy, metrics in bench_data["strategies"].items():
        if metrics and any(v is not None and v != 0 for v in metrics.values() if not isinstance(v, dict)):
            has_valid_metrics = True
            break
    if not has_valid_metrics:
        print(f"    VERIFY FAIL: {benchmark} has no valid metrics")
        return False
    return True


def aggregate_results(all_results: Dict) -> Dict:
    """Aggregate results across all models by strategy and category."""
    agg = {"by_strategy": defaultdict(lambda: defaultdict(list)),
        "by_category": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "by_model": {}}
    for model, data in all_results.items():
        model_metrics = {}
        for strategy, metrics in data.get("strategies", {}).items():
            if metrics:
                numeric_metrics = {}
                for k, v in metrics.items():
                    if v:
                        numeric_v = [float(x) for x in v if x is not None]
                        if numeric_v:
                            numeric_metrics[k] = numeric_v
                model_metrics[strategy] = {k: np.mean(v) for k, v in numeric_metrics.items()}
                for k, vals in numeric_metrics.items():
                    agg["by_strategy"][strategy][k].extend(vals)
        agg["by_model"][model] = model_metrics
        for bench, bench_data in data.get("benchmarks", {}).items():
            cat = bench_data.get("category", "other")
            for strategy, metrics in bench_data.get("strategies", {}).items():
                for k, v in metrics.items():
                    try:
                        agg["by_category"][cat][strategy][k].append(float(v))
                    except (ValueError, TypeError):
                        pass
    summary = {"strategies": {}, "categories": {}, "models": agg["by_model"]}
    for strat, metrics in agg["by_strategy"].items():
        if metrics:
            summary["strategies"][strat] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in metrics.items() if v}
    for cat, strats in agg["by_category"].items():
        summary["categories"][cat] = {}
        for strat, metrics in strats.items():
            if metrics:
                summary["categories"][cat][strat] = {
                    k: float(np.mean(v)) for k, v in metrics.items() if v}
    return summary


def plot_results(summary: Dict, output_dir: str):
    """Generate strategy comparison bar charts (legacy wrapper)."""
    os.makedirs(output_dir, exist_ok=True)


def plot_model(model_name: str, model_data: dict, output_dir: str):
    """Generate comprehensive visualizations for a single model."""
    os.makedirs(output_dir, exist_ok=True)
    results = model_data.get("benchmarks", model_data)
    strat_metrics = defaultdict(lambda: defaultdict(list))
    for bname, bdata in results.items():
        if not isinstance(bdata, dict) or bdata.get("no_data"):
            continue
        for strat, metrics in bdata.get("strategies", {}).items():
            if not metrics or not isinstance(metrics, dict):
                continue
            for k, v in metrics.items():
                if v is not None:
                    try:
                        strat_metrics[strat][k].append(float(v))
                    except (ValueError, TypeError):
                        pass
    if not strat_metrics:
        return
    strats = sorted(strat_metrics.keys(), key=lambda s: -(
        np.mean(strat_metrics[s].get('best_linear_acc', [0]))))
    safe = model_name.replace("/", "_")
    colors = plt.cm.tab10(np.linspace(0, 1, len(strats)))
    # Fig 1: Best-layer metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} - Best Layer Metrics', fontsize=14, fontweight='bold')
    for ax, metric, label in zip(axes.flat,
            ['best_linear_acc', 'best_nonlinear_acc', 'best_consistency', 'best_steer_acc'],
            ['Linear Accuracy', 'Nonlinear (MLP) Accuracy', 'Consistency', 'Steering Accuracy']):
        means = [np.mean(strat_metrics[s].get(metric, [0])) for s in strats]
        stds = [np.std(strat_metrics[s].get(metric, [0])) for s in strats]
        bars = ax.bar(range(len(strats)), means, yerr=stds, color=colors, capsize=3, alpha=0.85)
        ax.set_xticks(range(len(strats)))
        ax.set_xticklabels(strats, rotation=45, ha='right', fontsize=7)
        ax.set_title(label)
        ax.set_ylim(0, max(1.0, max(means) * 1.1) if means else 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{safe}_best_layer.png'), dpi=VIZ_DPI)
    plt.close()
    # Fig 2: Mean-across-layers + signal breadth
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'{model_name} - Multi-Layer Metrics', fontsize=14, fontweight='bold')
    for ax, metric, label in zip(axes[:2],
            ['mean_linear_acc', 'mean_consistency'], ['Mean Linear Acc', 'Mean Consistency']):
        means = [np.mean(strat_metrics[s].get(metric, [0])) for s in strats]
        ax.bar(range(len(strats)), means, color=colors, alpha=0.85)
        ax.set_xticks(range(len(strats)))
        ax.set_xticklabels(strats, rotation=45, ha='right', fontsize=7)
        ax.set_title(label)
    breadths = [np.mean(strat_metrics[s].get('signal_breadth', [0])) for s in strats]
    axes[2].bar(range(len(strats)), breadths, color=colors, alpha=0.85)
    axes[2].set_xticks(range(len(strats)))
    axes[2].set_xticklabels(strats, rotation=45, ha='right', fontsize=7)
    axes[2].set_title('Signal Breadth (layers > 0.6)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{safe}_multi_layer.png'), dpi=VIZ_DPI)
    plt.close()
    # Fig 3: Linear vs Nonlinear scatter (curse of dimensionality diagnostic)
    fig, ax = plt.subplots(figsize=(8, 8))
    for i, strat in enumerate(strats):
        lin = strat_metrics[strat].get('best_linear_acc', [])
        nlin = strat_metrics[strat].get('best_nonlinear_acc', [])
        n = min(len(lin), len(nlin))
        if n > 0:
            ax.scatter(lin[:n], nlin[:n], alpha=0.3, s=15, color=colors[i], label=strat)
    ax.plot([0.4, 1], [0.4, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Linear Accuracy (best layer)')
    ax.set_ylabel('Nonlinear Accuracy (best layer)')
    ax.set_title(f'{model_name} - Linear vs Nonlinear (curse of dim. diagnostic)')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.4, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{safe}_lin_vs_nlin.png'), dpi=VIZ_DPI)
    plt.close()
    # Fig 4: Per-category heatmap (best_linear_acc)
    cats = defaultdict(lambda: defaultdict(list))
    for bname, bdata in results.items():
        if not isinstance(bdata, dict) or bdata.get("no_data"):
            continue
        cat = bdata.get("category", "other")
        for strat, metrics in bdata.get("strategies", {}).items():
            if metrics and isinstance(metrics, dict) and "best_linear_acc" in metrics:
                try:
                    cats[cat][strat].append(float(metrics["best_linear_acc"]))
                except (ValueError, TypeError):
                    pass
    cat_names = sorted(cats.keys())
    if cat_names and strats:
        heatmap = np.zeros((len(cat_names), len(strats)))
        for i, cat in enumerate(cat_names):
            for j, strat in enumerate(strats):
                vals = cats[cat].get(strat, [])
                heatmap[i, j] = np.mean(vals) if vals else 0
        fig, ax = plt.subplots(figsize=(max(10, len(strats) * 0.8), max(6, len(cat_names) * 0.5)))
        im = ax.imshow(heatmap, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=1.0)
        ax.set_xticks(range(len(strats)))
        ax.set_xticklabels(strats, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(cat_names)))
        ax.set_yticklabels(cat_names, fontsize=8)
        for i, cat_names in enumerate(cat_names):
            for j, strats in enumerate(strats):
                ax.text(j, i, f'{heatmap[i,j]:.2f}', ha='center', va='center', fontsize=6)
        plt.colorbar(im, ax=ax, label='Best Linear Accuracy')
        ax.set_title(f'{model_name} - Strategy x Category (best linear acc)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{safe}_category_heatmap.png'), dpi=VIZ_DPI)
        plt.close()
    print(f"  Plots saved to {output_dir}/{safe}_*.png")




def analyze_model(data: dict, model: str):
    """Print summary metrics for a model."""
    if model not in data:
        print(f"Model {model} not found in checkpoint")
        return
    model_data = data[model]
    results = model_data.get("benchmarks", model_data)
    n_data = sum(1 for b in results.values() if isinstance(b, dict) and not b.get("no_data"))
    n_nodata = sum(1 for b in results.values() if isinstance(b, dict) and b.get("no_data"))
    strategy_metrics = defaultdict(lambda: defaultdict(list))
    for benchmark, bench_data in results.items():
        if not isinstance(bench_data, dict) or bench_data.get("no_data"):
            continue
        strats = bench_data.get("strategies", bench_data)
        for strat, metrics in strats.items():
            if not metrics or not isinstance(metrics, dict):
                continue
            for key in metrics:
                val = metrics[key]
                if val is not None:
                    try:
                        strategy_metrics[strat][key].append(float(val))
                    except (ValueError, TypeError):
                        pass
    print(f"\nResults for {model} ({n_data} benchmarks, {n_nodata} no-data)")
    print("=" * 100)
    is_multi = any("best_linear_acc" in m for m in strategy_metrics.values())
    if is_multi:
        strats_sorted = sorted(strategy_metrics.keys(), key=lambda s: -(
            sum(strategy_metrics[s].get('best_linear_acc', [0])) /
            max(len(strategy_metrics[s].get('best_linear_acc', [0])), 1)))
        print(f"\n  Best-layer (single-layer steering):")
        print(f"  {'Strategy':<18} {'LinAcc':>8} {'NLAcc':>8} {'Consist':>8} {'SteerAcc':>9} {'EffSize':>8} {'N':>5}")
        print(f"  {'-'*62}")
        for strat in strats_sorted:
            m = strategy_metrics[strat]
            n = len(m.get('best_linear_acc', []))
            if n == 0:
                continue
            avg = lambda k: sum(m.get(k, [0])) / max(len(m.get(k, [1])), 1)
            print(f"  {strat:<18} {avg('best_linear_acc'):>8.4f} {avg('best_nonlinear_acc'):>8.4f} "
                f"{avg('best_consistency'):>8.4f} {avg('best_steer_acc'):>9.4f} {avg('best_effect_size'):>8.2f} {n:>5}")
        print(f"\n  Mean-across-layers (multi-layer steering):")
        print(f"  {'Strategy':<18} {'LinAcc':>8} {'NLAcc':>8} {'Consist':>8} {'Breadth':>8} {'N':>5}")
        print(f"  {'-'*52}")
        for strat in strats_sorted:
            m = strategy_metrics[strat]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="strategy_analysis_results/checkpoint.json")
    parser.add_argument("--model", help="Specific model to analyze (default: all)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--output-dir", default="strategy_analysis_results")
    args = parser.parse_args()
    with open(args.checkpoint) as f:
        data = json.load(f)
    print(f"Checkpoint contains {len(data)} models: {list(data.keys())}")
    models = [args.model] if args.model else list(data.keys())
    for model in models:
        analyze_model(data, model)
        if args.plot:
            plot_model(model, data[model], args.output_dir)


if __name__ == "__main__":
    main()
