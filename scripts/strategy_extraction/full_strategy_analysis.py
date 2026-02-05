#!/usr/bin/env python3
"""Full extraction strategy analysis with comprehensive diagnostics."""
import argparse, json, struct, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np
import torch
import psycopg2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wisent.core.activations.core.diagnostics.metrics import (
    compute_pairwise_consistency, compute_linear_nonlinear_accuracy, compute_steering_quality)
from database import get_available_layers, compute_layer_aggregates

DB_CONFIG = {"host": "aws-0-eu-west-2.pooler.supabase.com", "port": 5432,
    "database": "postgres", "user": "postgres.rbqjqnouluslojmmnuqi",
    "password": "BsKuEnPFLCFurN4a"}
MODELS = ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen3-8B",
    "meta-llama/Llama-2-7b-chat-hf", "openai/gpt-oss-20b"]
CATEGORIES = {"truthfulness": ["truthfulqa", "toxigen"],
    "reasoning": ["arc", "hellaswag", "piqa", "siqa", "winogrande", "boolq"],
    "knowledge": ["mmlu", "openbookqa", "sciq"], "math": ["gsm8k", "math"],
    "code": ["humaneval", "mbpp"]}


def bytes_to_vector(data: bytes) -> np.ndarray:
    num_floats = len(data) // 4
    return np.array(struct.unpack(f'{num_floats}f', data))


def get_benchmark_category(benchmark_name: str) -> str:
    name_lower = benchmark_name.lower()
    for category, patterns in CATEGORIES.items():
        for pattern in patterns:
            if pattern in name_lower:
                return category
    return "other"


def get_benchmarks(conn) -> List[str]:
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT name FROM "ContrastivePairSet" ORDER BY name')
    benchmarks = [row[0] for row in cur.fetchall()]
    cur.close()
    return benchmarks


def get_model_id(conn, model_name: str) -> Optional[int]:
    cur = conn.cursor()
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    result = cur.fetchone()
    cur.close()
    return (result[0], result[1]) if result else (None, None)


def load_benchmark_activations(conn, model_id: int, layer: int, benchmark: str) -> Dict:
    cur = conn.cursor()
    cur.execute('''
        SELECT a."contrastivePairId", a."extractionStrategy", a."activationData", a."isPositive"
        FROM "Activation" a
        JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
        WHERE a."modelId" = %s AND a.layer = %s AND cps.name = %s
    ''', (model_id, layer, benchmark))
    rows = cur.fetchall()
    cur.close()

    raw = defaultdict(lambda: defaultdict(dict))
    for pair_id, strategy, data, is_pos in rows:
        key = "pos" if is_pos else "neg"
        raw[strategy][(pair_id, key)] = bytes_to_vector(data)

    result = {}
    for strategy, pairs in raw.items():
        pos_acts, neg_acts = [], []
        pair_ids = set(p[0] for p in pairs.keys())
        for pid in pair_ids:
            if (pid, "pos") in pairs and (pid, "neg") in pairs:
                pos_acts.append(pairs[(pid, "pos")])
                neg_acts.append(pairs[(pid, "neg")])
        if len(pos_acts) >= 5:
            result[strategy] = {"pos": np.array(pos_acts), "neg": np.array(neg_acts)}
    return result


def compute_full_metrics(pos: np.ndarray, neg: np.ndarray) -> Dict:
    """Compute comprehensive diagnostics metrics."""
    pos_t = torch.tensor(pos, dtype=torch.float32)
    neg_t = torch.tensor(neg, dtype=torch.float32)
    directions = pos_t - neg_t

    # Direction consistency
    consistency, consistency_std = compute_pairwise_consistency(directions)

    # Linear/nonlinear accuracy (use fewer CV folds for speed)
    linear_acc, nonlinear_acc = compute_linear_nonlinear_accuracy(pos_t, neg_t, cv_folds=3)

    # Steering quality
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
    # Check that at least one strategy has valid metrics
    has_valid_metrics = False
    for strategy, metrics in bench_data["strategies"].items():
        if metrics and any(v is not None and v != 0 for v in metrics.values() if not isinstance(v, dict)):
            has_valid_metrics = True
            break
    if not has_valid_metrics:
        print(f"    VERIFY FAIL: {benchmark} has no valid metrics")
        return False
    return True


def analyze_model(model_name: str, benchmarks: List[str], checkpoint_file: str, all_results: Dict, multi_layer: bool = False) -> Dict:
    conn = psycopg2.connect(**DB_CONFIG)
    model_id, num_layers = get_model_id(conn, model_name)
    if model_id is None:
        print(f"  Model {model_name} not found"); conn.close(); return {}
    existing = all_results.get(model_name, {})
    results = {"benchmarks": existing.get("benchmarks", {}), "strategies": defaultdict(lambda: defaultdict(list)), "num_layers": num_layers}
    for bench, bdata in results["benchmarks"].items():
        for strat, metrics in bdata.get("strategies", {}).items():
            for k, v in metrics.items():
                if not k.startswith("per_layer"): results["strategies"][strat][k].append(float(v) if isinstance(v, str) else v)

    processed_count = 0
    verified_count = 0
    failed_benchmarks = []

    for i, benchmark in enumerate(benchmarks):
        if benchmark in results["benchmarks"]: continue
        if (i + 1) % 20 == 0: print(f"    {i+1}/{len(benchmarks)} benchmarks...", flush=True)
        try:
            if multi_layer:
                layers = get_available_layers(conn, model_id, benchmark)
                if not layers: continue
                strat_per_layer = defaultdict(dict)
                for layer in layers:
                    acts = load_benchmark_activations(conn, model_id, layer, benchmark)
                    for strategy, data in acts.items():
                        if len(data["pos"]) >= 10: strat_per_layer[strategy][layer] = compute_full_metrics(data["pos"], data["neg"])
                bench_results = {}
                for strategy, per_layer in strat_per_layer.items():
                    if per_layer:
                        metrics = compute_layer_aggregates(per_layer)
                        bench_results[strategy] = metrics
                        for k, v in metrics.items():
                            if k not in ["best_layer", "total_layers"]: results["strategies"][strategy][k].append(v)
            else:
                acts = load_benchmark_activations(conn, model_id, num_layers // 2, benchmark)
                if not acts: continue
                bench_results = {}
                for strategy, data in acts.items():
                    if len(data["pos"]) >= 10:
                        metrics = compute_full_metrics(data["pos"], data["neg"])
                        bench_results[strategy] = metrics
                        for k, v in metrics.items(): results["strategies"][strategy][k].append(v)
            results["benchmarks"][benchmark] = {"category": get_benchmark_category(benchmark), "strategies": bench_results}
            processed_count += 1

            # VERIFICATION: Check that benchmark was actually saved correctly
            if verify_benchmark_completion(results, benchmark, model_name):
                verified_count += 1
                print(f"    VERIFIED: {benchmark} ({len(bench_results)} strategies)", flush=True)
            else:
                failed_benchmarks.append(benchmark)
                # Remove failed benchmark from results to retry next time
                del results["benchmarks"][benchmark]
                print(f"    REMOVED: {benchmark} (verification failed)", flush=True)

            all_results[model_name] = results
            with open(checkpoint_file, "w") as f: json.dump(all_results, f, default=str)

            # Double-check checkpoint file was written correctly
            with open(checkpoint_file, "r") as f:
                saved_data = json.load(f)
            if model_name not in saved_data or benchmark not in saved_data.get(model_name, {}).get("benchmarks", {}):
                if benchmark in results["benchmarks"]:
                    print(f"    CHECKPOINT FAIL: {benchmark} not saved to file, retrying...", flush=True)
                    with open(checkpoint_file, "w") as f: json.dump(all_results, f, default=str)

        except psycopg2.OperationalError as e:
            print(f"    DB ERROR: {benchmark} - {e}", flush=True)
            failed_benchmarks.append(benchmark)
            try: conn.close()
            except: pass
            conn = psycopg2.connect(**DB_CONFIG); model_id, _ = get_model_id(conn, model_name); continue
        except Exception as e:
            print(f"    ERROR: {benchmark} - {e}", flush=True)
            failed_benchmarks.append(benchmark)
            try: conn.rollback()
            except: pass
            continue

    conn.close()

    # Final summary
    print(f"\n  SUMMARY for {model_name}:")
    print(f"    Processed: {processed_count}, Verified: {verified_count}, Failed: {len(failed_benchmarks)}")
    if failed_benchmarks:
        print(f"    Failed benchmarks: {failed_benchmarks[:10]}{'...' if len(failed_benchmarks) > 10 else ''}")

    return results


def aggregate_results(all_results: Dict) -> Dict:
    agg = {"by_strategy": defaultdict(lambda: defaultdict(list)),
        "by_category": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "by_model": {}}

    for model, data in all_results.items():
        model_metrics = {}
        for strategy, metrics in data.get("strategies", {}).items():
            if metrics:  # Just check if there are any metrics
                model_metrics[strategy] = {k: np.mean(v) for k, v in metrics.items() if v}
                for k, vals in metrics.items():
                    agg["by_strategy"][strategy][k].extend(vals)
        agg["by_model"][model] = model_metrics

        for bench, bench_data in data.get("benchmarks", {}).items():
            cat = bench_data.get("category", "other")
            for strategy, metrics in bench_data.get("strategies", {}).items():
                for k, v in metrics.items():
                    agg["by_category"][cat][strategy][k].append(v)

    summary = {"strategies": {}, "categories": {}, "models": agg["by_model"]}
    for strat, metrics in agg["by_strategy"].items():
        if metrics:  # Just check if there are any metrics
            summary["strategies"][strat] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in metrics.items() if v}

    for cat, strats in agg["by_category"].items():
        summary["categories"][cat] = {}
        for strat, metrics in strats.items():
            if metrics:  # Just check if there are any metrics
                summary["categories"][cat][strat] = {
                    k: float(np.mean(v)) for k, v in metrics.items() if v}

    return summary


def plot_results(summary: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    strategies = sorted(summary["strategies"].keys())
    if not strategies:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.flat, ["linear_acc", "consistency", "steer_acc", "effect_size"]):
        vals = [summary["strategies"][s].get(metric, {}).get("mean", 0) for s in strategies]
        ax.bar(range(len(strategies)), vals)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=8)
        ax.set_title(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_metrics.png'), dpi=150)
    plt.close()


def print_summary(summary: Dict):
    print("\n" + "=" * 80 + "\nEXTRACTION STRATEGY ANALYSIS\n" + "=" * 80)
    if not summary["strategies"]: print("No data collected"); return
    sample = next(iter(summary["strategies"].values()), {})
    if "best_linear_acc" in sample:  # Multi-layer mode
        print("\nSingle-layer methods (best layer):")
        print(f"{'Strategy':<15} {'LinAcc':>8} {'Consist':>8} {'EffSize':>8}")
        for strat, m in sorted(summary["strategies"].items(), key=lambda x: x[1].get("best_linear_acc", {}).get("mean", 0), reverse=True):
            print(f"  {strat:<13} {m.get('best_linear_acc', {}).get('mean', 0):>8.3f} {m.get('best_consistency', {}).get('mean', 0):>8.3f} {m.get('best_effect_size', {}).get('mean', 0):>8.3f}")
        print("\nMulti-layer methods (mean across layers):")
        print(f"{'Strategy':<15} {'LinAcc':>8} {'Consist':>8} {'Breadth':>8}")
        for strat, m in sorted(summary["strategies"].items(), key=lambda x: x[1].get("mean_linear_acc", {}).get("mean", 0), reverse=True):
            print(f"  {strat:<13} {m.get('mean_linear_acc', {}).get('mean', 0):>8.3f} {m.get('mean_consistency', {}).get('mean', 0):>8.3f} {m.get('signal_breadth', {}).get('mean', 0):>8.1f}")
    else:
        print(f"\n{'Strategy':<15} {'LinAcc':>8} {'Consist':>8} {'EffSize':>8}")
        for strat, m in sorted(summary["strategies"].items(), key=lambda x: x[1].get("linear_acc", {}).get("mean", 0), reverse=True):
            print(f"  {strat:<13} {m.get('linear_acc', {}).get('mean', 0):>8.3f} {m.get('consistency', {}).get('mean', 0):>8.3f} {m.get('effect_size', {}).get('mean', 0):>8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Full extraction strategy analysis")
    parser.add_argument("--output-dir", default="strategy_analysis_results")
    parser.add_argument("--models", nargs="*", default=None, help="Models to analyze")
    parser.add_argument("--multi-layer", action="store_true", help="Analyze ALL layers (not just middle)")
    parser.add_argument("--benchmark", help="Analyze only this benchmark")
    args = parser.parse_args()

    models = args.models if args.models else MODELS
    if args.multi_layer: print("Multi-layer mode: analyzing ALL layers")

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    if args.benchmark:
        benchmarks = [args.benchmark]
        print(f"Analyzing single benchmark: {args.benchmark}")
    else:
        benchmarks = get_benchmarks(conn)
        print(f"Found {len(benchmarks)} benchmarks")
    conn.close()

    checkpoint_file = os.path.join(args.output_dir, "checkpoint.json")
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            all_results = json.load(f)
        print(f"Loaded checkpoint with {len(all_results)} models completed")
    for model in models:
        done_benchmarks = len(all_results.get(model, {}).get("benchmarks", {}))
        if done_benchmarks >= len(benchmarks):
            print(f"\nSkipping {model} (already done)")
            continue
        print(f"\nAnalyzing {model}... ({done_benchmarks}/{len(benchmarks)} done)")
        all_results[model] = analyze_model(model, benchmarks, checkpoint_file, all_results, args.multi_layer)

    print("\nAggregating results...")
    summary = aggregate_results(all_results)

    print_summary(summary)
    plot_results(summary, args.output_dir)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "raw": all_results}, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
