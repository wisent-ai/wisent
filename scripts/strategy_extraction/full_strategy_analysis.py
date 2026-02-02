#!/usr/bin/env python3
"""Full extraction strategy analysis with comprehensive diagnostics."""
import argparse, json, struct, os
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

DB_CONFIG = {"host": "aws-0-eu-west-2.pooler.supabase.com", "port": 5432,
    "database": "postgres", "user": "postgres.rbqjqnouluslojmmnuqi",
    "password": "BsKuEnPFLCFurN4a", "options": "-c statement_timeout=120000"}
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


def analyze_model(model_name: str, benchmarks: List[str], checkpoint_file: str, all_results: Dict) -> Dict:
    conn = psycopg2.connect(**DB_CONFIG)
    model_id, num_layers = get_model_id(conn, model_name)
    if model_id is None:
        print(f"  Model {model_name} not found")
        conn.close()
        return {}

    layer = num_layers // 2
    # Load existing partial results for this model if any
    existing = all_results.get(model_name, {})
    results = {"benchmarks": existing.get("benchmarks", {}),
               "strategies": defaultdict(lambda: defaultdict(list))}
    # Rebuild strategies from existing benchmarks
    for bench, bdata in results["benchmarks"].items():
        for strat, metrics in bdata.get("strategies", {}).items():
            for k, v in metrics.items():
                results["strategies"][strat][k].append(float(v) if isinstance(v, str) else v)

    for i, benchmark in enumerate(benchmarks):
        if benchmark in results["benchmarks"]:
            continue  # Already done
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(benchmarks)} benchmarks...", flush=True)
        try:
            acts = load_benchmark_activations(conn, model_id, layer, benchmark)
            if not acts:
                continue
            bench_results = {}
            for strategy, data in acts.items():
                if len(data["pos"]) < 10:
                    continue
                metrics = compute_full_metrics(data["pos"], data["neg"])
                bench_results[strategy] = metrics
                for k, v in metrics.items():
                    results["strategies"][strategy][k].append(v)
            results["benchmarks"][benchmark] = {
                "category": get_benchmark_category(benchmark), "strategies": bench_results}
            # Checkpoint after each benchmark
            all_results[model_name] = results
            with open(checkpoint_file, "w") as f:
                json.dump(all_results, f, default=str)
        except psycopg2.OperationalError:
            # Reconnect on connection errors
            try:
                conn.close()
            except:
                pass
            conn = psycopg2.connect(**DB_CONFIG)
            model_id, _ = get_model_id(conn, model_name)
            continue
        except Exception as e:
            try:
                conn.rollback()
            except:
                pass
            continue
    conn.close()
    return results


def aggregate_results(all_results: Dict) -> Dict:
    agg = {"by_strategy": defaultdict(lambda: defaultdict(list)),
        "by_category": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "by_model": {}}

    for model, data in all_results.items():
        model_metrics = {}
        for strategy, metrics in data.get("strategies", {}).items():
            if metrics.get("consistency"):
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
        if metrics.get("consistency"):
            summary["strategies"][strat] = {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in metrics.items() if v}

    for cat, strats in agg["by_category"].items():
        summary["categories"][cat] = {}
        for strat, metrics in strats.items():
            if metrics.get("consistency"):
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
    if not summary["strategies"]:
        print("No data collected"); return
    print(f"\n{'Strategy':<20} {'LinAcc':>8} {'Consist':>8} {'SteerAcc':>9}\n" + "-" * 50)
    for strat, m in sorted(summary["strategies"].items(), key=lambda x: x[1].get("consistency", {}).get("mean", 0), reverse=True):
        print(f"  {strat:<18} {m.get('linear_acc', {}).get('mean', 0):>8.3f} {m.get('consistency', {}).get('mean', 0):>8.3f} {m.get('steer_acc', {}).get('mean', 0):>9.3f}")


def main():
    parser = argparse.ArgumentParser(description="Full extraction strategy analysis")
    parser.add_argument("--output-dir", default="strategy_analysis_results")
    parser.add_argument("--models", nargs="*", default=None, help="Models to analyze")
    args = parser.parse_args()

    models = args.models if args.models else MODELS

    print("Connecting to database...")
    conn = psycopg2.connect(**DB_CONFIG)
    print("Getting benchmark list...")
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
        all_results[model] = analyze_model(model, benchmarks, checkpoint_file, all_results)

    print("\nAggregating results...")
    summary = aggregate_results(all_results)

    print_summary(summary)
    plot_results(summary, args.output_dir)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "raw": all_results}, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
