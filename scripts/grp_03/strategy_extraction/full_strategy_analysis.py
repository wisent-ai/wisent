#!/usr/bin/env python3
"""Full extraction strategy analysis - parallel workers with batch layer queries."""
import argparse, json, struct, os, sys, multiprocessing
from wisent.core.constants import COMPARISON_MAX_BATCH_SIZE
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*lbfgs failed to converge.*")
import numpy as np
import psycopg2
from database import get_available_layers, compute_layer_aggregates
from analyze_checkpoint import (
    compute_full_metrics, verify_benchmark_completion,
    aggregate_results, plot_results, print_summary)

DB_CONFIG = {"host": "aws-0-eu-west-2.pooler.supabase.com", "port": 5432,
    "database": "postgres", "user": "postgres.rbqjqnouluslojmmnuqi",
    "password": "REDACTED_DB_PASSWORD",
    "options": "-c statement_timeout=0"}
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


def get_model_id(conn, model_name: str):
    cur = conn.cursor()
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    result = cur.fetchone()
    cur.close()
    return (result[0], result[1]) if result else (None, None)


def make_connection(max_retries=5):
    """Create a DB connection with TCP keepalives and no statement_timeout."""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(
                **DB_CONFIG, keepalives=1, keepalives_idle=30,
                keepalives_interval=10, keepalives_count=5)
            conn.autocommit = True
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            cur.close()
            return conn
        except psycopg2.OperationalError as e:
            print(f"    Connection attempt {attempt+1}/{max_retries} failed: {e}", flush=True)
            if attempt == max_retries - 1:
                raise


def load_single_layer(conn, model_id: int, layer: int, benchmark: str) -> Dict:
    """Load activations for a single layer. Returns {strategy: {pos, neg}}."""
    cur = conn.cursor()
    cur.execute('''
        SELECT a."contrastivePairId", a."extractionStrategy", a."activationData", a."isPositive"
        FROM "Activation" a
        JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
        WHERE a."modelId" = %s AND a.layer = %s AND cps.name = %s
    ''', (model_id, layer, benchmark))
    rows = cur.fetchall()
    cur.close()
    raw = defaultdict(dict)
    for pair_id, strategy, data, is_pos in rows:
        raw[strategy][(pair_id, "pos" if is_pos else "neg")] = bytes_to_vector(data)
    result = {}
    for strategy, pairs in raw.items():
        pos_acts, neg_acts = [], []
        for pid in set(p[0] for p in pairs.keys()):
            if (pid, "pos") in pairs and (pid, "neg") in pairs:
                pos_acts.append(pairs[(pid, "pos")])
                neg_acts.append(pairs[(pid, "neg")])
        if len(pos_acts) >= 5:
            result[strategy] = {"pos": np.array(pos_acts), "neg": np.array(neg_acts)}
    return result


def _run_benchmark(conn, model_id, num_layers, benchmark, multi_layer):
    """Core benchmark processing logic. Returns dict, 'NO_DATA' sentinel, or None on error."""
    if multi_layer:
        layers = get_available_layers(conn, model_id, benchmark)
        if not layers:
            return "NO_DATA"
        strat_per_layer = defaultdict(dict)
        for layer in layers:
            acts = load_single_layer(conn, model_id, layer, benchmark)
            for strategy, data in acts.items():
                if len(data["pos"]) >= 10:
                    strat_per_layer[strategy][layer] = compute_full_metrics(data["pos"], data["neg"])
        bench_results = {}
        for strategy, per_layer in strat_per_layer.items():
            if per_layer:
                bench_results[strategy] = compute_layer_aggregates(per_layer)
    else:
        acts = load_single_layer(conn, model_id, num_layers // 2, benchmark)
        if not acts:
            return "NO_DATA"
        bench_results = {}
        for strategy, data in acts.items():
            if len(data["pos"]) >= 10:
                bench_results[strategy] = compute_full_metrics(data["pos"], data["neg"])
    return bench_results if bench_results else "NO_DATA"


def process_benchmark(args, max_retries=3):
    """Worker: process one (model, benchmark) pair. Retries on connection errors."""
    model_name, model_id, num_layers, benchmark, multi_layer = args
    for attempt in range(max_retries):
        try:
            conn = make_connection()
        except Exception as e:
            print(f"    CONN FAILED [{model_name}] {benchmark} (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            if attempt == max_retries - 1:
                return (model_name, benchmark, None)
            continue
        try:
            result = _run_benchmark(conn, model_id, num_layers, benchmark, multi_layer)
            conn.close()
            return (model_name, benchmark, result)
        except psycopg2.OperationalError as e:
            print(f"    DB ERROR [{model_name}] {benchmark} (attempt {attempt+1}/{max_retries}): {e}", flush=True)
            try: conn.close()
            except: pass
            if attempt == max_retries - 1:
                return (model_name, benchmark, None)
        except Exception as e:
            print(f"    ERROR [{model_name}] {benchmark}: {e}", flush=True)
            try: conn.close()
            except: pass
            return (model_name, benchmark, None)
    return (model_name, benchmark, None)


def load_checkpoints(models, output_dir):
    """Load all per-model checkpoint files."""
    all_results = {}
    for model in models:
        ms = model.replace("/", "_")
        cp = os.path.join(output_dir, f"checkpoint_{ms}.json")
        if os.path.exists(cp):
            with open(cp) as f:
                all_results.update(json.load(f))
    return all_results


def build_tasks(models, model_info, benchmarks, all_results, multi_layer):
    """Build flat task list from remaining benchmarks."""
    tasks = []
    for model in models:
        if model not in model_info:
            continue
        mid, nlayers = model_info[model]
        done = set(all_results.get(model, {}).get("benchmarks", {}).keys())
        remaining = [b for b in benchmarks if b not in done]
        print(f"  {model}: {len(done)} done, {len(remaining)} remaining")
        for b in remaining:
            tasks.append((model, mid, nlayers, b, multi_layer))
    return tasks


def _save_checkpoint(model_name, all_results, output_dir):
    ms = model_name.replace("/", "_")
    cp = os.path.join(output_dir, f"checkpoint_{ms}.json")
    with open(cp, "w") as f:
        json.dump({model_name: all_results[model_name]}, f, default=str)


def run_batch(tasks, workers, models, model_info, all_results, output_dir):
    """Run a batch of tasks. Returns (completed, failed) counts."""
    completed, failed = 0, 0
    with multiprocessing.Pool(workers) as pool:
        for model_name, benchmark, bench_results in pool.imap_unordered(process_benchmark, tasks):
            if bench_results is None:
                failed += 1
                continue
            if model_name not in all_results:
                all_results[model_name] = {"benchmarks": {}, "num_layers": model_info[model_name][1]}
            if bench_results == "NO_DATA":
                all_results[model_name]["benchmarks"][benchmark] = {
                    "category": get_benchmark_category(benchmark), "no_data": True, "strategies": {}}
                _save_checkpoint(model_name, all_results, output_dir)
                print(f"    NO_DATA [{model_name}] {benchmark}", flush=True)
                completed += 1
                continue
            all_results[model_name]["benchmarks"][benchmark] = {
                "category": get_benchmark_category(benchmark), "strategies": bench_results}
            if not verify_benchmark_completion(all_results[model_name], benchmark, model_name):
                del all_results[model_name]["benchmarks"][benchmark]
                failed += 1
                continue
            completed += 1
            _save_checkpoint(model_name, all_results, output_dir)
            if completed % 5 == 0:
                counts = {m: len(all_results.get(m, {}).get("benchmarks", {})) for m in models}
                print(f"  [{completed} done, {failed} failed] {counts}", flush=True)
    return completed, failed


def main():
    parser = argparse.ArgumentParser(description="Full extraction strategy analysis")
    parser.add_argument("--output-dir", default="strategy_analysis_results")
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--multi-layer", action="store_true")
    parser.add_argument("--benchmark", help="Single benchmark")
    parser.add_argument("--workers", type=int, default=COMPARISON_MAX_BATCH_SIZE)
    args = parser.parse_args()
    models = args.models or MODELS
    if args.multi_layer:
        print("Multi-layer mode: per-layer queries per benchmark")
    conn = make_connection()
    benchmarks = [args.benchmark] if args.benchmark else get_benchmarks(conn)
    print(f"Found {len(benchmarks)} benchmarks")
    model_info = {}
    for model in models:
        mid, nlayers = get_model_id(conn, model)
        if mid:
            model_info[model] = (mid, nlayers)
        else:
            print(f"  WARNING: Model {model} not found in DB")
    conn.close()
    os.makedirs(args.output_dir, exist_ok=True)
    total_completed, total_failed = 0, 0
    while True:
        all_results = load_checkpoints(models, args.output_dir)
        tasks = build_tasks(models, model_info, benchmarks, all_results, args.multi_layer)
        if not tasks:
            print("All tasks complete!", flush=True)
            break
        print(f"\nBatch: {len(tasks)} tasks with {args.workers} workers...", flush=True)
        try:
            c, f = run_batch(tasks, args.workers, models, model_info, all_results, args.output_dir)
            total_completed += c
            total_failed += f
            if c == 0 and f == len(tasks):
                print("All remaining tasks failed. Stopping.", flush=True)
                break
        except Exception as e:
            print(f"\nPool crashed: {e}. Restarting...", flush=True)
            continue
    # Final aggregation
    all_results = load_checkpoints(models, args.output_dir)
    print(f"\nTotal completed: {total_completed}, Total failed: {total_failed}")
    cp = os.path.join(args.output_dir, "checkpoint.json")
    with open(cp, "w") as f:
        json.dump(all_results, f, default=str)
    summary = aggregate_results(all_results)
    print_summary(summary)
    plot_results(summary, args.output_dir)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "raw": all_results}, f, indent=2, default=str)
    print(f"\nFull results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()
