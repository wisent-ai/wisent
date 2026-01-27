#!/usr/bin/env python3
"""
Extract all 7 strategies for benchmarks that don't have complete extraction.

Finds all benchmarks where some pairs are missing all 7 strategies and runs
extraction for them. Uses ON CONFLICT DO NOTHING to safely skip existing data.

Usage:
    python scripts/run_strategy_extraction.py --model meta-llama/Llama-3.2-1B-Instruct
    python scripts/run_strategy_extraction.py --model Qwen/Qwen3-8B --device cuda
"""

import argparse
import os
import sys
import time

import torch

# Add the repo root to path so we can import from scripts.strategy_extraction
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.insert(0, repo_root)

from scripts.strategy_extraction import (
    get_db_connection,
    get_incomplete_benchmarks,
    get_pairs_needing_strategies,
    extract_pair_all_strategies,
)
from scripts.strategy_extraction.database import get_model_id, get_benchmark_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--limit", type=int, default=500, help="Max pairs per benchmark")
    parser.add_argument("--benchmark", default=None, help="Single benchmark (optional)")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True,
        )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers", flush=True)

    conn = get_db_connection()
    model_id = get_model_id(conn, args.model)
    print(f"Model ID: {model_id}", flush=True)

    if args.benchmark:
        set_id = get_benchmark_id(conn, args.benchmark)
        benchmarks = [(set_id, args.benchmark, 1)]
    else:
        benchmarks = get_incomplete_benchmarks(conn, model_id)
        print(f"Found {len(benchmarks)} benchmarks needing extraction", flush=True)

    total_extracted = 0
    for i, (set_id, name, incomplete_count) in enumerate(benchmarks):
        print(f"\n[{i+1}/{len(benchmarks)}] {name} ({incomplete_count} pairs)", flush=True)
        start = time.time()

        pairs = get_pairs_needing_strategies(conn, model_id, set_id, args.limit)
        if not pairs:
            print(f"  No pairs need extraction", flush=True)
            continue

        print(f"  Extracting {len(pairs)} pairs with all 7 strategies...", flush=True)

        for pair_idx, (pair_id, pos_text, neg_text) in enumerate(pairs):
            extract_pair_all_strategies(
                model, tokenizer, args.device, model_id,
                pair_id, set_id, pos_text, neg_text, conn, num_layers
            )
            total_extracted += 1

            if (pair_idx + 1) % 10 == 0:
                print(f"    Processed {pair_idx + 1}/{len(pairs)} pairs", flush=True)

        elapsed = time.time() - start
        print(f"  Done in {elapsed:.1f}s", flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"COMPLETE! Extracted strategies for {total_extracted} pairs", flush=True)
    conn.close()


if __name__ == "__main__":
    main()
