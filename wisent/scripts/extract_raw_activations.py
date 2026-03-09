#!/usr/bin/env python3
"""
Extract raw activations for ALL missing benchmarks with 3 prompt formats.

This script:
1. Finds all benchmarks that have contrastive pairs in the database
2. Checks which benchmarks are missing raw activations for the given model
3. Extracts using 3 formats: chat, mc_balanced, role_play
4. Stores to RawActivation table (full sequence hidden states)

Extracts up to 500 pairs per benchmark (or maximum available).

Usage:
    python3 -m wisent.scripts.extract_raw_activations --model meta-llama/Llama-3.2-1B-Instruct
    python3 -m wisent.scripts.extract_raw_activations --model Qwen/Qwen3-8B --benchmark knowledge_qa/mmlu
"""

import argparse
import os
import sys
import time

print("[STARTUP] Starting extract_raw_activations.py...", flush=True)
print(f"[STARTUP] Python version: {sys.version}", flush=True)

print("[STARTUP] Importing psycopg2...", flush=True)
import psycopg2
print("[STARTUP] psycopg2 imported", flush=True)

print("[STARTUP] Importing torch...", flush=True)
import torch
print(f"[STARTUP] torch imported, version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)

from wisent.scripts._helpers.extract_raw_helpers import extract_benchmark
from wisent.scripts._helpers.extract_raw_db import (
    get_conn, reset_conn, get_or_create_model, get_missing_benchmarks,
)


def main():
    print("[MAIN] Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser(description="Extract raw activations for all missing benchmarks with 3 formats")
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--device", required=True, help="Device (cuda/mps/cpu)")
    parser.add_argument("--benchmark", default=None, help="Single benchmark to extract (optional)")
    parser.add_argument("--max-retries", type=int, required=True, help="Maximum retry attempts for DB operations")
    parser.add_argument("--log-interval", type=int, required=True, help="Progress logging interval")
    args = parser.parse_args()
    print(f"[MAIN] Args: model={args.model}, device={args.device}, benchmark={args.benchmark}", flush=True)

    print("[MAIN] Importing transformers...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("[MAIN] transformers imported", flush=True)

    print(f"[MAIN] Loading tokenizer for {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[MAIN] Tokenizer loaded, vocab_size={tokenizer.vocab_size}", flush=True)

    print(f"[MAIN] Loading model {args.model}...", flush=True)
    if args.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.float32, trust_remote_code=True)
        model = model.to("mps")
        actual_device = "mps"
    else:
        num_gpus = torch.cuda.device_count()
        print(f"[MAIN] Detected {num_gpus} GPUs", flush=True)
        use_device_map = "auto" if num_gpus > 1 else args.device
        print(f"[MAIN] Using device_map={use_device_map}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map=use_device_map, trust_remote_code=True)
        actual_device = next(model.parameters()).device
        print(f"[MAIN] Model device: {actual_device}", flush=True)
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"[MAIN] Model loaded: {num_layers} layers, device={actual_device}", flush=True)

    # Store actual device for use in extraction
    model._actual_device = str(actual_device)

    print("[MAIN] Connecting to database...", flush=True)
    conn = get_conn()
    print("[MAIN] Database connected", flush=True)

    model_id = get_or_create_model(conn, args.model, num_layers)
    print(f"[MAIN] Model ID: {model_id}", flush=True)

    if args.benchmark:
        cur = conn.cursor()
        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (args.benchmark,))
        result = cur.fetchone()
        if not result:
            print(f"ERROR: Benchmark {args.benchmark} not found", flush=True)
            return
        set_id = result[0]
        cur.close()

        print(f"\nExtracting single benchmark: {args.benchmark}", flush=True)
        extracted = extract_benchmark(model, tokenizer, model_id, args.benchmark, set_id,
                                       num_layers, args.device, get_conn, reset_conn, max_retries=args.max_retries, log_interval=args.log_interval)
        print(f"\nDone! Extracted {extracted} pairs", flush=True)
    else:
        missing = get_missing_benchmarks(conn, model_id, num_layers)
        print(f"\nFound {len(missing)} benchmarks needing extraction", flush=True)

        if not missing:
            print("All benchmarks are fully extracted!", flush=True)
            return

        total_extracted = 0
        for i, (set_id, benchmark_name, pair_count) in enumerate(missing):
            print(f"\n[{i+1}/{len(missing)}] {benchmark_name} ({pair_count} pairs in DB)", flush=True)
            start = time.time()

            extracted = extract_benchmark(model, tokenizer, model_id, benchmark_name, set_id,
                                           num_layers, args.device, get_conn, reset_conn, max_retries=args.max_retries, log_interval=args.log_interval)

            total_extracted += extracted
            elapsed = time.time() - start
            print(f"  Completed in {elapsed:.1f}s", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"COMPLETE! Total extracted: {total_extracted} pairs across {len(missing)} benchmarks", flush=True)


if __name__ == "__main__":
    main()
