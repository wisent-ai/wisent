#!/usr/bin/env python3
"""
Extract activations for ALL missing benchmarks for all models.
Designed to run on AWS with GPU.
"""

import argparse
import os
import struct
import time
import sys

import psycopg2
import torch

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]

if not DATABASE_URL:
    DATABASE_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'


def hidden_states_to_bytes(hidden_states: torch.Tensor) -> bytes:
    """Convert hidden_states tensor to bytes (float32) using numpy for speed."""
    import numpy as np
    arr = hidden_states.cpu().float().numpy()
    return arr.astype(np.float32).tobytes()


def get_db_connection():
    """Get database connection."""
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(
        db_url,
        connect_timeout=30,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
    conn.autocommit = True
    return conn


def get_missing_benchmarks(conn, model_id: int, target_pairs: int = 500) -> list:
    """Get list of benchmarks that need more extractions for this model.

    A benchmark is incomplete if it has fewer extracted pairs than:
    - target_pairs (default 500), OR
    - the total available pairs in the database (if less than target_pairs)

    Returns list of (set_id, name, pairs_needed) for incomplete benchmarks.
    """
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")  # 5 minute timeout for complex queries

    # Step 1: Get all benchmarks with pair counts (fast query)
    print("  Fetching benchmark pair counts...", flush=True)
    cur.execute('''
        SELECT cps.id, cps.name, COUNT(cp.id) as total_pairs
        FROM "ContrastivePairSet" cps
        INNER JOIN "ContrastivePair" cp ON cp."setId" = cps.id
        GROUP BY cps.id, cps.name
        HAVING COUNT(cp.id) > 0
        ORDER BY cps.name
    ''')
    benchmarks = cur.fetchall()
    print(f"  Found {len(benchmarks)} benchmarks with pairs", flush=True)

    # Step 2: For each benchmark, count extracted pairs (separate queries avoid timeout)
    missing = []
    complete = 0
    for i, (set_id, name, total_pairs) in enumerate(benchmarks):
        cur.execute('''
            SELECT COUNT(DISTINCT "contrastivePairId")
            FROM "Activation"
            WHERE "contrastivePairSetId" = %s AND "modelId" = %s
        ''', (set_id, model_id))
        extracted_pairs = cur.fetchone()[0]

        # Target is min of target_pairs or total available
        target = min(target_pairs, total_pairs)
        if extracted_pairs < target:
            pairs_needed = target - extracted_pairs
            missing.append((set_id, name, pairs_needed))
        else:
            complete += 1

        if (i + 1) % 50 == 0:
            print(f"  Checked {i + 1}/{len(benchmarks)} benchmarks...", flush=True)

    cur.close()
    print(f"Found {len(benchmarks)} benchmarks with pairs: {complete} complete, {len(missing)} need more extraction", flush=True)
    return missing


def get_or_create_pair(conn, set_id: int, prompt: str, positive: str, negative: str, pair_idx: int) -> int:
    """Get or create ContrastivePair."""
    cur = conn.cursor()

    cur.execute('''
        SELECT id FROM "ContrastivePair"
        WHERE "setId" = %s AND category = %s
    ''', (set_id, f"pair_{pair_idx}"))
    result = cur.fetchone()
    if result:
        cur.close()
        return result[0]

    positive_text = f"{prompt}\n\n{positive}"[:65000]
    negative_text = f"{prompt}\n\n{negative}"[:65000]

    cur.execute('''
        INSERT INTO "ContrastivePair" ("setId", "positiveExample", "negativeExample", "category", "createdAt", "updatedAt")
        VALUES (%s, %s, %s, %s, NOW(), NOW())
        RETURNING id
    ''', (set_id, positive_text, negative_text, f"pair_{pair_idx}"))
    pair_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return pair_id


def create_activation(conn, model_id: int, pair_id: int, set_id: int, layer: int,
                      activation_vec: torch.Tensor, is_positive: bool, strategy: str):
    """Create Activation record."""
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '120s'")  # 2 minute timeout for inserts

    neuron_count = activation_vec.shape[0]
    activation_bytes = hidden_states_to_bytes(activation_vec)

    cur.execute('''
        INSERT INTO "Activation"
        ("modelId", "contrastivePairId", "contrastivePairSetId", "layer", "neuronCount",
         "extractionStrategy", "activationData", "isPositive", "userId", "createdAt", "updatedAt")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'system', NOW(), NOW())
        ON CONFLICT DO NOTHING
    ''', (model_id, pair_id, set_id, layer, neuron_count, strategy,
          psycopg2.Binary(activation_bytes), is_positive))
    conn.commit()
    cur.close()


def extract_benchmark(model, tokenizer, model_id: int, benchmark_name: str, set_id: int,
                      conn, device: str, num_layers: int, limit: int = 500):
    """Extract activations for a single benchmark using EXISTING pairs from database.

    Only extracts pairs that don't already have activations for this model.
    """
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")  # 5 minute timeout for complex queries

    # Get pairs that DON'T already have activations for this model
    cur.execute('''
        SELECT cp.id, cp."positiveExample", cp."negativeExample"
        FROM "ContrastivePair" cp
        WHERE cp."setId" = %s
        AND NOT EXISTS (
            SELECT 1 FROM "Activation" a
            WHERE a."contrastivePairId" = cp.id AND a."modelId" = %s
        )
        ORDER BY cp.id
        LIMIT %s
    ''', (set_id, model_id, limit))
    db_pairs = cur.fetchall()
    cur.close()

    if not db_pairs:
        print(f"  All pairs already extracted for {benchmark_name}", flush=True)
        return 0

    print(f"  Extracting {len(db_pairs)} pairs (skipping already extracted)...", flush=True)
    extracted = 0

    for pair_idx, (pair_id, pos_text, neg_text) in enumerate(db_pairs):
        # pos_text and neg_text already contain the full example from database
        # Extract activations
        def get_hidden_states(text):
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.inference_mode():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            # Return last token hidden state for each layer
            return [out.hidden_states[i][0, -1, :] for i in range(1, len(out.hidden_states))]

        pos_hidden = get_hidden_states(pos_text)
        neg_hidden = get_hidden_states(neg_text)

        # Save all layers
        # Using "completion_last" because format is prompt\n\nresponse (not chat template)
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            create_activation(conn, model_id, pair_id, set_id, layer_num,
                              pos_hidden[layer_idx], True, "completion_last")
            create_activation(conn, model_id, pair_id, set_id, layer_num,
                              neg_hidden[layer_idx], False, "completion_last")

        del pos_hidden, neg_hidden
        extracted += 1

        if (pair_idx + 1) % 50 == 0:
            print(f"    Processed {pair_idx + 1}/{len(db_pairs)} pairs", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--limit", type=int, default=500, help="Max pairs per benchmark")
    parser.add_argument("--benchmark", default=None, help="Single benchmark to extract (optional)")
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to("mps")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers", flush=True)

    conn = get_db_connection()
    cur = conn.cursor()

    # Get model ID
    cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (args.model,))
    result = cur.fetchone()
    if not result:
        print(f"ERROR: Model {args.model} not found in database", flush=True)
        sys.exit(1)
    model_id = result[0]
    print(f"Model ID: {model_id}", flush=True)

    if args.benchmark:
        # Extract single benchmark
        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (args.benchmark,))
        result = cur.fetchone()
        if not result:
            print(f"ERROR: Benchmark {args.benchmark} not found", flush=True)
            sys.exit(1)
        set_id = result[0]

        print(f"Extracting single benchmark: {args.benchmark}", flush=True)
        extracted = extract_benchmark(model, tokenizer, model_id, args.benchmark, set_id,
                                       conn, args.device, num_layers, args.limit)
        print(f"Done! Extracted {extracted} pairs", flush=True)
    else:
        # Extract all incomplete benchmarks
        missing = get_missing_benchmarks(conn, model_id, args.limit)
        print(f"Found {len(missing)} incomplete benchmarks to extract", flush=True)

        if not missing:
            print("All benchmarks are complete!", flush=True)
            conn.close()
            return

        total_extracted = 0
        for i, (set_id, benchmark_name, pairs_needed) in enumerate(missing):
            print(f"\n[{i+1}/{len(missing)}] {benchmark_name} ({pairs_needed} pairs needed)", flush=True)
            start = time.time()

            extracted = extract_benchmark(model, tokenizer, model_id, benchmark_name, set_id,
                                           conn, args.device, num_layers, args.limit)

            total_extracted += extracted
            elapsed = time.time() - start
            print(f"  Extracted {extracted} pairs in {elapsed:.1f}s", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"COMPLETE! Total extracted: {total_extracted} pairs across {len(missing)} benchmarks", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
