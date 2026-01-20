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


def hidden_states_to_bytes(hidden_states: torch.Tensor) -> bytes:
    """Convert hidden_states tensor to bytes (float32)."""
    flat = hidden_states.cpu().float().flatten().tolist()
    return struct.pack(f'{len(flat)}f', *flat)


def get_db_connection():
    """Get database connection."""
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def get_missing_benchmarks(conn, model_id: int) -> list:
    """Get list of benchmark IDs missing for this model."""
    cur = conn.cursor()

    # Get all benchmarks
    cur.execute('SELECT id, name FROM "ContrastivePairSet" ORDER BY name')
    all_benchmarks = {row[0]: row[1] for row in cur.fetchall()}

    # Get benchmarks that have extractions for this model
    cur.execute('''
        SELECT DISTINCT "contrastivePairSetId"
        FROM "Activation"
        WHERE "modelId" = %s AND "contrastivePairSetId" IS NOT NULL
    ''', (model_id,))
    covered_ids = {row[0] for row in cur.fetchall()}

    # Return missing (id, name) pairs
    missing = [(bid, all_benchmarks[bid]) for bid in all_benchmarks.keys() if bid not in covered_ids]
    cur.close()
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
                      conn, device: str, num_layers: int, limit: int = 200):
    """Extract activations for a single benchmark using EXISTING pairs from database."""
    cur = conn.cursor()

    # Get existing pairs from database for this benchmark set
    cur.execute('''
        SELECT id, "positiveExample", "negativeExample"
        FROM "ContrastivePair"
        WHERE "setId" = %s
        ORDER BY id
        LIMIT %s
    ''', (set_id, limit))
    db_pairs = cur.fetchall()
    cur.close()

    if not db_pairs:
        print(f"  No existing pairs in database for {benchmark_name}", flush=True)
        return 0

    print(f"  Processing {len(db_pairs)} existing pairs from database...", flush=True)
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
        for layer_idx in range(num_layers):
            layer_num = layer_idx + 1
            create_activation(conn, model_id, pair_id, set_id, layer_num,
                              pos_hidden[layer_idx], True, "chat_last")
            create_activation(conn, model_id, pair_id, set_id, layer_num,
                              neg_hidden[layer_idx], False, "chat_last")

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
    parser.add_argument("--limit", type=int, default=200, help="Max pairs per benchmark")
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
        # Extract all missing benchmarks
        missing = get_missing_benchmarks(conn, model_id)
        print(f"Found {len(missing)} missing benchmarks", flush=True)

        total_extracted = 0
        for i, (set_id, benchmark_name) in enumerate(missing):
            print(f"\n[{i+1}/{len(missing)}] {benchmark_name}", flush=True)
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
