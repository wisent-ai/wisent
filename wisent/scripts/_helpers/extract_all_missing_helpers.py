"""
Benchmark extraction and main entry point for extract_all_missing.

Split from extract_all_missing.py to meet 300-line limit.
"""

import argparse
import sys
import time

import psycopg2
import torch

from wisent.core.utils.config_tools.constants import EXTRACTION_DEFAULT_PAIR_LIMIT, EXTRACTION_SMALL_BATCH_SIZE, PROGRESS_LOG_INTERVAL

from wisent.scripts.extract_all_missing import (
    hidden_states_to_bytes,
    get_conn,
    reset_conn,
    batch_create_activations,
    get_missing_benchmarks,
)


def extract_benchmark(model, tokenizer, model_id: int, benchmark_name: str, set_id: int,
                      device: str, num_layers: int, limit: int = EXTRACTION_DEFAULT_PAIR_LIMIT):
    """Extract activations for a single benchmark using EXISTING pairs from database.

    Only extracts pairs that don't already have activations for this model.
    """
    conn = get_conn()
    cur = conn.cursor()

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

    def get_hidden_states(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        # Return last token hidden state for each layer
        return [out.hidden_states[i][0, -1, :] for i in range(1, len(out.hidden_states))]

    # Process in batches of 10 pairs to reduce DB round trips
    batch_size = EXTRACTION_SMALL_BATCH_SIZE
    for batch_start in range(0, len(db_pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(db_pairs))
        batch_pairs = db_pairs[batch_start:batch_end]

        activations_batch = []
        for pair_id, pos_text, neg_text in batch_pairs:
            pos_hidden = get_hidden_states(pos_text)
            neg_hidden = get_hidden_states(neg_text)

            # Collect all layers for this pair
            for layer_idx in range(num_layers):
                layer_num = layer_idx + 1
                pos_bytes = hidden_states_to_bytes(pos_hidden[layer_idx])
                neg_bytes = hidden_states_to_bytes(neg_hidden[layer_idx])
                neuron_count = pos_hidden[layer_idx].shape[0]

                activations_batch.append((
                    model_id, pair_id, set_id, layer_num, neuron_count,
                    "chat_last", psycopg2.Binary(pos_bytes), True
                ))
                activations_batch.append((
                    model_id, pair_id, set_id, layer_num, neuron_count,
                    "chat_last", psycopg2.Binary(neg_bytes), False
                ))

            del pos_hidden, neg_hidden
            extracted += 1

        # Batch insert all activations for this batch of pairs
        batch_create_activations(activations_batch)

        if batch_end % PROGRESS_LOG_INTERVAL == 0 or batch_end == len(db_pairs):
            print(f"    Processed {batch_end}/{len(db_pairs)} pairs", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--limit", type=int, default=EXTRACTION_DEFAULT_PAIR_LIMIT, help="Max pairs per benchmark")
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

    conn = get_conn()
    cur = conn.cursor()

    # Get model ID
    cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (args.model,))
    result = cur.fetchone()
    if not result:
        print(f"ERROR: Model {args.model} not found in database", flush=True)
        sys.exit(1)
    model_id = result[0]
    cur.close()
    print(f"Model ID: {model_id}", flush=True)

    if args.benchmark:
        # Extract single benchmark
        conn = get_conn()
        cur = conn.cursor()
        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (args.benchmark,))
        result = cur.fetchone()
        cur.close()
        if not result:
            print(f"ERROR: Benchmark {args.benchmark} not found", flush=True)
            sys.exit(1)
        set_id = result[0]

        print(f"Extracting single benchmark: {args.benchmark}", flush=True)
        extracted = extract_benchmark(model, tokenizer, model_id, args.benchmark, set_id,
                                       args.device, num_layers, args.limit)
        print(f"Done! Extracted {extracted} pairs", flush=True)
    else:
        # Extract all incomplete benchmarks
        missing = get_missing_benchmarks(get_conn(), model_id, args.limit)
        print(f"Found {len(missing)} incomplete benchmarks to extract", flush=True)

        if not missing:
            print("All benchmarks are complete!", flush=True)
            reset_conn()
            return

        total_extracted = 0
        for i, (set_id, benchmark_name, pairs_needed) in enumerate(missing):
            print(f"\n[{i+1}/{len(missing)}] {benchmark_name} ({pairs_needed} pairs needed)", flush=True)
            start = time.time()

            extracted = extract_benchmark(model, tokenizer, model_id, benchmark_name, set_id,
                                           args.device, num_layers, args.limit)

            total_extracted += extracted
            elapsed = time.time() - start
            print(f"  Extracted {extracted} pairs in {elapsed:.1f}s", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"COMPLETE! Total extracted: {total_extracted} pairs across {len(missing)} benchmarks", flush=True)

    reset_conn()
