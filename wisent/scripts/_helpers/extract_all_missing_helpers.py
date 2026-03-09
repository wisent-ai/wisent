"""
Benchmark extraction and main entry point for extract_all_missing.

Split from extract_all_missing.py to meet 300-line limit.
"""

import argparse
import sys
import time

import psycopg2
import torch

from wisent.core.utils.config_tools.constants import RECURSION_INITIAL_DEPTH

from wisent.scripts.extract_all_missing import (
    hidden_states_to_bytes,
    get_conn,
    reset_conn,
    batch_create_activations,
    get_missing_benchmarks,
)


def extract_benchmark(model, tokenizer, model_id: int, benchmark_name: str, set_id: int,
                      device: str, num_layers: int, batch_size: int,
                      db_connect_wait_s: int, max_retries: int,
                      log_interval: int):
    """Extract activations for a single benchmark using EXISTING pairs from database.

    Only extracts pairs that don't already have activations for this model.
    """
    conn = get_conn(db_connect_wait_s)
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
    ''', (set_id, model_id))
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

    # Process in batches to reduce DB round trips
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
        batch_create_activations(activations_batch, max_retries=max_retries, db_connect_wait_s=db_connect_wait_s)

        if batch_end % log_interval == RECURSION_INITIAL_DEPTH or batch_end == len(db_pairs):
            print(f"    Processed {batch_end}/{len(db_pairs)} pairs", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    return extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--device", required=True, help="Device (cuda/mps/cpu)")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for extraction (number of pairs per DB round trip)")
    parser.add_argument("--benchmark", default=None, help="Single benchmark to extract (optional)")
    parser.add_argument("--db-connect-wait-s", type=int, required=True, help="Database connection wait seconds")
    parser.add_argument("--max-retries", type=int, required=True, help="Maximum retry attempts for DB operations")
    parser.add_argument("--log-interval", type=int, required=True, help="Progress logging interval")
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

    conn = get_conn(args.db_connect_wait_s)
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
        conn = get_conn(args.db_connect_wait_s)
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
                                       args.device, num_layers, args.batch_size,
                                       db_connect_wait_s=args.db_connect_wait_s, max_retries=args.max_retries,
                                       log_interval=args.log_interval)
        print(f"Done! Extracted {extracted} pairs", flush=True)
    else:
        # Extract all incomplete benchmarks
        missing = get_missing_benchmarks(get_conn(args.db_connect_wait_s), model_id, log_interval=args.log_interval)
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
                                           args.device, num_layers, args.batch_size,
                                           db_connect_wait_s=args.db_connect_wait_s, max_retries=args.max_retries,
                                           log_interval=args.log_interval)

            total_extracted += extracted
            elapsed = time.time() - start
            print(f"  Extracted {extracted} pairs in {elapsed:.1f}s", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"COMPLETE! Total extracted: {total_extracted} pairs across {len(missing)} benchmarks", flush=True)

    reset_conn()
