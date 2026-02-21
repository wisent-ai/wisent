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
    ./run_on_aws.sh python3 scripts/extract_all_missing_raw.py --model meta-llama/Llama-3.2-1B-Instruct
    ./run_on_aws.sh python3 scripts/extract_all_missing_raw.py --model Qwen/Qwen3-8B --benchmark knowledge_qa/mmlu
"""

import argparse
import time

import psycopg2
import torch

from extract_all_missing_raw_helpers import (
    get_conn,
    reset_conn,
    hidden_states_to_bytes,
    get_or_create_model,
    get_missing_benchmarks,
    check_pair_fully_extracted,
    batch_create_raw_activations,
)


def extract_benchmark(model, tokenizer, model_id: int, benchmark_name: str, set_id: int,
                      num_layers: int, device: str, limit: int = 500):
    """Extract raw activations for a single benchmark using 3 formats."""
    from wisent.core.activations import ExtractionStrategy, build_extraction_texts

    conn = get_conn()

    cur = conn.cursor()
    cur.execute('''
        SELECT id, "positiveExample", "negativeExample", category
        FROM "ContrastivePair"
        WHERE "setId" = %s
        ORDER BY id
        LIMIT %s
    ''', (set_id, limit))
    db_pairs = cur.fetchall()
    cur.close()

    if not db_pairs:
        print(f"  No pairs in database for {benchmark_name}", flush=True)
        return 0

    print(f"  Processing {len(db_pairs)} pairs with 3 formats...", flush=True)

    all_prompt_formats = [
        ("chat", ExtractionStrategy.CHAT_LAST),
        ("mc_balanced", ExtractionStrategy.MC_BALANCED),
        ("role_play", ExtractionStrategy.ROLE_PLAY),
    ]
    format_names = [f[0] for f in all_prompt_formats]

    def get_hidden_states(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        return [out.hidden_states[i].squeeze(0) for i in range(1, len(out.hidden_states))]

    extracted = 0
    skipped = 0

    for pair_idx, (pair_id, pos_example, neg_example, category) in enumerate(db_pairs):
        if "\n\n" in pos_example:
            prompt = pos_example.rsplit("\n\n", 1)[0]
            pos = pos_example.rsplit("\n\n", 1)[1]
        else:
            prompt = pos_example
            pos = ""

        if "\n\n" in neg_example:
            neg = neg_example.rsplit("\n\n", 1)[1]
        else:
            neg = neg_example

        if check_pair_fully_extracted(model_id, pair_id, num_layers, format_names):
            skipped += 1
            if skipped % 50 == 0:
                print(f"    [skipped {skipped} already-extracted pairs]", flush=True)
            continue

        activations_batch = []

        for prompt_format, strategy in all_prompt_formats:
            try:
                if strategy == ExtractionStrategy.MC_BALANCED:
                    pos_text, pos_answer, pos_prompt_only = build_extraction_texts(
                        strategy, prompt, pos, tokenizer, other_response=neg, is_positive=True
                    )
                    neg_text, neg_answer, neg_prompt_only = build_extraction_texts(
                        strategy, prompt, neg, tokenizer, other_response=pos, is_positive=False
                    )
                else:
                    pos_text, pos_answer, pos_prompt_only = build_extraction_texts(strategy, prompt, pos, tokenizer)
                    neg_text, neg_answer, neg_prompt_only = build_extraction_texts(strategy, prompt, neg, tokenizer)
            except Exception as e:
                print(f"    Error building texts for {prompt_format}: {e}", flush=True)
                continue

            pos_prompt_len = len(tokenizer(pos_prompt_only, add_special_tokens=False)["input_ids"]) if pos_prompt_only else 0
            neg_prompt_len = len(tokenizer(neg_prompt_only, add_special_tokens=False)["input_ids"]) if neg_prompt_only else 0

            pos_hidden = get_hidden_states(pos_text)
            neg_hidden = get_hidden_states(neg_text)

            for layer_idx in range(num_layers):
                layer_num = layer_idx + 1
                pos_bytes = hidden_states_to_bytes(pos_hidden[layer_idx])
                neg_bytes = hidden_states_to_bytes(neg_hidden[layer_idx])

                activations_batch.append((
                    model_id, pair_id, set_id, layer_num,
                    pos_hidden[layer_idx].shape[0], pos_hidden[layer_idx].shape[1],
                    pos_prompt_len, psycopg2.Binary(pos_bytes), pos_answer, True, prompt_format
                ))
                activations_batch.append((
                    model_id, pair_id, set_id, layer_num,
                    neg_hidden[layer_idx].shape[0], neg_hidden[layer_idx].shape[1],
                    neg_prompt_len, psycopg2.Binary(neg_bytes), neg_answer, False, prompt_format
                ))

            del pos_hidden, neg_hidden

        reset_conn()
        batch_create_raw_activations(activations_batch)
        extracted += 1

        if (pair_idx + 1) % 10 == 0:
            print(f"    Processed {pair_idx + 1}/{len(db_pairs)} pairs", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Done: extracted {extracted}, skipped {skipped}", flush=True)
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract raw activations for all missing benchmarks with 3 formats")
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--limit", type=int, default=500, help="Max pairs per benchmark (default: 500)")
    parser.add_argument("--benchmark", default=None, help="Single benchmark to extract (optional)")
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
        num_gpus = torch.cuda.device_count()
        use_device_map = "auto" if num_gpus > 1 else args.device
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map=use_device_map, trust_remote_code=True,
        )
    model.eval()

    num_layers = model.config.num_hidden_layers
    print(f"Model loaded: {num_layers} layers", flush=True)

    conn = get_conn()
    model_id = get_or_create_model(conn, args.model, num_layers)
    print(f"Model ID: {model_id}", flush=True)

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
                                       num_layers, args.device, args.limit)
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
                                           num_layers, args.device, args.limit)

            total_extracted += extracted
            elapsed = time.time() - start
            print(f"  Completed in {elapsed:.1f}s", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"COMPLETE! Total extracted: {total_extracted} pairs across {len(missing)} benchmarks", flush=True)


if __name__ == "__main__":
    main()
