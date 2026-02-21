#!/usr/bin/env python3
"""
Extract raw activations for truthfulqa_custom benchmark and save to database.
Modified from extract_all_benchmarks_db.py to extract only one benchmark.

Creates:
- ContrastivePairSet per benchmark
- ContrastivePair per pair
- RawActivation per layer/pair/side/promptFormat (positive/negative x chat/mc_balanced/role_play)

Extracts 3 prompt formats:
- chat: Standard chat template with Q+A
- mc_balanced: Multiple choice with balanced A/B assignment
- role_play: "Behave like person who answers Q with A" format
"""

import argparse
import time

import torch
import psycopg2

from extract_truthfulqa_custom_db_helpers import (
    get_conn,
    reset_conn,
    get_batch_size,
    hidden_states_to_bytes,
    get_or_create_model,
    get_or_create_pair_set,
    get_or_create_pair,
    check_pair_fully_extracted,
    batch_create_raw_activations,
    check_benchmark_done,
)
from extract_truthfulqa_custom_db_verify import verify_extraction


def extract_benchmark(model_name: str, benchmark: str = "truthfulqa_custom", device: str = "cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    from wisent.core.activations import ExtractionStrategy, build_extraction_texts

    # Initialize DB connection (uses auto-reconnect mechanism)
    conn = get_conn()

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s), using device={device}", flush=True)

    if device == "mps":
        # MPS doesn't support device_map
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # MPS works better with float32
            trust_remote_code=True,
        )
        model = model.to("mps")
    else:
        use_device_map = "auto" if num_gpus > 1 else device
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=use_device_map,
            trust_remote_code=True,
        )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    batch_size = get_batch_size(model.config)
    print(f"Model loaded: {num_layers} layers, hidden_dim={hidden_dim}, batch_size={batch_size}", flush=True)

    model_id = get_or_create_model(conn, model_name, num_layers)
    print(f"Model ID: {model_id}", flush=True)

    set_id = get_or_create_pair_set(conn, benchmark)
    print(f"Benchmark set ID: {set_id}", flush=True)

    # Generate pairs first to get pair count for completion check
    print(f"Generating contrastive pairs for {benchmark}...", flush=True)
    pairs = lm_build_contrastive_pairs(benchmark, None, limit=None)

    if not pairs:
        print("No pairs generated!", flush=True)
        conn.close()
        return

    pair_count = len(pairs)
    print(f"Generated {pair_count} pairs (batch size {batch_size})", flush=True)

    # All 3 prompt formats
    all_prompt_formats = [
        ("chat", ExtractionStrategy.CHAT_LAST),
        ("mc_balanced", ExtractionStrategy.MC_BALANCED),
        ("role_play", ExtractionStrategy.ROLE_PLAY),
    ]

    # Check which formats need extraction (now with proper completeness check)
    formats_to_extract = []
    for fmt, strategy in all_prompt_formats:
        if not check_benchmark_done(conn, model_id, set_id, fmt, num_layers, pair_count):
            formats_to_extract.append((fmt, strategy))

    if not formats_to_extract:
        print(f"SKIP (all formats exist for {model_name})", flush=True)
        conn.close()
        return

    format_names = [f[0] for f in formats_to_extract]
    print(f"Formats to extract: {format_names}", flush=True)
    start = time.time()
    skipped = 0
    extracted = 0

    # Forward pass helper (defined once, not per-pair)
    def get_hidden_states(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        return [out.hidden_states[i].squeeze(0) for i in range(1, len(out.hidden_states))]

    for batch_start in range(0, len(pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]

        for pair_idx_in_batch, pair in enumerate(batch_pairs):
            pair_idx = batch_start + pair_idx_in_batch

            prompt = pair.prompt
            pos = pair.positive_response.model_response if hasattr(pair.positive_response, 'model_response') else str(pair.positive_response)
            neg = pair.negative_response.model_response if hasattr(pair.negative_response, 'model_response') else str(pair.negative_response)

            # Create pair in DB (fast - just lookup or insert)
            pair_id = get_or_create_pair(set_id, prompt, pos, neg, pair_idx)

            # OPTIMIZATION: Skip if pair is fully extracted
            if check_pair_fully_extracted(model_id, pair_id, num_layers, format_names):
                skipped += 1
                if skipped % 50 == 0:
                    print(f"    [skipped {skipped} already-extracted pairs]", flush=True)
                continue

            print(f"    pair {pair_idx+1}/{len(pairs)}...", end="", flush=True)

            # Collect all activations for batch insert
            activations_batch = []

            # Extract for each prompt format that needs it
            for prompt_format, strategy in formats_to_extract:
                try:
                    # Build texts for this format
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
                    print(f"    Error building texts for {prompt_format}: {e}")
                    continue

                pos_prompt_len = len(tokenizer(pos_prompt_only, add_special_tokens=False)["input_ids"]) if pos_prompt_only else 0
                neg_prompt_len = len(tokenizer(neg_prompt_only, add_special_tokens=False)["input_ids"]) if neg_prompt_only else 0

                pos_hidden = get_hidden_states(pos_text)
                neg_hidden = get_hidden_states(neg_text)

                # Collect all layers for batch insert
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

            # Reset connection before batch insert to ensure fresh connection after long forward passes
            reset_conn()

            # Batch insert all activations for this pair
            batch_create_raw_activations(activations_batch)
            extracted += 1
            print(" done", flush=True)

        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"DONE in {time.time()-start:.1f}s (extracted {extracted}, skipped {skipped})", flush=True)
    reset_conn()
    print(f"\nExtraction complete for {benchmark} with {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--benchmark", default="truthfulqa_custom", help="Benchmark name")
    parser.add_argument("--device", default="cuda", help="Device (cuda or mps)")
    parser.add_argument("--verify-only", action="store_true", help="Only run verification, skip extraction")
    args = parser.parse_args()

    if args.verify_only:
        success = verify_extraction(args.model, args.benchmark)
        exit(0 if success else 1)
    else:
        extract_benchmark(args.model, args.benchmark, args.device)
        verify_extraction(args.model, args.benchmark)
