#!/usr/bin/env python3
"""
Extract raw activations for truthfulqa_custom benchmark and save to database.
Modified from extract_all_benchmarks_db.py to extract only one benchmark.

Creates:
- ContrastivePairSet per benchmark
- ContrastivePair per pair
- RawActivation per layer/pair/side/promptFormat (positive/negative × chat/mc_balanced/role_play)

Extracts 3 prompt formats:
- chat: Standard chat template with Q+A
- mc_balanced: Multiple choice with balanced A/B assignment
- role_play: "Behave like person who answers Q with A" format
"""

import argparse
import time
import struct
import os
from typing import Dict, List

import torch
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]  # Remove pgbouncer params


def get_batch_size(model_config) -> int:
    """Auto-adjust batch size based on model size to avoid OOM."""
    num_params_b = getattr(model_config, 'num_parameters', None)
    if num_params_b is None:
        hidden = model_config.hidden_size
        layers = model_config.num_hidden_layers
        num_params_b = (12 * hidden * hidden * layers) / 1e9

    if num_params_b < 2:
        return 10
    elif num_params_b < 3:
        return 5
    elif num_params_b < 5:
        return 2
    else:
        return 1


def hidden_states_to_bytes(hidden_states: torch.Tensor) -> bytes:
    """Convert hidden_states tensor to bytes (float32)."""
    flat = hidden_states.cpu().float().flatten().tolist()
    return struct.pack(f'{len(flat)}f', *flat)


def get_or_create_model(conn, model_name: str, num_layers: int = 32) -> int:
    """Get or create model in database."""
    cur = conn.cursor()
    cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    result = cur.fetchone()
    if result:
        cur.close()
        return result[0]

    optimal_layer = num_layers // 2
    cur.execute('''
        INSERT INTO "Model" ("name", "huggingFaceId", "userTag", "assistantTag", "userId", "isPublic", "numLayers", "optimalLayer", "createdAt", "updatedAt")
        VALUES (%s, %s, 'user', 'assistant', 'system', true, %s, %s, NOW(), NOW())
        RETURNING id
    ''', (model_name.split('/')[-1], model_name, num_layers, optimal_layer))
    model_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return model_id


def get_or_create_pair_set(conn, benchmark: str) -> int:
    """Get or create ContrastivePairSet."""
    cur = conn.cursor()

    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (benchmark,))
    result = cur.fetchone()
    if result:
        cur.close()
        return result[0]

    cur.execute('''
        INSERT INTO "ContrastivePairSet" ("name", "description", "userId", "isPublic", "createdAt", "updatedAt")
        VALUES (%s, %s, 'system', true, NOW(), NOW())
        RETURNING id
    ''', (benchmark, f"Benchmark: {benchmark}"))
    set_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return set_id


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


def check_activation_exists(conn, model_id: int, pair_id: int, layer: int, is_positive: bool, prompt_format: str) -> bool:
    """Check if RawActivation already exists."""
    cur = conn.cursor()
    cur.execute('''
        SELECT 1 FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairId" = %s AND layer = %s AND "isPositive" = %s AND "promptFormat" = %s
    ''', (model_id, pair_id, layer, is_positive, prompt_format))
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def create_raw_activation(
    conn,
    model_id: int,
    pair_id: int,
    set_id: int,
    layer: int,
    hidden_states: torch.Tensor,
    prompt_len: int,
    answer_text: str,
    is_positive: bool,
    prompt_format: str = "chat"
):
    """Create RawActivation record."""
    cur = conn.cursor()

    seq_len = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    hidden_bytes = hidden_states_to_bytes(hidden_states)

    cur.execute('''
        INSERT INTO "RawActivation"
        ("modelId", "contrastivePairId", "contrastivePairSetId", "layer", "seqLen", "hiddenDim", "promptLen", "hiddenStates", "answerText", "isPositive", "promptFormat", "createdAt")
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        ON CONFLICT ("modelId", "contrastivePairId", layer, "isPositive", "promptFormat") DO NOTHING
    ''', (model_id, pair_id, set_id, layer, seq_len, hidden_dim, prompt_len, psycopg2.Binary(hidden_bytes), answer_text, is_positive, prompt_format))
    conn.commit()
    cur.close()


def check_benchmark_done(conn, model_id: int, set_id: int, prompt_format: str, num_layers: int, pair_count: int) -> bool:
    """Check if benchmark already has ALL activations for a specific prompt format.

    Expected count = pair_count * num_layers * 2 (positive + negative).
    Requires at least 95% completeness to skip.
    """
    cur = conn.cursor()
    cur.execute('''
        SELECT COUNT(*) FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s AND "promptFormat" = %s
    ''', (model_id, set_id, prompt_format))
    count = cur.fetchone()[0]
    cur.close()

    expected = pair_count * num_layers * 2
    threshold = int(expected * 0.95)  # Require 95% completeness
    return count >= threshold


def extract_benchmark(model_name: str, benchmark: str = "truthfulqa_custom", device: str = "cuda"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    from wisent.core.activations.extraction_strategy import ExtractionStrategy, build_extraction_texts

    # Supabase transaction pooler (port 6543) may reassign backends after idle
    # Use session pooler (port 5432) to maintain same backend connection
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET default_transaction_read_only = off")
    cur.close()

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

    print(f"Formats to extract: {[f[0] for f in formats_to_extract]}", flush=True)
    start = time.time()

    for batch_start in range(0, len(pairs), batch_size):
        batch_end = min(batch_start + batch_size, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]

        for pair_idx_in_batch, pair in enumerate(batch_pairs):
            pair_idx = batch_start + pair_idx_in_batch

            prompt = pair.prompt
            pos = pair.positive_response.model_response if hasattr(pair.positive_response, 'model_response') else str(pair.positive_response)
            neg = pair.negative_response.model_response if hasattr(pair.negative_response, 'model_response') else str(pair.negative_response)

            # Create pair in DB
            pair_id = get_or_create_pair(conn, set_id, prompt, pos, neg, pair_idx)

            # Forward pass helper
            def get_hidden_states(text):
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.inference_mode():
                    out = model(**enc, output_hidden_states=True, use_cache=False)
                return [out.hidden_states[i].squeeze(0) for i in range(1, len(out.hidden_states))]

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

                # Save all layers for this format
                for layer_idx in range(num_layers):
                    layer_num = layer_idx + 1

                    create_raw_activation(
                        conn=conn,
                        model_id=model_id,
                        pair_id=pair_id,
                        set_id=set_id,
                        layer=layer_num,
                        hidden_states=pos_hidden[layer_idx],
                        prompt_len=pos_prompt_len,
                        answer_text=pos_answer,
                        is_positive=True,
                        prompt_format=prompt_format
                    )

                    create_raw_activation(
                        conn=conn,
                        model_id=model_id,
                        pair_id=pair_id,
                        set_id=set_id,
                        layer=layer_num,
                        hidden_states=neg_hidden[layer_idx],
                        prompt_len=neg_prompt_len,
                        answer_text=neg_answer,
                        is_positive=False,
                        prompt_format=prompt_format
                    )

                del pos_hidden, neg_hidden

        if device == "cuda":
            torch.cuda.empty_cache()
        print(f"  batch {batch_start//batch_size + 1}/{(len(pairs) + batch_size - 1)//batch_size} saved (pairs {batch_start+1}-{batch_end})", flush=True)

    print(f"DONE in {time.time()-start:.1f}s", flush=True)
    conn.close()
    print(f"\nExtraction complete for {benchmark} with {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--benchmark", default="truthfulqa_custom", help="Benchmark name")
    parser.add_argument("--device", default="cuda", help="Device (cuda or mps)")
    args = parser.parse_args()

    extract_benchmark(args.model, args.benchmark, args.device)
