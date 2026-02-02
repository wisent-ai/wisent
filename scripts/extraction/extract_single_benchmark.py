#!/usr/bin/env python3
"""
Extract activations for a single benchmark and save to database.
"""

import argparse
import os
import struct
import time

import psycopg2
import torch

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]


def hidden_states_to_bytes(hidden_states: torch.Tensor) -> bytes:
    """Convert hidden_states tensor to bytes (float32)."""
    flat = hidden_states.cpu().float().flatten().tolist()
    return struct.pack(f'{len(flat)}f', *flat)


def get_or_create_model(conn, model_name: str, num_layers: int) -> int:
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


def check_activation_exists(conn, model_id: int, pair_id: int, layer: int, strategy: str, is_positive: bool) -> bool:
    """Check if Activation already exists."""
    cur = conn.cursor()
    cur.execute('''
        SELECT 1 FROM "Activation"
        WHERE "modelId" = %s AND "contrastivePairId" = %s AND layer = %s AND "extractionStrategy" = %s AND "isPositive" = %s
    ''', (model_id, pair_id, layer, strategy, is_positive))
    exists = cur.fetchone() is not None
    cur.close()
    return exists


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


def extract_single_benchmark(model_name: str, benchmark: str, limit: int = 200, device: str = "cuda"):
    """Extract activations for a single benchmark."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import lm_build_contrastive_pairs
    from wisent.core.activations import ExtractionStrategy

    conn = psycopg2.connect(DATABASE_URL)

    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"Model loaded: {num_layers} layers, hidden_dim={hidden_dim}")

    model_id = get_or_create_model(conn, model_name, num_layers)
    print(f"Model ID: {model_id}")

    set_id = get_or_create_pair_set(conn, benchmark)
    print(f"Benchmark set ID: {set_id}")

    # Generate pairs
    print(f"Generating contrastive pairs for {benchmark}...")
    pairs = lm_build_contrastive_pairs(benchmark, None, limit=limit)
    print(f"Generated {len(pairs)} pairs")

    if not pairs:
        print("No pairs generated!")
        return

    # Extraction strategies to use
    strategies = [
        ("chat_last", ExtractionStrategy.CHAT_LAST),
    ]

    for pair_idx, pair in enumerate(pairs):
        print(f"[{pair_idx+1}/{len(pairs)}] Processing pair...", flush=True)

        prompt = pair.prompt
        pos = pair.positive_response.model_response if hasattr(pair.positive_response, 'model_response') else str(pair.positive_response)
        neg = pair.negative_response.model_response if hasattr(pair.negative_response, 'model_response') else str(pair.negative_response)

        pair_id = get_or_create_pair(conn, set_id, prompt, pos, neg, pair_idx)

        # Build chat texts
        pos_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": pos}]
        neg_messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": neg}]

        pos_text = tokenizer.apply_chat_template(pos_messages, tokenize=False, add_generation_prompt=False)
        neg_text = tokenizer.apply_chat_template(neg_messages, tokenize=False, add_generation_prompt=False)

        for strategy_name, strategy in strategies:
            # Check if already exists for middle layer
            mid_layer = num_layers // 2
            if check_activation_exists(conn, model_id, pair_id, mid_layer, strategy_name, True):
                print(f"  Skipping (already exists)")
                continue

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
                                  pos_hidden[layer_idx], True, strategy_name)
                create_activation(conn, model_id, pair_id, set_id, layer_num,
                                  neg_hidden[layer_idx], False, strategy_name)

            del pos_hidden, neg_hidden

        torch.cuda.empty_cache()

    conn.close()
    print(f"\nExtraction complete for {benchmark}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--benchmark", required=True, help="Benchmark name")
    parser.add_argument("--limit", type=int, default=200, help="Max pairs")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    extract_single_benchmark(args.model, args.benchmark, args.limit, args.device)
