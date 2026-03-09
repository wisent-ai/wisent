"""Helper functions for extract_raw_activations: extraction and DB batch operations."""

from __future__ import annotations
import struct

import psycopg2
from psycopg2.extras import execute_values
import torch

from wisent.core.utils.config_tools.constants import PROGRESS_LOG_INTERVAL_10, RECURSION_INITIAL_DEPTH


def hidden_states_to_bytes(hidden_states: torch.Tensor) -> bytes:
    """Convert hidden_states tensor to bytes (float32)."""
    flat = hidden_states.cpu().float().flatten().tolist()
    return struct.pack(f'{len(flat)}f', *flat)


def get_batch_size(model_config) -> int:
    """Auto-adjust batch size based on model size."""
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


def check_pair_fully_extracted(get_conn_fn, model_id: int, pair_id: int,
                                num_layers: int, formats: list) -> bool:
    """Check if a pair has all raw activations for all formats."""
    expected_count = num_layers * 2 * len(formats)
    try:
        conn = get_conn_fn()
        cur = conn.cursor()
        cur.execute('''
            SELECT COUNT(*) FROM "RawActivation"
            WHERE "modelId" = %s AND "contrastivePairId" = %s
        ''', (model_id, pair_id))
        actual_count = cur.fetchone()[0]
        cur.close()
        return actual_count >= expected_count
    except Exception:
        return False


def batch_create_raw_activations(get_conn_fn, reset_conn_fn, activations_data: list, max_retries: int, batch_size: int = None):
    """Batch insert multiple RawActivation records."""
    if not activations_data:
        return

    if batch_size is None:
        raise ValueError("batch_size is required for batch_create_raw_activations")

    for i in range(0, len(activations_data), batch_size):
        batch = activations_data[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                conn = get_conn_fn()
                cur = conn.cursor()
                execute_values(cur, '''
                    INSERT INTO "RawActivation"
                    ("modelId", "contrastivePairId", "contrastivePairSetId", "layer", "seqLen", "hiddenDim", "promptLen", "hiddenStates", "answerText", "isPositive", "promptFormat", "createdAt")
                    VALUES %s
                    ON CONFLICT ("modelId", "contrastivePairId", layer, "isPositive", "promptFormat") DO NOTHING
                ''', batch, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())")
                cur.close()
                break
            except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.errors.QueryCanceled) as e:
                print(f"  [DB batch error attempt {attempt+1}/{max_retries}: {e}]", flush=True)
                reset_conn_fn()
                if attempt == max_retries - 1:
                    raise


def extract_benchmark(model, tokenizer, model_id: int, benchmark_name: str, set_id: int,
                      num_layers: int, device: str, get_conn_fn, reset_conn_fn, max_retries: int, log_interval: int):
    """Extract raw activations for a single benchmark."""
    print(f"  [EXTRACT] Importing extraction strategy...", flush=True)
    from wisent.core.primitives.model_interface.core.activations import ExtractionStrategy, build_extraction_texts
    print(f"  [EXTRACT] Extraction strategy imported", flush=True)

    actual_device = getattr(model, '_actual_device', device)
    print(f"  [EXTRACT] Using device: {actual_device}", flush=True)

    print(f"  [EXTRACT] Fetching pairs from database...", flush=True)
    conn = get_conn_fn()

    cur = conn.cursor()
    cur.execute('''
        SELECT id, "positiveExample", "negativeExample", category
        FROM "ContrastivePair"
        WHERE "setId" = %s
        ORDER BY id
    ''', (set_id,))
    db_pairs = cur.fetchall()
    cur.close()
    print(f"  [EXTRACT] Fetched {len(db_pairs)} pairs from database", flush=True)

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
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length, add_special_tokens=False)
        enc = {k: v.to(actual_device) for k, v in enc.items()}
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        return [out.hidden_states[i].squeeze(0) for i in range(1, len(out.hidden_states))]

    extracted = 0
    skipped = 0

    for pair_idx, (pair_id, pos_example, neg_example, category) in enumerate(db_pairs):
        if pair_idx == 0:
            print(f"  [EXTRACT] Processing first pair (id={pair_id})...", flush=True)

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

        if check_pair_fully_extracted(get_conn_fn, model_id, pair_id, num_layers, format_names):
            skipped += 1
            if skipped % log_interval == RECURSION_INITIAL_DEPTH:
                print(f"    [skipped {skipped} already-extracted pairs]", flush=True)
            continue

        activations_batch = []

        for prompt_format, strategy in all_prompt_formats:
            try:
                if strategy == ExtractionStrategy.MC_BALANCED:
                    pos_text, pos_answer, pos_prompt_only = build_extraction_texts(
                        strategy, prompt, pos, tokenizer, other_response=neg, is_positive=True)
                    neg_text, neg_answer, neg_prompt_only = build_extraction_texts(
                        strategy, prompt, neg, tokenizer, other_response=pos, is_positive=False)
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

        reset_conn_fn()
        batch_create_raw_activations(get_conn_fn, reset_conn_fn, activations_batch, max_retries=max_retries)
        extracted += 1

        if (pair_idx + 1) % PROGRESS_LOG_INTERVAL_10 == 0:
            print(f"    Processed {pair_idx + 1}/{len(db_pairs)} pairs", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Done: extracted {extracted}, skipped {skipped}", flush=True)
    return extracted
