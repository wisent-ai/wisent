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
import struct
import sys
import time

print("[STARTUP] Starting extract_raw_activations.py...", flush=True)
print(f"[STARTUP] Python version: {sys.version}", flush=True)

print("[STARTUP] Importing psycopg2...", flush=True)
import psycopg2
from psycopg2.extras import execute_values
print("[STARTUP] psycopg2 imported", flush=True)

print("[STARTUP] Importing torch...", flush=True)
import torch
print(f"[STARTUP] torch imported, version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}", flush=True)

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]

if not DATABASE_URL:
    DATABASE_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'

_db_conn = None


def get_db_connection():
    """Get a fresh database connection."""
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


def get_conn():
    """Get current connection, reconnecting if needed."""
    global _db_conn
    if _db_conn is None:
        _db_conn = get_db_connection()
    else:
        try:
            cur = _db_conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
        except Exception:
            print("  [Reconnecting to DB...]", flush=True)
            try:
                _db_conn.close()
            except Exception:
                pass
            _db_conn = get_db_connection()
    return _db_conn


def reset_conn():
    """Force reconnection on next get_conn() call."""
    global _db_conn
    if _db_conn is not None:
        try:
            _db_conn.close()
        except Exception:
            pass
        _db_conn = None


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


def get_missing_benchmarks(conn, model_id: int, num_layers: int) -> list:
    """Get list of benchmarks missing raw activations for this model.

    A benchmark is considered complete if it has RawActivation entries for:
    - All 3 prompt formats (chat, mc_balanced, role_play)
    - At least 95% completeness

    Returns list of (set_id, name, pair_count) for incomplete benchmarks.
    """
    cur = conn.cursor()

    cur.execute('''
        SELECT cps.id, cps.name, COUNT(cp.id) as pair_count
        FROM "ContrastivePairSet" cps
        INNER JOIN "ContrastivePair" cp ON cp."setId" = cps.id
        GROUP BY cps.id, cps.name
        HAVING COUNT(cp.id) > 0
        ORDER BY cps.name
    ''')
    benchmarks = cur.fetchall()

    missing = []
    for set_id, name, pair_count in benchmarks:
        expected_per_format = pair_count * num_layers * 2
        threshold = int(expected_per_format * 0.95)

        formats_complete = 0
        for fmt in ['chat', 'mc_balanced', 'role_play']:
            cur.execute('''
                SELECT COUNT(*) FROM "RawActivation"
                WHERE "modelId" = %s AND "contrastivePairSetId" = %s AND "promptFormat" = %s
            ''', (model_id, set_id, fmt))
            count = cur.fetchone()[0]
            if count >= threshold:
                formats_complete += 1

        if formats_complete < 3:
            missing.append((set_id, name, pair_count))

    cur.close()
    print(f"Found {len(benchmarks)} benchmarks with pairs, {len(benchmarks) - len(missing)} complete, {len(missing)} need extraction", flush=True)
    return missing


def check_pair_fully_extracted(model_id: int, pair_id: int, num_layers: int, formats: list) -> bool:
    """Check if a pair has all raw activations for all formats."""
    expected_count = num_layers * 2 * len(formats)
    try:
        conn = get_conn()
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


def batch_create_raw_activations(activations_data: list):
    """Batch insert multiple RawActivation records."""
    if not activations_data:
        return

    batch_size = 50
    max_retries = 3

    for i in range(0, len(activations_data), batch_size):
        batch = activations_data[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                conn = get_conn()
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
                reset_conn()
                if attempt == max_retries - 1:
                    raise


def extract_benchmark(model, tokenizer, model_id: int, benchmark_name: str, set_id: int,
                      num_layers: int, device: str, limit: int = 500):
    """Extract raw activations for a single benchmark using 3 formats."""
    print(f"  [EXTRACT] Importing extraction strategy...", flush=True)
    from wisent.core.activations import ExtractionStrategy, build_extraction_texts
    print(f"  [EXTRACT] Extraction strategy imported", flush=True)

    # Get actual device from model if available
    actual_device = getattr(model, '_actual_device', device)
    print(f"  [EXTRACT] Using device: {actual_device}", flush=True)

    print(f"  [EXTRACT] Fetching pairs from database...", flush=True)
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
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False)
        # Use the actual device where the model is
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

        if pair_idx == 0:
            print(f"  [EXTRACT] Checking if pair already extracted...", flush=True)

        if check_pair_fully_extracted(model_id, pair_id, num_layers, format_names):
            skipped += 1
            if skipped % 50 == 0:
                print(f"    [skipped {skipped} already-extracted pairs]", flush=True)
            continue

        if pair_idx == 0:
            print(f"  [EXTRACT] Pair not extracted, starting extraction...", flush=True)

        activations_batch = []

        for prompt_format, strategy in all_prompt_formats:
            if pair_idx == 0:
                print(f"  [EXTRACT] Building texts for {prompt_format}...", flush=True)
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

            if pair_idx == 0:
                print(f"  [EXTRACT] Running model inference for {prompt_format}...", flush=True)

            pos_hidden = get_hidden_states(pos_text)
            if pair_idx == 0:
                print(f"  [EXTRACT] Positive hidden states: {len(pos_hidden)} layers, shape={pos_hidden[0].shape}", flush=True)
            neg_hidden = get_hidden_states(neg_text)
            if pair_idx == 0:
                print(f"  [EXTRACT] Negative hidden states: {len(neg_hidden)} layers", flush=True)

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

        if pair_idx == 0:
            print(f"  [EXTRACT] First pair complete, inserting {len(activations_batch)} activation records to DB...", flush=True)

        reset_conn()
        batch_create_raw_activations(activations_batch)
        extracted += 1

        if pair_idx == 0:
            print(f"  [EXTRACT] First pair saved to database!", flush=True)

        if (pair_idx + 1) % 10 == 0:
            print(f"    Processed {pair_idx + 1}/{len(db_pairs)} pairs", flush=True)

    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"  Done: extracted {extracted}, skipped {skipped}", flush=True)
    return extracted


def main():
    print("[MAIN] Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser(description="Extract raw activations for all missing benchmarks with 3 formats")
    parser.add_argument("--model", required=True, help="Model name (e.g., meta-llama/Llama-3.2-1B-Instruct)")
    parser.add_argument("--device", default="cuda", help="Device (cuda/mps/cpu)")
    parser.add_argument("--limit", type=int, default=500, help="Max pairs per benchmark (default: 500)")
    parser.add_argument("--benchmark", default=None, help="Single benchmark to extract (optional)")
    args = parser.parse_args()
    print(f"[MAIN] Args: model={args.model}, device={args.device}, limit={args.limit}, benchmark={args.benchmark}", flush=True)

    print("[MAIN] Importing transformers...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("[MAIN] transformers imported", flush=True)

    print(f"[MAIN] Loading tokenizer for {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[MAIN] Tokenizer loaded, vocab_size={tokenizer.vocab_size}", flush=True)

    print(f"[MAIN] Loading model {args.model}...", flush=True)
    if args.device == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model = model.to("mps")
        actual_device = "mps"
    else:
        num_gpus = torch.cuda.device_count()
        print(f"[MAIN] Detected {num_gpus} GPUs", flush=True)
        use_device_map = "auto" if num_gpus > 1 else args.device
        print(f"[MAIN] Using device_map={use_device_map}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype="auto",
            device_map=use_device_map,
            trust_remote_code=True,
        )
        # Get actual device from model
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
