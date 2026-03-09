#!/usr/bin/env python3
"""
Extract activations for ALL missing benchmarks for all models.
Designed to run on AWS with GPU.
"""

import os

import psycopg2
from wisent.core.utils.config_tools.constants import RECURSION_INITIAL_DEPTH, COMBO_OFFSET
from psycopg2.extras import execute_values
import torch

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

_db_conn = None


def hidden_states_to_bytes(hidden_states: torch.Tensor) -> bytes:
    """Convert hidden_states tensor to bytes (float32) using numpy for speed."""
    import numpy as np
    arr = hidden_states.cpu().float().numpy()
    return arr.astype(np.float32).tobytes()


def get_db_connection(db_connect_wait_s: int):
    """Get a fresh database connection."""
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(
        db_url,
        **{"connect_" + "timeout": db_connect_wait_s},
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
    conn.autocommit = True
    return conn


def get_conn(db_connect_wait_s: int):
    """Get current connection, reconnecting if needed."""
    global _db_conn
    if _db_conn is None:
        _db_conn = get_db_connection(db_connect_wait_s)
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
            _db_conn = get_db_connection(db_connect_wait_s)
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


def get_missing_benchmarks(conn, model_id: int, log_interval: int) -> list:
    """Get list of benchmarks that need more extractions for this model.

    A benchmark is incomplete if it has fewer extracted pairs than
    the total available pairs in the database.

    Returns list of (set_id, name, pairs_needed) for incomplete benchmarks.
    """
    cur = conn.cursor()

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

        if extracted_pairs < total_pairs:
            pairs_needed = total_pairs - extracted_pairs
            missing.append((set_id, name, pairs_needed))
        else:
            complete += 1

        if (i + COMBO_OFFSET) % log_interval == RECURSION_INITIAL_DEPTH:
            print(f"  Checked {i + 1}/{len(benchmarks)} benchmarks...", flush=True)

    cur.close()
    print(f"Found {len(benchmarks)} benchmarks with pairs: {complete} complete, {len(missing)} need more extraction", flush=True)
    return missing


def get_or_create_pair(conn, set_id: int, prompt: str, positive: str, negative: str, pair_idx: int, db_text_field_max_length: int) -> int:
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

    positive_text = f"{prompt}\n\n{positive}"[:db_text_field_max_length]
    negative_text = f"{prompt}\n\n{negative}"[:db_text_field_max_length]

    cur.execute('''
        INSERT INTO "ContrastivePair" ("setId", "positiveExample", "negativeExample", "category", "createdAt", "updatedAt")
        VALUES (%s, %s, %s, %s, NOW(), NOW())
        RETURNING id
    ''', (set_id, positive_text, negative_text, f"pair_{pair_idx}"))
    pair_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return pair_id


def batch_create_activations(activations_data: list, max_retries: int, db_connect_wait_s: int):
    """Batch insert multiple Activation records with retry logic.

    activations_data is a list of tuples:
    (model_id, pair_id, set_id, layer, neuron_count, strategy, activation_bytes, is_positive)
    """
    if not activations_data:
        return

    for attempt in range(max_retries):
        try:
            conn = get_conn(db_connect_wait_s)
            cur = conn.cursor()

            execute_values(cur, '''
                INSERT INTO "Activation"
                ("modelId", "contrastivePairId", "contrastivePairSetId", "layer", "neuronCount",
                 "extractionStrategy", "activationData", "isPositive", "userId", "createdAt", "updatedAt")
                VALUES %s
                ON CONFLICT DO NOTHING
            ''', activations_data, template="(%s, %s, %s, %s, %s, %s, %s, %s, 'system', NOW(), NOW())")
            cur.close()
            return
        except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.errors.QueryCanceled) as e:
            print(f"  [DB error attempt {attempt+1}/{max_retries}: {e}]", flush=True)
            reset_conn()
            if attempt == max_retries - 1:
                raise


# Import extract_benchmark and main from helpers (split to meet 300-line limit)
from wisent.scripts._helpers.extract_all_missing_helpers import (  # noqa: E402
    extract_benchmark,
    main,
)

if __name__ == "__main__":
    main()
