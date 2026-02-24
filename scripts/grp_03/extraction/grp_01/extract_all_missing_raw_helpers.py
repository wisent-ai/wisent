"""
Database helper functions for extract_all_missing_raw.py.

Contains connection management, model/pair CRUD, activation storage,
and batch insert utilities for the RawActivation table.
"""

import os
import psycopg2
from psycopg2.extras import execute_values
import torch
from wisent.core.constants import EXTRACTION_RAW_BATCH_SIZE

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
    """Convert hidden_states tensor to bytes (float32) using numpy for speed."""
    import numpy as np
    arr = hidden_states.cpu().float().numpy()
    return arr.astype(np.float32).tobytes()


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
    """Batch insert multiple RawActivation records.

    Uses small batches (5 records) and long timeout (5 min) because each record
    contains ~10MB of binary data.
    """
    if not activations_data:
        return

    batch_size = EXTRACTION_RAW_BATCH_SIZE  # Small batch - each record is ~10MB
    max_retries = 3

    total_batches = (len(activations_data) + batch_size - 1) // batch_size
    for batch_idx, i in enumerate(range(0, len(activations_data), batch_size)):
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
                if (batch_idx + 1) % 4 == 0 or batch_idx == total_batches - 1:
                    print(f"      [DB insert batch {batch_idx+1}/{total_batches}]", flush=True)
                break
            except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.errors.QueryCanceled) as e:
                print(f"  [DB batch error attempt {attempt+1}/{max_retries}: {e}]", flush=True)
                reset_conn()
                if attempt == max_retries - 1:
                    raise
