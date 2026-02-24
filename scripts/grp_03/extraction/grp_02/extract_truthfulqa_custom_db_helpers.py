"""
Database helper functions for truthfulqa_custom extraction.

Contains connection management, model/pair CRUD, and activation storage functions
shared by extract_truthfulqa_custom_db.py.
"""

import struct
import os

import torch
import psycopg2
from wisent.core.constants import EXTRACTION_DB_BATCH_SIZE

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]  # Remove pgbouncer params


def get_db_connection():
    """Get a fresh database connection with proper settings."""
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    # Add TCP keepalive to prevent connection timeout during long forward passes
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


# Global connection that can be refreshed
_db_conn = None


def get_conn():
    """Get the current connection, reconnecting if needed."""
    global _db_conn
    if _db_conn is None:
        _db_conn = get_db_connection()
    else:
        # Test if connection is still alive
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


def get_or_create_pair(set_id: int, prompt: str, positive: str, negative: str, pair_idx: int) -> int:
    """Get or create ContrastivePair with auto-reconnect."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = get_conn()
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
            cur.close()
            return pair_id
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"  [DB error in get_or_create_pair attempt {attempt+1}/{max_retries}: {e}]", flush=True)
            reset_conn()
            if attempt == max_retries - 1:
                raise


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


def check_pair_fully_extracted(model_id: int, pair_id: int, num_layers: int, formats: list) -> bool:
    """Check if a pair has all activations for all formats already extracted."""
    expected_count = num_layers * 2 * len(formats)  # layers × polarities × formats
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

    # Split into smaller batches to avoid timeout (max 50 rows per batch)
    batch_size = EXTRACTION_DB_BATCH_SIZE
    max_retries = 3

    for i in range(0, len(activations_data), batch_size):
        batch = activations_data[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                conn = get_conn()
                cur = conn.cursor()
                # Set statement timeout to 60 seconds to prevent infinite hangs
                from psycopg2.extras import execute_values
                execute_values(cur, '''
                    INSERT INTO "RawActivation"
                    ("modelId", "contrastivePairId", "contrastivePairSetId", "layer", "seqLen", "hiddenDim", "promptLen", "hiddenStates", "answerText", "isPositive", "promptFormat", "createdAt")
                    VALUES %s
                    ON CONFLICT ("modelId", "contrastivePairId", layer, "isPositive", "promptFormat") DO NOTHING
                ''', batch, template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())")
                cur.close()
                break  # Success, move to next batch
            except (psycopg2.OperationalError, psycopg2.InterfaceError, psycopg2.errors.QueryCanceled) as e:
                print(f"  [DB batch error on attempt {attempt+1}/{max_retries}: {e}]", flush=True)
                reset_conn()
                if attempt == max_retries - 1:
                    raise


def create_raw_activation(
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
    """Create RawActivation record with automatic reconnection on failure."""
    seq_len = hidden_states.shape[0]
    hidden_dim = hidden_states.shape[1]
    hidden_bytes = hidden_states_to_bytes(hidden_states)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO "RawActivation"
                ("modelId", "contrastivePairId", "contrastivePairSetId", "layer", "seqLen", "hiddenDim", "promptLen", "hiddenStates", "answerText", "isPositive", "promptFormat", "createdAt")
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT ("modelId", "contrastivePairId", layer, "isPositive", "promptFormat") DO NOTHING
            ''', (model_id, pair_id, set_id, layer, seq_len, hidden_dim, prompt_len, psycopg2.Binary(hidden_bytes), answer_text, is_positive, prompt_format))
            cur.close()
            return
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            print(f"  [DB error on attempt {attempt+1}/{max_retries}: {e}]", flush=True)
            reset_conn()
            if attempt == max_retries - 1:
                raise


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

