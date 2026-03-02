"""Database connection management for extract_raw_activations."""

from __future__ import annotations
import os
import psycopg2


DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is required")

_db_conn = None

# Preserved from original extract_raw_activations.py
_CONN_KW = {
    "connect_" + "timeout": 30,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}


def get_db_connection():
    """Get a fresh database connection."""
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(db_url, **_CONN_KW)
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
    """Get list of benchmarks missing raw activations for this model."""
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
    print(f"Found {len(benchmarks)} benchmarks, {len(benchmarks) - len(missing)} complete, {len(missing)} need extraction", flush=True)
    return missing
