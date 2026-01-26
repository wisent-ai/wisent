#!/usr/bin/env python3
"""Database functions for strategy extraction."""

import os
import numpy as np
import psycopg2
import torch

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]

if not DATABASE_URL:
    DATABASE_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'


def hidden_states_to_bytes(hidden_states: torch.Tensor) -> bytes:
    """Convert hidden_states tensor to bytes."""
    arr = hidden_states.cpu().float().numpy()
    return arr.astype(np.float32).tobytes()


def get_db_connection():
    """Get database connection with timeouts."""
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


def get_incomplete_benchmarks(conn, model_id: int) -> list:
    """Get benchmarks that have only 1 strategy and need all 7."""
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")

    cur.execute('''
        WITH strategy_counts AS (
            SELECT "contrastivePairSetId", "contrastivePairId",
                   COUNT(DISTINCT "extractionStrategy") as num_strategies
            FROM "Activation"
            WHERE "modelId" = %s
            GROUP BY "contrastivePairSetId", "contrastivePairId"
        )
        SELECT DISTINCT sc."contrastivePairSetId", cps.name,
               COUNT(DISTINCT sc."contrastivePairId") as incomplete_pairs
        FROM strategy_counts sc
        JOIN "ContrastivePairSet" cps ON cps.id = sc."contrastivePairSetId"
        WHERE sc.num_strategies = 1
        GROUP BY sc."contrastivePairSetId", cps.name
        ORDER BY incomplete_pairs DESC
    ''', (model_id,))

    results = cur.fetchall()
    cur.close()
    return results


def get_pairs_needing_strategies(conn, model_id: int, set_id: int, limit: int = 500) -> list:
    """Get pairs that have only 1 strategy and need the other 6."""
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")

    cur.execute('''
        SELECT cp.id, cp."positiveExample", cp."negativeExample"
        FROM "ContrastivePair" cp
        WHERE cp."setId" = %s
        AND cp.id IN (
            SELECT "contrastivePairId"
            FROM "Activation"
            WHERE "modelId" = %s AND "contrastivePairSetId" = %s
            GROUP BY "contrastivePairId"
            HAVING COUNT(DISTINCT "extractionStrategy") = 1
        )
        ORDER BY cp.id
        LIMIT %s
    ''', (set_id, model_id, set_id, limit))

    pairs = cur.fetchall()
    cur.close()
    return pairs


def create_activation(conn, model_id: int, pair_id: int, set_id: int, layer: int,
                      activation_vec: torch.Tensor, is_positive: bool, strategy: str):
    """Create Activation record in database."""
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '120s'")

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


def get_model_id(conn, model_name: str) -> int:
    """Get model ID from database by HuggingFace ID."""
    cur = conn.cursor()
    cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    result = cur.fetchone()
    cur.close()
    if not result:
        raise ValueError(f"Model {model_name} not found in database")
    return result[0]


def get_benchmark_id(conn, benchmark_name: str) -> int:
    """Get benchmark set ID from database by name."""
    cur = conn.cursor()
    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (benchmark_name,))
    result = cur.fetchone()
    cur.close()
    if not result:
        raise ValueError(f"Benchmark {benchmark_name} not found in database")
    return result[0]
