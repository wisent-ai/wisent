#!/usr/bin/env python3
"""Database functions for strategy extraction."""

import os
import numpy as np
import psycopg2
from psycopg2.extras import execute_values
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
    """Get ALL benchmarks where some pairs don't have all 7 strategies.

    This includes:
    - Benchmarks with pairs that have 0 strategies (no activations at all)
    - Benchmarks with pairs that have 1-6 strategies (partial extraction)

    Returns list of (set_id, name, incomplete_pairs_count).
    """
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '0'")

    cur.execute('''
        WITH benchmark_totals AS (
            -- All benchmarks with their total pair counts (capped at 500)
            SELECT
                cps.id as set_id,
                cps.name,
                LEAST(COUNT(cp.id), 500) as expected_pairs
            FROM "ContrastivePairSet" cps
            JOIN "ContrastivePair" cp ON cp."setId" = cps.id
            GROUP BY cps.id, cps.name
        ),
        pairs_with_all_strategies AS (
            -- Pairs that already have all 7 strategies for this model
            SELECT
                a."contrastivePairSetId" as set_id,
                a."contrastivePairId"
            FROM "Activation" a
            WHERE a."modelId" = %s
            GROUP BY a."contrastivePairSetId", a."contrastivePairId"
            HAVING COUNT(DISTINCT a."extractionStrategy") = 7
        ),
        complete_counts AS (
            -- Count of complete pairs per benchmark
            SELECT set_id, COUNT(*) as complete_pairs
            FROM pairs_with_all_strategies
            GROUP BY set_id
        )
        SELECT
            bt.set_id,
            bt.name,
            bt.expected_pairs - COALESCE(cc.complete_pairs, 0) as incomplete_pairs
        FROM benchmark_totals bt
        LEFT JOIN complete_counts cc ON bt.set_id = cc.set_id
        WHERE bt.expected_pairs > COALESCE(cc.complete_pairs, 0)
        ORDER BY incomplete_pairs DESC
    ''', (model_id,))

    results = cur.fetchall()
    cur.close()
    return results


def get_pairs_needing_strategies(conn, model_id: int, set_id: int, limit: int = 500) -> list:
    """Get pairs that have fewer than 7 strategies (including 0).

    Returns pairs that need extraction - either they have no activations at all
    or they have partial extraction (1-6 strategies).
    """
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '0'")

    cur.execute('''
        WITH pairs_with_complete_strategies AS (
            -- Pairs that already have all 7 strategies
            SELECT "contrastivePairId"
            FROM "Activation"
            WHERE "modelId" = %s AND "contrastivePairSetId" = %s
            GROUP BY "contrastivePairId"
            HAVING COUNT(DISTINCT "extractionStrategy") = 7
        )
        SELECT cp.id, cp."positiveExample", cp."negativeExample"
        FROM "ContrastivePair" cp
        WHERE cp."setId" = %s
        AND cp.id NOT IN (SELECT "contrastivePairId" FROM pairs_with_complete_strategies)
        ORDER BY cp.id
        LIMIT %s
    ''', (model_id, set_id, set_id, limit))

    pairs = cur.fetchall()
    cur.close()
    return pairs


def create_activations_batch(conn, records: list):
    """Batch insert multiple activation records at once.

    Args:
        conn: Database connection
        records: List of tuples (model_id, pair_id, set_id, layer, activation_vec, is_positive, strategy)
    """
    if not records:
        return

    cur = conn.cursor()
    cur.execute("SET statement_timeout = '0'")

    # Convert all activation tensors to bytes
    values = []
    for model_id, pair_id, set_id, layer, activation_vec, is_positive, strategy in records:
        neuron_count = activation_vec.shape[0]
        activation_bytes = hidden_states_to_bytes(activation_vec)
        values.append((
            model_id, pair_id, set_id, layer, neuron_count,
            strategy, psycopg2.Binary(activation_bytes), is_positive,
            'system'  # userId
        ))

    # Batch insert using execute_values (much faster than individual inserts)
    execute_values(
        cur,
        '''
        INSERT INTO "Activation"
        ("modelId", "contrastivePairId", "contrastivePairSetId", "layer", "neuronCount",
         "extractionStrategy", "activationData", "isPositive", "userId", "createdAt", "updatedAt")
        VALUES %s
        ON CONFLICT DO NOTHING
        ''',
        values,
        template='(%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())',
        page_size=100
    )
    conn.commit()
    cur.close()


def create_activation(conn, model_id: int, pair_id: int, set_id: int, layer: int,
                      activation_vec: torch.Tensor, is_positive: bool, strategy: str):
    """Create single Activation record (legacy, use create_activations_batch for better perf)."""
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '0'")

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
