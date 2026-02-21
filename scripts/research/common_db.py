"""
Database access functions for research analysis.

Contains database connection config, model lookups, activation loading,
and data conversion utilities.
"""

import os
import struct
from collections import defaultdict
from typing import Dict, List, Any

import numpy as np
import psycopg2

from common_data import ActivationData

# Database configuration - uses environment variables with fallback
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "aws-0-eu-west-2.pooler.supabase.com"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "postgres.rbqjqnouluslojmmnuqi"),
    "password": os.environ.get("DB_PASSWORD", "REDACTED_DB_PASSWORD"),
    "options": "-c statement_timeout=0",  # No timeout
}

# Models to analyze
RESEARCH_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.2-1B-Instruct",
    "openai/gpt-oss-20b",
    "Qwen/Qwen3-8B",
]


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get model info from database including number of layers."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE name = %s', (model_name,))
    result = cur.fetchone()
    cur.close()
    conn.close()

    if not result:
        raise ValueError(f"Model {model_name} not found in database")

    return {"id": result[0], "num_layers": result[1]}


def get_all_models_with_activations() -> List[Dict[str, Any]]:
    """Get all models that have activations in the database."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('''
        SELECT DISTINCT m.name, m."numLayers"
        FROM "Model" m
        JOIN "Activation" a ON m.id = a."modelId"
    ''')
    results = cur.fetchall()
    cur.close()
    conn.close()

    return [{"name": r[0], "num_layers": r[1]} for r in results]


def bytes_to_vector(data: bytes) -> np.ndarray:
    """Convert binary data back to float vector."""
    num_floats = len(data) // 4
    return np.array(struct.unpack(f'{num_floats}f', data))


def load_activations_from_db(model_name: str, layer: int = None, benchmark: str = None) -> Dict[str, List[ActivationData]]:
    """
    Load activations from database grouped by benchmark.

    Args:
        model_name: Name of the model (e.g., 'Qwen/Qwen3-8B')
        layer: Layer to load (default: middle layer)
        benchmark: Filter to specific benchmark (default: all benchmarks)

    Returns:
        Dict[benchmark_name -> List[ActivationData]]
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get model ID - use huggingFaceId field
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    result = cur.fetchone()
    if not result:
        raise ValueError(f"Model {model_name} not found in database")
    model_id, num_layers = result

    # Default to middle layer if not specified
    if layer is None:
        layer = num_layers // 2

    print(f"Loading activations for model {model_name} (id={model_id}), layer {layer}" + (f", benchmark={benchmark}" if benchmark else ""))

    # Query activations with benchmark info
    if benchmark:
        cur.execute('''
            SELECT
                a."contrastivePairId",
                a."contrastivePairSetId",
                cps.name as benchmark,
                a.layer,
                a."extractionStrategy",
                a."activationData",
                a."isPositive"
            FROM "Activation" a
            JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
            WHERE a."modelId" = %s AND a.layer = %s AND cps.name = %s
            ORDER BY a."contrastivePairSetId", a."contrastivePairId", a."extractionStrategy"
        ''', (model_id, layer, benchmark))
    else:
        cur.execute('''
            SELECT
                a."contrastivePairId",
                a."contrastivePairSetId",
                cps.name as benchmark,
                a.layer,
                a."extractionStrategy",
                a."activationData",
                a."isPositive"
            FROM "Activation" a
            JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
            WHERE a."modelId" = %s AND a.layer = %s
            ORDER BY a."contrastivePairSetId", a."contrastivePairId", a."extractionStrategy"
        ''', (model_id, layer))

    rows = cur.fetchall()
    print(f"Fetched {len(rows)} activation rows")

    # Group by benchmark and pair
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for row in rows:
        pair_id, set_id, benchmark, layer, strategy, activation_bytes, is_positive = row
        key = "positive" if is_positive else "negative"
        raw_data[benchmark][(pair_id, set_id)][strategy][key] = bytes_to_vector(activation_bytes)

    # Convert to ActivationData objects
    result = defaultdict(list)
    for benchmark, pairs in raw_data.items():
        for (pair_id, set_id), strategies in pairs.items():
            for strategy, activations in strategies.items():
                if "positive" in activations and "negative" in activations:
                    result[benchmark].append(ActivationData(
                        pair_id=pair_id,
                        set_id=set_id,
                        benchmark=benchmark,
                        layer=layer,
                        strategy=strategy,
                        positive_activation=activations["positive"],
                        negative_activation=activations["negative"],
                    ))

    cur.close()
    conn.close()

    return dict(result)
