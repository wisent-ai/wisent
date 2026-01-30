"""Supabase caching for nonsense activations."""

import os
import struct
from typing import Dict, Tuple, Optional
import numpy as np
import torch

# Database config from environment
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "aws-0-eu-west-2.pooler.supabase.com"),
    "port": int(os.environ.get("DB_PORT", 5432)),
    "database": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "postgres.rbqjqnouluslojmmnuqi"),
    "password": os.environ.get("DB_PASSWORD", "BsKuEnPFLCFurN4a"),
}


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Convert tensor to bytes for database storage."""
    arr = tensor.cpu().numpy().astype(np.float32).flatten()
    return struct.pack(f'{len(arr)}f', *arr)


def _bytes_to_tensor(data: bytes, shape: Tuple[int, ...]) -> torch.Tensor:
    """Convert bytes back to tensor."""
    num_floats = len(data) // 4
    arr = np.array(struct.unpack(f'{num_floats}f', data), dtype=np.float32)
    return torch.from_numpy(arr.reshape(shape))


def get_cached_nonsense_from_db(
    model_name: str, layer: int, n_pairs: int
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Get cached nonsense activations from Supabase.

    Returns activations if found with >= n_pairs, None otherwise.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute('''
            SELECT "positiveData", "negativeData", "nPairs", "hiddenDim"
            FROM "NonsenseCache"
            WHERE "modelName" = %s AND "layer" = %s AND "nPairs" >= %s
            ORDER BY "nPairs" ASC LIMIT 1
        ''', (model_name, layer, n_pairs))
        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            pos_data, neg_data, cached_n, hidden_dim = row
            pos = _bytes_to_tensor(pos_data, (cached_n, hidden_dim))
            neg = _bytes_to_tensor(neg_data, (cached_n, hidden_dim))
            return pos[:n_pairs], neg[:n_pairs]
        return None
    except Exception:
        return None


def cache_nonsense_to_db(
    model_name: str, layer: int, n_pairs: int,
    pos: torch.Tensor, neg: torch.Tensor
) -> bool:
    """Cache nonsense activations to Supabase. Returns True on success."""
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()

        cur.execute('''
            CREATE TABLE IF NOT EXISTS "NonsenseCache" (
                "id" SERIAL PRIMARY KEY,
                "modelName" TEXT NOT NULL,
                "layer" INTEGER NOT NULL,
                "nPairs" INTEGER NOT NULL,
                "hiddenDim" INTEGER NOT NULL,
                "positiveData" BYTEA NOT NULL,
                "negativeData" BYTEA NOT NULL,
                "createdAt" TIMESTAMP DEFAULT NOW(),
                UNIQUE("modelName", "layer", "nPairs")
            )
        ''')

        pos_bytes = _tensor_to_bytes(pos)
        neg_bytes = _tensor_to_bytes(neg)
        hidden_dim = pos.shape[1]

        cur.execute('''
            INSERT INTO "NonsenseCache" ("modelName", "layer", "nPairs", "hiddenDim", "positiveData", "negativeData")
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT ("modelName", "layer", "nPairs") DO UPDATE
            SET "positiveData" = EXCLUDED."positiveData",
                "negativeData" = EXCLUDED."negativeData",
                "createdAt" = NOW()
        ''', (model_name, layer, n_pairs, hidden_dim, pos_bytes, neg_bytes))

        conn.commit()
        cur.close()
        conn.close()
        return True
    except Exception:
        return False


def list_cached_nonsense() -> list:
    """List all cached nonsense entries in database."""
    try:
        import psycopg2
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute('''
            SELECT "modelName", "layer", "nPairs", "hiddenDim", "createdAt"
            FROM "NonsenseCache"
            ORDER BY "modelName", "layer"
        ''')
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return [{"model": r[0], "layer": r[1], "n_pairs": r[2], "hidden_dim": r[3], "created": r[4]} for r in rows]
    except Exception:
        return []
