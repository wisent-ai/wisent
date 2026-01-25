"""Database loaders for activations and pair texts from Supabase."""
from typing import Dict, Any, Optional, List, Tuple
import os
import struct
import torch


def _get_db_connection(database_url: Optional[str] = None):
    """Get database connection with proper URL handling."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 required for database loading. Install with: pip install psycopg2-binary")

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("No database URL provided. Set DATABASE_URL env var or pass database_url parameter.")

    # Handle Supabase pooler URL (use direct connection)
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = '300s'")
    cur.close()
    return conn


def _bytes_to_vector(data: bytes, seq_len: int, hidden_dim: int, strategy: str, prompt_len: int) -> List[float]:
    """Convert bytes to activation vector using extraction strategy."""
    num_floats = seq_len * hidden_dim
    values = struct.unpack(f'{num_floats}f', data)

    if strategy == "last_token":
        start = (seq_len - 1) * hidden_dim
        return list(values[start:start + hidden_dim])
    elif strategy == "first_token":
        idx = min(prompt_len, seq_len - 1)
        start = idx * hidden_dim
        return list(values[start:start + hidden_dim])
    else:
        start = (seq_len - 1) * hidden_dim
        return list(values[start:start + hidden_dim])


def load_activations_from_database(
    model_name: str,
    task_name: str,
    layer: int,
    prompt_format: str = "chat",
    extraction_strategy: str = "last_token",
    limit: Optional[int] = None,
    database_url: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load activations for a single layer directly from database.

    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        task_name: Name of the contrastive pair set (e.g., "truthfulqa_custom")
        layer: Layer number to load
        prompt_format: Prompt format used ("chat" or "completion")
        extraction_strategy: Token extraction strategy ("last_token" or "first_token")
        limit: Maximum number of pairs to load
        database_url: Optional database URL (defaults to DATABASE_URL env var)

    Returns:
        Tuple of (pos_activations, neg_activations) tensors
    """
    from collections import defaultdict

    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        # Get model ID
        cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Model {model_name} not found in database")
        model_id = result[0]

        # Get set ID
        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (task_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Benchmark {task_name} not found in database")
        set_id = result[0]

        # Load all activations for this layer
        limit_clause = f"LIMIT {limit * 2}" if limit else ""

        cur.execute(f'''
            SELECT "contrastivePairId", "seqLen", "hiddenDim", "promptLen", "hiddenStates", "isPositive"
            FROM "RawActivation"
            WHERE "modelId" = %s
              AND "contrastivePairSetId" = %s
              AND "promptFormat" = %s
              AND layer = %s
            ORDER BY "contrastivePairId", "isPositive"
            {limit_clause}
        ''', (model_id, set_id, prompt_format, layer))

        rows = cur.fetchall()

        # Group by pair_id
        pair_activations = defaultdict(dict)
        for row in rows:
            pair_id, seq_len, h_dim, prompt_len, hidden_bytes, is_positive = row
            activation = _bytes_to_vector(hidden_bytes, seq_len, h_dim, extraction_strategy, prompt_len or 0)
            key = "pos" if is_positive else "neg"
            pair_activations[pair_id][key] = activation

        # Convert to tensors - only pairs with both pos and neg
        pos_list, neg_list = [], []
        for pair_id, acts in sorted(pair_activations.items()):
            if "pos" in acts and "neg" in acts:
                pos_list.append(acts["pos"])
                neg_list.append(acts["neg"])

        return torch.tensor(pos_list, dtype=torch.float32), torch.tensor(neg_list, dtype=torch.float32)

    finally:
        cur.close()
        conn.close()


def load_pair_texts_from_database(
    task_name: str,
    limit: int = 200,
    database_url: Optional[str] = None,
) -> Dict[int, Dict[str, str]]:
    """
    Load contrastive pair texts from Supabase database.

    Args:
        task_name: Name of the contrastive pair set (e.g., "truthfulqa_custom")
        limit: Maximum number of pairs to load
        database_url: Optional database URL (defaults to DATABASE_URL env var)

    Returns:
        Dict mapping pair_id -> {prompt, positive, negative}
    """
    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        # Get set ID for task
        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (task_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"No ContrastivePairSet found with name '{task_name}'")
        set_id = result[0]

        # Load pairs
        cur.execute('''
            SELECT id, "positiveExample", "negativeExample"
            FROM "ContrastivePair"
            WHERE "setId" = %s
            ORDER BY id
            LIMIT %s
        ''', (set_id, limit))

        pairs = {}
        for row in cur.fetchall():
            pair_id, pos_example, neg_example = row

            # Parse prompt and responses
            if "\n\n" in pos_example:
                prompt = pos_example.rsplit("\n\n", 1)[0]
                pos_response = pos_example.rsplit("\n\n", 1)[1]
            else:
                prompt = pos_example
                pos_response = ""

            if "\n\n" in neg_example:
                neg_response = neg_example.rsplit("\n\n", 1)[1]
            else:
                neg_response = neg_example

            pairs[pair_id] = {
                "prompt": prompt,
                "positive": pos_response,
                "negative": neg_response,
            }

        return pairs

    finally:
        cur.close()
        conn.close()
