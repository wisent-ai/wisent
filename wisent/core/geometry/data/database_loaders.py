"""Database loaders for activations and pair texts from Supabase with local caching."""
from typing import Dict, Optional, List, Tuple
import os
import struct
import json
import torch
from .cache import get_cache_path, save_pair_texts_cache, save_activations_cache


def _get_db_connection(database_url: Optional[str] = None):
    """Get database connection with proper URL handling."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 required for database loading. Install with: pip install psycopg2-binary")

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("No database URL provided. Set DATABASE_URL env var or pass database_url parameter.")

    # Add sslmode if not present
    if "sslmode=" not in db_url:
        db_url += "?sslmode=require" if "?" not in db_url else "&sslmode=require"

    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.close()
    return conn


def _bytes_to_vector_simple(data: bytes) -> List[float]:
    """Convert bytes to activation vector (pre-extracted, single vector)."""
    num_floats = len(data) // 4
    return list(struct.unpack(f'{num_floats}f', data))


def load_activations_from_database(
    model_name: str,
    task_name: str,
    layer: int,
    prompt_format: str = "chat",
    extraction_strategy: str = "completion_last",
    limit: Optional[int] = None,
    database_url: Optional[str] = None,
    pair_ids: Optional[set] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load activations for a single layer from local cache or database.

    Args:
        model_name: HuggingFace model ID (e.g., "meta-llama/Llama-3.2-1B-Instruct")
        task_name: Name of the contrastive pair set (e.g., "truthfulqa_custom")
        layer: Layer number to load
        prompt_format: Prompt format used ("chat" or "completion")
        extraction_strategy: Extraction strategy - use ExtractionStrategy enum values:
            - "completion_last": completion format (prompt\\n\\nresponse) + last token
            - "completion_mean": completion format + mean of response tokens
            - "chat_last": chat template + last token
            - "chat_mean": chat template + mean of response tokens
        limit: Maximum number of pairs to load
        database_url: Optional database URL (defaults to DATABASE_URL env var)
        pair_ids: Optional set of pair IDs to filter (for train/test split)
        use_cache: Whether to use local cache (default: True)
        force_refresh: Force reload from database even if cache exists

    Returns:
        Tuple of (pos_activations, neg_activations) tensors with pair_id mapping
    """
    cache_path = get_cache_path(task_name, "activations", model_name=model_name, layer=layer)

    # Try loading from cache first
    if use_cache and not force_refresh and cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        pos_tensor, neg_tensor = cached["pos"], cached["neg"]
        cached_pair_ids = cached.get("pair_ids", list(range(len(pos_tensor))))
        if pair_ids is not None:
            pos_list, neg_list = [], []
            for i, pid in enumerate(cached_pair_ids):
                if pid in pair_ids:
                    pos_list.append(pos_tensor[i])
                    neg_list.append(neg_tensor[i])
            if pos_list:
                return torch.stack(pos_list), torch.stack(neg_list)
        else:
            if limit and len(pos_tensor) > limit:
                pos_tensor, neg_tensor = pos_tensor[:limit], neg_tensor[:limit]
            return pos_tensor, neg_tensor
    from collections import defaultdict
    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Model {model_name} not found in database")
        model_id = result[0]

        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (task_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Benchmark {task_name} not found in database")
        set_id = result[0]

        # Use extraction_strategy directly - no mapping needed
        # Valid values: completion_last, completion_mean, chat_last, chat_mean, etc.
        db_strategy = extraction_strategy

        cur.execute('''
            SELECT "contrastivePairId", "activationData", "isPositive"
            FROM "Activation"
            WHERE "modelId" = %s
              AND "contrastivePairSetId" = %s
              AND "extractionStrategy" = %s
              AND layer = %s
            ORDER BY "contrastivePairId", "isPositive"
        ''', (model_id, set_id, db_strategy, layer))

        rows = cur.fetchall()
        pair_activations = defaultdict(dict)
        for row in rows:
            pair_id, activation_bytes, is_positive = row
            pair_activations[pair_id]["pos" if is_positive else "neg"] = _bytes_to_vector_simple(activation_bytes)

        pos_list, neg_list, loaded_pair_ids = [], [], []
        for pair_id, acts in sorted(pair_activations.items()):
            if "pos" in acts and "neg" in acts:
                pos_list.append(acts["pos"])
                neg_list.append(acts["neg"])
                loaded_pair_ids.append(pair_id)

        all_pos_tensor = torch.tensor(pos_list, dtype=torch.float32)
        all_neg_tensor = torch.tensor(neg_list, dtype=torch.float32)

        if use_cache and len(pos_list) > 0:
            torch.save({"pos": all_pos_tensor, "neg": all_neg_tensor, "pair_ids": loaded_pair_ids}, cache_path)

        if pair_ids is not None:
            pos_filtered, neg_filtered = [], []
            for i, pid in enumerate(loaded_pair_ids):
                if pid in pair_ids:
                    pos_filtered.append(all_pos_tensor[i])
                    neg_filtered.append(all_neg_tensor[i])
            if pos_filtered:
                return torch.stack(pos_filtered), torch.stack(neg_filtered)
            return torch.tensor([]), torch.tensor([])

        if limit and len(all_pos_tensor) > limit:
            all_pos_tensor, all_neg_tensor = all_pos_tensor[:limit], all_neg_tensor[:limit]
        return all_pos_tensor, all_neg_tensor

    finally:
        cur.close()
        conn.close()


def load_available_layers_from_database(
    model_name: str, task_name: str, extraction_strategy: str = "completion_last", database_url: Optional[str] = None
) -> List[int]:
    """Query database to find all available layers for a model/task combination."""
    conn = _get_db_connection(database_url)
    cur = conn.cursor()
    try:
        cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Model {model_name} not found in database")
        model_id = result[0]
        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (task_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Benchmark {task_name} not found in database")
        set_id = result[0]
        # Use extraction_strategy directly - no mapping needed
        # Valid values: completion_last, completion_mean, chat_last, chat_mean, etc.
        db_strategy = extraction_strategy
        cur.execute('''
            SELECT DISTINCT layer FROM "Activation"
            WHERE "modelId" = %s AND "contrastivePairSetId" = %s AND "extractionStrategy" = %s
            ORDER BY layer
        ''', (model_id, set_id, db_strategy))
        return [row[0] for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def load_pair_texts_from_database(
    task_name: str,
    limit: int = 200,
    database_url: Optional[str] = None,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Dict[int, Dict[str, str]]:
    """
    Load contrastive pair texts from local cache or Supabase database.

    Args:
        task_name: Name of the contrastive pair set (e.g., "truthfulqa_custom")
        limit: Maximum number of pairs to load
        database_url: Optional database URL (defaults to DATABASE_URL env var)
        use_cache: Whether to use local cache (default: True)
        force_refresh: Force reload from database even if cache exists

    Returns:
        Dict mapping pair_id -> {prompt, positive, negative}
    """
    cache_path = get_cache_path(task_name, "pair_texts")

    # Try loading from cache first
    if use_cache and not force_refresh and cache_path.exists():
        print(f"  Loading pair texts from cache: {cache_path}", flush=True)
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        pairs = {int(k): v for k, v in cached_data.items()}
        if limit and len(pairs) > limit:
            pair_ids = sorted(pairs.keys())[:limit]
            pairs = {pid: pairs[pid] for pid in pair_ids}
        return pairs

    # Load from database
    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        # Get set ID for task
        cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (task_name,))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"No ContrastivePairSet found with name '{task_name}'")
        set_id = result[0]

        # Load pairs (load all for caching, apply limit after)
        cur.execute('''
            SELECT id, "positiveExample", "negativeExample"
            FROM "ContrastivePair"
            WHERE "setId" = %s
            ORDER BY id
        ''', (set_id,))

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

        # Save to cache
        if use_cache:
            print(f"  Caching {len(pairs)} pair texts to: {cache_path}")
            with open(cache_path, 'w') as f:
                json.dump(pairs, f)

        # Apply limit
        if limit and len(pairs) > limit:
            pair_ids = sorted(pairs.keys())[:limit]
            pairs = {pid: pairs[pid] for pid in pair_ids}

        return pairs

    finally:
        cur.close()
        conn.close()
