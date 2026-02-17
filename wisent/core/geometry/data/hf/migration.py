"""Migration logic: stream activations from Supabase to HuggingFace Hub."""
import os
import struct
from collections import defaultdict
from typing import List, Optional

import torch

from .hf_writers import (
    update_index,
    upload_activation_shard,
    upload_pair_texts,
    upload_raw_activation_shard,
)


def _get_db_connection(database_url: Optional[str] = None):
    """Get database connection (mirrors database_loaders pattern)."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError(
            "psycopg2 required for migration. "
            "Install with: pip install psycopg2-binary"
        )

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "No database URL provided. "
            "Set DATABASE_URL env var or pass --database-url."
        )

    if "sslmode=" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url += f"{separator}sslmode=require"

    return psycopg2.connect(db_url, connect_timeout=15)


def _bytes_to_vector(data: bytes) -> List[float]:
    """Convert BYTEA to float vector."""
    num_floats = len(data) // 4
    return list(struct.unpack(f"{num_floats}f", data))


def migrate_activation_table(
    model_name: str,
    task_name: str,
    strategy: str,
    database_url: Optional[str] = None,
    dry_run: bool = False,
) -> List[int]:
    """Migrate all layers for one (model, task, strategy) combo.

    Streams from Supabase one layer at a time, converts BYTEA to
    tensors, uploads as safetensors shards.

    Returns list of migrated layer numbers.
    """
    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        cur.execute(
            'SELECT id FROM "Model" WHERE "huggingFaceId" = %s',
            (model_name,),
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Model {model_name} not found in database")
        model_id = result[0]

        cur.execute(
            'SELECT id FROM "ContrastivePairSet" WHERE name = %s',
            (task_name,),
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Benchmark {task_name} not found in database")
        set_id = result[0]

        cur.execute(
            """SELECT DISTINCT layer FROM "Activation"
               WHERE "modelId" = %s AND "contrastivePairSetId" = %s
                 AND "extractionStrategy" = %s
               ORDER BY layer""",
            (model_id, set_id, strategy),
        )
        layers = [row[0] for row in cur.fetchall()]

        if not layers:
            print(f"  No layers found for {model_name}/{task_name}/{strategy}")
            return []

        print(
            f"  Found {len(layers)} layers for "
            f"{model_name}/{task_name}/{strategy}: {layers}"
        )

        migrated_layers = []
        for layer in layers:
            cur.execute(
                """SELECT "contrastivePairId", "activationData", "isPositive"
                   FROM "Activation"
                   WHERE "modelId" = %s AND "contrastivePairSetId" = %s
                     AND "extractionStrategy" = %s AND layer = %s
                   ORDER BY "contrastivePairId", "isPositive" """,
                (model_id, set_id, strategy, layer),
            )

            pair_acts = defaultdict(dict)
            for row in cur.fetchall():
                pair_id, act_bytes, is_positive = row
                key = "pos" if is_positive else "neg"
                pair_acts[pair_id][key] = _bytes_to_vector(act_bytes)

            pos_list, neg_list, pair_ids = [], [], []
            for pid, acts in sorted(pair_acts.items()):
                if "pos" in acts and "neg" in acts:
                    pos_list.append(acts["pos"])
                    neg_list.append(acts["neg"])
                    pair_ids.append(pid)

            if not pos_list:
                print(f"    Layer {layer}: no complete pairs, skipping")
                continue

            pos_tensor = torch.tensor(pos_list, dtype=torch.float32)
            neg_tensor = torch.tensor(neg_list, dtype=torch.float32)

            upload_activation_shard(
                model_name, task_name, strategy, layer,
                pos_tensor, neg_tensor, pair_ids,
                dry_run=dry_run,
            )
            migrated_layers.append(layer)

        if migrated_layers and not dry_run:
            update_index(model_name, task_name, strategy, migrated_layers)

        return migrated_layers

    finally:
        cur.close()
        conn.close()


def migrate_pair_texts(
    task_name: str,
    database_url: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Export pair texts from Supabase to HF.

    Returns number of pairs migrated.
    """
    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        cur.execute(
            'SELECT id FROM "ContrastivePairSet" WHERE name = %s',
            (task_name,),
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Benchmark {task_name} not found in database")
        set_id = result[0]

        cur.execute(
            """SELECT id, "positiveExample", "negativeExample"
               FROM "ContrastivePair"
               WHERE "setId" = %s ORDER BY id""",
            (set_id,),
        )

        pairs = {}
        for row in cur.fetchall():
            pair_id, pos_example, neg_example = row
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

        if not pairs:
            print(f"  No pairs found for {task_name}")
            return 0

        str_pairs = {str(k): v for k, v in pairs.items()}
        upload_pair_texts(task_name, str_pairs, dry_run=dry_run)
        return len(pairs)

    finally:
        cur.close()
        conn.close()


def migrate_raw_activation_table(
    model_name: str,
    task_name: str,
    prompt_format: str,
    chunk_size: int = 50,
    database_url: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Migrate raw activations in chunks (50 pairs per shard).

    Returns total number of chunks uploaded.
    """
    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        cur.execute(
            'SELECT id FROM "Model" WHERE "huggingFaceId" = %s',
            (model_name,),
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Model {model_name} not found")
        model_id = result[0]

        cur.execute(
            'SELECT id FROM "ContrastivePairSet" WHERE name = %s',
            (task_name,),
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Benchmark {task_name} not found")
        set_id = result[0]

        cur.execute(
            """SELECT DISTINCT layer FROM "RawActivation"
               WHERE "modelId" = %s AND "contrastivePairSetId" = %s
                 AND "promptFormat" = %s
               ORDER BY layer""",
            (model_id, set_id, prompt_format),
        )
        layers = [row[0] for row in cur.fetchall()]
        total_chunks = 0

        for layer in layers:
            cur.execute(
                """SELECT "contrastivePairId", "activationData", "isPositive"
                   FROM "RawActivation"
                   WHERE "modelId" = %s AND "contrastivePairSetId" = %s
                     AND "promptFormat" = %s AND layer = %s
                   ORDER BY "contrastivePairId", "isPositive" """,
                (model_id, set_id, prompt_format, layer),
            )

            pair_acts = defaultdict(dict)
            for row in cur.fetchall():
                pid, act_bytes, is_pos = row
                key = "pos" if is_pos else "neg"
                pair_acts[pid][key] = _bytes_to_vector(act_bytes)

            sorted_pairs = sorted(
                [(pid, a) for pid, a in pair_acts.items()
                 if "pos" in a and "neg" in a]
            )

            for ci in range(0, len(sorted_pairs), chunk_size):
                chunk = sorted_pairs[ci:ci + chunk_size]
                pos_list = [a["pos"] for _, a in chunk]
                neg_list = [a["neg"] for _, a in chunk]
                pids = [pid for pid, _ in chunk]

                pos_t = torch.tensor(pos_list, dtype=torch.float32)
                neg_t = torch.tensor(neg_list, dtype=torch.float32)

                upload_raw_activation_shard(
                    model_name, task_name, prompt_format, layer,
                    ci // chunk_size, pos_t, neg_t, pids,
                    dry_run=dry_run,
                )
                total_chunks += 1

        return total_chunks

    finally:
        cur.close()
        conn.close()
