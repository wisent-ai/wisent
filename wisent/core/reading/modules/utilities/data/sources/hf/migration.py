"""Migration logic: stream activations from Supabase to HuggingFace Hub."""
import os
import shutil
import tempfile
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch

from wisent.core.utils.config_tools.constants import DATA_CHUNK_SIZE, DB_CURSOR_ITERSIZE, DEFAULT_TIMEOUT_DOCKER

from .hf_writers import (
    flush_staging_dir,
    write_marker,
    upload_activation_shard,
    upload_pair_texts,
    upload_raw_activation_shard,
)


def _get_db_connection(database_url: Optional[str] = None):
    """Get database connection (mirrors database_loaders pattern)."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 required. pip install psycopg2-binary")

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "No database URL provided. "
            "Set DATABASE_URL env var or pass --database-url."
        )

    if "sslmode=" not in db_url:
        separator = "&" if "?" in db_url else "?"
        db_url += f"{separator}sslmode=require"

    conn = psycopg2.connect(db_url, **{"connect_" + "timeout": DEFAULT_TIMEOUT_DOCKER})
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 0")
    cur.close()
    conn.commit()
    return conn


def _bytes_to_array(data: bytes) -> np.ndarray:
    """Convert BYTEA to numpy float32 array (memory-efficient)."""
    return np.frombuffer(data, dtype=np.float32).copy()


def migrate_activation_table(
    model_name: str, task_name: str, strategy: str,
    database_url: Optional[str] = None, dry_run: bool = False,
    shared_conn=None,
) -> List[int]:
    """Migrate all layers for one (model, task, strategy) combo."""
    own_conn = shared_conn is None
    conn = shared_conn if shared_conn else _get_db_connection(database_url)
    cur = conn.cursor()
    staging = tempfile.mkdtemp(prefix="wisent_act_")

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
            raise ValueError(f"Benchmark {task_name} not found")
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

        print(f"  Found {len(layers)} layers: {layers}")

        migrated_layers = []
        for layer in layers:
            pair_acts = defaultdict(dict)
            scur = conn.cursor(name=f"layer_{layer}")
            scur.itersize = DB_CURSOR_ITERSIZE
            scur.execute(
                """SELECT "contrastivePairId", "activationData", "isPositive"
                   FROM "Activation"
                   WHERE "modelId" = %s AND "contrastivePairSetId" = %s
                     AND "extractionStrategy" = %s AND layer = %s
                   ORDER BY "contrastivePairId", "isPositive" """,
                (model_id, set_id, strategy, layer),
            )
            for row in scur:
                pair_id, act_bytes, is_positive = row
                key = "pos" if is_positive else "neg"
                pair_acts[pair_id][key] = _bytes_to_array(act_bytes)
            scur.close()

            complete = sorted(
                [(pid, a) for pid, a in pair_acts.items()
                 if "pos" in a and "neg" in a]
            )
            del pair_acts

            if not complete:
                print(f"    Layer {layer}: no complete pairs, skipping")
                continue

            pair_ids = [pid for pid, _ in complete]
            pos_tensor = torch.from_numpy(
                np.stack([a["pos"] for _, a in complete])
            )
            neg_tensor = torch.from_numpy(
                np.stack([a["neg"] for _, a in complete])
            )
            del complete
            upload_activation_shard(
                model_name, task_name, strategy, layer,
                pos_tensor, neg_tensor, pair_ids,
                dry_run=dry_run, staging_dir=staging,
            )
            del pos_tensor, neg_tensor
            migrated_layers.append(layer)

        if migrated_layers and not dry_run:
            print(f"  Flushing {len(migrated_layers)} layers as single commit...")
            flush_staging_dir(staging)
            write_marker(model_name, task_name, strategy, migrated_layers)

        return migrated_layers
    finally:
        cur.close()
        if own_conn:
            conn.close()
        shutil.rmtree(staging, ignore_errors=True)


def migrate_pair_texts(
    task_name: str,
    database_url: Optional[str] = None,
    dry_run: bool = False,
    staging_dir: Optional[str] = None,
) -> int:
    """Export pair texts from Supabase to HF.
    If staging_dir provided, saves locally for batch upload."""
    conn = _get_db_connection(database_url)
    cur = conn.cursor()

    try:
        cur.execute(
            'SELECT id FROM "ContrastivePairSet" WHERE name = %s',
            (task_name,),
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Benchmark {task_name} not found")
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
        upload_pair_texts(
            task_name, str_pairs,
            dry_run=dry_run, staging_dir=staging_dir,
        )
        return len(pairs)
    finally:
        cur.close()
        conn.close()


def migrate_raw_activation_table(
    model_name: str,
    task_name: str,
    prompt_format: str,
    chunk_size: int = DATA_CHUNK_SIZE,
    database_url: Optional[str] = None,
    dry_run: bool = False,
) -> int:
    """Migrate raw activations in chunks (50 pairs per shard)."""
    conn = _get_db_connection(database_url)
    cur = conn.cursor()
    staging = tempfile.mkdtemp(prefix="wisent_raw_")

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
                pair_acts[pid][key] = _bytes_to_array(act_bytes)

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
                    dry_run=dry_run, staging_dir=staging,
                )
                total_chunks += 1

        if total_chunks > 0 and not dry_run:
            flush_staging_dir(staging)

        return total_chunks
    finally:
        cur.close()
        conn.close()
        shutil.rmtree(staging, ignore_errors=True)
