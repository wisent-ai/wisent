"""Cleanup: drop Supabase Activation rows for combos that exist in HF Hub.

Uses parallel workers for throughput. Each worker has its own DB connection.
"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    INDEX_LAST,
    COMBO_OFFSET,
    MIN_COMBO_PATH_PARTS,
    PG_STATEMENT_NO_LIMIT,
    RECURSION_INITIAL_DEPTH,
    PARALLEL_CLEANUP_WORKERS,
)
from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    safe_name_to_model,
)

_print_lock = threading.Lock()


def _get_db_connection(database_url: Optional[str] = None):
    """Get database connection."""
    import psycopg2
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError("No DATABASE_URL.")
    parsed = urlparse(db_url)
    pg = {
        k: v[next(iter(range(len(v))))]
        for k, v in parse_qs(parsed.query).items()
        if k not in ("pgbouncer", "connection_limit")
    }
    if "sslmode" not in pg:
        pg["sslmode"] = "require"
    db_url = urlunparse(parsed._replace(query=urlencode(pg)))
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SET statement_" + "timeout = %s", (PG_STATEMENT_NO_LIMIT,))
    return conn


def _download_hf_index() -> dict:
    """Download and parse index.json from HF Hub."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename="index.json",
        repo_type=HF_REPO_TYPE, token=token,
    )
    with open(local_path, "r") as f:
        return json.load(f)


def _parse_index_combos(index: dict) -> list:
    """Parse index.json into (model, task, strategy, layers) tuples."""
    combos = []
    for key, layers in index.items():
        parts = key.split("/")
        if len(parts) < MIN_COMBO_PATH_PARTS:
            continue
        safe_model = parts[INDEX_FIRST]
        strategy = parts[INDEX_LAST]
        task = "/".join(parts[COMBO_OFFSET:INDEX_LAST])
        model = safe_name_to_model(safe_model)
        combos.append((model, task, strategy, sorted(layers)))
    return combos


def _worker_process_chunk(database_url, chunk, counter, lock):
    """Worker: persistent connection, processes a chunk of layers."""
    conn = _get_db_connection(database_url)
    cur = conn.cursor()
    for model, task, strategy, layer, model_id, set_id in chunk:
        deleted = RECURSION_INITIAL_DEPTH
        while True:
            try:
                cur.execute(
                    """DELETE FROM "Activation"
                       WHERE "modelId" = %s AND "contrastivePairSetId" = %s
                         AND "extractionStrategy" = %s AND layer = %s""",
                    (model_id, set_id, strategy, layer),
                )
                deleted = cur.rowcount
                conn.commit()
                break
            except Exception as exc:
                with lock:
                    print(f"  RETRY {model}/{task}/layer={layer}: {exc}")
                try:
                    conn.close()
                except Exception:
                    pass
                try:
                    conn = _get_db_connection(database_url)
                    cur = conn.cursor()
                except Exception:
                    continue
        if deleted > RECURSION_INITIAL_DEPTH:
            with lock:
                counter.append(deleted)
                print(
                    f"  Deleted {deleted} rows: "
                    f"{model}/{task}/{strategy}/layer={layer}"
                    f"  [total={sum(counter)}]"
                )
    try:
        conn.close()
    except Exception:
        pass


def cleanup_migrated_activations(
    database_url: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """Delete Supabase Activation rows that exist in HF Hub.

    Uses PARALLEL_CLEANUP_WORKERS threads with persistent connections.
    """
    print("Downloading HF index.json ...")
    index = _download_hf_index()
    combos = _parse_index_combos(index)
    print(f"Found {len(combos)} combos in HF index")

    conn = _get_db_connection(database_url)
    cur = conn.cursor()
    model_cache, set_cache = {}, {}
    total_skipped = RECURSION_INITIAL_DEPTH

    work_items = []
    for model, task, strategy, layers in combos:
        if model not in model_cache:
            cur.execute(
                'SELECT id FROM "Model" WHERE "huggingFaceId" = %s',
                (model,),
            )
            row = cur.fetchone()
            model_cache[model] = row[INDEX_FIRST] if row else None
        model_id = model_cache[model]
        if model_id is None:
            total_skipped += len(layers)
            continue
        if task not in set_cache:
            cur.execute(
                'SELECT id FROM "ContrastivePairSet" WHERE name = %s',
                (task,),
            )
            row = cur.fetchone()
            set_cache[task] = row[INDEX_FIRST] if row else None
        set_id = set_cache[task]
        if set_id is None:
            total_skipped += len(layers)
            continue
        for layer in layers:
            work_items.append(
                (model, task, strategy, layer, model_id, set_id)
            )
    conn.close()
    print(f"Work items: {len(work_items)} layers to process")
    print(f"Skipped (not in DB): {total_skipped} layer-slots")

    if dry_run:
        print("[DRY-RUN] parallel mode not supported")
        return {"dry_run": True, "combos_affected": RECURSION_INITIAL_DEPTH,
                "total_rows_deleted": RECURSION_INITIAL_DEPTH,
                "total_skipped": total_skipped, "details": []}

    workers = PARALLEL_CLEANUP_WORKERS
    chunk_size = len(work_items) // workers + bool(len(work_items) % workers)
    chunks = [work_items[i:i + chunk_size]
              for i in range(RECURSION_INITIAL_DEPTH, len(work_items),
                             chunk_size)]
    print(f"Starting {len(chunks)} workers, ~{chunk_size} items each")

    counter = []
    lock = threading.Lock()
    threads = []
    for chunk in chunks:
        t = threading.Thread(
            target=_worker_process_chunk,
            args=(database_url, chunk, counter, lock),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    total = sum(counter)
    print(f"\nDone. Total deleted this run: {total}")
    return {
        "dry_run": False, "combos_affected": len(work_items),
        "total_rows_deleted": total,
        "total_skipped": total_skipped, "details": [],
    }
