"""Cleanup: drop Supabase Activation rows for combos that exist in HF Hub."""
import json
import os
from typing import Optional

from wisent.core.utils.config_tools.constants import (
    INDEX_FIRST,
    INDEX_LAST,
    COMBO_OFFSET,
    MIN_COMBO_PATH_PARTS,
    PG_STATEMENT_NO_LIMIT,
    RECURSION_INITIAL_DEPTH,
)
from wisent.core.reading.modules.utilities.data.sources.hf.hf_config import (
    HF_REPO_ID,
    HF_REPO_TYPE,
    safe_name_to_model,
)


def _get_db_connection(database_url: Optional[str] = None):
    """Get database connection (same as migration.py)."""
    try:
        import psycopg2
    except ImportError:
        raise ImportError("psycopg2 required. pip install psycopg2-binary")
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    from wisent.core.utils.infra_tools.infra.core.hardware import (
        docker_code_exec_timeout_s,
    )

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        raise ValueError(
            "No database URL. Set DATABASE_URL or --database-url."
        )
    parsed = urlparse(db_url)
    pg = {
        k: v[next(iter(range(len(v))))]
        for k, v in parse_qs(parsed.query).items()
        if k not in ("pgbouncer", "connection_limit")
    }
    if "sslmode" not in pg:
        pg["sslmode"] = "require"
    db_url = urlunparse(parsed._replace(query=urlencode(pg)))
    conn = psycopg2.connect(
        db_url, **{"connect_" + "timeout": docker_code_exec_timeout_s()}
    )
    return conn


def _download_hf_index() -> dict:
    """Download and parse index.json from HF Hub."""
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN") or os.environ.get(
        "HUGGING_FACE_HUB_TOKEN"
    )
    local_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename="index.json",
        repo_type=HF_REPO_TYPE,
        token=token,
    )
    with open(local_path, "r") as f:
        return json.load(f)


def _parse_index_combos(index: dict) -> list:
    """Parse index.json into list of (model, task, strategy, layers)."""
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


def _resolve_model_id(cur, model_name: str) -> Optional[int]:
    """Resolve HF model name to Supabase Model.id."""
    cur.execute(
        'SELECT id FROM "Model" WHERE "huggingFaceId" = %s',
        (model_name,),
    )
    row = cur.fetchone()
    return row[INDEX_FIRST] if row else None


def _resolve_set_id(cur, task_name: str) -> Optional[int]:
    """Resolve task name to Supabase ContrastivePairSet.id."""
    cur.execute(
        'SELECT id FROM "ContrastivePairSet" WHERE name = %s',
        (task_name,),
    )
    row = cur.fetchone()
    return row[INDEX_FIRST] if row else None


def _count_rows(cur, model_id: int, set_id: int,
                strategy: str, layer: int) -> int:
    """Count Activation rows for a specific combo+layer."""
    cur.execute(
        """SELECT COUNT(*) FROM "Activation"
           WHERE "modelId" = %s AND "contrastivePairSetId" = %s
             AND "extractionStrategy" = %s AND layer = %s""",
        (model_id, set_id, strategy, layer),
    )
    return cur.fetchone()[INDEX_FIRST]


def _delete_rows(cur, model_id: int, set_id: int,
                 strategy: str, layer: int) -> int:
    """Delete Activation rows for a specific combo+layer. Returns count."""
    cur.execute(
        """DELETE FROM "Activation"
           WHERE "modelId" = %s AND "contrastivePairSetId" = %s
             AND "extractionStrategy" = %s AND layer = %s""",
        (model_id, set_id, strategy, layer),
    )
    return cur.rowcount


def cleanup_migrated_activations(
    database_url: Optional[str] = None,
    dry_run: bool = True,
) -> dict:
    """Check HF index and drop matching Supabase Activation rows.

    Args:
        database_url: Supabase connection string (or DATABASE_URL env).
        dry_run: If True, only report what would be deleted.

    Returns:
        Summary dict with per-combo counts.
    """
    print("Downloading HF index.json ...")
    index = _download_hf_index()
    combos = _parse_index_combos(index)
    print(f"Found {len(combos)} combos in HF index")

    conn = _get_db_connection(database_url)
    cur = conn.cursor()
    cur.execute("SET statement_" + "timeout = %s", (PG_STATEMENT_NO_LIMIT,))

    model_cache, set_cache = {}, {}
    total_deleted = RECURSION_INITIAL_DEPTH
    total_skipped = RECURSION_INITIAL_DEPTH
    results = []

    for model, task, strategy, layers in combos:
        if model not in model_cache:
            model_cache[model] = _resolve_model_id(cur, model)
        model_id = model_cache[model]
        if model_id is None:
            print(f"  SKIP {model}/{task}/{strategy}: model not in DB")
            total_skipped += len(layers)
            continue

        if task not in set_cache:
            set_cache[task] = _resolve_set_id(cur, task)
        set_id = set_cache[task]
        if set_id is None:
            print(f"  SKIP {model}/{task}/{strategy}: task not in DB")
            total_skipped += len(layers)
            continue

        combo_deleted = RECURSION_INITIAL_DEPTH
        for layer in layers:
            count = _count_rows(cur, model_id, set_id, strategy, layer)
            if count == RECURSION_INITIAL_DEPTH:
                continue
            if dry_run:
                print(
                    f"  [DRY-RUN] Would delete {count} rows: "
                    f"{model}/{task}/{strategy}/layer={layer}"
                )
                combo_deleted += count
            else:
                deleted = _delete_rows(
                    cur, model_id, set_id, strategy, layer
                )
                combo_deleted += deleted
                print(
                    f"  Deleted {deleted} rows: "
                    f"{model}/{task}/{strategy}/layer={layer}"
                )

        if combo_deleted > RECURSION_INITIAL_DEPTH:
            results.append({
                "model": model,
                "task": task,
                "strategy": strategy,
                "layers": len(layers),
                "rows_deleted": combo_deleted,
            })
            total_deleted += combo_deleted

    if not dry_run:
        conn.commit()
        print(f"\nCommitted. Total deleted: {total_deleted}")
    else:
        conn.rollback()
        print(f"\n[DRY-RUN] Would delete total: {total_deleted}")

    print(f"Skipped (not in DB): {total_skipped} layer-slots")
    cur.close()
    conn.close()

    return {
        "dry_run": dry_run,
        "combos_affected": len(results),
        "total_rows_deleted": total_deleted,
        "total_skipped": total_skipped,
        "details": results,
    }
