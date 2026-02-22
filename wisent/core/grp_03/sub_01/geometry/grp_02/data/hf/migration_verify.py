"""Migration orchestration and verification."""
import json
import shutil
import tempfile
from typing import Optional, Set

import torch

from .migration import (
    _get_db_connection,
    migrate_activation_table,
    migrate_pair_texts,
)
from .hf_config import HF_REPO_ID, HF_REPO_TYPE, model_to_safe_name
from .hf_writers import flush_staging_dir, _get_hf_token


def _load_migrated_keys() -> Set[str]:
    """Load already-migrated keys from markers/ dir in HF repo."""
    from huggingface_hub import HfApi
    token = _get_hf_token()
    api = HfApi(token=token)
    keys: Set[str] = set()
    try:
        all_files = api.list_repo_tree(
            repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE,
            path_in_repo="markers", recursive=True,
        )
        for f in all_files:
            rp = getattr(f, "rpath", "")
            if rp.endswith(".json"):
                parts = rp.split("/")
                if len(parts) == 4:
                    sm, bm, st = parts[1], parts[2], parts[3].replace(".json", "")
                    keys.add(f"{sm}/{bm}/{st}")
    except Exception as exc:
        print(f"  Warning: could not load markers: {exc}")
    print(f"  Already migrated: {len(keys)} combos (from markers)")
    return keys


def migrate_all(
    database_url: Optional[str] = None,
    dry_run: bool = False,
    combo_start: int = 0,
    combo_end: Optional[int] = None,
    skip_pair_texts: bool = False,
) -> None:
    """Discover all (model, benchmark, strategy) combos and migrate.
    Uses staging dirs to batch uploads and avoid HF rate limits."""
    conn = _get_db_connection(database_url)
    cur = conn.cursor()
    try:
        cur.execute(
            """SELECT DISTINCT m."huggingFaceId", cs.name,
                      a."extractionStrategy"
               FROM "Activation" a
               JOIN "Model" m ON a."modelId" = m.id
               JOIN "ContrastivePairSet" cs
                    ON a."contrastivePairSetId" = cs.id
               ORDER BY 1, 2, 3"""
        )
        combos = cur.fetchall()
    finally:
        cur.close()
        conn.close()
    total = len(combos)
    combos = combos[combo_start:combo_end]
    print(f"Found {total} total combos, processing slice [{combo_start}:{combo_end}] = {len(combos)}")

    if not skip_pair_texts:
        pair_staging = tempfile.mkdtemp(prefix="wisent_pair_texts_")
        migrated_tasks: set = set()
        try:
            for _, task, _ in combos:
                if task not in migrated_tasks:
                    print(f"\nStaging pair texts: {task}")
                    migrate_pair_texts(
                        task, database_url=database_url,
                        dry_run=dry_run, staging_dir=pair_staging,
                    )
                    migrated_tasks.add(task)
            if not dry_run and migrated_tasks:
                print(f"\nFlushing {len(migrated_tasks)} pair text files...")
                flush_staging_dir(pair_staging)
        finally:
            shutil.rmtree(pair_staging, ignore_errors=True)
    else:
        print("Skipping pair texts migration (--skip-pair-texts)")

    migrated_keys = _load_migrated_keys()
    for idx, (model, task, strategy) in enumerate(combos):
        global_idx = combo_start + idx
        safe = model_to_safe_name(model)
        key = f"{safe}/{task}/{strategy}"
        if key in migrated_keys:
            print(f"\n[{global_idx}] SKIP (already migrated): {model} / {task} / {strategy}")
            continue
        print(f"\n[{global_idx}] Migrating: {model} / {task} / {strategy}")
        migrate_activation_table(
            model, task, strategy,
            database_url=database_url, dry_run=dry_run,
        )


def verify_migration(
    model_name: str,
    task_name: str,
    strategy: str,
    layer: int,
    database_url: Optional[str] = None,
) -> bool:
    """Verify HF data matches Supabase for a single layer."""
    from ..database_loaders import load_activations_from_database
    from .hf_loaders import load_activations_from_hf

    print(
        f"  Verifying {model_name}/{task_name}"
        f"/{strategy}/layer_{layer}..."
    )

    db_pos, db_neg = load_activations_from_database(
        model_name, task_name, layer,
        extraction_strategy=strategy,
        database_url=database_url,
        use_cache=False,
    )

    hf_pos, hf_neg = load_activations_from_hf(
        model_name, task_name, layer,
        extraction_strategy=strategy,
        use_cache=False,
    )

    if db_pos.shape != hf_pos.shape:
        print(
            f"    MISMATCH: pos shape "
            f"DB={db_pos.shape} HF={hf_pos.shape}"
        )
        return False

    if db_neg.shape != hf_neg.shape:
        print(
            f"    MISMATCH: neg shape "
            f"DB={db_neg.shape} HF={hf_neg.shape}"
        )
        return False

    pos_match = torch.allclose(db_pos, hf_pos, atol=1e-6)
    neg_match = torch.allclose(db_neg, hf_neg, atol=1e-6)

    if pos_match and neg_match:
        n = db_pos.shape[0]
        d = db_pos.shape[1]
        print(f"    MATCH: {n} pairs, dim={d}")
        return True

    if not pos_match:
        diff = (db_pos - hf_pos).abs().max().item()
        print(f"    MISMATCH: pos max diff = {diff}")
    if not neg_match:
        diff = (db_neg - hf_neg).abs().max().item()
        print(f"    MISMATCH: neg max diff = {diff}")

    return False
