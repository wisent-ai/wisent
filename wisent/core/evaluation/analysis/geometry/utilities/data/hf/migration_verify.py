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
from wisent.core.constants import COMPARE_TOL, N_JOBS_SINGLE


def _load_migrated_keys() -> Set[str]:
    """Load already-migrated keys from markers/ dir in HF repo.
    Marker paths: markers/{model}/{benchmark_path}/{strategy}.json
    where benchmark_path may contain slashes (e.g. coding/humaneval)."""
    from huggingface_hub import HfApi
    token = _get_hf_token()
    api = HfApi(token=token)
    keys: Set[str] = set()
    try:
        info = api.dataset_info(repo_id=HF_REPO_ID, token=token)
        for s in (info.siblings or []):
            fn = s.rfilename
            if fn.startswith("markers/") and fn.endswith(".json"):
                inner = fn[len("markers/"):-len(".json")]
                parts = inner.split("/")
                if len(parts) >= 3:
                    model = parts[0]
                    strategy = parts[-1]
                    benchmark = "/".join(parts[1:-1])
                    keys.add(f"{model}/{benchmark}/{strategy}")
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
    reverse: bool = False,
) -> None:
    """Discover all (model, benchmark, strategy) combos and migrate.
    Uses staging dirs to batch uploads and avoid HF rate limits.
    Builds combos from Model x ContrastivePairSet x strategies
    (avoids scanning the massive Activation table which times out)."""
    strategies = [
        "chat_first", "chat_last", "chat_max_norm",
        "chat_mean", "chat_weighted", "mc_balanced", "role_play",
    ]
    conn = _get_db_connection(database_url)
    cur = conn.cursor()
    try:
        cur.execute(
            'SELECT "huggingFaceId" FROM "Model" ORDER BY id'
        )
        models = [r[0] for r in cur.fetchall()]
        cur.execute(
            'SELECT name FROM "ContrastivePairSet" ORDER BY id'
        )
        benchmarks = [r[0] for r in cur.fetchall()]
    finally:
        cur.close()
        conn.close()
    combos = [
        (m, bm, s)
        for m in models
        for bm in benchmarks
        for s in strategies
    ]
    combos.sort()
    total = len(combos)
    combos = combos[combo_start:combo_end]
    if reverse:
        combos = list(reversed(combos))
    direction = " (REVERSE)" if reverse else ""
    print(f"Found {total} total combos, processing slice [{combo_start}:{combo_end}] = {len(combos)}{direction}")

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
    end_val = combo_end if combo_end is not None else total
    shared_conn = _get_db_connection(database_url)
    for idx, (model, task, strategy) in enumerate(combos):
        global_idx = end_val - N_JOBS_SINGLE - idx if reverse else combo_start + idx
        safe = model_to_safe_name(model)
        key = f"{safe}/{task}/{strategy}"
        if key in migrated_keys:
            print(f"\n[{global_idx}] SKIP (already migrated): {model} / {task} / {strategy}")
            continue
        print(f"\n[{global_idx}] Migrating: {model} / {task} / {strategy}")
        try:
            migrate_activation_table(
                model, task, strategy,
                database_url=database_url, dry_run=dry_run,
                shared_conn=shared_conn,
            )
        except Exception as exc:
            if "closed the connection" in str(exc) or "connection" in str(exc).lower():
                print(f"  Connection lost, reconnecting...")
                try:
                    shared_conn.close()
                except Exception:
                    pass
                shared_conn = _get_db_connection(database_url)
                migrate_activation_table(
                    model, task, strategy,
                    database_url=database_url, dry_run=dry_run,
                    shared_conn=shared_conn,
                )
            else:
                raise
    try:
        shared_conn.close()
    except Exception:
        pass


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

    pos_match = torch.allclose(db_pos, hf_pos, atol=COMPARE_TOL)
    neg_match = torch.allclose(db_neg, hf_neg, atol=COMPARE_TOL)

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
