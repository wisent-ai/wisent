"""Migration orchestration and verification."""
from typing import Optional

import torch

from .migration import (
    _get_db_connection,
    migrate_activation_table,
    migrate_pair_texts,
)


def migrate_all(
    database_url: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Discover all (model, benchmark, strategy) combos and migrate."""
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

    print(f"Found {len(combos)} (model, benchmark, strategy) combinations")

    # Migrate pair texts first (deduplicated by task)
    migrated_tasks = set()
    for _, task, _ in combos:
        if task not in migrated_tasks:
            print(f"\nMigrating pair texts: {task}")
            migrate_pair_texts(
                task, database_url=database_url, dry_run=dry_run
            )
            migrated_tasks.add(task)

    # Migrate activations
    for model, task, strategy in combos:
        print(f"\nMigrating activations: {model} / {task} / {strategy}")
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
    """Verify HF data matches Supabase for a single layer.

    Loads from both sources and compares with torch.allclose.
    Returns True if data matches.
    """
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
