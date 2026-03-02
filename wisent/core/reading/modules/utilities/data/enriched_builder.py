"""Build enriched pairs JSON from database or by generation + activation collection."""
import json
import math
import os
import struct
from collections import defaultdict
from typing import Optional

from wisent.core.utils.config_tools.constants import BYTES_PER_MB


def build_enriched_from_db(
    model_name: str,
    task_name: str,
    work_dir: str,
    extraction_strategy: str,
    limit: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    database_url: Optional[str] = None,
) -> Optional[str]:
    """
    Build enriched pairs JSON from Supabase database.
    Returns path to the enriched file, or None if DB data unavailable.
    """
    try:
        import psycopg2
    except ImportError:
        print("  DB: psycopg2 not available, skipping DB lookup")
        return None

    db_url = database_url or os.environ.get("DATABASE_URL")
    if not db_url:
        print("  DB: No DATABASE_URL set, skipping DB lookup")
        return None
    if "sslmode=" not in db_url:
        db_url += "?sslmode=require" if "?" not in db_url else "&sslmode=require"

    try:
        conn = psycopg2.connect(db_url)
    except Exception as e:
        print(f"  DB: Connection failed: {e}")
        return None

    cur = conn.cursor()
    try:
        return _query_and_build(
            cur, model_name, task_name, work_dir,
            extraction_strategy, limit,
            num_attention_heads, num_key_value_heads)
    except Exception as e:
        print(f"  DB lookup failed: {e}")
        return None
    finally:
        cur.close()
        conn.close()


def _query_and_build(
    cur, model_name, task_name, work_dir,
    extraction_strategy, limit,
    num_attention_heads, num_key_value_heads,
):
    """Run DB queries and build enriched JSON. Returns file path or None."""
    cur.execute('SELECT id FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    row = cur.fetchone()
    if not row:
        print(f"  DB: Model {model_name} not found")
        return None
    model_id = row[0]

    cur.execute(
        'SELECT id, name FROM "ContrastivePairSet" WHERE name = %s OR name LIKE %s ORDER BY name LIMIT 1',
        (task_name, '%/' + task_name))
    row = cur.fetchone()
    if not row:
        print(f"  DB: Task {task_name} not found")
        return None
    set_id = row[0]
    print(f"  DB: Matched task '{task_name}' → '{row[1]}' (id={set_id})")

    cur.execute('''
        SELECT id, "positiveExample", "negativeExample"
        FROM "ContrastivePair" WHERE "setId" = %s ORDER BY id
    ''', (set_id,))
    pair_rows = cur.fetchall()
    if not pair_rows:
        print(f"  DB: No pairs for {task_name}")
        return None

    pairs_text = {}
    for pid, pos_ex, neg_ex in pair_rows:
        if "\n\n" in pos_ex:
            prompt = pos_ex.rsplit("\n\n", 1)[0]
            pos_resp = pos_ex.rsplit("\n\n", 1)[1]
        else:
            prompt, pos_resp = pos_ex, ""
        neg_resp = neg_ex.rsplit("\n\n", 1)[1] if "\n\n" in neg_ex else neg_ex
        pairs_text[pid] = {"prompt": prompt, "positive": pos_resp, "negative": neg_resp}

    cur.execute('''
        SELECT DISTINCT layer FROM "Activation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s AND "extractionStrategy" = %s
        ORDER BY layer
    ''', (model_id, set_id, extraction_strategy))
    layers = [r[0] for r in cur.fetchall()]
    if not layers:
        print(f"  DB: No activations for {model_name}/{task_name}/{extraction_strategy}")
        return None

    cur.execute('''
        SELECT "contrastivePairId", layer, "activationData", "isPositive"
        FROM "Activation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s AND "extractionStrategy" = %s
        ORDER BY "contrastivePairId", layer, "isPositive"
    ''', (model_id, set_id, extraction_strategy))

    act_data = defaultdict(lambda: defaultdict(dict))
    for pid, layer, blob, is_pos in cur.fetchall():
        n = len(blob) // 4
        act_data[pid][layer]["pos" if is_pos else "neg"] = list(struct.unpack(f'{n}f', blob))

    complete_pids = []
    for pid in sorted(pairs_text.keys()):
        if pid not in act_data:
            continue
        has_all = all(
            "pos" in act_data[pid].get(l, {}) and "neg" in act_data[pid].get(l, {})
            for l in layers)
        if has_all:
            complete_pids.append(pid)

    if not complete_pids:
        print(f"  DB: No complete pairs (all layers + both sides) for {task_name}")
        return None
    if limit and len(complete_pids) > limit:
        complete_pids = complete_pids[:limit]

    print(f"  DB: Found {len(complete_pids)} complete pairs across {len(layers)} layers")

    calibration_norms = {}
    for layer in layers:
        norms = []
        for pid in complete_pids:
            for side in ("pos", "neg"):
                vec = act_data[pid][layer][side]
                norms.append(math.sqrt(sum(x * x for x in vec)))
        calibration_norms[str(layer)] = sum(norms) / len(norms) if norms else 0.0

    enriched_pairs = _assemble_pairs(complete_pids, pairs_text, act_data, layers, task_name)

    output = {
        "task_name": task_name, "trait_label": task_name,
        "model": model_name, "layers": layers,
        "extraction_strategy": extraction_strategy,
        "extraction_component": "residual_stream",
        "raw_mode": False, "num_pairs": len(enriched_pairs),
        "calibration_norms": calibration_norms,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "pairs": enriched_pairs,
    }

    out_path = os.path.join(work_dir, "enriched_from_db.json")
    with open(out_path, 'w') as f:
        json.dump(output, f)

    size_mb = os.path.getsize(out_path) / BYTES_PER_MB
    print(f"  DB: Wrote enriched file ({size_mb:.1f} MB): {out_path}")
    return out_path


def _assemble_pairs(complete_pids, pairs_text, act_data, layers, task_name):
    """Assemble enriched pair dicts from text and activation data."""
    enriched = []
    for pid in complete_pids:
        pt = pairs_text[pid]
        pair_dict = {
            "prompt": pt["prompt"],
            "positive_response": {
                "model_response": pt["positive"],
                "layers_activations": {},
                "q_proj_activations": {}, "k_proj_activations": {},
            },
            "negative_response": {
                "model_response": pt["negative"],
                "layers_activations": {},
                "q_proj_activations": {}, "k_proj_activations": {},
            },
            "label": task_name, "trait_description": "",
        }
        for layer in layers:
            ls = str(layer)
            pair_dict["positive_response"]["layers_activations"][ls] = act_data[pid][layer]["pos"]
            pair_dict["negative_response"]["layers_activations"][ls] = act_data[pid][layer]["neg"]
        enriched.append(pair_dict)
    return enriched


def generate_and_collect_enriched(
    model_name, benchmark, work_dir, limit, device, cached_model=None,
):
    """Generate contrastive pairs and collect activations from model."""
    from wisent.core.utils.cli.optimize_steering.data.contrastive_pairs_data import (
        execute_generate_pairs_from_task,
    )
    from wisent.core.utils.cli.optimize_steering.data.activations_data import (
        execute_get_activations,
    )
    import argparse

    pairs_file = os.path.join(work_dir, "pairs.json")
    enriched_file = os.path.join(work_dir, "enriched_all_layers.json")

    print(f"  Generating contrastive pairs for {benchmark}...")
    execute_generate_pairs_from_task(argparse.Namespace(
        task_name=benchmark, output=pairs_file,
        limit=limit, verbose=False))

    num_layers = cached_model.num_layers if cached_model else 16
    print(f"  Collecting activations (all {num_layers} layers)...")
    execute_get_activations(argparse.Namespace(
        pairs_file=pairs_file, model=model_name,
        output=enriched_file, layers=None,
        extraction_strategy="chat_last", device=device,
        verbose=False, timing=False, raw=False,
        cached_model=cached_model))
    return enriched_file
