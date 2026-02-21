"""
Verification functions for truthfulqa_custom extraction.

Checks that all pairs, layers, formats, and polarities have been extracted
and reports any issues.
"""

import os
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]


def verify_extraction(model_name: str, benchmark: str = "truthfulqa_custom") -> bool:
    """
    Verify extraction was successful after extraction completes.

    Checks:
    1. All pairs have activations for all layers
    2. Both positive and negative activations exist for each pair
    3. All prompt formats were extracted
    4. No duplicates

    Returns True if verification passes, False otherwise.
    """
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()

    print(f"\n{'='*60}")
    print(f"VERIFICATION: {model_name} / {benchmark}")
    print(f"{'='*60}")

    issues = []

    # Get model info
    cur.execute('SELECT id, "numLayers" FROM "Model" WHERE "huggingFaceId" = %s', (model_name,))
    result = cur.fetchone()
    if not result:
        print(f"ERROR: Model {model_name} not found in database")
        cur.close()
        conn.close()
        return False
    model_id, num_layers = result
    print(f"Model ID: {model_id}, Layers: {num_layers}")

    # Get benchmark set info
    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (benchmark,))
    result = cur.fetchone()
    if not result:
        print(f"ERROR: Benchmark {benchmark} not found in database")
        cur.close()
        conn.close()
        return False
    set_id = result[0]
    print(f"Benchmark Set ID: {set_id}")

    # Get expected pair count
    cur.execute('SELECT COUNT(*) FROM "ContrastivePair" WHERE "setId" = %s', (set_id,))
    expected_pairs = cur.fetchone()[0]
    print(f"Expected pairs: {expected_pairs}")

    # Check 1: Count unique pairs with activations
    cur.execute('''
        SELECT COUNT(DISTINCT "contrastivePairId")
        FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s
    ''', (model_id, set_id))
    actual_pairs = cur.fetchone()[0]
    print(f"Pairs with activations: {actual_pairs}")

    if actual_pairs < expected_pairs:
        issues.append(f"Missing pairs: {expected_pairs - actual_pairs} pairs not extracted")

    # Check 2: Verify all layers extracted per pair
    cur.execute('''
        SELECT "contrastivePairId", "promptFormat", COUNT(DISTINCT layer) as layer_count
        FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s
        GROUP BY "contrastivePairId", "promptFormat"
        HAVING COUNT(DISTINCT layer) != %s
    ''', (model_id, set_id, num_layers))
    incomplete_layers = cur.fetchall()
    if incomplete_layers:
        issues.append(f"Incomplete layer coverage: {len(incomplete_layers)} pair/format combinations missing layers")
        for pair_id, fmt, layer_count in incomplete_layers[:5]:
            print(f"  - Pair {pair_id} ({fmt}): {layer_count}/{num_layers} layers")

    # Check 3: Verify both positive and negative exist per pair/layer/format
    cur.execute('''
        SELECT "contrastivePairId", layer, "promptFormat",
               COUNT(DISTINCT "isPositive") as polarity_count,
               ARRAY_AGG(DISTINCT "isPositive") as polarities
        FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s
        GROUP BY "contrastivePairId", layer, "promptFormat"
        HAVING COUNT(DISTINCT "isPositive") != 2
    ''', (model_id, set_id))
    incomplete_polarities = cur.fetchall()
    if incomplete_polarities:
        issues.append(f"Missing polarities: {len(incomplete_polarities)} pair/layer/format combinations missing pos or neg")
        for pair_id, layer, fmt, cnt, pols in incomplete_polarities[:5]:
            print(f"  - Pair {pair_id}, layer {layer} ({fmt}): only has {pols}")

    # Check 4: Verify all 3 prompt formats
    cur.execute('''
        SELECT "promptFormat", COUNT(DISTINCT "contrastivePairId") as pair_count
        FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s
        GROUP BY "promptFormat"
    ''', (model_id, set_id))
    format_counts = {row[0]: row[1] for row in cur.fetchall()}
    expected_formats = ["chat", "mc_balanced", "role_play"]
    for fmt in expected_formats:
        if fmt not in format_counts:
            issues.append(f"Missing prompt format: {fmt}")
        elif format_counts[fmt] < expected_pairs:
            issues.append(f"Incomplete format {fmt}: {format_counts[fmt]}/{expected_pairs} pairs")
    print(f"Prompt format coverage: {format_counts}")

    # Check 5: No duplicates
    cur.execute('''
        SELECT "contrastivePairId", layer, "isPositive", "promptFormat", COUNT(*) as cnt
        FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s
        GROUP BY "contrastivePairId", layer, "isPositive", "promptFormat"
        HAVING COUNT(*) > 1
    ''', (model_id, set_id))
    duplicates = cur.fetchall()
    if duplicates:
        issues.append(f"Duplicates found: {len(duplicates)} duplicate activations")
        for pair_id, layer, is_pos, fmt, cnt in duplicates[:5]:
            print(f"  - Pair {pair_id}, layer {layer}, pos={is_pos} ({fmt}): {cnt} copies")

    # Summary
    cur.execute('''
        SELECT COUNT(*) FROM "RawActivation"
        WHERE "modelId" = %s AND "contrastivePairSetId" = %s
    ''', (model_id, set_id))
    total_activations = cur.fetchone()[0]

    expected_total = expected_pairs * num_layers * 2 * 3  # pairs * layers * (pos+neg) * 3 formats

    print(f"\nTotal activations: {total_activations}")
    print(f"Expected total: {expected_total}")
    completeness = (total_activations / expected_total * 100) if expected_total > 0 else 0
    print(f"Completeness: {completeness:.1f}%")

    cur.close()
    conn.close()

    if issues:
        print(f"\n{'!'*60}")
        print(f"VERIFICATION FAILED - {len(issues)} issue(s):")
        for issue in issues:
            print(f"  - {issue}")
        print(f"{'!'*60}")
        return False
    else:
        print(f"\n{'*'*60}")
        print("VERIFICATION PASSED - All checks successful!")
        print(f"{'*'*60}")
        return True

