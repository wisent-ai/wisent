#!/usr/bin/env python3
"""Verify that extraction completed successfully."""
import psycopg2
import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres"
)

def main():
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()

    benchmark = "truthfulqa_custom"
    model = "openai/gpt-oss-20b"

    print("=== TruthfulQA Extraction Verification ===")
    print()

    # Total count
    cur.execute(
        "SELECT COUNT(*) FROM activations WHERE benchmark = %s AND model_name = %s",
        (benchmark, model)
    )
    total = cur.fetchone()[0]
    print(f"Total records: {total}")

    # Expected: 817 pairs x 24 layers x 3 formats x 2 (pos/neg) = 117,648
    expected = 817 * 24 * 3 * 2
    print(f"Expected records: {expected}")
    print(f"Match: {total == expected}")
    if total != expected:
        print(f"  MISSING: {expected - total} records")
    print()

    # Count by format
    cur.execute(
        "SELECT format, COUNT(*) FROM activations WHERE benchmark = %s AND model_name = %s GROUP BY format ORDER BY format",
        (benchmark, model)
    )
    print("Records by format:")
    for row in cur.fetchall():
        expected_per_format = 817 * 24 * 2  # pairs * layers * pos/neg
        status = "OK" if row[1] == expected_per_format else f"MISSING {expected_per_format - row[1]}"
        print(f"  {row[0]}: {row[1]} ({status})")
    print()

    # Count by label (pos/neg)
    cur.execute(
        "SELECT label, COUNT(*) FROM activations WHERE benchmark = %s AND model_name = %s GROUP BY label",
        (benchmark, model)
    )
    print("Records by label:")
    for row in cur.fetchall():
        expected_per_label = 817 * 24 * 3  # pairs * layers * formats
        status = "OK" if row[1] == expected_per_label else f"MISSING {expected_per_label - row[1]}"
        print(f"  {row[0]}: {row[1]} ({status})")
    print()

    # Check layers coverage
    cur.execute(
        "SELECT MIN(layer_idx), MAX(layer_idx), COUNT(DISTINCT layer_idx) FROM activations WHERE benchmark = %s AND model_name = %s",
        (benchmark, model)
    )
    row = cur.fetchone()
    print(f"Layers: min={row[0]}, max={row[1]}, distinct={row[2]} (expected 24)")
    print()

    # Check for NULL activations
    cur.execute(
        "SELECT COUNT(*) FROM activations WHERE benchmark = %s AND model_name = %s AND activation IS NULL",
        (benchmark, model)
    )
    nulls = cur.fetchone()[0]
    print(f"NULL activations: {nulls}")
    print()

    # Check distinct pair_ids
    cur.execute(
        "SELECT COUNT(DISTINCT pair_id) FROM activations WHERE benchmark = %s AND model_name = %s",
        (benchmark, model)
    )
    pairs = cur.fetchone()[0]
    print(f"Distinct pair_ids: {pairs} (expected 817)")
    print()

    # Check for any gaps in pair_ids
    cur.execute(
        "SELECT DISTINCT pair_id FROM activations WHERE benchmark = %s AND model_name = %s ORDER BY pair_id",
        (benchmark, model)
    )
    pair_ids = [r[0] for r in cur.fetchall()]
    if pair_ids:
        expected_ids = set(range(1, 818))
        actual_ids = set(pair_ids)
        missing_ids = expected_ids - actual_ids
        if missing_ids:
            print(f"Missing pair_ids: {sorted(missing_ids)[:20]}...")
        else:
            print("All pair_ids present (1-817)")
    print()

    # Sample a few records to verify data integrity
    cur.execute(
        """SELECT pair_id, layer_idx, format, label,
                  LENGTH(activation) as activation_size,
                  hidden_dim
           FROM activations
           WHERE benchmark = %s AND model_name = %s
           ORDER BY RANDOM()
           LIMIT 5""",
        (benchmark, model)
    )
    print("Sample records:")
    for row in cur.fetchall():
        print(f"  pair={row[0]}, layer={row[1]}, format={row[2]}, label={row[3]}, size={row[4]}, hidden_dim={row[5]}")

    conn.close()
    print()
    print("=== Verification Complete ===")

if __name__ == "__main__":
    main()
