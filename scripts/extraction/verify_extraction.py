#!/usr/bin/env python3
"""Verify extraction completion for a model.

Checks that each benchmark has all pairs (up to 500 or max available)
with all 7 extraction strategies present.
"""

import argparse
import psycopg2

# Same database as extraction script
DATABASE_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'


def create_indexes(cur):
    """Create indexes to speed up queries if they don't exist."""
    print("Creating indexes if needed...")
    indexes = [
        ('idx_activation_model_id', '"Activation"', '"modelId"'),
        ('idx_activation_model_strategy', '"Activation"', '"modelId", "extractionStrategy"'),
        ('idx_activation_model_pair', '"Activation"', '"modelId", "contrastivePairId"'),
        ('idx_activation_model_set_pair', '"Activation"', '"modelId", "contrastivePairSetId", "contrastivePairId"'),
    ]
    for idx_name, table, columns in indexes:
        try:
            cur.execute(f'CREATE INDEX IF NOT EXISTS {idx_name} ON {table} ({columns})')
            print(f"  Index {idx_name} ready")
        except Exception as e:
            print(f"  Index {idx_name} skipped: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name pattern to check")
    parser.add_argument("--create-indexes", action="store_true", help="Create indexes first")
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL, connect_timeout=30)
    conn.autocommit = True
    cur = conn.cursor()

    if args.create_indexes:
        create_indexes(cur)

    # Get model ID
    cur.execute(
        """SELECT id, "huggingFaceId" FROM "Model" WHERE "huggingFaceId" LIKE %s""",
        (f"%{args.model}%",)
    )
    models = cur.fetchall()
    print(f"=== Models matching '{args.model}' ===")
    for m in models:
        print(f"  {m[0]}: {m[1]}")

    if not models:
        print("No models found!")
        return

    model_id = models[0][0]
    model_name = models[0][1]
    print(f"\nVerifying model: {model_name} (ID: {model_id})")

    # For each benchmark:
    # 1. Get total pairs in benchmark (up to 500)
    # 2. Get pairs with all 7 strategies
    # 3. Compare
    cur.execute("""
        WITH benchmark_totals AS (
            -- Total pairs per benchmark (capped at 500)
            SELECT
                cps.id as set_id,
                cps.name,
                LEAST(COUNT(cp.id), 500) as expected_pairs
            FROM "ContrastivePairSet" cps
            JOIN "ContrastivePair" cp ON cp."setId" = cps.id
            GROUP BY cps.id, cps.name
        ),
        pairs_with_all_strategies AS (
            -- Pairs that have all 7 strategies for this model
            SELECT
                a."contrastivePairSetId" as set_id,
                a."contrastivePairId",
                COUNT(DISTINCT a."extractionStrategy") as strategy_count
            FROM "Activation" a
            WHERE a."modelId" = %s
            GROUP BY a."contrastivePairSetId", a."contrastivePairId"
            HAVING COUNT(DISTINCT a."extractionStrategy") = 7
        ),
        complete_pairs_per_benchmark AS (
            -- Count of pairs with all 7 strategies per benchmark
            SELECT
                set_id,
                COUNT(*) as complete_pairs
            FROM pairs_with_all_strategies
            GROUP BY set_id
        )
        SELECT
            bt.name,
            bt.expected_pairs,
            COALESCE(cpb.complete_pairs, 0) as complete_pairs,
            CASE
                WHEN COALESCE(cpb.complete_pairs, 0) >= bt.expected_pairs THEN 'OK'
                WHEN COALESCE(cpb.complete_pairs, 0) > 0 THEN 'PARTIAL'
                ELSE 'MISSING'
            END as status
        FROM benchmark_totals bt
        LEFT JOIN complete_pairs_per_benchmark cpb ON bt.set_id = cpb.set_id
        ORDER BY
            CASE
                WHEN COALESCE(cpb.complete_pairs, 0) >= bt.expected_pairs THEN 2
                WHEN COALESCE(cpb.complete_pairs, 0) > 0 THEN 1
                ELSE 0
            END,
            bt.name
    """, (model_id,))

    results = cur.fetchall()

    # Summary counts
    ok_count = sum(1 for r in results if r[3] == 'OK')
    partial_count = sum(1 for r in results if r[3] == 'PARTIAL')
    missing_count = sum(1 for r in results if r[3] == 'MISSING')

    print(f"\n=== SUMMARY ===")
    print(f"  OK (all pairs have 7 strategies): {ok_count}")
    print(f"  PARTIAL (some pairs have 7 strategies): {partial_count}")
    print(f"  MISSING (no pairs have 7 strategies): {missing_count}")
    print(f"  TOTAL BENCHMARKS: {len(results)}")

    # Show incomplete benchmarks
    incomplete = [r for r in results if r[3] != 'OK']
    if incomplete:
        print(f"\n=== INCOMPLETE BENCHMARKS ({len(incomplete)}) ===")
        for name, expected, complete, status in incomplete[:50]:  # Show first 50
            print(f"  [{status}] {name}: {complete}/{expected} pairs complete")
        if len(incomplete) > 50:
            print(f"  ... and {len(incomplete) - 50} more")

    # Show some OK benchmarks as confirmation
    ok_benchmarks = [r for r in results if r[3] == 'OK']
    if ok_benchmarks:
        print(f"\n=== COMPLETE BENCHMARKS (showing first 20 of {len(ok_benchmarks)}) ===")
        for name, expected, complete, status in ok_benchmarks[:20]:
            print(f"  [OK] {name}: {complete}/{expected} pairs complete")

    conn.close()

    # Exit with error code if not all benchmarks are complete
    if incomplete:
        print(f"\n VERIFICATION FAILED: {len(incomplete)} benchmarks incomplete")
        exit(1)
    else:
        print(f"\n VERIFICATION PASSED: All {len(results)} benchmarks complete")
        exit(0)


if __name__ == "__main__":
    main()
