#!/usr/bin/env python3
"""Verify extraction completion for a model."""

import argparse
import psycopg2

# Same database as extraction script
DATABASE_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name pattern to check")
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL, connect_timeout=30)
    cur = conn.cursor()

    # Get model ID by huggingFaceId
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

    # Count total activations
    cur.execute(
        """SELECT COUNT(*) FROM "Activation" WHERE "modelId" = %s""",
        (model_id,)
    )
    total = cur.fetchone()[0]
    print(f"\n=== Total Activations: {total:,} ===")

    # Count by strategy
    cur.execute(
        """SELECT "extractionStrategy", COUNT(*) FROM "Activation"
           WHERE "modelId" = %s
           GROUP BY "extractionStrategy"
           ORDER BY "extractionStrategy" """,
        (model_id,)
    )
    strategies = cur.fetchall()
    print("\n=== Activations by Strategy ===")
    for row in strategies:
        print(f"  {row[0]}: {row[1]:,}")

    # Check if all 7 strategies are present
    expected_strategies = {
        "chat_mean", "chat_first", "chat_last", "chat_max_norm",
        "chat_weighted", "role_play", "mc_balanced"
    }
    found_strategies = {row[0] for row in strategies}
    missing = expected_strategies - found_strategies
    if missing:
        print(f"\n WARNING: Missing strategies: {missing}")
    else:
        print(f"\n All 7 strategies present!")

    # Count unique pairs with activations
    cur.execute(
        """SELECT COUNT(DISTINCT "contrastivePairId") FROM "Activation" WHERE "modelId" = %s""",
        (model_id,)
    )
    unique_pairs = cur.fetchone()[0]
    print(f"\n=== Unique pairs with activations: {unique_pairs:,} ===")

    # Check strategy distribution per benchmark (sample)
    cur.execute(
        """SELECT cps.name, COUNT(DISTINCT a."extractionStrategy") as strategy_count, COUNT(DISTINCT a."contrastivePairId") as pair_count
           FROM "Activation" a
           JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
           WHERE a."modelId" = %s
           GROUP BY cps.name
           ORDER BY strategy_count, cps.name
           LIMIT 20""",
        (model_id,)
    )
    print("\n=== Benchmarks (first 20, ordered by strategy count) ===")
    for row in cur.fetchall():
        status = "OK" if row[1] == 7 else "INCOMPLETE"
        print(f"  [{status}] {row[0]}: {row[1]} strategies, {row[2]} pairs")

    conn.close()


if __name__ == "__main__":
    main()
