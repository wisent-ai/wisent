#!/usr/bin/env python3
"""Verify extraction completion for a model."""

import argparse
import psycopg2

DATABASE_URL = "postgresql://postgres:J7wBv963kMq6@bobloo.com:5432/postgres"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name pattern to check")
    args = parser.parse_args()

    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()

    # Get model ID
    cur.execute(
        """SELECT id, name FROM "Model" WHERE name LIKE %s""",
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
        """SELECT COUNT(*) FROM "Activation" WHERE model_id = %s""",
        (model_id,)
    )
    total = cur.fetchone()[0]
    print(f"\n=== Total Activations: {total:,} ===")

    # Count by strategy
    cur.execute(
        """SELECT strategy, COUNT(*) FROM "Activation"
           WHERE model_id = %s
           GROUP BY strategy
           ORDER BY strategy""",
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
        """SELECT COUNT(DISTINCT pair_id) FROM "Activation" WHERE model_id = %s""",
        (model_id,)
    )
    unique_pairs = cur.fetchone()[0]
    print(f"\n=== Unique pairs with activations: {unique_pairs:,} ===")

    # Check strategy distribution per benchmark (sample)
    cur.execute(
        """SELECT cs.name, COUNT(DISTINCT a.strategy) as strategy_count, COUNT(DISTINCT a.pair_id) as pair_count
           FROM "Activation" a
           JOIN "ContrastivePair" cp ON a.pair_id = cp.id
           JOIN "ContrastiveSet" cs ON cp.set_id = cs.id
           WHERE a.model_id = %s
           GROUP BY cs.name
           ORDER BY strategy_count, cs.name
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
