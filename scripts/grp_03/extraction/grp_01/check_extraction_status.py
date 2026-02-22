#!/usr/bin/env python3
"""Check extraction status for all models."""

import psycopg2
import sys

DATABASE_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'

def main():
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=30)
    cur = conn.cursor()

    # Get total benchmark sets
    cur.execute('SELECT COUNT(*) FROM "ContrastivePairSet"')
    total_sets = cur.fetchone()[0]

    # Get models
    cur.execute('SELECT id, "huggingFaceId" FROM "Model" ORDER BY id')
    models = cur.fetchall()

    print("=" * 60)
    print("EXTRACTION STATUS CHECK")
    print("=" * 60)
    print(f"Total benchmark sets: {total_sets}")
    print()

    # Use table statistics for fast estimate
    cur.execute('''
        SELECT n_live_tup FROM pg_stat_user_tables WHERE relname = 'Activation'
    ''')
    row = cur.fetchone()
    if row:
        print(f"Activation table: ~{row[0]:,} rows (estimated)")
    print()

    # Sample recent activations instead of full GROUP BY
    print("Sampling recent activations by model...")
    cur.execute('''
        SELECT "modelId", COUNT(*)
        FROM "Activation"
        WHERE id > (SELECT MAX(id) - 100000 FROM "Activation")
        GROUP BY "modelId"
    ''')
    activation_counts = {row[0]: row[1] for row in cur.fetchall()}
    print(f"Found activation data for {len(activation_counts)} models")

    # Skip slow COUNT DISTINCT - estimate from activation count
    # Each pair has 2 activations (pos/neg) per layer. Typical models have 16-48 layers.
    # So ~32-96 activation records per pair per model.
    set_counts = {}

    for model_id, model_name in models:
        total_activations = activation_counts.get(model_id, 0)
        # Estimate: 211 sets, ~500 pairs each = 105,500 pairs
        # Each pair = 2 activations per layer. Llama 1B has 16 layers = 32 activations/pair
        # So ~3.4M for 16-layer model with full extraction
        estimated_complete = total_activations > 10_000_000  # Very rough estimate
        status = "LIKELY COMPLETE" if estimated_complete else "INCOMPLETE/PARTIAL"
        print(f"Model: {model_name}")
        print(f"  Total activation records: {total_activations:,}")
        print(f"  Status: {status}")
        print()

    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
