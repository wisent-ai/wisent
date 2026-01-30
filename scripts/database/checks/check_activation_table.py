#!/usr/bin/env python3
"""Check how many benchmarks have all 7 strategies per model."""
import psycopg2

conn = psycopg2.connect(
    host='aws-0-eu-west-2.pooler.supabase.com',
    port=5432,
    database='postgres',
    user='postgres.rbqjqnouluslojmmnuqi',
    password='BsKuEnPFLCFurN4a'
)
cur = conn.cursor()
cur.execute("SET statement_timeout = '600s'")

# Get models
cur.execute('SELECT id, "huggingFaceId" FROM "Model" ORDER BY id')
models = cur.fetchall()

print("=" * 70)
print("ACTIVATION: Benchmarks by number of strategies per model")
print("=" * 70)

for model_id, model_name in models:
    print(f"\n{model_name}:", flush=True)

    # Count strategies per benchmark
    cur.execute('''
        SELECT strategy_count, COUNT(*) as benchmark_count
        FROM (
            SELECT "contrastivePairSetId", COUNT(DISTINCT "extractionStrategy") as strategy_count
            FROM "Activation"
            WHERE "modelId" = %s
            GROUP BY "contrastivePairSetId"
        ) sub
        GROUP BY strategy_count
        ORDER BY strategy_count
    ''', (model_id,))

    results = cur.fetchall()
    total = 0
    for strategy_count, benchmark_count in results:
        print(f"  {strategy_count} strategies: {benchmark_count} benchmarks")
        total += benchmark_count
    print(f"  Total: {total} benchmarks")

conn.close()
