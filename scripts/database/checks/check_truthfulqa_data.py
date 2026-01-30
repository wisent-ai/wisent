#!/usr/bin/env python3
"""Check what truthfulqa data exists in both tables."""
import psycopg2

conn = psycopg2.connect(
    host='aws-0-eu-west-2.pooler.supabase.com',
    port=5432,
    database='postgres',
    user='postgres.rbqjqnouluslojmmnuqi',
    password='BsKuEnPFLCFurN4a'
)
cur = conn.cursor()

# Check Activation table for truthfulqa
print("=== Activation Table ===")
cur.execute('''
    SELECT cps.name, COUNT(*) as cnt
    FROM "Activation" a
    JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
    WHERE cps.name LIKE '%truthful%'
    GROUP BY cps.name
    ORDER BY cnt DESC
''')
results = cur.fetchall()
if results:
    print("TruthfulQA benchmarks in Activation:")
    for r in results:
        print(f"  {r[0]}: {r[1]} rows")
else:
    print("No truthfulqa data in Activation table")

# Check what models have truthfulqa data
cur.execute('''
    SELECT m.name, COUNT(*) as cnt
    FROM "Activation" a
    JOIN "Model" m ON a."modelId" = m.id
    JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
    WHERE cps.name LIKE '%truthful%'
    GROUP BY m.name
    ORDER BY cnt DESC
''')
results = cur.fetchall()
if results:
    print("\nModels with truthfulqa data in Activation:")
    for r in results:
        print(f"  {r[0]}: {r[1]} rows")

# Check what extraction strategies exist
cur.execute('''
    SELECT DISTINCT a."extractionStrategy"
    FROM "Activation" a
    JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
    WHERE cps.name LIKE '%truthful%'
''')
results = cur.fetchall()
if results:
    print("\nExtraction strategies for truthfulqa:")
    for r in results:
        print(f"  {r[0]}")

print("\n=== RawActivation Table ===")
cur.execute('''
    SELECT cps.name, COUNT(*) as cnt
    FROM "RawActivation" a
    JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
    WHERE cps.name LIKE '%truthful%'
    GROUP BY cps.name
    ORDER BY cnt DESC
''')
results = cur.fetchall()
if results:
    print("TruthfulQA benchmarks in RawActivation:")
    for r in results:
        print(f"  {r[0]}: {r[1]} rows")

# Check what models have truthfulqa data in RawActivation
cur.execute('''
    SELECT m.name, COUNT(*) as cnt
    FROM "RawActivation" a
    JOIN "Model" m ON a."modelId" = m.id
    JOIN "ContrastivePairSet" cps ON a."contrastivePairSetId" = cps.id
    WHERE cps.name LIKE '%truthful%'
    GROUP BY m.name
    ORDER BY cnt DESC
''')
results = cur.fetchall()
if results:
    print("\nModels with truthfulqa data in RawActivation:")
    for r in results:
        print(f"  {r[0]}: {r[1]} rows")

conn.close()
