#!/usr/bin/env python3
"""Get schema details for RawActivation table."""
import psycopg2
import os

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres"
)

conn = psycopg2.connect(DATABASE_URL)
conn.autocommit = True
cur = conn.cursor()

# Get columns of RawActivation
cur.execute("""
    SELECT column_name, data_type
    FROM information_schema.columns
    WHERE table_name = 'RawActivation'
    ORDER BY ordinal_position
""")
print("RawActivation columns:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

print()

# Now verify the extraction with correct column names
cur.execute('''
    SELECT MIN("layer"), MAX("layer"), COUNT(DISTINCT "layer")
    FROM "RawActivation" ra
    JOIN "Model" m ON ra."modelId" = m."id"
    WHERE m."huggingFaceId" = 'openai/gpt-oss-20b'
''')
row = cur.fetchone()
print(f"Layers: min={row[0]}, max={row[1]}, distinct={row[2]} (expected 24)")

# Check for NULL activations
cur.execute('''
    SELECT COUNT(*) FROM "RawActivation" ra
    JOIN "Model" m ON ra."modelId" = m."id"
    WHERE m."huggingFaceId" = 'openai/gpt-oss-20b' AND ra."activation" IS NULL
''')
nulls = cur.fetchone()[0]
print(f"NULL activations: {nulls}")

# Check by isPositive
cur.execute('''
    SELECT ra."isPositive", COUNT(*)
    FROM "RawActivation" ra
    JOIN "Model" m ON ra."modelId" = m."id"
    WHERE m."huggingFaceId" = 'openai/gpt-oss-20b'
    GROUP BY ra."isPositive"
''')
print("By isPositive:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

# Count by format (via ContrastivePairSet name)
cur.execute('''
    SELECT cps."name", COUNT(*)
    FROM "RawActivation" ra
    JOIN "Model" m ON ra."modelId" = m."id"
    JOIN "ContrastivePair" cp ON ra."pairId" = cp."id"
    JOIN "ContrastivePairSet" cps ON cp."setId" = cps."id"
    WHERE m."huggingFaceId" = 'openai/gpt-oss-20b'
    GROUP BY cps."name"
    ORDER BY cps."name"
''')
print("\nBy ContrastivePairSet:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

# Sample records
cur.execute('''
    SELECT ra."id", ra."layer", ra."isPositive", LENGTH(ra."activation") as size
    FROM "RawActivation" ra
    JOIN "Model" m ON ra."modelId" = m."id"
    WHERE m."huggingFaceId" = 'openai/gpt-oss-20b'
    ORDER BY RANDOM()
    LIMIT 5
''')
print("\nSample records:")
for row in cur.fetchall():
    print(f"  id={row[0]}, layer={row[1]}, positive={row[2]}, size={row[3]}")

conn.close()
print("\n=== Verification Complete ===")
