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

    print("=== TruthfulQA Extraction Verification ===")
    print()

    # Check Model table
    cur.execute('SELECT COUNT(*) FROM "Model"')
    model_count = cur.fetchone()[0]
    print(f"Model records: {model_count}")

    cur.execute('SELECT "id", "name", "huggingFaceId", "numLayers" FROM "Model" ORDER BY "id" DESC LIMIT 5')
    print("Recent models:")
    for row in cur.fetchall():
        print(f"  id={row[0]}, name={row[1]}, hf_id={row[2]}, layers={row[3]}")
    print()

    # Check ContrastivePairSet table
    cur.execute('SELECT COUNT(*) FROM "ContrastivePairSet"')
    set_count = cur.fetchone()[0]
    print(f"ContrastivePairSet records: {set_count}")

    cur.execute('SELECT "id", "name", "description" FROM "ContrastivePairSet" ORDER BY "id" DESC LIMIT 5')
    print("Recent pair sets:")
    for row in cur.fetchall():
        print(f"  id={row[0]}, name={row[1]}, desc={row[2][:50] if row[2] else 'N/A'}...")
    print()

    # Check ContrastivePair table
    cur.execute('SELECT COUNT(*) FROM "ContrastivePair"')
    pair_count = cur.fetchone()[0]
    print(f"ContrastivePair records: {pair_count}")

    # Count pairs for truthfulqa_custom specifically
    cur.execute('''
        SELECT COUNT(*) FROM "ContrastivePair" cp
        JOIN "ContrastivePairSet" cps ON cp."setId" = cps."id"
        WHERE cps."name" LIKE '%truthfulqa_custom%'
    ''')
    tqa_pair_count = cur.fetchone()[0]
    print(f"TruthfulQA custom pairs: {tqa_pair_count} (expected 817 per format, ~2451 total for 3 formats)")
    print()

    # Check RawActivation table
    cur.execute('SELECT COUNT(*) FROM "RawActivation"')
    activation_count = cur.fetchone()[0]
    print(f"RawActivation records: {activation_count}")

    # Expected: 817 pairs x 24 layers x 3 formats x 2 (pos/neg) = 117,648
    expected = 817 * 24 * 3 * 2
    print(f"Expected for TruthfulQA: {expected}")
    print()

    # Check activations by model
    cur.execute('''
        SELECT m."name", COUNT(*) as cnt
        FROM "RawActivation" ra
        JOIN "Model" m ON ra."modelId" = m."id"
        GROUP BY m."name"
        ORDER BY cnt DESC
    ''')
    print("Activations by model:")
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}")
    print()

    # Check activations for openai/gpt-oss-20b specifically
    cur.execute('''
        SELECT COUNT(*) FROM "RawActivation" ra
        JOIN "Model" m ON ra."modelId" = m."id"
        WHERE m."huggingFaceId" = 'openai/gpt-oss-20b'
    ''')
    gpt_activations = cur.fetchone()[0]
    print(f"Activations for openai/gpt-oss-20b: {gpt_activations}")

    if gpt_activations > 0:
        # Check layers coverage
        cur.execute('''
            SELECT MIN(ra."layerIndex"), MAX(ra."layerIndex"), COUNT(DISTINCT ra."layerIndex")
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

        # Sample records
        cur.execute('''
            SELECT ra."id", ra."layerIndex", ra."isPositive", LENGTH(ra."activation") as size, m."huggingFaceId"
            FROM "RawActivation" ra
            JOIN "Model" m ON ra."modelId" = m."id"
            WHERE m."huggingFaceId" = 'openai/gpt-oss-20b'
            ORDER BY RANDOM()
            LIMIT 5
        ''')
        print("\nSample records:")
        for row in cur.fetchall():
            print(f"  id={row[0]}, layer={row[1]}, positive={row[2]}, size={row[3]}, model={row[4]}")

    conn.close()
    print()
    print("=== Verification Complete ===")

if __name__ == "__main__":
    main()
