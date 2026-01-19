#!/usr/bin/env python3
"""Check what's in the Activation table vs RawActivation table."""
import psycopg2

conn = psycopg2.connect(
    host='aws-0-eu-west-2.pooler.supabase.com',
    port=5432,
    database='postgres',
    user='postgres.rbqjqnouluslojmmnuqi',
    password='BsKuEnPFLCFurN4a'
)
cur = conn.cursor()

# Check Activation table
try:
    cur.execute('SELECT COUNT(*) FROM "Activation"')
    print(f"Activation table: {cur.fetchone()[0]} rows")

    cur.execute('''
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'Activation' ORDER BY ordinal_position
    ''')
    print("Activation columns:", [r[0] for r in cur.fetchall()])
except Exception as e:
    print(f"Activation table error: {e}")

print()

# Check RawActivation table
try:
    cur.execute('SELECT COUNT(*) FROM "RawActivation"')
    print(f"RawActivation table: {cur.fetchone()[0]} rows")

    cur.execute('''
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'RawActivation' ORDER BY ordinal_position
    ''')
    print("RawActivation columns:", [r[0] for r in cur.fetchall()])
except Exception as e:
    print(f"RawActivation table error: {e}")

conn.close()
