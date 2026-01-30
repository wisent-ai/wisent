#!/usr/bin/env python3
"""Delete all truthfulqa_custom data from database."""
import psycopg2

DB_CONFIG = {
    "host": "aws-0-eu-west-2.pooler.supabase.com",
    "port": 5432,
    "database": "postgres",
    "user": "postgres.rbqjqnouluslojmmnuqi",
    "password": "BsKuEnPFLCFurN4a",
}

conn = psycopg2.connect(**DB_CONFIG)
conn.autocommit = True
cur = conn.cursor()

cur.execute("SET statement_timeout = 0")

print("Deleting from Activation table (sets 250, 251)...", flush=True)
cur.execute('DELETE FROM "Activation" WHERE "contrastivePairSetId" IN (250, 251)')
print(f"Deleted {cur.rowcount} rows from Activation", flush=True)

print("Deleting from RawActivation table...", flush=True)
cur.execute('DELETE FROM "RawActivation" WHERE "contrastivePairSetId" IN (250, 251)')
print(f"Deleted {cur.rowcount} rows from RawActivation", flush=True)

print("Deleting from ContrastivePair table...", flush=True)
cur.execute('DELETE FROM "ContrastivePair" WHERE "setId" IN (250, 251)')
print(f"Deleted {cur.rowcount} rows from ContrastivePair", flush=True)

print("Deleting from ContrastivePairSet table...", flush=True)
cur.execute('DELETE FROM "ContrastivePairSet" WHERE id IN (250, 251)')
print(f"Deleted {cur.rowcount} rows from ContrastivePairSet", flush=True)

cur.close()
conn.close()
print("Done.", flush=True)
