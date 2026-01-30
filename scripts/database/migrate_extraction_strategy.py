#!/usr/bin/env python3
"""
Database migration: Rename 'chat_last' to 'completion_last' in Activation table.

The existing activations were extracted using completion format (prompt\n\nresponse)
but were incorrectly labeled as 'chat_last'. This migration fixes the naming.

Run with: python scripts/database/migrate_extraction_strategy.py
"""

import os
import psycopg2

DATABASE_URL = os.environ.get("DATABASE_URL")
if DATABASE_URL and '?' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.split('?')[0]

if not DATABASE_URL:
    DATABASE_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'


def get_db_connection():
    db_url = DATABASE_URL
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(db_url, connect_timeout=30)
    return conn


def main():
    print("Connecting to database...", flush=True)
    conn = get_db_connection()
    cur = conn.cursor()

    # Check current counts
    cur.execute('''
        SELECT "extractionStrategy", COUNT(*)
        FROM "Activation"
        GROUP BY "extractionStrategy"
    ''')
    print("\nCurrent extraction strategy counts:", flush=True)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}", flush=True)

    # Count records to migrate
    cur.execute('''
        SELECT COUNT(*) FROM "Activation" WHERE "extractionStrategy" = 'chat_last'
    ''')
    count = cur.fetchone()[0]
    print(f"\nRecords to migrate (chat_last -> completion_last): {count}", flush=True)

    if count == 0:
        print("No records to migrate. Done.", flush=True)
        conn.close()
        return

    # Confirm
    response = input("\nProceed with migration? (yes/no): ")
    if response.lower() != 'yes':
        print("Aborted.", flush=True)
        conn.close()
        return

    # Perform migration
    print("\nMigrating...", flush=True)
    cur.execute('''
        UPDATE "Activation"
        SET "extractionStrategy" = 'completion_last'
        WHERE "extractionStrategy" = 'chat_last'
    ''')
    conn.commit()
    print(f"  Updated {cur.rowcount} records", flush=True)

    # Verify
    cur.execute('''
        SELECT "extractionStrategy", COUNT(*)
        FROM "Activation"
        GROUP BY "extractionStrategy"
    ''')
    print("\nNew extraction strategy counts:", flush=True)
    for row in cur.fetchall():
        print(f"  {row[0]}: {row[1]}", flush=True)

    cur.close()
    conn.close()
    print("\nMigration complete.", flush=True)


if __name__ == "__main__":
    main()
