#!/usr/bin/env python3
"""Test script to verify Supabase write permissions."""
import psycopg2
import os

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:6543/postgres")

print(f"Testing connection to: {DATABASE_URL[:50]}...")

# Test 1: Basic connection
print("\n=== Test 1: Basic connection (no options) ===")
try:
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute("SHOW default_transaction_read_only")
    print(f"default_transaction_read_only: {cur.fetchone()[0]}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

# Test 2: With options parameter
print("\n=== Test 2: With options=-c default_transaction_read_only=off ===")
try:
    conn = psycopg2.connect(DATABASE_URL, options="-c default_transaction_read_only=off")
    cur = conn.cursor()
    cur.execute("SHOW default_transaction_read_only")
    print(f"default_transaction_read_only: {cur.fetchone()[0]}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

# Test 3: Session pooler with options
print("\n=== Test 3: Session pooler (port 5432) with options ===")
session_url = DATABASE_URL.replace(":6543", ":5432")
try:
    conn = psycopg2.connect(session_url, options="-c default_transaction_read_only=off")
    cur = conn.cursor()
    cur.execute("SHOW default_transaction_read_only")
    print(f"default_transaction_read_only: {cur.fetchone()[0]}")
    try:
        cur.execute("CREATE TEMP TABLE test_write (id int)")
        print("CREATE TEMP TABLE: SUCCESS")
    except Exception as e:
        print(f"CREATE TEMP TABLE: FAILED - {e}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

# Test 4: SET after connect + autocommit
print("\n=== Test 4: SET after connect + autocommit ===")
try:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET default_transaction_read_only = off")
    cur.execute("SHOW default_transaction_read_only")
    print(f"default_transaction_read_only: {cur.fetchone()[0]}")
    try:
        cur.execute("CREATE TEMP TABLE test_write2 (id int)")
        print("CREATE TEMP TABLE: SUCCESS")
    except Exception as e:
        print(f"CREATE TEMP TABLE: FAILED - {e}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

# Test 5: Session pooler with SET + autocommit
print("\n=== Test 5: Session pooler with SET + autocommit ===")
try:
    conn = psycopg2.connect(session_url)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET default_transaction_read_only = off")
    cur.execute("SHOW default_transaction_read_only")
    print(f"default_transaction_read_only: {cur.fetchone()[0]}")
    try:
        cur.execute("CREATE TEMP TABLE test_write3 (id int)")
        print("CREATE TEMP TABLE: SUCCESS")
    except Exception as e:
        print(f"CREATE TEMP TABLE: FAILED - {e}")
    conn.close()
except Exception as e:
    print(f"Error: {e}")

print("\n=== Tests complete ===")
