#!/usr/bin/env python3
"""Fix benchmarks one by one - extract and store to DB."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import os
import psycopg2
import logging
from datetime import datetime

from fix_benchmark_one_by_one_data import BENCHMARK_FIXES
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

DB_URL = os.environ["DATABASE_URL"]

def get_benchmark_status(conn):
    """Get current status of all benchmarks."""
    cur = conn.cursor()
    cur.execute('''
        SELECT cps.name, COUNT(cp.id) as pair_count
        FROM "ContrastivePairSet" cps
        LEFT JOIN "ContrastivePair" cp ON cps.id = cp."setId"
        GROUP BY cps.id, cps.name
        ORDER BY COUNT(cp.id) ASC
    ''')
    return cur.fetchall()


def extract_pairs(task_name: str, limit: int = 500):
    """Extract pairs using the appropriate extractor."""
    from wisent.extractors.lm_eval.lm_task_pairs_generation import build_contrastive_pairs
    return build_contrastive_pairs(task_name, limit=limit)


def store_pairs(conn, db_name: str, pairs, limit: int = 500):
    """Store extracted pairs to DB."""
    cur = conn.cursor()

    # Get set_id
    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (db_name,))
    result = cur.fetchone()
    if not result:
        log.warning(f"Benchmark {db_name} not found in DB")
        return 0
    set_id = result[0]

    # Get current count
    cur.execute('SELECT COUNT(*) FROM "ContrastivePair" WHERE "setId" = %s', (set_id,))
    current_count = cur.fetchone()[0]

    if current_count >= limit:
        log.info(f"Already has {current_count} pairs, skipping")
        return 0

    # Only add new pairs (skip existing ones)
    pairs_to_add = pairs[current_count:limit]
    if not pairs_to_add:
        log.info(f"No new pairs to add (extracted {len(pairs)}, have {current_count})")
        return 0

    log.info(f"Adding {len(pairs_to_add)} new pairs (current: {current_count})")

    count = 0
    for i, pair in enumerate(pairs_to_add):
        try:
            prompt = pair.prompt
            pos = pair.positive_response.model_response if hasattr(pair.positive_response, 'model_response') else str(pair.positive_response)
            neg = pair.negative_response.model_response if hasattr(pair.negative_response, 'model_response') else str(pair.negative_response)

            positive_text = f'{prompt}\n\n{pos}'
            negative_text = f'{prompt}\n\n{neg}'

            cur.execute('''
                INSERT INTO "ContrastivePair" ("setId", "positiveExample", "negativeExample", "category", "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, NOW(), NOW())
            ''', (set_id, positive_text[:65000], negative_text[:65000], f'pair_{current_count + i}'))
            count += 1
        except Exception as e:
            log.warning(f"Failed to store pair {i}: {e}")

    conn.commit()
    log.info(f"=> Stored {count} pairs (now has {current_count + count})")
    return count


def fix_single_benchmark(conn, db_name: str, task_name: str, limit: int = 500):
    """Fix a single benchmark - extract and store."""
    log.info(f"\n{'='*60}")
    log.info(f"FIXING: {db_name} (task: {task_name})")
    log.info(f"{'='*60}")

    # Check current status
    cur = conn.cursor()
    cur.execute('''
        SELECT COUNT(*) FROM "ContrastivePair" cp
        JOIN "ContrastivePairSet" cps ON cp."setId" = cps.id
        WHERE cps.name = %s
    ''', (db_name,))
    current = cur.fetchone()[0]
    log.info(f"Current pairs in DB: {current}")

    if current >= limit:
        log.info(f"Already at target ({limit}), skipping")
        return 0

    # Extract pairs
    try:
        log.info(f"Extracting pairs from task: {task_name}")
        pairs = extract_pairs(task_name, limit)
        log.info(f"Extracted {len(pairs)} pairs")

        if len(pairs) == 0:
            log.warning("No pairs extracted!")
            return 0

        # Log sample
        if pairs:
            sample = pairs[0]
            log.info(f"Sample prompt: {sample.prompt[:100]}...")
            log.info(f"Sample positive: {sample.positive_response.model_response[:50]}...")

    except Exception as e:
        log.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

    # Store pairs
    stored = store_pairs(conn, db_name, pairs, limit)

    # Verify final count
    cur.execute('''
        SELECT COUNT(*) FROM "ContrastivePair" cp
        JOIN "ContrastivePairSet" cps ON cp."setId" = cps.id
        WHERE cps.name = %s
    ''', (db_name,))
    final = cur.fetchone()[0]
    log.info(f"Final pairs in DB: {final}")

    return stored

def print_status_report(conn):
    """Print a status report of all benchmarks."""
    status = get_benchmark_status(conn)

    at_500 = sum(1 for _, count in status if count >= 500)
    under_500 = [(name, count) for name, count in status if count < 500]

    log.info(f"\n{'='*60}")
    log.info("BENCHMARK STATUS REPORT")
    log.info(f"{'='*60}")
    log.info(f"Total benchmarks: {len(status)}")
    log.info(f"At 500 pairs: {at_500}")
    log.info(f"Under 500: {len(under_500)}")
    log.info(f"Progress: {at_500}/{len(status)} = {at_500*100/len(status):.1f}%")

    log.info(f"\n{'='*60}")
    log.info("BENCHMARKS UNDER 500 PAIRS:")
    log.info(f"{'='*60}")

    # Group by category
    categories = {}
    for name, count in under_500:
        if '/' in name:
            cat = name.split('/')[0]
        else:
            cat = 'other'
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, count))

    for cat in sorted(categories.keys()):
        items = categories[cat]
        log.info(f"\n{cat}: {len(items)} benchmarks")
        for name, count in sorted(items, key=lambda x: x[1]):
            in_mapping = "✓" if name in BENCHMARK_FIXES else " "
            log.info(f"  [{in_mapping}] {name}: {count}")


def main():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = True

    if len(sys.argv) > 1:
        if sys.argv[1] == "--status":
            # Print status report
            print_status_report(conn)
        elif sys.argv[1] == "--all":
            # Fix all benchmarks in the mapping
            total = 0
            fixed = 0

            for db_name, task_name in BENCHMARK_FIXES.items():
                count = fix_single_benchmark(conn, db_name, task_name)
                if count > 0:
                    fixed += 1
                total += count

            log.info(f"\n{'='*60}")
            log.info(f"COMPLETE! Fixed {fixed} benchmarks, added {total} pairs total")
            print_status_report(conn)
        else:
            # Fix specific benchmark(s) passed as arguments
            for arg in sys.argv[1:]:
                if arg in BENCHMARK_FIXES:
                    fix_single_benchmark(conn, arg, BENCHMARK_FIXES[arg])
                else:
                    log.error(f"Unknown benchmark: {arg}")
                    log.info(f"Known benchmarks: {list(BENCHMARK_FIXES.keys())}")
    else:
        # Print usage
        log.info("Usage:")
        log.info("  python fix_benchmark_one_by_one.py --status       # Print status report")
        log.info("  python fix_benchmark_one_by_one.py --all          # Fix all benchmarks")
        log.info("  python fix_benchmark_one_by_one.py <db_name>      # Fix specific benchmark")
        log.info(f"\nKnown benchmarks: {list(BENCHMARK_FIXES.keys())}")

    conn.close()


if __name__ == "__main__":
    main()
