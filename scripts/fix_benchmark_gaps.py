#!/usr/bin/env python3
"""Fix benchmarks with extraction gaps - run with nohup."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# Benchmarks to fix with their HF extractor names
BENCHMARKS_TO_FIX = [
    ("coding/livecodebench_lite", "livecodebench_lite"),
    ("instruction_following/arena_hard", "arena_hard"),
    ("multilingual/basqueglue", "basqueglue"),
    ("reasoning_logic/planbench", "planbench"),
    ("math/math", "math"),
    ("hallucination_factuality/faithbench", "faithbench"),
    ("coding/oj_bench", "oj_bench"),
    ("math/aime", "aime"),
    ("reasoning_logic/babilong", "babilong"),
    ("science_medical/healthbench", "healthbench"),
    ("coding/livecodebench_v5", "livecodebench"),
    ("hallucination_factuality/chinese_simpleqa", "chinese_simpleqa"),
]

def fix_benchmark(conn, db_name: str, extractor_name: str, limit: int = 500):
    """Add missing pairs using HF extractor (don't delete existing)."""
    cur = conn.cursor()

    # Get set_id and current count
    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (db_name,))
    result = cur.fetchone()
    if not result:
        log.warning(f"Benchmark {db_name} not found in DB")
        return 0
    set_id = result[0]

    # Get current pair count
    cur.execute('SELECT COUNT(*) FROM "ContrastivePair" WHERE "setId" = %s', (set_id,))
    current_count = cur.fetchone()[0]

    if current_count >= limit:
        log.info(f"Already has {current_count} pairs, skipping {db_name}")
        return 0

    needed = limit - current_count
    log.info(f"Has {current_count} pairs, need {needed} more for {db_name}")

    # Get extractor
    try:
        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_registry import get_extractor
        extractor = get_extractor(extractor_name)
    except Exception as e:
        log.error(f"Failed to get extractor {extractor_name}: {e}")
        return 0

    # Extract pairs - get more than needed to skip existing ones
    try:
        pairs = extractor.extract_contrastive_pairs(limit=limit)
        log.info(f"Extracted {len(pairs)} pairs for {db_name}")
    except Exception as e:
        log.error(f"Failed to extract pairs for {db_name}: {e}")
        return 0

    # Skip existing pairs, only add new ones
    pairs_to_add = pairs[current_count:limit]
    if not pairs_to_add:
        log.info(f"No new pairs to add for {db_name}")
        return 0

    log.info(f"Adding {len(pairs_to_add)} new pairs for {db_name}")

    # Store pairs
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
    log.info(f"✓ Stored {count} pairs for {db_name}")
    return count

def main():
    conn = psycopg2.connect('postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres')
    conn.autocommit = True

    total = 0
    for db_name, extractor_name in BENCHMARKS_TO_FIX:
        log.info(f"\n{'='*60}")
        log.info(f"Fixing: {db_name} -> {extractor_name}")
        count = fix_benchmark(conn, db_name, extractor_name)
        total += count

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE! Total: {total} pairs created")
    conn.close()

if __name__ == "__main__":
    main()
