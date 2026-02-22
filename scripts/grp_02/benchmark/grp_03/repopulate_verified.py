#!/usr/bin/env python3
"""Repopulate benchmarks with verified working extractors."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# Verified working extractors: (db_name, extractor_type, extractor_name)
# extractor_type: "lm_eval" or "hf"
VERIFIED_BENCHMARKS = [
    # Reading comprehension - verified working
    ("reading_comprehension/drop", "lm_eval", "drop"),
    ("reading_comprehension/squad_completion", "lm_eval", "squad_completion"),
    ("reading_comprehension/race", "lm_eval", "race"),

    # French bench - fixed
    ("multilingual/french_bench", "lm_eval", "french_bench_hellaswag"),

    # Ethics - verified working
    ("ethics_values/hendrycks_ethics", "lm_eval", "hendrycks_ethics"),

    # Commonsense - verified working
    ("commonsense/wsc273", "lm_eval", "wsc273"),
    ("commonsense/winogender", "lm_eval", "winogender_mc"),

    # Multilingual - verified working
    ("multilingual/xquad", "lm_eval", "xquad"),

    # Others that were under-populated
    ("math/gsm8k", "lm_eval", "gsm8k"),
    ("math/arithmetic", "lm_eval", "arithmetic"),
    ("science_medical/sciq", "lm_eval", "sciq"),
    ("safety_bias/crows_pairs", "lm_eval", "crows_pairs"),
]


def extract_pairs(extractor_type: str, extractor_name: str, limit: int = 500):
    """Extract pairs using the specified extractor."""
    if extractor_type == "lm_eval":
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
        return build_contrastive_pairs(extractor_name, limit=limit)
    else:
        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_registry import get_extractor
        extractor = get_extractor(extractor_name)
        return extractor.extract_contrastive_pairs(limit=limit)


def repopulate_benchmark(conn, db_name: str, extractor_type: str, extractor_name: str, limit: int = 500):
    """Repopulate a benchmark - delete old pairs and add new ones."""
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

    log.info(f"Current count: {current_count}")

    # Extract new pairs
    try:
        pairs = extract_pairs(extractor_type, extractor_name, limit)
        log.info(f"Extracted {len(pairs)} pairs")
    except Exception as e:
        log.error(f"Failed to extract: {e}")
        return 0

    if not pairs or len(pairs) <= current_count:
        log.info(f"No improvement possible (extracted {len(pairs)}, have {current_count})")
        return 0

    # Delete old pairs (only if we have more new ones)
    try:
        cur.execute('DELETE FROM "ContrastivePair" WHERE "setId" = %s', (set_id,))
        log.info(f"Deleted {current_count} old pairs")
    except Exception as e:
        # Foreign key constraint - skip deletion, just add more
        log.warning(f"Cannot delete (foreign key): {e}")
        pairs = pairs[current_count:limit]
        if not pairs:
            return 0

    # Add new pairs
    count = 0
    start_idx = 0 if current_count == 0 else current_count

    for i, pair in enumerate(pairs):
        try:
            prompt = pair.prompt
            pos = pair.positive_response.model_response if hasattr(pair.positive_response, 'model_response') else str(pair.positive_response)
            neg = pair.negative_response.model_response if hasattr(pair.negative_response, 'model_response') else str(pair.negative_response)

            positive_text = f'{prompt}\n\n{pos}'
            negative_text = f'{prompt}\n\n{neg}'

            cur.execute('''
                INSERT INTO "ContrastivePair" ("setId", "positiveExample", "negativeExample", "category", "createdAt", "updatedAt")
                VALUES (%s, %s, %s, %s, NOW(), NOW())
            ''', (set_id, positive_text[:65000], negative_text[:65000], f'pair_{start_idx + i}'))
            count += 1
        except Exception as e:
            log.warning(f"Failed to store pair {i}: {e}")

    conn.commit()

    # Get final count
    cur.execute('SELECT COUNT(*) FROM "ContrastivePair" WHERE "setId" = %s', (set_id,))
    final_count = cur.fetchone()[0]

    log.info(f"=> Added {count} pairs (now has {final_count})")
    return count


def main():
    conn = psycopg2.connect('postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres')
    conn.autocommit = True

    total = 0
    fixed = 0

    for i, (db_name, ext_type, ext_name) in enumerate(VERIFIED_BENCHMARKS):
        log.info(f"\n[{i+1}/{len(VERIFIED_BENCHMARKS)}] {db_name} -> {ext_name}")
        count = repopulate_benchmark(conn, db_name, ext_type, ext_name)
        if count > 0:
            fixed += 1
        total += count

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE! Fixed {fixed} benchmarks, added {total} pairs total")
    conn.close()


if __name__ == "__main__":
    main()
