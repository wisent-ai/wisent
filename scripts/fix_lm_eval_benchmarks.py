#!/usr/bin/env python3
"""Fix benchmarks using lm-eval extractors - run with nohup."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# Benchmarks with lm-eval mappings that need more pairs (< 400)
# Format: (db_name, lm_eval_task)
BENCHMARKS_TO_FIX = [
    ("multilingual/evalita_LLM", "evalita_llm"),
    ("multilingual/french_bench", "french_bench"),
    ("multilingual/kobest", "kobest"),
    ("multilingual/csatqa", "csatqa"),
    ("multilingual/catalan_bench", "catalan_bench"),
    ("knowledge_qa/metabench", "metabench"),
    ("reasoning_logic/score", "score"),
    ("language_understanding/paloma", "paloma"),
    ("safety_bias/simple_cooccurrence_bias", "simple_cooccurrence_bias"),
    ("multilingual/galician_bench", "galician_bench"),
    ("multilingual/aexams", "aexams"),
    ("multilingual/AraDiCE", "aradice"),
    ("multilingual/spanish_bench", "spanish_bench"),
    ("multilingual/arabic_leaderboard_complete", "arabic_leaderboard_complete"),
    ("multilingual/arabic_leaderboard_light", "arabic_leaderboard_light"),
    ("ethics_values/model_written_evals", "model_written_evals"),
    ("science_medical/headqa", "headqa"),
]

def fix_benchmark(conn, db_name: str, lm_task: str, limit: int = 500):
    """Add missing pairs using lm-eval extractor."""
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

    # Get pairs using lm-eval
    try:
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
        pairs = build_contrastive_pairs(lm_task, limit=limit)
        log.info(f"Extracted {len(pairs)} pairs for {db_name}")
    except Exception as e:
        log.error(f"Failed to extract pairs for {db_name}: {e}")
        return 0

    if not pairs:
        log.info(f"No pairs generated for {db_name}")
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
    log.info(f"=> Stored {count} pairs for {db_name} (now has {current_count + count})")
    return count

def main():
    conn = psycopg2.connect('postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres')
    conn.autocommit = True

    total = 0
    fixed = 0
    for i, (db_name, lm_task) in enumerate(BENCHMARKS_TO_FIX):
        log.info(f"\n[{i+1}/{len(BENCHMARKS_TO_FIX)}] {db_name} -> {lm_task}")
        count = fix_benchmark(conn, db_name, lm_task)
        if count > 0:
            fixed += 1
        total += count

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE! Fixed {fixed} benchmarks, added {total} pairs total")
    conn.close()

if __name__ == "__main__":
    main()
