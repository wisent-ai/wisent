#!/usr/bin/env python3
"""Fix all remaining benchmarks with < 500 pairs."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# Benchmarks to fix with lm-eval extractors
LM_EVAL_BENCHMARKS = [
    ("commonsense/anli", "anli"),
    ("math/gsm8k", "gsm8k"),
    ("math/arithmetic", "arithmetic"),
    ("reading_comprehension/race", "race"),
    ("science_medical/sciq", "sciq"),
    ("safety_bias/crows_pairs", "crows_pairs"),
    ("multilingual/kobest", "kobest"),
    ("multilingual/csatqa", "csatqa"),
    ("multilingual/catalan_bench", "catalan_bench"),
    ("multilingual/galician_bench", "galician_bench"),
    ("multilingual/spanish_bench", "spanish_bench"),
    ("multilingual/french_bench", "french_bench"),
    ("multilingual/aexams", "aexams"),
    ("multilingual/AraDiCE", "aradice"),
    ("multilingual/arabic_leaderboard_complete", "arabic_leaderboard_complete"),
    ("multilingual/arabic_leaderboard_light", "arabic_leaderboard_light"),
    ("multilingual/global_mmlu", "global_mmlu"),
    ("multilingual/belebele", "belebele"),
    ("multilingual/ceval", "ceval"),
    ("multilingual/cmmlu", "cmmlu"),
    ("multilingual/tmmluplus", "tmmluplus"),
    ("multilingual/arabicmmlu", "arabicmmlu"),
    ("multilingual/xwinograd", "xwinograd"),
    ("multilingual/xcopa", "xcopa"),
    ("multilingual/xnli", "xnli"),
    ("multilingual/xstorycloze", "xstorycloze"),
    ("multilingual/darijammlu", "darijammlu"),
    ("multilingual/afrimmlu_direct_amh", "afrimmlu"),
    ("multilingual/ArabCulture", "arabculture"),
    ("multilingual/bertaqa", "bertaqa"),
    ("knowledge_qa/mmlu", "mmlu"),
    ("knowledge_qa/mmlu_pro", "mmlu"),
    ("knowledge_qa/mmlu-pro-plus", "mmlu"),
    ("knowledge_qa/mmlu_prox", "mmlu"),
    ("knowledge_qa/aclue", "aclue"),
    ("knowledge_qa/metabench", "metabench"),
    ("knowledge_qa/openbookqa", "openbookqa"),
    ("language_understanding/blimp", "blimp"),
    ("language_understanding/paloma", "paloma"),
    ("language_understanding/lambada_cloze", "lambada_cloze"),
    ("reasoning_logic/bbh", "bbh"),
    ("reasoning_logic/score", "score"),
    ("reasoning_logic/mastermind", "mastermind"),
    ("reasoning_logic/lingoly", "lingoly"),
    ("reasoning_logic/acp_bench", "acp_bench"),
    ("ethics_values/model_written_evals", "model_written_evals"),
    ("safety_bias/simple_cooccurrence_bias", "simple_cooccurrence_bias"),
    ("safety_bias/wmdp", "wmdp"),
    ("science_medical/headqa", "headqa"),
    ("science_medical/gpqa", "gpqa"),
    ("hallucination_factuality/truthfulqa", "truthfulqa"),
    ("math/hendrycks_math", "hendrycks_math"),
    ("math/afrimgsm_direct_amh", "afrimgsm"),
    ("commonsense/wsc273", "wsc273"),
    ("commonsense/prost", "prost"),
    ("commonsense/winogender", "winogender"),
]

def fix_with_lm_eval(conn, db_name: str, lm_task: str, limit: int = 500):
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

    for i, (db_name, lm_task) in enumerate(LM_EVAL_BENCHMARKS):
        log.info(f"\n[{i+1}/{len(LM_EVAL_BENCHMARKS)}] {db_name} -> {lm_task}")
        count = fix_with_lm_eval(conn, db_name, lm_task)
        if count > 0:
            fixed += 1
        total += count

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE! Fixed {fixed} benchmarks, added {total} pairs total")
    conn.close()

if __name__ == "__main__":
    main()
