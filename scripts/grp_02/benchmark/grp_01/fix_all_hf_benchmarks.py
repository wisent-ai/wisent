#!/usr/bin/env python3
"""Fix all benchmarks that have HF extractors - run with nohup."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# All benchmarks with HF extractors that have < 500 pairs
BENCHMARKS_TO_FIX = [
    ("math/cnmo_2024", "cnmo_2024"),
    ("math/cnmo", "cnmo"),
    ("math/livemathbench_cnmo_en", "livemathbench_cnmo_en"),
    ("math/hmmt_feb_2025", "hmmt_feb_2025"),
    ("math/hmmt", "hmmt"),
    ("coding/aider_polyglot", "aider_polyglot"),
    ("coding/scicode", "scicode"),
    ("multilingual/Tag", "tag"),
    ("reasoning_logic/babilong", "babilong"),
    ("tool_use_agents/seal_0", "seal_0"),
    ("tool_use_agents/seal", "seal"),
    ("coding/terminal_bench", "terminal_bench"),
    ("math/polymath_zh_medium", "polymath_zh_medium"),
    ("tool_use_agents/toolemu", "toolemu"),
    ("coding/multiple_cpp", "multiple_cpp"),
    ("coding/multiple_java", "multiple_java"),
    ("coding/multiple_go", "multiple_go"),
    ("coding/multiple_rs", "multiple_rs"),
    ("coding/multipl_e", "multipl_e"),
    ("coding/multiple_js", "multiple_js"),
    ("coding/multiple_py", "multiple_py"),
    ("coding/recode", "recode"),
    ("coding/humaneval_plus", "humaneval_plus"),
    ("coding/humanevalpack", "humanevalpack"),
    ("coding/humaneval", "humaneval"),
    ("tool_use_agents/tau_bench", "tau_bench"),
    ("coding/oj_bench", "oj_bench"),
    ("safety_bias/refusalbench", "refusalbench"),
    ("instruction_following/travelplanner", "travelplanner"),
    ("math/polymath_en_medium", "polymath_en_medium"),
    ("reasoning_logic/frames", "frames"),
    ("multilingual/lambada_multilingual", "lambada_multilingual"),
    ("math/polymath_zh_high", "polymath_zh_high"),
    ("multilingual/eus_exams", "eus_exams"),
    ("math/polymath_en_high", "polymath_en_high"),
    ("coding/mercury", "mercury"),
    ("reasoning_logic/inverse_scaling", "inverse_scaling"),
    ("safety_bias/curate", "curate"),
    ("instruction_following/mmmu", "mmmu"),
    ("hallucination_factuality/faithbench", "faithbench"),
    ("science_medical/healthbench", "healthbench"),
    ("knowledge_qa/mmlusr", "mmlusr"),
    ("coding/livecodebench_v6", "livecodebench_v6"),
    ("coding/livecodebench_v5", "livecodebench_v5"),
    ("multilingual/bangla_mmlu", "bangla_mmlu"),
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
    log.info(f"=> Stored {count} pairs for {db_name} (now has {current_count + count})")
    return count

def main():
    conn = psycopg2.connect('postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres')
    conn.autocommit = True

    total = 0
    fixed = 0
    for i, (db_name, extractor_name) in enumerate(BENCHMARKS_TO_FIX):
        log.info(f"\n[{i+1}/{len(BENCHMARKS_TO_FIX)}] {db_name} -> {extractor_name}")
        count = fix_benchmark(conn, db_name, extractor_name)
        if count > 0:
            fixed += 1
        total += count

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE! Fixed {fixed} benchmarks, added {total} pairs total")
    conn.close()

if __name__ == "__main__":
    main()
