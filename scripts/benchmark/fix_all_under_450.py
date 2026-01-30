#!/usr/bin/env python3
"""Fix all benchmarks under 450 pairs - try both HF and lm-eval extractors."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

# Mapping of db_name -> (hf_extractor, lm_eval_task)
# None means not available
BENCHMARK_EXTRACTORS = {
    "ethics_values/hendrycks_ethics": (None, "hendrycks_ethics"),
    "reading_comprehension/qasper": (None, "qasper"),
    "multilingual/kbl": (None, None),
    "multilingual/xnli_eu": (None, "xnli_eu"),
    "multilingual/polemo2": (None, "polemo2"),
    "instruction_following/groundcocoa": ("groundcocoa", None),
    "math/cnmo_2024": ("cnmo_2024", None),
    "math/cnmo": ("cnmo", None),
    "multilingual/evalita_LLM": (None, "evalita_llm"),
    "math/livemathbench_cnmo_en": ("livemathbench_cnmo_en", None),
    "math/hmmt_feb_2025": ("hmmt_feb_2025", None),
    "math/hmmt": ("hmmt", None),
    "coding/aider_polyglot": ("aider_polyglot", None),
    "multilingual/Tag": ("tag", "tag"),
    "coding/scicode": ("scicode", None),
    "multilingual/french_bench": (None, "french_bench"),
    "reasoning_logic/babilong": ("babilong", None),
    "reasoning_logic/babi": ("babilong", "babi"),
    "multilingual/kobest": (None, "kobest"),
    "reasoning_logic/acp_bench": (None, "acp_bench"),
    "tool_use_agents/seal_0": ("seal_0", None),
    "tool_use_agents/seal": ("seal", None),
    "coding/terminal_bench": ("terminal_bench", None),
    "math/polymath_zh_medium": ("polymath_zh_medium", None),
    "reading_comprehension/qa4mre": (None, "qa4mre"),
    "instruction_following/eq_bench": ("eq_bench", None),
    "tool_use_agents/toolemu": ("toolemu", None),
    "multilingual/csatqa": (None, "csatqa"),
    "coding/multiple_js": ("multiple_js", "multiple-js"),
    "coding/multiple_rs": ("multiple_rs", "multiple-rs"),
    "coding/multiple_go": ("multiple_go", "multiple-go"),
    "coding/multiple_java": ("multiple_java", "multiple-java"),
    "coding/multipl_e": ("multipl_e", "multiple-py"),
    "coding/multiple_cpp": ("multiple_cpp", "multiple-cpp"),
    "coding/multiple_py": ("multiple_py", "multiple-py"),
    "coding/recode": ("recode", None),
    "coding/humaneval": ("humaneval", "humaneval"),
    "coding/humaneval_plus": ("humaneval_plus", None),
    "coding/humanevalpack": ("humanevalpack", None),
    "tool_use_agents/tau_bench": ("tau_bench", None),
    "coding/oj_bench": ("oj_bench", None),
    "reading_comprehension/swde": (None, "swde"),
    "safety_bias/refusalbench": ("refusalbench", None),
    "instruction_following/travelplanner": ("travelplanner", None),
    "math/polymath_en_medium": ("polymath_en_medium", None),
    "multilingual/xquad": (None, "xquad"),
    "reading_comprehension/squad_completion": (None, "squad_completion"),
    "reading_comprehension/c4": (None, None),
    "multilingual/catalan_bench": (None, "catalan_bench"),
    "math/polymath_zh_high": ("polymath_zh_high", None),
    "commonsense/winogender": (None, "winogender"),
    "multilingual/eus_exams": ("eus_exams", None),
    "math/polymath_en_high": ("polymath_en_high", None),
    "knowledge_qa/metabench": ("metabench", "metabench"),
    "coding/mercury": ("mercury", None),
    "reasoning_logic/score": (None, "score"),
    "language_understanding/paloma": (None, "paloma"),
    "safety_bias/simple_cooccurrence_bias": (None, "simple_cooccurrence_bias"),
    "multilingual/eus_reading": ("eus_reading", None),
    "multilingual/galician_bench": (None, "galician_bench"),
    "reasoning_logic/inverse_scaling": ("inverse_scaling", None),
    "multilingual/aexams": (None, "aexams"),
    "commonsense/wsc273": (None, "wsc273"),
    "safety_bias/curate": ("curate", None),
    "multilingual/AraDiCE": (None, "aradice"),
    "multilingual/spanish_bench": (None, "spanish_bench"),
    "multilingual/arabic_leaderboard_complete": (None, "arabic_leaderboard_complete"),
    "multilingual/arabic_leaderboard_light": (None, "arabic_leaderboard_light"),
    "ethics_values/model_written_evals": (None, "model_written_evals"),
    "instruction_following/mmmu": ("mmmu", None),
    "hallucination_factuality/faithbench": ("faithbench", None),
    "science_medical/headqa": (None, "headqa"),
    "science_medical/healthbench": ("healthbench", None),
    "multilingual/eus_trivia": ("eus_trivia", None),
}

def try_hf_extractor(extractor_name: str, limit: int = 500):
    """Try to extract pairs using HF extractor."""
    try:
        from wisent.core.contrastive_pairs.huggingface_pairs.hf_extractor_registry import get_extractor
        extractor = get_extractor(extractor_name)
        pairs = extractor.extract_contrastive_pairs(limit=limit)
        return pairs
    except Exception as e:
        log.warning(f"HF extractor '{extractor_name}' failed: {e}")
        return None

def try_lm_eval_extractor(task_name: str, limit: int = 500):
    """Try to extract pairs using lm-eval extractor."""
    try:
        from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
        pairs = build_contrastive_pairs(task_name, limit=limit)
        return pairs
    except Exception as e:
        log.warning(f"lm-eval extractor '{task_name}' failed: {e}")
        return None

def fix_benchmark(conn, db_name: str, hf_extractor: str, lm_eval_task: str, limit: int = 500):
    """Fix a single benchmark."""
    cur = conn.cursor()

    # Get set_id and current count
    cur.execute('SELECT id FROM "ContrastivePairSet" WHERE name = %s', (db_name,))
    result = cur.fetchone()
    if not result:
        log.warning(f"Benchmark {db_name} not found in DB")
        return 0
    set_id = result[0]

    cur.execute('SELECT COUNT(*) FROM "ContrastivePair" WHERE "setId" = %s', (set_id,))
    current_count = cur.fetchone()[0]

    if current_count >= limit:
        log.info(f"Already has {current_count} pairs, skipping")
        return 0

    needed = limit - current_count
    log.info(f"Has {current_count} pairs, need {needed} more")

    # Try extractors
    pairs = None

    if hf_extractor:
        log.info(f"Trying HF extractor: {hf_extractor}")
        pairs = try_hf_extractor(hf_extractor, limit)

    if not pairs and lm_eval_task:
        log.info(f"Trying lm-eval extractor: {lm_eval_task}")
        pairs = try_lm_eval_extractor(lm_eval_task, limit)

    if not pairs:
        log.warning(f"No pairs extracted for {db_name}")
        return 0

    log.info(f"Extracted {len(pairs)} pairs")

    # Skip existing, add new
    pairs_to_add = pairs[current_count:limit]
    if not pairs_to_add:
        log.info(f"No new pairs to add (extractor returned {len(pairs)}, already have {current_count})")
        return 0

    log.info(f"Adding {len(pairs_to_add)} new pairs")

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

def main():
    conn = psycopg2.connect('postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres')
    conn.autocommit = True

    total = 0
    fixed = 0

    benchmarks = list(BENCHMARK_EXTRACTORS.items())

    for i, (db_name, (hf_ext, lm_task)) in enumerate(benchmarks):
        log.info(f"\n[{i+1}/{len(benchmarks)}] {db_name}")

        if not hf_ext and not lm_task:
            log.info(f"No extractor available, skipping")
            continue

        count = fix_benchmark(conn, db_name, hf_ext, lm_task)
        if count > 0:
            fixed += 1
        total += count

    log.info(f"\n{'='*60}")
    log.info(f"COMPLETE! Fixed {fixed} benchmarks, added {total} pairs total")
    conn.close()

if __name__ == "__main__":
    main()
