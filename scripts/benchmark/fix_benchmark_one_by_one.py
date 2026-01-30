#!/usr/bin/env python3
"""Fix benchmarks one by one - extract and store to DB."""

import sys
sys.path.insert(0, "/Users/lukaszbartoszcze/Documents/CodingProjects/Wisent/backends/wisent-open-source")

import psycopg2
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger(__name__)

DB_URL = 'postgresql://postgres.rbqjqnouluslojmmnuqi:BsKuEnPFLCFurN4a@aws-0-eu-west-2.pooler.supabase.com:5432/postgres'

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
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs
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


# Mapping of db_name -> task_name for benchmarks that need fixing
# Add entries here as you investigate each benchmark
# Format: "db_name": "lm_eval_task_name",  # current_count (status/notes)
BENCHMARK_FIXES = {
    # ==========================================================================
    # VERIFIED AT 500 OR MAX AVAILABLE
    # ==========================================================================
    "multilingual/kbl": "kbl",  # 448 (max - 64 subtasks)
    "instruction_following/groundcocoa": "groundcocoa",  # 500
    "reading_comprehension/drop": "drop",  # 500
    "reasoning_logic/babi": "babi",  # 500
    "reasoning_logic/babilong": "babilong",  # 500 (loads from 5 QA configs)
    "multilingual/kobest": "kobest",  # 500 (5 format handlers added)
    "commonsense/winogender": "winogender",  # 500 (HF extractor created)
    "reading_comprehension/squad_completion": "squad_completion",  # 500
    "reading_comprehension/race": "race",  # 500
    "math/gsm8k": "gsm8k",  # 500
    "math/arithmetic": "arithmetic",  # 500
    "science_medical/sciq": "sciq",  # 500
    "ethics_values/hendrycks_ethics": "hendrycks_ethics",  # 500
    "multilingual/xquad": "xquad",  # 492 (max - 12 subtasks)
    "multilingual/french_bench": "french_bench_hellaswag",  # 500
    "multilingual/evalita_LLM": "evalita-mp",  # 340 (max for MCQ subtasks)
    "commonsense/wsc273": "wsc273",  # 315 (max - dataset has 273 samples)
    "safety_bias/crows_pairs": "crows_pairs",  # 458 (max - 22 subtasks)
    "multilingual/xnli": "xnli",  # 495 (max - 15 subtasks)
    "multilingual/xstorycloze": "xstorycloze",  # 495 (max - 11 subtasks)
    "multilingual/xcopa": "xcopa",  # 495 (max - 11 subtasks)
    "multilingual/belebele": "belebele",  # 488 (max - 122 subtasks)
    "multilingual/xwinograd": "xwinograd",  # 481 (max - 6 subtasks)
    "reasoning_logic/bbh": "bbh",  # 486 (max - 27 subtasks)
    "reasoning_logic/inverse_scaling": "inverse_scaling",  # 300 (max)
    "reasoning_logic/lingoly": "lingoly",  # 470 (max)
    "knowledge_qa/mmlu": "mmlu",  # 456 (max - 57 subtasks)
    "knowledge_qa/mmlu_pro": "mmlu_pro",  # 490 (max - 14 subtasks)
    "knowledge_qa/aclue": "aclue",  # 495 (max - 15 subtasks)
    "safety_bias/wmdp": "wmdp",  # 498 (max - 3 subtasks)
    "science_medical/gpqa": "gpqa",  # 495 (max - 15 subtasks)
    "science_medical/headqa": "headqa",  # 396 (max - 2 subtasks)
    "hallucination_factuality/truthfulqa": "truthfulqa_mc1",  # 500

    # ==========================================================================
    # MATH BENCHMARKS - TO INVESTIGATE
    # ==========================================================================
    "math/cnmo_2024": "cnmo_2024",  # 18 - investigate
    "math/cnmo": "cnmo",  # 18 - investigate
    "math/livemathbench_cnmo_en": "livemathbench",  # 27 - investigate
    "math/hmmt": "hmmt",  # 30 - investigate
    "math/hmmt_feb_2025": "hmmt",  # 30 - investigate
    "math/polymath_zh_medium": "polymath",  # 125 - investigate
    "math/polymath_en_medium": "polymath",  # 187 - investigate
    "math/polymath_zh_high": "polymath",  # 204 - investigate
    "math/polymath_en_high": "polymath",  # 241 - investigate
    "math/afrimgsm_direct_amh": "afrimgsm_direct_amh",  # 496 - investigate
    "math/hendrycks_math": "hendrycks_math",  # 497 - investigate

    # ==========================================================================
    # CODING BENCHMARKS - TO INVESTIGATE
    # ==========================================================================
    "coding/aider_polyglot": "aider_polyglot",  # 42 - investigate
    "coding/scicode": "scicode",  # 65 - investigate
    "coding/terminal_bench": "terminal_bench",  # 112 - investigate
    "coding/multiple_py": "multiple-py",  # 161 - investigate
    "coding/multiple_rs": "multiple-rs",  # 161 - investigate
    "coding/multiple_js": "multiple-js",  # 161 - investigate
    "coding/multiple_go": "multiple-go",  # 161 - investigate
    "coding/multipl_e": "multiple-py",  # 161 - investigate
    "coding/multiple_java": "multiple-java",  # 161 - investigate
    "coding/multiple_cpp": "multiple-cpp",  # 161 - investigate
    "coding/humaneval": "humaneval",  # 164 - investigate
    "coding/humaneval_plus": "humaneval",  # 164 - investigate
    "coding/recode": "recode",  # 164 - investigate
    "coding/humanevalpack": "humanevalpack",  # 164 - investigate
    "coding/oj_bench": "oj_bench",  # 165 - investigate
    "coding/mercury": "mercury",  # 256 - investigate

    # ==========================================================================
    # MULTILINGUAL BENCHMARKS - TO INVESTIGATE
    # ==========================================================================
    "multilingual/Tag": "tag",  # 60 - investigate
    "multilingual/csatqa": "csatqa",  # 147 - investigate
    "multilingual/catalan_bench": "catalan_bench",  # 204 - investigate
    "multilingual/eus_exams": "eus_exams",  # 236 - investigate
    "multilingual/eus_reading": "eus_reading",  # 280 - investigate
    "multilingual/galician_bench": "galician_bench",  # 289 - investigate
    "multilingual/aexams": "aexams",  # 311 - investigate
    "multilingual/AraDiCE": "aradice",  # 344 - investigate
    "multilingual/spanish_bench": "spanish_bench",  # 355 - investigate
    "multilingual/arabic_leaderboard_light": "arabic_leaderboard_light",  # 372 - investigate
    "multilingual/arabic_leaderboard_complete": "arabic_leaderboard_complete",  # 372 - investigate
    "multilingual/global_mmlu": "global_mmlu",  # 450 - investigate
    "multilingual/eus_proficiency": "eus_proficiency",  # 459 - investigate
    "multilingual/ceval": "ceval-valid",  # 468 - investigate
    "multilingual/cmmlu": "cmmlu",  # 469 - investigate
    "multilingual/tmmluplus": "tmmluplus",  # 469 - investigate
    "multilingual/arabicmmlu": "arabicmmlu",  # 480 - investigate
    "multilingual/afrimmlu_direct_amh": "afrimmlu_direct_amh",  # 484 - investigate
    "multilingual/darijammlu": "darijammlu",  # 484 - investigate
    "multilingual/ArabCulture": "arabculture",  # 494 - investigate
    "multilingual/bertaqa": "bertaqa",  # 496 - investigate

    # ==========================================================================
    # REASONING/LOGIC BENCHMARKS - TO INVESTIGATE
    # ==========================================================================
    "reasoning_logic/acp_bench": "acp_bench",  # 104 - investigate
    "reasoning_logic/score": "score",  # 276 - investigate
    "reasoning_logic/mastermind": "mastermind",  # 498 - investigate

    # ==========================================================================
    # TOOL USE / AGENTS - TO INVESTIGATE
    # ==========================================================================
    "tool_use_agents/seal": "seal",  # 111 - investigate
    "tool_use_agents/seal_0": "seal",  # 111 - investigate
    "tool_use_agents/toolemu": "toolemu",  # 144 - investigate
    "tool_use_agents/tau_bench": "tau_bench",  # 165 - investigate

    # ==========================================================================
    # READING COMPREHENSION - TO INVESTIGATE
    # ==========================================================================
    "reading_comprehension/qa4mre": "qa4mre",  # 128 - investigate
    "reading_comprehension/swde": "swde",  # 166 - investigate
    "reading_comprehension/c4": "c4",  # 199 - investigate

    # ==========================================================================
    # INSTRUCTION FOLLOWING - TO INVESTIGATE
    # ==========================================================================
    "instruction_following/eq_bench": "eq_bench",  # 136 - investigate
    "instruction_following/travelplanner": "travelplanner",  # 180 - investigate
    "instruction_following/mmmu": "mmmu",  # 380 - investigate

    # ==========================================================================
    # SAFETY/BIAS - TO INVESTIGATE
    # ==========================================================================
    "safety_bias/refusalbench": "refusalbench",  # 169 - investigate
    "safety_bias/simple_cooccurrence_bias": "simple_cooccurrence_bias",  # 280 - investigate
    "safety_bias/curate": "curate",  # 336 - investigate

    # ==========================================================================
    # LANGUAGE UNDERSTANDING - TO INVESTIGATE
    # ==========================================================================
    "language_understanding/paloma": "paloma",  # 278 - investigate
    "language_understanding/blimp": "blimp",  # 469 - investigate

    # ==========================================================================
    # KNOWLEDGE QA - TO INVESTIGATE
    # ==========================================================================
    "knowledge_qa/metabench": "metabench",  # 241 - investigate
    "knowledge_qa/mmlu_prox": "mmlu_pro",  # 456 - investigate (same as mmlu_pro?)
    "knowledge_qa/mmlu-pro-plus": "mmlu_pro",  # 456 - investigate

    # ==========================================================================
    # ETHICS/VALUES - TO INVESTIGATE
    # ==========================================================================
    "ethics_values/model_written_evals": "model_written_evals",  # 374 - investigate

    # ==========================================================================
    # HALLUCINATION/FACTUALITY - TO INVESTIGATE
    # ==========================================================================
    "hallucination_factuality/faithbench": "faithbench",  # 386 - investigate

    # ==========================================================================
    # SCIENCE/MEDICAL - TO INVESTIGATE
    # ==========================================================================
    "science_medical/healthbench": "healthbench",  # 412 - investigate

    # ==========================================================================
    # COMMONSENSE - TO INVESTIGATE
    # ==========================================================================
    "commonsense/anli": "anli",  # 498 - investigate
}


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
