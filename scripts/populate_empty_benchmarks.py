#!/usr/bin/env python3
"""
Populate empty ContrastivePairSet records with actual pairs from lm-eval-harness.
"""

import argparse
import os
import sys

import psycopg2

# Map DB benchmark names to extractor names (lm-eval or HuggingFace)
DB_TO_EXTRACTOR = {
    # coding
    "coding/livecodebench_v5": "livecodebench",
    "coding/livecodebench_v6": "livecodebench",
    "coding/mercury": "mercury",
    "coding/terminal_bench": "terminal_bench",  # HF extractor
    # commonsense
    "commonsense/siqa": "siqa",
    "commonsense/storycloze": "storycloze",
    # ethics_values
    "ethics_values/model_written_evals": "model_written_evals",
    # hallucination_factuality
    "hallucination_factuality/browsecomp": "browsecomp",  # HF extractor
    "hallucination_factuality/okapi/truthfulqa_multilingual": "okapi_truthfulqa_multilingual",
    "hallucination_factuality/truthfulqa": "truthfulqa",
    "hallucination_factuality/truthfulqa_generation": "truthfulqa_gen",
    # instruction_following
    "instruction_following/mmmu": None,  # Multimodal - requires special handling
    # knowledge_qa
    "knowledge_qa/aclue": "aclue",
    "knowledge_qa/metabench": "metabench",
    "knowledge_qa/mmlu": "mmlu",
    "knowledge_qa/mmlu_pro": "mmlu",
    "knowledge_qa/mmlu-pro-plus": "mmlu",
    "knowledge_qa/mmlu_prox": "mmlu",
    # language_understanding
    "language_understanding/blimp": "blimp",
    "language_understanding/mc_taco": "mc_taco",
    "language_understanding/mutual": "mutual",
    "language_understanding/paloma": "paloma",
    "language_understanding/paws-x": "paws_x",
    # math
    "math/afrimgsm_direct_amh": "afrimgsm",
    "math/hendrycks_math": "hendrycks_math",
    "math/hrm8k": "hrm8k",
    "math/mathqa": "mathqa",
    "math/mgsm": "gsm8k",
    "math/polymath_en_high": "polymath_en_high",
    # multilingual
    "multilingual/aexams": "aexams",
    "multilingual/afrimmlu_direct_amh": "afrimmlu",
    "multilingual/ArabCulture": "arabculture",
    "multilingual/arabic_leaderboard_complete": "arabic_leaderboard_complete",
    "multilingual/arabic_leaderboard_light": "arabic_leaderboard_light",
    "multilingual/arabicmmlu": "arabicmmlu",
    "multilingual/AraDiCE": "aradice",
    "multilingual/belebele": "belebele",
    "multilingual/catalan_bench": "catalan_bench",
    "multilingual/ceval": "ceval",
    "multilingual/cmmlu": "cmmlu",
    "multilingual/copal_id": "copal_id",
    "multilingual/csatqa": "csatqa",
    "multilingual/darija_bench": None,  # No extractor available
    "multilingual/darijammlu": "darijammlu",
    "multilingual/eus_exams": None,  # No extractor available
    "multilingual/evalita_LLM": "evalita_llm",
    "multilingual/french_bench": "french_bench",
    "multilingual/galician_bench": "galician_bench",
    "multilingual/global_mmlu": "global_mmlu",
    "multilingual/haerae": "haerae",
    "multilingual/kobest": "kobest",
    "multilingual/lambada_multilingual_stablelm": None,  # Not loadable
    "multilingual/mlqa": None,  # No extractor available
    "multilingual/okapi/hellaswag_multilingual": "okapi_hellaswag_multilingual",
    "multilingual/okapi/mmlu_multilingual": "okapi_mmlu_multilingual",
    "multilingual/spanish_bench": "spanish_bench",
    "multilingual/Tag": "tag",
    "multilingual/tmmluplus": "tmmluplus",
    "multilingual/xcopa": "xcopa",
    "multilingual/xnli": "xnli",
    "multilingual/xstorycloze": "xstorycloze",
    "multilingual/xwinograd": "xwinograd",
    # reasoning_logic
    "reasoning_logic/bbh": "bbh",
    "reasoning_logic/fld": "fld",
    "reasoning_logic/inverse_scaling": None,  # No extractor available
    "reasoning_logic/lingoly": "lingoly",
    "reasoning_logic/logiqa": "logiqa",
    "reasoning_logic/logiqa2": "logiqa2",
    "reasoning_logic/mastermind": "mastermind",
    "reasoning_logic/score": "score",
    # safety_bias
    "safety_bias/bbq": "bbq",
    "safety_bias/curate": "curate",  # HF extractor
    "safety_bias/flames": "flames",  # HF extractor
    "safety_bias/politicalbias_qa": "politicalbias_qa",  # HF extractor
    "safety_bias/polyglottoxicityprompts": "polyglottoxicityprompts",  # HF extractor
    "safety_bias/realtoxicityprompts": "realtoxicityprompts",
    "safety_bias/refusalbench": "refusalbench",  # HF extractor
    "safety_bias/simple_cooccurrence_bias": "simple_cooccurrence_bias",
    "safety_bias/toxigen": "toxigen",
    "safety_bias/wmdp": "wmdp",
    # science_medical
    "science_medical/careqa": "careqa",
    "science_medical/fda": "fda",
    "science_medical/headqa": "headqa",
    "science_medical/healthbench": "healthbench",  # HF extractor
    "science_medical/kormedmcqa": "kormedmcqa",
    "science_medical/med_concepts_qa": None,  # No extractor available
    "science_medical/meddialog": "meddialog",
    "science_medical/medmcqa": "medmcqa",
    "science_medical/medqa": "medqa",
    "science_medical/pubmedqa": "pubmedqa",
    # tool_use_agents - HF extractors available
    "tool_use_agents/agentbench": "agentbench",  # HF extractor
    "tool_use_agents/finsearchcomp": "finsearchcomp",  # HF extractor
    "tool_use_agents/seal": "seal",  # HF extractor
    "tool_use_agents/seal_0": "seal_0",  # HF extractor
    "tool_use_agents/tau_bench": "tau_bench",  # HF extractor
    "tool_use_agents/toolbench": "toolbench",  # HF extractor
    "tool_use_agents/toolemu": "toolemu",  # HF extractor
    "tool_use_agents/toolllm": "toolllm",  # HF extractor
    # translation
    "translation/translation": "translation",
    "translation/wmt14_en_fr": "translation",
    "translation/wmt14_fr_en": "translation",
    "translation/wmt16_de_en": "translation",
    "translation/wmt16_en_de": "translation",
    "translation/wmt2016": "translation",
}

# Backwards compatibility alias
DB_TO_LM_EVAL = DB_TO_EXTRACTOR


def get_db_connection(db_url: str):
    """Get database connection."""
    if "pooler.supabase.com:6543" in db_url:
        db_url = db_url.replace(":6543", ":5432")
    conn = psycopg2.connect(db_url)
    conn.autocommit = True
    return conn


def get_empty_benchmarks(conn) -> list:
    """Get benchmarks with no pairs."""
    cur = conn.cursor()
    cur.execute('''
        SELECT cps.id, cps.name
        FROM "ContrastivePairSet" cps
        LEFT JOIN "ContrastivePair" cp ON cp."setId" = cps.id
        GROUP BY cps.id, cps.name
        HAVING COUNT(cp.id) = 0
        ORDER BY cps.name
    ''')
    result = cur.fetchall()
    cur.close()
    return result


def create_pair(conn, set_id: int, positive: str, negative: str, category: str) -> int:
    """Create a ContrastivePair record."""
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO "ContrastivePair" ("setId", "positiveExample", "negativeExample", "category", "createdAt", "updatedAt")
        VALUES (%s, %s, %s, %s, NOW(), NOW())
        RETURNING id
    ''', (set_id, positive[:65000], negative[:65000], category))
    pair_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return pair_id


def populate_benchmark(conn, set_id: int, db_name: str, lm_task: str, limit: int = 500) -> int:
    """Generate and store pairs for a benchmark."""
    from wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_pairs_generation import build_contrastive_pairs

    print(f"  Generating pairs using lm-eval task '{lm_task}'...", flush=True)
    try:
        pairs = build_contrastive_pairs(lm_task, limit=limit)
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
        return 0

    if not pairs:
        print(f"  No pairs generated", flush=True)
        return 0

    print(f"  Storing {len(pairs)} pairs...", flush=True)
    count = 0
    for i, pair in enumerate(pairs):
        prompt = pair.prompt
        pos = pair.positive_response.model_response if hasattr(pair.positive_response, 'model_response') else str(pair.positive_response)
        neg = pair.negative_response.model_response if hasattr(pair.negative_response, 'model_response') else str(pair.negative_response)

        positive_text = f"{prompt}\n\n{pos}"
        negative_text = f"{prompt}\n\n{neg}"

        create_pair(conn, set_id, positive_text, negative_text, f"pair_{i}")
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", default=os.environ.get("DATABASE_URL"),
                        help="Database URL")
    parser.add_argument("--limit", type=int, default=500, help="Max pairs per benchmark")
    parser.add_argument("--benchmark", help="Single benchmark to populate (optional)")
    args = parser.parse_args()

    if not args.db_url:
        print("ERROR: No database URL provided", flush=True)
        sys.exit(1)

    conn = get_db_connection(args.db_url)

    if args.benchmark:
        # Single benchmark mode
        cur = conn.cursor()
        cur.execute('SELECT id, name FROM "ContrastivePairSet" WHERE name = %s', (args.benchmark,))
        result = cur.fetchone()
        cur.close()

        if not result:
            print(f"ERROR: Benchmark '{args.benchmark}' not found", flush=True)
            sys.exit(1)

        set_id, name = result
        lm_task = DB_TO_LM_EVAL.get(name)
        if not lm_task:
            print(f"ERROR: No lm-eval mapping for '{name}'", flush=True)
            sys.exit(1)

        print(f"Populating: {name} -> {lm_task}", flush=True)
        count = populate_benchmark(conn, set_id, name, lm_task, args.limit)
        print(f"Done! Created {count} pairs", flush=True)
    else:
        # Populate all empty benchmarks
        empty = get_empty_benchmarks(conn)
        print(f"Found {len(empty)} empty benchmarks", flush=True)

        total = 0
        skipped = 0
        for i, (set_id, name) in enumerate(empty):
            lm_task = DB_TO_LM_EVAL.get(name)

            if not lm_task:
                print(f"[{i+1}/{len(empty)}] {name} - SKIPPED (no lm-eval mapping)", flush=True)
                skipped += 1
                continue

            print(f"\n[{i+1}/{len(empty)}] {name} -> {lm_task}", flush=True)
            count = populate_benchmark(conn, set_id, name, lm_task, args.limit)
            total += count
            print(f"  Created {count} pairs", flush=True)

        print(f"\n{'='*60}", flush=True)
        print(f"COMPLETE! Total: {total} pairs created, {skipped} skipped", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
