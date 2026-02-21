"""
Benchmark fix mapping data for fix_benchmark_one_by_one.py.

Maps database benchmark names to lm_eval task names.
Format: "db_name": "lm_eval_task_name",  # pair_count (status/notes)
"""

# Mapping of db_name -> task_name for benchmarks that need fixing
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
