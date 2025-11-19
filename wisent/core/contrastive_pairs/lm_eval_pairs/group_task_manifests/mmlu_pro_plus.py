"""Mmlu pro plus group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MMLU_PRO_PLUS_TASKS = {
    "mmlu-pro-plus": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_biology": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_business": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_chemistry": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_computer_science": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_economics": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_engineering": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_health": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_history": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_law": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_math": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_other": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_philosophy": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_physics": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_plus_psychology": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
}
