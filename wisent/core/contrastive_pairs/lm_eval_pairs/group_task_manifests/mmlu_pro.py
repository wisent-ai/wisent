"""Mmlu pro group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MMLU_PRO_TASKS = {
    "mmlu_pro": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_biology": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_business": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_chemistry": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_computer_science": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_economics": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_engineering": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_health": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_history": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_law": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_math": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_other": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_philosophy": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_physics": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
    "mmlu_pro_psychology": f"{BASE_IMPORT}mmlu_pro:MMLUProExtractor",
}
