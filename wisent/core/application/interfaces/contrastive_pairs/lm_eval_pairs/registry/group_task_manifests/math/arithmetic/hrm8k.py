"""Hrm8K group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

HRM8K_TASKS = {
    "hrm8k": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_en": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_gsm8k": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_gsm8k_en": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_ksm": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_ksm_en": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_math": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_math_en": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_mmmlu": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_mmmlu_en": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_omni_math": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
    "hrm8k_omni_math_en": f"{BASE_IMPORT}hrm8k:Hrm8kExtractor",
}
