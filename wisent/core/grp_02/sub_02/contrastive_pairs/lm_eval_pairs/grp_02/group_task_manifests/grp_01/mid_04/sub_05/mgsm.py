"""Mgsm group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MGSM_TASKS = {
    "mgsm": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_en": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_es": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_fr": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_de": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_ru": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_zh": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_ja": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_th": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_sw": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_bn": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_direct_te": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_bn": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_de": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_en": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_es": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_fr": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_ja": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_ru": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_sw": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_te": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_th": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_cot_native_zh": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_bn": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_de": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_en": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_es": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_fr": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_ja": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_ru": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_sw": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_te": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_th": f"{BASE_IMPORT}mgsm:MgsmExtractor",
    "mgsm_native_cot_zh": f"{BASE_IMPORT}mgsm:MgsmExtractor",
}
