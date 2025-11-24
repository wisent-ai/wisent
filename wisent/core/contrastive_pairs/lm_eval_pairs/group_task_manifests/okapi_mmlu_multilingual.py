"""Okapi mmlu multilingual group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# Individual mmlu tasks
_MMLU_TASKS = [
    "m_mmlu_ar", "m_mmlu_bn", "m_mmlu_ca", "m_mmlu_da", "m_mmlu_de", "m_mmlu_en",
    "m_mmlu_es", "m_mmlu_eu", "m_mmlu_fr", "m_mmlu_gu", "m_mmlu_hi", "m_mmlu_hr",
    "m_mmlu_hu", "m_mmlu_hy", "m_mmlu_id", "m_mmlu_is", "m_mmlu_it", "m_mmlu_kn",
    "m_mmlu_ml", "m_mmlu_mr", "m_mmlu_nb", "m_mmlu_ne", "m_mmlu_nl", "m_mmlu_pt",
    "m_mmlu_ro", "m_mmlu_ru", "m_mmlu_sk", "m_mmlu_sr", "m_mmlu_sv", "m_mmlu_ta",
    "m_mmlu_te", "m_mmlu_uk", "m_mmlu_vi", "m_mmlu_zh"
]

OKAPI_MMLU_MULTILINGUAL_TASKS = {
    "okapi_mmlu_multilingual": f"{BASE_IMPORT}okapi_mmlu_multilingual:OkapiMmluMultilingualExtractor",
    "okapi/mmlu_multilingual": f"{BASE_IMPORT}okapi_mmlu_multilingual:OkapiMmluMultilingualExtractor",
}

# Register all individual tasks
for task in _MMLU_TASKS:
    OKAPI_MMLU_MULTILINGUAL_TASKS[task] = f"{BASE_IMPORT}okapi_mmlu_multilingual:OkapiMmluMultilingualExtractor"
