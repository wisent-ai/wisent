"""M group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

M_TASKS = {
    "m_mmlu_en": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_sk": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_ml": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_pt": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_it": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_hr": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_sr": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_de": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_uk": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_ru": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_ta": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_nl": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_hy": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_zh": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_bn": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_mr": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_hu": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_is": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_es": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_ar": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_sv": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_nb": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_te": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_vi": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_da": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_id": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_ro": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_ca": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_hi": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_gu": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_eu": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_kn": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_ne": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu_fr": f"{BASE_IMPORT}okapi:OkapiExtractor",
    "m_mmlu": f"{BASE_IMPORT}m_mmlu:MMmluExtractor",
}
