"""Okapi truthfulqa multilingual group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# Individual truthfulqa tasks (mc1 and mc2 for each language)
_TRUTHFULQA_TASKS = [
    "truthfulqa_ar_mc1", "truthfulqa_ar_mc2", "truthfulqa_bn_mc1", "truthfulqa_bn_mc2",
    "truthfulqa_ca_mc1", "truthfulqa_ca_mc2", "truthfulqa_da_mc1", "truthfulqa_da_mc2",
    "truthfulqa_de_mc1", "truthfulqa_de_mc2", "truthfulqa_es_mc1", "truthfulqa_es_mc2",
    "truthfulqa_eu_mc1", "truthfulqa_eu_mc2", "truthfulqa_fr_mc1", "truthfulqa_fr_mc2",
    "truthfulqa_gu_mc1", "truthfulqa_gu_mc2", "truthfulqa_hi_mc1", "truthfulqa_hi_mc2",
    "truthfulqa_hr_mc1", "truthfulqa_hr_mc2", "truthfulqa_hu_mc1", "truthfulqa_hu_mc2",
    "truthfulqa_hy_mc1", "truthfulqa_hy_mc2", "truthfulqa_id_mc1", "truthfulqa_id_mc2",
    "truthfulqa_it_mc1", "truthfulqa_it_mc2", "truthfulqa_kn_mc1", "truthfulqa_kn_mc2",
    "truthfulqa_ml_mc1", "truthfulqa_ml_mc2", "truthfulqa_mr_mc1", "truthfulqa_mr_mc2",
    "truthfulqa_ne_mc1", "truthfulqa_ne_mc2", "truthfulqa_nl_mc1", "truthfulqa_nl_mc2",
    "truthfulqa_pt_mc1", "truthfulqa_pt_mc2", "truthfulqa_ro_mc1", "truthfulqa_ro_mc2",
    "truthfulqa_ru_mc1", "truthfulqa_ru_mc2", "truthfulqa_sk_mc1", "truthfulqa_sk_mc2",
    "truthfulqa_sr_mc1", "truthfulqa_sr_mc2", "truthfulqa_sv_mc1", "truthfulqa_sv_mc2",
    "truthfulqa_ta_mc1", "truthfulqa_ta_mc2", "truthfulqa_te_mc1", "truthfulqa_te_mc2",
    "truthfulqa_uk_mc1", "truthfulqa_uk_mc2", "truthfulqa_vi_mc1", "truthfulqa_vi_mc2",
    "truthfulqa_zh_mc1", "truthfulqa_zh_mc2",
]

OKAPI_TRUTHFULQA_MULTILINGUAL_TASKS = {
    "okapi_truthfulqa_multilingual": f"{BASE_IMPORT}okapi_truthfulqa_multilingual:OkapiTruthfulqaMultilingualExtractor",
    "okapi/truthfulqa_multilingual": f"{BASE_IMPORT}okapi_truthfulqa_multilingual:OkapiTruthfulqaMultilingualExtractor",
}

# Register all individual tasks
for task in _TRUTHFULQA_TASKS:
    OKAPI_TRUTHFULQA_MULTILINGUAL_TASKS[task] = f"{BASE_IMPORT}okapi_truthfulqa_multilingual:OkapiTruthfulqaMultilingualExtractor"
