"""Okapi hellaswag multilingual group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# Individual hellaswag tasks
_HELLASWAG_TASKS = [
    "hellaswag_ar", "hellaswag_bn", "hellaswag_ca", "hellaswag_da", "hellaswag_de",
    "hellaswag_es", "hellaswag_eu", "hellaswag_fr", "hellaswag_gu", "hellaswag_hi",
    "hellaswag_hr", "hellaswag_hu", "hellaswag_hy", "hellaswag_id", "hellaswag_it",
    "hellaswag_kn", "hellaswag_ml", "hellaswag_mr", "hellaswag_ne", "hellaswag_nl",
    "hellaswag_pt", "hellaswag_ro", "hellaswag_ru", "hellaswag_sk", "hellaswag_sr",
    "hellaswag_sv", "hellaswag_ta", "hellaswag_te", "hellaswag_uk", "hellaswag_vi"
]

OKAPI_HELLASWAG_MULTILINGUAL_TASKS = {
    "okapi_hellaswag_multilingual": f"{BASE_IMPORT}okapi_hellaswag_multilingual:OkapiHellaswagMultilingualExtractor",
    "okapi/hellaswag_multilingual": f"{BASE_IMPORT}okapi_hellaswag_multilingual:OkapiHellaswagMultilingualExtractor",
}

# Register all individual tasks
for task in _HELLASWAG_TASKS:
    OKAPI_HELLASWAG_MULTILINGUAL_TASKS[task] = f"{BASE_IMPORT}okapi_hellaswag_multilingual:OkapiHellaswagMultilingualExtractor"
