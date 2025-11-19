"""Okapi mmlu multilingual group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

OKAPI_MMLU_MULTILINGUAL_TASKS = {
    "okapi_mmlu_multilingual": f"{BASE_IMPORT}okapi_mmlu_multilingual:OkapiMmluMultilingualExtractor",
}
