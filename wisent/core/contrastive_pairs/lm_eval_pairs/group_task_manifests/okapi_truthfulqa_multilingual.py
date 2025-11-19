"""Okapi truthfulqa multilingual group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

OKAPI_TRUTHFULQA_MULTILINGUAL_TASKS = {
    "okapi_truthfulqa_multilingual": f"{BASE_IMPORT}okapi_truthfulqa_multilingual:OkapiTruthfulqaMultilingualExtractor",
}
