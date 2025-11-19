"""Okapi group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

OKAPI_TASKS = {
    "okapi": f"{BASE_IMPORT}okapi:OkapiExtractor",
}
