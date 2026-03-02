"""Okapi group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

OKAPI_TASKS = {
    "okapi": f"{BASE_IMPORT}okapi:OkapiExtractor",
}
