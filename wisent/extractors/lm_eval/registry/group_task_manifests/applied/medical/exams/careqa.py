"""Careqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

CAREQA_TASKS = {
    "careqa": f"{BASE_IMPORT}careqa:CareqaExtractor",
}
