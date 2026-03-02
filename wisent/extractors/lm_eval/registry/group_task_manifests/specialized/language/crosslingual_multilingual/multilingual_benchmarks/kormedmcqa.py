"""Kormedmcqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

KORMEDMCQA_TASKS = {
    "kormedmcqa": f"{BASE_IMPORT}kormedmcqa:KormedmcqaExtractor",
}
