"""Ruler group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

RULER_TASKS = {
    "ruler": f"{BASE_IMPORT}ruler:RulerExtractor",
}
