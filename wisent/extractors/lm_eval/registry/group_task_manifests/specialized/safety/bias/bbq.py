"""Bbq group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

BBQ_TASKS = {
    "bbq": f"{BASE_IMPORT}bbq:BbqExtractor",
}
