"""MedQA group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

MEDQA_TASKS = {
    "medqa_4options": f"{BASE_IMPORT}medqa:MedQAExtractor",
}
