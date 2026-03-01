"""MedQA group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

MEDQA_TASKS = {
    "medqa_4options": f"{BASE_IMPORT}medqa:MedQAExtractor",
}
