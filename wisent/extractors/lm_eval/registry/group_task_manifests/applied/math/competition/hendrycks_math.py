"""Hendrycks math group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

HENDRYCKS_MATH_TASKS = {
    "hendrycks_math": f"{BASE_IMPORT}hendrycks_math:HendrycksMathExtractor",
}
