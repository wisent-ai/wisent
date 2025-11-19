"""Hendrycks math group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

HENDRYCKS_MATH_TASKS = {
    "hendrycks_math": f"{BASE_IMPORT}hendrycks_math:HendrycksMathExtractor",
}
