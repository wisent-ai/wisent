"""Ruler group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

RULER_TASKS = {
    "ruler": f"{BASE_IMPORT}ruler:RulerExtractor",
}
