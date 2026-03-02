"""Afrobench belebele group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_BELEBELE_TASKS = {
    "afrobench_belebele": f"{BASE_IMPORT}afrobench_belebele:AfrobenchBelebeleExtractor",
}
