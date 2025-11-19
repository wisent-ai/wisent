"""Afrobench_afriqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_AFRIQA_TASKS = {
    "afrobench_afriqa": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
}
