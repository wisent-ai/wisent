"""Afrobench naijarc group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_NAIJARC_TASKS = {
    "afrobench_naijarc": f"{BASE_IMPORT}afrobench_naijarc:AfrobenchNaijarcExtractor",
}
