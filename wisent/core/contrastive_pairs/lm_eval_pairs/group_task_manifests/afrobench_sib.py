"""Afrobench sib group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_SIB_TASKS = {
    "afrobench_sib": f"{BASE_IMPORT}afrobench_sib:AfrobenchSibExtractor",
}
