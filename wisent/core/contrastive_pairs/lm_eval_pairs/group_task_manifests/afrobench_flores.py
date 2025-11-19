"""Afrobench flores group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_FLORES_TASKS = {
    "afrobench_flores": f"{BASE_IMPORT}afrobench_flores:AfrobenchFloresExtractor",
}
