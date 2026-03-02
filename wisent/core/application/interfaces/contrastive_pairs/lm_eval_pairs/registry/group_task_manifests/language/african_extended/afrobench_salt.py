"""Afrobench salt group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_SALT_TASKS = {
    "afrobench_salt": f"{BASE_IMPORT}afrobench_salt:AfrobenchSaltExtractor",
}
