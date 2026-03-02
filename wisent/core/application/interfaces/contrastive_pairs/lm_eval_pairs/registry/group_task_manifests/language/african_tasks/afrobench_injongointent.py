"""Afrobench injongointent group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_INJONGOINTENT_TASKS = {
    "afrobench_injongointent": f"{BASE_IMPORT}afrobench_injongointent:AfrobenchInjongointentExtractor",
}
