"""Afrobench_adr group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_ADR_TASKS = {
    "afrobench_adr": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
}
