"""Afrobench afrisenti group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_AFRISENTI_TASKS = {
    "afrobench_afrisenti": f"{BASE_IMPORT}afrobench_afrisenti:AfrobenchAfrisentiExtractor",
}
