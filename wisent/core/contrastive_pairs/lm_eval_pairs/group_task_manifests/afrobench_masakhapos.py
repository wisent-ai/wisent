"""Afrobench masakhapos group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_MASAKHAPOS_TASKS = {
    "afrobench_masakhapos": f"{BASE_IMPORT}afrobench_masakhapos:AfrobenchMasakhaposExtractor",
}
