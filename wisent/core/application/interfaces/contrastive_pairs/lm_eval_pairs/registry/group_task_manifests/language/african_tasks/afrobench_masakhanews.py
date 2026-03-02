"""Afrobench masakhanews group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_MASAKHANEWS_TASKS = {
    "afrobench_masakhanews": f"{BASE_IMPORT}afrobench_masakhanews:AfrobenchMasakhanewsExtractor",
}
