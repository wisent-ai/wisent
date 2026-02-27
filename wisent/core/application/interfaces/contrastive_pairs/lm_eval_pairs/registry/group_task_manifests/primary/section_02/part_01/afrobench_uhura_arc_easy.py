"""Afrobench uhura arc easy group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_UHURA_ARC_EASY_TASKS = {
    "afrobench_uhura-arc-easy": f"{BASE_IMPORT}afrobench_uhura_arc_easy:AfrobenchUhuraArcEasyExtractor",
}
