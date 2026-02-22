"""Afrobench ntrex group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_NTREX_TASKS = {
    "afrobench_ntrex": f"{BASE_IMPORT}afrobench_ntrex:AfrobenchNtrexExtractor",
}
