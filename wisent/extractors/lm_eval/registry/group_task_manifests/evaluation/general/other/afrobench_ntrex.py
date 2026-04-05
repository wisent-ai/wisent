"""Afrobench ntrex group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_NTREX_TASKS = {
    "afrobench_ntrex": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
