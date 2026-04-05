"""Afrobench belebele group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_BELEBELE_TASKS = {
    "afrobench_belebele": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
}
