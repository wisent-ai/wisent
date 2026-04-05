"""Afrobench masakhanews group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_MASAKHANEWS_TASKS = {
    "afrobench_masakhanews": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
}
