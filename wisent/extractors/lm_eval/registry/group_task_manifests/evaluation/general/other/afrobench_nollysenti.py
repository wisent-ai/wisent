"""Afrobench nollysenti group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_NOLLYSENTI_TASKS = {
    "afrobench_nollysenti": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
}
