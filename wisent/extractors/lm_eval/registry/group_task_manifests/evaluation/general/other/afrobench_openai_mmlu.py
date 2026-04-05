"""Afrobench openai mmlu group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_OPENAI_MMLU_TASKS = {
    "afrobench_openai_mmlu": f"{BASE_IMPORT}afrobench_mc:AfroBenchMultipleChoiceExtractor",
}
