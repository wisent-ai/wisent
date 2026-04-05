"""Afrobench flores group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_FLORES_TASKS = {
    "afrobench_flores": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
