"""Afrobench masakhaner group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_MASAKHANER_TASKS = {
    "afrobench_masakhaner": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
