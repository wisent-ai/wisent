"""Afrobench masakhapos group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_MASAKHAPOS_TASKS = {
    "afrobench_masakhapos": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
