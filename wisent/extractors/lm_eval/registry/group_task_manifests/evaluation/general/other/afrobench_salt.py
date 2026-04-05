"""Afrobench salt group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_SALT_TASKS = {
    "afrobench_salt": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
