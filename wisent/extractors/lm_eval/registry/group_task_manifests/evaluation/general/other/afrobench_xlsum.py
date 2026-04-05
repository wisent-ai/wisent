"""Afrobench xlsum group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_XLSUM_TASKS = {
    "afrobench_xlsum": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
