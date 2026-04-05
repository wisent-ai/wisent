"""Afrobench mafand group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFROBENCH_MAFAND_TASKS = {
    "afrobench_mafand": f"{BASE_IMPORT}afrobench_cot:AfroBenchCotExtractor",
}
