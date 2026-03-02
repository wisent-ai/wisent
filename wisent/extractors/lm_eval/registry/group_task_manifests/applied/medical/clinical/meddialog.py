"""Meddialog group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

MEDDIALOG_TASKS = {
    "meddialog_qsumm": f"{BASE_IMPORT}meddialog:MeddialogExtractor",
    "meddialog_qsumm_perplexity": f"{BASE_IMPORT}meddialog:MeddialogExtractor",
    "meddialog_raw_dialogues": f"{BASE_IMPORT}meddialog:MeddialogExtractor",
    "meddialog_raw_perplexity": f"{BASE_IMPORT}meddialog:MeddialogExtractor",
}
