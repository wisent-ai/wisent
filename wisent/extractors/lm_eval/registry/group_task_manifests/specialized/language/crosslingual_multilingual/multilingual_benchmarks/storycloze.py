"""Storycloze group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

STORYCLOZE_TASKS = {
    "storycloze": f"{BASE_IMPORT}xstorycloze:XStoryclozeExtractor",
}
