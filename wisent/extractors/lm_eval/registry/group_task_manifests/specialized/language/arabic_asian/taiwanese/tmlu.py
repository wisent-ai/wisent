"""Tmlu group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

TMLU_TASKS = {
    "tmlu": f"{BASE_IMPORT}tmlu:TmluExtractor",
}
