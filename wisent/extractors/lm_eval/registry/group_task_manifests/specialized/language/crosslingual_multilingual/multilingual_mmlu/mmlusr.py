"""Mmlusr group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

MMLUSR_TASKS = {
    "mmlusr": f"{BASE_IMPORT}mmlusr:MmlusrExtractor",
}
