"""Anli group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

ANLI_TASKS = {
    "anli": f"{BASE_IMPORT}anli:AnliExtractor",
}
