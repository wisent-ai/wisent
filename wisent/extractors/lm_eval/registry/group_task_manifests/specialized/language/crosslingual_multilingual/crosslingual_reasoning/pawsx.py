"""Pawsx group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

PAWSX_TASKS = {
    "pawsx": f"{BASE_IMPORT}pawsx:PawsxExtractor",
}
