"""Longbench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

LONGBENCH_TASKS = {
    "longbench": f"{BASE_IMPORT}longbench:LongbenchExtractor",
}
