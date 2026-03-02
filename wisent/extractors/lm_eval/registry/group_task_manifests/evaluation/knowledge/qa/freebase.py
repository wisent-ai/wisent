"""Freebase group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

FREEBASE_TASKS = {
    "webqs": f"{BASE_IMPORT}webqs:WebQSExtractor",
}
