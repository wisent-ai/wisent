"""Copal id group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

COPAL_ID_TASKS = {
    "copal_id": f"{BASE_IMPORT}copal_id:CopalIdExtractor",
    "copal_id_colloquial": f"{BASE_IMPORT}copal_id:CopalIdExtractor",
    "copal_id_standard": f"{BASE_IMPORT}copal_id:CopalIdExtractor",
}
