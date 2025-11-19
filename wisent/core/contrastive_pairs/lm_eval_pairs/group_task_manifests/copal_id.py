"""Copal id group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

COPAL_ID_TASKS = {
    "copal_id": f"{BASE_IMPORT}copal_id:CopalIdExtractor",
    "copal_id_colloquial": f"{BASE_IMPORT}copal_id:CopalIdExtractor",
    "copal_id_standard": f"{BASE_IMPORT}copal_id:CopalIdExtractor",
}
