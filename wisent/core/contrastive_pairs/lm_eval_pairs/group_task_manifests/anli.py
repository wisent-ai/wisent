"""Anli group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

ANLI_TASKS = {
    "anli": f"{BASE_IMPORT}anli:AnliExtractor",
}
