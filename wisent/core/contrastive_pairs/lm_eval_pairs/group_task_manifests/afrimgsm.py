"""Afrimgsm group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFRIMGSM_TASKS = {
    "afrimgsm": f"{BASE_IMPORT}afrimgsm:AfrimgsmExtractor",
    "afrimgsm_direct_amh": f"{BASE_IMPORT}afrimgsm:AfrimgsmExtractor",
}
