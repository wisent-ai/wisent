"""Afrimmlu group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFRIMMLU_TASKS = {
    "afrimmlu": f"{BASE_IMPORT}afrimmlu:AfrimmluExtractor",
    "afrimmlu_direct_amh": f"{BASE_IMPORT}afrimmlu:AfrimmluExtractor",
}
