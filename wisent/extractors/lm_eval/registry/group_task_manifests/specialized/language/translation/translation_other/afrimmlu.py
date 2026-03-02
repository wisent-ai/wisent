"""Afrimmlu group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

AFRIMMLU_TASKS = {
    "afrimmlu": f"{BASE_IMPORT}afrimmlu:AfrimmluExtractor",
    "afrimmlu_direct_amh": f"{BASE_IMPORT}afrimmlu:AfrimmluExtractor",
}
