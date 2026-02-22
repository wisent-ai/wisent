"""Headqa group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

HEADQA_TASKS = {
    "headqa": f"{BASE_IMPORT}headqa:HeadQAExtractor",
    "headqa_en": f"{BASE_IMPORT}headqa:HeadQAExtractor",
    "headqa_es": f"{BASE_IMPORT}headqa:HeadQAExtractor",
}
