"""Kobest group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

KOBEST_TASKS = {
    "kobest": f"{BASE_IMPORT}kobest:KobestExtractor",
    "kobest_boolq": f"{BASE_IMPORT}kobest:KobestExtractor",
    "kobest_copa": f"{BASE_IMPORT}kobest:KobestExtractor",
    "kobest_hellaswag": f"{BASE_IMPORT}kobest:KobestExtractor",
    "kobest_sentineg": f"{BASE_IMPORT}kobest:KobestExtractor",
    "kobest_wic": f"{BASE_IMPORT}kobest:KobestExtractor",
}
