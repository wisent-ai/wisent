"""Qasper group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

QASPER_TASKS = {
    "qasper": f"{BASE_IMPORT}qasper:QasperExtractor",
    "qasper_bool": f"{BASE_IMPORT}qasper:QasperExtractor",
    "qasper_freeform": f"{BASE_IMPORT}qasper:QasperExtractor",
}
