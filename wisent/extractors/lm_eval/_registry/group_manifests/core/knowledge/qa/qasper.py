"""Qasper group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

QASPER_TASKS = {
    "qasper": f"{BASE_IMPORT}qasper:QasperExtractor",
    "qasper_bool": f"{BASE_IMPORT}qasper:QasperExtractor",
    "qasper_freeform": f"{BASE_IMPORT}qasper:QasperExtractor",
}
