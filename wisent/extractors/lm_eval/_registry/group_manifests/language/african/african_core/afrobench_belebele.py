"""Afrobench belebele group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_BELEBELE_TASKS = {
    "afrobench_belebele": f"{BASE_IMPORT}afrobench_belebele:AfrobenchBelebeleExtractor",
}
