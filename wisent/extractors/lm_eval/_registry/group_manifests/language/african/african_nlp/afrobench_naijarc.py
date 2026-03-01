"""Afrobench naijarc group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_NAIJARC_TASKS = {
    "afrobench_naijarc": f"{BASE_IMPORT}afrobench_naijarc:AfrobenchNaijarcExtractor",
}
