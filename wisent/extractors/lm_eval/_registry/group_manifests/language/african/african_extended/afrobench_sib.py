"""Afrobench sib group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_SIB_TASKS = {
    "afrobench_sib": f"{BASE_IMPORT}afrobench_sib:AfrobenchSibExtractor",
}
