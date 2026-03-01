"""Afrobench masakhaner group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_MASAKHANER_TASKS = {
    "afrobench_masakhaner": f"{BASE_IMPORT}afrobench_masakhaner:AfrobenchMasakhanerExtractor",
}
