"""Afrobench flores group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_FLORES_TASKS = {
    "afrobench_flores": f"{BASE_IMPORT}afrobench_flores:AfrobenchFloresExtractor",
}
