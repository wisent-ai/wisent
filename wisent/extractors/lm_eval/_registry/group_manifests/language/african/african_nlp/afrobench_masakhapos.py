"""Afrobench masakhapos group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_MASAKHAPOS_TASKS = {
    "afrobench_masakhapos": f"{BASE_IMPORT}afrobench_masakhapos:AfrobenchMasakhaposExtractor",
}
