"""Afrobench_adr group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_ADR_TASKS = {
    "afrobench_adr": f"{BASE_IMPORT}afrobench:AfrobenchExtractor",
}
