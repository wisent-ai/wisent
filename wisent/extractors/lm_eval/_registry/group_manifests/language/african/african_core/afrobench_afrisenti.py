"""Afrobench afrisenti group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_AFRISENTI_TASKS = {
    "afrobench_afrisenti": f"{BASE_IMPORT}afrobench_afrisenti:AfrobenchAfrisentiExtractor",
}
