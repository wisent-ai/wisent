"""Afrobench xlsum group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFROBENCH_XLSUM_TASKS = {
    "afrobench_xlsum": f"{BASE_IMPORT}afrobench_xlsum:AfrobenchXlsumExtractor",
}
