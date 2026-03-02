"""Afrobench xlsum group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_XLSUM_TASKS = {
    "afrobench_xlsum": f"{BASE_IMPORT}afrobench_xlsum:AfrobenchXlsumExtractor",
}
