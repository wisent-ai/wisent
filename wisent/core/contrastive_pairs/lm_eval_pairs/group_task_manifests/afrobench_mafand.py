"""Afrobench mafand group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AFROBENCH_MAFAND_TASKS = {
    "afrobench_mafand": f"{BASE_IMPORT}afrobench_mafand:AfrobenchMafandExtractor",
}
