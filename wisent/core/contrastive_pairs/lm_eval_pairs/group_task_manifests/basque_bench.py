"""Basque bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

BASQUE_BENCH_TASKS = {
    "basque_bench": f"{BASE_IMPORT}basque_bench:BasqueBenchExtractor",
}
