"""Galician bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

GALICIAN_BENCH_TASKS = {
    "galician_bench": f"{BASE_IMPORT}galician_bench:GalicianBenchExtractor",
}
