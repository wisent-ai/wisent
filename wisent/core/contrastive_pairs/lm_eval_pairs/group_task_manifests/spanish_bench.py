"""Spanish bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

SPANISH_BENCH_TASKS = {
    "spanish_bench": f"{BASE_IMPORT}spanish_bench:SpanishBenchExtractor",
}
