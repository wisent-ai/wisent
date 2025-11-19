"""Portuguese bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

PORTUGUESE_BENCH_TASKS = {
    "portuguese_bench": f"{BASE_IMPORT}portuguese_bench:PortugueseBenchExtractor",
}
