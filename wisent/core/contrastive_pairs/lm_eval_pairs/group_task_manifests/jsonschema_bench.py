"""Jsonschema bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

JSONSCHEMA_BENCH_TASKS = {
    "jsonschema_bench": f"{BASE_IMPORT}jsonschema_bench:JsonschemaBenchExtractor",
}
