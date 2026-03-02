"""Jsonschema bench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

JSONSCHEMA_BENCH_TASKS = {
    "jsonschema_bench": f"{BASE_IMPORT}jsonschema_bench:JsonschemaBenchExtractor",
}
