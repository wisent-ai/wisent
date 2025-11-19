"""Tinybenchmarks group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

TINYBENCHMARKS_TASKS = {
    "tinyBenchmarks": f"{BASE_IMPORT}tinybenchmarks:TinybenchmarksExtractor",
}
