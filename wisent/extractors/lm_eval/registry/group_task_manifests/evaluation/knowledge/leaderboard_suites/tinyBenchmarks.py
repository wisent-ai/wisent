"""Tinybenchmarks group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

TINYBENCHMARKS_TASKS = {
    "tinyBenchmarks": f"{BASE_IMPORT}tinybenchmarks:TinybenchmarksExtractor",
}
