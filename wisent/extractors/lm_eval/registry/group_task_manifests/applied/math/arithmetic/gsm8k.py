"""Gsm8k group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

GSM8K_TASKS = {
    "gsm8k": f"{BASE_IMPORT}gsm8k:Gsm8kExtractor",
}
