"""Gsm8k group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

GSM8K_TASKS = {
    "gsm8k": f"{BASE_IMPORT}gsm8k:Gsm8kExtractor",
}
