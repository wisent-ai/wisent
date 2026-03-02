"""Gsm8k platinum group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

GSM8K_PLATINUM_TASKS = {
    "gsm8k_platinum": f"{BASE_IMPORT}gsm8k_platinum:Gsm8kPlatinumExtractor",
}
