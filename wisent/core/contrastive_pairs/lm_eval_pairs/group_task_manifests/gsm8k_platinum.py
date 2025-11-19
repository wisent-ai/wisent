"""Gsm8k platinum group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

GSM8K_PLATINUM_TASKS = {
    "gsm8k_platinum": f"{BASE_IMPORT}gsm8k_platinum:Gsm8kPlatinumExtractor",
}
