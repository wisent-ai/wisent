"""Self_consistency group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

SELF_CONSISTENCY_TASKS = {
    "self_consistency": f"{BASE_IMPORT}gsm8k:GSM8KExtractor",
    "gsm8k_cot_self_consistency": f"{BASE_IMPORT}gsm8k:GSM8KExtractor",
    "gsm8k_platinum_cot_self_consistency": f"{BASE_IMPORT}gsm8k:GSM8KExtractor",
}
