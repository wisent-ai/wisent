"""MathQA group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

MATHQA_TASKS = {
    "mathqa": f"{BASE_IMPORT}mathqa:MathQAExtractor",
}
