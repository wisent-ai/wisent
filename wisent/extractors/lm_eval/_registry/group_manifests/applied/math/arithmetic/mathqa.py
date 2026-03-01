"""MathQA group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

MATHQA_TASKS = {
    "mathqa": f"{BASE_IMPORT}mathqa:MathQAExtractor",
}
