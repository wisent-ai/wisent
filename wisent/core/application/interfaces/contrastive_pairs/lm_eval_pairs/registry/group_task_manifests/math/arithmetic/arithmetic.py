"""Arithmetic group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

ARITHMETIC_TASKS = {
    "arithmetic": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_1dc": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_2da": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_2dm": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_2ds": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_3da": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_3ds": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_4da": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_4ds": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_5da": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
    "arithmetic_5ds": f"{BASE_IMPORT}arithmetic:ArithmeticExtractor",
}
