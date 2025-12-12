"""Code2Text group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

# Code2Text tasks in lm-eval-harness
CODE_X_GLUE_TASKS = {
    "code2text": f"{BASE_IMPORT}code_x_glue:Code2TextExtractor",
    "code2text_go": f"{BASE_IMPORT}code_x_glue:Code2TextExtractor",
    "code2text_java": f"{BASE_IMPORT}code_x_glue:Code2TextExtractor",
    "code2text_javascript": f"{BASE_IMPORT}code_x_glue:Code2TextExtractor",
    "code2text_php": f"{BASE_IMPORT}code_x_glue:Code2TextExtractor",
    "code2text_python": f"{BASE_IMPORT}code_x_glue:Code2TextExtractor",
    "code2text_ruby": f"{BASE_IMPORT}code_x_glue:Code2TextExtractor",
}
