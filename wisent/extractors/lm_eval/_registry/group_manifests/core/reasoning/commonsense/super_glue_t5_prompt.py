"""Super-glue t5-prompt group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

SUPER_GLUE_T5_PROMPT_TASKS = {
    "super-glue-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-wsc-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-record-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-multirc-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-copa-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-boolq-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-rte-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-wic-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
    "super_glue-cb-t5-prompt": f"{BASE_IMPORT}super_glue_t5_prompt:SuperGlueT5PromptExtractor",
}
