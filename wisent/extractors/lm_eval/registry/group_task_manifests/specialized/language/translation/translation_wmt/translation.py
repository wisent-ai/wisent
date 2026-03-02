"""Translation group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

TRANSLATION_TASKS = {
    "translation": f"{BASE_IMPORT}translation:TranslationExtractor",
}
