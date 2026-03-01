"""Translation group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

TRANSLATION_TASKS = {
    "translation": f"{BASE_IMPORT}translation:TranslationExtractor",
}
