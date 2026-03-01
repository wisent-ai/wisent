"""Storycloze group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

STORYCLOZE_TASKS = {
    "storycloze": f"{BASE_IMPORT}xstorycloze:XStoryclozeExtractor",
}
