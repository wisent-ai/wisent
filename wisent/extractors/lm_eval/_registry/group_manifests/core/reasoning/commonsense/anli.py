"""Anli group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

ANLI_TASKS = {
    "anli": f"{BASE_IMPORT}anli:AnliExtractor",
}
