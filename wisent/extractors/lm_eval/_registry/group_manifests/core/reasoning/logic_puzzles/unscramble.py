"""Unscramble group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

UNSCRAMBLE_TASKS = {
    "unscramble": f"{BASE_IMPORT}unscramble:UnscrambleExtractor",
}
