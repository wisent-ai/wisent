"""Mastermind group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

MASTERMIND_TASKS = {
    "mastermind": f"{BASE_IMPORT}mastermind:MastermindExtractor",
}
