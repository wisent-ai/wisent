"""Mastermind group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

MASTERMIND_TASKS = {
    "mastermind": f"{BASE_IMPORT}mastermind:MastermindExtractor",
}
