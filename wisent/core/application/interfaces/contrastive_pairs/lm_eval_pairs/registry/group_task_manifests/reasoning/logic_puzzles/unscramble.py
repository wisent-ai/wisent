"""Unscramble group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

UNSCRAMBLE_TASKS = {
    "unscramble": f"{BASE_IMPORT}unscramble:UnscrambleExtractor",
}
