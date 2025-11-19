"""Winogender group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

WINOGENDER_TASKS = {
    "winogender": f"{BASE_IMPORT}winogender:WinogenderExtractor",
    "winogender_all": f"{BASE_IMPORT}winogender:WinogenderExtractor",
    "winogender_female": f"{BASE_IMPORT}winogender:WinogenderExtractor",
    "winogender_gotcha": f"{BASE_IMPORT}winogender:WinogenderExtractor",
    "winogender_gotcha_female": f"{BASE_IMPORT}winogender:WinogenderExtractor",
    "winogender_gotcha_male": f"{BASE_IMPORT}winogender:WinogenderExtractor",
    "winogender_male": f"{BASE_IMPORT}winogender:WinogenderExtractor",
    "winogender_neutral": f"{BASE_IMPORT}winogender:WinogenderExtractor",
}
