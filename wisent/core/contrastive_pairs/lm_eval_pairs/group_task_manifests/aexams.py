"""Aexams group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

AEXAMS_TASKS = {
    "aexams_IslamicStudies": f"{BASE_IMPORT}aexams:AexamsExtractor",
    "aexams_Biology": f"{BASE_IMPORT}aexams:AexamsExtractor",
    "aexams_Science": f"{BASE_IMPORT}aexams:AexamsExtractor",
    "aexams_Physics": f"{BASE_IMPORT}aexams:AexamsExtractor",
    "aexams_Social": f"{BASE_IMPORT}aexams:AexamsExtractor",
    "aexams": f"{BASE_IMPORT}aexams:AexamsExtractor",
}
