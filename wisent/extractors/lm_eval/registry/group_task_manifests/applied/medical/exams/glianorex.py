"""Glianorex group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval.lm_task_extractors."

GLIANOREX_TASKS = {
    "glianorex": f"{BASE_IMPORT}glianorex:GlianorexExtractor",
    "glianorex_en": f"{BASE_IMPORT}glianorex:GlianorexExtractor",
    "glianorex_fr": f"{BASE_IMPORT}glianorex:GlianorexExtractor",
}
