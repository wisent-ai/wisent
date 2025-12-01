"""Glianorex group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.core.contrastive_pairs.lm_eval_pairs.lm_task_extractors."

GLIANOREX_TASKS = {
    "glianorex": f"{BASE_IMPORT}glianorex:GlianorexExtractor",
    "glianorex_en": f"{BASE_IMPORT}glianorex:GlianorexExtractor",
    "glianorex_fr": f"{BASE_IMPORT}glianorex:GlianorexExtractor",
}
