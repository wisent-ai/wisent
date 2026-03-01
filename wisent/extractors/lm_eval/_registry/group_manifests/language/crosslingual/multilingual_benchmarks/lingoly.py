"""Lingoly group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

LINGOLY_TASKS = {
    "lingoly": f"{BASE_IMPORT}lingoly:LingolyExtractor",
    "lingoly_context": f"{BASE_IMPORT}lingoly:LingolyExtractor",
    "lingoly_nocontext": f"{BASE_IMPORT}lingoly:LingolyExtractor",
}
