"""Longbench group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

LONGBENCH_TASKS = {
    "longbench": f"{BASE_IMPORT}longbench:LongbenchExtractor",
}
