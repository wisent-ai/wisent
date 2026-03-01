"""Bbq group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

BBQ_TASKS = {
    "bbq": f"{BASE_IMPORT}bbq:BbqExtractor",
}
