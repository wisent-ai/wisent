"""Wsc273 group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

WSC273_TASKS = {
    "wsc273": f"{BASE_IMPORT}wsc273:Wsc273Extractor",
}
