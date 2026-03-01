"""Mmlusr group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

MMLUSR_TASKS = {
    "mmlusr": f"{BASE_IMPORT}mmlusr:MmlusrExtractor",
}
