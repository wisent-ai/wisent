"""Polemo2 group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

POLEMO2_TASKS = {
    "polemo2": f"{BASE_IMPORT}polemo2:Polemo2Extractor",
}
