"""Afrimgsm group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

AFRIMGSM_TASKS = {
    "afrimgsm": f"{BASE_IMPORT}afrimgsm:AfrimgsmExtractor",
    "afrimgsm_direct_amh": f"{BASE_IMPORT}afrimgsm:AfrimgsmExtractor",
}
