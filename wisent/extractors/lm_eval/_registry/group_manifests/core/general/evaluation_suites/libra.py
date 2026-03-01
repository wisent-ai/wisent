"""Libra group task manifest."""

from __future__ import annotations

BASE_IMPORT = "wisent.extractors.lm_eval."

LIBRA_TASKS = {
    "libra": f"{BASE_IMPORT}libra:LibraExtractor",
}
